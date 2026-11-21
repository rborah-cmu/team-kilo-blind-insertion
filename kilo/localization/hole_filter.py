"""Particle filter for finding the hole by pressing the peg tip on the box's top face.

Model: a press at point p "drops in" (the rounded tip falls below the top surface)
if and only if p is within ``detect_radius`` of the hole centre. A press that stays
at top height rules out that neighbourhood; a press that drops in confines the
hole to it. ``miss_prob`` allows for occasional misreads so one bad press does not
eliminate the true hole.

All positions are 2D (x, y) in the robot base frame, in metres.
"""

from __future__ import annotations

import numpy as np


class HoleSearchFilter:
    def __init__(
        self,
        center: np.ndarray,
        max_offset: float,
        detect_radius: float,
        num_particles: int = 4000,
        center_std: float = 0.0,
        miss_prob: float = 0.02,
        rng: np.random.Generator | None = None,
    ):
        """
        Args:
            center: estimated (x, y) of the box's top-face centre.
            max_offset: R, the hole lies within this radius of the true centre.
            detect_radius: a press within this distance of the hole drops in
                (about one hole radius for a rounded tip).
            num_particles: number of hypotheses for the hole centre.
            center_std: uncertainty of ``center``; widens the prior disc by 2 sigma.
            miss_prob: probability a press reads the wrong way.
            rng: random generator, for reproducibility.
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        self.center = np.asarray(center, dtype=float)
        self.radius = max_offset + 2.0 * center_std
        self.detect_radius = detect_radius
        self.miss_prob = miss_prob

        # Uniform samples over a disc: sqrt on the radius keeps density even.
        r = self.radius * np.sqrt(self.rng.uniform(size=num_particles))
        theta = self.rng.uniform(0.0, 2.0 * np.pi, size=num_particles)
        self.particles = self.center + np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        self.weights = np.full(num_particles, 1.0 / num_particles)

    def update_press(self, point: np.ndarray, dropped: bool) -> None:
        """Incorporate one press at ``point`` and whether the tip dropped in."""
        near = np.linalg.norm(self.particles - np.asarray(point, dtype=float), axis=1) < self.detect_radius
        hit, miss = 1.0 - self.miss_prob, self.miss_prob
        likelihood = np.where(near, hit, miss) if dropped else np.where(near, miss, hit)
        self.weights *= likelihood
        self.weights /= self.weights.sum()
        if self.effective_sample_size() < 0.5 * len(self.weights):
            self._resample()

    def update_seated(self, tip_xy: np.ndarray, std: float) -> None:
        """Incorporate where the tip came to rest after dropping in.

        Under compliant control the rounded tip slides toward the hole centre, so
        its resting (x, y) is a direct, noisy measurement of the centre. ``std``
        is about the radial clearance plus forward-kinematics error.
        """
        d2 = np.sum((self.particles - np.asarray(tip_xy, dtype=float)) ** 2, axis=1)
        self.weights *= np.exp(-0.5 * d2 / std ** 2) + 1e-300
        self.weights /= self.weights.sum()
        if self.effective_sample_size() < 0.5 * len(self.weights):
            self._resample(jitter_scale=0.5 * std)

    def next_press(self, num_candidates: int = 400) -> np.ndarray:
        """Pick the press that covers the most remaining probability.

        Candidates are drawn from the particles themselves. Covering the most
        mass maximizes what a "no drop" result eliminates, and it is also the
        press most likely to find the hole.
        """
        idx = self.rng.choice(len(self.particles), size=min(num_candidates, len(self.particles)),
                              replace=False, p=self.weights)
        candidates = self.particles[idx]
        dists = np.linalg.norm(candidates[:, None, :] - self.particles[None, :, :], axis=2)
        mass = (dists < self.detect_radius) @ self.weights
        return candidates[int(np.argmax(mass))].copy()

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """Weighted mean (x, y) and 2x2 covariance of the hole centre."""
        mean = self.weights @ self.particles
        diff = self.particles - mean
        cov = (self.weights[:, None] * diff).T @ diff
        return mean, cov

    def effective_sample_size(self) -> float:
        return 1.0 / np.sum(self.weights ** 2)

    def _resample(self, jitter_scale: float | None = None) -> None:
        """Systematic resampling, plus a small jitter to keep particles diverse."""
        n = len(self.weights)
        positions = (self.rng.uniform() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.weights), positions), n - 1)
        scale = 0.05 * self.detect_radius if jitter_scale is None else jitter_scale
        jitter = self.rng.normal(scale=scale, size=(n, 2))
        self.particles = self.particles[idx] + jitter
        self.weights = np.full(n, 1.0 / n)
