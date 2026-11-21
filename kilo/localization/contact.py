"""Contact detection from the Franka's estimated external joint torques."""

from __future__ import annotations

import numpy as np


class ContactDetector:
    """Flags contact when |tau_ext - bias| exceeds a threshold for several samples.

    The debounce (``debounce_samples`` consecutive samples over threshold) rejects
    single-sample spikes from model error. Call :meth:`calibrate` with free-space
    samples first so a constant model bias is not mistaken for contact.
    """

    def __init__(self, threshold: float | np.ndarray, debounce_samples: int = 5):
        if debounce_samples < 1:
            raise ValueError("debounce_samples must be at least 1")
        self.threshold = np.asarray(threshold, dtype=float)
        self.debounce_samples = debounce_samples
        self.bias: np.ndarray | None = None
        self._count = 0

    def calibrate(self, free_space_samples: np.ndarray) -> None:
        """Set the bias to the mean tau_ext over samples taken with no contact."""
        samples = np.atleast_2d(np.asarray(free_space_samples, dtype=float))
        self.bias = samples.mean(axis=0)
        self.reset()

    def reset(self) -> None:
        self._count = 0

    def update(self, tau_ext: np.ndarray) -> bool:
        """Feed one tau_ext sample (one value per joint). Returns True once in contact."""
        tau = np.asarray(tau_ext, dtype=float)
        residual = tau - self.bias if self.bias is not None else tau
        over = bool(np.any(np.abs(residual) > self.threshold))
        self._count = self._count + 1 if over else 0
        return self._count >= self.debounce_samples
