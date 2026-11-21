import numpy as np

from kilo.insertion import spiral_waypoints


def test_spiral_starts_at_center_and_reaches_max_radius():
    c = np.array([0.4, -0.1])
    pts = spiral_waypoints(c, pitch=0.002, max_radius=0.01, step=0.0005)
    assert np.allclose(pts[0], c)
    radii = np.linalg.norm(pts - c, axis=1)
    assert np.isclose(radii.max(), 0.01)
    assert np.all(np.diff(radii) >= -1e-12)  # never moves inward


def test_spiral_steps_are_roughly_even():
    pts = spiral_waypoints([0, 0], pitch=0.002, max_radius=0.01, step=0.0005)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert steps[5:].max() < 0.0006
