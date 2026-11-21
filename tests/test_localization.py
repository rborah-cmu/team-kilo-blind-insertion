import numpy as np

from kilo.localization import ContactDetector, HoleSearchFilter


def test_contact_detector_needs_consecutive_samples():
    det = ContactDetector(threshold=1.0, debounce_samples=3)
    det.calibrate(np.zeros((10, 7)))
    spike = np.full(7, 0.0)
    spike[2] = 2.0
    quiet = np.zeros(7)
    assert not det.update(spike)
    assert not det.update(quiet)  # streak broken
    assert not det.update(spike)
    assert not det.update(spike)
    assert det.update(spike)


def test_contact_detector_removes_bias():
    det = ContactDetector(threshold=1.0, debounce_samples=1)
    det.calibrate(np.full((5, 7), 3.0))
    assert not det.update(np.full(7, 3.5))
    assert det.update(np.full(7, 4.5))


def _search(seed, R=0.02, detect=0.005, max_presses=60):
    rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.uniform())
    a = rng.uniform(0, 2 * np.pi)
    true_hole = np.array([0.5 + r * np.cos(a), r * np.sin(a)])
    f = HoleSearchFilter(center=[0.5, 0.0], max_offset=R, detect_radius=detect,
                         num_particles=3000, rng=rng)
    for n in range(1, max_presses + 1):
        p = f.next_press()
        dropped = np.linalg.norm(p - true_hole) < detect
        f.update_press(p, dropped)
        if dropped:
            return n, true_hole, f
    return None, true_hole, f


def test_hole_search_finds_hole_within_budget():
    # Worst case is about (R / detect)^2 = 16 presses; allow margin.
    for seed in range(10):
        n, true_hole, f = _search(seed)
        assert n is not None and n <= 30, f"seed {seed}: not found in budget"
        mean, _ = f.estimate()
        # A single drop only confines the hole to the press neighbourhood.
        assert np.linalg.norm(mean - true_hole) < 2 * f.detect_radius


def test_seated_tip_pins_down_hole_centre():
    for seed in range(10):
        n, true_hole, f = _search(seed)
        rng = np.random.default_rng(100 + seed)
        seated = true_hole + rng.normal(scale=0.0005, size=2)
        f.update_seated(seated, std=0.001)
        mean, _ = f.estimate()
        assert np.linalg.norm(mean - true_hole) < 0.0015, f"seed {seed}"


def test_estimate_covariance_shrinks_after_drop():
    _, _, f0 = _search(0, max_presses=0)
    _, cov0 = f0.estimate()
    n, _, f = _search(0)
    _, cov = f.estimate()
    assert np.trace(cov) < np.trace(cov0)
