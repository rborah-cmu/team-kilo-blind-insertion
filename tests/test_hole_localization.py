import numpy as np
import pytest

from kilo.config import load_config
from kilo.localization.hole import localize_hole


@pytest.mark.parametrize("outcome", ["found", "unconfirmed", "no_surface_reference", "incomplete_press"])
def test_hole_search_uses_measured_depth_and_contact(outcome):
    cfg = load_config()
    cfg["hole_search"]["max_presses"] = 4
    center = np.array([.51, -.07])
    top = .058
    hole = center + cfg["hole"]["cad_offset"]
    targets = []

    def press(xy, target_z, label):
        targets.append(np.asarray(xy))
        if label == "descent-reference":
            if outcome == "no_surface_reference":
                return np.r_[xy, target_z], False
            return np.r_[xy, top], True
        assert target_z < top - cfg["hole_search"]["drop_depth"]
        if outcome == "found":
            return np.r_[xy, target_z + .0002], False
        if outcome == "incomplete_press":
            return np.r_[xy, top - .001], False
        return np.r_[xy, top - .0001], True

    result = {}
    localize_hole(center, 0., top, cfg, press, result, rng=0)
    assert result["status"] == outcome
    if outcome == "found":
        np.testing.assert_allclose(result["center"], hole, atol=.0002)
        assert len(result["presses"]) == 1
    elif outcome == "unconfirmed":
        assert len(result["presses"]) == cfg["hole_search"]["max_presses"]
        np.testing.assert_allclose(targets[1], hole)
        np.testing.assert_allclose(targets[2], 2 * center - hole)
        assert len(np.unique(targets, axis=0)) == len(targets)
