from pathlib import Path

import yaml

from kilo.localization.common import (
    EE_FLOAT_HEIGHT,
    FORCE_THRESHOLD,
    SEARCH_SPACE_CORNERS,
    TABLETOP_HEIGHT,
    VIRTUAL_PROBE_LENGTH,
)


def test_snake_search_config_matches_common():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with config_path.open() as f:
        cfg = yaml.safe_load(f)["snake_search"]

    assert SEARCH_SPACE_CORNERS.tolist() == cfg["corners"]
    assert TABLETOP_HEIGHT == cfg["tabletop_height"]
    assert EE_FLOAT_HEIGHT == cfg["ee_float_height"]
    assert VIRTUAL_PROBE_LENGTH == cfg["virtual_probe_length"]
    assert FORCE_THRESHOLD == cfg["force_threshold"]
