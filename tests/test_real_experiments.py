import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from kilo.config import REPO_ROOT
from kilo.robot.interface import ProbeResult, RobotState, SafetyStop
from scripts.real.phase1 import RunLog, cmd_touch


def test_touch_saves_completed_probe_before_failed_return(tmp_path):
    cfg = yaml.safe_load((REPO_ROOT / "configs/real.yaml").read_text())
    cfg["motion"]["align_wrist"] = False
    state = RobotState(0, np.array([.36, -.075, .1]), 0., np.zeros(3), np.zeros(3))
    result = ProbeResult(False, state.tip + [.12, 0, 0], np.array([2., 0, 0]),
                         2.14, .117, np.zeros(3), [[0, *state.tip, 0, 0, 0]])
    calls = []
    def move(*args):
        calls.append(args)
        if len(calls) == 2:
            raise SafetyStop("return failed")
    robot = SimpleNamespace(state=lambda: state, move=move, probe=lambda *args: result)
    args = SimpleNamespace(backend="sim", direction=[1, 0], distance=.12, repeats=1)
    log = RunLog("touch", tmp_path)
    with pytest.raises(SafetyStop, match="return failed"):
        cmd_touch(args, cfg, robot, log)
    saved = json.loads((log.dir / "result.json").read_text())
    assert saved["probes"][0]["contact"] is False
    assert saved["probes"][0]["peak"] == 2.14
    assert saved["probes"][0]["distance"] == .117
    assert (log.dir / "samples.npz").exists()
