import mujoco
import numpy as np
from kilo.config import load_config, REPO_ROOT
from kilo.insertion.env import PegInsertionEnv


def test_contact_grasp_holds_then_releases():
    cfg = load_config(REPO_ROOT / 'configs/contact_grasp.yaml')
    cfg['insertion']['start_clearance'] = .04
    env = PegInsertionEnv(cfg)
    try:
        env.reset(seed=0)
        assert env.model.joint('peg_free').type[0] == mujoco.mjtJoint.mjJNT_FREE
        assert np.all(env._grip_loads() > 1.)
        before = env._relative_grasp_tip()
        target = env._site_pos(env.control_site_id)
        for _ in range(1000):
            env._controller(target)
            mujoco.mj_step(env.model, env.data)
        env._forward()
        assert np.linalg.norm(env._relative_grasp_tip()-before) < .002
        # Opening the jaws must actually release the peg: no hidden attachment.
        env.data.ctrl[7] = .04
        for _ in range(300):
            env._controller(target)
            mujoco.mj_step(env.model, env.data)
        env._forward()
        assert np.linalg.norm(env._relative_grasp_tip()-before) > .015
        _, _, terminated, _, info = env.step(np.zeros(3))
        assert terminated and info['dropped'] and not info['success']
    finally:
        env.close()


def test_contact_reset_is_repeatable_and_grip_not_counted_as_fixture_force():
    cfg = load_config(REPO_ROOT / 'configs/contact_grasp.yaml')
    env = PegInsertionEnv(cfg)
    try:
        obs, _ = env.reset(seed=2)
        state = env.data.qpos.copy()
        assert env._grip_loads().sum() > 10.
        assert env._contact_load() < .01
        assert env.observation_space.contains(obs)
        env.reset(seed=2)
        np.testing.assert_allclose(env.data.qpos, state, atol=1e-10)
    finally:
        env.close()
