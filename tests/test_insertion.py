import numpy as np
import mujoco
import torch
from stable_baselines3 import PPO

from kilo.config import load_config
from kilo.insertion.env import FRAME_FEATURES, PegInsertionEnv
from kilo.insertion.policy import AsymmetricInsertionPolicy
from kilo.insertion.reference import InsertionReference


def test_aligned_spiral_baseline_reaches_success_depth():
    cfg = load_config()
    cfg["insertion"]["train_noise_xy"] = 0.0
    cfg["insertion"]["grasp_offset_std"] = 0.0
    cfg["insertion"]["grasp_tilt_std_rad"] = 0.0
    env = PegInsertionEnv(config=cfg)
    try:
        observation, _ = env.reset(seed=123)
        assert observation['history'].shape == (cfg["insertion"]["history_frames"] * FRAME_FEATURES,)
        assert env.observation_space.contains(observation)

        for _ in range(env.max_steps):
            observation, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            if terminated or truncated:
                break

        assert info["success"]
        assert info["depth"] >= cfg["insertion"]["success_depth"]
        assert info["radial_error"] <= (
            cfg["hole"]["radius"] - cfg["peg"]["shaft_radius"]
            + cfg["insertion"]["success_xy_tolerance"]
        )
    finally:
        env.close()


def test_reference_limits_preload_and_stops_sweeping_on_capture():
    cfg = load_config()['insertion']
    tip = np.array([0., 0., 0.05])
    ref = InsertionReference(cfg, tip[:2], tip[2], tip)
    # The measured touch plane differs from the fixture height by 0.7 mm.
    tip[2] += 0.0007
    for _ in range(100):
        target = ref.step(tip, 0.0381, force_z=0.5)
        assert target[2] >= tip[2] - cfg['search_z_lead'] - 1e-12
    assert ref.phase == ref.SEARCH
    tip[2] -= 0.0006
    target = ref.step(tip, 0.0381, force_z=0.5)
    assert ref.phase == ref.CAPTURE
    locked_xy = target[:2].copy()
    tip[0] += 0.0002
    tip[2] -= 0.004
    target = ref.step(tip, 0.0381, force_z=0.5)
    assert ref.phase == ref.INSERT
    np.testing.assert_array_equal(target[:2], locked_xy)


def test_grasp_error_does_not_change_nominal_tool_observation():
    env = PegInsertionEnv(use_tau_ext=False)
    try:
        env.reset(seed=123)
        before = env._current_frame().copy()
        actual_before = env._site_pos(env.peg_site_id)
        env.model.body_pos[env.peg_body_id] += [0.001, -0.002, 0.001]
        mujoco.mj_forward(env.model, env.data)
        assert np.linalg.norm(actual_before - env._site_pos(env.peg_site_id)) > 0.002
        np.testing.assert_allclose(env._current_frame(), before, atol=1e-7)
    finally:
        env.close()


def test_randomized_reset_has_no_peg_penetration():
    env = PegInsertionEnv()
    try:
        # This draw produced a 5 kN artificial impulse with the old 1 mm hover.
        observation, _ = env.reset(seed=20013)
        assert all(np.isfinite(value).all() for value in observation.values())
        for contact in env.data.contact:
            if env.peg_body_id in env.model.geom_bodyid[[contact.geom1, contact.geom2]]:
                assert contact.dist >= 0.0
    finally:
        env.close()


def test_privileged_state_changes_value_but_cannot_change_actor(tmp_path):
    torch.set_num_threads(1)
    env = PegInsertionEnv()
    try:
        model = PPO(AsymmetricInsertionPolicy, env, n_steps=8, batch_size=8, seed=0)
        obs, _ = env.reset(seed=123)
        tensor, _ = model.policy.obs_to_tensor(obs)
        action_before, _ = model.predict(obs, deterministic=True)
        value_before = model.policy.predict_values(tensor)
        obs['privileged'] += 10.0
        tensor, _ = model.policy.obs_to_tensor(obs)
        action_after, _ = model.predict(obs, deterministic=True)
        value_after = model.policy.predict_values(tensor)
        np.testing.assert_array_equal(action_before, action_after)
        assert not torch.allclose(value_before, value_after)
        # A critic update may teach the shared encoder, but direct privileged
        # inputs must have zero derivative into the action distribution.
        tensor['privileged'].requires_grad_(True)
        mean = model.policy.get_distribution(tensor).distribution.mean
        grad = torch.autograd.grad(mean.sum(), tensor['privileged'])[0]
        assert torch.count_nonzero(grad) == 0
        path = tmp_path / 'policy'
        model.save(path)
        restored = PPO.load(path, env=env)
        np.testing.assert_array_equal(restored.predict(obs, deterministic=True)[0], action_before)
    finally:
        env.close()
