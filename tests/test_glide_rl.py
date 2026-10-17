from pathlib import Path
import torch

from rl.ppo_agent import ActorCritic
from sim.glide_gps_env import GlideGpsTargetEnv
from sim.glide_rl_env import GlidePulloutEnv


def test_glide_rl_env_spaces_and_step():
    env = GlidePulloutEnv(release_altitude_m=1000.0, randomize_drop=False)
    obs, info = env.reset(seed=42, options={"deterministic": True})
    assert obs.shape == (10,)
    assert env.action_space.shape == (2,)
    assert info["altitude_m"] == 1000.0
    assert info["pitch_deg"] < -85.0


def test_trained_glide_rl_policy_reaches_over_1300m_unshaped():
    ckpt_untrained = Path("checkpoints/glide_rl/stage_0_untrained.pt")
    ckpt_trained = Path("checkpoints/glide_rl/stage_4_master.pt")
    assert ckpt_untrained.exists() and ckpt_trained.exists()

    def run_policy(ckpt_path: Path):
        model = ActorCritic(obs_dim=10, act_dim=2, hidden_dim=64)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        env = GlidePulloutEnv(release_altitude_m=1000.0, randomize_drop=False)
        obs, info = env.reset(seed=100, options={"deterministic": True})
        max_pitch = info["pitch_deg"]
        while True:
            act = model.act_deterministic(obs)
            obs, _, term, trunc, info = env.step(act)
            if info["pitch_deg"] > max_pitch:
                max_pitch = info["pitch_deg"]
            if term or trunc:
                break
        return info["xy_distance_m"], max_pitch

    dist_0, _ = run_policy(ckpt_untrained)
    dist_4, pitch_4 = run_policy(ckpt_trained)

    assert dist_0 < 25.0
    assert dist_4 > 1300.0
    assert pitch_4 > -25.0


def test_gps_target_curriculum_stages_static_linear_aggressive():
    stages = [
        ("gps_1_static", "static", 25.0),
        ("gps_2_linear", "linear", 30.0),
        ("gps_3_aggressive", "aggressive", 190.0),
    ]
    for stage_id, mode, max_allowed_err in stages:
        ckpt_path = Path(f"checkpoints/glide_rl/{stage_id}.pt")
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ActorCritic(obs_dim=14, act_dim=2, hidden_dim=64)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        env = GlideGpsTargetEnv(target_mode=mode, release_altitude_m=1000.0, randomize_drop=False)
        obs, info = env.reset(seed=200, options={"deterministic": True})
        max_bank = 0.0
        while True:
            act = model.act_deterministic(obs)
            obs, _, term, trunc, info = env.step(act)
            max_bank = max(max_bank, abs(info["roll_deg"]))
            if term or trunc:
                break
        assert info["target_error_m"] < max_allowed_err
        if mode == "aggressive":
            assert max_bank > 45.0
