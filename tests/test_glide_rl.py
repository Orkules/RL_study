from pathlib import Path
import torch

from rl.ppo_agent import ActorCritic
from sim.glide_gps_env import GlideGpsTargetEnv
from sim.glide_rl_env import GlidePulloutEnv
from sim.pursuit_evasion_env import PursuitEvasionAdversarialEnv
from train_pursuit_evasion_loop import evaluate_10_duel_trials


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


def test_3d_balls_and_ground_and_linear_and_curved_targets():
    stages = [
        ("gps_1_static", "balls_3d", 35.0),
        ("gps_1_static", "static", 35.0),
        ("gps_2_linear", "linear", 35.0),
        ("gps_3_aggressive", "curved", 45.0),
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
        while True:
            act = model.act_deterministic(obs)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        assert info["target_error_m"] < max_allowed_err
        if mode == "balls_3d":
            assert info["mean_ball_min_dist_m"] < 55.0


def test_adversarial_pursuit_evasion_car_vs_missile_50_to_300_kmh():
    m_ckpt = Path("checkpoints/glide_rl/gps_4_car_pursuit.pt")
    c_ckpt = Path("checkpoints/glide_rl/car_evader_best.pt")
    assert m_ckpt.exists() and c_ckpt.exists()

    missile_model = ActorCritic(obs_dim=14, act_dim=2, hidden_dim=64)
    car_model = ActorCritic(obs_dim=16, act_dim=2, hidden_dim=64)
    missile_model.load_state_dict(torch.load(m_ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    car_model.load_state_dict(torch.load(c_ckpt, map_location="cpu", weights_only=False)["model_state_dict"])

    for spd_kmh in [50.0, 110.0, 210.0, 300.0]:
        res = evaluate_10_duel_trials(missile_model, car_model, speed_kmh=spd_kmh, num_trials=10, seed=5000)
        assert res["hits"] >= 9
