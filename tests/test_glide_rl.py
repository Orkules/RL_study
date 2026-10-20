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


def test_24d_inav_ground_and_linear_and_curved_targets():
    stages = [
        ("gps_1_static", "static", 15.0),
        ("gps_2_linear", "linear", 20.0),
        ("gps_3_aggressive", "curved", 65.0),
    ]
    for stage_id, mode, max_allowed_err in stages:
        ckpt_path = Path(f"checkpoints/glide_rl/{stage_id}.pt")
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ActorCritic(obs_dim=24, act_dim=2, hidden_dim=64)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        env = GlideGpsTargetEnv(target_mode=mode, release_altitude_m=1000.0, randomize_drop=False)
        obs, info = env.reset(seed=200, options={"deterministic": True})
        assert obs.shape == (24,)
        assert info["boundary_radius_m"] == 930.0
        assert info["boundary_diameter_m"] == 1860.0
        while True:
            act = model.act_deterministic(obs)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        assert info["target_error_m"] < max_allowed_err

    # Also verify 100 m, 200 m, 600 m, and 800 m targets in all 4 quadrants hit within < 3.0 m average error and > 88% shortest-path efficiency!
    ckpt1 = torch.load("checkpoints/glide_rl/gps_1_static.pt", map_location="cpu", weights_only=False)
    m1 = ActorCritic(obs_dim=24, act_dim=2, hidden_dim=64)
    m1.load_state_dict(ckpt1["model_state_dict"])
    errs_q, effs_q = [], []
    for tn, te in [(100.0, 0.0), (0.0, 100.0), (200.0, 0.0), (0.0, 200.0), (-200.0, 0.0), (-500.0, -400.0), (600.0, 400.0), (0.0, -800.0)]:
        env_q = GlideGpsTargetEnv(target_mode="static", release_altitude_m=1000.0, randomize_drop=False)
        obs, info = env_q.reset(options={"deterministic": True, "target_ne": [tn, te]})
        while True:
            obs, _, tm, tr, info = env_q.step(m1.act_deterministic(obs))
            if tm or tr:
                break
        errs_q.append(info["target_error_m"])
        effs_q.append(info["shortest_path_efficiency"])
    assert sum(errs_q) / len(errs_q) <= 3.0
    assert sum(effs_q) / len(effs_q) >= 0.88


def test_adversarial_pursuit_evasion_car_vs_missile_strict_boundary_and_hit():
    m_ckpt = Path("checkpoints/glide_rl/gps_4_car_pursuit.pt")
    c_ckpt = Path("checkpoints/glide_rl/car_evader_best.pt")
    assert m_ckpt.exists() and c_ckpt.exists()

    missile_model = ActorCritic(obs_dim=24, act_dim=2, hidden_dim=64)
    car_model = ActorCritic(obs_dim=16, act_dim=2, hidden_dim=64)
    missile_model.load_state_dict(torch.load(m_ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    car_model.load_state_dict(torch.load(c_ckpt, map_location="cpu", weights_only=False)["model_state_dict"])

    res50 = evaluate_10_duel_trials(missile_model, car_model, speed_kmh=50.0, num_trials=10, seed=5000)
    assert res50["hits"] >= 9
    assert res50["mean_min_dist_m"] <= 3.0

    # Also verify edge cases on gps_1_static.pt (High-Speed 90-deg Left Break, 90-deg Right Break, 180-deg Rear Break, Upside-Down Launch):
    s_ckpt = Path("checkpoints/glide_rl/gps_1_static.pt")
    m1 = ActorCritic(obs_dim=24, act_dim=2, hidden_dim=64)
    m1.load_state_dict(torch.load(s_ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    env_ec = GlideGpsTargetEnv(target_mode="static", release_altitude_m=1000.0, randomize_drop=False)
    for ec_opts in [
        {"altitude_m": 500.0, "angles_deg": [0.0, -15.0, 0.0], "speed_ms": 42.0, "target_ne": [100.0, -320.0]},
        {"altitude_m": 500.0, "angles_deg": [0.0, -15.0, 0.0], "speed_ms": 42.0, "target_ne": [100.0, +320.0]},
        {"altitude_m": 700.0, "angles_deg": [0.0, -20.0, 0.0], "speed_ms": 35.0, "target_ne": [-380.0, 50.0]},
        {"altitude_m": 650.0, "angles_deg": [180.0, -25.0, 45.0], "speed_ms": 38.0, "target_ne": [-300.0, -300.0]},
        {"altitude_m": 600.0, "angles_deg": [90.0, -30.0, -60.0], "speed_ms": 40.0, "target_ne": [280.0, 260.0]},
    ]:
        obs_ec, info_ec = env_ec.reset(options={"deterministic": True, **ec_opts})
        while True:
            obs_ec, _, tm_ec, tr_ec, info_ec = env_ec.step(m1.act_deterministic(obs_ec))
            if tm_ec or tr_ec:
                break
        assert info_ec["target_error_m"] <= 3.0


def test_fpv_powered_3d_balls_env_and_checkpoint():
    from sim.fpv_balls_env import FpvPoweredBallsEnv

    ckpt_path = Path("checkpoints/glide_rl/fpv_balls_master.pt")
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ActorCritic(obs_dim=25, act_dim=3, hidden_dim=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = FpvPoweredBallsEnv(num_balls=5, randomize_start=False)
    obs, info = env.reset(options={"deterministic": True})
    assert obs.shape == (25,)
    assert info["shortest_path_length_m"] > 500.0
    assert info["ideal_min_time_s"] > 5.0

    max_spd = 0.0
    while True:
        act = model.act_deterministic(obs)
        assert act.shape == (3,)
        obs, _, term, trunc, info = env.step(act)
        max_spd = max(max_spd, info["airspeed_ms"])
        if term or trunc:
            break

    assert info["balls_reached"] >= 1
    assert max_spd > 55.0
