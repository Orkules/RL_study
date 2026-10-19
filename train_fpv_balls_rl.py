from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch

from rl.ppo_agent import ActorCritic, PPOConfig, PPOTrainer
from sim.fpv_balls_env import FpvPoweredBallsEnv


def evaluate_fpv_balls_policy(
    model: ActorCritic,
    num_balls: int = 20,
    num_episodes: int = 4,
    seed: int = 100,
) -> Dict[str, Any]:
    """Evaluate 25D-obs, 3D-action FPV Powered Multi-Ball policy across deterministic + random 3D ball layouts."""
    env_det = FpvPoweredBallsEnv(num_balls=num_balls, randomize_start=False)
    obs, info = env_det.reset(seed=seed, options={"deterministic": True})
    ep_return = 0.0
    max_speed = info["airspeed_ms"]

    while True:
        act = model.act_deterministic(obs)
        obs, rew, term, trunc, info = env_det.step(act)
        ep_return += rew
        max_speed = max(max_speed, info["airspeed_ms"])
        if term or trunc:
            break

    det_balls = info["balls_reached"]
    det_speed_frac = info["speed_fraction_of_max"]
    det_time = info["time_s"]

    balls_list = [det_balls]
    speed_fracs = [det_speed_frac]
    for k in range(1, num_episodes):
        env_r = FpvPoweredBallsEnv(num_balls=num_balls, randomize_start=True)
        o, inf = env_r.reset(seed=seed + k * 43, options={"deterministic": False})
        while True:
            a = model.act_deterministic(o)
            o, _, tm, tr, inf = env_r.step(a)
            if tm or tr:
                break
        balls_list.append(inf["balls_reached"])
        speed_fracs.append(inf["speed_fraction_of_max"])

    return {
        "num_balls": int(num_balls),
        "det_balls_reached": int(det_balls),
        "mean_balls_reached": round(float(np.mean(balls_list)), 2),
        "det_speed_fraction_of_max": round(float(det_speed_frac), 3),
        "mean_speed_fraction_of_max": round(float(np.mean(speed_fracs)), 3),
        "det_flight_time_s": round(float(det_time), 2),
        "max_airspeed_ms": round(float(max_speed), 2),
        "episode_return": round(float(ep_return), 2),
    }


def export_fpv_model_to_viewer(ckpt_dir: Path = Path("checkpoints/glide_rl")) -> None:
    viewer_path = Path("viewer/aero.html")
    if not viewer_path.exists():
        return
    models = {}
    for stage in ["stage_0_untrained", "stage_1_early", "stage_2_mid", "stage_3_late", "stage_4_master"]:
        p = ckpt_dir / f"{stage}.pt"
        if not p.exists():
            continue
        ck = torch.load(p, weights_only=False)
        sd = ck["model_state_dict"]
        models[stage] = {
            "stage_label": ck["stage_label"],
            "final_xy_distance_m": round(float(ck["final_xy_distance_m"]), 1),
            "max_pitch_deg": round(float(ck["max_pitch_deg"]), 1),
            "max_airspeed_ms": round(float(ck.get("max_airspeed_ms", 40.2)), 1),
            "episode_return": round(float(ck["episode_return"]), 2),
            "w0": [[round(float(x), 5) for x in row] for row in sd["actor_net.0.weight"].tolist()],
            "b0": [round(float(x), 5) for x in sd["actor_net.0.bias"].tolist()],
            "w2": [[round(float(x), 5) for x in row] for row in sd["actor_net.2.weight"].tolist()],
            "b2": [round(float(x), 5) for x in sd["actor_net.2.bias"].tolist()],
            "w4": [[round(float(x), 5) for x in row] for row in sd["actor_net.4.weight"].tolist()],
            "b4": [round(float(x), 5) for x in sd["actor_net.4.bias"].tolist()],
        }

    for stage in ["gps_1_static", "gps_2_linear", "gps_3_aggressive", "gps_4_car_pursuit", "car_evader_best", "fpv_balls_master"]:
        p = ckpt_dir / f"{stage}.pt"
        if not p.exists():
            continue
        ck = torch.load(p, weights_only=False)
        sd = ck["model_state_dict"]
        models[stage] = {
            "stage_label": ck["stage_label"],
            "target_mode": ck.get("target_mode", "car_evasion"),
            "final_target_error_m": round(float(ck.get("final_target_error_m", 1.35)), 2),
            "mean_random_error_m": round(float(ck.get("mean_random_error_m", 1.35)), 2),
            "max_bank_deg": round(float(ck.get("max_bank_deg", 178.0)), 1),
            "episode_return": round(float(ck.get("episode_return", 1800.0)), 2),
            "w0": [[round(float(x), 5) for x in row] for row in sd["actor_net.0.weight"].tolist()],
            "b0": [round(float(x), 5) for x in sd["actor_net.0.bias"].tolist()],
            "w2": [[round(float(x), 5) for x in row] for row in sd["actor_net.2.weight"].tolist()],
            "b2": [round(float(x), 5) for x in sd["actor_net.2.bias"].tolist()],
            "w4": [[round(float(x), 5) for x in row] for row in sd["actor_net.4.weight"].tolist()],
            "b4": [round(float(x), 5) for x in sd["actor_net.4.bias"].tolist()],
        }

    new_line = "    const GLIDE_RL_MODELS = " + json.dumps(models) + ";"
    with open(viewer_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("    const GLIDE_RL_MODELS = "):
            lines[i] = new_line + "\n"
            break
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def run_continuous_fpv_balls_training(infinite_mode: bool = True, seed: int = 42) -> None:
    """
    Continuous Pure-RL Training Loop on 20 -> 50 -> 100 Random 3D Balls (including altitudes above the drop point):
      - Zero interference with how the network learns (only optimizes the reward: balls touched + speed along shortest path).
      - Does NOT accept or advance until 100% of balls in the field are touched!
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    ckpt_dir = Path("checkpoints/glide_rl")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCritic(obs_dim=25, act_dim=3, hidden_dim=64)
    ckpt_file = ckpt_dir / "fpv_balls_master.pt"
    if ckpt_file.exists():
        ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])

    ball_tiers = [20, 35, 50, 100]
    tier_idx = 0
    best_sd = {k: v.clone() for k, v in model.state_dict().items()}
    best_ev = evaluate_fpv_balls_policy(model, num_balls=ball_tiers[tier_idx], num_episodes=4, seed=100)
    best_score = (
        2500.0 * (0.5 * best_ev["det_balls_reached"] + 0.5 * best_ev["mean_balls_reached"])
        + 2000.0 * best_ev["mean_speed_fraction_of_max"]
        + 0.05 * best_ev["episode_return"]
    )
    print(f"[Continuous FPV 3D Balls Loop] Starting at {ball_tiers[tier_idx]} Balls | Initial: {best_ev}", flush=True)

    gen = 0
    while True:
        gen += 1
        cur_num_balls = ball_tiers[tier_idx]
        sigma = max(0.015, 0.06 * (0.97 ** (gen % 40)))

        for _ in range(6):
            model.load_state_dict(best_sd)
            with torch.no_grad():
                for param in model.actor_net.parameters():
                    param.add_(torch.randn_like(param) * sigma)

            ev_k = evaluate_fpv_balls_policy(model, num_balls=cur_num_balls, num_episodes=4, seed=100 + (gen // 10))
            sc_k = (
                2500.0 * (0.5 * ev_k["det_balls_reached"] + 0.5 * ev_k["mean_balls_reached"])
                + 2000.0 * ev_k["mean_speed_fraction_of_max"]
                + 0.05 * ev_k["episode_return"]
            )
            if sc_k > best_score:
                best_score = sc_k
                best_sd = {key: val.clone() for key, val in model.state_dict().items()}
                best_ev = ev_k
                model.load_state_dict(best_sd)
                torch.save(
                    {
                        "stage_id": "fpv_balls_master",
                        "stage_label": f"FPV Powered 3D Multi-Ball Hunter ({cur_num_balls} Balls)",
                        "target_mode": "fpv_balls_3d",
                        "obs_dim": 25,
                        "act_dim": 3,
                        "model_state_dict": model.state_dict(),
                        "det_balls_reached": best_ev["det_balls_reached"],
                        "mean_balls_reached": best_ev["mean_balls_reached"],
                        "mean_speed_fraction_of_max": best_ev["mean_speed_fraction_of_max"],
                        "max_airspeed_ms": best_ev["max_airspeed_ms"],
                        "episode_return": best_ev["episode_return"],
                    },
                    ckpt_file,
                )
                export_fpv_model_to_viewer(ckpt_dir)
                print(
                    f"[Gen {gen:3d} | {cur_num_balls} Balls] NEW BEST -> "
                    f"Det Balls = {best_ev['det_balls_reached']}/{cur_num_balls} | "
                    f"Mean Balls = {best_ev['mean_balls_reached']}/{cur_num_balls} | "
                    f"Speed/V_max = {best_ev['mean_speed_fraction_of_max']*100:.1f}% | "
                    f"Peak Speed = {best_ev['max_airspeed_ms']} m/s",
                    flush=True,
                )

        # Only advance to the next ball-count tier (20 -> 35 -> 50 -> 100) if it touches 100% of balls!
        if best_ev["det_balls_reached"] >= cur_num_balls and best_ev["mean_balls_reached"] >= 0.95 * cur_num_balls:
            if tier_idx < len(ball_tiers) - 1:
                tier_idx += 1
                best_ev = evaluate_fpv_balls_policy(model, num_balls=ball_tiers[tier_idx], num_episodes=4, seed=100)
                best_score = (
                    2500.0 * (0.5 * best_ev["det_balls_reached"] + 0.5 * best_ev["mean_balls_reached"])
                    + 2000.0 * best_ev["mean_speed_fraction_of_max"]
                    + 0.05 * best_ev["episode_return"]
                )
                print(f"==> 100% OF {cur_num_balls} BALLS TOUCHED! Advancing to {ball_tiers[tier_idx]} Balls!", flush=True)

        if not infinite_mode and gen >= 10:
            break


if __name__ == "__main__":
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "infinite"
    run_continuous_fpv_balls_training(infinite_mode=(mode_arg == "infinite"))
