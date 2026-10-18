from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch

from rl.ppo_agent import ActorCritic, PPOConfig, PPOTrainer
from sim.glide_gps_env import GlideGpsTargetEnv


def evaluate_gps_policy(
    model: ActorCritic,
    target_mode: str = "aggressive",
    altitude_m: float = 1000.0,
    num_episodes: int = 5,
    seed: int = 200,
) -> Dict[str, Any]:
    """Evaluate 14D policy on 1,000 m GPS target distance-decrease across deterministic + random test cases."""
    env_det = GlideGpsTargetEnv(target_mode=target_mode, release_altitude_m=altitude_m, randomize_drop=False)
    obs, info = env_det.reset(seed=seed, options={"deterministic": True, "altitude_m": altitude_m})
    init_dist = info["target_error_m"]
    trajectory: List[Dict[str, Any]] = [info]
    ep_return = 0.0
    max_bank_deg = 0.0
    inverted_steps = 0

    while True:
        act = model.act_deterministic(obs)
        obs, rew, term, trunc, info = env_det.step(act)
        ep_return += rew
        max_bank_deg = max(max_bank_deg, abs(info["roll_deg"]))
        if abs(info["roll_deg"]) > 120.0:
            inverted_steps += 1
        trajectory.append(info)
        if term or trunc:
            break

    det_closed = info["total_distance_closed_m"]
    det_final_err = info["target_error_m"]

    closed_list = [det_closed]
    errs = [det_final_err]
    for k in range(1, num_episodes):
        env_r = GlideGpsTargetEnv(target_mode=target_mode, release_altitude_m=altitude_m, randomize_drop=True)
        o, inf = env_r.reset(seed=seed + k * 37)
        while True:
            a = model.act_deterministic(o)
            o, _, tm, tr, inf = env_r.step(a)
            if tm or tr:
                break
        closed_list.append(float(inf["total_distance_closed_m"]))
        errs.append(float(inf["target_error_m"]))

    return {
        "target_mode": target_mode,
        "initial_target_dist_m": round(float(init_dist), 2),
        "total_distance_closed_m": round(float(det_closed), 2),
        "mean_random_closed_m": round(float(np.mean(closed_list)), 2),
        "final_target_error_m": round(float(det_final_err), 2),
        "mean_random_error_m": round(float(np.mean(errs)), 2),
        "episode_return": round(float(ep_return), 2),
        "max_bank_deg": round(float(max_bank_deg), 2),
        "inverted_flight_s": round(float(inverted_steps * env_det.control_dt), 2),
        "flight_time_s": round(float(info["time_s"]), 2),
        "trajectory": trajectory,
    }


def run_pure_ppo_stage(
    model: ActorCritic,
    target_mode: str,
    stage_id: str,
    stage_label: str,
    num_updates: int = 20,
    num_envs: int = 6,
    rollout_steps: int = 512,
    lr: float = 1e-3,
    seed: int = 100,
    ckpt_dir: Path = Path("checkpoints/glide_rl"),
) -> Dict[str, Any]:
    """Train 100% pure PPO from scratch — rewarded strictly for decreasing distance to target!"""
    envs = [GlideGpsTargetEnv(target_mode=target_mode, release_altitude_m=1000.0, randomize_drop=True) for _ in range(num_envs)]
    ppo = PPOTrainer(
        model,
        PPOConfig(
            lr=lr,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            ent_coef=0.003,
            update_epochs=8,
            minibatch_size=256,
        ),
    )

    obs_list = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed + i)
        obs_list.append(o)
    obs = np.stack(obs_list, axis=0)

    best_score = -1e9
    best_state = None
    best_eval = None

    for update in range(num_updates + 1):
        ev = evaluate_gps_policy(model, target_mode=target_mode, altitude_m=1000.0, num_episodes=5, seed=700)
        # Maximize total distance decreased toward target across deterministic and random episodes!
        score = 0.5 * ev["total_distance_closed_m"] + 0.5 * ev["mean_random_closed_m"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_eval = ev

        if update % 4 == 0 or update == num_updates:
            print(
                f"[{stage_label}] update={update:2d}/{num_updates} | "
                f"Dist Decreased = {ev['total_distance_closed_m']:+7.1f} m (Random Avg: {ev['mean_random_closed_m']:+7.1f} m) | "
                f"Max Roll = {ev['max_bank_deg']:5.1f}° | "
                f"Inverted Time = {ev['inverted_flight_s']:4.1f}s"
            )

        if update == num_updates:
            break

        frac = 1.0 - (update / float(num_updates))
        for pg in ppo.optimizer.param_groups:
            pg["lr"] = max(1e-4, lr * frac)

        obs_buf = np.zeros((rollout_steps, num_envs, 14), dtype=np.float32)
        act_buf = np.zeros((rollout_steps, num_envs, 2), dtype=np.float32)
        logp_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        rew_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        done_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        val_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        for step in range(rollout_steps):
            obs_buf[step] = obs
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                act_t, logp_t, _, val_t = model.get_action_and_value(obs_t)

            actions = act_t.cpu().numpy()
            act_buf[step] = actions
            logp_buf[step] = logp_t.cpu().numpy()
            val_buf[step] = val_t.cpu().numpy()

            next_obs_list = []
            for env_idx, env in enumerate(envs):
                n_obs, rew, term, trunc, _ = env.step(actions[env_idx])
                done = term or trunc
                rew_buf[step, env_idx] = rew
                done_buf[step, env_idx] = 1.0 if done else 0.0
                if done:
                    n_obs, _ = env.reset()
                next_obs_list.append(n_obs)
            obs = np.stack(next_obs_list, axis=0)

        with torch.no_grad():
            next_val = model.critic_net(torch.as_tensor(obs, dtype=torch.float32)).squeeze(-1).cpu().numpy()

        adv_buf = np.zeros_like(rew_buf)
        last_gae = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(rollout_steps)):
            next_non_terminal = 1.0 - done_buf[t]
            next_values = next_val if t == rollout_steps - 1 else val_buf[t + 1]
            delta = rew_buf[t] + ppo.cfg.gamma * next_values * next_non_terminal - val_buf[t]
            last_gae = delta + ppo.cfg.gamma * ppo.cfg.gae_lambda * next_non_terminal * last_gae
            adv_buf[t] = last_gae

        ret_buf = adv_buf + val_buf

        ppo.update(
            obs_buf.reshape(-1, 14),
            act_buf.reshape(-1, 2),
            logp_buf.reshape(-1),
            ret_buf.reshape(-1),
            adv_buf.reshape(-1),
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_file = ckpt_dir / f"{stage_id}.pt"
    torch.save(
        {
            "stage_id": stage_id,
            "stage_label": stage_label,
            "target_mode": target_mode,
            "obs_dim": 14,
            "model_state_dict": model.state_dict(),
            "total_distance_closed_m": best_eval["total_distance_closed_m"],
            "mean_random_closed_m": best_eval["mean_random_closed_m"],
            "final_target_error_m": best_eval["final_target_error_m"],
            "mean_random_error_m": best_eval["mean_random_error_m"],
            "initial_target_dist_m": best_eval["initial_target_dist_m"],
            "max_bank_deg": best_eval["max_bank_deg"],
            "inverted_flight_s": best_eval["inverted_flight_s"],
            "episode_return": best_eval["episode_return"],
        },
        ckpt_file,
    )
    print(
        f"==> Completed [{stage_label}] | Saved {ckpt_file} | "
        f"Best Dist Decreased = {best_eval['total_distance_closed_m']:+.1f} m | "
        f"Random Avg Decreased = {best_eval['mean_random_closed_m']:+.1f} m | "
        f"Max Roll = {best_eval['max_bank_deg']:.1f}° | "
        f"Inverted Flight = {best_eval['inverted_flight_s']:.1f}s\n"
    )
    return best_eval


def train_all_gps_stages() -> Dict[str, Any]:
    np.random.seed(42)
    torch.manual_seed(42)

    ckpt_dir = Path("checkpoints/glide_rl")
    out_dir = Path("artifacts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCritic(obs_dim=14, act_dim=2, hidden_dim=64)

    t0 = time.time()

    # Stage 1: Static Desired GPS Coordinate (Pure PPO from scratch on distance-decrease)
    res_static = run_pure_ppo_stage(
        model=model,
        target_mode="static",
        stage_id="gps_1_static",
        stage_label="GPS Stage 1: Static Desired Coordinate",
        num_updates=16,
        lr=1.2e-3,
        seed=101,
        ckpt_dir=ckpt_dir,
    )

    # Stage 2: Linear Straight-Line Moving Coordinate (Pure PPO on distance-decrease)
    res_linear = run_pure_ppo_stage(
        model=model,
        target_mode="linear",
        stage_id="gps_2_linear",
        stage_label="GPS Stage 2: Linear Straight-Line Moving Coordinate",
        num_updates=16,
        lr=8e-4,
        seed=202,
        ckpt_dir=ckpt_dir,
    )

    # Stage 3: Mirror-Point Flip Across Axis (Pure PPO on distance-decrease — learns to flip/roll 180° & fly inverted!)
    res_aggr = run_pure_ppo_stage(
        model=model,
        target_mode="aggressive",
        stage_id="gps_3_aggressive",
        stage_label="GPS Stage 3: Mirror-Point Flip & Inverted Flight",
        num_updates=20,
        lr=8e-4,
        seed=303,
        ckpt_dir=ckpt_dir,
    )

    elapsed = round(time.time() - t0, 2)
    summary = {
        "elapsed_seconds": elapsed,
        "gps_1_static": res_static,
        "gps_2_linear": res_linear,
        "gps_3_aggressive": res_aggr,
    }
    with open(out_dir / "glide_gps_training_progress.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"All 3 Pure-PPO GPS stages completed in {elapsed:.1f}s!")
    return summary


if __name__ == "__main__":
    train_all_gps_stages()
