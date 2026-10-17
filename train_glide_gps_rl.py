from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from rl.ppo_agent import ActorCritic, PPOConfig, PPOTrainer
from sim.glide_gps_env import GlideGpsTargetEnv


def guidance_teacher_action(obs: np.ndarray, lead_gain: float = 0.65) -> np.ndarray:
    """Proportional-navigation warm-start teacher for 14D GPS coordinate interception."""
    pitch_n = float(obs[0])
    sin_roll = float(obs[1])
    p_norm = float(obs[3])
    r_norm = float(obs[5])
    vd_norm = float(obs[7])
    alt_n = float(obs[8])
    rel_x = float(obs[9])
    rel_y = float(obs[10])
    tv_x = float(obs[11])
    tv_y = float(obs[12])

    vz_mps = max(12.0, vd_norm * 50.0)
    t_go = min(22.0, (alt_n * 1000.0) / vz_mps)
    pred_x = rel_x + (tv_x * 25.0 * lead_gain * t_go) / 1000.0
    pred_y = rel_y + (tv_y * 25.0 * lead_gain * t_go) / 1000.0

    horiz_dist = float(np.hypot(pred_x, pred_y))
    req_glide_ratio = horiz_dist / max(0.04, alt_n)
    pitch_cmd = float(np.clip(0.05 + 0.42 * (req_glide_ratio - 0.35), 0.05, 0.72))

    pullout_gate = float(np.clip((pitch_n + 0.82) * 3.2, 0.0, 1.0))
    bearing = float(np.arctan2(pred_y, max(0.04, pred_x)))
    desired_roll = float(np.clip(1.95 * bearing, -0.85, 0.85)) * pullout_gate
    roll_cmd = float(np.clip(1.35 * (desired_roll - sin_roll) - 0.28 * p_norm - 0.12 * r_norm, -1.0, 1.0))
    return np.array([pitch_cmd, roll_cmd], dtype=np.float32)


def pretrain_actor_for_gps_stage(
    model: ActorCritic,
    target_mode: str,
    lead_gain: float,
    num_episodes: int = 35,
    epochs: int = 25,
    seed: int = 42,
) -> None:
    """Warm-start the 14D ActorCritic on trajectories in the target environment before PPO fine-tuning."""
    env = GlideGpsTargetEnv(target_mode=target_mode, release_altitude_m=1000.0, randomize_drop=True)
    obs_data: List[np.ndarray] = []
    act_data: List[np.ndarray] = []
    ret_data: List[float] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep * 13, options={"deterministic": (ep == 0)})
        ep_obs: List[np.ndarray] = []
        ep_act: List[np.ndarray] = []
        ep_rew: List[float] = []
        while True:
            act = guidance_teacher_action(obs, lead_gain=lead_gain)
            # Add slight exploration noise on half the episodes to cover neighbor states
            if ep % 2 == 1:
                noisy_act = np.clip(act + np.random.normal(0.0, 0.03, size=2).astype(np.float32), -1.0, 1.0)
            else:
                noisy_act = act
            ep_obs.append(obs)
            ep_act.append(act)
            obs, rew, term, trunc, _ = env.step(noisy_act)
            ep_rew.append(float(rew))
            if term or trunc:
                break

        g = 0.0
        ep_ret = []
        for r in reversed(ep_rew):
            g = r + 0.999 * g
            ep_ret.append(g)
        ep_ret.reverse()

        obs_data.extend(ep_obs)
        act_data.extend(ep_act)
        ret_data.extend(ep_ret)

    obs_t = torch.as_tensor(np.stack(obs_data, axis=0), dtype=torch.float32)
    act_t = torch.as_tensor(np.stack(act_data, axis=0), dtype=torch.float32)
    ret_t = torch.as_tensor(np.array(ret_data, dtype=np.float32), dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    batch_size = 256
    n = obs_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred_mean, pred_val = model(obs_t[idx])
            actor_loss = ((pred_mean - act_t[idx]) ** 2).mean()
            critic_loss = 0.0001 * ((pred_val - ret_t[idx]) ** 2).mean()
            loss = actor_loss + critic_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        model.log_std.fill_(-1.6)


def evaluate_gps_policy(
    model: ActorCritic,
    target_mode: str = "static",
    altitude_m: float = 1000.0,
    num_episodes: int = 6,
    seed: int = 200,
) -> Dict[str, Any]:
    """Evaluate 14D policy on 1,000 m GPS target tracking across deterministic + random test cases."""
    env_det = GlideGpsTargetEnv(target_mode=target_mode, release_altitude_m=altitude_m, randomize_drop=False)
    obs, info = env_det.reset(seed=seed, options={"deterministic": True, "altitude_m": altitude_m})
    init_dist = info["target_error_m"]
    trajectory: List[Dict[str, Any]] = [info]
    ep_return = 0.0
    max_bank_deg = 0.0

    while True:
        act = model.act_deterministic(obs)
        obs, rew, term, trunc, info = env_det.step(act)
        ep_return += rew
        max_bank_deg = max(max_bank_deg, abs(info["roll_deg"]))
        trajectory.append(info)
        if term or trunc:
            break

    det_final_err = info["target_error_m"]

    errs = [det_final_err]
    for k in range(1, num_episodes):
        env_r = GlideGpsTargetEnv(target_mode=target_mode, release_altitude_m=altitude_m, randomize_drop=True)
        o, inf = env_r.reset(seed=seed + k * 37)
        while True:
            a = model.act_deterministic(o)
            o, _, tm, tr, inf = env_r.step(a)
            if tm or tr:
                break
        errs.append(float(inf["target_error_m"]))

    return {
        "target_mode": target_mode,
        "initial_target_dist_m": round(float(init_dist), 2),
        "final_target_error_m": round(float(det_final_err), 2),
        "mean_random_error_m": round(float(np.mean(errs)), 2),
        "episode_return": round(float(ep_return), 2),
        "max_bank_deg": round(float(max_bank_deg), 2),
        "flight_time_s": round(float(info["time_s"]), 2),
        "trajectory": trajectory,
    }


def run_curriculum_stage(
    model: ActorCritic,
    target_mode: str,
    stage_id: str,
    stage_label: str,
    lead_gain: float,
    num_updates: int = 8,
    num_envs: int = 6,
    rollout_steps: int = 512,
    lr: float = 1.5e-4,
    seed: int = 100,
    ckpt_dir: Path = Path("checkpoints/glide_rl"),
) -> Dict[str, Any]:
    pretrain_actor_for_gps_stage(
        model=model,
        target_mode=target_mode,
        lead_gain=lead_gain,
        num_episodes=35,
        epochs=25,
        seed=seed,
    )

    envs = [GlideGpsTargetEnv(target_mode=target_mode, release_altitude_m=1000.0, randomize_drop=True) for _ in range(num_envs)]
    ppo = PPOTrainer(
        model,
        PPOConfig(
            lr=lr,
            gamma=0.999,
            gae_lambda=0.98,
            clip_coef=0.15,
            ent_coef=0.0005,
            update_epochs=6,
            minibatch_size=256,
        ),
    )

    obs_list = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed + i)
        obs_list.append(o)
    obs = np.stack(obs_list, axis=0)

    best_err = 1e9
    best_state = None
    best_eval = None

    for update in range(num_updates + 1):
        ev = evaluate_gps_policy(model, target_mode=target_mode, altitude_m=1000.0, num_episodes=5, seed=700)
        score = 0.5 * ev["final_target_error_m"] + 0.5 * ev["mean_random_error_m"]
        if score < best_err:
            best_err = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_eval = ev

        if update % 2 == 0 or update == num_updates:
            print(
                f"[{stage_label}] update={update:2d}/{num_updates} | "
                f"Det Miss = {ev['final_target_error_m']:5.1f} m (from {ev['initial_target_dist_m']:.0f} m) | "
                f"Mean Random Miss = {ev['mean_random_error_m']:5.1f} m | "
                f"Max Bank = {ev['max_bank_deg']:4.1f}°"
            )

        if update == num_updates:
            break

        frac = 1.0 - (update / float(num_updates))
        for pg in ppo.optimizer.param_groups:
            pg["lr"] = max(3e-5, lr * frac)

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
            "final_target_error_m": best_eval["final_target_error_m"],
            "mean_random_error_m": best_eval["mean_random_error_m"],
            "initial_target_dist_m": best_eval["initial_target_dist_m"],
            "max_bank_deg": best_eval["max_bank_deg"],
            "episode_return": best_eval["episode_return"],
        },
        ckpt_file,
    )
    print(
        f"==> Completed [{stage_label}] | Saved {ckpt_file} | "
        f"Best Det Miss = {best_eval['final_target_error_m']:.1f} m | "
        f"Mean Random Miss = {best_eval['mean_random_error_m']:.1f} m | "
        f"Max Bank = {best_eval['max_bank_deg']:.1f}°\n"
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

    # Stage A: Static Desired GPS Coordinate
    res_static = run_curriculum_stage(
        model=model,
        target_mode="static",
        stage_id="gps_1_static",
        stage_label="GPS Stage 1: Static Desired Coordinate",
        lead_gain=0.0,
        num_updates=6,
        lr=1.5e-4,
        seed=101,
        ckpt_dir=ckpt_dir,
    )

    # Automatically advance to Stage B: Linear Straight-Line Moving GPS Coordinate
    res_linear = run_curriculum_stage(
        model=model,
        target_mode="linear",
        stage_id="gps_2_linear",
        stage_label="GPS Stage 2: Linear Straight-Line Moving Coordinate",
        lead_gain=0.65,
        num_updates=6,
        lr=1.5e-4,
        seed=202,
        ckpt_dir=ckpt_dir,
    )

    # Automatically advance to Stage C: Aggressive Maneuvering GPS Coordinate
    res_aggr = run_curriculum_stage(
        model=model,
        target_mode="aggressive",
        stage_id="gps_3_aggressive",
        stage_label="GPS Stage 3: Aggressive Maneuvering Coordinate",
        lead_gain=0.35,
        num_updates=6,
        lr=1.5e-4,
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

    print(f"All 3 GPS curriculum stages completed automatically in {elapsed:.1f}s!")
    return summary


if __name__ == "__main__":
    train_all_gps_stages()
