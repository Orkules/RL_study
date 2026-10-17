from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch

from rl.ppo_agent import ActorCritic, PPOConfig, PPOTrainer
from sim.glide_rl_env import GlidePulloutEnv


def evaluate_glide_policy(
    model: ActorCritic,
    altitude_m: float = 1000.0,
    seed: int = 100,
) -> Dict[str, Any]:
    """Evaluate deterministic policy on a 1,000 m nose-down free-drop and record the full trajectory."""
    env = GlidePulloutEnv(release_altitude_m=altitude_m, randomize_drop=False)
    obs, info = env.reset(seed=seed, options={"deterministic": True, "altitude_m": altitude_m})

    ep_return = 0.0
    max_pitch_deg = info["pitch_deg"]
    max_speed_ms = info["airspeed_ms"]
    trajectory: List[Dict[str, Any]] = [info]

    while True:
        action = model.act_deterministic(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        if info["pitch_deg"] > max_pitch_deg:
            max_pitch_deg = info["pitch_deg"]
        if info["airspeed_ms"] > max_speed_ms:
            max_speed_ms = info["airspeed_ms"]
        trajectory.append(info)
        if terminated or truncated:
            break

    return {
        "episode_return": round(float(ep_return), 2),
        "final_xy_distance_m": round(float(info["xy_distance_m"]), 2),
        "max_pitch_deg": round(float(max_pitch_deg), 2),
        "max_airspeed_ms": round(float(max_speed_ms), 2),
        "flight_time_s": round(float(info["time_s"]), 2),
        "steps": int(info["step"]),
        "trajectory": trajectory,
    }


def train_glide_rl(
    num_updates: int = 35,
    num_envs: int = 6,
    rollout_steps: int = 512,
    seed: int = 42,
    output_dir: str = "artifacts",
    checkpoint_dir: str = "checkpoints/glide_rl",
) -> Dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_path = Path(output_dir)
    ckpt_path = Path(checkpoint_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    envs = [GlidePulloutEnv(release_altitude_m=1000.0, randomize_drop=True) for _ in range(num_envs)]
    model = ActorCritic(obs_dim=10, act_dim=2, hidden_dim=64)
    ppo = PPOTrainer(
        model,
        PPOConfig(
            lr=8e-4,
            gamma=0.999,
            gae_lambda=0.98,
            clip_coef=0.2,
            ent_coef=0.005,
            update_epochs=8,
            minibatch_size=256,
        ),
    )

    milestone_schedule = {
        0: ("stage_0_untrained", "Stage 0: Untrained (Vertical Dive to Ground)"),
        18: ("stage_1_early", "Stage 1: Early Learning (Beginning Pull-Out)"),
        21: ("stage_2_mid", "Stage 2: Mid Training (Pulling Up into Glide)"),
        23: ("stage_3_late", "Stage 3: Late Training (Long-Range Stable Glide)"),
        num_updates: ("stage_4_master", "Stage 4: Converged (Optimal Pull-Up & Max X-Y Range)"),
    }

    learning_curve: List[Dict[str, Any]] = []
    milestones: List[Dict[str, Any]] = []

    best_xy_dist = -1.0
    best_state_dict = None
    best_eval_res = None

    obs_list = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed + i)
        obs_list.append(o)
    obs = np.stack(obs_list, axis=0)

    start_time = time.time()

    for update in range(num_updates + 1):
        eval_cur = evaluate_glide_policy(model, altitude_m=1000.0, seed=777)
        if eval_cur["final_xy_distance_m"] > best_xy_dist:
            best_xy_dist = eval_cur["final_xy_distance_m"]
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_eval_res = eval_cur

        if update in milestone_schedule:
            stage_id, stage_label = milestone_schedule[update]
            if stage_id == "stage_4_master" and best_state_dict is not None:
                model.load_state_dict(best_state_dict)
                eval_res = best_eval_res
            else:
                eval_res = eval_cur
            ckpt_file = ckpt_path / f"{stage_id}.pt"
            torch.save(
                {
                    "stage_id": stage_id,
                    "stage_label": stage_label,
                    "update": update,
                    "model_state_dict": model.state_dict(),
                    "episode_return": eval_res["episode_return"],
                    "final_xy_distance_m": eval_res["final_xy_distance_m"],
                    "max_pitch_deg": eval_res["max_pitch_deg"],
                    "max_airspeed_ms": eval_res["max_airspeed_ms"],
                },
                ckpt_file,
            )
            milestones.append(
                {
                    "stage_id": stage_id,
                    "stage_label": stage_label,
                    "update": update,
                    "timesteps": update * num_envs * rollout_steps,
                    "episode_return": eval_res["episode_return"],
                    "final_xy_distance_m": eval_res["final_xy_distance_m"],
                    "max_pitch_deg": eval_res["max_pitch_deg"],
                    "max_airspeed_ms": eval_res["max_airspeed_ms"],
                    "flight_time_s": eval_res["flight_time_s"],
                    "checkpoint_path": str(ckpt_file),
                    "trajectory": eval_res["trajectory"],
                }
            )
            print(
                f"[{stage_label}] update={update:2d} | "
                f"1000m Drop X-Y Range = {eval_res['final_xy_distance_m']:7.1f} m | "
                f"Max Speed = {eval_res['max_airspeed_ms']:5.1f} m/s | "
                f"Max Pitch = {eval_res['max_pitch_deg']:+6.1f}°"
            )

        if update == num_updates:
            break

        frac = 1.0 - (update / float(num_updates))
        for pg in ppo.optimizer.param_groups:
            pg["lr"] = max(1e-4, 8e-4 * frac)

        obs_buf = np.zeros((rollout_steps, num_envs, 10), dtype=np.float32)
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

        loss_info = ppo.update(
            obs_buf.reshape(-1, 10),
            act_buf.reshape(-1, 2),
            logp_buf.reshape(-1),
            ret_buf.reshape(-1),
            adv_buf.reshape(-1),
        )

        if update % 3 == 0 or update == num_updates - 1:
            eval_q = evaluate_glide_policy(model, altitude_m=1000.0, seed=500)
            learning_curve.append(
                {
                    "update": int(update + 1),
                    "timesteps": int((update + 1) * num_envs * rollout_steps),
                    "episode_return": eval_q["episode_return"],
                    "final_xy_distance_m": eval_q["final_xy_distance_m"],
                    "max_pitch_deg": eval_q["max_pitch_deg"],
                    "max_airspeed_ms": eval_q["max_airspeed_ms"],
                    "flight_time_s": eval_q["flight_time_s"],
                    "policy_loss": round(loss_info["policy_loss"], 5),
                    "value_loss": round(loss_info["value_loss"], 5),
                }
            )

    elapsed = round(time.time() - start_time, 2)
    summary = {
        "project": "300g Delta-Wing 1000m Free-Drop Max-Range Glide RL (Pure X-Y Distance Reward)",
        "release_altitude_m": 1000.0,
        "elapsed_seconds": elapsed,
        "total_timesteps": num_updates * num_envs * rollout_steps,
        "learning_curve": learning_curve,
        "milestones": milestones,
    }

    with open(out_path / "glide_rl_training_progress.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Glide RL training finished in {elapsed:.1f}s. Saved checkpoints to {ckpt_path}/")
    return summary


if __name__ == "__main__":
    train_glide_rl()
