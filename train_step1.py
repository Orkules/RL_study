from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch

from rl.ppo_agent import ActorCritic, PPOConfig, PPOTrainer
from sim.tethered_drone_env import TetheredDroneEnv


def evaluate_policy(
    model: ActorCritic,
    n_episodes: int = 5,
    record_frames: int = 300,
    seed: int = 100,
) -> Dict[str, Any]:
    """Evaluate current policy and record a deterministic trajectory for visual inspection."""
    env = TetheredDroneEnv(initial_tilt_range_deg=25.0, terminate_on_wire_limit=True)
    visual_env = TetheredDroneEnv(initial_tilt_range_deg=25.0, terminate_on_wire_limit=False)
    returns: List[float] = []
    final_tilts: List[float] = []
    lengths: List[int] = []
    recorded_trajectory: List[Dict[str, Any]] = []

    # Record a full uninterrupted visual trajectory on the 4-wire rig
    v_obs, v_info = visual_env.reset(
        seed=seed,
        options={
            "initial_angles_deg": [22.0, -18.0, 12.0],
            "initial_omega": [1.2, -1.0, 0.8],
        },
    )
    recorded_trajectory.append(v_info)
    for _ in range(record_frames):
        v_act = model.act_deterministic(v_obs)
        v_obs, _, _, _, v_info = visual_env.step(v_act)
        recorded_trajectory.append(v_info)

    for ep in range(n_episodes):
        if ep == 0:
            obs, info = env.reset(
                seed=seed,
                options={
                    "initial_angles_deg": [22.0, -18.0, 12.0],
                    "initial_omega": [1.2, -1.0, 0.8],
                },
            )
        else:
            obs, info = env.reset(seed=seed + ep)

        ep_ret = 0.0
        steps = 0
        for _ in range(env.cfg.max_steps):
            action = model.act_deterministic(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            steps += 1
            if terminated or truncated:
                break

        returns.append(float(ep_ret))
        final_tilts.append(float(info["tilt_deg"]))
        lengths.append(int(steps))

    return {
        "mean_return": round(float(np.mean(returns)), 2),
        "mean_final_tilt_deg": round(float(np.mean(final_tilts)), 2),
        "mean_episode_steps": round(float(np.mean(lengths)), 1),
        "trajectory": recorded_trajectory,
    }


def train_step1(
    num_updates: int = 85,
    num_envs: int = 8,
    rollout_steps: int = 256,
    seed: int = 42,
    output_dir: str = "artifacts",
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_path = Path(output_dir)
    ckpt_path = Path(checkpoint_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    envs = [
        TetheredDroneEnv(initial_tilt_range_deg=26.0, domain_randomization=True)
        for _ in range(num_envs)
    ]
    model = ActorCritic(obs_dim=6, act_dim=3, hidden_dim=64)
    ppo = PPOTrainer(
        model,
        PPOConfig(
            lr=4e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            ent_coef=0.0002,
            update_epochs=8,
            minibatch_size=256,
        ),
    )

    milestone_Schedule = {
        0: ("stage_0_random", "Stage 0: Untrained (Random Network)"),
        25: ("stage_1_early", "Stage 1: Early Learning (Discovering Thrust & Gimbal)"),
        30: ("stage_2_mid", "Stage 2: Mid Training (Wobbly Balance)"),
        38: ("stage_3_late", "Stage 3: Late Training (Damping Oscillations)"),
        num_updates: ("stage_4_stabilized", "Stage 4: Converged (Locked Upright Stabilization)"),
    }

    learning_curve: List[Dict[str, Any]] = []
    milestones: List[Dict[str, Any]] = []

    obs_list = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed + i)
        obs_list.append(o)
    obs = np.stack(obs_list, axis=0)

    start_time = time.time()

    for update in range(num_updates + 1):
        if update in milestone_Schedule:
            stage_id, stage_label = milestone_Schedule[update]
            eval_res = evaluate_policy(model, n_episodes=5, record_frames=300, seed=777)
            ckpt_file = ckpt_path / f"{stage_id}.pt"
            torch.save(
                {
                    "stage_id": stage_id,
                    "stage_label": stage_label,
                    "update": update,
                    "model_state_dict": model.state_dict(),
                    "mean_return": eval_res["mean_return"],
                    "mean_final_tilt_deg": eval_res["mean_final_tilt_deg"],
                },
                ckpt_file,
            )
            milestones.append(
                {
                    "stage_id": stage_id,
                    "stage_label": stage_label,
                    "update": update,
                    "timesteps": update * num_envs * rollout_steps,
                    "mean_return": eval_res["mean_return"],
                    "mean_final_tilt_deg": eval_res["mean_final_tilt_deg"],
                    "mean_episode_steps": eval_res["mean_episode_steps"],
                    "checkpoint_path": str(ckpt_file),
                    "trajectory": eval_res["trajectory"],
                }
            )
            print(
                f"[{stage_label}] update={update:3d} | "
                f"return={eval_res['mean_return']:7.2f} | "
                f"final_tilt={eval_res['mean_final_tilt_deg']:5.2f} deg | "
                f"steps={eval_res['mean_episode_steps']:5.1f}"
            )

        if update == num_updates:
            break

        # Linearly anneal learning rate for crisp final stabilization
        frac = 1.0 - (update / float(num_updates))
        lr_now = max(5e-5, 4e-4 * frac)
        for param_group in ppo.optimizer.param_groups:
            param_group["lr"] = lr_now

        # Collect rollout across parallel environments
        obs_buf = np.zeros((rollout_steps, num_envs, 6), dtype=np.float32)
        act_buf = np.zeros((rollout_steps, num_envs, 3), dtype=np.float32)
        logp_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        rew_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        done_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        val_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        step_tilts: List[float] = []

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
                n_obs, rew, term, trunc, info = env.step(actions[env_idx])
                done = term or trunc
                rew_buf[step, env_idx] = rew
                done_buf[step, env_idx] = 1.0 if done else 0.0
                step_tilts.append(info["tilt_deg"])
                if done:
                    n_obs, _ = env.reset()
                next_obs_list.append(n_obs)
            obs = np.stack(next_obs_list, axis=0)

        # Compute GAE advantages and returns
        with torch.no_grad():
            next_val = model.critic_net(torch.as_tensor(obs, dtype=torch.float32)).squeeze(-1).cpu().numpy()

        adv_buf = np.zeros_like(rew_buf)
        last_gae = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(rollout_steps)):
            if t == rollout_steps - 1:
                next_non_terminal = 1.0 - done_buf[t]
                next_values = next_val
            else:
                next_non_terminal = 1.0 - done_buf[t]
                next_values = val_buf[t + 1]
            delta = rew_buf[t] + ppo.cfg.gamma * next_values * next_non_terminal - val_buf[t]
            last_gae = delta + ppo.cfg.gamma * ppo.cfg.gae_lambda * next_non_terminal * last_gae
            adv_buf[t] = last_gae

        ret_buf = adv_buf + val_buf

        loss_info = ppo.update(
            obs_buf.reshape(-1, 6),
            act_buf.reshape(-1, 3),
            logp_buf.reshape(-1),
            ret_buf.reshape(-1),
            adv_buf.reshape(-1),
        )

        if update % 5 == 0 or update == num_updates - 1:
            eval_quick = evaluate_policy(model, n_episodes=3, record_frames=1, seed=500)
            learning_curve.append(
                {
                    "update": int(update + 1),
                    "timesteps": int((update + 1) * num_envs * rollout_steps),
                    "mean_return": eval_quick["mean_return"],
                    "mean_final_tilt_deg": eval_quick["mean_final_tilt_deg"],
                    "mean_episode_steps": eval_quick["mean_episode_steps"],
                    "rollout_avg_tilt_deg": round(float(np.mean(step_tilts)), 2),
                    "policy_loss": round(loss_info["policy_loss"], 5),
                    "value_loss": round(loss_info["value_loss"], 5),
                }
            )

    elapsed = round(time.time() - start_time, 2)
    summary = {
        "project": "Step 1: 20cm Tethered Single-Propeller TVC Rocket-Drone Stabilization",
        "elapsed_seconds": elapsed,
        "total_timesteps": num_updates * num_envs * rollout_steps,
        "learning_curve": learning_curve,
        "milestones": milestones,
    }

    with open(out_path / "step1_training_progress.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Training completed in {elapsed:.1f}s. Saved checkpoints to {ckpt_path}/")
    return summary


if __name__ == "__main__":
    train_step1()
