from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from rl.ppo_agent import ActorCritic, PPOConfig, PPOTrainer
from sim.fpv_balls_env import FpvPoweredBallsEnv


def evaluate_fpv_balls_policy(
    model: ActorCritic,
    num_balls: int = 5,
    num_episodes: int = 6,
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
        "det_balls_reached": int(det_balls),
        "mean_balls_reached": round(float(np.mean(balls_list)), 2),
        "det_speed_fraction_of_max": round(float(det_speed_frac), 3),
        "mean_speed_fraction_of_max": round(float(np.mean(speed_fracs)), 3),
        "det_flight_time_s": round(float(det_time), 2),
        "max_airspeed_ms": round(float(max_speed), 2),
        "episode_return": round(float(ep_return), 2),
    }


def train_fpv_balls_pure_rl(
    num_updates: int = 45,
    num_envs: int = 8,
    rollout_steps: int = 256,
    lr: float = 2.5e-4,
    seed: int = 42,
    ckpt_dir: Path = Path("checkpoints/glide_rl"),
) -> Dict[str, Any]:
    """
    100% Pure Reinforcement Learning (PPO + Reward-Weighted Self-Imitation of Top Episodes)
    for the 3D-action [pitch, roll, motor_throttle] FPV Powered Delta-Wing in Unrestricted 3D Ball Space.
    Zero hand-coded flight controllers or heuristics!
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCritic(obs_dim=25, act_dim=3, hidden_dim=64)
    # Initialize from existing 24D->2D weights for the first 24 obs and 2 flap actions if available,
    # leaving the 3rd action (motor throttle) and 25th obs to be learned purely from RL reward!
    static_ckpt = ckpt_dir / "gps_1_static.pt"
    if static_ckpt.exists():
        ck = torch.load(static_ckpt, map_location="cpu", weights_only=False)
        sd_old = ck["model_state_dict"]
        with torch.no_grad():
            model.actor_net[0].weight.zero_()
            model.actor_net[0].weight[:, :15].copy_(sd_old["actor_net.0.weight"][:, :15])
            model.actor_net[0].bias.copy_(sd_old["actor_net.0.bias"])
            model.actor_net[2].weight.copy_(sd_old["actor_net.2.weight"])
            model.actor_net[2].bias.copy_(sd_old["actor_net.2.bias"])
            model.actor_net[4].weight.zero_()
            model.actor_net[4].weight[:2, :].copy_(sd_old["actor_net.4.weight"][:2, :])
            model.actor_net[4].bias[:2].copy_(sd_old["actor_net.4.bias"][:2])
            # Start motor throttle bias at +1.2 (tanh(1.2) = 0.83 -> 92% FPV motor power)
            model.actor_net[4].bias[2] = 1.2
            model.log_std.fill_(-1.4)

    envs = [FpvPoweredBallsEnv(num_balls=5, randomize_start=True) for _ in range(num_envs)]
    ppo = PPOTrainer(
        model,
        PPOConfig(
            lr=lr,
            gamma=0.995,
            gae_lambda=0.97,
            clip_coef=0.18,
            ent_coef=0.002,
            update_epochs=6,
            minibatch_size=256,
        ),
    )

    obs_list = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed + i * 29)
        obs_list.append(o)
    obs = np.stack(obs_list, axis=0)

    best_score = -1e9
    best_state = None
    best_eval = None

    for update in range(num_updates + 1):
        ev = evaluate_fpv_balls_policy(model, num_balls=5, num_episodes=6, seed=300)
        score = 1500.0 * (0.5 * ev["det_balls_reached"] + 0.5 * ev["mean_balls_reached"]) + 1000.0 * ev["mean_speed_fraction_of_max"] + 0.1 * ev["episode_return"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_eval = ev

        if update % 5 == 0 or update == num_updates:
            print(
                f"[FPV Powered 3D Balls Pure-RL] update={update:2d}/{num_updates} | "
                f"Det Balls = {ev['det_balls_reached']}/5 | "
                f"Mean Balls = {ev['mean_balls_reached']:.2f}/5 | "
                f"Speed/V_max = {ev['mean_speed_fraction_of_max']*100:.1f}% | "
                f"Peak Speed = {ev['max_airspeed_ms']:.1f} m/s",
                flush=True,
            )

        if update == num_updates:
            break

        frac = 1.0 - (update / float(num_updates))
        for pg in ppo.optimizer.param_groups:
            pg["lr"] = max(3e-5, lr * frac)

        obs_buf = np.zeros((rollout_steps, num_envs, 25), dtype=np.float32)
        act_buf = np.zeros((rollout_steps, num_envs, 3), dtype=np.float32)
        logp_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        rew_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        done_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        val_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        for step in range(rollout_steps):
            obs_buf[step] = obs
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                act_t, logp_t, _, val_t = model.get_action_and_value(obs_t)

            actions = np.clip(act_t.cpu().numpy(), -1.0, 1.0)
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
            obs_buf.reshape(-1, 25),
            act_buf.reshape(-1, 3),
            logp_buf.reshape(-1),
            ret_buf.reshape(-1),
            adv_buf.reshape(-1),
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_file = ckpt_dir / "fpv_balls_master.pt"
    torch.save(
        {
            "stage_id": "fpv_balls_master",
            "stage_label": "FPV Powered 3D Multi-Ball Hunter (3D Action: Pitch, Roll, Motor Power)",
            "target_mode": "fpv_balls_3d",
            "obs_dim": 25,
            "act_dim": 3,
            "model_state_dict": model.state_dict(),
            "det_balls_reached": best_eval["det_balls_reached"],
            "mean_balls_reached": best_eval["mean_balls_reached"],
            "mean_speed_fraction_of_max": best_eval["mean_speed_fraction_of_max"],
            "max_airspeed_ms": best_eval["max_airspeed_ms"],
            "episode_return": best_eval["episode_return"],
        },
        ckpt_file,
    )
    print(f"==> Saved {ckpt_file} | Best Eval: {best_eval}", flush=True)
    return best_eval


if __name__ == "__main__":
    train_fpv_balls_pure_rl(num_updates=30, num_envs=8, rollout_steps=256, lr=2.0e-4, seed=42)
