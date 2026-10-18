from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from rl.ppo_agent import ActorCritic, PPOConfig, PPOTrainer
from sim.glide_gps_env import GlideGpsTargetEnv


def evaluate_gps_policy(
    model: ActorCritic,
    target_mode: str = "static",
    altitude_m: float = 1000.0,
    num_episodes: int = 6,
    seed: int = 200,
) -> Dict[str, Any]:
    """Evaluate 24D INAV + Relative Target policy across deterministic + random 360-deg test cases."""
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
    num_updates: int = 40,
    num_envs: int = 8,
    rollout_steps: int = 256,
    lr: float = 1.5e-4,
    init_log_std: float = -1.2,
    seed: int = 100,
    ckpt_dir: Path = Path("checkpoints/glide_rl"),
) -> Dict[str, Any]:
    """
    100% Pure Reinforcement Learning (PPO) with ZERO hand-coded guidance, ZERO behavioral cloning,
    and ZERO angle or action restrictions.
    The network explores with Gaussian policy noise and updates strictly via PPO policy gradients
    from the environment reward (distance closing, <= 3.0 m hit bonus, fast arrival bonus, and anti-oscillation penalty).
    """
    with torch.no_grad():
        model.log_std.fill_(init_log_std)

    envs = [GlideGpsTargetEnv(target_mode=target_mode, release_altitude_m=1000.0, randomize_drop=True) for _ in range(num_envs)]
    ppo = PPOTrainer(
        model,
        PPOConfig(
            lr=lr,
            gamma=0.995,
            gae_lambda=0.97,
            clip_coef=0.15,
            ent_coef=0.001,
            update_epochs=6,
            minibatch_size=256,
        ),
    )

    obs_list = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed + i * 19)
        obs_list.append(o)
    obs = np.stack(obs_list, axis=0)

    best_err = 1e9
    best_state = None
    best_eval = None

    for update in range(num_updates + 1):
        ev = evaluate_gps_policy(model, target_mode=target_mode, altitude_m=1000.0, num_episodes=6, seed=700)
        score = 0.55 * ev["final_target_error_m"] + 0.45 * ev["mean_random_error_m"]
        if score < best_err:
            best_err = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_eval = ev

        if update % 4 == 0 or update == num_updates:
            print(
                f"[{stage_label}] update={update:3d}/{num_updates} | "
                f"Det Miss = {ev['final_target_error_m']:5.2f} m | "
                f"Random Avg Miss = {ev['mean_random_error_m']:5.2f} m | "
                f"Flight Time = {ev['flight_time_s']:4.1f} s | "
                f"Max Roll = {ev['max_bank_deg']:5.1f} deg",
                flush=True,
            )

        if update == num_updates:
            break

        frac = 1.0 - (update / float(num_updates))
        for pg in ppo.optimizer.param_groups:
            pg["lr"] = max(2e-5, lr * frac)

        obs_buf = np.zeros((rollout_steps, num_envs, 24), dtype=np.float32)
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
            obs_buf.reshape(-1, 24),
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
            "obs_dim": 24,
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
        f"Best Det Miss = {best_eval['final_target_error_m']:.2f} m | "
        f"Random Avg Miss = {best_eval['mean_random_error_m']:.2f} m\n",
        flush=True,
    )
    return best_eval


if __name__ == "__main__":
    import sys

    stage_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    np.random.seed(42)
    torch.manual_seed(42)
    ckpt_dir = Path("checkpoints/glide_rl")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = ActorCritic(obs_dim=24, act_dim=2, hidden_dim=64)

    if stage_arg in ("static", "all"):
        if (ckpt_dir / "gps_1_static.pt").exists():
            ck = torch.load(ckpt_dir / "gps_1_static.pt", weights_only=False)
            if ck.get("obs_dim") == 24:
                model.load_state_dict(ck["model_state_dict"])
        run_pure_ppo_stage(
            model=model,
            target_mode="static",
            stage_id="gps_1_static",
            stage_label="GPS Stage 1: Pure-RL Multi-Angle/Speed Target Homing (<3m Error)",
            num_updates=24,
            num_envs=8,
            lr=1.0e-4,
            init_log_std=-1.5,
            seed=101,
            ckpt_dir=ckpt_dir,
        )

    if stage_arg in ("linear", "all"):
        if (ckpt_dir / "gps_2_linear.pt").exists():
            ck = torch.load(ckpt_dir / "gps_2_linear.pt", weights_only=False)
            if ck.get("obs_dim") == 24:
                model.load_state_dict(ck["model_state_dict"])
        elif (ckpt_dir / "gps_1_static.pt").exists():
            ck = torch.load(ckpt_dir / "gps_1_static.pt", weights_only=False)
            if ck.get("obs_dim") == 24:
                model.load_state_dict(ck["model_state_dict"])
        run_pure_ppo_stage(
            model=model,
            target_mode="linear",
            stage_id="gps_2_linear",
            stage_label="GPS Stage 2: Pure-RL Tangent-Chord Crossing Car (<3m Hit)",
            num_updates=24,
            num_envs=8,
            lr=1.0e-4,
            init_log_std=-1.5,
            seed=404,
            ckpt_dir=ckpt_dir,
        )

    if stage_arg in ("curved", "all"):
        if (ckpt_dir / "gps_3_aggressive.pt").exists():
            ck = torch.load(ckpt_dir / "gps_3_aggressive.pt", weights_only=False)
            if ck.get("obs_dim") == 24:
                model.load_state_dict(ck["model_state_dict"])
        elif (ckpt_dir / "gps_2_linear.pt").exists():
            ck = torch.load(ckpt_dir / "gps_2_linear.pt", weights_only=False)
            if ck.get("obs_dim") == 24:
                model.load_state_dict(ck["model_state_dict"])
        run_pure_ppo_stage(
            model=model,
            target_mode="curved",
            stage_id="gps_3_aggressive",
            stage_label="GPS Stage 3: Pure-RL Curved Moving Target (8-20 m/s)",
            num_updates=24,
            num_envs=8,
            lr=1.0e-4,
            init_log_std=-1.5,
            seed=505,
            ckpt_dir=ckpt_dir,
        )
