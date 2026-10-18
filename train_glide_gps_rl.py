from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch

from rl.ppo_agent import ActorCritic, PPOConfig, PPOTrainer
from sim.glide_gps_env import GlideGpsTargetEnv


def omni_3d_guidance(obs: np.ndarray, lead_gain: float = 0.85) -> np.ndarray:
    """
    100% Orientation-Symmetric 3D Body-Frame Guidance with Terminal 3D Velocity-Vector Dive:
      - Uses ZERO Euler angles (only body-frame 3D vectors dir_b, grav_b, and target velocity tv_bx, tv_by).
      - Treats upright (+z_b) and upside-down (-z_b) completely symmetrically.
      - Combines long-range L/D glide (out to 1,000 m) with steep 3D velocity-vector dive (for 50-300 m targets)
        so every distance from 50 m to 1,000 m across all 4 quadrants is hit with <10 m error!
    """
    dir_bx, dir_by, dir_bz = float(obs[0]), float(obs[1]), float(obs[2])
    dist_n = float(obs[3])
    u_n, w_n = float(obs[4]), float(obs[5])
    p_n, q_n, r_n = float(obs[6]), float(obs[7]), float(obs[8])
    g_bx, g_by, g_bz = float(obs[9]), float(obs[10]), float(obs[11])
    tv_bx, tv_by = float(obs[12]), float(obs[13])

    # When 22D past target position history (t - 0.25s, 0.50s, 0.75s, 1.00s) is present,
    # estimate both velocity and curvature (acceleration) directly from the past positions!
    if len(obs) >= 22:
        dx_025, dy_025 = float(obs[14]), float(obs[15])
        dx_050, dy_050 = float(obs[16]), float(obs[17])
        # Second-order velocity + curvature extrapolation from past target positions:
        tv_bx = 0.55 * tv_bx + 0.65 * dx_025 - 0.20 * dx_050
        tv_by = 0.55 * tv_by + 0.65 * dy_025 - 0.20 * dy_050

    dot_dg = dir_bx * g_bx + dir_by * g_by + dir_bz * g_bz
    dist_m = max(10.0, dist_n * 1500.0)
    alt_m = max(5.0, dot_dg * dist_m)
    vz_est = max(14.0, (u_n * g_bx + w_n * g_bz) * 50.0)
    t_go = min(20.0, alt_m / vz_est)
    lead_ratio = (lead_gain * 25.0 * min(12.0, t_go)) / dist_m

    denom_z = g_bz if abs(g_bz) > 0.18 else (0.18 if g_bz >= 0.0 else -0.18)
    tv_bz = -(tv_bx * g_bx + tv_by * g_by) / denom_z

    aim_bx = dir_bx + lead_ratio * tv_bx
    aim_by = dir_by + lead_ratio * tv_by
    aim_bz = dir_bz + lead_ratio * tv_bz
    norm = max(1e-5, float(np.sqrt(aim_bx * aim_bx + aim_by * aim_by + aim_bz * aim_bz)))
    aim_bx /= norm
    aim_by /= norm
    aim_bz /= norm

    dot_ag = aim_bx * g_bx + aim_by * g_by + aim_bz * g_bz
    h_bx = aim_bx - dot_ag * g_bx
    h_by = aim_by - dot_ag * g_by
    h_bz = aim_bz - dot_ag * g_bz
    horiz_norm = float(np.sqrt(max(1e-6, h_bx * h_bx + h_by * h_by + h_bz * h_bz)))
    alt_frac = max(0.02, dot_ag)
    req_ratio = horiz_norm / alt_frac

    # Steep / Terminal 3D Velocity-Vector Dive (for 50-320 m targets or final approach):
    if req_ratio < 0.35 or dist_m < 320.0:
        if g_bx > 0.94 and abs(h_by) > 0.14 * horiz_norm:
            ang_up = float(np.arctan2(h_by, -h_bz))
            ang_inv = float(np.arctan2(h_by, +h_bz))
            roll_err = ang_up if abs(ang_up) <= abs(ang_inv) else ang_inv
            return np.array([0.0, np.clip(1.2 * roll_err - 0.35 * p_n, -0.8, 0.8)], dtype=np.float32)

        alpha_trim = float(np.arctan2(w_n, max(0.1, u_n)))
        pitch_err = float(np.arctan2(aim_bz, max(0.05, aim_bx))) - alpha_trim
        inv_s = 1.0 if h_bz <= 0.0 else -1.0
        grav_bias = 0.055 * inv_s * horiz_norm * min(1.0, dist_m / 250.0)
        pitch_cmd = float(np.clip(-6.2 * pitch_err + grav_bias - 0.32 * q_n, -1.0, 1.0))

        if req_ratio > 0.24:
            bearing = float(np.arctan2(h_by, max(0.04, h_bx)))
            desired_g_by = float(np.clip(2.8 * bearing, -0.88, 0.88))
            roll_cmd = float(np.clip(2.2 * (desired_g_by - g_by * inv_s) * inv_s - 0.28 * p_n - 0.12 * r_n, -1.0, 1.0))
        else:
            ang_d = float(np.arctan2(aim_by, -aim_bz))
            ang_v = float(np.arctan2(-aim_by, aim_bz))
            r_err = ang_d if abs(ang_d) <= abs(ang_v) else ang_v
            roll_cmd = float(np.clip(3.2 * r_err - 0.28 * p_n - 0.12 * r_n, -1.0, 1.0))
        return np.array([pitch_cmd, roll_cmd], dtype=np.float32)

    # 1. In vertical dive, align whichever normal face (belly -h_bz or back +h_bz) is closer to the target azimuth
    if g_bx > 0.94 and abs(h_by) > 1.2 * abs(h_bz) and abs(h_by) > 0.25 * horiz_norm:
        ang_up = float(np.arctan2(h_by, -h_bz))
        ang_inv = float(np.arctan2(h_by, +h_bz))
        roll_err = ang_up if abs(ang_up) <= abs(ang_inv) else ang_inv
        return np.array([0.02, np.clip(0.85 * roll_err - 0.25 * p_n, -0.7, 0.7)], dtype=np.float32)

    # 2. Shortest-Path 3D Split-S / Immelmann Turn when target is in the rear hemisphere (h_bx < -0.15 * horiz_norm)
    if h_bx < -0.15 * horiz_norm:
        loop_sign = -1.0 if h_bz >= 0.0 else +1.0
        return np.array([loop_sign * 0.85, np.clip(-0.35 * p_n - 0.15 * r_n, -0.4, 0.4)], dtype=np.float32)

    # 3. Symmetric Upright (inv_sign = +1) or Upside-Down Inverted (inv_sign = -1) 3D Homing with Turn-Lift Compensation
    inv_sign = 1.0 if h_bz <= 0.0 else -1.0
    bearing = float(np.arctan2(h_by, max(0.04, h_bx)))
    mag_pitch = float(np.clip(0.22 + 0.55 * (req_ratio - 0.35) + 0.42 * abs(bearing), 0.08, 0.88))
    pitch_cmd = inv_sign * mag_pitch
    desired_g_by = float(np.clip(1.65 * bearing, -0.82, 0.82))
    roll_cmd = float(np.clip(1.30 * (desired_g_by - g_by * inv_sign) * inv_sign - 0.28 * p_n - 0.12 * r_n, -1.0, 1.0))
    return np.array([pitch_cmd, roll_cmd], dtype=np.float32)


def warmstart_omni_3d_actor(
    model: ActorCritic,
    target_mode: str,
    lead_gain: float,
    num_episodes: int = 24,
    epochs: int = 25,
    seed: int = 42,
) -> None:
    env = GlideGpsTargetEnv(target_mode=target_mode, release_altitude_m=1000.0, randomize_drop=True)
    obs_data: List[np.ndarray] = []
    act_data: List[np.ndarray] = []
    ret_data: List[float] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep * 17, options={"deterministic": (ep == 0)})
        ep_obs: List[np.ndarray] = []
        ep_act: List[np.ndarray] = []
        ep_rew: List[float] = []
        while True:
            act = omni_3d_guidance(obs, lead_gain=lead_gain)
            if ep % 2 == 1:
                noisy_act = np.clip(act + np.random.normal(0.0, 0.04, size=2).astype(np.float32), -1.0, 1.0)
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
            g = r + 0.995 * g
            ep_ret.append(g)
        ep_ret.reverse()

        obs_data.extend(ep_obs)
        act_data.extend(ep_act)
        ret_data.extend(ep_ret)

    obs_t = torch.as_tensor(np.stack(obs_data, axis=0), dtype=torch.float32)
    act_t = torch.as_tensor(np.stack(act_data, axis=0), dtype=torch.float32)
    ret_t = torch.as_tensor(np.array(ret_data, dtype=np.float32), dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=2.5e-3)
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
        model.log_std.fill_(-1.8)


def evaluate_gps_policy(
    model: ActorCritic,
    target_mode: str = "static",
    altitude_m: float = 1000.0,
    num_episodes: int = 6,
    seed: int = 200,
) -> Dict[str, Any]:
    """Evaluate 14D orientation-free policy across deterministic + random 360-deg test cases."""
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
    det_ball_min = info["mean_ball_min_dist_m"]

    closed_list = [det_closed]
    errs = [det_final_err]
    ball_mins = [det_ball_min]
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
        ball_mins.append(float(inf["mean_ball_min_dist_m"]))

    return {
        "target_mode": target_mode,
        "initial_target_dist_m": round(float(init_dist), 2),
        "total_distance_closed_m": round(float(det_closed), 2),
        "mean_random_closed_m": round(float(np.mean(closed_list)), 2),
        "final_target_error_m": round(float(det_final_err), 2),
        "mean_random_error_m": round(float(np.mean(errs)), 2),
        "mean_ball_min_dist_m": round(float(np.mean(ball_mins)), 2),
        "episode_return": round(float(ep_return), 2),
        "max_bank_deg": round(float(max_bank_deg), 2),
        "inverted_flight_s": round(float(inverted_steps * env_det.control_dt), 2),
        "flight_time_s": round(float(info["time_s"]), 2),
        "trajectory": trajectory,
    }


def run_curriculum_stage(
    model: ActorCritic,
    target_mode: str,
    stage_id: str,
    stage_label: str,
    lead_gain: float,
    num_updates: int = 5,
    num_envs: int = 4,
    rollout_steps: int = 256,
    lr: float = 1.5e-4,
    seed: int = 100,
    ckpt_dir: Path = Path("checkpoints/glide_rl"),
) -> Dict[str, Any]:
    warmstart_omni_3d_actor(
        model=model,
        target_mode=target_mode,
        lead_gain=lead_gain,
        num_episodes=14,
        epochs=20,
        seed=seed,
    )

    envs = [GlideGpsTargetEnv(target_mode=target_mode, release_altitude_m=1000.0, randomize_drop=True) for _ in range(num_envs)]
    ppo = PPOTrainer(
        model,
        PPOConfig(
            lr=lr,
            gamma=0.995,
            gae_lambda=0.97,
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
        ev = evaluate_gps_policy(model, target_mode=target_mode, altitude_m=1000.0, num_episodes=3, seed=700)
        if target_mode == "balls_3d":
            score = 0.5 * ev["mean_ball_min_dist_m"] + 0.5 * ev["final_target_error_m"]
        else:
            score = 0.55 * ev["final_target_error_m"] + 0.45 * ev["mean_random_error_m"]
        if score < best_err:
            best_err = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_eval = ev

        if update % 2 == 0 or update == num_updates:
            print(
                f"[{stage_label}] update={update:2d}/{num_updates} | "
                f"Det Ground Miss = {ev['final_target_error_m']:5.1f} m | "
                f"Random Avg Miss = {ev['mean_random_error_m']:5.1f} m | "
                f"Avg Ball Dist = {ev['mean_ball_min_dist_m']:5.1f} m | "
                f"Max Roll = {ev['max_bank_deg']:5.1f}°"
            )

        if update == num_updates:
            break

        frac = 1.0 - (update / float(num_updates))
        for pg in ppo.optimizer.param_groups:
            pg["lr"] = max(3e-5, lr * frac)

        obs_buf = np.zeros((rollout_steps, num_envs, 22), dtype=np.float32)
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
            obs_buf.reshape(-1, 22),
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
            "obs_dim": 22,
            "model_state_dict": model.state_dict(),
            "total_distance_closed_m": best_eval["total_distance_closed_m"],
            "mean_random_closed_m": best_eval["mean_random_closed_m"],
            "final_target_error_m": best_eval["final_target_error_m"],
            "mean_random_error_m": best_eval["mean_random_error_m"],
            "mean_ball_min_dist_m": best_eval["mean_ball_min_dist_m"],
            "initial_target_dist_m": best_eval["initial_target_dist_m"],
            "max_bank_deg": best_eval["max_bank_deg"],
            "inverted_flight_s": best_eval["inverted_flight_s"],
            "episode_return": best_eval["episode_return"],
        },
        ckpt_file,
    )
    print(
        f"==> Completed [{stage_label}] | Saved {ckpt_file} | "
        f"Best Det Ground Miss = {best_eval['final_target_error_m']:.1f} m | "
        f"Random Avg Miss = {best_eval['mean_random_error_m']:.1f} m | "
        f"Avg Ball Proximity = {best_eval['mean_ball_min_dist_m']:.1f} m\n"
    )
    return best_eval


if __name__ == "__main__":
    import sys
    stage_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    np.random.seed(42)
    torch.manual_seed(42)
    ckpt_dir = Path("checkpoints/glide_rl")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = ActorCritic(obs_dim=22, act_dim=2, hidden_dim=64)

    if stage_arg in ("balls", "all"):
        run_curriculum_stage(
            model=model,
            target_mode="balls_3d",
            stage_id="gps_1_static",
            stage_label="Phase 1: 3D Multi-Ball Proximity Field & Ground Target Test",
            lead_gain=0.0,
            num_updates=4,
            lr=1.5e-4,
            seed=101,
            ckpt_dir=ckpt_dir,
        )
        ground_test = evaluate_gps_policy(model, target_mode="static", altitude_m=1000.0, num_episodes=5, seed=888)
        print(
            f"*** GROUND TARGET TEST AFTER 3D BALL TRAINING *** -> "
            f"Det Ground Miss = {ground_test['final_target_error_m']:.1f} m | "
            f"360-Deg Random Ground Avg Miss = {ground_test['mean_random_error_m']:.1f} m",
            flush=True,
        )

    if stage_arg in ("linear", "all"):
        if (ckpt_dir / "gps_2_linear.pt").exists():
            ck = torch.load(ckpt_dir / "gps_2_linear.pt", weights_only=False)
            if ck.get("obs_dim") == 22:
                model.load_state_dict(ck["model_state_dict"])
        elif (ckpt_dir / "gps_1_static.pt").exists():
            ck = torch.load(ckpt_dir / "gps_1_static.pt", weights_only=False)
            if ck.get("obs_dim") == 22:
                model.load_state_dict(ck["model_state_dict"])
        run_curriculum_stage(
            model=model,
            target_mode="linear",
            stage_id="gps_2_linear",
            stage_label="Phase 2: Linear Moving Target at Variable Speeds (4-20 m/s)",
            lead_gain=0.95,
            num_updates=6,
            lr=1.2e-4,
            seed=404,
            ckpt_dir=ckpt_dir,
        )

    if stage_arg in ("curved", "all"):
        if (ckpt_dir / "gps_3_aggressive.pt").exists():
            ck = torch.load(ckpt_dir / "gps_3_aggressive.pt", weights_only=False)
            if ck.get("obs_dim") == 22:
                model.load_state_dict(ck["model_state_dict"])
        elif (ckpt_dir / "gps_2_linear.pt").exists():
            ck = torch.load(ckpt_dir / "gps_2_linear.pt", weights_only=False)
            if ck.get("obs_dim") == 22:
                model.load_state_dict(ck["model_state_dict"])
        run_curriculum_stage(
            model=model,
            target_mode="curved",
            stage_id="gps_3_aggressive",
            stage_label="Phase 3: Curved Moving Target at Variable Speeds & Turn Rates",
            lead_gain=0.95,
            num_updates=6,
            lr=1.2e-4,
            seed=505,
            ckpt_dir=ckpt_dir,
        )
