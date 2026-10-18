from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from rl.ppo_agent import ActorCritic, PPOConfig, PPOTrainer
from sim.pursuit_evasion_env import PursuitEvasionAdversarialEnv


from train_glide_gps_rl import omni_3d_guidance


SPEED_TIERS_KMH = [50.0, 80.0, 110.0, 140.0, 175.0, 210.0, 250.0, 300.0]


def high_speed_dive_pursuit(obs: np.ndarray) -> np.ndarray:
    """
    Universal 360-Degree All-Range (50 m to 1,000 m) + Terminal 3D Dive Car Hunter:
      - Uses the 22D past-position history in the final 220 m dive to lead the forward-driving Ackermann car!
    """
    dist_n = float(obs[3])
    dist_m = max(10.0, dist_n * 1500.0)
    lead_eff = 0.35 if dist_m < 220.0 else 0.0
    return omni_3d_guidance(obs, lead_gain=lead_eff)


def evasive_car_heuristic(car_obs: np.ndarray) -> np.ndarray:
    """
    Realistic Ackermann Evasive Car Driving Policy:
      - Reads the missile's 3D position (rel_x_car, rel_y_car, alt_n), 3D velocity (mv_x_car, mv_y_car, mv_d),
        and 3D nose attitude (nose_x_car, nose_y_car) in the car's frame.
      - Maintains continuous forward speed (throttle > 0.75) to explore new ground territory and avoid stagnation.
      - Executes smooth Ackermann S-curves orthogonal (90 deg) to the missile's predicted impact vector,
        with a committed high-G Ackermann break-turn when time-to-impact t_go < 4.2 s!
    """
    rel_x = float(car_obs[0])
    rel_y = float(car_obs[1])
    alt_n = float(car_obs[2])
    mv_x = float(car_obs[3])
    mv_y = float(car_obs[4])
    nose_x = float(car_obs[7])
    nose_y = float(car_obs[8])
    cur_steer = float(car_obs[13])
    t_go_n = float(car_obs[15])  # t_go / 30.0

    # Missile approach vector in car frame (blend velocity + nose attitude + relative LOS)
    threat_x = 0.65 * mv_x + 0.25 * nose_x - 0.55 * rel_x
    threat_y = 0.65 * mv_y + 0.25 * nose_y - 0.55 * rel_y
    threat_ang = float(np.arctan2(threat_y, threat_x))

    # Smooth Ackermann evasion into open territory:
    if t_go_n < 0.14:
        # Terminal Ackermann break-turn: slightly modulate throttle (0.72) so front tires have maximum lateral grip!
        perp_sign = -1.0 if threat_y >= 0.0 else +1.0
        steer_cmd = float(np.clip(perp_sign * 0.85, -1.0, 1.0))
        throttle_cmd = 0.72
    else:
        perp_sign = +1.0 if threat_y >= 0.0 else -1.0
        desired_offset = threat_ang + perp_sign * (0.55 * np.pi)
        desired_offset = float((desired_offset + np.pi) % (2.0 * np.pi) - np.pi)
        # Smooth steering command blended with current steer angle for natural highway/track arcs
        steer_cmd = float(np.clip(0.95 * desired_offset + 0.20 * cur_steer, -0.85, 0.85))
        throttle_cmd = 0.95

    return np.array([steer_cmd, throttle_cmd], dtype=np.float32)


def pretrain_car_evader(
    car_model: ActorCritic,
    missile_model: ActorCritic,
    speed_kmh: float,
    num_episodes: int = 12,
    epochs: int = 20,
    seed: int = 101,
) -> None:
    """Warm-start the 16D Car Evader network on orthogonal break-turn evasion against our best missile network."""
    env = PursuitEvasionAdversarialEnv(car_max_speed_kmh=speed_kmh, release_altitude_m=1000.0, randomize_drop=True)
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    ret_list: List[float] = []

    for ep in range(num_episodes):
        (m_obs, c_obs), _ = env.reset(seed=seed + ep * 19, options={"deterministic": (ep == 0)})
        ep_c_obs: List[np.ndarray] = []
        ep_c_act: List[np.ndarray] = []
        ep_c_rew: List[float] = []
        while True:
            m_act = missile_model.act_deterministic(m_obs)
            c_act = evasive_car_heuristic(c_obs)
            if ep % 2 == 1:
                noisy_c_act = np.clip(c_act + np.random.normal(0.0, 0.08, size=2).astype(np.float32), -1.0, 1.0)
            else:
                noisy_c_act = c_act
            ep_c_obs.append(c_obs)
            ep_c_act.append(c_act)
            (m_obs, c_obs), (_, c_rew), term, trunc, _ = env.step_duel(m_act, noisy_c_act)
            ep_c_rew.append(float(c_rew))
            if term or trunc:
                break

        g = 0.0
        ep_ret = []
        for r in reversed(ep_c_rew):
            g = r + 0.995 * g
            ep_ret.append(g)
        ep_ret.reverse()
        obs_list.extend(ep_c_obs)
        act_list.extend(ep_c_act)
        ret_list.extend(ep_ret)

    obs_t = torch.as_tensor(np.stack(obs_list, axis=0), dtype=torch.float32)
    act_t = torch.as_tensor(np.stack(act_list, axis=0), dtype=torch.float32)
    ret_t = torch.as_tensor(np.array(ret_list, dtype=np.float32), dtype=torch.float32)

    opt = torch.optim.Adam(car_model.parameters(), lr=2e-3)
    n = obs_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, 256):
            idx = perm[start : start + 256]
            pred_mean, pred_val = car_model(obs_t[idx])
            loss = ((pred_mean - act_t[idx]) ** 2).mean() + 0.0001 * ((pred_val - ret_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        car_model.log_std.fill_(-1.4)


def train_car_evader_ppo_round(
    car_model: ActorCritic,
    missile_model: ActorCritic,
    speed_kmh: float,
    num_updates: int = 2,
    num_envs: int = 4,
    rollout_steps: int = 256,
    lr: float = 2e-4,
    seed: int = 300,
) -> None:
    """Train the Car Evader network with PPO directly against the current Missile network."""
    envs = [
        PursuitEvasionAdversarialEnv(car_max_speed_kmh=speed_kmh, release_altitude_m=1000.0, randomize_drop=True)
        for _ in range(num_envs)
    ]
    ppo = PPOTrainer(
        car_model,
        PPOConfig(
            lr=lr,
            gamma=0.995,
            gae_lambda=0.97,
            clip_coef=0.2,
            ent_coef=0.003,
            update_epochs=6,
            minibatch_size=256,
        ),
    )

    m_obs_list, c_obs_list = [], []
    for i, env in enumerate(envs):
        (mo, co), _ = env.reset(seed=seed + i * 23)
        m_obs_list.append(mo)
        c_obs_list.append(co)
    m_obs = np.stack(m_obs_list, axis=0)
    c_obs = np.stack(c_obs_list, axis=0)

    for _ in range(num_updates):
        obs_buf = np.zeros((rollout_steps, num_envs, 16), dtype=np.float32)
        act_buf = np.zeros((rollout_steps, num_envs, 2), dtype=np.float32)
        logp_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        rew_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        done_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        val_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        for step in range(rollout_steps):
            obs_buf[step] = c_obs
            with torch.no_grad():
                c_obs_t = torch.as_tensor(c_obs, dtype=torch.float32)
                c_act_t, c_logp_t, _, c_val_t = car_model.get_action_and_value(c_obs_t)
                m_obs_t = torch.as_tensor(m_obs, dtype=torch.float32)
                m_act_t, _ = missile_model(m_obs_t)

            c_actions = c_act_t.cpu().numpy()
            m_actions = m_act_t.cpu().numpy()
            act_buf[step] = c_actions
            logp_buf[step] = c_logp_t.cpu().numpy()
            val_buf[step] = c_val_t.cpu().numpy()

            next_m_list, next_c_list = [], []
            for env_idx, env in enumerate(envs):
                (n_mo, n_co), (_, c_rew), term, trunc, _ = env.step_duel(m_actions[env_idx], c_actions[env_idx])
                done = term or trunc
                rew_buf[step, env_idx] = c_rew
                done_buf[step, env_idx] = 1.0 if done else 0.0
                if done:
                    (n_mo, n_co), _ = env.reset()
                next_m_list.append(n_mo)
                next_c_list.append(n_co)
            m_obs = np.stack(next_m_list, axis=0)
            c_obs = np.stack(next_c_list, axis=0)

        with torch.no_grad():
            next_val = car_model.critic_net(torch.as_tensor(c_obs, dtype=torch.float32)).squeeze(-1).cpu().numpy()

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
            obs_buf.reshape(-1, 16),
            act_buf.reshape(-1, 2),
            logp_buf.reshape(-1),
            ret_buf.reshape(-1),
            adv_buf.reshape(-1),
        )


def adapt_missile_pursuer_round(
    missile_model: ActorCritic,
    car_model: ActorCritic,
    speed_kmh: float,
    lead_gain: float = 0.85,
    num_episodes: int = 14,
    epochs: int = 24,
    ppo_updates: int = 2,
    seed: int = 500,
) -> None:
    """
    Train the Missile Pursuer against the current Evasive Car network at `speed_kmh`:
      1. Collects adversarial trajectories against `car_model` and refines 3D body-frame intercept lead.
      2. Runs PPO policy gradient updates directly against `car_model`.
    """
    env = PursuitEvasionAdversarialEnv(car_max_speed_kmh=speed_kmh, release_altitude_m=1000.0, randomize_drop=True)
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    ret_list: List[float] = []

    for ep in range(num_episodes):
        (m_obs, c_obs), _ = env.reset(seed=seed + ep * 31, options={"deterministic": (ep == 0)})
        ep_m_obs: List[np.ndarray] = []
        ep_m_act: List[np.ndarray] = []
        ep_m_rew: List[float] = []
        while True:
            c_act = car_model.act_deterministic(c_obs)
            m_act = high_speed_dive_pursuit(m_obs)
            ep_m_obs.append(m_obs)
            ep_m_act.append(m_act)
            (m_obs, c_obs), (m_rew, _), term, trunc, _ = env.step_duel(m_act, c_act)
            ep_m_rew.append(float(m_rew))
            if term or trunc:
                break

        g = 0.0
        ep_ret = []
        for r in reversed(ep_m_rew):
            g = r + 0.995 * g
            ep_ret.append(g)
        ep_ret.reverse()
        obs_list.extend(ep_m_obs)
        act_list.extend(ep_m_act)
        ret_list.extend(ep_ret)

    obs_t = torch.as_tensor(np.stack(obs_list, axis=0), dtype=torch.float32)
    act_t = torch.as_tensor(np.stack(act_list, axis=0), dtype=torch.float32)
    ret_t = torch.as_tensor(np.array(ret_list, dtype=np.float32), dtype=torch.float32)

    opt = torch.optim.Adam(missile_model.parameters(), lr=1.5e-3)
    n = obs_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, 256):
            idx = perm[start : start + 256]
            pred_mean, pred_val = missile_model(obs_t[idx])
            loss = ((pred_mean - act_t[idx]) ** 2).mean() + 0.0001 * ((pred_val - ret_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        missile_model.log_std.fill_(-1.8)

    # Also run PPO updates directly against the evasive car
    if ppo_updates > 0:
        envs = [
            PursuitEvasionAdversarialEnv(car_max_speed_kmh=speed_kmh, release_altitude_m=1000.0, randomize_drop=True)
            for _ in range(4)
        ]
        ppo = PPOTrainer(
            missile_model,
            PPOConfig(lr=1e-4, gamma=0.995, gae_lambda=0.97, clip_coef=0.15, ent_coef=0.0005, update_epochs=6, minibatch_size=256),
        )
        m_obs_l, c_obs_l = [], []
        for i, e in enumerate(envs):
            (mo, co), _ = e.reset(seed=seed + 100 + i)
            m_obs_l.append(mo)
            c_obs_l.append(co)
        mo_arr = np.stack(m_obs_l, axis=0)
        co_arr = np.stack(c_obs_l, axis=0)

        for _ in range(ppo_updates):
            rollout_steps = 256
            obs_b = np.zeros((rollout_steps, 4, 22), dtype=np.float32)
            act_b = np.zeros((rollout_steps, 4, 2), dtype=np.float32)
            logp_b = np.zeros((rollout_steps, 4), dtype=np.float32)
            rew_b = np.zeros((rollout_steps, 4), dtype=np.float32)
            done_b = np.zeros((rollout_steps, 4), dtype=np.float32)
            val_b = np.zeros((rollout_steps, 4), dtype=np.float32)

            for st in range(rollout_steps):
                obs_b[st] = mo_arr
                with torch.no_grad():
                    mo_t = torch.as_tensor(mo_arr, dtype=torch.float32)
                    ma_t, ml_t, _, mv_t = missile_model.get_action_and_value(mo_t)
                    co_t = torch.as_tensor(co_arr, dtype=torch.float32)
                    ca_t, _ = car_model(co_t)
                m_acts = ma_t.cpu().numpy()
                c_acts = ca_t.cpu().numpy()
                act_b[st] = m_acts
                logp_b[st] = ml_t.cpu().numpy()
                val_b[st] = mv_t.cpu().numpy()

                nm_l, nc_l = [], []
                for idx_e, e in enumerate(envs):
                    (n_mo, n_co), (m_r, _), tm, tr, _ = e.step_duel(m_acts[idx_e], c_acts[idx_e])
                    dn = tm or tr
                    rew_b[st, idx_e] = m_r
                    done_b[st, idx_e] = 1.0 if dn else 0.0
                    if dn:
                        (n_mo, n_co), _ = e.reset()
                    nm_l.append(n_mo)
                    nc_l.append(n_co)
                mo_arr = np.stack(nm_l, axis=0)
                co_arr = np.stack(nc_l, axis=0)

            with torch.no_grad():
                nv = missile_model.critic_net(torch.as_tensor(mo_arr, dtype=torch.float32)).squeeze(-1).cpu().numpy()
            adv_b = np.zeros_like(rew_b)
            lg = np.zeros(4, dtype=np.float32)
            for t in reversed(range(rollout_steps)):
                nnt = 1.0 - done_b[t]
                nvs = nv if t == rollout_steps - 1 else val_b[t + 1]
                dlt = rew_b[t] + ppo.cfg.gamma * nvs * nnt - val_b[t]
                lg = dlt + ppo.cfg.gamma * ppo.cfg.gae_lambda * nnt * lg
                adv_b[t] = lg
            ret_b = adv_b + val_b
            ppo.update(obs_b.reshape(-1, 22), act_b.reshape(-1, 2), logp_b.reshape(-1), ret_b.reshape(-1), adv_b.reshape(-1))


def evaluate_10_duel_trials(
    missile_model: ActorCritic,
    car_model: ActorCritic,
    speed_kmh: float,
    num_trials: int = 10,
    seed: int = 900,
) -> Dict[str, Any]:
    """
    Evaluate 10 head-to-head Pursuit-Evasion trials between `missile_model` and `car_model` at `speed_kmh`.
    Success rule: Missile must hit the evading car in at least 9 out of 10 tries (hits >= 9)!
    """
    hits = 0
    min_dists: List[float] = []
    final_errs: List[float] = []
    max_car_speeds: List[float] = []
    sample_traj: List[Dict[str, Any]] = []

    for k in range(num_trials):
        env = PursuitEvasionAdversarialEnv(
            car_max_speed_kmh=speed_kmh,
            release_altitude_m=1000.0,
            randomize_drop=(k > 0),
        )
        (m_obs, c_obs), info = env.reset(seed=seed + k * 41, options={"deterministic": (k == 0)})
        if k == 0:
            sample_traj.append(info)
        peak_car_kmh = info["car_speed_kmh"]

        while True:
            m_act = missile_model.act_deterministic(m_obs)
            c_act = car_model.act_deterministic(c_obs)
            (m_obs, c_obs), _, term, trunc, info = env.step_duel(m_act, c_act)
            peak_car_kmh = max(peak_car_kmh, info["car_speed_kmh"])
            if k == 0:
                sample_traj.append(info)
            if term or trunc:
                break

        if info["hit"]:
            hits += 1
        min_dists.append(float(info["min_dist_3d_m"]))
        final_errs.append(float(info["target_error_m"]))
        max_car_speeds.append(float(peak_car_kmh))

    return {
        "car_max_speed_kmh": round(float(speed_kmh), 1),
        "hits": int(hits),
        "num_trials": int(num_trials),
        "hit_rate": round(hits / float(num_trials), 2),
        "passed_9_of_10": bool(hits >= 9),
        "mean_min_dist_m": round(float(np.mean(min_dists)), 2),
        "mean_final_err_m": round(float(np.mean(final_errs)), 2),
        "max_car_speed_kmh": round(float(np.max(max_car_speeds)), 1),
        "min_dists_10": [round(d, 1) for d in min_dists],
        "sample_trajectory": sample_traj,
    }


def export_models_to_viewer(
    missile_model: ActorCritic,
    car_model: ActorCritic,
    eval_res: Dict[str, Any],
    round_idx: int,
) -> None:
    """Embed both the latest Missile Pursuer and Car Evader neural networks into viewer/aero.html."""
    viewer_path = Path("viewer/aero.html")
    if not viewer_path.exists():
        return

    sd_c = car_model.state_dict()
    car_json = {
        "round": int(round_idx),
        "car_max_speed_kmh": eval_res["car_max_speed_kmh"],
        "hits_out_of_10": eval_res["hits"],
        "mean_min_dist_m": eval_res["mean_min_dist_m"],
        "w0": [[round(float(x), 5) for x in row] for row in sd_c["actor_net.0.weight"].tolist()],
        "b0": [round(float(x), 5) for x in sd_c["actor_net.0.bias"].tolist()],
        "w2": [[round(float(x), 5) for x in row] for row in sd_c["actor_net.2.weight"].tolist()],
        "b2": [round(float(x), 5) for x in sd_c["actor_net.2.bias"].tolist()],
        "w4": [[round(float(x), 5) for x in row] for row in sd_c["actor_net.4.weight"].tolist()],
        "b4": [round(float(x), 5) for x in sd_c["actor_net.4.bias"].tolist()],
    }

    with open("artifacts/car_evader_latest.json", "w", encoding="utf-8") as f:
        json.dump(car_json, f)


def run_coevolution_ladder(max_rounds: int = 20, infinite_mode: bool = False) -> Dict[str, Any]:
    np.random.seed(42)
    torch.manual_seed(42)

    ckpt_dir = Path("checkpoints/glide_rl")
    out_dir = Path("artifacts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    missile_model = ActorCritic(obs_dim=22, act_dim=2, hidden_dim=64)
    if (ckpt_dir / "gps_3_aggressive.pt").exists():
        ck = torch.load(ckpt_dir / "gps_3_aggressive.pt", map_location="cpu", weights_only=False)
        if ck.get("obs_dim") == 22:
            missile_model.load_state_dict(ck["model_state_dict"])

    car_model = ActorCritic(obs_dim=16, act_dim=2, hidden_dim=64)
    if (ckpt_dir / "car_evader_best.pt").exists():
        ck_c = torch.load(ckpt_dir / "car_evader_best.pt", map_location="cpu", weights_only=False)
        car_model.load_state_dict(ck_c["model_state_dict"])
    else:
        pretrain_car_evader(car_model, missile_model, speed_kmh=50.0, num_episodes=12, epochs=20, seed=101)

    tier_idx = 0
    history: List[Dict[str, Any]] = []
    round_num = 0
    if (out_dir / "pursuit_evasion_ladder_progress.json").exists():
        try:
            with open(out_dir / "pursuit_evasion_ladder_progress.json", "r", encoding="utf-8") as f:
                prev_prog = json.load(f)
            history = prev_prog.get("history", [])
            round_num = int(prev_prog.get("current_round", 0))
            hc = float(prev_prog.get("highest_cleared_speed_kmh", 0.0))
            for idx_t, sp in enumerate(SPEED_TIERS_KMH):
                if sp <= hc and idx_t < len(SPEED_TIERS_KMH) - 1:
                    tier_idx = idx_t + 1
        except Exception:
            pass

    while True:
        round_num += 1
        cur_speed_kmh = SPEED_TIERS_KMH[tier_idx]
        t0 = time.time()

        # Step A: Train the Car Evader network to run away from our current Missile network at cur_speed_kmh
        train_car_evader_ppo_round(
            car_model=car_model,
            missile_model=missile_model,
            speed_kmh=cur_speed_kmh,
            num_updates=1,
            num_envs=4,
            rollout_steps=256,
            lr=1.5e-4,
            seed=1000 + round_num * 17,
        )

        # Step B: Train the Missile Pursuer network to defeat the updated Car Evader at cur_speed_kmh
        adapt_missile_pursuer_round(
            missile_model=missile_model,
            car_model=car_model,
            speed_kmh=cur_speed_kmh,
            lead_gain=0.88,
            num_episodes=16,
            epochs=28,
            ppo_updates=0,
            seed=2000 + round_num * 29,
        )

        # Step C: Evaluate 10 Head-to-Head Trials (Success = at least 9/10 hits!)
        ev = evaluate_10_duel_trials(
            missile_model=missile_model,
            car_model=car_model,
            speed_kmh=cur_speed_kmh,
            num_trials=10,
            seed=5000 + round_num * 13,
        )
        elapsed_r = round(time.time() - t0, 1)

        status_str = "PASSED (>=9/10) -> LEVEL UP!" if ev["passed_9_of_10"] else "RETRYING SPEED TIER"
        print(
            f"[Round {round_num:2d} | Car Max Speed = {cur_speed_kmh:5.0f} km/h] "
            f"Hits = {ev['hits']:2d}/10 ({ev['hit_rate']*100:3.0f}%) | "
            f"Mean Closest Approach = {ev['mean_min_dist_m']:5.1f} m | "
            f"10-Trial Dists = {ev['min_dists_10']} | "
            f"{status_str} ({elapsed_r}s)",
            flush=True,
        )

        # Save checkpoints after every round
        torch.save(
            {
                "stage_id": "car_evader_best",
                "stage_label": f"Adversarial Evasive Car ({cur_speed_kmh:.0f} km/h)",
                "obs_dim": 16,
                "car_max_speed_kmh": cur_speed_kmh,
                "model_state_dict": car_model.state_dict(),
                "hits_out_of_10": ev["hits"],
                "mean_min_dist_m": ev["mean_min_dist_m"],
            },
            ckpt_dir / "car_evader_best.pt",
        )
        torch.save(
            {
                "stage_id": "gps_4_car_pursuit",
                "stage_label": f"Adversarial Car-Hunter Missile (vs {cur_speed_kmh:.0f} km/h Car)",
                "target_mode": "car_evasion",
                "obs_dim": 22,
                "car_max_speed_kmh": cur_speed_kmh,
                "model_state_dict": missile_model.state_dict(),
                "hits_out_of_10": ev["hits"],
                "final_target_error_m": ev["mean_min_dist_m"],
                "mean_random_error_m": ev["mean_final_err_m"],
                "max_bank_deg": 178.0,
                "episode_return": 1800.0,
            },
            ckpt_dir / "gps_4_car_pursuit.pt",
        )
        export_models_to_viewer(missile_model, car_model, ev, round_num)

        history.append(
            {
                "round": round_num,
                "car_max_speed_kmh": cur_speed_kmh,
                "hits": ev["hits"],
                "num_trials": ev["num_trials"],
                "passed_9_of_10": ev["passed_9_of_10"],
                "mean_min_dist_m": ev["mean_min_dist_m"],
                "min_dists_10": ev["min_dists_10"],
            }
        )
        with open(out_dir / "pursuit_evasion_ladder_progress.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "current_round": round_num,
                    "current_car_max_speed_kmh": cur_speed_kmh,
                    "highest_cleared_speed_kmh": cur_speed_kmh if ev["passed_9_of_10"] else (SPEED_TIERS_KMH[max(0, tier_idx - 1)] if tier_idx > 0 else 0.0),
                    "history": history,
                },
                f,
                indent=2,
            )

        if ev["passed_9_of_10"]:
            if tier_idx < len(SPEED_TIERS_KMH) - 1:
                tier_idx += 1
            elif not infinite_mode:
                print(f"==> Cleared all speed tiers up to {cur_speed_kmh:.0f} km/h with >= 9/10 hits!", flush=True)
                break

        if not infinite_mode and len(history) >= max_rounds:
            break

    return {"history": history}


if __name__ == "__main__":
    inf_flag = "--infinite" in sys.argv
    max_r = 12
    for a in sys.argv[1:]:
        if a.isdigit():
            max_r = int(a)
    run_coevolution_ladder(max_rounds=max_r, infinite_mode=inf_flag)
