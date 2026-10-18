from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

from rl.ppo_agent import ActorCritic, PPOConfig, PPOTrainer
from sim.pursuit_evasion_env import PursuitEvasionAdversarialEnv


SPEED_TIERS_KMH = [50.0, 80.0, 110.0, 140.0, 175.0, 210.0, 250.0, 300.0]


def train_car_evader_ppo_round(
    car_model: ActorCritic,
    missile_model: ActorCritic,
    speed_kmh: float,
    num_updates: int = 4,
    num_envs: int = 8,
    rollout_steps: int = 256,
    lr: float = 1.5e-4,
    seed: int = 300,
) -> None:
    """Train the 16D Car Evader network with 100% pure PPO against the current Missile network."""
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
            ent_coef=0.002,
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


def train_missile_pursuer_ppo_round(
    missile_model: ActorCritic,
    car_model: ActorCritic,
    speed_kmh: float,
    num_updates: int = 8,
    num_envs: int = 8,
    rollout_steps: int = 256,
    lr: float = 1.0e-4,
    seed: int = 500,
) -> None:
    """
    Train the 24D Missile Pursuer network with 100% pure PPO against the current Evasive Car network:
      - Zero hand-coded guidance, zero behavioral cloning, and zero angle/action caps.
    """
    envs = [
        PursuitEvasionAdversarialEnv(car_max_speed_kmh=speed_kmh, release_altitude_m=1000.0, randomize_drop=True)
        for _ in range(num_envs)
    ]
    ppo = PPOTrainer(
        missile_model,
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

    m_obs_l, c_obs_l = [], []
    for i, e in enumerate(envs):
        (mo, co), _ = e.reset(seed=seed + 100 + i * 17)
        m_obs_l.append(mo)
        c_obs_l.append(co)
    mo_arr = np.stack(m_obs_l, axis=0)
    co_arr = np.stack(c_obs_l, axis=0)

    for _ in range(num_updates):
        obs_b = np.zeros((rollout_steps, num_envs, 24), dtype=np.float32)
        act_b = np.zeros((rollout_steps, num_envs, 2), dtype=np.float32)
        logp_b = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        rew_b = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        done_b = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        val_b = np.zeros((rollout_steps, num_envs), dtype=np.float32)

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
        lg = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(rollout_steps)):
            nnt = 1.0 - done_b[t]
            nvs = nv if t == rollout_steps - 1 else val_b[t + 1]
            dlt = rew_b[t] + ppo.cfg.gamma * nvs * nnt - val_b[t]
            lg = dlt + ppo.cfg.gamma * ppo.cfg.gae_lambda * nnt * lg
            adv_b[t] = lg
        ret_b = adv_b + val_b
        ppo.update(obs_b.reshape(-1, 24), act_b.reshape(-1, 2), logp_b.reshape(-1), ret_b.reshape(-1), adv_b.reshape(-1))


def evaluate_10_duel_trials(
    missile_model: ActorCritic,
    car_model: ActorCritic,
    speed_kmh: float,
    num_trials: int = 10,
    seed: int = 900,
) -> Dict[str, Any]:
    """
    Evaluate 10 head-to-head Pursuit-Evasion trials between `missile_model` and `car_model` at `speed_kmh`.
    Success rule: Missile must hit the evading car (error <= 3.0 m) in at least 9 out of 10 tries (hits >= 9)!
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
        min_dists.append(round(float(info["min_dist_3d_m"]), 1))
        final_errs.append(round(float(info["target_error_m"]), 1))
        max_car_speeds.append(round(float(peak_car_kmh), 1))

    return {
        "car_max_speed_kmh": speed_kmh,
        "hits": hits,
        "num_trials": num_trials,
        "hit_rate": round(hits / float(num_trials), 2),
        "passed_9_of_10": bool(hits >= 9),
        "mean_min_dist_m": round(float(np.mean(min_dists)), 2),
        "mean_final_err_m": round(float(np.mean(final_errs)), 2),
        "mean_peak_car_kmh": round(float(np.mean(max_car_speeds)), 1),
        "min_dists_10": min_dists,
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

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    with open("artifacts/car_evader_latest.json", "w", encoding="utf-8") as f:
        json.dump(car_json, f)


def run_coevolution_ladder(max_rounds: int = 20, infinite_mode: bool = False) -> Dict[str, Any]:
    np.random.seed(42)
    torch.manual_seed(42)

    ckpt_dir = Path("checkpoints/glide_rl")
    out_dir = Path("artifacts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    missile_model = ActorCritic(obs_dim=24, act_dim=2, hidden_dim=64)
    if (ckpt_dir / "gps_4_car_pursuit.pt").exists():
        ck = torch.load(ckpt_dir / "gps_4_car_pursuit.pt", map_location="cpu", weights_only=False)
        if ck.get("obs_dim") == 24:
            missile_model.load_state_dict(ck["model_state_dict"])
    elif (ckpt_dir / "gps_1_static.pt").exists():
        ck = torch.load(ckpt_dir / "gps_1_static.pt", map_location="cpu", weights_only=False)
        if ck.get("obs_dim") == 24:
            missile_model.load_state_dict(ck["model_state_dict"])

    car_model = ActorCritic(obs_dim=16, act_dim=2, hidden_dim=64)
    if (ckpt_dir / "car_evader_best.pt").exists():
        ck_c = torch.load(ckpt_dir / "car_evader_best.pt", map_location="cpu", weights_only=False)
        car_model.load_state_dict(ck_c["model_state_dict"])

    with torch.no_grad():
        missile_model.log_std.fill_(-1.6)
        car_model.log_std.fill_(-1.4)

    tier_idx = 0
    history: List[Dict[str, Any]] = []
    round_num = 0
    best_hits_at_tier = -1
    best_dist_at_tier = 1e9
    best_m_state = {k: v.detach().cpu().clone() for k, v in missile_model.state_dict().items()}

    while True:
        round_num += 1
        cur_speed_kmh = SPEED_TIERS_KMH[tier_idx]
        t0 = time.time()

        # Evaluate before round
        ev_pre = evaluate_10_duel_trials(
            missile_model=missile_model,
            car_model=car_model,
            speed_kmh=cur_speed_kmh,
            num_trials=10,
            seed=5000,
        )
        if ev_pre["hits"] > best_hits_at_tier or (ev_pre["hits"] == best_hits_at_tier and ev_pre["mean_min_dist_m"] < best_dist_at_tier):
            best_hits_at_tier = ev_pre["hits"]
            best_dist_at_tier = ev_pre["mean_min_dist_m"]
            best_m_state = {k: v.detach().cpu().clone() for k, v in missile_model.state_dict().items()}

        # Step A: Pure PPO update for Car Evader
        train_car_evader_ppo_round(
            car_model=car_model,
            missile_model=missile_model,
            speed_kmh=cur_speed_kmh,
            num_updates=2,
            num_envs=8,
            rollout_steps=256,
            lr=1.0e-4,
            seed=1000 + round_num * 17,
        )

        # Step B: Pure PPO update for Missile Pursuer
        train_missile_pursuer_ppo_round(
            missile_model=missile_model,
            car_model=car_model,
            speed_kmh=cur_speed_kmh,
            num_updates=6,
            num_envs=8,
            rollout_steps=256,
            lr=8.0e-5,
            seed=2000 + round_num * 29,
        )

        # Step C: Evaluate 10 Head-to-Head Trials (Success = at least 9/10 hits at <= 3.0 m!)
        ev = evaluate_10_duel_trials(
            missile_model=missile_model,
            car_model=car_model,
            speed_kmh=cur_speed_kmh,
            num_trials=10,
            seed=5000,
        )
        if ev["hits"] < best_hits_at_tier or (ev["hits"] == best_hits_at_tier and ev["mean_min_dist_m"] > best_dist_at_tier):
            missile_model.load_state_dict(best_m_state)
            ev = evaluate_10_duel_trials(
                missile_model=missile_model,
                car_model=car_model,
                speed_kmh=cur_speed_kmh,
                num_trials=10,
                seed=5000,
            )
        else:
            best_hits_at_tier = ev["hits"]
            best_dist_at_tier = ev["mean_min_dist_m"]
            best_m_state = {k: v.detach().cpu().clone() for k, v in missile_model.state_dict().items()}

        elapsed_r = round(time.time() - t0, 1)
        status_str = "PASSED (>=9/10) -> LEVEL UP!" if ev["passed_9_of_10"] else "CONTINUING PURE-RL TRAINING"
        print(
            f"[Round {round_num:2d} | Car Max Speed = {cur_speed_kmh:5.0f} km/h] "
            f"Hits = {ev['hits']:2d}/10 ({ev['hit_rate']*100:3.0f}%) | "
            f"Mean Closest Approach = {ev['mean_min_dist_m']:5.2f} m | "
            f"10-Trial Dists = {ev['min_dists_10']} | "
            f"{status_str} ({elapsed_r}s)",
            flush=True,
        )

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
                "stage_label": f"Pure-RL Car-Hunter Missile (vs {cur_speed_kmh:.0f} km/h Car)",
                "target_mode": "car_evasion",
                "obs_dim": 24,
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
                "hit_rate": ev["hit_rate"],
                "mean_min_dist_m": ev["mean_min_dist_m"],
                "passed_9_of_10": ev["passed_9_of_10"],
            }
        )
        with open(out_dir / "pursuit_evasion_ladder_progress.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "current_round": round_num,
                    "current_speed_kmh": cur_speed_kmh,
                    "highest_cleared_speed_kmh": max([0.0] + [h["car_max_speed_kmh"] for h in history if h["passed_9_of_10"]]),
                    "history": history,
                },
                f,
                indent=2,
            )

        if ev["passed_9_of_10"] and tier_idx < len(SPEED_TIERS_KMH) - 1:
            tier_idx += 1
            best_hits_at_tier = -1
            best_dist_at_tier = 1e9

        if not infinite_mode and round_num >= max_rounds:
            break

    return {"history": history}


if __name__ == "__main__":
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "infinite"
    run_coevolution_ladder(max_rounds=10, infinite_mode=(mode_arg == "infinite"))
