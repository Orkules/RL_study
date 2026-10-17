from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim.aero_glide_sim import (
    SIM_MODE_FREE_DROP,
    AeroGlideSimulator,
    AirframeAeroConfig,
    euler_to_rotation_matrix,
    integrate_rotation_so3,
    rotation_matrix_to_euler,
)
from sim.wing_linkage import (
    LinkageGeometry,
    MAX_WING_FLAP_DEG,
    wing_flap_deg_to_servo_deg,
)


class GlidePulloutEnv(gym.Env):
    """
    Gymnasium RL Environment for the 300 g Delta-Wing Airframe Free-Drop Pull-Out & Max-Range Glide.

    Task (No Camera — Basic Aerodynamic Flight Control):
      - Dropped at altitude (default 500 m) with the nose pointing straight down (-90 deg pitch, 0 m/s).
      - Rewards:
        1. Horizontal X-Y distance progress: reach as far as possible across the X-Y ground surface.
        2. Pitch pull-up / leveling: starting from -90 deg vertical dive, reward pulling the nose up
           toward horizontal airplane flight (and temporary positive pitch climb / roll-up pull-out).
        3. Roll & angular rate stability: keep wings level (roll ~ 0 deg) and avoid tumbling or stalling.

    Observation Space (10D IMU + Flight State, normalized):
      0: pitch_norm   - Pitch angle theta / (pi / 2) in [-1, +1] (-1 = straight down -90 deg, 0 = level 0 deg)
      1: sin_roll     - sin(roll)
      2: cos_roll     - cos(roll)
      3: gyro_p       - Roll angular rate p / 10.0 (rad/s)
      4: gyro_q       - Pitch angular rate q / 10.0 (rad/s)
      5: gyro_r       - Yaw angular rate r / 10.0 (rad/s)
      6: airspeed_n   - Airspeed V / 50.0 (m/s)
      7: vz_norm      - Vertical descent rate vd / 50.0 (m/s)
      8: prev_act_r   - Previous Right Flap command in [-1, +1]
      9: prev_act_l   - Previous Left Flap command in [-1, +1]

    Action Space (2D Continuous in [-1, +1] — Remote Controller Stick):
      0: pitch_cmd    - Symmetric Pitch (Elevator) command in [-1, +1] -> [-MAX_WING_FLAP_DEG, +MAX_WING_FLAP_DEG]
      1: roll_cmd     - Differential Roll (Aileron) command in [-1, +1] -> [-20 deg, +20 deg]
    """

    metadata = {"render_modes": ["dict"], "render_fps": 20}

    def __init__(
        self,
        release_altitude_m: float = 1000.0,
        control_dt: float = 0.05,          # 20 Hz RL policy rate (5x 10ms physics steps per RL step)
        max_wing_flap_deg: float = MAX_WING_FLAP_DEG,
        randomize_drop: bool = True,
    ) -> None:
        super().__init__()
        self.release_altitude_m = float(release_altitude_m)
        self.control_dt = float(control_dt)
        self.max_wing_flap_deg = float(max_wing_flap_deg)
        self.randomize_drop = bool(randomize_drop)

        self.cfg = AirframeAeroConfig(max_wing_flap_deg=self.max_wing_flap_deg)
        self.geom = LinkageGeometry()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Fast internal 6-DOF state (identical math to AeroGlideSimulator without per-step MJData overhead during training)
        self.R_wb = np.eye(3, dtype=np.float64)
        self.angles = np.zeros(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self.pos_ned = np.zeros(3, dtype=np.float64)
        self.vel_ned = np.zeros(3, dtype=np.float64)
        self.delta_r_deg = 0.0
        self.delta_l_deg = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 1600  # up to 80 seconds so a 1,000 m max-range glide (~53 s) always reaches ground
        self.prev_xy_dist = 0.0
        self.prev_alt_m = self.release_altitude_m

    def _get_obs(self) -> np.ndarray:
        roll, pitch, _ = self.angles
        p, q, r = self.omega
        airspeed = float(np.linalg.norm(self.vel_ned))
        vd = float(self.vel_ned[2])
        return np.array(
            [
                pitch / (np.pi / 2.0),
                np.sin(roll),
                np.cos(roll),
                p / 10.0,
                q / 10.0,
                r / 10.0,
                airspeed / 50.0,
                vd / 50.0,
                float(self.last_action[0]),
                float(self.last_action[1]),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        alt = float(options.get("altitude_m", self.release_altitude_m))
        if self.randomize_drop and not options.get("deterministic", False):
            # Slight initial attitude/rate perturbation around -90 deg nose-down drop
            roll0 = float(self.np_random.uniform(-8.0, 8.0))
            pitch0 = float(self.np_random.uniform(-89.5, -84.0))
            yaw0 = float(self.np_random.uniform(-5.0, 5.0))
            self.omega = self.np_random.uniform(-0.3, 0.3, size=3).astype(np.float64)
        else:
            roll0, pitch0, yaw0 = 0.0, -89.5, 0.0
            self.omega = np.zeros(3, dtype=np.float64)

        self.angles = np.deg2rad([roll0, pitch0, yaw0])
        self.R_wb = euler_to_rotation_matrix(self.angles[0], self.angles[1], self.angles[2])
        self.pos_ned = np.array([0.0, 0.0, -alt], dtype=np.float64)
        self.vel_ned = np.array([0.0, 0.0, 0.1], dtype=np.float64)

        self.delta_r_deg = 0.0
        self.delta_l_deg = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.prev_xy_dist = 0.0
        self.prev_alt_m = alt

        return self._get_obs(), self.get_info()

    def _step_physics_substep(self, cmd_r_deg: float, cmd_l_deg: float, sub_dt: float) -> Tuple[float, float]:
        alpha_s = sub_dt / (self.cfg.servo_tau_s + sub_dt)
        self.delta_r_deg += alpha_s * (cmd_r_deg - self.delta_r_deg)
        self.delta_l_deg += alpha_s * (cmd_l_deg - self.delta_l_deg)

        alt_m = max(0.0, float(-self.pos_ned[2]))
        rho = float(self.cfg.rho_sea_level * np.exp(-alt_m / 9500.0))

        v_rel_body = self.R_wb.T @ self.vel_ned
        u, v, w = float(v_rel_body[0]), float(v_rel_body[1]), float(v_rel_body[2])
        airspeed = float(np.linalg.norm(v_rel_body))
        q_dyn = 0.5 * rho * (airspeed ** 2)

        if airspeed > 1e-4:
            alpha = float(np.arctan2(w, u))
            beta = float(np.arcsin(np.clip(v / airspeed, -1.0, 1.0)))
        else:
            alpha = 0.0
            beta = 0.0

        sa, ca = np.sin(alpha), np.cos(alpha)
        sb, cb = np.sin(beta), np.cos(beta)

        cl_wing = self.cfg.cl_alpha_wing * sa * ca + self.cfg.cl_vortex_wing * (sa ** 2) * ca * np.sign(sa)
        cd_wing = self.cfg.cd0_body_wing + 1.35 * (sa ** 2) + 0.35 * (sb ** 2)
        lift_wing = q_dyn * self.cfg.wing_area_m2 * cl_wing
        lift_fuse = q_dyn * self.cfg.wing_area_m2 * (0.95 * sa * ca)
        drag_wing = q_dyn * self.cfg.wing_area_m2 * cd_wing

        fx_wing = -drag_wing * ca + lift_wing * sa
        fz_wing = -drag_wing * sa - lift_wing * ca
        fx_fuse = lift_fuse * sa
        fz_fuse = -lift_fuse * ca

        tau_wing_y = 0.0633 * fz_wing
        fy_fin = q_dyn * self.cfg.fin_area_m2 * (self.cfg.cy_beta_fin * sb * cb)
        tau_fin_z = -0.0633 * fy_fin

        p, q, r = float(self.omega[0]), float(self.omega[1]), float(self.omega[2])

        # Right & Left Flaps
        r_x = -0.0690
        S_f = self.cfg.flap_area_each_m2

        def eval_flap(y_sign: float, d_deg: float) -> Tuple[float, float, float, float, float]:
            r_y = y_sign * 0.0711
            wz_loc = w - q * r_x + p * r_y
            ux_loc = u - r * r_y
            v_loc = float(np.sqrt(ux_loc * ux_loc + v * v + wz_loc * wz_loc))
            q_loc = 0.5 * rho * (v_loc ** 2)
            a_loc = float(np.arctan2(wz_loc, ux_loc)) if v_loc > 1e-4 else 0.0
            d_rad = np.deg2rad(d_deg)
            eff = d_rad - 0.35 * a_loc
            sd, cd = np.sin(d_rad), np.cos(d_rad)
            se, ce = np.sin(eff), np.cos(eff)
            cn = self.cfg.cl_delta_flap * sd * cd + 1.15 * se * ce
            cd_f = self.cfg.cd_delta_flap * (sd ** 2) + 0.25 * (se ** 2)
            fz = q_loc * S_f * cn
            fx = -q_loc * S_f * cd_f
            tx = r_y * fz
            ty = -r_x * fz + q_loc * S_f * self.cfg.mean_aero_chord_m * self.cfg.cm_delta_flap * sd * cd
            tz = -r_y * fx
            return fx, fz, tx, ty, tz

        fx_r, fz_r, tx_r, ty_r, tz_r = eval_flap(+1.0, self.delta_r_deg)
        fx_l, fz_l, tx_l, ty_l, tz_l = eval_flap(-1.0, self.delta_l_deg)

        b, c = self.cfg.wingspan_m, self.cfg.mean_aero_chord_m
        damp_scale = 0.5 * rho * airspeed * self.cfg.wing_area_m2
        tau_damp_x = -(damp_scale * (b ** 2) * self.cfg.cl_p_damp + self.cfg.viscous_rot_damp) * p
        tau_damp_y = -(damp_scale * (c ** 2) * self.cfg.cm_q_damp * 4.5 + self.cfg.viscous_rot_damp) * q
        tau_damp_z = -(damp_scale * (b ** 2) * self.cfg.cn_r_damp + self.cfg.viscous_rot_damp) * r

        f_body = np.array([fx_wing + fx_fuse + fx_r + fx_l, fy_fin, fz_wing + fz_fuse + fz_r + fz_l], dtype=np.float64)
        tau_body = np.array(
            [
                tx_r + tx_l + tau_damp_x,
                tau_wing_y + ty_r + ty_l + tau_damp_y,
                tau_fin_z + tz_r + tz_l + tau_damp_z,
            ],
            dtype=np.float64,
        )

        ixx, iyy, izz = self.cfg.ixx, self.cfg.iyy, self.cfg.izz
        dwx = (tau_body[0] - (izz - iyy) * q * r) / ixx
        dwy = (tau_body[1] - (ixx - izz) * p * r) / iyy
        dwz = (tau_body[2] - (iyy - ixx) * p * q) / izz

        self.omega[0] = np.clip(self.omega[0] + dwx * sub_dt, -35.0, 35.0)
        self.omega[1] = np.clip(self.omega[1] + dwy * sub_dt, -35.0, 35.0)
        self.omega[2] = np.clip(self.omega[2] + dwz * sub_dt, -35.0, 35.0)

        self.R_wb = integrate_rotation_so3(self.R_wb, self.omega, sub_dt)
        self.angles = rotation_matrix_to_euler(self.R_wb)

        f_world = self.R_wb @ f_body
        accel_ned = f_world / self.cfg.mass_kg + np.array([0.0, 0.0, self.cfg.gravity], dtype=np.float64)
        self.vel_ned += accel_ned * sub_dt
        self.pos_ned += self.vel_ned * sub_dt

        return alpha, airspeed

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.step_count += 1

        # Action 0 = Symmetric Pitch command in [-50, +50] deg; Action 1 = Differential Roll command in [-20, +20] deg
        pitch_flap_deg = float(act[0]) * self.max_wing_flap_deg
        roll_flap_deg = float(act[1]) * 20.0
        cmd_r_deg = float(np.clip(pitch_flap_deg + roll_flap_deg, -self.max_wing_flap_deg, +self.max_wing_flap_deg))
        cmd_l_deg = float(np.clip(pitch_flap_deg - roll_flap_deg, -self.max_wing_flap_deg, +self.max_wing_flap_deg))

        # Integrate 10 substeps of 5 ms (total 50 ms per RL decision step)
        n_sub = 10
        sub_dt = self.control_dt / n_sub
        alpha = 0.0
        for _ in range(n_sub):
            if -self.pos_ned[2] <= 0.0:
                self.pos_ned[2] = 0.0
                break
            alpha, _ = self._step_physics_substep(cmd_r_deg, cmd_l_deg, sub_dt)

        cur_xy_dist = float(np.hypot(self.pos_ned[0], self.pos_ned[1]))
        delta_xy = cur_xy_dist - self.prev_xy_dist
        self.prev_xy_dist = cur_xy_dist

        # --- Reward Function (100% Unshaped Pure Horizontal X-Y Distance ONLY) ---
        # Zero attitude penalties, zero roll/pitch/rate/slew shaping, zero mid-air termination.
        # The network is rewarded strictly for how far it travels across the X-Y ground plane.
        reward = 5.0 * delta_xy

        hit_ground = bool(-self.pos_ned[2] <= 0.0)
        terminated = hit_ground
        if hit_ground:
            reward += 1.0 * cur_xy_dist

        truncated = bool(self.step_count >= self.max_steps)
        self.last_action = act.copy()

        return self._get_obs(), float(reward), terminated, truncated, self.get_info()

    def get_info(self) -> Dict[str, Any]:
        roll_deg, pitch_deg, yaw_deg = np.rad2deg(self.angles)
        xy_dist = float(np.hypot(self.pos_ned[0], self.pos_ned[1]))
        alt_m = float(max(0.0, -self.pos_ned[2]))
        airspeed = float(np.linalg.norm(self.vel_ned))
        return {
            "step": int(self.step_count),
            "time_s": round(self.step_count * self.control_dt, 2),
            "altitude_m": round(alt_m, 2),
            "xy_distance_m": round(xy_dist, 2),
            "pos_x_m": round(float(self.pos_ned[0]), 2),
            "pos_y_m": round(float(self.pos_ned[1]), 2),
            "airspeed_ms": round(airspeed, 2),
            "roll_deg": round(float(roll_deg), 2),
            "pitch_deg": round(float(pitch_deg), 2),
            "yaw_deg": round(float(yaw_deg), 2),
            "delta_r_deg": round(float(self.delta_r_deg), 2),
            "delta_l_deg": round(float(self.delta_l_deg), 2),
            "servo_r_deg": round(wing_flap_deg_to_servo_deg(self.delta_r_deg, self.geom), 2),
            "servo_l_deg": round(wing_flap_deg_to_servo_deg(self.delta_l_deg, self.geom), 2),
        }
