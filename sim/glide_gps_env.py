from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim.aero_glide_sim import (
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


def compute_inav_circular_attitude(R_wb: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Exact INAV imuUpdateEulerAngles (src/main/flight/imu.c) circularly normalized with (sin, cos)
    so there is zero discontinuity at +-180 deg roll or 0/360 deg yaw:
      - roll  = atan2(R_wb[2, 1], R_wb[2, 2])
      - pitch = (pi / 2) - acos(-R_wb[2, 0]) = asin(-R_wb[2, 0])
      - yaw   = -atan2(R_wb[1, 0], R_wb[0, 0])
    Returns: (sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw)
    """
    roll_rad = float(np.arctan2(R_wb[2, 1], R_wb[2, 2]))
    sin_pitch = float(np.clip(-R_wb[2, 0], -1.0, 1.0))
    cos_pitch = float(np.sqrt(max(0.0, 1.0 - sin_pitch * sin_pitch)))
    yaw_rad = float(-np.arctan2(R_wb[1, 0], R_wb[0, 0]))
    return (
        float(np.sin(roll_rad)),
        float(np.cos(roll_rad)),
        sin_pitch,
        cos_pitch,
        float(np.sin(yaw_rad)),
        float(np.cos(yaw_rad)),
    )


class GlideGpsTargetEnv(gym.Env):
    """
    300 g Delta-Wing Airframe Guided Homing Environment with 24D INAV + Relative Target Observation:
      - Boundary circle around origin (0, 0): Diameter = 2 * (altitude - 70 m), i.e. Radius = altitude - 70 m.
      - Strict target hit threshold: hit_radius_m = 3.0 m (error <= 3.0 m = Hit (1), > 3.0 m = Miss (0)).
      - Observation Space (24D — Zero Clipping, Circularly Normalized INAV Attitude + Relative Coordinates):
          0..2  : Relative target coordinates (dx, dy, dz) = (target_ned - pos_ned) / 1000.0 (unclipped)
          3..4  : Body airspeed components (u / 50.0, w / 50.0)
          5..7  : Body gyroscope angular rates (p / 10.0, q / 10.0, r / 10.0)
          8..13 : Circularly normalized INAV attitude (sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw)
          14..15: Instantaneous 1-step target movement (target(t) - target(t - 0.05s)) / 0.05s / 25.0 in body frame
          16..23: Previous target displacements (target(t) - target(t - dt)) at dt = 0.25s, 0.50s, 0.75s, 1.00s in body frame
    """

    metadata = {"render_modes": ["dict"], "render_fps": 20}

    def __init__(
        self,
        target_mode: str = "static",  # "static", "linear", "curved", or "aggressive"
        release_altitude_m: float = 1000.0,
        control_dt: float = 0.05,
        max_wing_flap_deg: float = MAX_WING_FLAP_DEG,
        randomize_drop: bool = True,
        hit_radius_m: float = 3.0,
    ) -> None:
        super().__init__()
        assert target_mode in ("static", "linear", "curved", "aggressive")
        self.target_mode = target_mode
        self.release_altitude_m = float(release_altitude_m)
        self.control_dt = float(control_dt)
        self.max_wing_flap_deg = float(max_wing_flap_deg)
        self.randomize_drop = bool(randomize_drop)
        self.hit_radius_m = float(hit_radius_m)

        self.cfg = AirframeAeroConfig(max_wing_flap_deg=self.max_wing_flap_deg)
        self.geom = LinkageGeometry()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.R_wb = np.eye(3, dtype=np.float64)
        self.angles = np.zeros(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self.pos_ned = np.array([0.0, 0.0, -1000.0], dtype=np.float64)
        self.vel_ned = np.array([0.0, 0.0, 0.1], dtype=np.float64)
        self.delta_r_deg = 0.0
        self.delta_l_deg = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_omega = np.zeros(3, dtype=np.float64)
        self.prev_domega = np.zeros(3, dtype=np.float64)

        self.current_release_altitude_m = self.release_altitude_m
        self.boundary_radius_m = max(50.0, self.release_altitude_m - 70.0)

        self.target_ned = np.array([680.0, 240.0, 0.0], dtype=np.float64)
        self.target_ne = self.target_ned[:2]
        self.target_vel_ne = np.zeros(2, dtype=np.float64)
        self.target_speed_ms = 0.0
        self.target_hdg_rad = 0.0
        self.target_turn_rate_rads = 0.0
        self.target_turn_freq = 0.0
        self.target_history_ned: List[np.ndarray] = []

        self.step_count = 0
        self.max_steps = 1600
        self.prev_dist_3d = 0.0
        self.initial_dist_3d = 1000.0
        self.total_path_length_m = 0.0
        self.total_distance_closed_m = 0.0
        self.total_oscillation_penalty = 0.0

    def _compute_dist_3d(self) -> float:
        return float(np.linalg.norm(self.target_ned - self.pos_ned))

    def _get_obs(self) -> np.ndarray:
        p, q, r = self.omega

        # Relative target coordinates (target - pos) projected into body frame:
        # Both unit direction (dir_body) and unclipped distance (dist_3d / 1000.0)
        d_ned = self.target_ned - self.pos_ned
        dist_3d = max(1.0, float(np.linalg.norm(d_ned)))
        dir_body = (self.R_wb.T @ d_ned) / dist_3d

        # Body-frame airspeed components (u, w)
        v_body = self.R_wb.T @ self.vel_ned

        # Circularly normalized INAV attitude (sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw)
        s_roll, c_roll, s_pitch, c_pitch, s_yaw, c_yaw = compute_inav_circular_attitude(self.R_wb)

        # Past Target Positions (5 and 10 steps ago = 0.25s and 0.50s ago) in all 3 body axes (x_b, y_b, z_b)
        n_hist = len(self.target_history_ned)
        past_1 = self.target_history_ned[max(0, n_hist - 2)] if n_hist > 1 else self.target_ned
        past_5 = self.target_history_ned[max(0, n_hist - 6)] if n_hist > 5 else past_1
        past_10 = self.target_history_ned[max(0, n_hist - 11)] if n_hist > 10 else past_5

        step_delta_body = self.R_wb.T @ ((self.target_ned - past_1) / self.control_dt)
        d025_body = self.R_wb.T @ (self.target_ned - past_5)
        d050_body = self.R_wb.T @ (self.target_ned - past_10)

        return np.array(
            [
                float(dir_body[0]),
                float(dir_body[1]),
                float(dir_body[2]),
                dist_3d / 1000.0,
                float(v_body[0]) / 50.0,
                float(v_body[2]) / 50.0,
                float(p) / 10.0,
                float(q) / 10.0,
                float(r) / 10.0,
                s_roll,
                c_roll,
                s_pitch,
                c_pitch,
                s_yaw,
                c_yaw,
                float(step_delta_body[0]) / 25.0,
                float(step_delta_body[1]) / 25.0,
                float(step_delta_body[2]) / 25.0,
                float(d025_body[0]) / 6.25,
                float(d025_body[1]) / 6.25,
                float(d025_body[2]) / 6.25,
                float(d050_body[0]) / 12.5,
                float(d050_body[1]) / 12.5,
                float(d050_body[2]) / 12.5,
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
        deterministic = bool(options.get("deterministic", False))

        if "angles_deg" in options or "speed_ms" in options:
            roll0, pitch0, yaw0 = options.get("angles_deg", [0.0, -89.5, 0.0])
            spd0 = float(options.get("speed_ms", 0.1))
            self.omega = np.zeros(3, dtype=np.float64)
            self.angles = np.deg2rad([roll0, pitch0, yaw0])
            self.R_wb = euler_to_rotation_matrix(self.angles[0], self.angles[1], self.angles[2])
            self.pos_ned = np.array([0.0, 0.0, -alt], dtype=np.float64)
            self.vel_ned = self.R_wb @ np.array([spd0, 0.0, 0.0], dtype=np.float64)
        elif self.randomize_drop and not deterministic:
            if self.np_random.uniform(0.0, 1.0) < 0.50:
                alt = float(self.np_random.uniform(450.0, 1000.0))
                roll0 = float(self.np_random.uniform(-180.0, 180.0))
                pitch0 = float(self.np_random.uniform(-88.0, 15.0))
                yaw0 = float(self.np_random.uniform(-180.0, 180.0))
                spd0 = float(self.np_random.uniform(12.0, 52.0))
                self.omega = self.np_random.uniform(-0.3, 0.3, size=3).astype(np.float64)
                self.angles = np.deg2rad([roll0, pitch0, yaw0])
                self.R_wb = euler_to_rotation_matrix(self.angles[0], self.angles[1], self.angles[2])
                self.pos_ned = np.array([0.0, 0.0, -alt], dtype=np.float64)
                self.vel_ned = self.R_wb @ np.array([spd0, 0.0, 0.0], dtype=np.float64)
            else:
                roll0 = float(self.np_random.uniform(-180.0, 180.0))
                pitch0 = float(self.np_random.uniform(-89.5, -84.0))
                yaw0 = float(self.np_random.uniform(-180.0, 180.0))
                self.omega = self.np_random.uniform(-0.2, 0.2, size=3).astype(np.float64)
                self.angles = np.deg2rad([roll0, pitch0, yaw0])
                self.R_wb = euler_to_rotation_matrix(self.angles[0], self.angles[1], self.angles[2])
                self.pos_ned = np.array([0.0, 0.0, -alt], dtype=np.float64)
                self.vel_ned = np.array([0.0, 0.0, 0.1], dtype=np.float64)
        else:
            roll0, pitch0, yaw0 = 0.0, -89.5, 0.0
            self.omega = np.zeros(3, dtype=np.float64)
            self.angles = np.deg2rad([roll0, pitch0, yaw0])
            self.R_wb = euler_to_rotation_matrix(self.angles[0], self.angles[1], self.angles[2])
            self.pos_ned = np.array([0.0, 0.0, -alt], dtype=np.float64)
            self.vel_ned = np.array([0.0, 0.0, 0.1], dtype=np.float64)

        # Boundary circle diameter = 2 * (alt - 70) -> radius = alt - 70
        self.current_release_altitude_m = alt
        self.boundary_radius_m = max(50.0, alt - 70.0)

        self.delta_r_deg = 0.0
        self.delta_l_deg = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_omega = self.omega.copy()
        self.prev_domega = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.total_distance_closed_m = 0.0
        self.total_oscillation_penalty = 0.0
        self.target_turn_rate_rads = 0.0
        self.target_turn_freq = 0.0

        if "target_ne" in options:
            t_ne = np.array(options["target_ne"], dtype=np.float64)
            self.target_ned = np.array([t_ne[0], t_ne[1], 0.0], dtype=np.float64)
            self.target_ne = self.target_ned[:2]
            self.target_vel_ne = np.array(options.get("target_vel_ne", [0.0, 0.0]), dtype=np.float64)
        elif deterministic:
            if self.target_mode == "static":
                self.target_ned = np.array([680.0, 240.0, 0.0], dtype=np.float64)
                self.target_vel_ne = np.array([0.0, 0.0], dtype=np.float64)
            elif self.target_mode == "linear":
                self.target_ned = np.array([520.0, -220.0, 0.0], dtype=np.float64)
                self.target_speed_ms = 11.0
                self.target_hdg_rad = np.arctan2(9.5, 5.5)
                self.target_vel_ne = np.array(
                    [self.target_speed_ms * np.cos(self.target_hdg_rad), self.target_speed_ms * np.sin(self.target_hdg_rad)],
                    dtype=np.float64,
                )
            else:
                self.target_ned = np.array([500.0, -240.0, 0.0], dtype=np.float64)
                self.target_speed_ms = 14.0
                self.target_hdg_rad = 1.25
                self.target_turn_rate_rads = 0.095
                self.target_turn_freq = 0.14
                self.target_vel_ne = np.array(
                    [self.target_speed_ms * np.cos(self.target_hdg_rad), self.target_speed_ms * np.sin(self.target_hdg_rad)],
                    dtype=np.float64,
                )
            self.target_ne = self.target_ned[:2]
        else:
            if self.target_mode == "static":
                r_init = float(self.np_random.uniform(50.0, self.boundary_radius_m))
                az_init = float(self.np_random.uniform(-np.pi, np.pi))
                self.target_ned = np.array([r_init * np.cos(az_init), r_init * np.sin(az_init), 0.0], dtype=np.float64)
                self.target_vel_ne = np.zeros(2, dtype=np.float64)
            elif self.target_mode == "linear":
                r_init = float(self.np_random.uniform(120.0, min(self.boundary_radius_m * 0.75, 680.0)))
                az_init = float(self.np_random.uniform(-np.pi, np.pi))
                self.target_ned = np.array([r_init * np.cos(az_init), r_init * np.sin(az_init), 0.0], dtype=np.float64)
                self.target_speed_ms = float(self.np_random.uniform(4.0, 20.0))
                tangent_dir = az_init + float(self.np_random.choice([-1.0, 1.0])) * float(self.np_random.uniform(1.3, 2.2))
                self.target_hdg_rad = tangent_dir
                self.target_vel_ne = np.array(
                    [self.target_speed_ms * np.cos(self.target_hdg_rad), self.target_speed_ms * np.sin(self.target_hdg_rad)],
                    dtype=np.float64,
                )
            else:
                r_init = float(self.np_random.uniform(120.0, min(self.boundary_radius_m * 0.70, 650.0)))
                az_init = float(self.np_random.uniform(-np.pi, np.pi))
                self.target_ned = np.array([r_init * np.cos(az_init), r_init * np.sin(az_init), 0.0], dtype=np.float64)
                self.target_speed_ms = float(self.np_random.uniform(8.0, 20.0))
                tangent_dir = az_init + float(self.np_random.choice([-1.0, 1.0])) * float(self.np_random.uniform(1.3, 2.2))
                self.target_hdg_rad = tangent_dir
                self.target_turn_rate_rads = float(self.np_random.uniform(-0.14, 0.14))
                self.target_turn_freq = float(self.np_random.uniform(0.10, 0.22))
                self.target_vel_ne = np.array(
                    [self.target_speed_ms * np.cos(self.target_hdg_rad), self.target_speed_ms * np.sin(self.target_hdg_rad)],
                    dtype=np.float64,
                )
            self.target_ne = self.target_ned[:2]

        self.target_history_ned = []
        for h_i in range(20, -1, -1):
            dt_back = h_i * self.control_dt
            past_pt = np.array(
                [
                    self.target_ned[0] - self.target_vel_ne[0] * dt_back,
                    self.target_ned[1] - self.target_vel_ne[1] * dt_back,
                    self.target_ned[2],
                ],
                dtype=np.float64,
            )
            self.target_history_ned.append(past_pt)

        self.prev_dist_3d = self._compute_dist_3d()
        self.initial_dist_3d = max(1.0, self.prev_dist_3d)
        self.total_path_length_m = 0.0

        return self._get_obs(), self.get_info()

    def _step_physics_substep(self, cmd_r_deg: float, cmd_l_deg: float, sub_dt: float) -> None:
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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.step_count += 1
        t_sec = self.step_count * self.control_dt

        if self.target_mode in ("curved", "aggressive"):
            inst_turn = self.target_turn_rate_rads * np.cos(self.target_turn_freq * t_sec)
            self.target_hdg_rad += inst_turn * self.control_dt
            self.target_vel_ne[0] = self.target_speed_ms * np.cos(self.target_hdg_rad)
            self.target_vel_ne[1] = self.target_speed_ms * np.sin(self.target_hdg_rad)

        if self.target_mode in ("linear", "curved", "aggressive"):
            self.target_ned[0] += self.target_vel_ne[0] * self.control_dt
            self.target_ned[1] += self.target_vel_ne[1] * self.control_dt
            r_t = float(np.hypot(self.target_ned[0], self.target_ned[1]))
            if r_t > self.boundary_radius_m:
                self.target_ned[:2] *= self.boundary_radius_m / r_t
                self.target_vel_ne *= -1.0
                self.target_hdg_rad = float(np.arctan2(self.target_vel_ne[1], self.target_vel_ne[0]))
            self.target_ne = self.target_ned[:2]
            self.prev_dist_3d = self._compute_dist_3d()

        self.target_history_ned.append(self.target_ned.copy())
        if len(self.target_history_ned) > 25:
            self.target_history_ned.pop(0)

        pitch_flap_deg = float(act[0]) * self.max_wing_flap_deg
        roll_flap_deg = float(act[1]) * 35.0
        cmd_r_deg = float(np.clip(pitch_flap_deg + roll_flap_deg, -self.max_wing_flap_deg, +self.max_wing_flap_deg))
        cmd_l_deg = float(np.clip(pitch_flap_deg - roll_flap_deg, -self.max_wing_flap_deg, +self.max_wing_flap_deg))

        n_sub = 10
        sub_dt = self.control_dt / n_sub
        prev_pos_step = self.pos_ned.copy()
        for _ in range(n_sub):
            if -self.pos_ned[2] <= 0.0:
                self.pos_ned[2] = 0.0
                break
            self._step_physics_substep(cmd_r_deg, cmd_l_deg, sub_dt)
        step_path_len = float(np.linalg.norm(self.pos_ned - prev_pos_step))
        self.total_path_length_m += step_path_len

        cur_dist_3d = self._compute_dist_3d()
        cur_dist_2d = float(
            np.hypot(self.target_ned[0] - self.pos_ned[0], self.target_ned[1] - self.pos_ned[1])
        )
        dist_dec = self.prev_dist_3d - cur_dist_3d
        self.prev_dist_3d = cur_dist_3d
        self.total_distance_closed_m += dist_dec

        # Anti-Oscillation / Vibration Penalty (penalizes rapid control sign reversals / jerk, NOT smooth turns!):
        da_cur = act - self.last_action
        da_prev = self.last_action - self.prev_action
        sign_flip_mask = (da_cur * da_prev < -1e-4).astype(np.float32)
        action_oscillation = float(np.sum(sign_flip_mask * ((da_cur - da_prev) ** 2)))

        domega = self.omega - self.prev_omega
        omega_flip_mask = (domega * self.prev_domega < -1e-3).astype(np.float64)
        rate_oscillation = float(np.sum(omega_flip_mask * ((domega - self.prev_domega) ** 2))) * 0.02

        osc_penalty = 2.5 * action_oscillation + rate_oscillation
        self.total_oscillation_penalty += osc_penalty

        self.prev_action = self.last_action.copy()
        self.last_action = act.copy()
        self.prev_domega = domega.copy()
        self.prev_omega = self.omega.copy()

        # Step reward: distance closing minus oscillation penalty (with light detour penalty)
        wasted_detour = max(0.0, step_path_len - max(0.0, dist_dec))
        reward = 6.0 * dist_dec - 0.35 * wasted_detour - osc_penalty

        hit_ground = bool(-self.pos_ned[2] <= 0.0)
        if hit_ground:
            # End-of-flight evaluation: overwhelming weight on <= 3.0 m target hit accuracy!
            path_eff = min(1.0, self.initial_dist_3d / max(1.0, self.total_path_length_m))
            hit_bonus = 6000.0 * float(np.exp(-((cur_dist_2d / self.hit_radius_m) ** 2)))
            near_bonus = max(0.0, 1500.0 - 15.0 * cur_dist_2d)
            reward += (hit_bonus + near_bonus) * (0.85 + 0.15 * path_eff) - 0.5 * self.total_oscillation_penalty

        terminated = hit_ground
        truncated = bool(self.step_count >= self.max_steps)

        return self._get_obs(), float(reward), terminated, truncated, self.get_info()

    def get_info(self) -> Dict[str, Any]:
        roll_deg, pitch_deg, yaw_deg = np.rad2deg(self.angles)
        xy_dist = float(np.hypot(self.pos_ned[0], self.pos_ned[1]))
        target_err = float(
            np.hypot(self.target_ned[0] - self.pos_ned[0], self.target_ned[1] - self.pos_ned[1])
        )
        alt_m = float(max(0.0, -self.pos_ned[2]))
        airspeed = float(np.linalg.norm(self.vel_ned))
        path_eff = min(1.0, self.initial_dist_3d / max(1.0, self.total_path_length_m))
        return {
            "step": int(self.step_count),
            "time_s": round(self.step_count * self.control_dt, 2),
            "altitude_m": round(alt_m, 2),
            "xy_distance_m": round(xy_dist, 2),
            "pos_x_m": round(float(self.pos_ned[0]), 2),
            "pos_y_m": round(float(self.pos_ned[1]), 2),
            "target_x_m": round(float(self.target_ned[0]), 2),
            "target_y_m": round(float(self.target_ned[1]), 2),
            "target_alt_m": round(float(-self.target_ned[2]), 2),
            "target_vx_ms": round(float(self.target_vel_ne[0]), 2),
            "target_vy_ms": round(float(self.target_vel_ne[1]), 2),
            "target_error_m": round(target_err, 2),
            "hit": bool(alt_m <= 0.1 and target_err <= self.hit_radius_m),
            "boundary_radius_m": round(float(self.boundary_radius_m), 1),
            "boundary_diameter_m": round(float(2.0 * self.boundary_radius_m), 1),
            "shortest_path_efficiency": round(float(path_eff), 4),
            "total_oscillation_penalty": round(float(self.total_oscillation_penalty), 3),
            "total_distance_closed_m": round(float(self.total_distance_closed_m), 2),
            "airspeed_ms": round(airspeed, 2),
            "roll_deg": round(float(roll_deg), 2),
            "pitch_deg": round(float(pitch_deg), 2),
            "yaw_deg": round(float(yaw_deg), 2),
            "delta_r_deg": round(float(self.delta_r_deg), 2),
            "delta_l_deg": round(float(self.delta_l_deg), 2),
            "servo_r_deg": round(wing_flap_deg_to_servo_deg(self.delta_r_deg, self.geom), 2),
            "servo_l_deg": round(wing_flap_deg_to_servo_deg(self.delta_l_deg, self.geom), 2),
        }
