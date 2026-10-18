from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim.aero_glide_sim import (
    AirframeAeroConfig,
    euler_to_rotation_matrix,
    integrate_rotation_so3,
    rotation_matrix_to_euler,
)
from sim.glide_gps_env import compute_inav_circular_attitude
from sim.wing_linkage import (
    LinkageGeometry,
    MAX_WING_FLAP_DEG,
    wing_flap_deg_to_servo_deg,
)


@dataclass
class EvaderCarConfig:
    """
    Realistic Ackermann Kinematic-Dynamic Bicycle Model for the Evasive Car:
      - Wheelbase L = 2.65 m, track width W = 1.56 m
      - Max front-wheel steering angle delta_max = 30.0 deg (0.5236 rad) -> R_min = 4.59 m
      - Max steering rack rate = 50.0 deg/s + steering lag tau_steer = 0.18 s
      - Forward-only driving (v >= 0.0 m/s): braking decelerates to 0.0 m/s, never instant reverse
      - Realistic engine forward acceleration a_accel_max = 4.8 m/s^2, braking a_brake_max = 8.0 m/s^2
      - Tire cornering lateral grip limit a_lat_max = 0.92 * 9.81 = 9.03 m/s^2
      - Strict target hit capture radius = 3.0 m on the ground
    """

    wheelbase_m: float = 2.65
    track_width_m: float = 1.56
    max_steer_deg: float = 30.0
    max_steer_rate_degs: float = 50.0
    max_accel_ms2: float = 4.8
    max_brake_ms2: float = 8.0
    max_lat_g: float = 0.92
    drag_coeff: float = 0.0022
    rolling_resist_ms2: float = 0.18
    steer_tau_s: float = 0.18
    hit_radius_m: float = 3.0


class PursuitEvasionAdversarialEnv(gym.Env):
    """
    Two-Agent Competitive Pursuit-Evasion Environment:
      - Boundary Circle around origin (0, 0): Diameter = 2 * (altitude - 70 m), i.e. Radius = altitude - 70 m.
      - Strict Hit Threshold: hit_radius_m = 3.0 m (anything > 3.0 m is a Miss (0)).
      - Pursuer (Missile / Glider): 24D Observation (Relative Target Coordinates + Circular INAV Attitude + Gyro + Past Target Displacements, unclipped).
      - Evader (Ackermann Car on Ground): 16D Observation in Car Frame (knows Missile Position, Velocity, and Attitude).
    """

    metadata = {"render_modes": ["dict"], "render_fps": 20}

    def __init__(
        self,
        car_max_speed_kmh: float = 50.0,
        release_altitude_m: float = 1000.0,
        control_dt: float = 0.05,
        randomize_drop: bool = True,
    ) -> None:
        super().__init__()
        self.car_max_speed_kmh = float(car_max_speed_kmh)
        self.car_max_speed_ms = self.car_max_speed_kmh / 3.6
        self.release_altitude_m = float(release_altitude_m)
        self.control_dt = float(control_dt)
        self.randomize_drop = bool(randomize_drop)

        self.aero_cfg = AirframeAeroConfig(max_wing_flap_deg=MAX_WING_FLAP_DEG)
        self.car_cfg = EvaderCarConfig()
        self.geom = LinkageGeometry()

        self.missile_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        self.car_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.R_wb = np.eye(3, dtype=np.float64)
        self.angles = np.zeros(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self.pos_ned = np.array([0.0, 0.0, -1000.0], dtype=np.float64)
        self.vel_ned = np.zeros(3, dtype=np.float64)
        self.delta_r_deg = 0.0
        self.delta_l_deg = 0.0
        self.last_m_action = np.zeros(2, dtype=np.float32)
        self.prev_m_action = np.zeros(2, dtype=np.float32)
        self.prev_omega = np.zeros(3, dtype=np.float64)
        self.prev_domega = np.zeros(3, dtype=np.float64)
        self.total_oscillation_penalty = 0.0

        self.current_release_altitude_m = self.release_altitude_m
        self.boundary_radius_m = max(50.0, self.release_altitude_m - 70.0)

        self.car_pos_ne = np.array([280.0, -180.0], dtype=np.float64)
        self.car_spawn_center = self.car_pos_ne.copy()
        self.car_yaw = 0.0
        self.car_speed = 0.0
        self.car_steer = 0.0
        self.car_vel_ne = np.zeros(2, dtype=np.float64)
        self.target_history_ned: list[np.ndarray] = []
        self.visited_cells: set[Tuple[int, int]] = set()
        self.max_spawn_displacement_m = 0.0

        self.step_count = 0
        self.max_steps = 1600
        self.prev_dist_3d = 0.0
        self.min_dist_3d = 1e6

    def set_car_max_speed_kmh(self, speed_kmh: float) -> None:
        self.car_max_speed_kmh = float(speed_kmh)
        self.car_max_speed_ms = self.car_max_speed_kmh / 3.6

    def _compute_dist_3d(self) -> float:
        return float(
            np.sqrt(
                (self.car_pos_ne[0] - self.pos_ned[0]) ** 2
                + (self.car_pos_ne[1] - self.pos_ned[1]) ** 2
                + (0.0 - self.pos_ned[2]) ** 2
            )
        )

    def get_missile_obs(self) -> np.ndarray:
        p, q, r = self.omega
        cur_car_ned = np.array([float(self.car_pos_ne[0]), float(self.car_pos_ne[1]), 0.0], dtype=np.float64)
        d_ned = cur_car_ned - self.pos_ned
        dist_3d = max(1.0, float(np.linalg.norm(d_ned)))
        dir_body = (self.R_wb.T @ d_ned) / dist_3d
        v_body = self.R_wb.T @ self.vel_ned
        s_roll, c_roll, s_pitch, c_pitch, s_yaw, _ = compute_inav_circular_attitude(self.R_wb)

        hist_features: list[float] = []
        n_hist = len(self.target_history_ned)
        for k_back, norm_scale in [(5, 6.25), (10, 12.5), (15, 18.75), (20, 25.0)]:
            idx_h = max(0, n_hist - 1 - k_back)
            past_ned = self.target_history_ned[idx_h] if n_hist > 0 else cur_car_ned
            delta_ned = cur_car_ned - past_ned
            delta_body = self.R_wb.T @ delta_ned
            hist_features.append(float(delta_body[0] / norm_scale))
            hist_features.append(float(delta_body[1] / norm_scale))

        past_1 = self.target_history_ned[max(0, n_hist - 2)] if n_hist > 1 else cur_car_ned
        step_delta_body = self.R_wb.T @ ((cur_car_ned - past_1) / self.control_dt)

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
                float(step_delta_body[0]) / 25.0,
                float(step_delta_body[1]) / 25.0,
                *hist_features,
            ],
            dtype=np.float32,
        )

    def get_car_obs(self) -> np.ndarray:
        cy, sy = float(np.cos(self.car_yaw)), float(np.sin(self.car_yaw))

        d_n = float(self.pos_ned[0] - self.car_pos_ne[0])
        d_e = float(self.pos_ned[1] - self.car_pos_ne[1])
        alt_m = max(0.0, float(-self.pos_ned[2]))
        rel_x_car = cy * d_n + sy * d_e
        rel_y_car = -sy * d_n + cy * d_e

        mv_n, mv_e, mv_d = float(self.vel_ned[0]), float(self.vel_ned[1]), float(self.vel_ned[2])
        mv_x_car = cy * mv_n + sy * mv_e
        mv_y_car = -sy * mv_n + cy * mv_e
        missile_speed = float(np.linalg.norm(self.vel_ned))

        nose_n, nose_e, nose_d = float(self.R_wb[0, 0]), float(self.R_wb[1, 0]), float(self.R_wb[2, 0])
        nose_x_car = cy * nose_n + sy * nose_e
        nose_y_car = -sy * nose_n + cy * nose_e

        belly_n, belly_e = float(self.R_wb[0, 2]), float(self.R_wb[1, 2])
        belly_x_car = cy * belly_n + sy * belly_e
        belly_y_car = -sy * belly_n + cy * belly_e

        max_steer_rad = np.deg2rad(self.car_cfg.max_steer_deg)
        r_car_origin = float(np.hypot(self.car_pos_ne[0], self.car_pos_ne[1]))
        boundary_margin_norm = (self.boundary_radius_m - r_car_origin) / max(100.0, self.boundary_radius_m)

        return np.array(
            [
                rel_x_car / 1000.0,
                rel_y_car / 1000.0,
                alt_m / 1000.0,
                mv_x_car / 60.0,
                mv_y_car / 60.0,
                mv_d / 60.0,
                missile_speed / 60.0,
                nose_x_car,
                nose_y_car,
                nose_d,
                belly_x_car,
                belly_y_car,
                self.car_speed / 85.0,
                float(self.car_steer / max_steer_rad),
                self.car_max_speed_ms / 85.0,
                boundary_margin_norm,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        alt = float(options.get("altitude_m", self.release_altitude_m))
        deterministic = bool(options.get("deterministic", False))

        if self.randomize_drop and not deterministic:
            roll0 = float(self.np_random.uniform(-180.0, 180.0))
            pitch0 = float(self.np_random.uniform(-89.5, -85.0))
            yaw0 = float(self.np_random.uniform(-180.0, 180.0))
            self.omega = self.np_random.uniform(-0.15, 0.15, size=3).astype(np.float64)
        else:
            roll0, pitch0, yaw0 = 0.0, -89.5, 0.0
            self.omega = np.zeros(3, dtype=np.float64)

        self.current_release_altitude_m = alt
        # Circle boundary: Diameter = 2 * (altitude - 70 m) -> Radius = altitude - 70 m
        self.boundary_radius_m = max(50.0, alt - 70.0)

        self.angles = np.deg2rad([roll0, pitch0, yaw0])
        self.R_wb = euler_to_rotation_matrix(self.angles[0], self.angles[1], self.angles[2])
        self.pos_ned = np.array([0.0, 0.0, -alt], dtype=np.float64)
        self.vel_ned = np.array([0.0, 0.0, 0.1], dtype=np.float64)
        self.delta_r_deg = 0.0
        self.delta_l_deg = 0.0
        self.last_m_action = np.zeros(2, dtype=np.float32)
        self.prev_m_action = np.zeros(2, dtype=np.float32)
        self.prev_omega = self.omega.copy()
        self.prev_domega = np.zeros(3, dtype=np.float64)
        self.total_oscillation_penalty = 0.0

        if "car_pos_ne" in options:
            self.car_pos_ne = np.array(options["car_pos_ne"], dtype=np.float64)
            self.car_yaw = float(options.get("car_yaw", 0.0))
            self.car_speed = float(options.get("car_speed", 0.65 * self.car_max_speed_ms))
        elif deterministic:
            self.car_pos_ne = np.array([280.0, -180.0], dtype=np.float64)
            self.car_yaw = 1.25
            self.car_speed = 0.65 * self.car_max_speed_ms
        else:
            r_init = float(self.np_random.uniform(80.0, self.boundary_radius_m * 0.65))
            az_init = float(self.np_random.uniform(-np.pi, np.pi))
            self.car_pos_ne = np.array([r_init * np.cos(az_init), r_init * np.sin(az_init)], dtype=np.float64)
            self.car_yaw = float(self.np_random.uniform(-np.pi, np.pi))
            self.car_speed = float(self.np_random.uniform(0.4, 0.85)) * self.car_max_speed_ms
        self.car_spawn_center = self.car_pos_ne.copy()

        self.car_steer = 0.0
        self.car_vel_ne = np.array(
            [self.car_speed * np.cos(self.car_yaw), self.car_speed * np.sin(self.car_yaw)],
            dtype=np.float64,
        )
        self.target_history_ned = []
        for h_i in range(20, -1, -1):
            dt_back = h_i * self.control_dt
            self.target_history_ned.append(
                np.array(
                    [
                        self.car_pos_ne[0] - self.car_vel_ne[0] * dt_back,
                        self.car_pos_ne[1] - self.car_vel_ne[1] * dt_back,
                        0.0,
                    ],
                    dtype=np.float64,
                )
            )

        self.step_count = 0
        self.prev_dist_3d = self._compute_dist_3d()
        self.min_dist_3d = self.prev_dist_3d
        self.visited_cells = {(int(np.floor(self.car_pos_ne[0] / 12.0)), int(np.floor(self.car_pos_ne[1] / 12.0)))}
        self.max_spawn_displacement_m = 0.0

        return (self.get_missile_obs(), self.get_car_obs()), self.get_info()

    def _step_car_substep(self, steer_cmd: float, throttle_cmd: float, sub_dt: float) -> None:
        """
        Realistic Ackermann Bicycle Model bounded ONLY by the circle of diameter = 2 * (altitude - 70 m)
        centered at origin (0, 0) (radius = self.boundary_radius_m = altitude - 70 m).
        """
        max_steer_rad = np.deg2rad(self.car_cfg.max_steer_deg)
        max_lat_acc = self.car_cfg.max_lat_g * 9.81
        if self.car_speed > 2.5:
            speed_steer_limit = float(np.arctan((self.car_cfg.wheelbase_m * max_lat_acc) / (self.car_speed ** 2)))
            eff_max_steer = min(max_steer_rad, speed_steer_limit)
        else:
            eff_max_steer = max_steer_rad

        target_steer = float(np.clip(steer_cmd, -1.0, 1.0)) * eff_max_steer
        alpha_st = sub_dt / (self.car_cfg.steer_tau_s + sub_dt)
        desired_delta = alpha_st * (target_steer - self.car_steer)
        max_step_steer = np.deg2rad(self.car_cfg.max_steer_rate_degs) * sub_dt
        self.car_steer += float(np.clip(desired_delta, -max_step_steer, max_step_steer))
        self.car_steer = float(np.clip(self.car_steer, -eff_max_steer, eff_max_steer))

        if throttle_cmd >= 0.0:
            speed_ratio = float(np.clip(self.car_speed / max(5.0, self.car_max_speed_ms), 0.0, 1.0))
            power_taper = max(0.20, 1.0 - 0.55 * (speed_ratio ** 1.4))
            lon_acc = float(throttle_cmd) * self.car_cfg.max_accel_ms2 * power_taper
        else:
            lon_acc = float(throttle_cmd) * self.car_cfg.max_brake_ms2

        drag_acc = (
            self.car_cfg.drag_coeff * (self.car_speed ** 2)
            + (self.car_cfg.rolling_resist_ms2 if self.car_speed > 0.05 else 0.0)
            + 0.35 * abs(np.tan(self.car_steer)) * (self.car_speed / max(10.0, self.car_max_speed_ms))
        )
        self.car_speed = float(np.clip(self.car_speed + (lon_acc - drag_acc) * sub_dt, 0.0, self.car_max_speed_ms))

        beta_slip = float(np.arctan(0.5 * np.tan(self.car_steer)))
        yaw_rate = (self.car_speed / self.car_cfg.wheelbase_m) * np.tan(self.car_steer) * np.cos(beta_slip)
        if self.car_speed > 1.0:
            max_yaw_rate = max_lat_acc / self.car_speed
            yaw_rate = float(np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate))

        self.car_yaw = float((self.car_yaw + yaw_rate * sub_dt + np.pi) % (2.0 * np.pi) - np.pi)
        course_ang = self.car_yaw + beta_slip
        self.car_vel_ne[0] = self.car_speed * np.cos(course_ang)
        self.car_vel_ne[1] = self.car_speed * np.sin(course_ang)
        self.car_pos_ne += self.car_vel_ne * sub_dt

        # Enforce ONLY the outer boundary circle around (0, 0) with radius = altitude - 70 m (diameter = 2*(altitude - 70 m))
        r_origin = float(np.hypot(self.car_pos_ne[0], self.car_pos_ne[1]))
        if r_origin > self.boundary_radius_m - 25.0:
            inward_yaw = float(np.arctan2(-self.car_pos_ne[1], -self.car_pos_ne[0]))
            yaw_diff = float((inward_yaw - self.car_yaw + np.pi) % (2.0 * np.pi) - np.pi)
            turn_rate_in = min(max_lat_acc / max(2.5, self.car_speed), 1.8)
            self.car_yaw = float((self.car_yaw + turn_rate_in * sub_dt * np.sign(yaw_diff) + np.pi) % (2.0 * np.pi) - np.pi)
            if r_origin > self.boundary_radius_m:
                self.car_pos_ne *= self.boundary_radius_m / r_origin

    def _step_missile_substep(self, cmd_r_deg: float, cmd_l_deg: float, sub_dt: float) -> None:
        alpha_s = sub_dt / (self.aero_cfg.servo_tau_s + sub_dt)
        self.delta_r_deg += alpha_s * (cmd_r_deg - self.delta_r_deg)
        self.delta_l_deg += alpha_s * (cmd_l_deg - self.delta_l_deg)

        alt_m = max(0.0, float(-self.pos_ned[2]))
        rho = float(self.aero_cfg.rho_sea_level * np.exp(-alt_m / 9500.0))

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

        cl_wing = self.aero_cfg.cl_alpha_wing * sa * ca + self.aero_cfg.cl_vortex_wing * (sa ** 2) * ca * np.sign(sa)
        cd_wing = self.aero_cfg.cd0_body_wing + 1.35 * (sa ** 2) + 0.35 * (sb ** 2)
        lift_wing = q_dyn * self.aero_cfg.wing_area_m2 * cl_wing
        lift_fuse = q_dyn * self.aero_cfg.wing_area_m2 * (0.95 * sa * ca)
        drag_wing = q_dyn * self.aero_cfg.wing_area_m2 * cd_wing

        fx_wing = -drag_wing * ca + lift_wing * sa
        fz_wing = -drag_wing * sa - lift_wing * ca
        fx_fuse = lift_fuse * sa
        fz_fuse = -lift_fuse * ca

        tau_wing_y = 0.0633 * fz_wing
        fy_fin = q_dyn * self.aero_cfg.fin_area_m2 * (self.aero_cfg.cy_beta_fin * sb * cb)
        tau_fin_z = -0.0633 * fy_fin

        p, q, r = float(self.omega[0]), float(self.omega[1]), float(self.omega[2])

        r_x = -0.0690
        S_f = self.aero_cfg.flap_area_each_m2

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
            cn = self.aero_cfg.cl_delta_flap * sd * cd + 1.15 * se * ce
            cd_f = self.aero_cfg.cd_delta_flap * (sd ** 2) + 0.25 * (se ** 2)
            fz = q_loc * S_f * cn
            fx = -q_loc * S_f * cd_f
            tx = r_y * fz
            ty = -r_x * fz + q_loc * S_f * self.aero_cfg.mean_aero_chord_m * self.aero_cfg.cm_delta_flap * sd * cd
            tz = -r_y * fx
            return fx, fz, tx, ty, tz

        fx_r, fz_r, tx_r, ty_r, tz_r = eval_flap(+1.0, self.delta_r_deg)
        fx_l, fz_l, tx_l, ty_l, tz_l = eval_flap(-1.0, self.delta_l_deg)

        b, c = self.aero_cfg.wingspan_m, self.aero_cfg.mean_aero_chord_m
        damp_scale = 0.5 * rho * airspeed * self.aero_cfg.wing_area_m2
        tau_damp_x = -(damp_scale * (b ** 2) * self.aero_cfg.cl_p_damp + self.aero_cfg.viscous_rot_damp) * p
        tau_damp_y = -(damp_scale * (c ** 2) * self.aero_cfg.cm_q_damp * 4.5 + self.aero_cfg.viscous_rot_damp) * q
        tau_damp_z = -(damp_scale * (b ** 2) * self.aero_cfg.cn_r_damp + self.aero_cfg.viscous_rot_damp) * r

        f_body = np.array([fx_wing + fx_fuse + fx_r + fx_l, fy_fin, fz_wing + fz_fuse + fz_r + fz_l], dtype=np.float64)
        tau_body = np.array(
            [
                tx_r + tx_l + tau_damp_x,
                tau_wing_y + ty_r + ty_l + tau_damp_y,
                tau_fin_z + tz_r + tz_l + tau_damp_z,
            ],
            dtype=np.float64,
        )

        ixx, iyy, izz = self.aero_cfg.ixx, self.aero_cfg.iyy, self.aero_cfg.izz
        dwx = (tau_body[0] - (izz - iyy) * q * r) / ixx
        dwy = (tau_body[1] - (ixx - izz) * p * r) / iyy
        dwz = (tau_body[2] - (iyy - ixx) * p * q) / izz

        self.omega[0] = np.clip(self.omega[0] + dwx * sub_dt, -35.0, 35.0)
        self.omega[1] = np.clip(self.omega[1] + dwy * sub_dt, -35.0, 35.0)
        self.omega[2] = np.clip(self.omega[2] + dwz * sub_dt, -35.0, 35.0)

        self.R_wb = integrate_rotation_so3(self.R_wb, self.omega, sub_dt)
        self.angles = rotation_matrix_to_euler(self.R_wb)

        f_world = self.R_wb @ f_body
        accel_ned = f_world / self.aero_cfg.mass_kg + np.array([0.0, 0.0, self.aero_cfg.gravity], dtype=np.float64)
        self.vel_ned += accel_ned * sub_dt
        self.pos_ned += self.vel_ned * sub_dt

    def step_duel(
        self, missile_action: np.ndarray, car_action: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, bool, Dict[str, Any]]:
        m_act = np.clip(np.asarray(missile_action, dtype=np.float32), -1.0, 1.0)
        c_act = np.clip(np.asarray(car_action, dtype=np.float32), -1.0, 1.0)
        self.step_count += 1

        pitch_flap_deg = float(m_act[0]) * MAX_WING_FLAP_DEG
        roll_flap_deg = float(m_act[1]) * 35.0
        cmd_r_deg = float(np.clip(pitch_flap_deg + roll_flap_deg, -MAX_WING_FLAP_DEG, +MAX_WING_FLAP_DEG))
        cmd_l_deg = float(np.clip(pitch_flap_deg - roll_flap_deg, -MAX_WING_FLAP_DEG, +MAX_WING_FLAP_DEG))

        steer_cmd = float(c_act[0])
        throttle_cmd = float(c_act[1])

        n_sub = 10
        sub_dt = self.control_dt / n_sub
        for _ in range(n_sub):
            if -self.pos_ned[2] <= 0.0:
                self.pos_ned[2] = 0.0
                break
            self._step_car_substep(steer_cmd, throttle_cmd, sub_dt)
            self._step_missile_substep(cmd_r_deg, cmd_l_deg, sub_dt)
            d3 = self._compute_dist_3d()
            if d3 < self.min_dist_3d:
                self.min_dist_3d = d3

        self.target_history_ned.append(np.array([float(self.car_pos_ne[0]), float(self.car_pos_ne[1]), 0.0], dtype=np.float64))
        if len(self.target_history_ned) > 25:
            self.target_history_ned.pop(0)

        cur_dist_3d = self._compute_dist_3d()
        cur_dist_2d = float(np.hypot(self.car_pos_ne[0] - self.pos_ned[0], self.car_pos_ne[1] - self.pos_ned[1]))
        dist_dec = self.prev_dist_3d - cur_dist_3d
        self.prev_dist_3d = cur_dist_3d

        # Anti-Oscillation / Vibration Penalty for Missile:
        da_cur = m_act - self.last_m_action
        da_prev = self.last_m_action - self.prev_m_action
        sign_flip_mask = (da_cur * da_prev < -1e-4).astype(np.float32)
        action_osc = float(np.sum(sign_flip_mask * ((da_cur - da_prev) ** 2)))
        domega = self.omega - self.prev_omega
        omega_flip = (domega * self.prev_domega < -1e-3).astype(np.float64)
        rate_osc = float(np.sum(omega_flip * ((domega - self.prev_domega) ** 2))) * 0.02
        osc_penalty = 2.5 * action_osc + rate_osc
        self.total_oscillation_penalty += osc_penalty
        self.prev_m_action = self.last_m_action.copy()
        self.last_m_action = m_act.copy()
        self.prev_domega = domega.copy()
        self.prev_omega = self.omega.copy()

        # Spatial Exploration / New-Area Bonus for the Car (12 m x 12 m grid cells + expanding radius from spawn):
        cell_xy = (int(np.floor(self.car_pos_ne[0] / 12.0)), int(np.floor(self.car_pos_ne[1] / 12.0)))
        new_area_bonus = 0.0
        if cell_xy not in self.visited_cells:
            self.visited_cells.add(cell_xy)
            new_area_bonus += 6.5

        spawn_disp = float(np.hypot(self.car_pos_ne[0] - self.car_spawn_center[0], self.car_pos_ne[1] - self.car_spawn_center[1]))
        if spawn_disp > self.max_spawn_displacement_m:
            new_area_bonus += 1.5 * (spawn_disp - self.max_spawn_displacement_m)
            self.max_spawn_displacement_m = spawn_disp

        alt_m = max(1.0, float(-self.pos_ned[2]))
        vz_down = max(8.0, float(self.vel_ned[2]))
        t_imp = min(20.0, alt_m / vz_down)
        pred_imp_n = float(self.pos_ned[0] + self.vel_ned[0] * t_imp)
        pred_imp_e = float(self.pos_ned[1] + self.vel_ned[1] * t_imp)
        dist_to_pred_imp = float(np.hypot(self.car_pos_ne[0] - pred_imp_n, self.car_pos_ne[1] - pred_imp_e))
        forward_speed_bonus = 0.8 * (self.car_speed / max(5.0, self.car_max_speed_ms))
        stagnation_penalty = -1.5 if self.car_speed < 0.20 * self.car_max_speed_ms else 0.0

        missile_reward = 5.0 * dist_dec - osc_penalty
        car_reward = (
            -2.0 * dist_dec
            + new_area_bonus
            + forward_speed_bonus
            + 0.15 * (dist_to_pred_imp / 50.0)
            + stagnation_penalty
        )

        hit_ground = bool(-self.pos_ned[2] <= 0.0)
        # Strict 3.0 m hit definition (anything > 3.0 m is a Miss (0)!)
        hit_car = bool(hit_ground and cur_dist_2d <= self.car_cfg.hit_radius_m)
        terminated = hit_ground
        if terminated:
            if hit_car:
                missile_reward += 6000.0 - 500.0 * cur_dist_2d
                car_reward -= 2000.0
            else:
                missile_reward += max(0.0, 1200.0 - 10.0 * cur_dist_2d)
                car_reward += min(1500.0, 10.0 * cur_dist_2d)

        truncated = bool(self.step_count >= self.max_steps)

        return (
            (self.get_missile_obs(), self.get_car_obs()),
            (float(missile_reward), float(car_reward)),
            terminated,
            truncated,
            self.get_info(hit_car=hit_car),
        )

    def get_info(self, hit_car: bool = False) -> Dict[str, Any]:
        roll_deg, pitch_deg, yaw_deg = np.rad2deg(self.angles)
        target_err = float(np.hypot(self.car_pos_ne[0] - self.pos_ned[0], self.car_pos_ne[1] - self.pos_ned[1]))
        alt_m = float(max(0.0, -self.pos_ned[2]))
        airspeed = float(np.linalg.norm(self.vel_ned))
        return {
            "step": int(self.step_count),
            "time_s": round(self.step_count * self.control_dt, 2),
            "altitude_m": round(alt_m, 2),
            "pos_x_m": round(float(self.pos_ned[0]), 2),
            "pos_y_m": round(float(self.pos_ned[1]), 2),
            "car_x_m": round(float(self.car_pos_ne[0]), 2),
            "car_y_m": round(float(self.car_pos_ne[1]), 2),
            "car_yaw_deg": round(float(np.rad2deg(self.car_yaw)), 2),
            "car_speed_ms": round(float(self.car_speed), 2),
            "car_speed_kmh": round(float(self.car_speed * 3.6), 1),
            "car_max_speed_kmh": round(self.car_max_speed_kmh, 1),
            "car_steer_deg": round(float(np.rad2deg(self.car_steer)), 2),
            "car_visited_cells": int(len(self.visited_cells)),
            "car_spawn_displacement_m": round(float(self.max_spawn_displacement_m), 2),
            "boundary_radius_m": round(float(self.boundary_radius_m), 1),
            "boundary_diameter_m": round(float(2.0 * self.boundary_radius_m), 1),
            "target_error_m": round(target_err, 2),
            "min_dist_3d_m": round(float(target_err if alt_m <= 0.1 else self.min_dist_3d), 2),
            "hit": bool(hit_car or (alt_m <= 0.1 and target_err <= self.car_cfg.hit_radius_m)),
            "total_oscillation_penalty": round(float(self.total_oscillation_penalty), 3),
            "airspeed_ms": round(airspeed, 2),
            "roll_deg": round(float(roll_deg), 2),
            "pitch_deg": round(float(pitch_deg), 2),
            "yaw_deg": round(float(yaw_deg), 2),
            "delta_r_deg": round(float(self.delta_r_deg), 2),
            "delta_l_deg": round(float(self.delta_l_deg), 2),
        }
