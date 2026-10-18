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


class GlideGpsTargetEnv(gym.Env):
    """
    Gymnasium RL Environment for the 300 g Delta-Wing Airframe Guided to a Desired GPS Coordinate:
      - Stage 1 ("static"): Static GPS coordinate (N_target, E_target) on the ground.
      - Stage 2 ("linear"): GPS coordinate moving in a straight line at constant linear speed while the glider flies.
      - Stage 3 ("aggressive"): Mirror-Point Flip Across the Axis — starts at point (X_0, Y_0) and repeatedly flips
        to the exact mirror point (-X_0, -Y_0) across the origin/axis while the glider is in the air!

    Pure Distance-Decreasing Reward Rule (User Specification):
      - The algorithm is rewarded ONLY when the horizontal distance to the current target is decreasing:
          reward = 5.0 * (prev_dist_to_target - cur_dist_to_target)
      - Whenever the target moves or flips to its mirror point across the axis, prev_dist_to_target is reset to the
        new target position, and the network immediately earns positive reward again for decreasing the distance
        toward the new target!
      - It does NOT need to touch the target on the ground (zero terminal landing bonus, zero attitude/roll penalties).
      - The network discovers on its own how to pull a half-loop / roll over to 180 deg and fly upside-down (inverted)
        whenever the target flips across the axis!

    Observation Space (14D Singularity-Free Body-Frame IMU + GPS Target Vector):
      0: pitch_norm     - Pitch angle theta / (pi / 2)
      1: sin_roll       - sin(roll)
      2: cos_roll       - cos(roll) (+1 = upright, -1 = upside-down inverted flight!)
      3: gyro_p         - Roll rate p / 10.0 (rad/s)
      4: gyro_q         - Pitch rate q / 10.0 (rad/s)
      5: gyro_r         - Yaw rate r / 10.0 (rad/s)
      6: airspeed_n     - Airspeed V / 50.0 (m/s)
      7: vz_norm        - Vertical descent rate vd / 50.0 (m/s)
      8: alt_norm       - Current altitude / 1000.0
      9: err_bx         - 3D Target unit vector along Body Nose Axis (+x_b) in [-1, +1]
     10: err_by         - 3D Target unit vector along Body Right Wing Axis (+y_b) in [-1, +1]
     11: err_bz         - 3D Target unit vector along Body Belly Axis (+z_b) in [-1, +1]
     12: rel_x_horiz    - Horizontal target vector in heading frame (cos(yaw)*dN + sin(yaw)*dE) / 1000.0
     13: rel_y_horiz    - Horizontal target vector in heading frame (-sin(yaw)*dN + cos(yaw)*dE) / 1000.0
    """

    metadata = {"render_modes": ["dict"], "render_fps": 20}

    def __init__(
        self,
        target_mode: str = "static",  # "static", "linear", or "aggressive" (mirror-point flips)
        release_altitude_m: float = 1000.0,
        control_dt: float = 0.05,
        max_wing_flap_deg: float = MAX_WING_FLAP_DEG,
        randomize_drop: bool = True,
    ) -> None:
        super().__init__()
        assert target_mode in ("static", "linear", "aggressive")
        self.target_mode = target_mode
        self.release_altitude_m = float(release_altitude_m)
        self.control_dt = float(control_dt)
        self.max_wing_flap_deg = float(max_wing_flap_deg)
        self.randomize_drop = bool(randomize_drop)

        self.cfg = AirframeAeroConfig(max_wing_flap_deg=self.max_wing_flap_deg)
        self.geom = LinkageGeometry()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.R_wb = np.eye(3, dtype=np.float64)
        self.angles = np.zeros(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self.pos_ned = np.array([0.0, 0.0, -1000.0], dtype=np.float64)
        self.vel_ned = np.zeros(3, dtype=np.float64)
        self.delta_r_deg = 0.0
        self.delta_l_deg = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)

        self.target_ne = np.array([800.0, 0.0], dtype=np.float64)
        self.target_vel_ne = np.zeros(2, dtype=np.float64)
        self.aggressive_schedule: List[Tuple[int, float, float, float, float]] = []

        self.step_count = 0
        self.max_steps = 1600
        self.prev_dist_to_target = 0.0
        self.total_distance_closed_m = 0.0

    def _get_obs(self) -> np.ndarray:
        roll, pitch, yaw = self.angles
        p, q, r = self.omega
        airspeed = float(np.linalg.norm(self.vel_ned))
        vd = float(self.vel_ned[2])
        alt_m = max(0.0, float(-self.pos_ned[2]))

        # Horizontal vector from airframe to target (dN, dE, 0) projected into 3D Body Frame!
        # Using horizontal d_ned = [dN, dE, 0] ensures err_b points strictly toward the horizontal target direction
        # in the body frame: if err_b[2] > 0, target is on the belly side (+z_b); if err_b[2] < 0, target is on the dorsal/top side (-z_b)!
        d_horiz_ned = np.array(
            [
                float(self.target_ne[0] - self.pos_ned[0]),
                float(self.target_ne[1] - self.pos_ned[1]),
                0.0,
            ],
            dtype=np.float64,
        )
        dist_h = max(1.0, float(np.linalg.norm(d_horiz_ned)))
        unit_b = (self.R_wb.T @ d_horiz_ned) / dist_h

        cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
        rel_x_horiz = cy * d_horiz_ned[0] + sy * d_horiz_ned[1]
        rel_y_horiz = -sy * d_horiz_ned[0] + cy * d_horiz_ned[1]

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
                alt_m / 1000.0,
                float(unit_b[0]),
                float(unit_b[1]),
                float(unit_b[2]),
                np.clip(rel_x_horiz / 1000.0, -3.0, 3.0),
                np.clip(rel_y_horiz / 1000.0, -3.0, 3.0),
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

        if self.randomize_drop and not deterministic:
            roll0 = float(self.np_random.uniform(-4.0, 4.0))
            pitch0 = float(self.np_random.uniform(-89.5, -86.0))
            yaw0 = float(self.np_random.uniform(-15.0, 15.0))
            self.omega = self.np_random.uniform(-0.15, 0.15, size=3).astype(np.float64)
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
        self.total_distance_closed_m = 0.0
        self.aggressive_schedule = []

        if "target_ne" in options:
            self.target_ne = np.array(options["target_ne"], dtype=np.float64)
            self.target_vel_ne = np.array(options.get("target_vel_ne", [0.0, 0.0]), dtype=np.float64)
        elif deterministic:
            if self.target_mode == "static":
                self.target_ne = np.array([850.0, 260.0], dtype=np.float64)
                self.target_vel_ne = np.array([0.0, 0.0], dtype=np.float64)
            elif self.target_mode == "linear":
                self.target_ne = np.array([650.0, -200.0], dtype=np.float64)
                self.target_vel_ne = np.array([4.5, 9.0], dtype=np.float64)
            else:
                # "aggressive": Starts at (+800, +220), then flips across the axis to its exact mirror point
                # (-800, -220), then flips back to (+800, -220), then (-800, +220)!
                self.target_ne = np.array([+800.0, +220.0], dtype=np.float64)
                self.target_vel_ne = np.array([0.0, 0.0], dtype=np.float64)
                self.aggressive_schedule = [
                    (160, -800.0, -220.0, 0.0, 0.0),
                    (360, +800.0, -220.0, 0.0, 0.0),
                    (560, -800.0, +220.0, 0.0, 0.0),
                ]
        else:
            if self.target_mode == "static":
                r_init = float(self.np_random.uniform(500.0, 1000.0))
                az_init = float(self.np_random.uniform(-np.pi, np.pi))
                self.target_ne = np.array([r_init * np.cos(az_init), r_init * np.sin(az_init)], dtype=np.float64)
                self.target_vel_ne = np.zeros(2, dtype=np.float64)
            elif self.target_mode == "linear":
                r_init = float(self.np_random.uniform(550.0, 900.0))
                az_init = float(self.np_random.uniform(-np.pi, np.pi))
                self.target_ne = np.array([r_init * np.cos(az_init), r_init * np.sin(az_init)], dtype=np.float64)
                spd = float(self.np_random.uniform(5.0, 12.0))
                hdg = float(self.np_random.uniform(-np.pi, np.pi))
                self.target_vel_ne = np.array([spd * np.cos(hdg), spd * np.sin(hdg)], dtype=np.float64)
            else:
                # "aggressive": Start at random (X_0, Y_0) and flip to exact mirror point (-X_0, -Y_0) across the axis multiple times!
                r_init = float(self.np_random.uniform(650.0, 950.0))
                az_init = float(self.np_random.uniform(-np.pi, np.pi))
                x0 = r_init * np.cos(az_init)
                y0 = r_init * np.sin(az_init)
                self.target_ne = np.array([x0, y0], dtype=np.float64)
                self.target_vel_ne = np.zeros(2, dtype=np.float64)
                s1 = int(self.np_random.integers(140, 185))
                s2 = s1 + int(self.np_random.integers(170, 220))
                s3 = s2 + int(self.np_random.integers(170, 220))
                self.aggressive_schedule = [
                    (s1, -x0, -y0, 0.0, 0.0),
                    (s2, +x0, -y0, 0.0, 0.0),
                    (s3, -x0, +y0, 0.0, 0.0),
                ]

        self.prev_dist_to_target = float(
            np.hypot(self.target_ne[0] - self.pos_ned[0], self.target_ne[1] - self.pos_ned[1])
        )

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

        # Trigger mirror-point flips across the axis in "aggressive" mode
        if self.target_mode == "aggressive":
            for trig_step, tn, te, vn, ve in self.aggressive_schedule:
                if self.step_count == trig_step:
                    self.target_ne[0] = tn
                    self.target_ne[1] = te
                    self.target_vel_ne[0] = vn
                    self.target_vel_ne[1] = ve
                    # Reset prev_dist_to_target to the new mirror target so decreasing distance to the new target rewards again!
                    self.prev_dist_to_target = float(
                        np.hypot(self.target_ne[0] - self.pos_ned[0], self.target_ne[1] - self.pos_ned[1])
                    )
                    break

        if self.target_mode in ("linear", "aggressive"):
            self.target_ne += self.target_vel_ne * self.control_dt
            if self.target_ne[0] < -1250.0 or self.target_ne[0] > 1250.0:
                self.target_vel_ne[0] *= -1.0
                self.target_ne[0] = np.clip(self.target_ne[0], -1250.0, 1250.0)
            if self.target_ne[1] < -1250.0 or self.target_ne[1] > 1250.0:
                self.target_vel_ne[1] *= -1.0
                self.target_ne[1] = np.clip(self.target_ne[1], -1250.0, 1250.0)
            # Update prev_dist_to_target for target motion so reward measures the glider's own closing motion
            self.prev_dist_to_target = float(
                np.hypot(self.target_ne[0] - self.pos_ned[0], self.target_ne[1] - self.pos_ned[1])
            )

        pitch_flap_deg = float(act[0]) * self.max_wing_flap_deg
        roll_flap_deg = float(act[1]) * 20.0
        cmd_r_deg = float(np.clip(pitch_flap_deg + roll_flap_deg, -self.max_wing_flap_deg, +self.max_wing_flap_deg))
        cmd_l_deg = float(np.clip(pitch_flap_deg - roll_flap_deg, -self.max_wing_flap_deg, +self.max_wing_flap_deg))

        n_sub = 10
        sub_dt = self.control_dt / n_sub
        for _ in range(n_sub):
            if -self.pos_ned[2] <= 0.0:
                self.pos_ned[2] = 0.0
                break
            self._step_physics_substep(cmd_r_deg, cmd_l_deg, sub_dt)

        cur_dist_2d = float(
            np.hypot(self.target_ne[0] - self.pos_ned[0], self.target_ne[1] - self.pos_ned[1])
        )
        distance_decrease = self.prev_dist_to_target - cur_dist_2d
        self.prev_dist_to_target = cur_dist_2d
        self.total_distance_closed_m += distance_decrease

        # --- 100% Pure Distance-Decrease Reward (User Specification) ---
        # Positive when distance to target is decreasing; negative when increasing.
        # Does NOT need to touch the target on the ground — zero terminal bonus, zero attitude penalties!
        reward = 5.0 * distance_decrease

        hit_ground = bool(-self.pos_ned[2] <= 0.0)
        terminated = hit_ground
        truncated = bool(self.step_count >= self.max_steps)
        self.last_action = act.copy()

        return self._get_obs(), float(reward), terminated, truncated, self.get_info()

    def get_info(self) -> Dict[str, Any]:
        roll_deg, pitch_deg, yaw_deg = np.rad2deg(self.angles)
        xy_dist = float(np.hypot(self.pos_ned[0], self.pos_ned[1]))
        target_err = float(
            np.hypot(self.target_ne[0] - self.pos_ned[0], self.target_ne[1] - self.pos_ned[1])
        )
        alt_m = float(max(0.0, -self.pos_ned[2]))
        airspeed = float(np.linalg.norm(self.vel_ned))
        return {
            "step": int(self.step_count),
            "time_s": round(self.step_count * self.control_dt, 2),
            "altitude_m": round(alt_m, 2),
            "xy_distance_m": round(xy_dist, 2),
            "pos_x_m": round(float(self.pos_ned[0]), 2),
            "pos_y_m": round(float(self.pos_ned[1]), 2),
            "target_x_m": round(float(self.target_ne[0]), 2),
            "target_y_m": round(float(self.target_ne[1]), 2),
            "target_vx_ms": round(float(self.target_vel_ne[0]), 2),
            "target_vy_ms": round(float(self.target_vel_ne[1]), 2),
            "target_error_m": round(target_err, 2),
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
