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
    100% Orientation-Free 3D Body-Frame Gymnasium RL Environment for the 300 g Delta-Wing Airframe:
      - NO Euler angles (no roll, no pitch, no yaw) are ever given to the neural network!
      - The network has NO concept of "upright" vs "upside-down" — it is 100% symmetric and free on all 3 axes.
      - Every observation is expressed strictly in the airframe's own local 3D Body Frame (+x_b Nose, +y_b Right Wing, +z_b Belly).

    Supported Training & Test Modes (`target_mode`):
      1. "balls_3d" (3D Multi-Ball Proximity Field in Space):
         - Spawns 6 balls at different 3D positions (North, East, Altitude) in the air (not lined up in the same direction)
           plus 1 final ball on the ground (Altitude = 0 m).
         - Balls are NEVER collected/deleted! Instead, each ball has a continuous closeness score
           `s_i = exp(-min_dist_i / 110.0)` in `[0, 1]` based on how close the glider has passed to ball `i`.
         - As the glider gets close to MORE different balls, the cumulative closeness score `S = sum_i s_i` increases,
           and the reward gain scales superlinearly (`Total Potential = 400 * S + 180 * S^2`) alongside 3D distance-decrease
           toward the nearest unvisited ball below the glider (`gain = 1.0 + 0.55 * S`).
      2. "static" (Ground Target Test):
         - Static GPS target `(N_t, E_t, 0)` placed anywhere 360 deg around `(0, 0)` at radii `150–1,000 m`.
      3. "linear" (Linear Moving Target at Different Speeds):
         - Target moves in a straight line at varied speeds (`4 m/s` slow up to `24 m/s` fast) in any 360 deg direction.
      4. "curved" / "aggressive" (Curved Moving Target at Different Speeds):
         - Target moves along a continuous curved trajectory (`v(t)` rotates smoothly with angular turn rate `omega_t`
           in `[-0.14, +0.14] rad/s` at speeds `8–22 m/s`, creating circular arcs and S-curves on the ground).

    Observation Space (14D Pure 3D Body-Frame State — Zero Euler Angles):
      0: dir_bx         - 3D unit vector to active target/ball along Body Nose (+x_b) in [-1, +1]
      1: dir_by         - 3D unit vector to active target/ball along Body Right Wing (+y_b) in [-1, +1]
      2: dir_bz         - 3D unit vector to active target/ball along Body Belly (+z_b) in [-1, +1]
      3: dist_3d_norm   - 3D distance to active target/ball / 1500.0
      4: u_norm         - Body forward airspeed u / 50.0 (m/s)
      5: v_norm         - Body lateral airspeed v / 50.0 (m/s)
      6: w_norm         - Body normal airspeed w / 50.0 (m/s)
      7: p_norm         - Body roll rate p / 10.0 (rad/s)
      8: q_norm         - Body pitch rate q / 10.0 (rad/s)
      9: r_norm         - Body yaw rate r / 10.0 (rad/s)
     10: grav_bx        - Gravity unit vector along Body Nose (+x_b) = R_wb[2, 0] in [-1, +1]
     11: grav_by        - Gravity unit vector along Body Right Wing (+y_b) = R_wb[2, 1] in [-1, +1]
     12: grav_bz        - Gravity unit vector along Body Belly (+z_b) = R_wb[2, 2] in [-1, +1]
     13: tv_by          - Target lateral velocity in Body Right Wing (+y_b) / 25.0
    """

    metadata = {"render_modes": ["dict"], "render_fps": 20}

    def __init__(
        self,
        target_mode: str = "static",  # "balls_3d", "static", "linear", "curved", or "aggressive"
        release_altitude_m: float = 1000.0,
        control_dt: float = 0.05,
        max_wing_flap_deg: float = MAX_WING_FLAP_DEG,
        randomize_drop: bool = True,
    ) -> None:
        super().__init__()
        assert target_mode in ("balls_3d", "static", "linear", "curved", "aggressive")
        self.target_mode = target_mode
        self.release_altitude_m = float(release_altitude_m)
        self.control_dt = float(control_dt)
        self.max_wing_flap_deg = float(max_wing_flap_deg)
        self.randomize_drop = bool(randomize_drop)

        self.cfg = AirframeAeroConfig(max_wing_flap_deg=self.max_wing_flap_deg)
        self.geom = LinkageGeometry()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
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

        # Active 3D target point in NED [North_m, East_m, Down_m] and velocity [vN, vE]
        self.target_ned = np.array([680.0, 240.0, 0.0], dtype=np.float64)
        self.target_ne = self.target_ned[:2]  # alias for ground N, E
        self.target_vel_ne = np.zeros(2, dtype=np.float64)
        self.target_speed_ms = 0.0
        self.target_hdg_rad = 0.0
        self.target_turn_rate_rads = 0.0
        self.target_turn_freq = 0.0
        # Target position history buffer (21 steps = 1.00 s of past target positions at 20 Hz)
        self.target_history_ned: List[np.ndarray] = []

        # 3D Balls list for "balls_3d" mode: each ball is [North_m, East_m, Down_m (-Alt)]
        self.balls_ned: np.ndarray = np.zeros((1, 3), dtype=np.float64)
        self.ball_min_dists: np.ndarray = np.full(1, 1e6, dtype=np.float64)
        self.active_ball_idx: int = 0
        self.prev_ball_potential: float = 0.0

        self.step_count = 0
        self.max_steps = 1600
        self.prev_dist_3d = 0.0
        self.initial_dist_3d = 1000.0
        self.total_path_length_m = 0.0
        self.total_distance_closed_m = 0.0

    def _select_active_ball(self) -> None:
        """
        In 'balls_3d' mode, balls are never deleted/collected.
        The sensor points to the next ball at or below the glider's current altitude band so the glider
        weaves from ball to ball through the 3D space all the way to the ground ball.
        """
        if self.target_mode != "balls_3d":
            return
        cur_alt = max(0.0, float(-self.pos_ned[2]))
        chosen_idx = len(self.balls_ned) - 1
        for i in range(len(self.balls_ned)):
            ball_alt = float(-self.balls_ned[i, 2])
            if cur_alt >= ball_alt + 25.0:
                chosen_idx = i
                break
        if chosen_idx != self.active_ball_idx:
            self.active_ball_idx = chosen_idx
            self.target_ned = self.balls_ned[self.active_ball_idx].copy()
            self.target_ne = self.target_ned[:2]
            self.target_history_ned = [self.target_ned.copy() for _ in range(21)]
            self.prev_dist_3d = self._compute_dist_3d()

    def _compute_dist_3d(self) -> float:
        return float(np.linalg.norm(self.target_ned - self.pos_ned))

    def _compute_balls_potential(self) -> float:
        """
        Continuous multi-ball proximity potential (balls are NOT collected/deleted!):
          - For each ball i, closeness score s_i = exp(-min_dist_i / 90.0) in (0, 1].
          - Cumulative closeness S = sum_i s_i.
          - Escalating gain: Potential = 350.0 * S + 160.0 * (S ** 2).
            The more distinct balls the glider gets close to, the steeper the marginal reward multiplier!
        """
        closeness = np.exp(-self.ball_min_dists / 90.0)
        s_tot = float(np.sum(closeness))
        return 350.0 * s_tot + 160.0 * (s_tot ** 2)

    def _get_obs(self) -> np.ndarray:
        p, q, r = self.omega

        # 3D vector from current airframe position (N, E, D) to active 3D target/ball (N_t, E_t, D_t)
        d_ned = self.target_ned - self.pos_ned
        dist_3d = max(1.0, float(np.linalg.norm(d_ned)))
        dir_body = (self.R_wb.T @ d_ned) / dist_3d

        # Body-frame airspeed components (u, v, w)
        v_body = self.R_wb.T @ self.vel_ned

        # Past Target Positions (5, 10, 15, 20 steps ago = 0.25s, 0.50s, 0.75s, 1.00s ago)
        # Displacements from current target_ned projected into the missile's 3D Body Frame!
        hist_features: List[float] = []
        n_hist = len(self.target_history_ned)
        for k_back, norm_scale in [(5, 6.25), (10, 12.5), (15, 18.75), (20, 25.0)]:
            idx_h = max(0, n_hist - 1 - k_back)
            past_ned = self.target_history_ned[idx_h] if n_hist > 0 else self.target_ned
            delta_ned = self.target_ned - past_ned
            delta_body = self.R_wb.T @ delta_ned
            hist_features.append(float(np.clip(delta_body[0] / norm_scale, -3.5, 3.5)))
            hist_features.append(float(np.clip(delta_body[1] / norm_scale, -3.5, 3.5)))

        #Also include instantaneous 1-step target displacement (0.05s ago) for ultra-fast reaction to sudden RC car jinks!
        past_1 = self.target_history_ned[max(0, n_hist - 2)] if n_hist > 1 else self.target_ned
        step_delta_body = self.R_wb.T @ ((self.target_ned - past_1) / self.control_dt)

        return np.array(
            [
                float(dir_body[0]),
                float(dir_body[1]),
                float(dir_body[2]),
                np.clip(dist_3d / 1500.0, 0.0, 3.0),
                float(v_body[0]) / 50.0,
                float(v_body[2]) / 50.0,
                float(p) / 10.0,
                float(q) / 10.0,
                float(r) / 10.0,
                float(self.R_wb[2, 0]),
                float(self.R_wb[2, 1]),
                float(self.R_wb[2, 2]),
                np.clip(float(step_delta_body[0]) / 25.0, -3.5, 3.5),
                np.clip(float(step_delta_body[1]) / 25.0, -3.5, 3.5),
                *hist_features,
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
            # 50% of training episodes start from a random mid-air state (random angle, random speed 12-52 m/s, random altitude 450-1000 m)
            # to mimic chasing a target when the target suddenly switches to a completely different direction!
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

        self.delta_r_deg = 0.0
        self.delta_l_deg = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.total_distance_closed_m = 0.0
        self.target_turn_rate_rads = 0.0
        self.target_turn_freq = 0.0

        if "target_ne" in options:
            t_ne = np.array(options["target_ne"], dtype=np.float64)
            self.target_ned = np.array([t_ne[0], t_ne[1], 0.0], dtype=np.float64)
            self.target_ne = self.target_ned[:2]
            self.target_vel_ne = np.array(options.get("target_vel_ne", [0.0, 0.0]), dtype=np.float64)
            self.balls_ned = self.target_ned.reshape(1, 3).copy()
            self.ball_min_dists = np.array([self._compute_dist_3d()], dtype=np.float64)
        elif self.target_mode == "balls_3d":
            # Spawn 4 air balls in ALL 4 QUADRANTS (+N/+E, -N/+E, -N/-E, +N/-E) + 1 final ground ball!
            alt_levels = [760.0, 560.0, 360.0, 180.0, 0.0]
            balls = []
            if deterministic:
                offsets = [
                    (+130.0, +130.0),  # Quadrant I (+N, +E)
                    (-140.0, +160.0),  # Quadrant II (-N, +E)
                    (-220.0, -180.0),  # Quadrant III (-N, -E)
                    (+260.0, -220.0),  # Quadrant IV (+N, -E)
                    (+460.0, +160.0),  # Final Ground Ball
                ]
                for (bn, be), ba in zip(offsets, alt_levels):
                    balls.append([bn, be, -ba])
            else:
                # Place each ball in a different quadrant around the origin (0, 0) so all 4 quadrants are covered!
                quad_signs = [(+1.0, +1.0), (-1.0, +1.0), (-1.0, -1.0), (+1.0, -1.0)]
                perm_q = self.np_random.permutation(4)
                for q_i in range(4):
                    sn, se = quad_signs[int(perm_q[q_i])]
                    r_b = float(self.np_random.uniform(110.0 + 45.0 * q_i, 190.0 + 65.0 * q_i))
                    ang_b = float(self.np_random.uniform(0.25, 1.32))
                    balls.append([sn * r_b * np.cos(ang_b), se * r_b * np.sin(ang_b), -alt_levels[q_i]])
                # Final ground ball anywhere 360 deg at radius 220..650 m
                r_g = float(self.np_random.uniform(220.0, 650.0))
                az_g = float(self.np_random.uniform(-np.pi, np.pi))
                balls.append([r_g * np.cos(az_g), r_g * np.sin(az_g), 0.0])
            self.balls_ned = np.array(balls, dtype=np.float64)
            self.active_ball_idx = 0
            self.target_ned = self.balls_ned[0].copy()
            self.target_ne = self.target_ned[:2]
            self.target_vel_ne = np.zeros(2, dtype=np.float64)
            dists0 = np.linalg.norm(self.balls_ned - self.pos_ned.reshape(1, 3), axis=1)
            self.ball_min_dists = dists0.copy()
            self.prev_ball_potential = self._compute_balls_potential()
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
            else:  # "curved" or "aggressive" — Curved Moving Target at high speed!
                self.target_ned = np.array([500.0, -240.0, 0.0], dtype=np.float64)
                self.target_speed_ms = 14.0
                self.target_hdg_rad = 1.25
                self.target_turn_rate_rads = 0.095  # smooth curved arc / S-turn
                self.target_turn_freq = 0.16
                self.target_vel_ne = np.array(
                    [self.target_speed_ms * np.cos(self.target_hdg_rad), self.target_speed_ms * np.sin(self.target_hdg_rad)],
                    dtype=np.float64,
                )
            self.target_ne = self.target_ned[:2]
            self.balls_ned = self.target_ned.reshape(1, 3).copy()
            self.ball_min_dists = np.array([self._compute_dist_3d()], dtype=np.float64)
        else:
            if self.target_mode == "static":
                max_reach = min(920.0, alt * 0.92)
                r_init = float(self.np_random.uniform(50.0, max_reach))
                az_init = float(self.np_random.uniform(-np.pi, np.pi))
                self.target_ned = np.array([r_init * np.cos(az_init), r_init * np.sin(az_init), 0.0], dtype=np.float64)
                self.target_vel_ne = np.zeros(2, dtype=np.float64)
            elif self.target_mode == "linear":
                # Linear moving target at different speeds (4 m/s to 20 m/s) across all 4 quadrants!
                max_reach = min(680.0, alt * 0.68)
                r_init = float(self.np_random.uniform(120.0, max_reach))
                az_init = float(self.np_random.uniform(-np.pi, np.pi))
                self.target_ned = np.array([r_init * np.cos(az_init), r_init * np.sin(az_init), 0.0], dtype=np.float64)
                self.target_speed_ms = float(self.np_random.uniform(4.0, 20.0))
                tangent_dir = az_init + float(self.np_random.choice([-1.0, 1.0])) * float(self.np_random.uniform(1.3, 2.2))
                self.target_hdg_rad = tangent_dir
                self.target_vel_ne = np.array(
                    [self.target_speed_ms * np.cos(self.target_hdg_rad), self.target_speed_ms * np.sin(self.target_hdg_rad)],
                    dtype=np.float64,
                )
            else:  # "curved" or "aggressive" — Curved Moving Target across all 4 quadrants!
                max_reach = min(650.0, alt * 0.65)
                r_init = float(self.np_random.uniform(120.0, max_reach))
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
            self.balls_ned = self.target_ned.reshape(1, 3).copy()
            self.ball_min_dists = np.array([self._compute_dist_3d()], dtype=np.float64)

        # Pre-populate 21 steps (1.00 s) of target position history so even at step 0 the past trajectory is well-defined
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

        # Update moving target position in "linear" or "curved"/"aggressive" modes
        if self.target_mode in ("curved", "aggressive"):
            # Curved moving target: heading rotates continuously (circular arc + sinusoidal S-curve modulation)
            inst_turn = self.target_turn_rate_rads * np.cos(self.target_turn_freq * t_sec)
            self.target_hdg_rad += inst_turn * self.control_dt
            self.target_vel_ne[0] = self.target_speed_ms * np.cos(self.target_hdg_rad)
            self.target_vel_ne[1] = self.target_speed_ms * np.sin(self.target_hdg_rad)

        if self.target_mode in ("linear", "curved", "aggressive"):
            self.target_ned[0] += self.target_vel_ne[0] * self.control_dt
            self.target_ned[1] += self.target_vel_ne[1] * self.control_dt
            if self.target_ned[0] < -1150.0 or self.target_ned[0] > 1150.0:
                self.target_vel_ne[0] *= -1.0
                self.target_hdg_rad = float(np.arctan2(self.target_vel_ne[1], self.target_vel_ne[0]))
                self.target_ned[0] = np.clip(self.target_ned[0], -1150.0, 1150.0)
            if self.target_ned[1] < -1150.0 or self.target_ned[1] > 1150.0:
                self.target_vel_ne[1] *= -1.0
                self.target_hdg_rad = float(np.arctan2(self.target_vel_ne[1], self.target_vel_ne[0]))
                self.target_ned[1] = np.clip(self.target_ned[1], -1150.0, 1150.0)
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
            if self.target_mode == "balls_3d":
                dists = np.linalg.norm(self.balls_ned - self.pos_ned.reshape(1, 3), axis=1)
                self.ball_min_dists = np.minimum(self.ball_min_dists, dists)
        step_path_len = float(np.linalg.norm(self.pos_ned - prev_pos_step))
        self.total_path_length_m += step_path_len

        if self.target_mode == "balls_3d":
            # Escalating multi-ball closeness reward (balls are NOT collected/deleted!)
            cur_pot = self._compute_balls_potential()
            pot_gain = cur_pot - self.prev_ball_potential
            self.prev_ball_potential = cur_pot

            cur_dist_3d = self._compute_dist_3d()
            dist_dec = self.prev_dist_3d - cur_dist_3d
            self.prev_dist_3d = cur_dist_3d
            self.total_distance_closed_m += dist_dec

            closeness_sum = float(np.sum(np.exp(-self.ball_min_dists / 90.0)))
            multiplier = 1.0 + 0.60 * closeness_sum
            reward = pot_gain + 4.0 * multiplier * dist_dec
            self._select_active_ball()
        else:
            cur_dist_3d = self._compute_dist_3d()
            cur_dist_2d = float(
                np.hypot(self.target_ned[0] - self.pos_ned[0], self.target_ned[1] - self.pos_ned[1])
            )
            dist_dec = self.prev_dist_3d - cur_dist_3d
            self.prev_dist_3d = cur_dist_3d
            self.total_distance_closed_m += dist_dec

            # --- Shortest-Path Efficiency + Sub-3m Precision Reward ---
            # 1. Reward closing 3D distance along the shortest 3D Euclidean line, penalized by wasted detour path (step_path_len - dist_dec)
            wasted_detour = max(0.0, step_path_len - max(0.0, dist_dec))
            reward = 6.0 * dist_dec - 1.5 * wasted_detour
            if -self.pos_ned[2] <= 0.0:
                path_eff = min(1.0, self.initial_dist_3d / max(1.0, self.total_path_length_m))
                # Massive sub-3m precision bonus + shortest-path efficiency multiplier on landing!
                sub_3m_bonus = 2500.0 * float(np.exp(-((cur_dist_2d / 3.0) ** 2)))
                reward += (max(0.0, 1500.0 - 10.0 * cur_dist_2d) + sub_3m_bonus) * (0.5 + 0.5 * path_eff)

        hit_ground = bool(-self.pos_ned[2] <= 0.0)
        terminated = hit_ground
        truncated = bool(self.step_count >= self.max_steps)
        self.last_action = act.copy()

        return self._get_obs(), float(reward), terminated, truncated, self.get_info()

    def get_info(self) -> Dict[str, Any]:
        roll_deg, pitch_deg, yaw_deg = np.rad2deg(self.angles)
        xy_dist = float(np.hypot(self.pos_ned[0], self.pos_ned[1]))
        target_err = float(
            np.hypot(self.target_ned[0] - self.pos_ned[0], self.target_ned[1] - self.pos_ned[1])
        )
        alt_m = float(max(0.0, -self.pos_ned[2]))
        airspeed = float(np.linalg.norm(self.vel_ned))
        closeness_scores = np.exp(-self.ball_min_dists / 90.0)
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
            "shortest_path_efficiency": round(float(path_eff), 4),
            "mean_ball_min_dist_m": round(float(np.mean(self.ball_min_dists)), 2),
            "cumulative_ball_closeness": round(float(np.sum(closeness_scores)), 2),
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
