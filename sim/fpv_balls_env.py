from __future__ import annotations

from dataclasses import dataclass
import itertools
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
from sim.glide_gps_env import compute_inav_circular_attitude
from sim.wing_linkage import (
    LinkageGeometry,
    MAX_WING_FLAP_DEG,
    wing_flap_deg_to_servo_deg,
)


@dataclass
class FpvPusherMotorConfig:
    """
    Standard Rear-Mounted FPV Drone Brushless Pusher Motor + Propeller (e.g. 2207 2400KV 4S/6S with 5x4.3x3 FPV prop):
      - Mounted at rear tail center (x_b = -0.0913 m, y_b = 0, z_b = 0) pushing forward along +x_b.
      - Prop diameter D = 0.127 m (5.0 inch), geometric pitch p_prop = 0.109 m (4.3 inch).
      - Max RPM = 32,000 RPM -> max prop pitch efflux velocity V_pitch_max = (RPM/60) * p_prop * 1.65 = 96.0 m/s.
      - Max static thrust T_max = 11.5 N (~3.3:1 thrust-to-weight ratio on 355 g powered airframe).
      - Advance-ratio thrust model: T(throttle, u) = throttle * T_max * clip(1.0 - max(0, u) / (V_pitch_max * sqrt(max(0.15, throttle))), 0.0, 1.2)
      - Prop-wash slipstream velocity over rear fins/flaps: v_wash = sqrt(max(0, u)^2 + 2 * T / (rho * A_disk)) - max(0, u)
      - Motor first-order ESC/rotor spool-up lag tau_motor_s = 0.025 s.
    """

    total_mass_kg: float = 0.355  # 300 g airframe + 55 g FPV motor/ESC/prop
    prop_diameter_m: float = 0.127  # 5-inch FPV propeller
    max_rpm: float = 32000.0
    max_static_thrust_n: float = 11.5
    max_pitch_efflux_ms: float = 96.0
    motor_tau_s: float = 0.025
    max_achievable_speed_ms: float = 75.0  # Reference maximum cruising/diving speed V_max for path-speed reward


def solve_shortest_3d_ball_path(start_pos_ned: np.ndarray, balls_ned: np.ndarray) -> Tuple[List[int], float, List[float]]:
    """
    Find the shortest 3D Euclidean path (Traveling Salesperson tour) starting at `start_pos_ned`
    and visiting all `N` balls in `balls_ned`.
    Uses exact permutation search for N <= 6 and fast Nearest-Neighbor + 2-Opt refinement for N > 6 (20 to 100+ balls).
    """
    n_balls = len(balls_ned)
    if n_balls <= 6:
        best_len = float("inf")
        best_order: List[int] = list(range(n_balls))
        best_legs: List[float] = []
        for perm in itertools.permutations(range(n_balls)):
            cur_p = start_pos_ned
            tot_d = 0.0
            legs: List[float] = []
            for idx in perm:
                d = float(np.linalg.norm(balls_ned[idx] - cur_p))
                tot_d += d
                legs.append(d)
                if tot_d >= best_len:
                    break
                cur_p = balls_ned[idx]
            if tot_d < best_len:
                best_len = tot_d
                best_order = list(perm)
                best_legs = legs
        return best_order, float(best_len), best_legs

    # Fast Nearest-Neighbor + 2-Opt TSP for 10, 20, 50, or 100+ balls:
    unvisited = set(range(n_balls))
    order: List[int] = []
    cur_p = start_pos_ned
    while unvisited:
        nxt = min(unvisited, key=lambda idx: float(np.linalg.norm(balls_ned[idx] - cur_p)))
        order.append(nxt)
        unvisited.remove(nxt)
        cur_p = balls_ned[nxt]

    # 2-Opt local search refinement:
    def tour_pt(k: int) -> np.ndarray:
        return start_pos_ned if k < 0 else balls_ned[order[k]]

    improved = True
    iters = 0
    while improved and iters < 25:
        improved = False
        iters += 1
        for i in range(n_balls - 1):
            for j in range(i + 1, n_balls):
                p_prev = tour_pt(i - 1)
                p_i = tour_pt(i)
                p_j = tour_pt(j)
                p_next = tour_pt(j + 1) if j + 1 < n_balls else None
                d_old = float(np.linalg.norm(p_i - p_prev)) + (float(np.linalg.norm(p_next - p_j)) if p_next is not None else 0.0)
                d_new = float(np.linalg.norm(p_j - p_prev)) + (float(np.linalg.norm(p_next - p_i)) if p_next is not None else 0.0)
                if d_new + 1e-4 < d_old:
                    order[i : j + 1] = reversed(order[i : j + 1])
                    improved = True

    legs_out: List[float] = []
    tot_len = 0.0
    cur_p = start_pos_ned
    for idx in order:
        d = float(np.linalg.norm(balls_ned[idx] - cur_p))
        legs_out.append(d)
        tot_len += d
        cur_p = balls_ned[idx]
    return order, float(tot_len), legs_out


class FpvPoweredBallsEnv(gym.Env):
    """
    Powered 300g+55g Delta-Wing with Rear FPV Brushless Pusher Motor & Propeller in Unrestricted 3D Multi-Ball Space:
      - Action Space (3D in [-1, +1]):
          0: pitch_flap_cmd in [-1, +1] -> [-50 deg, +50 deg]
          1: roll_flap_cmd  in [-1, +1] -> [-50 deg, +50 deg]
          2: motor_power_cmd in [-1, +1] -> mapped to throttle in [0.0, 1.0] ("gas" / motor power)
      - Observation Space (25D — Zero Clipping):
          0..2  : Unit 3D vector to active uncollected ball in body frame (dir_bx, dir_by, dir_bz)
          3     : 3D distance to active ball / 1000.0
          4..6  : Body airspeed components (u / 60.0, v / 60.0, w / 60.0)
          7..9  : Body gyroscope angular rates (p / 10.0, q / 10.0, r / 10.0)
          10..15: Circularly normalized INAV attitude (sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw)
          16    : Current FPV motor throttle / power state in [0.0, 1.0]
          17    : Current airspeed ratio V / V_max (V / 75.0 m/s)
          18..20: Unit 3D vector from active ball to the NEXT ball on the optimal path in body frame (lookahead)
          21    : 3D distance from active ball to next ball / 1000.0
          22    : Fraction of balls remaining (balls_remaining / num_balls)
          23..24: Previous pitch and roll flap commands (last_action[0], last_action[1])
      - Reward Function (Strictly Outcome-Based — Zero Hand-Coded Flight Rules):
          1. Touching Balls: The drone must physically touch each ball (3D distance <= ball_touch_radius_m).
             Touching ball `k` gives a large touch reward (+1500.0 * (1 + 0.25 * balls_touched)).
          2. Speed / Time Efficiency Along Shortest Path:
             We compute the exact shortest 3D path `L_path` through all balls and the theoretical minimum time
             `T_ideal = L_path / V_max` (where `V_max = 75.0 m/s` is the maximum speed the powered object can reach).
             For each ball touched (and upon finishing all balls), we reward the network proportionally to how close
             its realized average speed `V_realized = leg_dist / leg_time` (and total `L_path / total_time`) is to `V_max`!
    """

    metadata = {"render_modes": ["dict"], "render_fps": 20}

    def __init__(
        self,
        num_balls: int = 20,
        ball_touch_radius_m: float = 3.0,
        release_altitude_m: float = 500.0,
        control_dt: float = 0.05,
        max_wing_flap_deg: float = MAX_WING_FLAP_DEG,
        randomize_start: bool = True,
    ) -> None:
        super().__init__()
        self.num_balls = int(num_balls)
        self.ball_touch_radius_m = float(ball_touch_radius_m)
        self.release_altitude_m = float(release_altitude_m)
        self.control_dt = float(control_dt)
        self.max_wing_flap_deg = float(max_wing_flap_deg)
        self.randomize_start = bool(randomize_start)

        self.cfg = AirframeAeroConfig(max_wing_flap_deg=self.max_wing_flap_deg)
        self.motor_cfg = FpvPusherMotorConfig()
        self.geom = LinkageGeometry()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.R_wb = np.eye(3, dtype=np.float64)
        self.angles = np.zeros(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self.pos_ned = np.array([0.0, 0.0, -500.0], dtype=np.float64)
        self.vel_ned = np.array([25.0, 0.0, 0.0], dtype=np.float64)
        self.delta_r_deg = 0.0
        self.delta_l_deg = 0.0
        self.motor_throttle = 0.5
        self.motor_thrust_n = 0.0
        self.motor_rpm = 0.0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)

        self.balls_ned = np.zeros((self.num_balls, 3), dtype=np.float64)
        self.balls_touched = np.zeros(self.num_balls, dtype=bool)
        self.optimal_order: List[int] = list(range(self.num_balls))
        self.optimal_leg_lengths: List[float] = [200.0] * self.num_balls
        self.shortest_total_path_m: float = 1000.0
        self.ideal_min_time_s: float = 15.0
        self.current_tour_ptr: int = 0
        self.leg_start_step: int = 0
        self.balls_reached_count: int = 0
        self.distance_along_tour_covered_m: float = 0.0

        self.step_count = 0
        self.max_steps = max(1200, self.num_balls * 200)
        self.prev_dist_to_active = 0.0
        self.total_path_flown_m = 0.0
        self.total_oscillation_penalty = 0.0

    def _active_ball_idx(self) -> int:
        while self.current_tour_ptr < self.num_balls and self.balls_touched[self.optimal_order[self.current_tour_ptr]]:
            self.current_tour_ptr += 1
        if self.current_tour_ptr < self.num_balls:
            return self.optimal_order[self.current_tour_ptr]
        return self.optimal_order[-1]

    def _next_ball_idx(self) -> int:
        nxt_ptr = self.current_tour_ptr + 1
        while nxt_ptr < self.num_balls and self.balls_touched[self.optimal_order[nxt_ptr]]:
            nxt_ptr += 1
        if nxt_ptr < self.num_balls:
            return self.optimal_order[nxt_ptr]
        return self._active_ball_idx()

    def _compute_dist_to_active(self) -> float:
        idx = self._active_ball_idx()
        return float(np.linalg.norm(self.balls_ned[idx] - self.pos_ned))

    def _get_obs(self) -> np.ndarray:
        p, q, r = self.omega
        act_idx = self._active_ball_idx()
        nxt_idx = self._next_ball_idx()

        active_ned = self.balls_ned[act_idx]
        d_ned = active_ned - self.pos_ned
        dist_3d = max(0.5, float(np.linalg.norm(d_ned)))
        dir_body = (self.R_wb.T @ d_ned) / dist_3d

        # Lookahead vector from active ball to the next ball on the tour
        if nxt_idx != act_idx:
            d_nxt_ned = self.balls_ned[nxt_idx] - active_ned
            dist_nxt = max(0.5, float(np.linalg.norm(d_nxt_ned)))
            dir_nxt_body = (self.R_wb.T @ d_nxt_ned) / dist_nxt
        else:
            dist_nxt = 0.0
            dir_nxt_body = np.zeros(3, dtype=np.float64)

        v_body = self.R_wb.T @ self.vel_ned
        airspeed = float(np.linalg.norm(v_body))
        s_roll, c_roll, s_pitch, c_pitch, s_yaw, c_yaw = compute_inav_circular_attitude(self.R_wb)
        rem_frac = float(self.num_balls - self.balls_reached_count) / float(self.num_balls)

        # Indices 0..14 match the exact 24D observation layout of GlideGpsTargetEnv!
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
                float(v_body[1]) / 50.0,
                float(self.motor_throttle),
                airspeed / self.motor_cfg.max_achievable_speed_ms,
                float(dir_nxt_body[0]),
                float(dir_nxt_body[1]),
                float(dir_nxt_body[2]),
                dist_nxt / 1000.0,
                rem_frac,
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
        deterministic = bool(options.get("deterministic", False))

        if "start_pos_ned" in options:
            self.pos_ned = np.array(options["start_pos_ned"], dtype=np.float64)
        elif self.randomize_start and not deterministic:
            alt0 = float(self.np_random.uniform(200.0, 800.0))
            sx0 = float(self.np_random.uniform(-300.0, 300.0))
            sy0 = float(self.np_random.uniform(-300.0, 300.0))
            self.pos_ned = np.array([sx0, sy0, -alt0], dtype=np.float64)
        else:
            self.pos_ned = np.array([0.0, 0.0, -self.release_altitude_m], dtype=np.float64)

        if "angles_deg" in options:
            r0, p0, y0 = options["angles_deg"]
        elif self.randomize_start and not deterministic:
            r0 = float(self.np_random.uniform(-180.0, 180.0))
            p0 = float(self.np_random.uniform(-65.0, 45.0))
            y0 = float(self.np_random.uniform(-180.0, 180.0))
        else:
            r0, p0, y0 = 0.0, 0.0, 0.0

        self.angles = np.deg2rad([r0, p0, y0])
        self.R_wb = euler_to_rotation_matrix(self.angles[0], self.angles[1], self.angles[2])

        spd0 = float(options.get("speed_ms", float(self.np_random.uniform(15.0, 55.0)) if (self.randomize_start and not deterministic) else 28.0))
        self.vel_ned = self.R_wb @ np.array([spd0, 0.0, 0.0], dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)

        self.delta_r_deg = 0.0
        self.delta_l_deg = 0.0
        self.motor_throttle = 0.6
        self.motor_thrust_n = 0.6 * self.motor_cfg.max_static_thrust_n
        self.motor_rpm = 0.6 * self.motor_cfg.max_rpm
        self.last_action = np.array([0.0, 0.0, 0.2], dtype=np.float32)
        self.prev_action = self.last_action.copy()

        # Spawn N balls completely randomly in 3D space (including altitudes well ABOVE the starting point: up to 1,400 m!)
        if "balls_ned" in options:
            self.balls_ned = np.array(options["balls_ned"], dtype=np.float64)
        elif deterministic:
            base_balls = [
                [160.0, -140.0, -580.0],
                [340.0, 190.0, -420.0],
                [120.0, 380.0, -640.0],
                [-240.0, 220.0, -310.0],
                [-380.0, -180.0, -490.0],
                [-150.0, -420.0, -780.0],
                [280.0, -360.0, -920.0],
                [520.0, -120.0, -1120.0],
                [460.0, 320.0, -860.0],
                [-80.0, 540.0, -1050.0],
                [-460.0, 380.0, -740.0],
                [-580.0, -60.0, -960.0],
                [-320.0, -520.0, -620.0],
                [80.0, -580.0, -360.0],
                [480.0, -440.0, -240.0],
                [640.0, 80.0, -480.0],
                [320.0, 560.0, -720.0],
                [-220.0, 620.0, -1220.0],
                [-620.0, 240.0, -1320.0],
                [-480.0, -380.0, -1150.0],
            ]
            while len(base_balls) < self.num_balls:
                k = len(base_balls)
                ang = k * 2.39996
                rad = 220.0 + (k % 7) * 95.0
                alt = 120.0 + ((k * 173) % 1250)
                base_balls.append([rad * np.cos(ang), rad * np.sin(ang), -alt])
            self.balls_ned = np.array(base_balls[: self.num_balls], dtype=np.float64)
        else:
            balls = []
            for _ in range(self.num_balls):
                bx = float(self.np_random.uniform(-900.0, 900.0))
                by = float(self.np_random.uniform(-900.0, 900.0))
                bz = -float(self.np_random.uniform(60.0, 1400.0))
                balls.append([bx, by, bz])
            self.balls_ned = np.array(balls, dtype=np.float64)

        self.balls_touched = np.zeros(self.num_balls, dtype=bool)
        self.optimal_order, self.shortest_total_path_m, self.optimal_leg_lengths = solve_shortest_3d_ball_path(
            self.pos_ned, self.balls_ned
        )
        self.ideal_min_time_s = max(1.0, self.shortest_total_path_m / self.motor_cfg.max_achievable_speed_ms)
        self.current_tour_ptr = 0
        self.leg_start_step = 0
        self.balls_reached_count = 0
        self.distance_along_tour_covered_m = 0.0

        self.step_count = 0
        self.prev_dist_to_active = self._compute_dist_to_active()
        self.total_path_flown_m = 0.0
        self.total_oscillation_penalty = 0.0

        return self._get_obs(), self.get_info()

    def _step_physics_substep(self, cmd_r_deg: float, cmd_l_deg: float, target_throttle: float, sub_dt: float) -> None:
        alpha_s = sub_dt / (self.cfg.servo_tau_s + sub_dt)
        self.delta_r_deg += alpha_s * (cmd_r_deg - self.delta_r_deg)
        self.delta_l_deg += alpha_s * (cmd_l_deg - self.delta_l_deg)

        alpha_m = sub_dt / (self.motor_cfg.motor_tau_s + sub_dt)
        self.motor_throttle += alpha_m * (target_throttle - self.motor_throttle)
        self.motor_throttle = float(np.clip(self.motor_throttle, 0.0, 1.0))

        alt_m = max(0.0, float(-self.pos_ned[2]))
        rho = float(self.cfg.rho_sea_level * np.exp(-alt_m / 9500.0))

        v_rel_body = self.R_wb.T @ self.vel_ned
        u, v, w = float(v_rel_body[0]), float(v_rel_body[1]), float(v_rel_body[2])
        airspeed = float(np.linalg.norm(v_rel_body))
        q_dyn = 0.5 * rho * (airspeed ** 2)

        # Rear FPV Brushless Pusher Motor + Propeller Thrust Calculation:
        eff_pitch_speed = self.motor_cfg.max_pitch_efflux_ms * np.sqrt(max(0.12, self.motor_throttle))
        advance_factor = float(np.clip(1.0 - max(0.0, u) / eff_pitch_speed, 0.0, 1.25))
        self.motor_thrust_n = self.motor_throttle * self.motor_cfg.max_static_thrust_n * advance_factor
        self.motor_rpm = self.motor_throttle * self.motor_cfg.max_rpm

        # Propeller slipstream induced velocity over rear control surfaces:
        a_disk = np.pi * ((0.5 * self.motor_cfg.prop_diameter_m) ** 2)
        v_wash = float(np.sqrt(max(0.0, u) ** 2 + (2.0 * self.motor_thrust_n) / max(0.2, rho * a_disk)) - max(0.0, u))

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
            # Include 35% of prop-wash velocity over the inner trailing-edge flaps so flaps have authority even at low airspeed!
            ux_loc = u + 0.35 * v_wash - r * r_y
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
        damp_scale = 0.5 * rho * (airspeed + 0.25 * v_wash) * self.cfg.wing_area_m2
        tau_damp_x = -(damp_scale * (b ** 2) * self.cfg.cl_p_damp + self.cfg.viscous_rot_damp) * p
        tau_damp_y = -(damp_scale * (c ** 2) * self.cfg.cm_q_damp * 4.5 + self.cfg.viscous_rot_damp) * q
        tau_damp_z = -(damp_scale * (b ** 2) * self.cfg.cn_r_damp + self.cfg.viscous_rot_damp) * r

        # Total body force includes +self.motor_thrust_n along +x_b (Nose):
        f_body = np.array(
            [
                self.motor_thrust_n + fx_wing + fx_fuse + fx_r + fx_l,
                fy_fin,
                fz_wing + fz_fuse + fz_r + fz_l,
            ],
            dtype=np.float64,
        )
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
        accel_ned = f_world / self.motor_cfg.total_mass_kg + np.array([0.0, 0.0, self.cfg.gravity], dtype=np.float64)
        self.vel_ned += accel_ned * sub_dt
        self.pos_ned += self.vel_ned * sub_dt

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.step_count += 1

        pitch_flap_deg = float(act[0]) * self.max_wing_flap_deg
        roll_flap_deg = float(act[1]) * self.max_wing_flap_deg
        cmd_r_deg = float(np.clip(pitch_flap_deg + roll_flap_deg, -self.max_wing_flap_deg, +self.max_wing_flap_deg))
        cmd_l_deg = float(np.clip(pitch_flap_deg - roll_flap_deg, -self.max_wing_flap_deg, +self.max_wing_flap_deg))
        # Map act[2] in [-1, +1] to FPV motor throttle in [0.0, 1.0]:
        target_throttle = 0.5 * (float(act[2]) + 1.0)

        n_sub = 10
        sub_dt = self.control_dt / n_sub
        prev_pos_step = self.pos_ned.copy()

        balls_newly_touched = 0
        speed_bonus_this_step = 0.0

        for _ in range(n_sub):
            self._step_physics_substep(cmd_r_deg, cmd_l_deg, target_throttle, sub_dt)
            # Check if the drone physically touched any uncollected ball (dist <= ball_touch_radius_m)!
            dists = np.linalg.norm(self.balls_ned - self.pos_ned.reshape(1, 3), axis=1)
            for b_i in range(self.num_balls):
                if not self.balls_touched[b_i] and dists[b_i] <= self.ball_touch_radius_m:
                    self.balls_touched[b_i] = True
                    self.balls_reached_count += 1
                    balls_newly_touched += 1

                    # Calculate speed efficiency for this leg: V_realized / V_max
                    leg_steps = max(1, self.step_count - self.leg_start_step)
                    leg_time_s = leg_steps * self.control_dt
                    leg_idx = min(self.current_tour_ptr, len(self.optimal_leg_lengths) - 1)
                    leg_dist_m = self.optimal_leg_lengths[leg_idx]
                    self.distance_along_tour_covered_m += leg_dist_m
                    v_realized = leg_dist_m / leg_time_s
                    speed_ratio = float(np.clip(v_realized / self.motor_cfg.max_achievable_speed_ms, 0.0, 1.5))
                    speed_bonus_this_step += 1200.0 * speed_ratio

                    self.leg_start_step = self.step_count
                    self._active_ball_idx()
                    self.prev_dist_to_active = self._compute_dist_to_active()

        step_path_len = float(np.linalg.norm(self.pos_ned - prev_pos_step))
        self.total_path_flown_m += step_path_len

        cur_dist = self._compute_dist_to_active()
        dist_dec = self.prev_dist_to_active - cur_dist
        self.prev_dist_to_active = cur_dist

        # Anti-oscillation penalty on control sign-flipping
        da_cur = act[:2] - self.last_action[:2]
        da_prev = self.last_action[:2] - self.prev_action[:2]
        flip_mask = (da_cur * da_prev < -1e-4).astype(np.float32)
        osc_pen = 2.0 * float(np.sum(flip_mask * ((da_cur - da_prev) ** 2)))
        self.total_oscillation_penalty += osc_pen
        self.prev_action = self.last_action.copy()
        self.last_action = act.copy()

        # Reward:
        # 1. Large discrete reward for each ball physically touched (+1500 * escalating multiplier)
        # 2. Speed bonus proportional to V_realized / V_max along the shortest path
        # 3. Continuous progress reward toward the active ball + terminal proximity shaping (< 30m) + 3D velocity alignment
        touch_reward = 0.0
        if balls_newly_touched > 0:
            touch_reward = 1500.0 * balls_newly_touched * (1.0 + 0.25 * self.balls_reached_count)

        # Velocity alignment toward the active ball + closest-approach potential + miss-orbit penalty:
        act_idx = self._active_ball_idx()
        d_to_ball = self.balls_ned[act_idx] - self.pos_ned
        dist_to_ball = max(1.0, float(np.linalg.norm(d_to_ball)))
        unit_to_ball = d_to_ball / dist_to_ball
        closing_vel_ms = float(np.dot(self.vel_ned, unit_to_ball))
        vel_align_bonus = 0.25 * (closing_vel_ms / self.motor_cfg.max_achievable_speed_ms)
        near_weight = 1.0 + 6.0 * float(np.exp(-cur_dist / 30.0))

        # Penalize flying away (dist_dec < 0) when within 80 m of the active ball without touching it!
        miss_orbit_pen = 1.5 * abs(dist_dec) if (cur_dist < 80.0 and dist_dec < -0.1 and balls_newly_touched == 0) else 0.0

        reward = 6.0 * near_weight * dist_dec + vel_align_bonus + touch_reward + speed_bonus_this_step - osc_pen - miss_orbit_pen

        all_balls_finished = bool(self.balls_reached_count >= self.num_balls)
        crashed_ground = bool(-self.pos_ned[2] <= 0.0)

        if all_balls_finished:
            total_time_s = max(0.05, self.step_count * self.control_dt)
            overall_avg_speed_ms = self.shortest_total_path_m / total_time_s
            overall_speed_ratio = float(np.clip(overall_avg_speed_ms / self.motor_cfg.max_achievable_speed_ms, 0.0, 1.5))
            # Huge completion + maximum-speed bonus when all balls are touched!
            reward += 5000.0 + 4000.0 * overall_speed_ratio

        if crashed_ground and not all_balls_finished:
            self.pos_ned[2] = 0.0
            reward -= 500.0

        terminated = all_balls_finished or crashed_ground
        truncated = bool(self.step_count >= self.max_steps)

        return self._get_obs(), float(reward), terminated, truncated, self.get_info()

    def get_info(self) -> Dict[str, Any]:
        roll_deg, pitch_deg, yaw_deg = np.rad2deg(self.angles)
        alt_m = float(max(0.0, -self.pos_ned[2]))
        airspeed = float(np.linalg.norm(self.vel_ned))
        elapsed_s = max(0.05, self.step_count * self.control_dt)
        realized_tour_speed_ms = self.distance_along_tour_covered_m / elapsed_s
        speed_fraction_of_max = realized_tour_speed_ms / self.motor_cfg.max_achievable_speed_ms
        return {
            "step": int(self.step_count),
            "time_s": round(self.step_count * self.control_dt, 2),
            "altitude_m": round(alt_m, 2),
            "pos_x_m": round(float(self.pos_ned[0]), 2),
            "pos_y_m": round(float(self.pos_ned[1]), 2),
            "airspeed_ms": round(airspeed, 2),
            "motor_throttle": round(float(self.motor_throttle), 3),
            "motor_thrust_n": round(float(self.motor_thrust_n), 2),
            "motor_rpm": round(float(self.motor_rpm), 0),
            "balls_reached": int(self.balls_reached_count),
            "num_balls": int(self.num_balls),
            "all_balls_finished": bool(self.balls_reached_count >= self.num_balls),
            "shortest_path_length_m": round(float(self.shortest_total_path_m), 2),
            "ideal_min_time_s": round(float(self.ideal_min_time_s), 2),
            "realized_tour_speed_ms": round(float(realized_tour_speed_ms), 2),
            "speed_fraction_of_max": round(float(speed_fraction_of_max), 3),
            "active_ball_dist_m": round(float(self._compute_dist_to_active()), 2),
            "roll_deg": round(float(roll_deg), 2),
            "pitch_deg": round(float(pitch_deg), 2),
            "yaw_deg": round(float(yaw_deg), 2),
            "delta_r_deg": round(float(self.delta_r_deg), 2),
            "delta_l_deg": round(float(self.delta_l_deg), 2),
            "servo_r_deg": round(wing_flap_deg_to_servo_deg(self.delta_r_deg, self.geom), 2),
            "servo_l_deg": round(wing_flap_deg_to_servo_deg(self.delta_l_deg, self.geom), 2),
        }
