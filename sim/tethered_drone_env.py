from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass
class DroneConfig:
    """Physical parameters for the 20 cm single-propeller TVC rocket-drone."""

    # Geometry & mass (20 cm tall vertical cylinder, 5 cm diameter)
    body_length: float = 0.20          # meters (20 cm)
    body_radius: float = 0.025         # meters (2.5 cm radius -> 5 cm diameter)
    mass: float = 0.180                # kg (180 g compact FPV TVC drone)

    # Principal moments of inertia (kg * m^2) for a 20 cm cylinder + electronics/battery
    ixx: float = 0.00072
    iyy: float = 0.00072
    izz: float = 0.00012

    # Moment arms relative to Center of Mass (COM)
    # Motor + 2-axis servo gimbal is at the bottom of the 20 cm tube (-0.10 m from COM)
    motor_z_offset: float = -0.10      # meters
    # 4-wire suspension collar is slightly below COM so the drone naturally wants to tip
    # over under gravity unless stabilized by the bottom gimbaled propeller
    tether_collar_z: float = -0.022    # meters
    tether_collar_radius: float = 0.032

    # Propulsion & 2-axis servo gimbal
    max_thrust: float = 3.60           # Newtons (approx 2:1 thrust-to-weight ratio)
    hover_throttle: float = 0.49       # Throttle where thrust ~= weight (0.18 * 9.81 / 3.6)
    max_servo_angle_deg: float = 18.0  # Max gimbal tilt in pitch & roll (degrees)
    servo_tau: float = 0.025           # Servo 1st-order response time constant (s)
    motor_tau: float = 0.035           # Motor spool-up response time constant (s)
    max_rpm: float = 28000.0           # Max propeller RPM

    # Single-propeller reaction torque & anti-spin fins (mix of passive trim + wind damping)
    prop_reaction_torque_coeff: float = 0.018   # N*m at full throttle
    max_slipstream_vel: float = 16.0            # m/s prop-wash speed at full throttle
    fin_counter_torque_coeff: float = 0.000039  # Counter-torque from angled fins * v_flow^2 (matches prop torque at hover)
    fin_yaw_damping_coeff: float = 0.00065      # Aerodynamic yaw damping * v_flow * omega_z
    body_rot_damping: float = 0.0012            # Natural air rotational damping

    # 4-wire test rig geometry (meters in world frame)
    rig_half_width: float = 0.42
    rig_top_z: float = 0.82
    suspend_height_z: float = 0.45
    wire_soft_limit_rad: float = 0.68  # ~39 deg where slack wires begin pulling hard
    wire_max_tilt_rad: float = 0.84    # ~48 deg safety wire limit (resets episode)

    # Simulation timing
    dt: float = 0.01                   # 100 Hz physics & IMU control rate
    max_steps: int = 500               # 5.0 seconds per episode
    gravity: float = 9.81

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles (roll, pitch, yaw in radians) to 3x3 rotation matrix R_wb."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


class TetheredDroneEnv(gym.Env):
    """
    Step 1 Environment: 20 cm Single-Propeller TVC Rocket-Drone Suspended by 4 Wires.

    Observation Space (6D IMU Input):
      0: roll          - IMU roll angle (rad, normalized by pi/2)
      1: pitch         - IMU pitch angle (rad, normalized by pi/2)
      2: yaw_rate_err  - IMU yaw drift / heading angle (rad, normalized by pi)
      3: gyro_x        - Gyroscope roll angular rate p (rad/s, scaled by 1/8)
      4: gyro_y        - Gyroscope pitch angular rate q (rad/s, scaled by 1/8)
      5: gyro_z        - Gyroscope yaw angular rate r (rad/s, scaled by 1/8)

    Action Space (3D Continuous in [-1, 1]):
      0: servo_1       - Pitch gimbal servo command (-1 -> -18 deg, +1 -> +18 deg)
      1: servo_2       - Roll gimbal servo command  (-1 -> -18 deg, +1 -> +18 deg)
      2: motor_speed   - Propeller speed command    (-1 -> 0% RPM,   +1 -> 100% RPM)
    """

    metadata = {"render_modes": ["human", "dict"], "render_fps": 100}

    def __init__(
        self,
        config: Optional[DroneConfig] = None,
        initial_tilt_range_deg: float = 28.0,
        domain_randomization: bool = False,
        terminate_on_wire_limit: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = config or DroneConfig()
        self.initial_tilt_range_deg = initial_tilt_range_deg
        self.domain_randomization = domain_randomization
        self.terminate_on_wire_limit = terminate_on_wire_limit
        self.render_mode = render_mode

        # 6D IMU observation: [roll_norm, pitch_norm, yaw_norm, gx_norm, gy_norm, gz_norm]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # 3D Action: [servo_1 (pitch), servo_2 (roll), motor_speed]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # World-frame anchor coordinates of the 4 top rig corners
        w = self.cfg.rig_half_width
        zt = self.cfg.rig_top_z
        self.rig_anchors_world = np.array(
            [
                [w, w, zt],
                [-w, w, zt],
                [-w, -w, zt],
                [w, -w, zt],
            ],
            dtype=np.float64,
        )

        # Local-frame attachment lugs on the drone collar (4 corners around cylinder)
        r_c = self.cfg.tether_collar_radius / np.sqrt(2.0)
        z_c = self.cfg.tether_collar_z
        self.collar_lugs_body = np.array(
            [
                [r_c, r_c, z_c],
                [-r_c, r_c, z_c],
                [-r_c, -r_c, z_c],
                [r_c, -r_c, z_c],
            ],
            dtype=np.float64,
        )

        # Internal state variables
        self.angles = np.zeros(3, dtype=np.float64)     # [roll, pitch, yaw] in rad
        self.omega = np.zeros(3, dtype=np.float64)      # [wx, wy, wz] body gyro in rad/s
        self.servo_angles = np.zeros(2, dtype=np.float64)  # actual [servo_1, servo_2] in rad
        self.motor_throttle = 0.5                       # actual normalized motor speed [0, 1]
        self.wind_vel = np.zeros(3, dtype=np.float64)   # ambient airflow (m/s)
        self.external_torque = np.zeros(3, dtype=np.float64)
        self.wire_tensions = np.zeros(4, dtype=np.float64)
        self.last_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0

    def _get_obs(self) -> np.ndarray:
        roll, pitch, yaw = self.angles
        wx, wy, wz = self.omega
        obs = np.array(
            [
                roll / 0.6,
                pitch / 0.6,
                np.arctan2(np.sin(yaw), np.cos(yaw)) / 1.0,
                wx / 5.0,
                wy / 5.0,
                wz / 5.0,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        tilt_deg = float(options.get("initial_tilt_deg", self.initial_tilt_range_deg))
        tilt_rad = np.deg2rad(tilt_deg)

        if "initial_angles_deg" in options:
            init_a = np.deg2rad(np.asarray(options["initial_angles_deg"], dtype=np.float64))
            self.angles = init_a.copy()
        else:
            # Start from a random tilted orientation and random angular velocity
            roll0 = self.np_random.uniform(-tilt_rad, tilt_rad)
            pitch0 = self.np_random.uniform(-tilt_rad, tilt_rad)
            # Ensure initial tilt is non-trivial when tilt_deg > 5
            if tilt_deg >= 8.0 and np.hypot(roll0, pitch0) < np.deg2rad(6.0):
                angle_dir = self.np_random.uniform(0, 2 * np.pi)
                mag = self.np_random.uniform(np.deg2rad(8.0), tilt_rad)
                roll0 = mag * np.cos(angle_dir)
                pitch0 = mag * np.sin(angle_dir)
            yaw0 = self.np_random.uniform(-0.35, 0.35)
            self.angles = np.array([roll0, pitch0, yaw0], dtype=np.float64)

        if "initial_omega" in options:
            self.omega = np.asarray(options["initial_omega"], dtype=np.float64).copy()
        else:
            self.omega = self.np_random.uniform(-1.5, 1.5, size=3).astype(np.float64)

        self.servo_angles = np.zeros(2, dtype=np.float64)
        self.motor_throttle = float(self.cfg.hover_throttle)
        self.last_action = np.array([0.0, 0.0, 2.0 * self.cfg.hover_throttle - 1.0], dtype=np.float32)
        self.external_torque = np.zeros(3, dtype=np.float64)

        if self.domain_randomization:
            self.wind_vel = self.np_random.uniform(-1.5, 1.5, size=3)
        else:
            self.wind_vel = np.zeros(3, dtype=np.float64)

        self.step_count = 0
        self._update_wire_tensions()

        return self._get_obs(), self.get_telemetry()

    def apply_disturbance_impulse(self, torque_impulse: Tuple[float, float, float]) -> None:
        """Apply an interactive external push/impulse to test recovery in the viewer."""
        ixx, iyy, izz = self.cfg.ixx, self.cfg.iyy, self.cfg.izz
        imp = np.asarray(torque_impulse, dtype=np.float64)
        self.omega += imp / np.array([ixx, iyy, izz])

    def _update_wire_tensions(self) -> np.ndarray:
        """Compute realistic tension in each of the 4 suspension wires."""
        roll, pitch, yaw = self.angles
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        com_world = np.array([0.0, 0.0, self.cfg.suspend_height_z])

        # Base vertical load carried by the 4 wires when thrust < weight
        net_downward_force = max(0.0, self.cfg.mass * self.cfg.gravity - self.motor_throttle * self.cfg.max_thrust * np.cos(roll) * np.cos(pitch))
        base_wire_tension = 0.45 + 0.35 * net_downward_force

        # Compute tension in each of the 4 suspension wires based on tilt direction & load
        tensions = np.zeros(4, dtype=np.float64)
        R_pr = euler_to_rotation_matrix(roll, pitch, 0.0)
        for i in range(4):
            lug_world = com_world + R @ self.collar_lugs_body[i]
            # Stretch comes primarily from pitch/roll tilting against the wire corner
            lug_pr = com_world + R_pr @ self.collar_lugs_body[i]
            dist_pr = float(np.linalg.norm(self.rig_anchors_world[i] - lug_pr))
            lug_upright = com_world + self.collar_lugs_body[i]
            nom_dist = float(np.linalg.norm(self.rig_anchors_world[i] - lug_upright))
            stretch = max(0.0, dist_pr - nom_dist)
            tensions[i] = base_wire_tension + 95.0 * stretch
        self.wire_tensions = tensions
        return tensions

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.step_count += 1

        max_servo_rad = np.deg2rad(self.cfg.max_servo_angle_deg)
        cmd_servo_1 = act[0] * max_servo_rad          # Pitch gimbal target (rad)
        cmd_servo_2 = act[1] * max_servo_rad          # Roll gimbal target (rad)
        cmd_throttle = 0.5 * (act[2] + 1.0)           # Map [-1, 1] -> [0, 1]

        # Sub-step physics integration (2 substeps of 5ms for crisp rotational accuracy)
        n_substeps = 2
        dt_sub = self.cfg.dt / n_substeps

        for _ in range(n_substeps):
            # 1. Actuator first-order dynamics (2 servos + 1 brushless motor)
            alpha_s = dt_sub / (self.cfg.servo_tau + dt_sub)
            alpha_m = dt_sub / (self.cfg.motor_tau + dt_sub)
            self.servo_angles[0] += alpha_s * (cmd_servo_1 - self.servo_angles[0])
            self.servo_angles[1] += alpha_s * (cmd_servo_2 - self.servo_angles[1])
            self.motor_throttle += alpha_m * (cmd_throttle - self.motor_throttle)

            s1_pitch, s2_roll = self.servo_angles
            throttle = np.clip(self.motor_throttle, 0.0, 1.0)

            # 2. Thrust vector in body frame from bottom gimbaled propeller
            thrust_mag = throttle * self.cfg.max_thrust
            # Tilting servo_1 (pitch) deflects thrust in body X; servo_2 (roll) deflects in body Y
            fx_b = -thrust_mag * np.sin(s1_pitch) * np.cos(s2_roll)
            fy_b = thrust_mag * np.sin(s2_roll) * np.cos(s1_pitch)
            fz_b = thrust_mag * np.cos(s1_pitch) * np.cos(s2_roll)
            f_thrust_body = np.array([fx_b, fy_b, fz_b], dtype=np.float64)

            # Moment arm from tether pivot / COM to bottom motor
            r_motor = np.array([0.0, 0.0, self.cfg.motor_z_offset], dtype=np.float64)
            tau_tvc = np.cross(r_motor, f_thrust_body)
            # Explicit verification of signs:
            # r_motor = [0, 0, -0.10]
            # cross([0, 0, -0.10], [fx, fy, fz]) = [0.10 * fy, -0.10 * fx, 0]
            # So positive s2_roll -> fy_b > 0 -> positive roll torque tau_x
            # Positive s1_pitch -> fx_b < 0 -> positive pitch torque tau_y

            # 3. Gravity toppling torque (inverted pendulum about the 4-wire suspension collar)
            roll, pitch, yaw = self.angles
            # Since the suspension collar is at z = -0.022 m below COM, the COM is +0.022 m above
            # the pivot, creating a toppling torque when tilted:
            com_above_collar = abs(self.cfg.tether_collar_z)
            tau_grav_x = self.cfg.mass * self.cfg.gravity * com_above_collar * np.sin(roll) * np.cos(pitch)
            tau_grav_y = self.cfg.mass * self.cfg.gravity * com_above_collar * np.sin(pitch) * np.cos(roll)
            tau_grav = np.array([tau_grav_x, tau_grav_y, 0.0], dtype=np.float64)

            # 4. Single-propeller reaction torque + 4 anti-spin fins in prop-wash & wind
            # Prop spins CW -> exerts CCW reaction torque (+z) on body proportional to throttle^2
            tau_prop_yaw = self.cfg.prop_reaction_torque_coeff * (throttle ** 2)

            # Airflow speed over the bottom fins = propeller slipstream + ambient axial wind
            v_slipstream = self.cfg.max_slipstream_vel * np.sqrt(max(throttle, 0.001))
            v_ambient = float(np.linalg.norm(self.wind_vel))
            v_flow = v_slipstream + v_ambient

            # Passive cambered fins reject spin at speed (proportional to v_flow^2)
            # plus aerodynamic yaw rate damping (proportional to v_flow * omega_z)
            wx, wy, wz = self.omega
            tau_fin_yaw = (
                -self.cfg.fin_counter_torque_coeff * (v_flow ** 2)
                - self.cfg.fin_yaw_damping_coeff * v_flow * wz
            )

            # Slight yaw restoring torque from the 4 crossed suspension wires when twisted
            tau_wire_yaw = -0.0035 * np.sin(yaw)

            # Soft wire elastic restoring torque when approaching wire slack limit
            tilt_mag = np.hypot(roll, pitch)
            tau_wire_limit = np.zeros(2, dtype=np.float64)
            if tilt_mag > self.cfg.wire_soft_limit_rad:
                excess = tilt_mag - self.cfg.wire_soft_limit_rad
                tau_wire_limit[0] = -2.2 * excess * (roll / (tilt_mag + 1e-6)) - 0.015 * wx
                tau_wire_limit[1] = -2.2 * excess * (pitch / (tilt_mag + 1e-6)) - 0.015 * wy

            # Aerodynamic pitch/roll damping from fins in slipstream
            aero_pr_damp = (self.cfg.body_rot_damping + 0.00018 * v_flow) * np.array([wx, wy])

            # Total torque in body frame
            tau_x = tau_tvc[0] + tau_grav[0] + tau_wire_limit[0] - aero_pr_damp[0] + self.external_torque[0]
            tau_y = tau_tvc[1] + tau_grav[1] + tau_wire_limit[1] - aero_pr_damp[1] + self.external_torque[1]
            tau_z = tau_prop_yaw + tau_fin_yaw + tau_wire_yaw - 0.0004 * wz + self.external_torque[2]

            # 5. Rigid-body Euler rotational equations: I * d(omega)/dt + omega x (I * omega) = tau
            ixx, iyy, izz = self.cfg.ixx, self.cfg.iyy, self.cfg.izz
            dwx = (tau_x - (izz - iyy) * wy * wz) / ixx
            dwy = (tau_y - (ixx - izz) * wx * wz) / iyy
            dwz = (tau_z - (iyy - ixx) * wx * wy) / izz

            self.omega[0] += dwx * dt_sub
            self.omega[1] += dwy * dt_sub
            self.omega[2] += dwz * dt_sub
            self.omega = np.clip(self.omega, -25.0, 25.0)

            # 6. Integrate Euler angles from body angular rates
            wx, wy, wz = self.omega
            cr, sr = np.cos(roll), np.sin(roll)
            cp = np.clip(np.cos(pitch), 0.15, 1.0)
            tp = np.sin(pitch) / cp

            droll = wx + (wy * sr + wz * cr) * tp
            dpitch = wy * cr - wz * sr
            dyaw = (wy * sr + wz * cr) / cp

            self.angles[0] += droll * dt_sub
            self.angles[1] += dpitch * dt_sub
            self.angles[2] = np.arctan2(
                np.sin(self.angles[2] + dyaw * dt_sub),
                np.cos(self.angles[2] + dyaw * dt_sub),
            )

        # Decay any temporary external disturbance torque
        self.external_torque *= 0.85
        self._update_wire_tensions()

        # Compute RL reward for Step 1: upright stabilization + spin suppression + smooth servos
        roll, pitch, yaw = self.angles
        wx, wy, wz = self.omega
        tilt_rad = float(np.hypot(roll, pitch))

        # Alive bonus + upright precision shaping (never profitable to hit the safety wires)
        tilt_sq = float(roll ** 2 + pitch ** 2)
        alive_bonus = 1.0
        upright_bonus = (
            1.2 * float(np.exp(-8.0 * tilt_sq))
            + 0.8 * float(np.exp(-50.0 * tilt_sq))
            + 0.2 * float(np.exp(-8.0 * (yaw ** 2)))
        )
        angle_penalty = 1.5 * tilt_rad + 0.8 * tilt_sq + 0.15 * abs(float(yaw))
        gyro_penalty = 0.015 * float(wx ** 2 + wy ** 2) + 0.010 * float(wz ** 2)
        throttle_target_penalty = 0.20 * float((self.motor_throttle - 0.57) ** 2)
        action_slew = float(np.sum((act - self.last_action) ** 2))
        servo_effort = 0.015 * float(act[0] ** 2 + act[1] ** 2) + 0.015 * action_slew

        reward = (
            alive_bonus
            + upright_bonus
            - angle_penalty
            - gyro_penalty
            - throttle_target_penalty
            - servo_effort
        )

        hit_wire_limit = bool(tilt_rad > self.cfg.wire_max_tilt_rad)
        terminated = bool(hit_wire_limit and self.terminate_on_wire_limit)
        if hit_wire_limit:
            reward -= 5.0

        truncated = bool(self.step_count >= self.cfg.max_steps)
        self.last_action = act.astype(np.float32)

        return self._get_obs(), float(reward), terminated, truncated, self.get_telemetry()

    def get_telemetry(self) -> Dict[str, Any]:
        """Return rich 3D state & telemetry for the interactive visual simulator."""
        roll, pitch, yaw = self.angles
        wx, wy, wz = self.omega
        s1_pitch, s2_roll = self.servo_angles
        throttle = float(self.motor_throttle)
        v_slipstream = self.cfg.max_slipstream_vel * np.sqrt(max(throttle, 0.001))

        R = euler_to_rotation_matrix(roll, pitch, yaw)
        com_world = np.array([0.0, 0.0, self.cfg.suspend_height_z])
        lugs_world = [
            (com_world + R @ self.collar_lugs_body[i]).round(4).tolist()
            for i in range(4)
        ]

        return {
            "step": int(self.step_count),
            "time_s": round(self.step_count * self.cfg.dt, 3),
            "roll_deg": round(float(np.rad2deg(roll)), 2),
            "pitch_deg": round(float(np.rad2deg(pitch)), 2),
            "yaw_deg": round(float(np.rad2deg(yaw)), 2),
            "tilt_deg": round(float(np.rad2deg(np.hypot(roll, pitch))), 2),
            "gyro_xyz_rads": [round(float(wx), 3), round(float(wy), 3), round(float(wz), 3)],
            "servo_1_deg": round(float(np.rad2deg(s1_pitch)), 2),
            "servo_2_deg": round(float(np.rad2deg(s2_roll)), 2),
            "motor_throttle_pct": round(throttle * 100.0, 1),
            "motor_rpm": round(throttle * self.cfg.max_rpm, 0),
            "thrust_n": round(throttle * self.cfg.max_thrust, 3),
            "fin_airflow_ms": round(float(v_slipstream + np.linalg.norm(self.wind_vel)), 2),
            "wire_tensions_n": [round(float(t), 2) for t in self.wire_tensions],
            "com_world": com_world.tolist(),
            "lugs_world": lugs_world,
            "rig_anchors_world": self.rig_anchors_world.tolist(),
        }
