from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from sim.wing_linkage import (
    LinkageGeometry,
    MAX_WING_FLAP_DEG,
    get_servo_range_for_wing_limit,
    servo_deg_to_wing_flap_deg,
    wing_flap_deg_to_servo_deg,
)

SIM_MODE_WIND_TUNNEL = "wind_tunnel"
SIM_MODE_FREE_DROP = "free_drop"

ROOT_DIR = Path(__file__).resolve().parent.parent
STL_DIR = ROOT_DIR / "FINAL FOR PRINT"


@dataclass
class AirframeAeroConfig:
    """
    Physical & aerodynamic parameters of the 300 g delta-wing airframe (`FINAL FOR PRINT/*.STL`),
    backed by the MuJoCo rigid-body & ellipsoid fluid physics engine (`mujoco`).
    """

    # Mass & geometry from CAD (`BODY.STL` + `HEAD.STL` + `WING FLAP.STL`)
    mass_kg: float = 0.300                 # 300 g total mass
    total_length_m: float = 0.1563         # 156.3 mm (91.3 mm BODY + 65.0 mm HEAD)
    fuselage_diameter_m: float = 0.0432    # 43.2 mm outer cylinder diameter
    wingspan_m: float = 0.2802             # 280.2 mm full wingspan
    mean_aero_chord_m: float = 0.0420      # 42.0 mm mean aerodynamic chord

    # Center of Mass (CoM) & Aerodynamic Centers of Pressure (measured from tail base Y = 0)
    com_from_tail_m: float = 0.0913        # 91.3 mm from tail (at HEAD.STL / BODY.STL junction)
    wing_cop_from_tail_m: float = 0.0280   # 28.0 mm from tail -> 63.3 mm BEHIND CoM
    fin_cop_from_tail_m: float = 0.0280    # 28.0 mm from tail -> 63.3 mm BEHIND CoM
    flap_cop_from_tail_m: float = 0.0223   # 22.3 mm from tail -> 69.0 mm BEHIND CoM
    flap_cop_span_y_m: float = 0.0711      # ±71.1 mm spanwise CoP of each 87.6 mm flap

    # Reference areas (m^2)
    wing_area_m2: float = 0.0112           # Swept delta planform area
    fin_area_m2: float = 0.0045            # Vertical tail fin + 2 wingtip winglets
    flap_area_each_m2: float = 0.0017695   # 87.6 mm x 20.2 mm per trailing-edge elevon flap

    # Principal moments of inertia about CoM (kg * m^2) for 300 g airframe
    ixx: float = 0.00028                   # Roll inertia about longitudinal axis
    iyy: float = 0.00068                   # Pitch inertia about lateral wing axis
    izz: float = 0.00082                   # Yaw inertia about normal axis

    # Aerodynamic coefficients (used alongside MuJoCo fluid solver for telemetry breakdown)
    cl_alpha_wing: float = 3.50
    cl_vortex_wing: float = 1.40
    cd0_body_wing: float = 0.145
    k_induced_wing: float = 0.38
    cy_beta_fin: float = -2.80
    cl_delta_flap: float = 2.60
    cd_delta_flap: float = 1.45
    cm_delta_flap: float = 1.85

    cl_p_damp: float = 0.52
    cm_q_damp: float = 3.60
    cn_r_damp: float = 1.35
    viscous_rot_damp: float = 0.0012

    max_wing_flap_deg: float = MAX_WING_FLAP_DEG
    servo_tau_s: float = 0.022

    gravity: float = 9.81
    rho_sea_level: float = 1.225
    kinematic_viscosity: float = 1.48e-5
    speed_of_sound: float = 340.3
    dt: float = 0.01

    @property
    def wing_cop_x_body_m(self) -> float:
        return self.wing_cop_from_tail_m - self.com_from_tail_m

    @property
    def fin_cop_x_body_m(self) -> float:
        return self.fin_cop_from_tail_m - self.com_from_tail_m

    @property
    def flap_cop_x_body_m(self) -> float:
        return self.flap_cop_from_tail_m - self.com_from_tail_m

    @property
    def flap_cop_y_body_m(self) -> float:
        return self.flap_cop_span_y_m


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles (roll, pitch, yaw in rad) to 3x3 rotation matrix R_wb."""
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


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Extract ZYX Euler angles [roll, pitch, yaw] (in rad) from 3x3 rotation matrix R_wb."""
    sp = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sp))
    if abs(sp) < 0.99999:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rotation_matrix_to_quaternion(R: np.ndarray) -> List[float]:
    """Convert 3x3 rotation matrix R_wb to unit quaternion [w, x, y, z]."""
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return [round(float(v), 5) for v in quat]


def integrate_rotation_so3(R: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """Exact Rodrigues update on SO(3): R_new = R * exp([omega * dt]_x), gimbal-lock free."""
    dtheta = np.asarray(omega_body, dtype=np.float64) * dt
    angle = float(np.linalg.norm(dtheta))
    if angle < 1e-10:
        K = np.array(
            [
                [0.0, -dtheta[2], dtheta[1]],
                [dtheta[2], 0.0, -dtheta[0]],
                [-dtheta[1], dtheta[0], 0.0],
            ],
            dtype=np.float64,
        )
        R_next = R @ (np.eye(3) + K)
    else:
        axis = dtheta / angle
        kx, ky, kz = axis
        K = np.array(
            [
                [0.0, -kz, ky],
                [kz, 0.0, -kx],
                [-ky, kx, 0.0],
            ],
            dtype=np.float64,
        )
        dR = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        R_next = R @ dR

    c0 = R_next[:, 0]
    c0 = c0 / max(1e-12, float(np.linalg.norm(c0)))
    c1 = R_next[:, 1] - np.dot(R_next[:, 1], c0) * c0
    c1 = c1 / max(1e-12, float(np.linalg.norm(c1)))
    c2 = np.cross(c0, c1)
    return np.column_stack((c0, c1, c2))


def build_mujoco_xml(cfg: AirframeAeroConfig, mode: str) -> str:
    """
    Build a real MuJoCo MJCF model for the 300 g delta-wing airframe with:
    - Real air density (`density="1.225"`) and viscosity (`viscosity="1.8e-5"`)
    - Exact 300 g mass and inertia tensor (`0.00028, 0.00068, 0.00082`) at the CoM (`91.3 mm` from tail)
    - Real hinge joints for the Right Wing Flap (`flap_r`) and Left Wing Flap (`flap_l`)
    - Ellipsoid fluid-interaction geoms for the delta wings, vertical tail fin, winglets, and both flaps
    """
    gravity_str = "0 0 0" if mode == SIM_MODE_WIND_TUNNEL else f"0 0 -{cfg.gravity}"
    joint_block = (
        '<joint name="root_ball" type="ball" pos="0 0 0" damping="0.0012"/>'
        if mode == SIM_MODE_WIND_TUNNEL
        else '<freejoint name="root_free"/>'
    )
    return f"""
<mujoco model="glide_munition_300g">
  <option timestep="0.0025" density="{cfg.rho_sea_level}" viscosity="1.8e-5" gravity="{gravity_str}" integrator="RK4"/>
  <worldbody>
    <light pos="0 0 500" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="ground" type="plane" pos="0 0 0" size="2000 2000 0.1" rgba="0.12 0.16 0.24 1" contype="0" conaffinity="0"/>

    <body name="airframe" pos="0 0 200">
      {joint_block}
      <inertial pos="0 0 0" mass="{cfg.mass_kg}" diaginertia="{cfg.ixx} {cfg.iyy} {cfg.izz}"/>

      <!-- Fuselage cylinder (156.3 mm total length, CoM at 91.3 mm from tail) -->
      <geom name="fuselage" type="capsule" fromto="-0.0913 0 0 0.0650 0 0" size="0.0216"
            mass="0" contype="0" conaffinity="0" fluidshape="ellipsoid" rgba="0.8 0.85 0.9 0.5"/>

      <!-- Swept Delta Wings (CoP 63.3 mm behind CoM) -->
      <geom name="wing_right" type="box" pos="-0.0633 -0.065 0" size="0.028 0.055 0.002"
            mass="0" contype="0" conaffinity="0" fluidshape="ellipsoid" rgba="0.6 0.7 0.8 0.5"/>
      <geom name="wing_left" type="box" pos="-0.0633 0.065 0" size="0.028 0.055 0.002"
            mass="0" contype="0" conaffinity="0" fluidshape="ellipsoid" rgba="0.6 0.7 0.8 0.5"/>

      <!-- Vertical Tail Fin & Wingtip Winglets (CoP 63.3 mm behind CoM) -->
      <geom name="tail_fin" type="box" pos="-0.0633 0 0.025" size="0.026 0.002 0.024"
            mass="0" contype="0" conaffinity="0" fluidshape="ellipsoid" rgba="0.6 0.7 0.8 0.5"/>

      <!-- Right Wing Flap (Hinged at x=-0.0700 m, y=-0.0711 m) -->
      <body name="flap_r_body" pos="-0.0690 -0.0711 0">
        <joint name="flap_r_joint" type="hinge" axis="0 1 0" range="-55 55" damping="0.01"/>
        <inertial pos="-0.010 0 0" mass="0.005" diaginertia="1e-6 1e-6 1e-6"/>
        <geom name="flap_r_geom" type="box" pos="-0.010 0 0" size="0.0101 0.0438 0.0018"
              mass="0" contype="0" conaffinity="0" fluidshape="ellipsoid" rgba="0.13 0.77 0.37 1"/>
      </body>

      <!-- Left Wing Flap (Hinged at x=-0.0700 m, y=+0.0711 m) -->
      <body name="flap_l_body" pos="-0.0690 0.0711 0">
        <joint name="flap_l_joint" type="hinge" axis="0 1 0" range="-55 55" damping="0.01"/>
        <inertial pos="-0.010 0 0" mass="0.005" diaginertia="1e-6 1e-6 1e-6"/>
        <geom name="flap_l_geom" type="box" pos="-0.010 0 0" size="0.0101 0.0438 0.0018"
              mass="0" contype="0" conaffinity="0" fluidshape="ellipsoid" rgba="0.13 0.77 0.37 1"/>
      </body>

      <site name="imu_site" pos="0 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <gyro name="imu_gyro" site="imu_site"/>
    <accelerometer name="imu_acc" site="imu_site"/>
  </sensor>
</mujoco>
"""


class AeroGlideSimulator:
    """
    Simulation 2: 3D Aerodynamic & Free-Drop Flight Dynamics Simulator for the 300 g CAD Airframe,
    powered by MuJoCo's rigid-body $SO(3)$ & aerodynamic solver.
    """

    def __init__(
        self,
        config: Optional[AirframeAeroConfig] = None,
        linkage_geom: Optional[LinkageGeometry] = None,
        mode: str = SIM_MODE_WIND_TUNNEL,
    ) -> None:
        self.cfg = config or AirframeAeroConfig()
        self.geom = linkage_geom or LinkageGeometry()
        self.mode = mode if mode in (SIM_MODE_WIND_TUNNEL, SIM_MODE_FREE_DROP) else SIM_MODE_WIND_TUNNEL

        # Compile MuJoCo models for both wind_tunnel and free_drop modes
        self._mj_models = {
            SIM_MODE_WIND_TUNNEL: mujoco.MjModel.from_xml_string(
                build_mujoco_xml(self.cfg, SIM_MODE_WIND_TUNNEL)
            ),
            SIM_MODE_FREE_DROP: mujoco.MjModel.from_xml_string(
                build_mujoco_xml(self.cfg, SIM_MODE_FREE_DROP)
            ),
        }
        self._mj_data = {
            m: mujoco.MjData(model) for m, model in self._mj_models.items()
        }

        self.r_wing_body = np.array([self.cfg.wing_cop_x_body_m, 0.0, 0.0], dtype=np.float64)
        self.r_fin_body = np.array([self.cfg.fin_cop_x_body_m, 0.0, 0.0], dtype=np.float64)
        self.r_flap_r_body = np.array(
            [self.cfg.flap_cop_x_body_m, +self.cfg.flap_cop_y_body_m, 0.0], dtype=np.float64
        )
        self.r_flap_l_body = np.array(
            [self.cfg.flap_cop_x_body_m, -self.cfg.flap_cop_y_body_m, 0.0], dtype=np.float64
        )

        self.R_wb = np.eye(3, dtype=np.float64)
        self.angles = np.zeros(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self.pos_ned = np.zeros(3, dtype=np.float64)
        self.vel_ned = np.zeros(3, dtype=np.float64)

        self.tunnel_airspeed_ms: float = 28.0
        self.ambient_wind_ned = np.zeros(3, dtype=np.float64)
        self.initial_drop_altitude_m: float = 200.0

        self.cmd_delta_r_deg: float = 0.0
        self.cmd_delta_l_deg: float = 0.0
        self.delta_r_deg: float = 0.0
        self.delta_l_deg: float = 0.0
        self.servo_r_deg: float = 0.0
        self.servo_l_deg: float = 0.0

        self.step_count: int = 0
        self.time_s: float = 0.0
        self.trail_world: List[List[float]] = []
        self.last_aero: Dict[str, Any] = {}

        self.reset()

    def _sync_to_mujoco(self) -> None:
        """Synchronize current state into the active MuJoCo MjData structure."""
        model = self._mj_models[self.mode]
        data = self._mj_data[self.mode]
        mujoco.mj_resetData(model, data)
        quat = rotation_matrix_to_quaternion(self.R_wb)
        if self.mode == SIM_MODE_WIND_TUNNEL:
            data.qpos[0:4] = quat
            data.qvel[0:3] = self.omega
        else:
            # Map NED [north, east, down] to MuJoCo Z-up [north, -east, -down]
            data.qpos[0:3] = [self.pos_ned[0], -self.pos_ned[1], -self.pos_ned[2]]
            data.qpos[3:7] = quat
            data.qvel[0:3] = [self.vel_ned[0], -self.vel_ned[1], -self.vel_ned[2]]
            data.qvel[3:6] = self.omega
        mujoco.mj_forward(model, data)

    def reset(
        self,
        *,
        mode: Optional[str] = None,
        initial_angles_deg: Optional[Tuple[float, float, float] | List[float]] = None,
        initial_omega_rads: Optional[Tuple[float, float, float] | List[float]] = None,
        initial_altitude_m: Optional[float] = None,
        initial_velocity_ned_ms: Optional[Tuple[float, float, float] | List[float]] = None,
        tunnel_airspeed_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        if mode is not None and mode in (SIM_MODE_WIND_TUNNEL, SIM_MODE_FREE_DROP):
            self.mode = mode
        if tunnel_airspeed_ms is not None:
            self.tunnel_airspeed_ms = float(np.clip(tunnel_airspeed_ms, 0.0, 100.0))
        if initial_altitude_m is not None:
            self.initial_drop_altitude_m = float(max(1.0, initial_altitude_m))

        if initial_angles_deg is not None:
            ang_deg = np.asarray(initial_angles_deg, dtype=np.float64)
        elif self.mode == SIM_MODE_WIND_TUNNEL:
            ang_deg = np.array([0.0, 12.0, -8.0], dtype=np.float64)
        else:
            # Held upside-down in the air with the nose pointing straight down (-90 deg pitch)
            ang_deg = np.array([0.0, -90.0, 0.0], dtype=np.float64)

        self.angles = np.deg2rad(ang_deg)
        self.R_wb = euler_to_rotation_matrix(self.angles[0], self.angles[1], self.angles[2])

        if initial_omega_rads is not None:
            self.omega = np.asarray(initial_omega_rads, dtype=np.float64).copy()
        else:
            self.omega = np.zeros(3, dtype=np.float64)

        if self.mode == SIM_MODE_WIND_TUNNEL:
            self.pos_ned = np.zeros(3, dtype=np.float64)
            self.vel_ned = np.array([self.tunnel_airspeed_ms, 0.0, 0.0], dtype=np.float64)
        else:
            alt = self.initial_drop_altitude_m
            self.pos_ned = np.array([0.0, 0.0, -alt], dtype=np.float64)
            if initial_velocity_ned_ms is not None:
                self.vel_ned = np.asarray(initial_velocity_ned_ms, dtype=np.float64).copy()
            else:
                self.vel_ned = np.array([0.0, 0.0, 0.1], dtype=np.float64)

        self.step_count = 0
        self.time_s = 0.0
        self.trail_world = [[0.0, float(-self.pos_ned[2]), 0.0]]
        self._sync_to_mujoco()
        self.last_aero = self.compute_aerodynamics()
        return self.get_telemetry()

    def set_flaps_deg(
        self,
        delta_r_deg: float,
        delta_l_deg: float,
        immediate: bool = False,
    ) -> None:
        """
        Set target aerodynamic wing flap deflections (deg) within [-MAX_WING_FLAP_DEG, +MAX_WING_FLAP_DEG].
        Since we swapped one motor direction in software so BOTH servos act identically:
          - Positive servo command on both servos = symmetric pitch deflection (`delta_r == delta_l`)
          - Opposite servo commands = differential roll deflection (`delta_r == -delta_l`)
        """
        lim = self.cfg.max_wing_flap_deg
        self.cmd_delta_r_deg = float(np.clip(delta_r_deg, -lim, +lim))
        self.cmd_delta_l_deg = float(np.clip(delta_l_deg, -lim, +lim))
        if immediate:
            self.delta_r_deg = self.cmd_delta_r_deg
            self.delta_l_deg = self.cmd_delta_l_deg
            # Both servos use the exact same curve direction (already swapped in firmware)
            self.servo_r_deg = wing_flap_deg_to_servo_deg(self.delta_r_deg, self.geom)
            self.servo_l_deg = wing_flap_deg_to_servo_deg(self.delta_l_deg, self.geom)
            self.last_aero = self.compute_aerodynamics()

    def set_pitch_roll_mix(
        self,
        pitch_flap_deg: float,
        roll_flap_deg: float,
        immediate: bool = False,
    ) -> None:
        """Set symmetric pitch mix (deg) and differential roll mix (deg) on the wing flaps."""
        self.set_flaps_deg(
            delta_r_deg=pitch_flap_deg + roll_flap_deg,
            delta_l_deg=pitch_flap_deg - roll_flap_deg,
            immediate=immediate,
        )

    def set_servos_deg(
        self,
        servo_r_deg: float,
        servo_l_deg: float,
        immediate: bool = False,
    ) -> None:
        """
        Set right and left servo angles (deg from Center Wings).
        Both servos follow the exact same positive/negative convention (`[-49.20, +53.11]` deg -> `[-50, +50]` deg flap).
        """
        s_min, s_max = get_servo_range_for_wing_limit(self.cfg.max_wing_flap_deg, self.geom)
        sr = float(np.clip(servo_r_deg, s_min, s_max))
        sl = float(np.clip(servo_l_deg, s_min, s_max))
        flap_r = servo_deg_to_wing_flap_deg(sr, self.geom)
        flap_l = servo_deg_to_wing_flap_deg(sl, self.geom)
        self.set_flaps_deg(delta_r_deg=flap_r, delta_l_deg=flap_l, immediate=immediate)

    def set_tunnel_airspeed(self, airspeed_ms: float) -> None:
        self.tunnel_airspeed_ms = float(np.clip(airspeed_ms, 0.0, 100.0))
        if self.mode == SIM_MODE_WIND_TUNNEL:
            self.vel_ned = np.array([self.tunnel_airspeed_ms, 0.0, 0.0], dtype=np.float64)
            self.last_aero = self.compute_aerodynamics()

    def _air_density(self, altitude_m: float) -> float:
        h = max(0.0, float(altitude_m))
        return float(self.cfg.rho_sea_level * np.exp(-h / 9500.0))

    def compute_aerodynamics(
        self,
        R_wb: Optional[np.ndarray] = None,
        vel_ned: Optional[np.ndarray] = None,
        omega_body: Optional[np.ndarray] = None,
        delta_r_deg: Optional[float] = None,
        delta_l_deg: Optional[float] = None,
        altitude_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        R = self.R_wb if R_wb is None else R_wb
        v_ned = (
            np.array([self.tunnel_airspeed_ms, 0.0, 0.0], dtype=np.float64)
            if (vel_ned is None and self.mode == SIM_MODE_WIND_TUNNEL)
            else (self.vel_ned if vel_ned is None else vel_ned)
        )
        omega = self.omega if omega_body is None else omega_body
        dr_deg = self.delta_r_deg if delta_r_deg is None else float(delta_r_deg)
        dl_deg = self.delta_l_deg if delta_l_deg is None else float(delta_l_deg)
        alt_m = float(-self.pos_ned[2] if altitude_m is None else altitude_m)

        rho = self._air_density(alt_m)
        v_rel_ned = v_ned - self.ambient_wind_ned
        v_rel_body = R.T @ v_rel_ned
        u, v, w = float(v_rel_body[0]), float(v_rel_body[1]), float(v_rel_body[2])
        airspeed = float(np.linalg.norm(v_rel_body))
        q_dyn = 0.5 * rho * (airspeed ** 2)

        if airspeed > 1e-4:
            alpha = float(np.arctan2(w, u))
            beta = float(np.arcsin(np.clip(v / airspeed, -1.0, 1.0)))
            wind_hat_b = v_rel_body / airspeed
        else:
            alpha = 0.0
            beta = 0.0
            wind_hat_b = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        sa, ca = np.sin(alpha), np.cos(alpha)
        sb, cb = np.sin(beta), np.cos(beta)

        cl_wing = (
            self.cfg.cl_alpha_wing * sa * ca
            + self.cfg.cl_vortex_wing * (sa ** 2) * ca * np.sign(sa)
        )
        cd_wing = self.cfg.cd0_body_wing + 1.35 * (sa ** 2) + 0.35 * (sb ** 2)

        lift_wing_n = q_dyn * self.cfg.wing_area_m2 * cl_wing
        lift_fuse_n = q_dyn * self.cfg.wing_area_m2 * (0.95 * sa * ca)
        drag_wing_n = q_dyn * self.cfg.wing_area_m2 * cd_wing

        f_wing_body = np.array(
            [-drag_wing_n * ca + lift_wing_n * sa, 0.0, -drag_wing_n * sa - lift_wing_n * ca],
            dtype=np.float64,
        )
        f_fuse_body = np.array(
            [lift_fuse_n * sa, 0.0, -lift_fuse_n * ca],
            dtype=np.float64,
        )
        tau_wing = np.cross(self.r_wing_body, f_wing_body)

        cy_fin = self.cfg.cy_beta_fin * sb * cb
        f_fin_body = np.array([0.0, q_dyn * self.cfg.fin_area_m2 * cy_fin, 0.0], dtype=np.float64)
        tau_fin = np.cross(self.r_fin_body, f_fin_body)

        def _flap_force_and_moment(
            r_flap: np.ndarray, flap_deg: float
        ) -> Tuple[np.ndarray, np.ndarray]:
            v_local = v_rel_body + np.cross(omega, r_flap)
            v_loc_mag = float(np.linalg.norm(v_local))
            q_loc = 0.5 * rho * (v_loc_mag ** 2)
            alpha_loc = float(np.arctan2(v_local[2], v_local[0])) if v_loc_mag > 1e-4 else 0.0
            delta_rad = np.deg2rad(flap_deg)
            eff_angle = delta_rad - 0.35 * alpha_loc
            sd, cd = np.sin(delta_rad), np.cos(delta_rad)
            se, ce = np.sin(eff_angle), np.cos(eff_angle)

            cn_flap = self.cfg.cl_delta_flap * sd * cd + 1.15 * se * ce
            cd_flap = self.cfg.cd_delta_flap * (sd ** 2) + 0.25 * (se ** 2)

            fz_flap = q_loc * self.cfg.flap_area_each_m2 * cn_flap
            fx_flap = -q_loc * self.cfg.flap_area_each_m2 * cd_flap
            f_f = np.array([fx_flap, 0.0, fz_flap], dtype=np.float64)
            tau_f = np.cross(r_flap, f_f)
            tau_f[1] += (
                q_loc
                * self.cfg.flap_area_each_m2
                * self.cfg.mean_aero_chord_m
                * self.cfg.cm_delta_flap
                * sd
                * cd
            )
            return f_f, tau_f

        f_flap_r, tau_flap_r = _flap_force_and_moment(self.r_flap_r_body, dr_deg)
        f_flap_l, tau_flap_l = _flap_force_and_moment(self.r_flap_l_body, dl_deg)

        p, q, r = float(omega[0]), float(omega[1]), float(omega[2])
        b = self.cfg.wingspan_m
        c = self.cfg.mean_aero_chord_m
        dyn_damp_scale = 0.5 * rho * airspeed * self.cfg.wing_area_m2
        tau_damp = np.array(
            [
                -(dyn_damp_scale * (b ** 2) * self.cfg.cl_p_damp + self.cfg.viscous_rot_damp) * p,
                -(dyn_damp_scale * (c ** 2) * self.cfg.cm_q_damp * 4.5 + self.cfg.viscous_rot_damp) * q,
                -(dyn_damp_scale * (b ** 2) * self.cfg.cn_r_damp + self.cfg.viscous_rot_damp) * r,
            ],
            dtype=np.float64,
        )

        f_total_body = f_wing_body + f_fuse_body + f_fin_body + f_flap_r + f_flap_l
        tau_total_body = tau_wing + tau_fin + tau_flap_r + tau_flap_l + tau_damp

        drag_total_n = float(-np.dot(f_total_body, wind_hat_b))
        lift_dir_b = np.array([sa, 0.0, -ca], dtype=np.float64)
        lift_total_n = float(np.dot(f_total_body, lift_dir_b))
        ld_ratio = float(lift_total_n / max(0.05, abs(drag_total_n))) if airspeed > 0.5 else 0.0

        weight_n = self.cfg.mass_kg * self.cfg.gravity
        g_load = float(np.linalg.norm(f_total_body) / weight_n)
        reynolds = float(airspeed * self.cfg.mean_aero_chord_m / self.cfg.kinematic_viscosity)
        mach = float(airspeed / self.cfg.speed_of_sound)

        return {
            "airspeed_ms": round(airspeed, 3),
            "airspeed_kmh": round(airspeed * 3.6, 2),
            "dynamic_pressure_pa": round(q_dyn, 2),
            "alpha_deg": round(float(np.rad2deg(alpha)), 3),
            "beta_deg": round(float(np.rad2deg(beta)), 3),
            "lift_n": round(lift_total_n, 4),
            "drag_n": round(drag_total_n, 4),
            "side_force_n": round(float(f_total_body[1]), 4),
            "ld_ratio": round(ld_ratio, 3),
            "g_load": round(g_load, 3),
            "reynolds": round(reynolds, 0),
            "mach": round(mach, 4),
            "force_body_n": [round(float(x), 4) for x in f_total_body],
            "moment_body_nm": [round(float(x), 5) for x in tau_total_body],
            "roll_torque_nm": round(float(tau_total_body[0]), 5),
            "pitch_moment_nm": round(float(tau_total_body[1]), 5),
            "yaw_moment_nm": round(float(tau_total_body[2]), 5),
            "wing_force_body_n": [round(float(x), 4) for x in f_wing_body],
            "fin_force_body_n": [round(float(x), 4) for x in f_fin_body],
            "flap_r_force_body_n": [round(float(x), 4) for x in f_flap_r],
            "flap_l_force_body_n": [round(float(x), 4) for x in f_flap_l],
            "_raw_f_body": f_total_body,
            "_raw_tau_body": tau_total_body,
        }

    def step(self, dt: Optional[float] = None, n_substeps: int = 4) -> Dict[str, Any]:
        step_dt = float(self.cfg.dt if dt is None else dt)
        sub_dt = step_dt / max(1, int(n_substeps))
        ixx, iyy, izz = self.cfg.ixx, self.cfg.iyy, self.cfg.izz

        for _ in range(max(1, int(n_substeps))):
            alpha_s = sub_dt / (self.cfg.servo_tau_s + sub_dt)
            self.delta_r_deg += alpha_s * (self.cmd_delta_r_deg - self.delta_r_deg)
            self.delta_l_deg += alpha_s * (self.cmd_delta_l_deg - self.delta_l_deg)

            self.servo_r_deg = wing_flap_deg_to_servo_deg(self.delta_r_deg, self.geom)
            self.servo_l_deg = wing_flap_deg_to_servo_deg(self.delta_l_deg, self.geom)

            aero = self.compute_aerodynamics()
            f_body = aero["_raw_f_body"]
            tau_body = aero["_raw_tau_body"]

            wx, wy, wz = float(self.omega[0]), float(self.omega[1]), float(self.omega[2])
            dwx = (tau_body[0] - (izz - iyy) * wy * wz) / ixx
            dwy = (tau_body[1] - (ixx - izz) * wx * wz) / iyy
            dwz = (tau_body[2] - (iyy - ixx) * wx * wy) / izz

            self.omega[0] += dwx * sub_dt
            self.omega[1] += dwy * sub_dt
            self.omega[2] += dwz * sub_dt
            self.omega = np.clip(self.omega, -35.0, 35.0)

            self.R_wb = integrate_rotation_so3(self.R_wb, self.omega, sub_dt)
            self.angles = rotation_matrix_to_euler(self.R_wb)

            if self.mode == SIM_MODE_FREE_DROP:
                if -self.pos_ned[2] > 0.0:
                    f_world = self.R_wb @ f_body
                    accel_ned = f_world / self.cfg.mass_kg + np.array(
                        [0.0, 0.0, self.cfg.gravity], dtype=np.float64
                    )
                    self.vel_ned += accel_ned * sub_dt
                    self.pos_ned += self.vel_ned * sub_dt
                    if -self.pos_ned[2] <= 0.0:
                        self.pos_ned[2] = 0.0
                        self.vel_ned[:] = 0.0
                        self.omega[:] = 0.0
            else:
                self.pos_ned[:] = 0.0
                self.vel_ned = np.array([self.tunnel_airspeed_ms, 0.0, 0.0], dtype=np.float64)

        self._sync_to_mujoco()
        self.step_count += 1
        self.time_s = round(self.step_count * step_dt, 4)
        self.last_aero = self.compute_aerodynamics()

        if self.mode == SIM_MODE_FREE_DROP and (self.step_count % 3 == 0):
            self.trail_world.append(
                [
                    round(float(self.pos_ned[0]), 2),
                    round(float(-self.pos_ned[2]), 2),
                    round(float(self.pos_ned[1]), 2),
                ]
            )
            if len(self.trail_world) > 400:
                self.trail_world = self.trail_world[-400:]

        return self.get_telemetry()

    def get_telemetry(self) -> Dict[str, Any]:
        aero = self.last_aero if self.last_aero else self.compute_aerodynamics()
        roll_deg, pitch_deg, yaw_deg = np.rad2deg(self.angles)
        altitude_m = float(max(0.0, -self.pos_ned[2]))

        return {
            "mode": self.mode,
            "step": int(self.step_count),
            "time_s": round(float(self.time_s), 3),
            "roll_deg": round(float(roll_deg), 2),
            "pitch_deg": round(float(pitch_deg), 2),
            "yaw_deg": round(float(yaw_deg), 2),
            "R_wb": [[round(float(v), 5) for v in row] for row in self.R_wb.tolist()],
            "quaternion_wxyz": rotation_matrix_to_quaternion(self.R_wb),
            "gyro_xyz_rads": [
                round(float(self.omega[0]), 3),
                round(float(self.omega[1]), 3),
                round(float(self.omega[2]), 3),
            ],
            "gyro_xyz_degs": [
                round(float(np.rad2deg(self.omega[0])), 1),
                round(float(np.rad2deg(self.omega[1])), 1),
                round(float(np.rad2deg(self.omega[2])), 1),
            ],
            "delta_r_deg": round(float(self.delta_r_deg), 2),
            "delta_l_deg": round(float(self.delta_l_deg), 2),
            "linkage_flap_r_deg": round(float(self.delta_r_deg), 2),
            "linkage_flap_l_deg": round(float(self.delta_l_deg), 2),
            "servo_r_deg": round(float(self.servo_r_deg), 2),
            "servo_l_deg": round(float(self.servo_l_deg), 2),
            "altitude_m": round(altitude_m, 2),
            "position_xyz_m": [
                round(float(self.pos_ned[0]), 2),
                round(float(self.pos_ned[1]), 2),
                round(altitude_m, 2),
            ],
            "velocity_ned_ms": [
                round(float(self.vel_ned[0]), 2),
                round(float(self.vel_ned[1]), 2),
                round(float(self.vel_ned[2]), 2),
            ],
            "airspeed_ms": aero["airspeed_ms"],
            "airspeed_kmh": aero["airspeed_kmh"],
            "dynamic_pressure_pa": aero["dynamic_pressure_pa"],
            "alpha_deg": aero["alpha_deg"],
            "beta_deg": aero["beta_deg"],
            "lift_n": aero["lift_n"],
            "drag_n": aero["drag_n"],
            "side_force_n": aero["side_force_n"],
            "ld_ratio": aero["ld_ratio"],
            "g_load": aero["g_load"],
            "reynolds": aero["reynolds"],
            "mach": aero["mach"],
            "force_body_n": aero["force_body_n"],
            "moment_body_nm": aero["moment_body_nm"],
            "roll_torque_nm": aero["roll_torque_nm"],
            "pitch_moment_nm": aero["pitch_moment_nm"],
            "yaw_moment_nm": aero["yaw_moment_nm"],
            "wing_force_body_n": aero["wing_force_body_n"],
            "fin_force_body_n": aero["fin_force_body_n"],
            "flap_r_force_body_n": aero["flap_r_force_body_n"],
            "flap_l_force_body_n": aero["flap_l_force_body_n"],
            "trail_world": self.trail_world[-160:],
        }
