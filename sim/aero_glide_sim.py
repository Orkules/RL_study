from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from sim.wing_linkage import (
    MAX_WING_FLAP_DEG,
    LinkageGeometry,
    get_servo_range_for_wing_limit,
    servo_deg_to_wing_flap_deg,
    wing_flap_deg_to_servo_deg,
)

SIM_MODE_WIND_TUNNEL = "wind_tunnel"
SIM_MODE_FREE_DROP = "free_drop"


@dataclass
class AirframeAeroConfig:
    """
    Exact physical, mass, CAD geometry, and aerodynamic specifications for Simulation 2
    extracted from FINAL FOR PRINT/*.STL.

    Coordinate Systems:
      - CAD Assembly Frame (mm):
          Y_cad: 0.0 mm at BODY.STL tail base -> 91.3 mm at BODY/HEAD joint -> 156.3 mm at HEAD nose tip
          X_cad: -140.1 mm (Left Wingtip) .. +140.1 mm (Right Wingtip), wingspan = 280.2 mm
          Z_cad: Normal to wing plane (0.0 mm at wing hinge axis)
      - Flight Dynamics Body Frame (m) centered at Center of Mass (Y_cad = 91.3 mm):
          +x_b: Forward along fuselage toward nose tip (+Y_cad)
          +y_b: Out the Right Wing (+X_cad)
          +z_b: Ventral / downward (-Z_cad), completing right-handed frame (x_b x y_b = z_b)
    """

    # Airframe mass & environment
    mass_kg: float = 0.300                 # 300 g total mass
    gravity: float = 9.81                  # m/s^2
    rho_sea_level: float = 1.225           # kg/m^3 standard air density
    kinematic_viscosity: float = 1.48e-5   # m^2/s air kinematic viscosity
    speed_of_sound: float = 340.3          # m/s

    # Overall CAD dimensions (m)
    total_length_m: float = 0.1563         # 156.3 mm (91.3 mm BODY + 65.0 mm HEAD)
    body_length_m: float = 0.0913          # 91.3 mm
    head_length_m: float = 0.0650          # 65.0 mm
    fuselage_diameter_m: float = 0.0432    # 43.2 mm
    wingspan_m: float = 0.2802             # 280.2 mm
    wing_root_chord_m: float = 0.0568      # ~57 mm (Y = 1.3 .. 58.1 mm)
    mean_aero_chord_m: float = 0.0420      # 42.0 mm mean aerodynamic chord

    # Reference aerodynamic surface areas (m^2)
    wing_area_m2: float = 0.0112           # Swept delta wings + fuselage planform area
    fin_area_m2: float = 0.0045            # Vertical tail fin + wingtip winglets side area
    flap_span_m: float = 0.0876            # 87.6 mm each (|X| = 27.3 .. 114.9 mm)
    flap_chord_m: float = 0.0202           # 20.2 mm chord
    flap_area_each_m2: float = 0.0017695   # 0.0876 m * 0.0202 m per wing flap

    # Center of Mass (CoM) & Centers of Pressure (CoP)
    com_from_tail_m: float = 0.0913        # 91.3 mm from tail base (at BODY/HEAD junction)
    wing_cop_from_tail_m: float = 0.0280   # 28.0 mm from tail base
    wing_cop_x_body_m: float = -0.0633     # 63.3 mm BEHIND CoM -> weathercock pitch stability
    fin_cop_x_body_m: float = -0.0633      # 63.3 mm BEHIND CoM -> weathercock yaw stability
    flap_cop_x_body_m: float = -0.0690     # 69.0 mm BEHIND CoM (22.3 mm from tail base)
    flap_cop_y_body_m: float = 0.0711      # ±71.1 mm spanwise CoP of each wing flap

    # Principal moments of inertia about CoM (kg * m^2) for 300 g body
    ixx: float = 0.00028                   # Roll inertia about longitudinal axis x_b
    iyy: float = 0.00068                   # Pitch inertia about spanwise axis y_b
    izz: float = 0.00082                   # Yaw inertia about vertical axis z_b

    # Nonlinear aerodynamic coefficients
    cd0_parasitic: float = 0.145           # Zero-AoA parasitic + blunt-base drag of body/wings/fins/linkages
    cl_alpha_wing: float = 3.50            # Delta wing lift slope (rad^-1) acting at Wing CoP (-63.3 mm)
    cl_alpha_fuselage: float = 0.95        # Nose cone (HEAD.STL) + forward fuselage lift slope near CoM
    cl_vortex_wing: float = 1.40           # Polhamus vortex lift term sin^2(alpha)*cos(alpha)
    cd_induced_wing: float = 1.35          # High-angle crossflow/induced drag factor sin^2(alpha)
    cy_beta_fin: float = 2.80              # Vertical fin & winglet side-force slope (rad^-1)

    # Trailing-edge wing flap nonlinear model (up to ±50 deg)
    cl_delta_flap: float = 2.60            # Direct flap normal/lift pressure sin(delta)*cos(delta)
    cm_delta_camber: float = 1.85          # Wing-flap chordwise camber pitching couple
    cd_delta_flap: float = 1.45            # Nonlinear flat-plate flap drag sin^2(delta)
    flap_alpha_coupling: float = 0.35      # Local AoA coupling into flap normal pressure

    # Aerodynamic rotational rate damping coefficients
    cl_p_damp: float = 0.52                # Wing & winglet roll damping C_lp
    cm_q_damp: float = 3.60                # Wing pitch rate damping C_mq
    cn_r_damp: float = 1.35                # Fin & winglet yaw rate damping C_nr
    viscous_rot_damp: float = 0.0012       # Low-airspeed viscous rotational damping (N*m*s/rad)

    # Servo & simulation timing
    max_wing_flap_deg: float = MAX_WING_FLAP_DEG
    servo_tau_s: float = 0.022             # SG90 first-order response time constant (s)
    dt: float = 0.01                       # 100 Hz default physics timestep (s)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles (roll, pitch, yaw in rad) to 3x3 rotation matrix R_wb (body -> world)."""
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
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= max(1e-12, float(np.linalg.norm(q)))
    return [round(float(v), 5) for v in q]


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

    # Gram-Schmidt orthonormalization to preserve exact SO(3) orthogonality
    c0 = R_next[:, 0]
    c0 = c0 / max(1e-12, float(np.linalg.norm(c0)))
    c1 = R_next[:, 1] - np.dot(R_next[:, 1], c0) * c0
    c1 = c1 / max(1e-12, float(np.linalg.norm(c1)))
    c2 = np.cross(c0, c1)
    return np.column_stack((c0, c1, c2))


class AeroGlideSimulator:
    """
    Simulation 2: 3D Aerodynamic & Free-Drop Flight Dynamics Simulator for the 300 g CAD Airframe.

    Supports two interactive operating modes:
      1. `wind_tunnel`:
         CoM is held at the center of the 3D wind tunnel while the body rotates freely in 3D
         (Roll, Pitch, Yaw / Angle of Attack alpha & Sideslip beta) in response to freestream
         airflow (0..100 m/s) and nonlinear 4-bar wing flap deflections (up to ±50 deg).
      2. `free_drop`:
         Releases the 300 g body in a 3D world from configurable altitude (e.g., 200 m or 500 m)
         under gravity (9.81 m/s^2) with full 6-DOF rigid-body flight dynamics.
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

        # 3D positions of aerodynamic Centers of Pressure in the CoM-centered body frame (m)
        self.r_wing_body = np.array([self.cfg.wing_cop_x_body_m, 0.0, 0.0], dtype=np.float64)
        self.r_fin_body = np.array([self.cfg.fin_cop_x_body_m, 0.0, 0.0], dtype=np.float64)
        self.r_flap_r_body = np.array(
            [self.cfg.flap_cop_x_body_m, +self.cfg.flap_cop_y_body_m, 0.0], dtype=np.float64
        )
        self.r_flap_l_body = np.array(
            [self.cfg.flap_cop_x_body_m, -self.cfg.flap_cop_y_body_m, 0.0], dtype=np.float64
        )

        # State vectors
        self.R_wb = np.eye(3, dtype=np.float64)
        self.angles = np.zeros(3, dtype=np.float64)         # [roll, pitch, yaw] in rad
        self.omega = np.zeros(3, dtype=np.float64)          # [p, q, r] body angular rates in rad/s
        self.pos_ned = np.zeros(3, dtype=np.float64)        # [north_x, east_y, down_z] in m
        self.vel_ned = np.zeros(3, dtype=np.float64)        # [vn, ve, vd] in m/s

        # Wind-tunnel & environment settings
        self.tunnel_airspeed_ms: float = 28.0               # Default 28 m/s (~100.8 km/h)
        self.ambient_wind_ned = np.zeros(3, dtype=np.float64)
        self.initial_drop_altitude_m: float = 200.0

        # Servo & wing flap state (deg)
        # Convention:
        #   delta_r_deg > 0: Right wing flap trailing edge deflected UP (+Z_cad / -z_b)
        #   delta_l_deg > 0: Left wing flap trailing edge deflected UP (+Z_cad / -z_b)
        #   Symmetric pitch deflection: delta_r_deg == delta_l_deg (+ pitches nose UP to +alpha_trim)
        #   Differential roll deflection: delta_r_deg == -delta_l_deg (+ rolls right, p > 0)
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
            # Free-drop starts slightly nose-down (-15 deg pitch) with forward launch speed
            ang_deg = np.array([0.0, -15.0, 0.0], dtype=np.float64)

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
                self.vel_ned = np.array([12.0, 0.0, 2.0], dtype=np.float64)

        self.step_count = 0
        self.time_s = 0.0
        self.trail_world = [[0.0, float(-self.pos_ned[2]), 0.0]]
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
          - Symmetric pitch deflection: delta_r_deg == delta_l_deg
          - Differential roll deflection: delta_r_deg == -delta_l_deg
        """
        lim = self.cfg.max_wing_flap_deg
        self.cmd_delta_r_deg = float(np.clip(delta_r_deg, -lim, +lim))
        self.cmd_delta_l_deg = float(np.clip(delta_l_deg, -lim, +lim))
        if immediate:
            self.delta_r_deg = self.cmd_delta_r_deg
            self.delta_l_deg = self.cmd_delta_l_deg
            self.servo_r_deg = wing_flap_deg_to_servo_deg(+self.delta_r_deg, self.geom)
            # Left wing linkage is mounted on the -Z face (side = -1 in 3D CAD assembly)
            self.servo_l_deg = wing_flap_deg_to_servo_deg(-self.delta_l_deg, self.geom)
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
        Set SG90 right and left servo angles (deg from Center Wings) and solve the 4-bar linkage.
        Note: In the 3D CAD assembly, the Left servo faces -X (side = -1), so linkage rotation
        +flap_l_linkage corresponds to -delta_l_deg in body vertical coordinates.
        """
        s_min, s_max = get_servo_range_for_wing_limit(self.cfg.max_wing_flap_deg, self.geom)
        sr = float(np.clip(servo_r_deg, s_min, s_max))
        sl = float(np.clip(servo_l_deg, s_min, s_max))
        flap_r_link = servo_deg_to_wing_flap_deg(sr, self.geom)
        flap_l_link = servo_deg_to_wing_flap_deg(sl, self.geom)
        self.set_flaps_deg(delta_r_deg=+flap_r_link, delta_l_deg=-flap_l_link, immediate=immediate)

    def set_tunnel_airspeed(self, airspeed_ms: float) -> None:
        """Set wind-tunnel freestream airspeed in [0, 100] m/s (0..360 km/h)."""
        self.tunnel_airspeed_ms = float(np.clip(airspeed_ms, 0.0, 100.0))
        if self.mode == SIM_MODE_WIND_TUNNEL:
            self.vel_ned = np.array([self.tunnel_airspeed_ms, 0.0, 0.0], dtype=np.float64)
            self.last_aero = self.compute_aerodynamics()

    def _air_density(self, altitude_m: float) -> float:
        """ISA troposphere air density model vs altitude (m)."""
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
        """
        Compute exact 3D aerodynamic forces and moments on the 300 g airframe:
          1. Swept Delta Wings (CoP 63.3 mm behind CoM -> weathercock pitch stability)
          2. Vertical Tail Fin & Wingtip Winglets (CoP 63.3 mm behind CoM -> weathercock yaw stability & roll damping)
          3. Right & Left Trailing-Edge Wing Flaps (CoP at X = ±71.1 mm, 69.0 mm behind CoM,
             nonlinear sin(delta)*cos(delta) pressure lift + sin^2(delta) flat-plate drag)
        """
        R = self.R_wb if R_wb is None else R_wb
        v_ned = (
            np.array([self.tunnel_airspeed_ms, 0.0, 0.0], dtype=np.float64)
            if (vel_ned is None and self.mode == SIM_MODE_WIND_TUNNEL)
            else (self.vel_ned if vel_ned is None else vel_ned)
        )
        omega = self.omega if omega_body is None else omega_body
        dr_deg = self.delta_r_deg if delta_r_deg is None else float(delta_r_deg)
        dl_deg = self.delta_l_deg if delta_l_deg is None else float(delta_l_deg)
        alt_m = float(-self.pos_ned[2]) if altitude_m is None else float(altitude_m)

        # Relative airspeed vector in body frame: v_rel_b = R_wb^T * (v_vehicle - v_wind)
        v_rel_ned = v_ned - self.ambient_wind_ned
        v_rel_b = R.T @ v_rel_ned
        u, v, w = float(v_rel_b[0]), float(v_rel_b[1]), float(v_rel_b[2])
        airspeed = float(np.linalg.norm(v_rel_b))

        rho = self._air_density(alt_m)
        q_dyn = 0.5 * rho * (airspeed ** 2)

        if airspeed > 1e-4:
            alpha = float(np.arctan2(w, u))
            beta = float(np.arcsin(np.clip(v / airspeed, -1.0, 1.0)))
            wind_hat_b = v_rel_b / airspeed
        else:
            alpha = 0.0
            beta = 0.0
            wind_hat_b = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        sa, ca = np.sin(alpha), np.cos(alpha)
        sb, cb = np.sin(beta), np.cos(beta)

        # -------------------------------------------------------------------------
        # 1. Swept Delta Wings + Fuselage Body (CoP at x_b = -63.3 mm behind CoM)
        # -------------------------------------------------------------------------
        # Nonlinear Polhamus delta-wing lift + high-angle flat-plate crossflow
        cl_wing = (
            self.cfg.cl_alpha_wing * sa * ca
            + self.cfg.cl_vortex_wing * (sa ** 2) * ca * np.sign(sa)
        )
        cd_wing = self.cfg.cd0_parasitic + self.cfg.cd_induced_wing * (sa ** 2) + 0.35 * (sb ** 2)

        lift_wing_n = q_dyn * self.cfg.wing_area_m2 * cl_wing
        lift_fuse_n = q_dyn * self.cfg.wing_area_m2 * (self.cfg.cl_alpha_fuselage * sa * ca)
        drag_wing_n = q_dyn * self.cfg.wing_area_m2 * cd_wing

        # Convert wing lift (perpendicular to airflow in x-z plane) and drag (opposing airflow) to body frame:
        # At alpha > 0, lift is in -z_b (upward) and drag is in -x_b (aft):
        fx_wing = -drag_wing_n * ca + lift_wing_n * sa
        fz_wing = -drag_wing_n * sa - lift_wing_n * ca
        f_wing_body = np.array([fx_wing, 0.0, fz_wing], dtype=np.float64)
        f_fuse_body = np.array([lift_fuse_n * sa, 0.0, -lift_fuse_n * ca], dtype=np.float64)

        # Weathercock pitch moment from Wing CoP 63.3 mm behind CoM: tau_wing = r_wing x F_wing
        tau_wing = np.cross(self.r_wing_body, f_wing_body)

        # -------------------------------------------------------------------------
        # 2. Vertical Tail Fin & Wingtip Winglets (CoP at x_b = -63.3 mm behind CoM)
        # -------------------------------------------------------------------------
        cy_fin = -self.cfg.cy_beta_fin * sb * cb
        fy_fin = q_dyn * self.cfg.fin_area_m2 * cy_fin
        f_fin_body = np.array([0.0, fy_fin, 0.0], dtype=np.float64)

        # Weathercock yaw moment from Fin CoP 63.3 mm behind CoM: tau_fin = r_fin x F_fin
        tau_fin = np.cross(self.r_fin_body, f_fin_body)

        # -------------------------------------------------------------------------
        # 3. Trailing-Edge Wing Flaps (Right & Left, X = ±71.1 mm, 69.0 mm behind CoM)
        # -------------------------------------------------------------------------
        dr_rad = float(np.deg2rad(dr_deg))
        dl_rad = float(np.deg2rad(dl_deg))

        def _flap_force_and_moment(
            r_flap: np.ndarray, delta_rad: float
        ) -> Tuple[np.ndarray, np.ndarray, float, float]:
            # Local airflow at the flap CoP including body angular velocity (omega x r_flap)
            v_loc = v_rel_b + np.cross(omega, r_flap)
            v_loc_mag = float(np.linalg.norm(v_loc))
            q_loc = 0.5 * rho * (v_loc_mag ** 2)
            alpha_loc = float(np.arctan2(v_loc[2], v_loc[0])) if v_loc_mag > 1e-4 else 0.0

            # Effective flap deflection angle relative to local flow
            # Convention: delta_rad > 0 = trailing edge deflected UP (+Z_cad / -z_b),
            # which produces downward normal force (+z_b) on the tail -> nose-UP pitch moment!
            eff_ang = delta_rad - self.cfg.flap_alpha_coupling * alpha_loc
            sd, cd = np.sin(delta_rad), np.cos(delta_rad)
            se, ce = np.sin(eff_ang), np.cos(eff_ang)

            # Nonlinear flap normal/lift force (sin(delta)*cos(delta)) and flap drag (sin^2(delta))
            cn_flap = self.cfg.cl_delta_flap * sd * cd + 1.15 * se * ce
            cd_flap = self.cfg.cd_delta_flap * (sd ** 2) + 0.25 * (se ** 2)

            # Force in body frame: +cn_flap pushes in +z_b (downward on tail when trailing edge is UP)
            fz_f = q_loc * self.cfg.flap_area_each_m2 * cn_flap
            fx_f = -q_loc * self.cfg.flap_area_each_m2 * cd_flap * np.sign(v_loc[0] if abs(v_loc[0]) > 1e-4 else 1.0)
            f_f = np.array([fx_f, 0.0, fz_f], dtype=np.float64)

            # Exact 3D moment from r_flap x F_flap + chordwise wing-camber couple
            tau_f = np.cross(r_flap, f_f)
            tau_f[1] += (
                q_loc
                * self.cfg.flap_area_each_m2
                * self.cfg.mean_aero_chord_m
                * self.cfg.cm_delta_camber
                * sd
                * cd
            )
            return f_f, tau_f, -fz_f * ca, abs(fx_f)

        f_flap_r, tau_flap_r, lift_fr, drag_fr = _flap_force_and_moment(self.r_flap_r_body, dr_rad)
        f_flap_l, tau_flap_l, lift_fl, drag_fl = _flap_force_and_moment(self.r_flap_l_body, dl_rad)

        # -------------------------------------------------------------------------
        # 4. Aerodynamic Rotational Rate Damping (p, q, r) from Wings & Winglets
        # -------------------------------------------------------------------------
        p, q, r = float(omega[0]), float(omega[1]), float(omega[2])
        b = self.cfg.wingspan_m
        c = self.cfg.mean_aero_chord_m
        s_ref = self.cfg.wing_area_m2

        dyn_damp_scale = 0.5 * rho * airspeed * s_ref
        tau_damp = np.array(
            [
                -(dyn_damp_scale * (b ** 2) * self.cfg.cl_p_damp + self.cfg.viscous_rot_damp) * p,
                -(dyn_damp_scale * (c ** 2) * self.cfg.cm_q_damp * 4.5 + self.cfg.viscous_rot_damp) * q,
                -(dyn_damp_scale * (b ** 2) * self.cfg.cn_r_damp + self.cfg.viscous_rot_damp) * r,
            ],
            dtype=np.float64,
        )

        # Total 3D aerodynamic force and moment in body frame
        f_total_body = f_wing_body + f_fuse_body + f_fin_body + f_flap_r + f_flap_l
        tau_total_body = tau_wing + tau_fin + tau_flap_r + tau_flap_l + tau_damp

        # Resolve total aerodynamic Lift (perpendicular to velocity in longitudinal plane) & Drag (along -v_rel)
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
        """Advance the 3D aerodynamic simulation by one control step."""
        step_dt = float(self.cfg.dt if dt is None else dt)
        sub_dt = step_dt / max(1, int(n_substeps))

        ixx, iyy, izz = self.cfg.ixx, self.cfg.iyy, self.cfg.izz

        for _ in range(max(1, int(n_substeps))):
            # 1. SG90 servo & linkage actuator dynamics
            alpha_s = sub_dt / (self.cfg.servo_tau_s + sub_dt)
            self.delta_r_deg += alpha_s * (self.cmd_delta_r_deg - self.delta_r_deg)
            self.delta_l_deg += alpha_s * (self.cmd_delta_l_deg - self.delta_l_deg)

            self.servo_r_deg = wing_flap_deg_to_servo_deg(+self.delta_r_deg, self.geom)
            self.servo_l_deg = wing_flap_deg_to_servo_deg(-self.delta_l_deg, self.geom)

            # 2. Evaluate 3D aerodynamic forces and moments
            aero = self.compute_aerodynamics()
            f_body = aero["_raw_f_body"]
            tau_body = aero["_raw_tau_body"]

            # 3. Rigid-body Euler rotational dynamics: I * d(omega)/dt + omega x (I * omega) = tau
            wx, wy, wz = float(self.omega[0]), float(self.omega[1]), float(self.omega[2])
            dwx = (tau_body[0] - (izz - iyy) * wy * wz) / ixx
            dwy = (tau_body[1] - (ixx - izz) * wx * wz) / iyy
            dwz = (tau_body[2] - (iyy - ixx) * wx * wy) / izz

            self.omega[0] += dwx * sub_dt
            self.omega[1] += dwy * sub_dt
            self.omega[2] += dwz * sub_dt
            self.omega = np.clip(self.omega, -35.0, 35.0)

            # 4. Integrate 3D orientation on SO(3) (gimbal-lock free)
            self.R_wb = integrate_rotation_so3(self.R_wb, self.omega, sub_dt)
            self.angles = rotation_matrix_to_euler(self.R_wb)

            # 5. Translational dynamics (Free-Drop Glide Mode vs Wind-Tunnel Mode)
            if self.mode == SIM_MODE_FREE_DROP:
                if -self.pos_ned[2] > 0.0:
                    f_world = self.R_wb @ f_body
                    accel_ned = f_world / self.cfg.mass_kg + np.array(
                        [0.0, 0.0, self.cfg.gravity], dtype=np.float64
                    )
                    self.vel_ned += accel_ned * sub_dt
                    self.pos_ned += self.vel_ned * sub_dt
                    # Ground contact clamp at altitude = 0 m
                    if -self.pos_ned[2] <= 0.0:
                        self.pos_ned[2] = 0.0
                        self.vel_ned[:] = 0.0
                        self.omega[:] = 0.0
            else:
                self.pos_ned[:] = 0.0
                self.vel_ned = np.array([self.tunnel_airspeed_ms, 0.0, 0.0], dtype=np.float64)

        self.step_count += 1
        self.time_s = round(self.step_count * step_dt, 4)
        self.last_aero = self.compute_aerodynamics()

        if self.mode == SIM_MODE_FREE_DROP and (self.step_count % 3 == 0):
            # Store 3D trajectory trail in Three.js world coordinates [X_east, Y_alt, Z_south]
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
        """Return complete telemetry dictionary for WebSocket streaming and unit tests."""
        aero = self.last_aero if self.last_aero else self.compute_aerodynamics()
        roll_deg, pitch_deg, yaw_deg = np.rad2deg(self.angles)
        altitude_m = float(max(0.0, -self.pos_ned[2]))

        return {
            "mode": self.mode,
            "step": int(self.step_count),
            "time_s": round(float(self.time_s), 3),
            # Attitude & angular rates
            "roll_deg": round(float(roll_deg), 2),
            "pitch_deg": round(float(pitch_deg), 2),
            "yaw_deg": round(float(yaw_deg), 2),
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
            # Servos & 4-bar linkage wing flaps
            "delta_r_deg": round(float(self.delta_r_deg), 2),
            "delta_l_deg": round(float(self.delta_l_deg), 2),
            "linkage_flap_r_deg": round(float(+self.delta_r_deg), 2),
            "linkage_flap_l_deg": round(float(-self.delta_l_deg), 2),
            "servo_r_deg": round(float(self.servo_r_deg), 2),
            "servo_l_deg": round(float(self.servo_l_deg), 2),
            # 3D position & velocity
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
            "vertical_speed_ms": round(float(-self.vel_ned[2]), 2),
            # Aerodynamic metrics
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
            # Physical reference constants
            "mass_g": round(self.cfg.mass_kg * 1000.0, 1),
            "com_from_tail_mm": round(self.cfg.com_from_tail_m * 1000.0, 1),
            "wing_cop_from_tail_mm": round(self.cfg.wing_cop_from_tail_m * 1000.0, 1),
            "flap_cop_behind_com_mm": round(abs(self.cfg.flap_cop_x_body_m) * 1000.0, 1),
            "trail_world": self.trail_world[-120:],
        }
