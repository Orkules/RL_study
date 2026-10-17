from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

# Configurable symmetric wing flap limit (change to 40.0 or 50.0 as needed)
MAX_WING_FLAP_DEG: float = 50.0


@dataclass(frozen=True)
class LinkageGeometry:
    """Exact 2-circle + connecting rod CAD geometry extracted from FINAL FOR PRINT/*.STL."""

    r_arm_mm: float = 13.0000          # WING MOTOR ARM shaft-to-tip hole radius (mm)
    l_rod_mm: float = 16.9100          # CONROD hole-1 to hole-2 length (mm)
    r_flap_mm: float = 13.0000         # WING FLAP hinge-to-horn hole radius (mm)
    dz0_horn_mm: float = 12.4555       # Horn pin Z offset from wing hinge at 0 deg flap (mm)
    dy0_horn_mm: float = -3.7230       # Horn pin Y offset from wing hinge at 0 deg flap (mm)
    zs_servo_mm: float = 7.0000        # BODY servo hole Z offset from centerline (mm)
    ys_servo_mm: float = 35.8950       # BODY servo hole Y height (mm)
    zh_hinge_mm: float = 0.0000        # BODY wing hinge hole Z offset (mm)
    yh_hinge_mm: float = 21.2936       # BODY wing hinge hole Y height (mm)
    theta0_arm_deg: float = -13.4290   # World arm angle where Wing Flap = 0.0 deg


def servo_deg_to_wing_flap_deg(
    servo_deg: float, geom: LinkageGeometry = LinkageGeometry()
) -> float:
    """Forward 4-bar equation: given Servo angle (deg from Center Wings), return Wing Flap angle (deg)."""
    phi0 = np.arctan2(geom.dy0_horn_mm, geom.dz0_horn_mm)
    theta = np.deg2rad(geom.theta0_arm_deg + servo_deg)

    az = geom.zs_servo_mm + geom.r_arm_mm * np.cos(theta)
    ay = geom.ys_servo_mm + geom.r_arm_mm * np.sin(theta)

    dx = az - geom.zh_hinge_mm
    dy = ay - geom.yh_hinge_mm
    d = float(
        np.clip(
            np.hypot(dx, dy),
            abs(geom.r_flap_mm - geom.l_rod_mm) + 1e-5,
            geom.r_flap_mm + geom.l_rod_mm - 1e-5,
        )
    )
    a = (geom.r_flap_mm**2 - geom.l_rod_mm**2 + d**2) / (2.0 * d)
    h = float(np.sqrt(max(0.0, geom.r_flap_mm**2 - a**2)))

    pz = geom.zh_hinge_mm + a * (dx / d) + h * (dy / d)
    py = geom.yh_hinge_mm + a * (dy / d) - h * (dx / d)

    phi = np.arctan2(py - geom.yh_hinge_mm, pz - geom.zh_hinge_mm)
    return float(np.rad2deg(phi - phi0))


def wing_flap_deg_to_servo_deg(
    flap_deg: float, geom: LinkageGeometry = LinkageGeometry()
) -> float:
    """
    Exact analytical inverse 4-bar equation:
    Given desired Wing Flap angle (e.g. in [-MAX_WING_FLAP_DEG, +MAX_WING_FLAP_DEG]),
    returns the exact Servo angle (in degrees from Center Wings) required to achieve it.
    """
    phi0 = np.arctan2(geom.dy0_horn_mm, geom.dz0_horn_mm)
    phi = phi0 + np.deg2rad(flap_deg)

    # Exact position of the Wing Flap horn pin P = (pz, py)
    pz = geom.zh_hinge_mm + geom.r_flap_mm * np.cos(phi)
    py = geom.yh_hinge_mm + geom.r_flap_mm * np.sin(phi)

    # Circle-circle intersection around Servo center S = (zs, ys) with radius r_arm
    # and around P = (pz, py) with radius l_rod:
    dx = pz - geom.zs_servo_mm
    dy = py - geom.ys_servo_mm
    d = float(
        np.clip(
            np.hypot(dx, dy),
            abs(geom.r_arm_mm - geom.l_rod_mm) + 1e-5,
            geom.r_arm_mm + geom.l_rod_mm - 1e-5,
        )
    )
    a = (geom.r_arm_mm**2 - geom.l_rod_mm**2 + d**2) / (2.0 * d)
    h = float(np.sqrt(max(0.0, geom.r_arm_mm**2 - a**2)))

    # Pick the outward arm solution (az > zs)
    az = geom.zs_servo_mm + a * (dx / d) - h * (dy / d)
    ay = geom.ys_servo_mm + a * (dy / d) + h * (dx / d)

    theta_deg = float(np.rad2deg(np.arctan2(ay - geom.ys_servo_mm, az - geom.zs_servo_mm)))
    return theta_deg - geom.theta0_arm_deg


def get_servo_range_for_wing_limit(
    max_wing_flap_deg: float = MAX_WING_FLAP_DEG,
    geom: LinkageGeometry = LinkageGeometry(),
) -> Tuple[float, float]:
    """
    Compute the exact (min_servo_deg, max_servo_deg) corresponding to
    (-max_wing_flap_deg, +max_wing_flap_deg).
    Example:
      max_wing_flap_deg = 50.0 -> (-49.20 deg, +53.11 deg)
      max_wing_flap_deg = 40.0 -> (-39.61 deg, +41.70 deg)
    """
    s_min = wing_flap_deg_to_servo_deg(-abs(max_wing_flap_deg), geom)
    s_max = wing_flap_deg_to_servo_deg(+abs(max_wing_flap_deg), geom)
    return (round(s_min, 2), round(s_max, 2))


def normalized_action_to_servo_deg(
    u_norm: float,
    max_wing_flap_deg: float = MAX_WING_FLAP_DEG,
    geom: LinkageGeometry = LinkageGeometry(),
) -> float:
    """Map RL action u in [-1, 1] to exact Servo angle (deg) for symmetric wing deflection."""
    u = float(np.clip(u_norm, -1.0, 1.0))
    target_flap_deg = u * max_wing_flap_deg
    return wing_flap_deg_to_servo_deg(target_flap_deg, geom)
