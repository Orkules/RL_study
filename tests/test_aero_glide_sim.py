from __future__ import annotations

import numpy as np

from sim.aero_glide_sim import (
    SIM_MODE_FREE_DROP,
    SIM_MODE_WIND_TUNNEL,
    AeroGlideSimulator,
    AirframeAeroConfig,
)
from sim.wing_linkage import MAX_WING_FLAP_DEG, get_servo_range_for_wing_limit


def test_airframe_physical_and_cad_specs():
    cfg = AirframeAeroConfig()
    assert cfg.mass_kg == 0.300
    assert abs(cfg.total_length_m - 0.1563) < 1e-6
    assert abs(cfg.fuselage_diameter_m - 0.0432) < 1e-6
    assert abs(cfg.wingspan_m - 0.2802) < 1e-6
    assert abs(cfg.com_from_tail_m - 0.0913) < 1e-6
    assert abs(cfg.wing_cop_from_tail_m - 0.0280) < 1e-6
    assert abs(cfg.wing_cop_x_body_m - (-0.0633)) < 1e-6
    assert abs(cfg.flap_cop_x_body_m - (-0.0690)) < 1e-6
    assert abs(cfg.flap_cop_y_body_m - 0.0711) < 1e-6
    assert MAX_WING_FLAP_DEG == 50.0
    assert get_servo_range_for_wing_limit(50.0) == (-49.20, 53.11)


def test_weathercock_stability_in_pitch_and_yaw():
    sim = AeroGlideSimulator(mode=SIM_MODE_WIND_TUNNEL)
    sim.reset(
        mode=SIM_MODE_WIND_TUNNEL,
        initial_angles_deg=[0.0, 24.0, -18.0],
        initial_omega_rads=[0.0, 0.0, 0.0],
        tunnel_airspeed_ms=35.0,
    )
    sim.set_flaps_deg(0.0, 0.0, immediate=True)

    initial_telemetry = sim.get_telemetry()
    # At +24 deg nose-up pitch (alpha > 0), CoP 63.3 mm behind CoM produces nose-down restoring moment (M < 0)
    assert initial_telemetry["alpha_deg"] > 20.0
    assert initial_telemetry["pitch_moment_nm"] < -0.05
    # At -18 deg nose-left yaw (beta > 0), vertical fin behind CoM produces nose-right restoring moment (N > 0)
    assert initial_telemetry["beta_deg"] > 10.0
    assert initial_telemetry["yaw_moment_nm"] > 0.02

    # Run wind-tunnel dynamics and verify the body weathercocks to near 0 deg AoA and 0 deg sideslip
    for _ in range(300):
        telemetry = sim.step()

    assert abs(telemetry["alpha_deg"]) < 1.0
    assert abs(telemetry["beta_deg"]) < 1.0
    assert abs(telemetry["pitch_deg"]) < 1.0
    assert abs(telemetry["yaw_deg"]) < 1.0


def test_symmetric_pitch_flap_deflection_trims_to_aoa():
    sim = AeroGlideSimulator(mode=SIM_MODE_WIND_TUNNEL)
    sim.reset(
        mode=SIM_MODE_WIND_TUNNEL,
        initial_angles_deg=[0.0, 0.0, 0.0],
        initial_omega_rads=[0.0, 0.0, 0.0],
        tunnel_airspeed_ms=35.0,
    )
    # Deflect both wing flaps symmetrically (+25 deg pitch command)
    sim.set_pitch_roll_mix(pitch_flap_deg=25.0, roll_flap_deg=0.0, immediate=True)

    # Immediately at alpha = 0, symmetric +25 deg flap deflection creates nose-up pitch moment (M > 0)
    t0 = sim.get_telemetry()
    assert t0["pitch_moment_nm"] > 0.05
    assert abs(t0["roll_torque_nm"]) < 1e-6

    for _ in range(300):
        t_pos = sim.step()

    alpha_trim_pos = t_pos["alpha_deg"]
    assert alpha_trim_pos > 8.0
    assert t_pos["lift_n"] > 1.0
    assert abs(t_pos["gyro_xyz_rads"][0]) < 1e-3

    # Opposite symmetric pitch deflection (-25 deg) trims to symmetric negative Angle of Attack
    sim.set_pitch_roll_mix(pitch_flap_deg=-25.0, roll_flap_deg=0.0, immediate=True)
    for _ in range(350):
        t_neg = sim.step()

    alpha_trim_neg = t_neg["alpha_deg"]
    assert alpha_trim_neg < -8.0
    assert abs(alpha_trim_pos + alpha_trim_neg) < 1.0


def test_differential_roll_flap_deflection_generates_roll_rate():
    sim = AeroGlideSimulator(mode=SIM_MODE_WIND_TUNNEL)
    sim.reset(
        mode=SIM_MODE_WIND_TUNNEL,
        initial_angles_deg=[0.0, 0.0, 0.0],
        initial_omega_rads=[0.0, 0.0, 0.0],
        tunnel_airspeed_ms=35.0,
    )
    # Differential roll deflection: delta_R = +30 deg, delta_L = -30 deg
    sim.set_pitch_roll_mix(pitch_flap_deg=0.0, roll_flap_deg=30.0, immediate=True)

    t0 = sim.get_telemetry()
    assert t0["roll_torque_nm"] > 0.03
    assert abs(t0["pitch_moment_nm"]) < 1e-5

    for _ in range(150):
        telemetry = sim.step()

    p_rate = telemetry["gyro_xyz_rads"][0]
    assert p_rate > 8.0
    assert abs(telemetry["alpha_deg"]) < 1.0


def test_free_drop_accelerates_to_terminal_and_glide_velocity():
    # 1. Zero-flap ballistic free-drop reaches steady terminal velocity (Drag == Weight -> 1.0 G)
    sim_drop = AeroGlideSimulator(mode=SIM_MODE_FREE_DROP)
    sim_drop.reset(
        mode=SIM_MODE_FREE_DROP,
        initial_altitude_m=500.0,
        initial_angles_deg=[0.0, -85.0, 0.0],
        initial_velocity_ned_ms=[1.0, 0.0, 5.0],
    )
    sim_drop.set_pitch_roll_mix(pitch_flap_deg=0.0, roll_flap_deg=0.0, immediate=True)
    for _ in range(900):
        t_term = sim_drop.step()

    assert t_term["altitude_m"] < 300.0
    assert 40.0 < t_term["airspeed_ms"] < 85.0
    assert 0.85 < t_term["g_load"] < 1.15

    # 2. Gliding free-drop with symmetric pitch-up flap trim pulls out and glides horizontally
    sim = AeroGlideSimulator(mode=SIM_MODE_FREE_DROP)
    sim.reset(
        mode=SIM_MODE_FREE_DROP,
        initial_altitude_m=500.0,
        initial_angles_deg=[0.0, -15.0, 0.0],
        initial_velocity_ned_ms=[18.0, 0.0, 4.0],
    )
    sim.set_pitch_roll_mix(pitch_flap_deg=14.0, roll_flap_deg=0.0, immediate=True)

    t_start = sim.get_telemetry()
    assert t_start["altitude_m"] == 500.0
    initial_speed = t_start["airspeed_ms"]

    for _ in range(800):  # 8.0 seconds of free-drop glide flight
        t_end = sim.step()

    # Altitude decreases while airspeed stabilizes in glide regime and horizontal distance advances
    assert t_end["altitude_m"] < 450.0
    assert t_end["altitude_m"] > 100.0
    assert t_end["airspeed_ms"] > initial_speed + 8.0
    assert 22.0 < t_end["airspeed_ms"] < 65.0
    assert t_end["position_xyz_m"][0] > 120.0
    assert t_end["lift_n"] > 1.5
    assert 0.70 < t_end["g_load"] < 1.45


def test_viewer_aero_html_and_server_routes():
    from pathlib import Path
    from viewer.server import AERO_HTML, VIEWER_HTML, app

    assert VIEWER_HTML.exists()
    assert AERO_HTML.exists()
    aero_content = Path(AERO_HTML).read_text(encoding="utf-8")
    assert "EMBEDDED_STL_B64" in aero_content
    assert "BODY.STL" in aero_content
    assert "HEAD.STL" in aero_content
    assert "WING FLAP.STL" in aero_content

    route_paths = {getattr(r, "path", None) for r in app.routes}
    assert "/" in route_paths
    assert "/aero" in route_paths
    assert "/ws/sim" in route_paths
    assert "/ws/aero" in route_paths

