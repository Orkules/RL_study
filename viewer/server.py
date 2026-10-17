from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch

from rl.ppo_agent import ActorCritic
from sim.aero_glide_sim import AeroGlideSimulator, AirframeAeroConfig
from sim.tethered_drone_env import TetheredDroneEnv

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
STL_DIR = ROOT_DIR / "FINAL FOR PRINT"
VIEWER_HTML = Path(__file__).resolve().parent / "index.html"
AERO_HTML = Path(__file__).resolve().parent / "aero.html"

app = FastAPI(title="3D CAD Linkage & Aerodynamic Flight Simulators (FINAL FOR PRINT)")

if STL_DIR.exists():
    app.mount("/stl", StaticFiles(directory=str(STL_DIR)), name="stl")


def load_stage_models() -> Dict[str, ActorCritic]:
    models: Dict[str, ActorCritic] = {}
    stage_ids = [
        "stage_0_random",
        "stage_1_early",
        "stage_2_mid",
        "stage_3_late",
        "stage_4_stabilized",
    ]
    for sid in stage_ids:
        model = ActorCritic(obs_dim=6, act_dim=3, hidden_dim=64)
        ckpt_file = CHECKPOINT_DIR / f"{sid}.pt"
        if ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models[sid] = model
    return models


STAGE_MODELS = load_stage_models()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(VIEWER_HTML)


@app.get("/aero")
async def aero_viewer() -> FileResponse:
    return FileResponse(AERO_HTML)


@app.get("/api/aero-config")
async def get_aero_config() -> JSONResponse:
    return JSONResponse(AirframeAeroConfig().to_dict())


@app.get("/api/training-progress")
async def get_training_progress() -> JSONResponse:
    progress_file = ARTIFACTS_DIR / "step1_training_progress.json"
    if not progress_file.exists():
        return JSONResponse({"error": "Training artifacts not found."}, status_code=404)
    data = json.loads(progress_file.read_text(encoding="utf-8"))
    return JSONResponse(data)


@app.websocket("/ws/sim")
async def sim_websocket(ws: WebSocket) -> None:
    await ws.accept()
    env = TetheredDroneEnv(initial_tilt_range_deg=24.0, terminate_on_wire_limit=False)
    obs, info = env.reset(
        seed=42,
        options={
            "initial_angles_deg": [22.0, -18.0, 12.0],
            "initial_omega": [1.2, -1.0, 0.8],
        },
    )
    current_stage = "stage_4_stabilized"
    manual_mode = True
    manual_action = np.array([0.0, 0.0, 0.14], dtype=np.float32)
    paused = False

    try:
        while True:
            try:
                msg_raw = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                msg = json.loads(msg_raw)
                cmd = msg.get("type")
                if cmd == "set_stage":
                    stage_id = msg.get("stage_id", "stage_4_stabilized")
                    if stage_id in STAGE_MODELS:
                        current_stage = stage_id
                        manual_mode = False
                        obs, info = env.reset(
                            options={
                                "initial_angles_deg": [22.0, -18.0, 12.0],
                                "initial_omega": [1.2, -1.0, 0.8],
                            }
                        )
                elif cmd == "set_manual":
                    manual_mode = bool(msg.get("enabled", True))
                    if "servo_1" in msg:
                        manual_action[0] = float(msg["servo_1"])
                    if "servo_2" in msg:
                        manual_action[1] = float(msg["servo_2"])
                    if "motor_speed" in msg:
                        manual_action[2] = float(msg["motor_speed"])
            except asyncio.TimeoutError:
                pass

            if not paused:
                if manual_mode:
                    action = manual_action.copy()
                else:
                    model = STAGE_MODELS.get(current_stage, STAGE_MODELS["stage_4_stabilized"])
                    action = model.act_deterministic(obs)
                obs, reward, _, _, info = env.step(action)
            else:
                reward = 0.0
                action = env.last_action

            await ws.send_text(
                json.dumps(
                    {
                        "stage_id": current_stage,
                        "manual_mode": manual_mode,
                        "reward": round(float(reward), 3),
                        "net_output": [round(float(x), 3) for x in action.tolist()],
                        "telemetry": info,
                    }
                )
            )
            await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/aero")
async def aero_websocket(ws: WebSocket) -> None:
    await ws.accept()
    sim = AeroGlideSimulator(mode="wind_tunnel")
    paused = False

    try:
        while True:
            try:
                msg_raw = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                msg = json.loads(msg_raw)
                cmd = msg.get("type")
                if cmd == "set_mode":
                    mode = str(msg.get("mode", "wind_tunnel"))
                    alt = float(msg.get("altitude_m", 200.0))
                    airspeed = float(msg.get("airspeed_ms", sim.tunnel_airspeed_ms))
                    sim.reset(
                        mode=mode,
                        initial_altitude_m=alt,
                        tunnel_airspeed_ms=airspeed,
                    )
                elif cmd == "reset":
                    alt = float(msg.get("altitude_m", sim.initial_drop_altitude_m))
                    angles = msg.get("initial_angles_deg")
                    sim.reset(
                        mode=sim.mode,
                        initial_altitude_m=alt,
                        initial_angles_deg=angles,
                        tunnel_airspeed_ms=sim.tunnel_airspeed_ms,
                    )
                elif cmd == "set_airspeed":
                    sim.set_tunnel_airspeed(float(msg.get("airspeed_ms", 28.0)))
                elif cmd == "set_mix":
                    sim.set_pitch_roll_mix(
                        pitch_flap_deg=float(msg.get("pitch_deg", 0.0)),
                        roll_flap_deg=float(msg.get("roll_deg", 0.0)),
                    )
                elif cmd == "set_servos":
                    sim.set_servos_deg(
                        servo_r_deg=float(msg.get("servo_r_deg", 0.0)),
                        servo_l_deg=float(msg.get("servo_l_deg", 0.0)),
                    )
                elif cmd == "set_flaps":
                    sim.set_flaps_deg(
                        delta_r_deg=float(msg.get("delta_r_deg", 0.0)),
                        delta_l_deg=float(msg.get("delta_l_deg", 0.0)),
                    )
                elif cmd == "kick_attitude":
                    d_angles = np.deg2rad(
                        np.asarray(msg.get("angles_deg", [0.0, 22.0, -15.0]), dtype=np.float64)
                    )
                    sim.angles = d_angles
                    from sim.aero_glide_sim import euler_to_rotation_matrix
                    sim.R_wb = euler_to_rotation_matrix(sim.angles[0], sim.angles[1], sim.angles[2])
                    sim.omega[:] = np.asarray(msg.get("omega_rads", [0.0, 0.0, 0.0]), dtype=np.float64)
                elif cmd == "set_paused":
                    paused = bool(msg.get("paused", False))
            except asyncio.TimeoutError:
                pass

            telemetry = sim.step() if not paused else sim.get_telemetry()
            telemetry["paused"] = paused
            await ws.send_text(json.dumps(telemetry))
            await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        return

