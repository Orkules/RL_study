# RL_study — 3D CAD Linkage Simulator, 3D Aerodynamic & Free-Drop Flight Simulator, and Step 1 Tethered RL

This repository contains **3 distinct simulations** built around the **9 printable `.STL` parts** in `FINAL FOR PRINT/`:

1. **Simulation 1 — 3D Mechanical CAD & 4-Bar Wing Flap Linkage Simulator (`viewer/index.html`, `sim/wing_linkage.py`):**
   - Served at **`http://localhost:8000/`** (and works 100% standalone by opening `viewer/index.html`).
   - Assembles all 9 `.STL` parts in millimeter coordinates, verifies flush SG90 servo placement, inspects the concentric pivot pins with adjustable shell opacity, and solves the exact forward/inverse 4-bar linkage kinematics (`MAX_WING_FLAP_DEG = 50.0` $\iff$ servo range `[-49.20°, +53.11°]`).
2. **Simulation 2 — 3D Aerodynamic & Free-Drop Flight Dynamics Simulator (`viewer/aero.html`, `sim/aero_glide_sim.py`):**
   - Served at **`http://localhost:8000/aero`** with live WebSocket telemetry at `/ws/aero` (and works 100% standalone by opening `viewer/aero.html` directly in a browser with all 9 `.STL` meshes embedded).
   - Models the **300 g** (`0.300 kg`) airframe (`156.3 mm` total length, `Ø43.2 mm` fuselage, `280.2 mm` wingspan) with its **Center of Mass (CoM)** at the `HEAD.STL`/`BODY.STL` joint (`91.3 mm` from tail base, `~58.4%` forward) and **Wing/Fin Center of Pressure (CoP)** at `28.0 mm` from tail base (`63.3 mm` behind CoM), providing strong passive weathercock stability in pitch and yaw.
   - Features both **Wind-Tunnel Mode** (`0..100 m/s` / `0..360 km/h` with fixed CoM and free 3D attitude rotation) and **Free-Drop Glide Mode** (`200 m` / `500 m` 3D world drop with gravity, 6-DOF rigid-body dynamics, trajectory ribbon trail, and follow camera).
3. **Simulation 3 — Step 1 RL Environment (`sim/tethered_drone_env.py`, `rl/ppo_agent.py`, `train_step1.py`):**
   - PPO reinforcement learning environment and trained actor-critic checkpoints (`checkpoints/stage_0_random.pt` .. `stage_4_stabilized.pt`).

For full geometric specifications, pin coordinates, and kinematics derivations, see [`docs/CAD_LINKAGE_SIMULATOR.md`](docs/CAD_LINKAGE_SIMULATOR.md).

---

## 1. Simulation 1: 3D Mechanical CAD & 4-Bar Wing Flap Linkage Simulator (`viewer/index.html`)

### Assembled `.STL` Parts (`FINAL FOR PRINT/`)

All 9 `.STL` meshes are embedded directly in `viewer/index.html` and `viewer/aero.html` (and mounted at `/stl` by `viewer/server.py`) and assembled in millimeter world coordinates (`+Y` up along the fuselage axis, `±X` along the left/right wingspan, `±Z` depth):

| Part File | Role in Assembly | Assembly Placement / Transformation |
|---|---|---|
| `BODY.STL` | Main fuselage & swept wings with hinge & servo pockets | Static shell (`dx=140.10, dy=0.00, dz=51.605 mm`), rendered semi-transparent (`22%` default opacity) |
| `HEAD.STL` | Upper aerodynamic nose cone (`Y = 91.29 .. 156.28 mm`) | Static shell (`dx=21.60, dy=91.29, dz=21.60 mm`), rendered semi-transparent (`25%` default opacity) |
| `MAIN CAGE-BASE.STL` | Internal lower avionics/servo bay frame | Static internal frame (`dx=9.95, dy=0.00, dz=19.90 mm`) |
| `MAIN CAGE-TOP.STL` | Internal upper structural cage | Static internal frame (`dx=19.95, dy=0.00, dz=19.95 mm`) |
| `BAT&AIRTAG CAGE.STL` | Upper internal battery & AirTag holder | Static internal frame (`dx=19.90, dy=49.00, dz=16.80 mm`) |
| `PARA BASKET.STL` | Parachute / lower bay basket | Static internal frame (`dx=21.60, dy=-2.00, dz=21.55 mm`) |
| `WING MOTOR ARM.STL` | Servo horn / crank arm (`×2`, Left & Right) | Rotates around the `BODY.STL` servo shaft bore `(Y = 35.8950 mm, Z = ±7.0000 mm)` |
| `CONROD.STL` | 4-bar connecting rod link (`×2`, Left & Right) | Couples the `WING MOTOR ARM.STL` tip pin to the `WING FLAP.STL` control horn pin |
| `WING FLAP.STL` | Elevon / wing flap control surface (`×2`, Left & Right) | Rotates purely about the fixed `BODY.STL` wing hinge axis `(Y = 21.2936 mm, Z = 0.0000 mm)` |

### Configurable Symmetric Wing Flap Limit (`MAX_WING_FLAP_DEG`) & Servo Ranges

Both `sim/wing_linkage.py`, `viewer/index.html`, and `viewer/aero.html` define:

```python
# Configurable symmetric wing flap limit (change to 40.0 or 50.0 as needed)
MAX_WING_FLAP_DEG: float = 50.0
```

| Target Symmetric Wing Flap Limit (`MAX_WING_FLAP_DEG`) | Exact Servo Angle for `-MAX` Flap | Exact Servo Angle for `0.0°` Center Flap | Exact Servo Angle for `+MAX` Flap | Relative Servo Travel Range |
|---:|---:|---:|---:|---|
| **`±50.0°`** (`MAX_WING_FLAP_DEG = 50.0`) | **`-49.20°`** | **`0.00°`** | **`+53.11°`** | **`[-49.20°, +53.11°]`** |
| **`±40.0°`** (`MAX_WING_FLAP_DEG = 40.0`) | **`-39.61°`** | **`0.00°`** | **`+41.70°`** | **`[-39.61°, +41.70°]`** |

---

## 2. Simulation 2: 3D Aerodynamic & Free-Drop Flight Dynamics Simulator (`viewer/aero.html` & `sim/aero_glide_sim.py`)

### Physical & Aerodynamic Specifications

- **Total Mass:** `300 g` (`0.300 kg`)
- **Airframe Dimensions:** Total length `156.3 mm` (`BODY.STL` `0..91.3 mm` + `HEAD.STL` `91.3..156.3 mm`), fuselage diameter `43.2 mm`, wingspan `280.2 mm`.
- **Center of Mass (CoM):** Located at the connection between `HEAD.STL` and `BODY.STL` at **`91.3 mm` from the tail base** (`~58.4%` forward toward the nose).
- **Aerodynamic Surfaces & Centers of Pressure:**
  1. **Swept Delta Wings (Left & Right):** Span `280.2 mm`, root chord `~57 mm` (`Y = 1.3 .. 58.1 mm`), Center of Pressure at `~28.0 mm` from the tail base (**`63.3 mm` behind the CoM** $\implies$ strong passive weathercock pitch stability $C_{m\alpha} < 0$).
  2. **Vertical Tail Fin & Wingtip Winglets (`BODY.STL`):** CoP `63.3 mm` behind the CoM, providing passive yaw weathercock stability ($C_{n\beta} > 0$) and aerodynamic roll/yaw rate damping ($C_{lp}, C_{nr}$).
  3. **Two Trailing-Edge Wing Flaps (`WING FLAP.STL`):** Span `87.6 mm` each (`|X| = 27.3 .. 114.9 mm`), chord `20.2 mm`, CoP at `X = ±71.1 mm` and `69.0 mm` behind the CoM, using a nonlinear aerodynamic model ($\sin\delta\cos\delta$ normal pressure lift + $\sin^2\delta$ flat-plate flap drag) driven through the exact 4-bar linkage (`MAX_WING_FLAP_DEG = 50.0`).

### Two Interactive Modes in `viewer/aero.html`

1. **Wind-Tunnel Mode (Interactive Speed & Attitude Testing):**
   - Holds the CoM (`91.3 mm` from tail) at the center of the 3D wind tunnel while the body rotates freely in 3D (Roll, Pitch, Yaw / Angle of Attack $\alpha$ and Sideslip $\beta$).
   - Interactive **Airspeed slider** (`0 m/s` to `100 m/s` / `0` to `360 km/h`) and **4-Bar Wing Flap / Servo sliders** (`Center Wings (0°)`, `Pitch Mix`, `Roll Mix`, `Right Servo`, `Left Servo`).
   - Displays live 3D Lift, Drag, Flap Normal Force, and Velocity vectors, animated airflow streamlines, and telemetry readouts ($\alpha, \beta, p, q, r, \text{Lift}, \text{Drag}, L/D, \text{Roll/Pitch/Yaw moments}$).
2. **Free-Drop Glide Mode (3D World Drop from Altitude):**
   - Releases the `300 g` airframe in a 3D world from configurable altitude (`200 m`, `500 m`, or `50..600 m`) under gravity (`9.81 m/s²`) with full 6-DOF rigid-body flight dynamics.
   - Camera follows the falling/gliding vehicle in 3D with a ground grid, 3D trajectory ribbon trail, velocity vector arrow, and live HUD while the user steers the servos in real time.

---

## Launching the Simulators

```bash
python3 -m uvicorn viewer.server:app --host 0.0.0.0 --port 8000
```

- **Simulation 1 (CAD & 4-Bar Linkage Review):** `http://localhost:8000/` (or open `viewer/index.html` directly)
- **Simulation 2 (3D Aerodynamic & Free-Drop Simulator):** `http://localhost:8000/aero` (or open `viewer/aero.html` directly)

### Running Unit Tests

```bash
python3 -m pytest -v
```
