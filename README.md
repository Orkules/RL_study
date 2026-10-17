# RL_study — 3D CAD Linkage Simulator, 3D Aerodynamic & Free-Drop Flight Simulator, and Step 1 Tethered RL

This repository contains **3 distinct simulations** built around the **9 printable `.STL` parts** in `FINAL FOR PRINT/`:

1. **Simulation 1 — 3D Mechanical CAD & 4-Bar Wing Flap Linkage Simulator (`viewer/index.html`, `sim/wing_linkage.py`):**
   - Served at **`http://localhost:8000/`** (and works 100% standalone by opening `viewer/index.html`).
   - Assembles all 9 `.STL` parts in millimeter coordinates, verifies centered flush SG90 servo placement (`X = 0.0 mm, Y = 28.0 mm, Z = ±7.0 mm`), inspects the concentric pivot pins with adjustable shell opacity, and solves the exact forward/inverse 4-bar linkage kinematics (`MAX_WING_FLAP_DEG = 50.0` $\iff$ servo range `[-49.20°, +53.11°]`, `wing_flap_deg_to_servo_deg`).
   - Full documentation: **[`docs/CAD_LINKAGE_SIMULATOR.md`](docs/CAD_LINKAGE_SIMULATOR.md)**
2. **Simulation 2 — 3D Aerodynamic Wind-Tunnel & Free-Drop Flight Dynamics Simulator (`viewer/aero.html`, `sim/aero_glide_sim.py`):**
   - Served at **`http://localhost:8000/aero`** with live WebSocket telemetry at `/ws/aero` (and works 100% standalone by opening `viewer/aero.html` directly in a browser with all 9 `.STL` meshes embedded).
   - Models the **300 g** (`0.300 kg`) airframe (`156.3 mm` total length, `Ø43.2 mm` fuselage, `280.2 mm` wingspan) with its **Center of Mass (CoM)** at the `HEAD.STL`/`BODY.STL` joint (`91.3 mm` from tail base, `~58.4%` forward) and **Wing/Fin Center of Pressure (CoP)** at `28.0 mm` from tail base (`63.3 mm` behind CoM), backed by MuJoCo (`mujoco`) rigid-body & ellipsoid fluid dynamics and gimbal-lock-free $SO(3)$ rotation matrix integration (`integrate_rotation_so3`).
   - Full documentation: **[`docs/AERODYNAMIC_SIMULATOR.md`](docs/AERODYNAMIC_SIMULATOR.md)**
3. **Simulation 3 — Step 1 RL Environment (`sim/tethered_drone_env.py`, `rl/ppo_agent.py`, `train_step1.py`):**
   - PPO reinforcement learning environment and trained actor-critic checkpoints (`checkpoints/stage_0_random.pt` .. `stage_4_stabilized.pt`).

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

## 2. Simulation 2: 3D Aerodynamic Wind-Tunnel & Free-Drop Flight Dynamics Simulator (`viewer/aero.html` & `sim/aero_glide_sim.py`)

See **[`docs/AERODYNAMIC_SIMULATOR.md`](docs/AERODYNAMIC_SIMULATOR.md)** for complete equations, MuJoCo MJCF formulation, and aerodynamic analysis.

### Physical & Aerodynamic Specifications

- **Total Mass:** `300 g` (`0.300 kg`)
- **Airframe Dimensions:** Total length `156.3 mm` (`BODY.STL` `0..91.3 mm` + `HEAD.STL` `91.3..156.3 mm`), fuselage diameter `43.2 mm`, wingspan `280.2 mm`, mean aerodynamic chord `42.0 mm`.
- **Center of Mass (CoM):** Located at the connection between `HEAD.STL` and `BODY.STL` at **`91.3 mm` from the tail base** (`~58.4%` forward toward the nose).
- **Aerodynamic Surfaces & Centers of Pressure:**
  1. **Swept Delta Wings (Left & Right):** Span `280.2 mm`, planform area `0.0112 m²`, Center of Pressure at `28.0 mm` from the tail base (**`63.3 mm` behind the CoM** $\implies$ strong passive weathercock pitch stability $C_{m\alpha} < 0$).
  2. **Vertical Tail Fin & Wingtip Winglets (`BODY.STL`):** Reference area `0.0045 m²`, CoP at `28.0 mm` from tail (**`63.3 mm` behind the CoM**), providing passive yaw weathercock stability ($C_{n\beta} > 0$) and aerodynamic roll/yaw rate damping ($C_{lp}, C_{nr}$).
  3. **Two Trailing-Edge Wing Flaps (`WING FLAP.STL`):** Span `87.6 mm` each (`|X| = 27.3 .. 114.9 mm`), chord `20.2 mm`, CoP at `X = ±71.1 mm` and `22.3 mm` from tail (**`69.0 mm` behind the CoM**), driven through the exact 4-bar linkage (`MAX_WING_FLAP_DEG = 50.0`) with software-matched servo polarity (`Pitch Mix` $\delta_R = \delta_L$, `Roll Mix` $\delta_R = -\delta_L$).

### Two Interactive Modes in `viewer/aero.html`

1. **Mode 1 — 3D Wind-Tunnel Mode (`SIM_MODE_WIND_TUNNEL`):**
   - Holds the CoM (`91.3 mm` from tail) at the center of the 3D wind tunnel while the body rotates freely in 3-DOF (Roll, Pitch, Yaw / Angle of Attack $\alpha$ and Sideslip $\beta$) using gimbal-lock-free $SO(3)$ integration.
   - Interactive **Freestream Airspeed slider** (`0 m/s` to `100 m/s` / `0` to `360 km/h`) with presets (`15, 35, 60, 90 m/s`), **`Kick AoA (+25° Pitch / -18° Yaw)`** disturbance button, and **4-Bar Wing Flap / Servo sliders** (`Center Wings (0°)`, `Pitch Mix`, `Roll Mix`, `Right Servo`, `Left Servo`).
   - Displays live 3D Lift (green), Drag (red), Flap Normal Force (cyan), and Velocity (purple) vector arrows, CoM (yellow) & CoP (pink) spheres, animated airflow streamlines, and telemetry readouts ($\alpha, \beta, p, q, r, \text{Lift}, \text{Drag}, L/D, \text{Roll/Pitch/Yaw moments}$).
2. **Mode 2 — 3D World Free-Drop Glide Mode (`SIM_MODE_FREE_DROP`):**
   - Releases the `300 g` airframe from rest (`0 m/s`) pointing straight down (**`-90°` nose-down pitch**) at a configurable altitude (`50 .. 600 m` slider, plus one-click `Drop from 200 m` and `Drop from 500 m` buttons) under gravity (`9.81 m/s²`).
   - High-contrast sky-blue atmosphere and matte-green ground plane (zero fog) with a 3D orange trajectory trail + breadcrumb spheres, a live bottom-right **2D Altitude-vs-Distance trajectory profile map**, **`Follow Object`** vs **`Full Trajectory View`** camera presets, and a bottom-left **3-arrow `X · Y · Z` World Axes gizmo**.

### Summary of Glide Performance

- **Peak Lift-to-Drag Ratio ($L/D_{\max}$):** **`1.56 : 1`** achieved at **`+26.5°` symmetric pitch flap** (`+27.22°` SG90 servo angle), trimming the airframe at **`+9.9°` Angle of Attack** (`-32.7°` glide slope angle at `~35.6 m/s` / `128 km/h`).
- **Horizontal Glide Reach vs. Release Altitude (from `-90°` vertical nose-down release at rest):**
  - **`100 m` drop:** `23.8 m` (fixed `+26.5°` flap) / `~75 m` (optimal pull-out)
  - **`300 m` drop:** `251.5 m` (fixed `+26.5°` flap) / `~355 m` (optimal pull-out)
  - **`500 m` drop:** `571.0 m` (fixed `+26.5°` flap) / `~665 m` (optimal pull-out)
  - **`1,000 m` drop:** `1,343.0 m` (`1.34 km`, fixed `+26.5°` flap) / `~1,445 m` (`1.45 km`, optimal pull-out)

---

## Launching the Simulators

### 1. FastAPI & WebSocket Server (`viewer/server.py`)

```bash
python3 -m uvicorn viewer.server:app --host 0.0.0.0 --port 8000
```

- **Simulation 1 (3D CAD & 4-Bar Linkage Review):** `http://localhost:8000/`
- **Simulation 2 (3D Aerodynamic Wind-Tunnel & Free-Drop Simulator):** `http://localhost:8000/aero`

### 2. Standalone Browser Mode (No Server Needed)

Open `viewer/index.html` or `viewer/aero.html` directly in any modern browser (`file://`). All 9 `.STL` meshes and the 4-bar linkage + 6-DOF $SO(3)$ aerodynamic physics engine are self-contained inside the HTML files.

### 3. Running Unit Tests

```bash
python3 -m pytest -v
```
