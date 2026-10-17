# RL_study — 3D STL CAD & 4-Bar Wing Flap Linkage Simulator + Step 1 Tethered Rig

> **Current Viewer Mode (`viewer/index.html`, `viewer/server.py`, `sim/wing_linkage.py`):**
> The interactive 3D web application served at `http://localhost:8000` is currently configured as the **3D Mechanical CAD & 4-Bar Linkage Inspection Simulator** (**NOT the RL training simulation yet**). It was built specifically to load and visualize all **9 printable `.STL` parts** from `FINAL FOR PRINT/`, verify the mechanical assembly and flush SG90 servo placement, inspect the concentric pivot pins with semi-transparent shell walls, and derive the exact closed-form mathematical relationship between the two servo motor angles and the two wing flap deflection angles before wiring the kinematics into the RL flight controller.

For full geometric specifications, pin coordinates, and kinematics derivations, see [`docs/CAD_LINKAGE_SIMULATOR.md`](docs/CAD_LINKAGE_SIMULATOR.md).

---

## 1. 3D Mechanical CAD & 4-Bar Wing Flap Linkage Simulator

### Assembled `.STL` Parts (`FINAL FOR PRINT/`)

All 9 `.STL` meshes are embedded directly in `viewer/index.html` (and mounted at `/stl` by `viewer/server.py`) and assembled in millimeter world coordinates (`+Y` up along the fuselage axis, `±X` along the left/right wingspan, `±Z` depth):

| Part File | Role in Assembly | Assembly Placement / Transformation |
|---|---|---|
| `BODY.STL` | Main fuselage & swept wings with hinge & servo pockets | Static shell (`dx=140.10, dy=0.00, dz=51.605 mm`), rendered semi-transparent (`22%` default opacity) |
| `HEAD.STL` | Upper aerodynamic nose cone (`Y = 91.29 .. 191.29 mm`) | Static shell (`dx=21.60, dy=91.29, dz=21.60 mm`), rendered semi-transparent (`25%` default opacity) |
| `MAIN CAGE-BASE.STL` | Internal lower avionics/servo bay frame | Static internal frame (`dx=9.95, dy=0.00, dz=19.90 mm`) |
| `MAIN CAGE-TOP.STL` | Internal upper structural cage | Static internal frame (`dx=19.95, dy=0.00, dz=19.95 mm`) |
| `BAT&AIRTAG CAGE.STL` | Upper internal battery & AirTag holder | Static internal frame (`dx=19.90, dy=49.00, dz=16.80 mm`) |
| `PARA BASKET.STL` | Parachute / lower bay basket | Static internal frame (`dx=21.60, dy=-2.00, dz=21.55 mm`) |
| `WING MOTOR ARM.STL` | Servo horn / crank arm (`×2`, Left & Right) | Rotates around the `BODY.STL` servo shaft bore `(Y = 35.8950 mm, Z = ±7.0000 mm)` |
| `CONROD.STL` | 4-bar connecting rod link (`×2`, Left & Right) | Couples the `WING MOTOR ARM.STL` tip pin to the `WING FLAP.STL` control horn pin |
| `WING FLAP.STL` | Elevon / wing flap control surface (`×2`, Left & Right) | Rotates purely about the fixed `BODY.STL` wing hinge axis `(Y = 21.2936 mm, Z = 0.0000 mm)` |

Two **SG90 micro servos** (`19.9 × 23.0 × 11.8 mm`) are mounted centered inside the fuselage (`X ∈ [-9.95, +9.95] mm`, flush with the `±10.0 mm` inner pocket walls) at `(X = 0.0, Y = 28.0, Z = ±7.0 mm)`, with their output shafts emerging on opposite sides (`+X` for Right Wing at `Z = +7.0 mm`, `-X` for Left Wing at `Z = -7.0 mm`).

### Exact Sub-Millimeter Pin Bore Coordinates & 4-Bar Link Lengths

Measured directly from the triangle vertex loops of the `.STL` meshes in `FINAL FOR PRINT/`:

- **Crank Radius (`WING MOTOR ARM.STL` shaft center to tip pin center):**
  - `R_arm = 13.0000 mm` (shaft bore center `(3.8000, 3.3000, 15.2500)` to tip pin bore `(3.8000, 5.1000, 2.2500)`)
- **Connecting Rod Length (`CONROD.STL` bore-to-bore center distance):**
  - `L_rod = 16.9100 mm` (bore 1 `(2.2000, 0.7500, 2.2000)` to bore 2 `(19.1100, 0.7500, 2.2000)`)
- **Flap Horn Radius (`WING FLAP.STL` hinge axis to control horn pin center):**
  - `R_flap = 13.0000 mm` (`ΔZ = +12.4555 mm`, `ΔY = -3.7230 mm` at `0°` neutral wing flap; $\sqrt{12.4555^2 + (-3.7230)^2} = 13.0000\text{ mm}$)
- **`BODY.STL` Fixed Wing Hinge Hole Axis:**
  - `Y = 21.2936 mm, Z = 0.0000 mm` (inner bracket pin at `X = ±27.0 mm`, outer wingtip pin at `X = ±124.0 mm`)
- **`BODY.STL` Fixed Servo Output Shaft Hole:**
  - `Y = 35.8950 mm, Z = ±7.0000 mm` (`Z = +7.0000 mm` for Right Servo, `Z = -7.0000 mm` for Left Servo)
- **Neutral Reference (`0.0°` Servo $\iff$ `0.0°` Wing Flap):**
  - World crank arm angle $\theta_0 = -13.4290^\circ$ corresponds to exactly `0.0°` wing flap deflection.

### Configurable Symmetric Wing Flap Limit (`MAX_WING_FLAP_DEG`) & Servo Ranges

Both `sim/wing_linkage.py` and `viewer/index.html` define a single top-level constant:

```python
# Configurable symmetric wing flap limit (change to 40.0 or 50.0 as needed)
MAX_WING_FLAP_DEG: float = 50.0
```

Changing `MAX_WING_FLAP_DEG` between `50.0` and `40.0` automatically recalculates the exact asymmetric servo angle range required to produce symmetric `[-MAX_WING_FLAP_DEG, +MAX_WING_FLAP_DEG]` wing deflection via `get_servo_range_for_wing_limit()` and `wing_flap_deg_to_servo_deg()`:

| Target Symmetric Wing Flap Limit (`MAX_WING_FLAP_DEG`) | Exact Servo Angle for `-MAX` Flap | Exact Servo Angle for `0.0°` Center Flap | Exact Servo Angle for `+MAX` Flap | Relative Servo Travel Range |
|---:|---:|---:|---:|---|
| **`±50.0°`** (`MAX_WING_FLAP_DEG = 50.0`) | **`-49.20°`** | **`0.00°`** | **`+53.11°`** | **`[-49.20°, +53.11°]`** |
| **`±40.0°`** (`MAX_WING_FLAP_DEG = 40.0`) | **`-39.61°`** | **`0.00°`** | **`+41.70°`** | **`[-39.61°, +41.70°]`** |

### Physical Hard Stops on `BODY.STL`

- **Lower Protrusion Hard Stop:** At **`-54.0°` servo angle** (**`-55.1°` wing flap**), the lower eye of `CONROD.STL` / `WING FLAP.STL` horn contacts the lower wing-root protrusion on `BODY.STL`.
- **Upper Body Wall Hard Stop:** At **`+104.3°` servo angle** (**`+72.0°` wing flap**), the upper blade edge of `WING MOTOR ARM.STL` contacts the upper wing slot wall of `BODY.STL`.
- Both `±50.0°` and `±40.0°` symmetric wing flap operating envelopes remain **100% clear** of the `BODY.STL` physical hard stops.

### Launching the 3D CAD & Linkage Simulator

```bash
python3 -m uvicorn viewer.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser to:
- Inspect the 9 `.STL` parts in 3D (`3D Full View`, `Zoom Pins & Linkage`, `Rear Wing View`) and adjust `BODY.STL` / `HEAD.STL` shell transparency.
- Use **Center Wings (0°)**, **Move Together (Roll)**, **Move Opposite (Pitch)**, or individual **Right/Left Servo Angle** sliders.
- Observe real-time **Right/Left Wing Flap** angles, **3D BODY Collision Check** status, and the live **Wing Flap (°) vs. Servo (°)** mathematical curve.

---

## 2. Step 1 RL Environment (Tethered Rig Baseline)

Alongside the 3D CAD linkage simulator, this repository includes the Step 1 PPO reinforcement learning environment (`sim/tethered_drone_env.py`, `rl/ppo_agent.py`, `train_step1.py`):

- **Body:** 20 cm tall vertical cylinder (`radius = 2.5 cm`, `mass = 180 g`) suspended at `h = 0.45 m` inside a test cage by **4 corner suspension wires**.
- **Neural Network Interface:**
  - **Inputs (6D IMU):** `[roll, pitch, yaw, gyro_x, gyro_y, gyro_z]`
  - **Outputs (3D Actuators):** `[servo_1, servo_2, motor_speed]`

### Running Tests

```bash
pytest
```
