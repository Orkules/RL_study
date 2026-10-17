# Simulation 2: 3D Aerodynamic Wind-Tunnel & Free-Drop Flight Dynamics Simulator

## Overview of the Simulator Suite

This repository provides two interactive 3D engineering & flight-physics simulators built around the **9 printable `.STL` parts** in `FINAL FOR PRINT/`:

1. **Simulation 1 — 3D Mechanical CAD & 4-Bar Linkage Review Simulator (`viewer/index.html` / `/`, `sim/wing_linkage.py`):**
   - Assembles all 9 `.STL` parts (`BODY.STL`, `HEAD.STL`, `MAIN CAGE-BASE.STL`, `MAIN CAGE-TOP.STL`, `BAT&AIRTAG CAGE.STL`, `PARA BASKET.STL`, and `×2` `WING MOTOR ARM.STL`, `CONROD.STL`, `WING FLAP.STL`) in millimeter coordinates.
   - Places two centered, flush SG90 servos (`X = 0.0 mm`, `Y = 28.0 mm`, `Z = ±7.0 mm`, output shafts at `Y = 35.8950 mm, Z = ±7.0000 mm`) and concentric metallic pivot pins (`Ø2.0 mm`).
   - Solves exact closed-form forward (`servo_deg_to_wing_flap_deg`) and inverse (`wing_flap_deg_to_servo_deg`) 4-bar linkage kinematics with configurable symmetric flap limits (`MAX_WING_FLAP_DEG = 50.0` $\iff$ servo range `[-49.20°, +53.11°]`; or `40.0°` $\iff$ `[-39.61°, +41.70°]`).
   - Documented in [`docs/CAD_LINKAGE_SIMULATOR.md`](CAD_LINKAGE_SIMULATOR.md).

2. **Simulation 2 — 3D Aerodynamic Wind-Tunnel & Free-Drop Flight Dynamics Simulator (`viewer/aero.html` / `/aero`, `sim/aero_glide_sim.py`):**
   - Models the **300 g (`0.300 kg`)** delta-wing airframe in 3D airflow and full 6-DOF free-drop flight, backed by MuJoCo (`mujoco`) rigid-body & ellipsoid fluid dynamics (`build_mujoco_xml`) and gimbal-lock-free $SO(3)$ rotation matrix integration (`integrate_rotation_so3`).
   - Directly couples the exact 4-bar servo linkage (`wing_flap_deg_to_servo_deg`) to the aerodynamic trailing-edge elevon flaps (`WING FLAP.STL`).
   - Runs both via the FastAPI server (`http://localhost:8000/aero` + `/ws/aero`) and **100% standalone** by opening `viewer/aero.html` directly in any browser (all 9 `.STL` meshes and the 100 Hz 6-DOF $SO(3)$ solver are embedded in `viewer/aero.html`).

---

## 1. Physical, Mass & Aerodynamic Geometry (`AirframeAeroConfig`)

All physical dimensions and moment arms in `sim/aero_glide_sim.py` and `viewer/aero.html` are derived directly from the assembled `FINAL FOR PRINT/*.STL` geometry:

| Parameter | Symbol / Field | Value | Physical / CAD Basis |
|---|---|---|---|
| **Total Mass** | `mass_kg` | **`0.300 kg` (`300 g`)** | Complete airframe with servos, battery, and nose cone |
| **Total Airframe Length** | `total_length_m` | **`0.1563 m` (`156.3 mm`)** | `BODY.STL` (`0 .. 91.3 mm`) + `HEAD.STL` (`91.3 .. 156.3 mm`) |
| **Fuselage Outer Diameter** | `fuselage_diameter_m` | **`0.0432 m` (`43.2 mm`)** | Cylindrical body diameter (`R = 21.6 mm`) |
| **Full Wingspan** | `wingspan_m` | **`0.2802 m` (`280.2 mm`)** | Swept delta wingtip-to-wingtip span (`X ∈ [-140.1, +140.1] mm`) |
| **Mean Aerodynamic Chord** | `mean_aero_chord_m` | **`0.0420 m` (`42.0 mm`)** | Delta wing reference chord $c$ |
| **Center of Mass (CoM)** | `com_from_tail_m` | **`0.0913 m` (`91.3 mm`)** | Located at the `HEAD.STL` / `BODY.STL` junction (`~58.4%` forward from tail base) |
| **Delta Wing CoP** | `wing_cop_from_tail_m` | **`0.0280 m` (`28.0 mm`)** | **`63.3 mm` behind CoM** (`wing_cop_x_body_m = -0.0633 m`) |
| **Vertical Tail Fin & Winglets CoP** | `fin_cop_from_tail_m` | **`0.0280 m` (`28.0 mm`)** | **`63.3 mm` behind CoM** (`fin_cop_x_body_m = -0.0633 m`) |
| **Trailing-Edge Wing Flap CoP** | `flap_cop_from_tail_m` | **`0.0223 m` (`22.3 mm`)** | **`69.0 mm` behind CoM** (`flap_cop_x_body_m = -0.0690 m`) |
| **Wing Flap Spanwise Moment Arm** | `flap_cop_span_y_m` | **`±0.0711 m` (`±71.1 mm`)** | Mid-span of each `87.6 mm × 20.2 mm` `WING FLAP.STL` (`|X| = 27.3 .. 114.9 mm`) |
| **Principal Moments of Inertia** | `(ixx, iyy, izz)` | **`(2.8, 6.8, 8.2) × 10⁻⁴ kg·m²`** | Roll ($I_{xx}$), Pitch ($I_{yy}$), and Yaw ($I_{zz}$) inertia about CoM |

---

## 2. Aerodynamic Force, Moment & Linkage Equations

### 2.1 Relative Airflow, Angle of Attack ($\alpha$) & Sideslip ($\beta$)

Given the body-to-world rotation matrix $\mathbf{R}_{wb} \in SO(3)$ and vehicle velocity $\mathbf{v}_{\text{NED}}$ (or freestream wind-tunnel speed $V_\infty$ along North), the relative airflow velocity in body coordinates is:

\[
\mathbf{v}_{\text{rel},b} = \begin{bmatrix} u \\ v \\ w \end{bmatrix} = \mathbf{R}_{wb}^\top \left(\mathbf{v}_{\text{NED}} - \mathbf{v}_{\text{wind,NED}}\right), \qquad V = \|\mathbf{v}_{\text{rel},b}\|, \qquad q_{\infty} = \frac{1}{2}\rho(h)V^2
\]

where air density $\rho(h) = 1.225 \exp(-h / 9500)\text{ kg/m}^3$, and:

\[
\alpha = \operatorname{atan2}(w, u), \qquad \beta = \arcsin\!\left(\operatorname{clip}\!\left(\frac{v}{V}, -1, 1\right)\right)
\]

### 2.2 Swept Delta Wing, Fuselage & Vertical Fin Weathercock Stability

- **Swept Delta Wing Lift & Polhamus Vortex Lift:**
  \[
  C_{L,\text{wing}}(\alpha) = C_{L\alpha}\sin\alpha\cos\alpha + C_{L,\text{vortex}}\sin^2\alpha\cos\alpha\operatorname{sign}(\alpha)
  \]
- **Body + Wing Drag Polar:**
  \[
  C_{D,\text{wing}}(\alpha, \beta) = C_{D0} + 1.35\sin^2\alpha + 0.35\sin^2\beta
  \]
- **Passive Pitch & Yaw Weathercock Restoring Moments:**
  Because the Delta Wing CoP and Vertical Fin CoP lie **`63.3 mm` aft of the CoM** ($\mathbf{r}_{\text{wing},b} = \mathbf{r}_{\text{fin},b} = [-0.0633, 0, 0]^\top\text{ m}$), any positive Angle of Attack ($\alpha > 0$) generates an upward wing normal force behind the CoM, creating a strong nose-down restoring pitch moment ($C_{m\alpha} < 0$). Similarly, any sideslip ($\beta \neq 0$) produces a lateral fin force behind the CoM, creating a strong yaw weathercock restoring moment ($C_{n\beta} > 0$).

### 2.3 Trailing-Edge Elevon Flaps & Software-Matched Servo Mixing

Each `WING FLAP.STL` (`S_{\text{flap}} = 1.7695 \times 10^{-3}\text{ m}^2$) is positioned at $\mathbf{r}_{\text{flap},R/L} = [-0.0690, \pm 0.0711, 0]^\top\text{ m}$ relative to the CoM and experiences local velocity $\mathbf{v}_{\text{loc}} = \mathbf{v}_{\text{rel},b} + \boldsymbol{\omega} \times \mathbf{r}_{\text{flap}}$.

Because one servo's polarity is inverted in firmware so **both servos share the exact same sign convention**:
- **Symmetric Pitch Mix ($\delta_R = \delta_L = \delta_{\text{pitch}}$):** Both flaps deflect together to pitch the nose up or down and trim the vehicle to a steady Angle of Attack $\alpha_{\text{trim}}$.
- **Differential Roll Mix ($\delta_R = +\delta_{\text{roll}},\, \delta_L = -\delta_{\text{roll}}$):** The right and left flaps deflect in opposite directions, generating pure axial roll torque $\tau_x$ about the fuselage centerline.
- **Combined Elevon Mixing (`set_pitch_roll_mix`):**
  \[
  \delta_R = \operatorname{clip}(\delta_{\text{pitch}} + \delta_{\text{roll}}, -50^\circ, +50^\circ), \qquad \delta_L = \operatorname{clip}(\delta_{\text{pitch}} - \delta_{\text{roll}}, -50^\circ, +50^\circ)
  \]
  and the exact SG90 servo angles are computed continuously via `wing_flap_deg_to_servo_deg`.

### 2.4 Gimbal-Lock-Free $SO(3)$ Attitude Integration

Because free-drop launches begin at **$-90^\circ$ pitch (nose straight down)** where standard ZYX Euler angle kinematic differential equations suffer from $\cos(\pm 90^\circ) = 0$ gimbal lock, both `sim/aero_glide_sim.py` and `viewer/aero.html` integrate the full $3\times 3$ rotation matrix $\mathbf{R}_{wb} \in SO(3)$ directly via the exact Rodrigues exponential map at $400\text{ Hz}$ ($4$ substeps per $10\text{ ms}$ frame):

\[
\mathbf{R}_{wb}(t + \Delta t) = \mathbf{R}_{wb}(t)\exp\!\left([\boldsymbol{\omega}\Delta t]_\times\right)
\]

---

## 3. Detailed Breakdown of the Two Operating Modes (`viewer/aero.html`)

### Mode 1: 3D Wind-Tunnel Mode (`SIM_MODE_WIND_TUNNEL`)

- **Fixed CoM with Free 3-DOF Attitude Rotation:** The Center of Mass (`91.3 mm` from the tail base, marked by the yellow sphere) is pinned at the center of the 3D wind tunnel `(0, 0, 0)` while the airframe rotates freely in Roll, Pitch, and Yaw under aerodynamic moments and gyroscopic coupling.
- **Freestream Airspeed Control (`0 .. 100 m/s` / `0 .. 360 km/h`):** Interactive slider plus one-click presets for `15 m/s (54 km/h)`, `35 m/s (126 km/h)`, `60 m/s (216 km/h)`, and `90 m/s (324 km/h)`, accompanied by animated 3D airflow streamlines whose speed scales with airspeed.
- **Weathercock Stability & Disturbance Testing:** Clicking **`Kick AoA (+25° Pitch / -18° Yaw)`** perturbs the nose attitude; with flaps centered (`0°`), the CoP `63.3 mm` behind the CoM rapidly weathercocks the airframe back to $\alpha \approx 0^\circ, \beta \approx 0^\circ$.
- **Live 3D Force & Velocity Vector Arrows:**
  - **Yellow Sphere:** Center of Mass (`91.3 mm` from tail).
  - **Pink Sphere:** Wing/Fin Center of Pressure (`28.0 mm` from tail, `63.3 mm` behind CoM).
  - **Green Arrow:** Total aerodynamic Lift vector.
  - **Red Arrow:** Total aerodynamic Drag vector.
  - **Cyan Arrows (`×2`):** Right and Left Wing Flap normal force vectors at `X = ±71.1 mm`.
  - **Purple Arrow:** Relative airspeed / velocity vector.

### Mode 2: 3D World Free-Drop Glide Mode (`SIM_MODE_FREE_DROP`)

- **Nose-Down (`-90°` Pitch) Release from Rest:** Just like dropping a real vertical munition from a carrier drone, each drop starts suspended stationary in the air with the nose pointing straight down (`pitch = -90.0°`, `V ≈ 0 m/s`) at a configurable release altitude (`50 .. 600 m` slider, plus one-click **`Drop from 200 m`** and **`Drop from 500 m`** buttons).
- **Clear Sky-Blue & Matte-Green Ground World (Zero Fog):** Uses a high-contrast sky-blue atmosphere (`#38bdf8`) and flat matte-green ground plane (`#2d6a4f`) with distance rings (`100 m`, `200 m`, `400 m`, `600 m`), a vertical plumb line, and a ground shadow marker directly beneath the vehicle so descent toward the ground is unmistakable.
- **3D Orange Trajectory Trail & Live 2D Trajectory Profile Map:**
  - In the 3D viewport, a continuous red/orange trajectory line and 3D breadcrumb spheres trace the complete flight path from the release point all the way to ground touch-down (`Alt = 0 m`).
  - In the bottom-right corner, a live **2D Side-Profile Trajectory Map (`Full Drop Trajectory Path: Altitude vs. Distance`)** plots the exact pull-out curve and glide slope from release (`(0, H_0)`) to current horizontal distance and altitude.
- **Camera View Presets:**
  - **`Follow Object`:** Tracks alongside and slightly above the falling/gliding airframe so the vehicle, its 4-bar flap motion, the trailing orange trajectory path, the sky above, and the green ground below are all simultaneously visible.
  - **`Full Trajectory View`:** Zooms out to frame the entire drop column from release altitude to ground impact, dynamically scaling the airframe and trail markers so the full trajectory arc is clearly visible.
  - **`Side AoA View`** & **`Rear Tail View`:** Inspect pitch angle of attack and differential roll deflection close-up.
- **Bottom-Left 3-Arrow `X · Y · Z` World Axes Gizmo:** Live 3D coordinate triad (`X` Red Forward/North, `Y` Green Right/Span, `Z` Blue Up/Altitude) that rotates dynamically with the camera orientation.

---

## 4. Summary of Aerodynamic & Free-Drop Glide Performance

### 4.1 Best Steady Glide Trim (`L/D` Ratio)

Sweeping symmetric Pitch Flap Mix across `[0°, +50°]` in `AeroGlideSimulator`:

| Metric | Value | Notes |
|---|---|---|
| **Optimal Symmetric Pitch Flap** | **`+26.5°`** (`δ_R = δ_L = +26.5°`) | Corresponding SG90 Servo Angle: **`+27.22°`** (`wing_flap_deg_to_servo_deg(26.5)`) |
| **Trimmed Angle of Attack ($\alpha_{\text{trim}}$)** | **`+9.89°` (`~9.9°`)** | Static pitch equilibrium where flap nose-up moment balances wing nose-down restoring moment |
| **Peak Lift-to-Drag Ratio ($L/D_{\max}$)** | **`1.56 : 1`** (`1.558`) | Typical for a low-aspect-ratio (`AR ≈ 7.0` span/chord, compact `Ø43.2 mm` fuselage) swept delta munition |
| ** Equilibrium Glide Slope Angle ($\gamma$)** | **`-32.7°`** | $\gamma = -\arctan(1 / 1.558)$, giving **`1.56 m` of forward travel per `1.0 m` of altitude lost** in steady glide |
| **Steady Equilibrium Glide Speed** | **`~35.6 m/s` (`128 km/h`)** | Speed where $L \cos|\gamma| + D \sin|\gamma| = m g$ (`2.943 N` for `300 g`) |
| **Zero-Flap Vertical Terminal Velocity** | **`~61.7 m/s` (`222 km/h`)** | Ballistic vertical dive with flaps neutral (`δ_R = δ_L = 0°`, `1.00 G` drag equilibrium) |

### 4.2 Horizontal Glide Reach vs. Release Altitude

When released from rest (`0 m/s`) pointing straight down (`-90°` pitch), the airframe first spends ~`60–85 m` of vertical drop accelerating and pulling its nose up from `-90°` into the `-32.7°` equilibrium glide slope before settling into steady `1.56 : 1` glide:

| Release Altitude ($H_0$) | Horizontal Reach (Fixed `+26.5°` Flap from Release) | Max Horizontal Reach (Optimal Pull-Out + `+26.5°` Glide) | Time to Ground Impact |
|---:|---:|---:|---:|
| **`100 m`** | **`23.8 m`** | **`~75 m`** | `5.0 s` |
| **`200 m`** | **`102.0 m`** | **`~205 m`** | `8.4 s` |
| **`300 m`** | **`251.5 m`** | **`~355 m`** | `13.3 s` |
| **`500 m`** | **`571.0 m`** | **`~665 m`** | `24.5 s` |
| **`1,000 m`** | **`1,343.0 m` (`1.34 km`)** | **`~1,445 m` (`1.45 km`)** | `51.1 s` |

---

## 5. Running the Simulators & Automated Tests

### 5.1 Standalone Browser Mode (Zero Server Required)

Both `viewer/index.html` and `viewer/aero.html` have all 9 binary `.STL` files base64-embedded along with the exact 4-bar linkage and 6-DOF $SO(3)$ aerodynamic physics engine. Open either file directly in Chrome, Firefox, Edge, or Safari:
- `viewer/index.html` — **Simulation 1: 3D Mechanical CAD & 4-Bar Linkage Simulator**
- `viewer/aero.html` — **Simulation 2: 3D Aerodynamic Wind-Tunnel & Free-Drop Simulator**

### 5.2 FastAPI & WebSocket Server Mode

```bash
python3 -m uvicorn viewer.server:app --host 0.0.0.0 --port 8000
```

- **Simulation 1 (3D CAD & Linkage):** `http://localhost:8000/`
- **Simulation 2 (3D Aerodynamics & Free-Drop):** `http://localhost:8000/aero`
- **Live Aerodynamic WebSocket Endpoint:** `ws://localhost:8000/ws/aero`

### 5.3 Running the Unit Test Suite

```bash
python3 -m pytest -v
```

`tests/test_aero_glide_sim.py` verifies:
- `test_airframe_physical_and_cad_specs` (`300 g` mass, `91.3 mm` CoM, `28.0 mm` wing/fin CoP, `MAX_WING_FLAP_DEG = 50.0`).
- `test_weathercock_stability_in_pitch_and_yaw` (passive restoring moments $C_{m\alpha} < 0, C_{n\beta} > 0$ converge perturbed attitudes to $0^\circ$ AoA and $0^\circ$ sideslip).
- `test_symmetric_pitch_flap_deflection_trims_to_aoa` (`±25°` symmetric pitch flap trims to symmetric positive/negative AoA with zero roll torque).
- `test_differential_roll_flap_deflection_generates_roll_rate` (`±30°` differential flap deflection generates pure axial roll rate).
- `test_free_drop_accelerates_to_terminal_and_glide_velocity` (ballistic terminal velocity & positive-lift forward glide from `500 m`).
- `test_viewer_aero_html_and_server_routes` (embedded `.STL` integrity and FastAPI route verification).
