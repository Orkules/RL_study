# 3D Mechanical CAD & 4-Bar Wing Flap Linkage Simulator

## Overview & Purpose

The interactive 3D viewer (`viewer/index.html`, `viewer/server.py`) and Python kinematics module (`sim/wing_linkage.py`) form the **3D Mechanical CAD & 4-Bar Linkage Inspection Simulator** (**not the RL training simulation yet**).

Its purpose is to:
1. Load and assemble all **9 `.STL` CAD parts** from `FINAL FOR PRINT/` in their exact millimeter coordinate frames.
2. Visually and mathematically verify that all 6 linkage pivot pins pass concentrically through their mating bores while `WING FLAP.STL` rotates purely around the fixed `BODY.STL` wing hinge axis.
3. Derive the exact closed-form forward and inverse 4-bar linkage equations relating **SG90 Servo Angle** (`servo_deg`) and **Wing Flap Angle** (`flap_deg`), including physical `BODY.STL` hard-stop boundaries and configurable symmetric wing flap limits (`MAX_WING_FLAP_DEG = 50.0` or `40.0`).

---

## 1. 9-Part `.STL` Assembly (`FINAL FOR PRINT/`)

The assembly uses a right-handed millimeter coordinate system:
- **`X` axis:** Wingspan (`+X` Right Wing, `-X` Left Wing, `X = 0` fuselage centerline)
- **`Y` axis:** Vertical fuselage axis (`Y = 0` at `BODY.STL` base, `+Y` toward `HEAD.STL` nose cone)
- **`Z` axis:** Fuselage thickness/depth (`Z = 0` on the wing mid-plane / wing hinge axis)

| `.STL` File | Quantity | World Transform / Kinematic Role |
|---|---:|---|
| `BODY.STL` | 1 | Translated by `(-140.10, 0.00, -51.605) mm` so center is at `X = 0`, base at `Y = 0`, and wing hinge axis at `Z = 0`. Rendered semi-transparent (`0.22` opacity). |
| `HEAD.STL` | 1 | Translated by `(-21.60, +91.29, -21.60) mm` to sit atop `BODY.STL` (`Y = 91.29 .. 191.29 mm`). Rendered semi-transparent (`0.25` opacity). |
| `MAIN CAGE-BASE.STL` | 1 | Translated by `(-9.95, 0.00, -19.90) mm` inside the lower fuselage bay. |
| `MAIN CAGE-TOP.STL` | 1 | Translated by `(-19.95, 0.00, -19.95) mm` inside the mid fuselage bay. |
| `BAT&AIRTAG CAGE.STL` | 1 | Translated by `(-19.90, +49.00, -16.80) mm` in the upper fuselage bay. |
| `PARA BASKET.STL` | 1 | Translated by `(-21.60, -2.00, -21.55) mm` at the base of the fuselage. |
| `WING MOTOR ARM.STL` | 2 (R/L) | Rigid crank arm pivoting around the servo shaft axis at `(Y = 35.8950 mm, Z = ±7.0000 mm)`. |
| `CONROD.STL` | 2 (R/L) | Rigid connecting rod coupling the servo arm tip pin to the wing flap horn pin. |
| `WING FLAP.STL` | 2 (R/L) | Rigid control surface pivoting strictly around the fixed wing hinge axis `(Y = 21.2936 mm, Z = 0.0000 mm)`. |

### Centered Flush SG90 Servos

Each SG90 servo body (`19.9 × 23.0 × 11.8 mm`) is centered across `X = 0` (`X ∈ [-9.95, +9.95] mm`) so that it sits flush within the `±10.0 mm` internal servo bay walls of `BODY.STL` and `MAIN CAGE-BASE.STL`:
- **Right Wing Servo:** Case centered at `(X = 0.0, Y = 28.0, Z = +7.0) mm`; output shaft pin along `+X` at `(Y = 35.8950, Z = +7.0000) mm`.
- **Left Wing Servo:** Case centered at `(X = 0.0, Y = 28.0, Z = -7.0) mm`; output shaft pin along `-X` at `(Y = 35.8950, Z = -7.0000) mm`.

---

## 2. Sub-Millimeter Pin Bore Coordinates & 4-Bar Link Constants

Every bore center was extracted directly from the circular vertex loops of the `.STL` files:

| Parameter | Symbol | Exact Value | CAD Source |
|---|---|---:|---|
| Crank Arm Radius | `R_arm` | `13.0000 mm` | `WING MOTOR ARM.STL`: shaft bore `(3.8000, 3.3000, 15.2500)` to tip bore `(3.8000, 5.1000, 2.2500)` |
| Connecting Rod Length | `L_rod` | `16.9100 mm` | `CONROD.STL`: bore 1 `(2.2000, 0.7500, 2.2000)` to bore 2 `(19.1100, 0.7500, 2.2000)` |
| Wing Flap Horn Radius | `R_flap` | `13.0000 mm` | `WING FLAP.STL`: hinge axis `(Y=20.0122, Z=14.4555)` to horn bore `(Y=16.2892, Z=2.0000)` |
| Horn Offset at `0°` Flap | `(dZ0, dY0)` | `(+12.4555, -3.7230) mm` | $\phi_0 = \text{atan2}(-3.7230, 12.4555) = -16.6435^\circ$ |
| `BODY.STL` Wing Hinge Hole | `(Yh, Zh)` | `(21.2936, 0.0000) mm` | Inner hinge bracket (`X = ±27.0 mm`) & outer wingtip bracket (`X = ±124.0 mm`) |
| `BODY.STL` Servo Shaft Hole | `(Ys, Zs)` | `(35.8950, ±7.0000) mm` | Right servo at `Z = +7.0000 mm`, Left servo at `Z = -7.0000 mm` |
| Neutral Arm Angle (`0°` Flap) | `theta0_arm_deg` | `-13.4290°` | World crank angle $\theta$ where `flap_deg == 0.0°` and `servo_deg == 0.0°` |

---

## 3. Forward & Inverse 4-Bar Equations (`sim/wing_linkage.py`)

### Forward Kinematics: `servo_deg_to_wing_flap_deg(servo_deg)`

Given servo rotation $\Delta\theta_{\text{servo}}$ (in degrees from the `0°` neutral wing position):
1. World arm angle: $\theta = \text{rad}(-13.4290^\circ + \Delta\theta_{\text{servo}})$
2. Arm tip pin $A = (A_z, A_y)$ on the servo circle of radius $R_{\text{arm}} = 13.0\text{ mm}$:
   \[
   A_z = 7.0 + 13.0 \cos\theta, \quad A_y = 35.8950 + 13.0 \sin\theta
   \]
3. Flap horn pin $P = (P_z, P_y)$ is the intersection of the circle around hinge $H = (0.0, 21.2936)$ with radius $R_{\text{flap}} = 13.0\text{ mm}$ and the circle around $A$ with radius $L_{\text{rod}} = 16.91\text{ mm}$.
4. Wing flap angle:
   \[
   \delta_{\text{flap}} = \text{deg}\!\left(\text{atan2}(P_y - 21.2936,\, P_z - 0.0) - \phi_0\right)
   \]

### Inverse Kinematics: `wing_flap_deg_to_servo_deg(flap_deg)`

Given desired wing flap deflection $\delta_{\text{flap}} \in [-\text{MAX\_WING\_FLAP\_DEG}, +\text{MAX\_WING\_FLAP\_DEG}]$:
1. Horn angle $\phi = \phi_0 + \text{rad}(\delta_{\text{flap}})$ gives exact horn pin location $P = (P_z, P_y)$:
   \[
   P_z = 13.0 \cos\phi, \quad P_y = 21.2936 + 13.0 \sin\phi
   \]
2. Arm tip pin $A = (A_z, A_y)$ is found by intersecting the circle around servo shaft $S = (7.0, 35.8950)$ of radius $R_{\text{arm}} = 13.0\text{ mm}$ with the circle around $P$ of radius $L_{\text{rod}} = 16.91\text{ mm}$.
3. Required servo angle:
   \[
   \Delta\theta_{\text{servo}} = \text{deg}\!\left(\text{atan2}(A_y - 35.8950,\, A_z - 7.0)\right) - (-13.4290^\circ)
   \]

### Configurable Wing Flap Limit (`MAX_WING_FLAP_DEG`)

In `sim/wing_linkage.py` (and `viewer/index.html`):

```python
MAX_WING_FLAP_DEG: float = 50.0
```

Calling `get_servo_range_for_wing_limit(MAX_WING_FLAP_DEG)` returns the exact servo endpoints `(min_servo_deg, max_servo_deg)`:
- **`MAX_WING_FLAP_DEG = 50.0`** $\implies$ Servo range **`[-49.20°, +53.11°]`**
- **`MAX_WING_FLAP_DEG = 40.0`** $\implies$ Servo range **`[-39.61°, +41.70°]`**

### Physical Collision Limits on `BODY.STL`

- **Lower Protrusion Stop:** `-54.0°` servo angle $\implies$ `-55.1°` wing flap angle (lower eye of `CONROD.STL` / `WING FLAP.STL` horn meets the `1.87 mm` lower wing-root protrusion on `BODY.STL`).
- **Upper Body Wall Stop:** `+104.3°` servo angle $\implies$ `+72.0°` wing flap angle (upper edge of `WING MOTOR ARM.STL` meets the upper internal wing slot wall on `BODY.STL`).
