# RL_study — Step 1: 20cm Tethered Single-Propeller TVC Rocket-Drone

A step-by-step reinforcement learning simulation and interactive 3D test rig for a **20 cm cylindrical single-propeller thrust-vector-controlled (TVC) FPV drone**.

## Hardware & Physics Model (Step 1: Tethered Rig)

- **Body:** 20 cm tall vertical cylinder (`radius = 2.5 cm`, `mass = 180 g`) suspended at `h = 0.45 m` inside a test cage by **4 corner suspension wires** so it cannot touch the ground or translate during initial attitude learning.
- **Propulsion & Gimbal (Bottom-Mounted):**
  - 1 brushless motor + propeller at the bottom (`z = -0.10 m` from Center of Mass).
  - 2-axis servo gimbal (`servo_1` pitch, `servo_2` roll, `±18°` deflection).
- **Anti-Spin Fins & Slipstream Wind Physics:**
  - Single-propeller reaction torque is counteracted by **4 bottom passive aerodynamic fins** sitting in the propeller slipstream + ambient airflow (`v_flow`). At higher motor speeds, the slipstream dynamic pressure over the fins rejects body spin.
- **Neural Network Interface:**
  - **Inputs (6D IMU):** `[roll, pitch, yaw, gyro_x, gyro_y, gyro_z]`
  - **Outputs (3D Actuators):** `[servo_1, servo_2, motor_speed]`

## Training Progression (PPO from Scratch)

| Milestone Checkpoint | Training Update | Episode Return | Final Tilt (°) | Behavior |
|---|---:|---:|---:|---|
| `stage_0_random` | 0 | -311.2 | 39.6° | Falls immediately and hangs caught by the 4 suspension wires |
| `stage_1_early` | 25 | +281.1 | 38.0° | Discovers bottom motor thrust & gimbal direction; fights gravity for ~2.5s |
| `stage_2_mid` | 30 | +749.7 | 11.8° | Stays off the wires for the full episode with a visible wobble |
| `stage_3_late` | 38 | +1222.1 | 3.2° | Damps rotational oscillations using gyroscope feedback |
| `stage_4_stabilized` | 85 | +1316.6 | **0.87°** | Locked upright stabilization + active disturbance rejection |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the RL Agent from Scratch
```bash
python3 train_step1.py
```
Saves milestone checkpoints (`stage_0_random.pt` through `stage_4_stabilized.pt`) in `checkpoints/` and training telemetry in `artifacts/step1_training_progress.json`.

### 3. Enter the Interactive 3D Simulation Viewer
```bash
python3 -m uvicorn viewer.server:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser to orbit/zoom the 3D rig, switch between training stages (`0` to `4`), apply external roll/pitch/yaw impulses, or take manual control of the 2 servos and motor speed.
