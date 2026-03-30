# mujoco-so100

MuJoCo scenes and tools for the [SO-ARM100/101](https://github.com/TheRobotStudio/SO-ARM100) robotic arm. Includes sim-only viewers, keyboard/mouse teleop, and a real-robot bridge via [lerobot](https://github.com/huggingface/lerobot).

## Scripts

| Script | Description |
|--------|-------------|
| `01_simple_scene.py` | Minimal viewer — loads the arm, nothing else. Good starting point. |
| `02_arm_viewer.py` | Arm on a wooden table with studio lighting and a looping demo trajectory. |
| `03_keyboard_teleop.py` | Sim-only: control the end-effector with arrow keys / W S. Orange sphere shows the IK target. |
| `04_mouse_teleop.py` | Sim-only: click-and-drag teleop. Spawn 1–6 arms with `--n-robots N`. |
| `05_lerobot_teleop.py` | **Real robot teleop**: MuJoCo IK keyboard control that sends joint angles to a physical SO-101 via lerobot. |
| `06_calibrate_visual.py` | Visual calibration tool for wrist_roll joint mapping between sim and real robot. |

---

## Setup

### 1. Create a conda environment

```bash
conda create -n mujoco-so100 python=3.12 -y
conda activate mujoco-so100
```

### 2. Install dependencies

```bash
# Core (all scripts)
pip install mujoco numpy

# For real-robot teleop (05, 06) — requires lerobot with Dynamixel/Feetech support
pip install lerobot[dynamixel]
```

On macOS, use `mjpython` instead of `python` for viewer scripts.
On **Windows/Linux**, regular `python` works.

### 3. Get the URDF

URDF files are in `assets/SO100_urdf/` and `assets/SO101_urdf/` at the repo root.

---

## Usage

### Sim-only scripts (01–04)

```bash
# 01 — just look at the arm
python 01_simple_scene.py --urdf ../../assets/SO101_urdf/so101_new_calib.urdf

# 02 — presentation scene with looping motion
python 02_arm_viewer.py --urdf ../../assets/SO101_urdf/so101_new_calib.urdf

# 03 — keyboard teleop (sim only)
python 03_keyboard_teleop.py --urdf ../../assets/SO101_urdf/so101_new_calib.urdf

# 04 — mouse drag teleop, 3 arms
python 04_mouse_teleop.py --urdf ../../assets/SO101_urdf/so101_new_calib.urdf --n-robots 3
```

### Real robot teleop (05)

Controls a physical SO-101 arm via keyboard. MuJoCo provides the IK solver and visual
feedback; lerobot handles motor communication over serial.

```bash
# Sim-only mode (no robot connected) — test IK and controls
python 05_lerobot_teleop.py --urdf ../../assets/SO101_urdf/so101_new_calib.urdf

# With real robot on COM3
python 05_lerobot_teleop.py --urdf ../../assets/SO101_urdf/so101_new_calib.urdf --port COM3

# Diagnose a joint mapping (sweep + read back)
python 05_lerobot_teleop.py --urdf ../../assets/SO101_urdf/so101_new_calib.urdf --port COM3 --diagnose wrist_roll
```

#### Keyboard controls

| Key | Action |
|-----|--------|
| `←` / `→` | Move EE left / right (Y axis) |
| `↑` / `↓` | Move EE forward / back (X axis) |
| `W` / `S` | Move EE up / down (Z axis) |
| `A` / `D` | Roll wrist left / right |
| `O` / `C` | Open / close gripper |
| `R` | Reset to home position |
| `Q` | Quit |

#### Joint corrections

The mapping between MuJoCo joint angles and lerobot motor commands is defined by
`JOINT_CORRECTIONS` at the top of `05_lerobot_teleop.py`:

```python
JOINT_CORRECTIONS = {
    "shoulder_pan":  {"scale": 1.0, "offset": 0.0},
    "shoulder_lift": {"scale": 1.0, "offset": 0.0},
    "elbow_flex":    {"scale": 1.0, "offset": 0.0},
    "wrist_flex":    {"scale": 1.0, "offset": 90.0},
    "wrist_roll":    {"scale": 1.0, "offset": 105.0},
}
```

Each joint is mapped as: `real_degrees = mujoco_degrees * scale + offset`.

If a joint doesn't match between sim and real robot:
1. Use `--diagnose <joint_name>` to sweep the joint and see where sim/real diverge
2. Use `06_calibrate_visual.py` for visual alignment (sim shows target, you match by hand)
3. Update the correction values in the dict

### Visual calibration (06)

Interactive tool for calibrating wrist_roll. The sim shows target poses; you move the
real robot to match and press ENTER. Outputs the correct scale/offset values.

```bash
python 06_calibrate_visual.py --urdf ../../assets/SO101_urdf/so101_new_calib.urdf --port COM3
```

---

## How it works

### IK solver

All teleop scripts use MuJoCo's built-in Jacobian (`mj_jacBody`) with damped
least-squares to solve inverse kinematics. This runs entirely on CPU with no
external IK library (no placo needed). The solver controls `shoulder_pan`,
`shoulder_lift`, `elbow_flex`, and `wrist_flex` for 3D end-effector positioning.
`wrist_roll` and `gripper` are controlled independently via keyboard.

### Real robot bridge (05)

The bridge converts MuJoCo's joint solution (`data.qpos` in radians) to lerobot
motor commands (degrees with calibration offsets), then sends them via
`SOFollower.send_action()` over the Feetech serial bus at 30 Hz.

---

## Requirements

```
python   >= 3.10
mujoco   >= 3.0
numpy
lerobot  (for scripts 05, 06 only)
```
