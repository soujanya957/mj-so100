# mujoco-so100

Simple MuJoCo scenes for the [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) robotic arm.

## Scenes

| Script | Description |
|--------|-------------|
| `01_simple_scene.py` | Minimal viewer — loads the arm, nothing else. Good starting point. |
| `02_arm_viewer.py` | Arm on a wooden table with studio lighting and a looping demo trajectory. |
| `03_keyboard_teleop.py` | Control the end-effector with arrow keys / W S. Orange sphere shows the IK target. |
| `04_mouse_teleop.py` | Click-and-drag teleop. Spawn 1–6 arms with `--n-robots N`. |

---

## Setup

### 1. Create a conda environment

```bash
conda create -n mujoco-so100 python=3.11 -y
conda activate mujoco-so100
```

### 2. Install MuJoCo

MuJoCo 3.x ships as a Python wheel — no separate binary install needed.

```bash
pip install mujoco numpy
```

On macOS, use `mjpython` (bundled with the wheel) instead of `python` to launch
viewer scripts. It sets up the required OpenGL context automatically:

```bash
# macOS
mjpython 01_simple_scene.py --urdf path/to/so100.urdf

# Linux / Windows
python 01_simple_scene.py --urdf path/to/so100.urdf
```

### 3. Get the SO100 URDF

Clone the official repo (or copy your existing `SO100_urdf/` folder):

```bash
git clone https://github.com/TheRobotStudio/SO-ARM100
# URDF is at SO-ARM100/urdf/so100.urdf  (path may vary by release)
```

---

## Usage

```bash
conda activate mujoco-so100

# 01 — just look at the arm
mjpython 01_simple_scene.py --urdf SO100_urdf/so100.urdf

# 02 — presentation scene with looping motion
mjpython 02_arm_viewer.py --urdf SO100_urdf/so100.urdf

# 03 — keyboard teleop (arrow keys + W/S + R to reset)
mjpython 03_keyboard_teleop.py --urdf SO100_urdf/so100.urdf

# 04 — mouse drag teleop, 1 arm (default)
mjpython 04_mouse_teleop.py --urdf SO100_urdf/so100.urdf

# 04 — mouse drag teleop, 3 arms side by side
mjpython 04_mouse_teleop.py --urdf SO100_urdf/so100.urdf --n-robots 3
```

### Keyboard teleop controls

| Key | Action |
|-----|--------|
| `←` / `→` | Move EE left / right (X) |
| `↑` / `↓` | Move EE forward / back (Y, toward wall) |
| `W` / `S` | Move EE up / down (Z) |
| `R` | Reset to home |
| `Q` | Quit |

### Mouse teleop controls

In the MuJoCo viewer, **Ctrl+click** a colored sphere then drag it to move that arm.
Each arm gets its own color (red, green, blue, …).

Standard viewer controls still work: left-drag to orbit, scroll to zoom, right-drag to pan.

---

## Requirements

```
python  >= 3.10
mujoco  >= 3.0
numpy
```
