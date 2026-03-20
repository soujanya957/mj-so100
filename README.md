# mujoco-so100

Simple MuJoCo scenes for the [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) robotic arm.

## Scenes

| Script | Description |
|--------|-------------|
| `01_simple_scene.py` | Minimal viewer — loads the arm, nothing else. Good starting point. |
| `02_arm_viewer.py` | Arm on a wooden table with studio lighting and a looping demo trajectory. |
| `03_keyboard_teleop.py` | Control the end-effector with arrow keys / W S. Orange sphere shows the IK target. |
| `04_mouse_teleop.py` | Click-and-drag teleop. Spawn 1–6 arms with `--n-robots N`. Same scene layout as the shadow art designer. |

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
pip install mujoco
```

On macOS, use `mjpython` (bundled with the wheel) instead of `python` to launch
viewer scripts. It sets up the required OpenGL context automatically:

```bash
# macOS
mjpython 01_simple_scene.py --urdf path/to/so100.urdf

# Linux / Windows
python 01_simple_scene.py --urdf path/to/so100.urdf
```

### 3. Install optional dependencies

```bash
pip install numpy opencv-python
```

`opencv-python` is only needed for the shadow-frame extraction in `04_mouse_teleop.py`.
Every script works without it — you'll just see a warning.

### 4. Get the SO100 URDF

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

## Shadow frames

`04_mouse_teleop.py` writes rendered shadow frames to `/tmp/shadow_frame.npy` at 20 Hz.
These can be read by a shadow viewer running in a second terminal:

```python
import numpy as np, time
while True:
    frame = np.load("/tmp/shadow_frame.npy")
    # do something with frame (uint8 H×W, 0=shadow 255=lit)
    time.sleep(0.05)
```

---

## Requirements summary

```
python  >= 3.10
mujoco  >= 3.0
numpy
opencv-python   (optional, for shadow extraction)
```
