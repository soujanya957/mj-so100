"""
03_keyboard_teleop.py - Control SO100 end-effector with keyboard
================================================================
Usage:
    mjpython 03_keyboard_teleop.py --urdf path/to/so100.urdf

Keys:
    Arrow Left / Right   move EE left / right   (X axis)
    Arrow Up   / Down    move EE forward / back  (Y axis, toward wall)
    W / S                move EE up / down       (Z axis)
    R                    reset to home position
    Q                    quit

The orange sphere shows the current IK target.
EE world position is printed live to the terminal.
"""

import argparse
import os
import re
import shutil
import tempfile
import time

import numpy as np
import mujoco
import mujoco.viewer


IK_DAMPING = 1e-3
IK_ITERS = 40
IK_ALPHA = 0.5
CTRL_HZ = 60
STEP = 0.008  # metres moved per key event

TABLE_Y = 0.30
HOME = np.array([0.0, TABLE_Y - 0.10, 0.25])

# MuJoCo GLFW key codes
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265


def patch_urdf(xml):
    def add_collision(m):
        lc = m.group(1)
        if "<collision>" in lc:
            return m.group(0)
        mm = re.search(r'<mesh filename="([^"]+)"', lc)
        if not mm:
            return m.group(0)
        cb = (
            "\n    <collision>\n      <geometry>\n"
            f'        <mesh filename="{mm.group(1)}"/>\n'
            "      </geometry>\n    </collision>"
        )
        return m.group(0).replace("</link>", cb + "\n  </link>")

    return re.sub(r"<link[^>]*>(.*?)</link>", add_collision, xml, flags=re.DOTALL)


def load_model(urdf_path):
    abs_urdf = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(abs_urdf)

    with open(abs_urdf) as f:
        xml = patch_urdf(f.read())

    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "so100.urdf"), "w") as f:
        f.write(xml)
    for root, _, files in os.walk(urdf_dir):
        for fn in files:
            if fn.lower().endswith(".stl"):
                shutil.copy(os.path.join(root, fn), os.path.join(tmpdir, fn))

    orig = os.getcwd()
    os.chdir(tmpdir)
    try:
        m0 = mujoco.MjModel.from_xml_path("so100.urdf")
        mujoco.mj_saveLastXML("robot.xml", m0)
        with open("robot.xml") as f:
            robot_xml = f.read()

        asset_m = re.search(r"<asset>(.*?)</asset>", robot_xml, re.DOTALL)
        wb_m = re.search(r"<worldbody>(.*?)</worldbody>", robot_xml, re.DOTALL)
        assets = asset_m.group(1) if asset_m else ""
        wb = wb_m.group(1) if wb_m else ""

        scene_xml = f"""<mujoco model="so100_keyboard_teleop">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <headlight ambient="0.2 0.2 0.2" diffuse="0.55 0.55 0.55" specular="0 0 0"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>{assets}</asset>
  <worldbody>
    <light name="key" pos="1 1 2" dir="-0.4 -0.4 -1"
           diffuse="1.0 1.0 0.9" castshadow="true" cutoff="60" exponent="3"/>
    <geom name="floor" type="plane" size="3 3 0.1"
          pos="0 0 -0.77" rgba="0.28 0.28 0.28 1"/>
    <geom name="backwall" type="box" size="1.0 0.02 0.8"
          pos="0 -0.80 0.20" rgba="0.93 0.93 0.91 1"/>
    <body name="table" pos="0 {TABLE_Y:.2f} 0">
      <geom name="tabletop" type="box" size="0.50 0.35 0.015"
            pos="0 0 -0.015" rgba="0.82 0.68 0.48 1"/>
      <geom name="leg_fl" type="box" size="0.02 0.02 0.37"
            pos=" 0.45  0.30 -0.40" rgba="0.65 0.53 0.36 1"/>
      <geom name="leg_fr" type="box" size="0.02 0.02 0.37"
            pos=" 0.45 -0.30 -0.40" rgba="0.65 0.53 0.36 1"/>
      <geom name="leg_bl" type="box" size="0.02 0.02 0.37"
            pos="-0.45  0.30 -0.40" rgba="0.65 0.53 0.36 1"/>
      <geom name="leg_br" type="box" size="0.02 0.02 0.37"
            pos="-0.45 -0.30 -0.40" rgba="0.65 0.53 0.36 1"/>
    </body>
    <body name="robot" pos="0 {TABLE_Y:.2f} 0">
      {wb}
    </body>
    <!-- IK target marker — the orange sphere you're steering -->
    <body name="ee_target" pos="0 {TABLE_Y - 0.10:.2f} 0.25" mocap="true">
      <geom type="sphere" size="0.022" rgba="1 0.45 0 0.85"/>
    </body>
  </worldbody>
</mujoco>"""

        with open("scene.xml", "w") as f:
            f.write(scene_xml)
        model = mujoco.MjModel.from_xml_path("scene.xml")
    finally:
        os.chdir(orig)

    shutil.rmtree(tmpdir, ignore_errors=True)
    model.dof_damping[:] = 10.0
    return model, mujoco.MjData(model)


def ik_step(model, data, target, ee_id):
    for _ in range(IK_ITERS):
        mujoco.mj_fwdPosition(model, data)
        err = target - data.xpos[ee_id]
        if np.linalg.norm(err) < 5e-4:
            break
        J = np.zeros((6, model.nv))
        mujoco.mj_jacBody(model, data, J[:3], J[3:], ee_id)
        Jp = J[:3]
        dq = IK_ALPHA * Jp.T @ np.linalg.solve(Jp @ Jp.T + IK_DAMPING * np.eye(3), err)
        data.qpos[: model.nv] += np.clip(dq, -0.1, 0.1)
        mujoco.mj_normalizeQuat(model, data.qpos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    args = parser.parse_args()

    model, data = load_model(args.urdf)
    mujoco.mj_forward(model, data)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    if ee_id == -1:
        raise ValueError("Body 'gripper' not found in URDF.")

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_target")
    mocap_id = model.body_mocapid[target_body_id]

    target = HOME.copy()
    data.mocap_pos[mocap_id] = target.copy()

    # Key state: pressed once per event (key_callback fires on press, not hold)
    nudge = {"dx": 0.0, "dy": 0.0, "dz": 0.0}
    state = {"quit": False, "reset": False}

    def key_cb(k):
        if k == ord("Q"):
            state["quit"] = True
        elif k == ord("R"):
            state["reset"] = True
        elif k == KEY_RIGHT:
            nudge["dx"] += STEP
        elif k == KEY_LEFT:
            nudge["dx"] -= STEP
        elif k == KEY_UP:
            nudge["dy"] -= STEP  # forward = more negative Y (toward wall)
        elif k == KEY_DOWN:
            nudge["dy"] += STEP
        elif k == ord("W"):
            nudge["dz"] += STEP
        elif k == ord("S"):
            nudge["dz"] -= STEP

    print("Keys: ←/→=X  ↑/↓=Y(depth)  W/S=Z(height)  R=reset  Q=quit\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.lookat[:] = [0, TABLE_Y, 0.2]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18

        while viewer.is_running() and not state["quit"]:
            t0 = time.time()

            if state["reset"]:
                target = HOME.copy()
                data.qpos[:] = 0.0
                state["reset"] = False
                nudge["dx"] = nudge["dy"] = nudge["dz"] = 0.0

            # Consume accumulated nudges
            target[0] += nudge["dx"]
            target[1] += nudge["dy"]
            target[2] += nudge["dz"]
            nudge["dx"] = nudge["dy"] = nudge["dz"] = 0.0

            # Workspace clamp
            target[0] = np.clip(target[0], -0.28, 0.28)
            target[1] = np.clip(target[1], TABLE_Y - 0.40, TABLE_Y + 0.10)
            target[2] = np.clip(target[2], 0.02, 0.42)

            data.mocap_pos[mocap_id] = target.copy()
            ik_step(model, data, target, ee_id)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()

            ee = data.xpos[ee_id]
            print(
                f"\rEE [{ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f}]  "
                f"target [{target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}]",
                end="",
            )

            time.sleep(max(0, 1 / CTRL_HZ - (time.time() - t0)))

    print("\nDone.")


if __name__ == "__main__":
    main()
