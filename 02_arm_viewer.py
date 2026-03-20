"""
02_arm_viewer.py - SO100 arm on a table, nicely lit, looping trajectory
========================================================================
Usage:
    mjpython 02_arm_viewer.py --urdf path/to/so100.urdf

Shows the arm on a wooden table with a white back wall and studio lighting.
The arm plays a slow looping sweep through its workspace. Press Q to quit.
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
IK_ITERS = 200
IK_ALPHA = 0.4
CTRL_HZ = 60
SEG_DUR = 3.0  # seconds per segment

TABLE_Y = 0.30
TABLE_HX, TABLE_HY, TABLE_THK = 0.50, 0.35, 0.015
LEG_H = 0.37

# World-space waypoints [x, y, z] for the looping demo sweep
WAYPOINTS = np.array([
    [ 0.00,  TABLE_Y - 0.10,  0.30],
    [ 0.15,  TABLE_Y - 0.15,  0.20],
    [ 0.00,  TABLE_Y - 0.22,  0.12],
    [-0.15,  TABLE_Y - 0.15,  0.20],
    [ 0.00,  TABLE_Y - 0.10,  0.30],
])


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

        scene_xml = f"""<mujoco model="so100_arm_viewer">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <headlight ambient="0.15 0.15 0.15" diffuse="0.5 0.5 0.5" specular="0 0 0"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>{assets}</asset>
  <worldbody>
    <light name="key" pos="1.0 1.0 2.0" dir="-0.4 -0.4 -1"
           diffuse="1.2 1.2 1.1" castshadow="true" cutoff="60" exponent="3"/>
    <light name="fill" pos="-1.0 0.5 1.5" dir="0.4 -0.2 -1"
           diffuse="0.4 0.4 0.5" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1"
          pos="0 0 -0.77" rgba="0.22 0.22 0.22 1"/>
    <geom name="backwall" type="box" size="1.5 0.02 1.0"
          pos="0 -0.90 0.20" rgba="0.95 0.95 0.93 1"/>
    <body name="table" pos="0 {TABLE_Y:.2f} 0">
      <geom name="tabletop" type="box"
            size="{TABLE_HX} {TABLE_HY} {TABLE_THK}" pos="0 0 -{TABLE_THK}"
            rgba="0.82 0.68 0.48 1"/>
      <geom name="leg_fl" type="box" size="0.02 0.02 {LEG_H}"
            pos=" {TABLE_HX-0.05:.3f}  {TABLE_HY-0.05:.3f} -{LEG_H+TABLE_THK*2:.3f}"
            rgba="0.65 0.53 0.36 1"/>
      <geom name="leg_fr" type="box" size="0.02 0.02 {LEG_H}"
            pos=" {TABLE_HX-0.05:.3f} -{TABLE_HY-0.05:.3f} -{LEG_H+TABLE_THK*2:.3f}"
            rgba="0.65 0.53 0.36 1"/>
      <geom name="leg_bl" type="box" size="0.02 0.02 {LEG_H}"
            pos="-{TABLE_HX-0.05:.3f}  {TABLE_HY-0.05:.3f} -{LEG_H+TABLE_THK*2:.3f}"
            rgba="0.65 0.53 0.36 1"/>
      <geom name="leg_br" type="box" size="0.02 0.02 {LEG_H}"
            pos="-{TABLE_HX-0.05:.3f} -{TABLE_HY-0.05:.3f} -{LEG_H+TABLE_THK*2:.3f}"
            rgba="0.65 0.53 0.36 1"/>
    </body>
    <body name="robot" pos="0 {TABLE_Y:.2f} 0">
      {wb}
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


def ik_to(model, data, target, ee_id):
    for _ in range(IK_ITERS):
        mujoco.mj_fwdPosition(model, data)
        err = target - data.xpos[ee_id]
        if np.linalg.norm(err) < 5e-4:
            break
        J = np.zeros((6, model.nv))
        mujoco.mj_jacBody(model, data, J[:3], J[3:], ee_id)
        Jp = J[:3]
        dq = IK_ALPHA * Jp.T @ np.linalg.solve(Jp @ Jp.T + IK_DAMPING * np.eye(3), err)
        data.qpos[:model.nv] += np.clip(dq, -0.1, 0.1)
        mujoco.mj_normalizeQuat(model, data.qpos)


def build_traj(model, data, ee_id):
    steps = int(SEG_DUR * CTRL_HZ)
    wps = list(WAYPOINTS) + [WAYPOINTS[0]]
    traj = []
    for i in range(len(wps) - 1):
        ik_to(model, data, wps[i], ee_id)
        q0 = data.qpos[:model.nv].copy()
        ik_to(model, data, wps[i + 1], ee_id)
        q1 = data.qpos[:model.nv].copy()
        for s in range(steps):
            t = s / steps
            t = t * t * (3 - 2 * t)  # smooth-step
            traj.append(q0 + t * (q1 - q0))
    return np.array(traj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    args = parser.parse_args()

    model, data = load_model(args.urdf)
    mujoco.mj_forward(model, data)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    if ee_id == -1:
        raise ValueError("Body 'gripper' not found in URDF.")

    print("Building looping trajectory (IK)...")
    traj = build_traj(model, data, ee_id)
    print(f"[✓] {len(traj)} frames ({len(traj)/CTRL_HZ:.1f}s loop) — press Q to quit\n")

    state = {"step": 0, "quit": False}

    def key_cb(k):
        if k == ord("Q"):
            state["quit"] = True

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.lookat[:] = [0, TABLE_Y, 0.20]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18

        while viewer.is_running() and not state["quit"]:
            t0 = time.time()
            data.qpos[:model.nv] = traj[state["step"] % len(traj)]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            state["step"] += 1
            time.sleep(max(0, 1 / CTRL_HZ - (time.time() - t0)))


if __name__ == "__main__":
    main()
