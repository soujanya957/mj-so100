"""
06_calibrate_visual.py - Visual wrist_roll calibration
=======================================================
Usage:
    python 06_calibrate_visual.py --urdf ../../assets/SO101_urdf/so101_new_calib.urdf --port COM3

Shows 3 wrist_roll positions in the sim viewer.
For each, move the real robot's wrist to match, then press ENTER.
Outputs the correct offset to use in 05_lerobot_teleop.py.
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


WRIST_ROLL_POSES_DEG = [-45.0, 0.0, 45.0, 90.0]


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
    with open(os.path.join(tmpdir, "robot.urdf"), "w") as f:
        f.write(xml)
    for root, _, files in os.walk(urdf_dir):
        for fn in files:
            if fn.lower().endswith(".stl"):
                shutil.copy(os.path.join(root, fn), os.path.join(tmpdir, fn))
    orig = os.getcwd()
    os.chdir(tmpdir)
    try:
        m0 = mujoco.MjModel.from_xml_path("robot.urdf")
        mujoco.mj_saveLastXML("robot.xml", m0)
        with open("robot.xml") as f:
            robot_xml = f.read()
        asset_m = re.search(r"<asset>(.*?)</asset>", robot_xml, re.DOTALL)
        wb_m = re.search(r"<worldbody>(.*?)</worldbody>", robot_xml, re.DOTALL)
        assets = asset_m.group(1) if asset_m else ""
        wb = wb_m.group(1) if wb_m else ""
        scene_xml = f"""<mujoco model="so101_calibration">
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
    <body name="robot" pos="0 0 0">
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


def connect_robot(port, robot_id):
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    config = SOFollowerRobotConfig(port=port, id=robot_id, use_degrees=True)
    robot = SOFollower(config)
    robot.connect()
    return robot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", default="so101")
    args = parser.parse_args()

    model, data = load_model(args.urdf)
    mujoco.mj_forward(model, data)

    wrist_roll_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_roll")
    wrist_roll_idx = model.jnt_qposadr[wrist_roll_jid]

    print(f"Connecting to robot on {args.port}...")
    robot = connect_robot(args.port, args.robot_id)
    print("Disabling torque — you can now move the arm by hand.")
    robot.bus.disable_torque()

    state = {"ready": False}

    def key_cb(k):
        if k == 257:  # ENTER
            state["ready"] = True

    print("\n=== Wrist Roll Visual Calibration ===")
    print("For each pose:")
    print("  1. Look at the sim — it shows wrist_roll at a target angle")
    print("  2. Rotate the real robot's wrist to match")
    print("  3. Press ENTER in the viewer when aligned\n")

    points = []

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.15]
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        for i, sim_deg in enumerate(WRIST_ROLL_POSES_DEG):
            print(f"--- Pose {i + 1}/{len(WRIST_ROLL_POSES_DEG)}: wrist_roll = {sim_deg:+.0f} deg ---")

            # Set sim
            data.qpos[wrist_roll_idx] = np.deg2rad(sim_deg)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()

            print(f"  Move real wrist to match sim, then press ENTER in viewer...")
            state["ready"] = False
            while not state["ready"] and viewer.is_running():
                viewer.sync()
                time.sleep(0.05)

            if not viewer.is_running():
                break

            obs = robot.get_observation()
            real_deg = obs["wrist_roll.pos"]
            delta = real_deg - sim_deg
            points.append((sim_deg, real_deg, delta))
            print(f"  Sim: {sim_deg:+8.1f}  Real: {real_deg:+8.1f}  Delta: {delta:+8.1f}\n")

    # Compute result
    if len(points) >= 2:
        sim_pts = np.array([p[0] for p in points])
        real_pts = np.array([p[1] for p in points])

        A = np.vstack([sim_pts, np.ones(len(sim_pts))]).T
        scale, offset = np.linalg.lstsq(A, real_pts, rcond=None)[0]

        print("\n========================================")
        print("  WRIST ROLL CALIBRATION RESULT")
        print("========================================")
        print(f"\n  Points collected:")
        for sim_d, real_d, delta in points:
            print(f"    sim={sim_d:+8.1f}  real={real_d:+8.1f}  delta={delta:+8.1f}")

        if abs(scale - 1.0) < 0.1:
            avg_offset = np.mean(real_pts - sim_pts)
            print(f"\n  Scale is ~1.0, using average offset: {avg_offset:.1f}")
            print(f'\n  Update in 05_lerobot_teleop.py:')
            print(f'    "wrist_roll": {{"scale": 1.0, "offset": {avg_offset:.1f}}},')
        else:
            print(f"\n  Linear fit: real = {scale:.4f} * sim + {offset:.1f}")
            print(f'\n  Update in 05_lerobot_teleop.py:')
            print(f'    "wrist_roll": {{"scale": {scale:.4f}, "offset": {offset:.1f}}},')
    else:
        print("Not enough data points collected.")

    robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
