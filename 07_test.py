import argparse
import os
import re
import time
import numpy as np
import mujoco
import mujoco.viewer
import tempfile
import shutil

# ── CONFIG ──
CTRL_HZ = 30
DEG_STEP = 2.0
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
KEY_UP, KEY_DOWN = 265, 264
JOINT_KEYS = {ord(str(i + 1)): i for i in range(6)}

# ── CALIBRATION OFFSETS ──
# real_deg = sim_deg + offset
# Since sim is +90 deg (anti-clockwise) relative to real,
# we subtract 90 to bring the real motor to the sim position.
JOINT_OFFSETS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": -90.0,  # Adjust this if it needs to be +90
    "gripper": 0.0,
}


def load_simple_model(urdf_path):
    abs_urdf = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(abs_urdf)
    assets_dir = os.path.join(urdf_dir, "assets")
    tmp_dir = tempfile.mkdtemp()

    if os.path.exists(assets_dir):
        for item in os.listdir(assets_dir):
            if item.lower().endswith(".stl"):
                shutil.copy(os.path.join(assets_dir, item), tmp_dir)

    with open(abs_urdf, "r") as f:
        xml = f.read()

    def strip_mesh_path(match):
        return f'filename="{os.path.basename(match.group(1))}"'

    xml = re.sub(r'filename="([^"]+\.[sS][tT][lL])"', strip_mesh_path, xml)
    tmp_urdf_path = os.path.join(tmp_dir, "robot.urdf")
    with open(tmp_urdf_path, "w") as f:
        f.write(xml)

    original_cwd = os.getcwd()
    os.chdir(tmp_dir)
    try:
        model = mujoco.MjModel.from_xml_path("robot.urdf")
        data = mujoco.MjData(model)
    finally:
        os.chdir(original_cwd)
    return model, data, tmp_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--port", required=True)
    args = parser.parse_args()

    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    print(f"Connecting to robot on {args.port}...")
    config = SOFollowerRobotConfig(port=args.port, id="so101", use_degrees=True)
    robot = SOFollower(config)
    robot.connect()

    print("Syncing simulation to real robot position...")
    obs = robot.get_observation()
    model, data, tmp_dir = load_simple_model(args.urdf)

    # Map real robot -> Sim (Initial Sync)
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid != -1:
            key = f"{name}.pos"
            if key in obs:
                # To sync sim to real: sim = real - offset
                real_deg = obs[key]
                sim_deg = real_deg - JOINT_OFFSETS.get(name, 0.0)
                data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(sim_deg)

    mujoco.mj_forward(model, data)

    selected_joint_idx = 0
    nudge = 0.0

    def key_cb(k):
        nonlocal selected_joint_idx, nudge
        if k in JOINT_KEYS:
            selected_joint_idx = JOINT_KEYS[k]
        elif k == KEY_UP:
            nudge = DEG_STEP
        elif k == KEY_DOWN:
            nudge = -DEG_STEP

    print(f"\n--- OFFSET CALIBRATION MODE ---")
    print(f"Current Wrist Offset: {JOINT_OFFSETS['wrist_roll']}°")
    print(f"1-6: Select Joint | Arrows: Move | Q: Quit")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        while viewer.is_running():
            t_loop = time.time()

            if nudge != 0:
                name = JOINT_NAMES[selected_joint_idx]
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                q_addr = model.jnt_qposadr[jid]

                # Move Sim
                data.qpos[q_addr] += np.deg2rad(nudge)

                # Send to hardware
                action = {}
                for j_name in JOINT_NAMES:
                    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
                    sim_deg = np.rad2deg(data.qpos[model.jnt_qposadr[j_id]])
                    # real_deg = sim_deg + offset
                    action[f"{j_name}.pos"] = float(
                        sim_deg + JOINT_OFFSETS.get(j_name, 0.0)
                    )

                robot.send_action(action)
                nudge = 0.0

            mujoco.mj_forward(model, data)
            viewer.sync()

            sim_val = np.rad2deg(
                data.qpos[
                    model.jnt_qposadr[
                        mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            JOINT_NAMES[selected_joint_idx],
                        )
                    ]
                ]
            )
            print(
                f"\rActive: [{JOINT_NAMES[selected_joint_idx]}] Sim Angle: {sim_val:6.1f}° | Real: {sim_val + JOINT_OFFSETS.get(JOINT_NAMES[selected_joint_idx], 0):6.1f}°   ",
                end="",
            )

            time.sleep(max(0, 1 / CTRL_HZ - (time.time() - t_loop)))

    robot.disconnect()
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
