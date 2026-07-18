"""
05_lerobot_teleop.py - MuJoCo IK keyboard teleop → real SO-101 via lerobot
==========================================================================
Usage:
    python 05_lerobot_teleop.py --urdf ../../assets/SO101_urdf/so101_new_calib.urdf --port COM3

Controls (focus the MuJoCo viewer window):
    Arrow Left / Right   move EE left / right   (Y axis)
    Arrow Up   / Down    move EE forward / back  (X axis)
    W / S                move EE up / down       (Z axis)
    A / D                roll wrist left / right
    O / C                open / close gripper
    R                    reset to home position
    Q                    quit

The orange sphere shows the current IK target.
Joint angles solved by MuJoCo are sent to the real robot each frame.
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


# ── IK config ──
IK_DAMPING = 1e-3
IK_ITERS = 80
IK_ALPHA = 0.4
CTRL_HZ = 30
EE_BODY = "gripper_link"
STEP = 0.008  # metres per key event
ROLL_STEP = 0.1  # radians per key event for wrist roll

# Joint name order matching MuJoCo qpos indices and lerobot motor names
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# MuJoCo GLFW key codes
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265


def patch_urdf(xml):
    """Add <collision> to visual-only links so MuJoCo creates geoms."""

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

        scene_xml = f"""<mujoco model="so101_lerobot_teleop">
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
    <!-- IK target marker -->
    <body name="ee_target" pos="0 0 0.25" mocap="true">
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


def ik_step(model, data, target, ee_id, wrist_roll_dof, gripper_dof):
    """Damped least-squares IK, position only. Excludes wrist_roll and gripper."""
    # Use DOF-space indices (jnt_dofadr), not qpos-space indices (jnt_qposadr)
    ik_indices = [
        i for i in range(model.nv) if i != wrist_roll_dof and i != gripper_dof
    ]

    for _ in range(IK_ITERS):
        mujoco.mj_fwdPosition(model, data)
        err = target - data.xpos[ee_id]
        if np.linalg.norm(err) < 5e-4:
            break
        J = np.zeros((6, model.nv))
        mujoco.mj_jacBody(model, data, J[:3], J[3:], ee_id)
        Jp = J[:3][:, ik_indices]
        dq = IK_ALPHA * Jp.T @ np.linalg.solve(Jp @ Jp.T + IK_DAMPING * np.eye(3), err)
        for k, dof_idx in enumerate(ik_indices):
            jid = model.dof_jntid[dof_idx]
            qpos_idx = model.jnt_qposadr[jid]
            data.qpos[qpos_idx] += np.clip(dq[k], -0.1, 0.1)

        # Clamp limited joints
        for dof_idx in ik_indices:
            jid = model.dof_jntid[dof_idx]
            if model.jnt_limited[jid]:
                qpos_idx = model.jnt_qposadr[jid]
                data.qpos[qpos_idx] = np.clip(
                    data.qpos[qpos_idx],
                    model.jnt_range[jid, 0],
                    model.jnt_range[jid, 1],
                )


def connect_robot(port, robot_id):
    """Connect to real SO-101 via lerobot."""
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    config = SOFollowerRobotConfig(
        port=port,
        id=robot_id,
        use_degrees=True,
    )
    robot = SOFollower(config)
    robot.connect()
    return robot


# Corrections for joints where MuJoCo and lerobot conventions differ
# real_deg = mujoco_deg * scale + offset
JOINT_CORRECTIONS = {
    "shoulder_pan": {"scale": 0.9447, "offset": 1.47},
    "shoulder_lift": {"scale": 0.2949, "offset": 21.93},
    "elbow_flex": {"scale": 0.3529, "offset": 18.59},
    "wrist_flex": {"scale": 1.0067, "offset": -3.43},
    "wrist_roll": {"scale": 0.9288, "offset": -7.81},
}


def mujoco_qpos_to_lerobot_action(model, data, gripper_deg):
    """Convert MuJoCo joint angles to lerobot action dict (degrees).

    Uses name-based joint lookup so joint order in URDF doesn't matter.
    NOTE: JOINT_CORRECTIONS must be recalibrated via 06_calibrate_visual.py
    if the robot does not track correctly.
    """
    action = {}
    for name in JOINT_NAMES[:5]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        val_rad = data.qpos[model.jnt_qposadr[jid]]
        val_deg = np.rad2deg(val_rad)
        cor = JOINT_CORRECTIONS[name]
        action[f"{name}.pos"] = float(val_deg * cor["scale"] + cor["offset"])
    action["gripper.pos"] = float(gripper_deg)
    return action


def diagnose_joint(robot, model, joint_name, steps=20):
    """
    Sweep a joint through its MuJoCo range, send to real robot, read back.
    Prints a table of: sim_deg -> sent_deg -> real_readback_deg
    When real stops changing but sim keeps going, that's the offset/limit mismatch.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid == -1:
        print(f"Joint '{joint_name}' not found in model.")
        return

    sim_lo, sim_hi = model.jnt_range[jid]
    sim_lo_deg = np.rad2deg(sim_lo)
    sim_hi_deg = np.rad2deg(sim_hi)
    cor = JOINT_CORRECTIONS[joint_name]

    print(f"\n=== Diagnosing '{joint_name}' ===")
    print(f"MuJoCo URDF range: {sim_lo_deg:.1f} to {sim_hi_deg:.1f} deg")
    print(f"Current correction: scale={cor['scale']}, offset={cor['offset']}")
    print(f"Sweeping {steps} steps from min to max...\n")
    print(
        f"{'Step':>4}  {'Sim(deg)':>10}  {'Sent(deg)':>10}  {'Real(deg)':>10}  {'Delta':>10}"
    )
    print("-" * 56)

    sweep = np.linspace(sim_lo_deg, sim_hi_deg, steps)
    results = []

    for i, sim_deg in enumerate(sweep):
        sent_deg = sim_deg * cor["scale"] + cor["offset"]

        # Send only this joint, keep others at current
        obs = robot.get_observation()
        action = {f"{n}.pos": obs[f"{n}.pos"] for n in JOINT_NAMES}
        action[f"{joint_name}.pos"] = float(sent_deg)
        robot.send_action(action)

        time.sleep(0.15)  # let motor settle

        # Read back
        obs = robot.get_observation()
        real_deg = obs[f"{joint_name}.pos"]
        delta = real_deg - sent_deg

        results.append((sim_deg, sent_deg, real_deg, delta))
        print(
            f"{i:4d}  {sim_deg:10.2f}  {sent_deg:10.2f}  {real_deg:10.2f}  {delta:+10.2f}"
        )

    # Analyze: find where real motor stopped tracking
    real_vals = [r[2] for r in results]
    real_range = max(real_vals) - min(real_vals)
    sim_range = sim_hi_deg - sim_lo_deg

    print(f"\n--- Analysis ---")
    print(f"Sim range:  {sim_range:.1f} deg")
    print(f"Real range: {real_range:.1f} deg")
    print(
        f"Real min:   {min(real_vals):.1f} deg  (at sim {results[real_vals.index(min(real_vals))][0]:.1f})"
    )
    print(
        f"Real max:   {max(real_vals):.1f} deg  (at sim {results[real_vals.index(max(real_vals))][0]:.1f})"
    )

    # Check for saturation (real stops moving)
    stall_lo = None
    stall_hi = None
    for i in range(1, len(results)):
        if abs(real_vals[i] - real_vals[i - 1]) < 0.5 and stall_lo is None:
            stall_lo = results[i][0]
        if abs(real_vals[-(i + 1)] - real_vals[-i]) < 0.5 and stall_hi is None:
            stall_hi = results[-(i + 1)][0]

    if stall_lo is not None:
        print(f"Real motor stalls at LOW end around sim={stall_lo:.1f} deg")
    if stall_hi is not None:
        print(f"Real motor stalls at HIGH end around sim={stall_hi:.1f} deg")

    # Suggest correction
    # Find the linear fit between sim and real in the region where real is moving
    moving = [
        (s, r)
        for s, _, r, _ in results
        if r > min(real_vals) + 1.0 and r < max(real_vals) - 1.0
    ]
    if len(moving) >= 2:
        sim_pts = np.array([m[0] for m in moving])
        real_pts = np.array([m[1] for m in moving])
        # Linear fit: real = scale * sim + offset
        A = np.vstack([sim_pts, np.ones(len(sim_pts))]).T
        scale_fit, offset_fit = np.linalg.lstsq(A, real_pts, rcond=None)[0]
        print(f"\nSuggested correction:")
        print(
            f'    "{joint_name}": {{"scale": {scale_fit:.4f}, "offset": {offset_fit:.2f}}},'
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True, help="Path to SO-101 URDF")
    parser.add_argument(
        "--port",
        default=None,
        help="COM port for real robot (e.g. COM3). Omit for sim-only.",
    )
    parser.add_argument(
        "--robot-id", default="so101", help="Robot ID for lerobot calibration"
    )
    parser.add_argument(
        "--diagnose",
        default=None,
        help="Diagnose a joint mapping (e.g. --diagnose wrist_roll)",
    )
    args = parser.parse_args()

    model, data = load_model(args.urdf)
    mujoco.mj_forward(model, data)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    if ee_id == -1:
        all_bodies = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(model.nbody)
        ]
        print(f"Body '{EE_BODY}' not found. Available bodies:")
        for i, name in enumerate(all_bodies):
            print(f"  [{i}] {name}")
        raise ValueError(
            f"Body '{EE_BODY}' not found. Pick one from the list above and set EE_BODY."
        )

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_target")
    mocap_id = model.body_mocapid[target_body_id]

    # Look up wrist_roll and gripper DOF indices once (used by IK and main loop)
    wrist_roll_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_roll")
    wrist_roll_qpos = model.jnt_qposadr[wrist_roll_jid]
    wrist_roll_dof = model.jnt_dofadr[wrist_roll_jid]
    grip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    grip_qpos = model.jnt_qposadr[grip_jid]
    gripper_dof = model.jnt_dofadr[grip_jid]

    # Start from current EE position
    home = data.xpos[ee_id].copy()
    target = home.copy()
    data.mocap_pos[mocap_id] = target.copy()

    gripper_deg = 50.0  # mid-range (0=closed, 100=open)
    GRIPPER_STEP = 5.0

    # Connect to real robot if port specified
    robot = None
    if args.port:
        print(f"Connecting to real robot on {args.port}...")
        robot = connect_robot(args.port, args.robot_id)
        print("Robot connected!")

        if args.diagnose:
            diagnose_joint(robot, model, args.diagnose)
            robot.disconnect()
            return
        print("Starting teleop...")
    else:
        print("Sim-only mode (no --port). Starting teleop...")

    nudge = {"dx": 0.0, "dy": 0.0, "dz": 0.0, "roll": 0.0}
    state = {"quit": False, "reset": False, "grip_open": False, "grip_close": False}

    def key_cb(k):
        if k == ord("Q"):
            state["quit"] = True
        elif k == ord("R"):
            state["reset"] = True
        elif k == KEY_UP:
            nudge["dx"] += STEP
        elif k == KEY_DOWN:
            nudge["dx"] -= STEP
        elif k == KEY_RIGHT:
            nudge["dy"] -= STEP
        elif k == KEY_LEFT:
            nudge["dy"] += STEP
        elif k == ord("W"):
            nudge["dz"] += STEP
        elif k == ord("S"):
            nudge["dz"] -= STEP
        elif k == ord("O"):
            state["grip_open"] = True
        elif k == ord("C"):
            state["grip_close"] = True
        elif k == ord("A"):
            nudge["roll"] += ROLL_STEP
        elif k == ord("D"):
            nudge["roll"] -= ROLL_STEP

    print("\nKeys: arrow=X/Y  W/S=Z  A/D=wrist roll  O/C=gripper  R=reset  Q=quit\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.15]
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        while viewer.is_running() and not state["quit"]:
            t0 = time.time()

            if state["reset"]:
                target = home.copy()
                data.qpos[:] = 0.0
                gripper_deg = 50.0
                state["reset"] = False
                nudge["dx"] = nudge["dy"] = nudge["dz"] = 0.0

            # Gripper
            if state["grip_open"]:
                gripper_deg = min(100.0, gripper_deg + GRIPPER_STEP)
                state["grip_open"] = False
            if state["grip_close"]:
                gripper_deg = max(0.0, gripper_deg - GRIPPER_STEP)
                state["grip_close"] = False

            # Consume accumulated nudges
            target[0] += nudge["dx"]
            target[1] += nudge["dy"]
            target[2] += nudge["dz"]
            nudge["dx"] = nudge["dy"] = nudge["dz"] = 0.0

            # Workspace clamp
            target[0] = np.clip(target[0], -0.30, 0.30)
            target[1] = np.clip(target[1], -0.30, 0.30)
            target[2] = np.clip(target[2], 0.02, 0.42)

            # Apply wrist roll directly via name-looked-up qpos index
            data.qpos[wrist_roll_qpos] += nudge["roll"]
            nudge["roll"] = 0.0
            # Clamp wrist roll to real motor's usable range (~-57 to +163 deg sim)
            data.qpos[wrist_roll_qpos] = np.clip(
                data.qpos[wrist_roll_qpos],
                np.deg2rad(-57.0),
                np.deg2rad(163.0),
            )

            data.mocap_pos[mocap_id] = target.copy()
            ik_step(model, data, target, ee_id, wrist_roll_dof, gripper_dof)

            # Update gripper joint in sim — map gripper_deg 0-100% to joint range
            grip_lo, grip_hi = model.jnt_range[grip_jid]
            data.qpos[grip_qpos] = grip_lo + (gripper_deg / 100.0) * (grip_hi - grip_lo)

            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()

            # Send to real robot
            if robot is not None:
                action = mujoco_qpos_to_lerobot_action(model, data, gripper_deg)
                try:
                    robot.send_action(action)
                except Exception as e:
                    print(f"\rRobot error: {e}", end="")

            ee = data.xpos[ee_id]
            print(
                f"\rEE [{ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f}]  "
                f"grip {gripper_deg:5.1f}%  "
                f"target [{target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}]",
                end="",
            )

            time.sleep(max(0, 1 / CTRL_HZ - (time.time() - t0)))

    if robot is not None:
        robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
