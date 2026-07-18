"""
06_calibrate_visual.py - Auto-sweep joint calibration for SO-101
=================================================================
Usage:
    # Interactive joint selection, auto-discovers limits:
    python 06_calibrate_visual.py --urdf <path> --port <port>

    # Pass joint directly:
    python 06_calibrate_visual.py --urdf <path> --port <port> --joint wrist_roll

    # Skip limit-finding, use explicit bounds:
    python 06_calibrate_visual.py --urdf <path> --port <port> --joint elbow_flex --lo -90 --hi 90

Procedure:
  Phase 1 — Limit discovery (unless --lo/--hi are given):
    Creeps slowly in + direction until the motor stalls → records hi limit.
    Then creeps in - direction until stall → records lo limit.
    Backs off a few degrees from each stop.

  Phase 2 — Calibration sweep (4 passes: fwd, bwd, fwd, bwd):
    The real robot follows the sim automatically. Torque stays ON.
    Other joints hold their current real positions.
    Fits: real = scale × sim + offset

  Output: ready-to-paste JOINT_CORRECTIONS entry for 05_lerobot_teleop.py.

Tips:
  - RMSE < 2° is good. RMSE > 5° means the motor hit a stop mid-sweep.
  - If limits are found too conservatively, re-run with --lo / --hi to override.
  - --steps controls sweep density (default 25/pass = 100 points total).
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import mujoco
import mujoco.viewer


ALL_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Sweep
SETTLE_SEC = 0.15  # seconds to wait after each command during calibration sweep

# Limit-finding
CREEP_STEP_DEG = 2.0  # degrees per step while searching for limits
CREEP_SETTLE = 0.25  # seconds to settle per creep step (slower than sweep)
STALL_THRESH = 0.5  # real-position delta below this counts as a stall
STALL_CONFIRM = 4  # consecutive stall readings needed to confirm limit
STALL_BACKOFF = 6.0  # degrees to retreat from the confirmed limit
MAX_CREEP_DEG = 300.0  # absolute safety ceiling — never command beyond ±this


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


def pick_joint():
    """Interactive joint selection before the viewer opens."""
    print("\nAvailable joints:")
    for i, name in enumerate(ALL_JOINTS):
        print(f"  [{i + 1}] {name}")
    print()
    while True:
        raw = input("Choose joint number or name: ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(ALL_JOINTS):
                return ALL_JOINTS[idx]
        elif raw in ALL_JOINTS:
            return raw
        print("  Not recognised — try again.")


def _set_sim(joint_name, sim_deg, model, data, viewer, qpos_idx):
    """Update the sim to show sim_deg for joint_name and sync the viewer."""
    data.qpos[qpos_idx] = np.deg2rad(sim_deg)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    viewer.sync()


def _cmd_robot(joint_name, sim_deg, robot, base_action):
    """Send sim_deg to the real robot for joint_name; other joints unchanged."""
    action = dict(base_action)
    action[f"{joint_name}.pos"] = float(sim_deg)
    robot.send_action(action)


def find_limits(joint_name, model, data, robot, viewer, qpos_idx, base_action):
    """
    Creep the joint toward its + limit until the motor stalls, record hi.
    Then creep toward - limit, record lo.
    Returns (lo_deg, hi_deg) backed off from the physical stops.
    """
    print(f"\n  --- Limit discovery for {joint_name} ---")
    print(f"  Creeping +{CREEP_STEP_DEG}°/step until stall, then reversing...")

    def creep(direction, start_deg):
        """Creep in direction (+1/-1) from start_deg. Returns limit sim_deg."""
        current = start_deg
        prev_real = None
        stall_run = 0

        while viewer.is_running():
            current += direction * CREEP_STEP_DEG
            if abs(current) > MAX_CREEP_DEG:
                print(f"  Safety ceiling ±{MAX_CREEP_DEG}° reached.")
                break

            _set_sim(joint_name, current, model, data, viewer, qpos_idx)
            _cmd_robot(joint_name, current, robot, base_action)
            time.sleep(CREEP_SETTLE)

            obs = robot.get_observation()
            real_deg = float(obs[f"{joint_name}.pos"])

            moving = abs(real_deg - prev_real) if prev_real is not None else 999.0
            stall_run = (stall_run + 1) if moving < STALL_THRESH else 0
            prev_real = real_deg

            tag = "STALL" if stall_run >= STALL_CONFIRM else "    "
            print(
                f"  {tag}  sim={current:+8.2f}°  real={real_deg:+8.2f}°"
                f"  Δreal={moving:+6.2f}°  stall_run={stall_run}"
            )

            if stall_run >= STALL_CONFIRM:
                # Back off from the physical stop
                limit = (
                    current
                    - direction * CREEP_STEP_DEG * STALL_CONFIRM
                    - direction * STALL_BACKOFF
                )
                print(
                    f"  → Limit confirmed.  Using {limit:+.1f}° (backed off {STALL_BACKOFF}°)"
                )
                return limit

        # Viewer closed or safety ceiling — use last position
        return current - direction * STALL_BACKOFF

    # Start both searches from 0
    hi_deg = creep(+1, 0.0)
    lo_deg = creep(-1, 0.0)
    print(f"\n  Discovered range: {lo_deg:.1f}° to {hi_deg:.1f}°")
    return lo_deg, hi_deg


def auto_sweep(
    joint_name, model, data, robot, viewer, lo_deg, hi_deg, n_steps, base_action
):
    """
    4-pass calibration sweep (fwd, bwd, fwd, bwd).
    Commands sim_deg directly (no correction) so the fit IS the correction.
    Returns (scale, offset, points) or None if viewer closed early.
    """
    forward = np.linspace(lo_deg, hi_deg, n_steps)
    backward = np.linspace(hi_deg, lo_deg, n_steps)
    passes = [
        ("fwd-1", forward),
        ("bwd-1", backward),
        ("fwd-2", forward),
        ("bwd-2", backward),
    ]
    total = len(passes) * n_steps
    done = 0
    points = []

    print(f"\n  --- Calibration sweep ---")
    print(
        f"  {lo_deg:.1f}° → {hi_deg:.1f}°  |  {n_steps} steps/pass  |  4 passes  |  {total} points total"
    )
    print(f"\n  {'Pass':<7} {'#':>6}  {'Sim(°)':>9}  {'Real(°)':>9}  {'Delta':>8}")
    print(f"  {'-' * 50}")

    for pass_name, sweep_pts in passes:
        for sim_deg in sweep_pts:
            if not viewer.is_running():
                return None

            _set_sim(
                joint_name, sim_deg, model, data, viewer, qpos_idx=None
            )  # qpos_idx resolved below
            _cmd_robot(joint_name, sim_deg, robot, base_action)
            time.sleep(SETTLE_SEC)

            obs = robot.get_observation()
            real_deg = float(obs[f"{joint_name}.pos"])
            points.append((sim_deg, real_deg))
            done += 1

            print(
                f"  {pass_name:<7} {done:>4}/{total}"
                f"  {sim_deg:>+9.2f}  {real_deg:>+9.2f}  {real_deg - sim_deg:>+8.2f}"
            )

    sim_pts = np.array([p[0] for p in points])
    real_pts = np.array([p[1] for p in points])
    A = np.vstack([sim_pts, np.ones(len(sim_pts))]).T
    scale, offset = np.linalg.lstsq(A, real_pts, rcond=None)[0]
    return float(scale), float(offset), points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", default="so101")
    parser.add_argument(
        "--joint",
        default=None,
        choices=ALL_JOINTS,
        help="Joint to calibrate. Omit for interactive menu.",
    )
    parser.add_argument(
        "--steps", type=int, default=25, help="Steps per sweep pass (default 25)."
    )
    parser.add_argument(
        "--lo",
        type=float,
        default=None,
        help="Override sweep low bound (°). Skips limit-finding.",
    )
    parser.add_argument(
        "--hi",
        type=float,
        default=None,
        help="Override sweep high bound (°). Skips limit-finding.",
    )
    args = parser.parse_args()

    joint_name = args.joint if args.joint else pick_joint()
    print(f"\nCalibrating: {joint_name}")

    model, data = load_model(args.urdf)
    mujoco.mj_forward(model, data)

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid == -1:
        print(f"ERROR: joint '{joint_name}' not found in model.")
        sys.exit(1)
    qpos_idx = model.jnt_qposadr[jid]

    # Monkey-patch qpos_idx into _set_sim's closure properly
    def set_sim(sim_deg):
        data.qpos[qpos_idx] = np.deg2rad(sim_deg)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

    print(f"\nConnecting to {args.port}...")
    robot = connect_robot(args.port, args.robot_id)
    print("Connected. Torque ON — the real robot will follow the sim.\n")
    print("Close the viewer window or Ctrl-C to abort.\n")

    # Snapshot other joints to keep them stable throughout
    obs = robot.get_observation()
    base_action = {f"{jn}.pos": float(obs[f"{jn}.pos"]) for jn in ALL_JOINTS}

    result = None
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0.15]
            viewer.cam.distance = 0.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

            # ── Phase 1: find limits ──────────────────────────────────────
            if args.lo is not None and args.hi is not None:
                lo_deg, hi_deg = args.lo, args.hi
                print(f"  Using explicit bounds: {lo_deg:.1f}° to {hi_deg:.1f}°")
            else:
                # Override only the axis the user left as None
                auto_lo, auto_hi = find_limits(
                    joint_name, model, data, robot, viewer, qpos_idx, base_action
                )
                lo_deg = args.lo if args.lo is not None else auto_lo
                hi_deg = args.hi if args.hi is not None else auto_hi

            if not viewer.is_running():
                print("Viewer closed during limit discovery.")
                robot.disconnect()
                return

            # ── Phase 2: calibration sweep ────────────────────────────────
            # Redefine inner helpers with the real qpos_idx in scope
            def _set(sim_deg):
                data.qpos[qpos_idx] = np.deg2rad(sim_deg)
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                viewer.sync()

            def _cmd(sim_deg):
                action = dict(base_action)
                action[f"{joint_name}.pos"] = float(sim_deg)
                robot.send_action(action)

            forward = np.linspace(lo_deg, hi_deg, args.steps)
            backward = np.linspace(hi_deg, lo_deg, args.steps)
            passes = [
                ("fwd-1", forward),
                ("bwd-1", backward),
                ("fwd-2", forward),
                ("bwd-2", backward),
            ]
            total = len(passes) * args.steps
            done = 0
            points = []

            print(f"\n  --- Calibration sweep ---")
            print(
                f"  {lo_deg:.1f}° → {hi_deg:.1f}°  |  {args.steps} steps/pass  |  4 passes  |  {total} pts"
            )
            print(
                f"\n  {'Pass':<7} {'#':>6}  {'Sim(°)':>9}  {'Real(°)':>9}  {'Delta':>8}"
            )
            print(f"  {'-' * 50}")

            for pass_name, sweep_pts in passes:
                for sim_deg in sweep_pts:
                    if not viewer.is_running():
                        break
                    _set(sim_deg)
                    _cmd(sim_deg)
                    time.sleep(SETTLE_SEC)

                    obs = robot.get_observation()
                    real_deg = float(obs[f"{joint_name}.pos"])
                    points.append((sim_deg, real_deg))
                    done += 1
                    print(
                        f"  {pass_name:<7} {done:>4}/{total}"
                        f"  {sim_deg:>+9.2f}  {real_deg:>+9.2f}"
                        f"  {real_deg - sim_deg:>+8.2f}"
                    )
                if not viewer.is_running():
                    break

            if len(points) >= 4:
                sim_pts = np.array([p[0] for p in points])
                real_pts = np.array([p[1] for p in points])
                A = np.vstack([sim_pts, np.ones(len(sim_pts))]).T
                scale, offset = np.linalg.lstsq(A, real_pts, rcond=None)[0]
                result = (float(scale), float(offset), points)

    except KeyboardInterrupt:
        print("\nAborted.")

    robot.disconnect()

    if result is None:
        print("Not enough data collected.")
        return

    scale, offset, points = result
    sim_pts = np.array([p[0] for p in points])
    real_pts = np.array([p[1] for p in points])
    residuals = real_pts - (scale * sim_pts + offset)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    print(f"\n{'=' * 56}")
    print(f"  RESULT: {joint_name}")
    print(f"{'=' * 56}")
    print(f"  real = {scale:.4f} × sim + {offset:.2f}")
    print(f"  RMSE: {rmse:.2f}°   (< 2° good,  > 5° motor hit a stop)")
    print(f"  Points used: {len(points)}")

    if abs(scale) < 0.5 or abs(scale) > 2.0:
        print(f"\n  WARNING: scale {scale:.4f} is far from 1.0.")
        print(f"  The motor likely hit a mechanical stop. Re-run with tighter bounds:")
        margin = (hi_deg - lo_deg) * 0.1
        print(f"    --lo {lo_deg + margin:.0f}  --hi {hi_deg - margin:.0f}")

    print(f"\n  Paste into JOINT_CORRECTIONS in 05_lerobot_teleop.py:")
    print(f'    "{joint_name}": {{"scale": {scale:.4f}, "offset": {offset:.2f}}},')
    print()


if __name__ == "__main__":
    main()
