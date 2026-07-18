"""
07-ee-position-test.py - End-effector (Cartesian) test for one SO-100/101 arm
==============================================================================
Drive the gripper tip in x/y/z instead of joint-by-joint. Hardware only.

    # 1. FIRST: verify the kinematic model without moving anything
    python 07-ee-position-test.py --port /dev/tty.usbmodemXXXX --id sr101A --probe

    # 2. Then, once the numbers look right:
    python 07-ee-position-test.py --port /dev/tty.usbmodemXXXX --id sr101A

WHY PROBE FIRST
    so_kin.py ships with *nominal* link lengths and *unverified* joint sign
    conventions - there is no URDF in this repo to derive them from. --probe
    reads the arm's real joint angles, prints the predicted tip position, and
    moves nothing. Pose the arm by hand, compare the printed x/y/z against a
    ruler, and fix L0..L3 / CONV in so_kin.py until they agree. Commanding
    Cartesian targets through a wrong model will drive joints into hard stops.

See README.md for finding ports and calibration.
"""

import argparse
import time

import so_kin
from so_hw import CTRL_HZ, JOINT_NAMES, KeyReader, _read_joint_state, connect_follower

STEP_M = 0.005  # metres per keypress
STEP_PITCH = 3.0  # degrees per keypress
GRIPPER_STEP = 3.0


def _probe(robot):
    """Print live joint angles and predicted tip pose. Sends no actions."""
    print("Probe mode - nothing will move. Pose the arm by hand. Ctrl-C to quit.\n")
    try:
        while True:
            j = _read_joint_state(robot)
            x, y, z, pitch = so_kin.forward(j)
            angles = "  ".join(f"{n[:5]}={j[n]:7.1f}" for n in JOINT_NAMES if n in j)
            print(
                f"\r{angles}  ->  x={x * 1000:7.1f} y={y * 1000:7.1f} "
                f"z={z * 1000:7.1f} mm  pitch={pitch:6.1f}deg   ",
                end="",
                flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n")


def _print_controls():
    print(
        "\nEnd-effector control\n"
        "  arrows      x +/- (up/down), y +/- (right/left)\n"
        "  w / s       z up / down\n"
        "  a / d       tool pitch -/+\n"
        "  o / c       gripper open / close\n"
        "  [ / ]       step size\n"
        "  e           toggle elbow up/down solution\n"
        "  h           return to the pose held at startup\n"
        "  x           quit\n"
    )


def _run(robot):
    joints = _read_joint_state(robot)
    x, y, z, pitch = so_kin.forward(joints)
    gripper = joints.get("gripper", 0.0)
    roll = joints.get("wrist_roll", 0.0)
    home = (x, y, z, pitch)
    step = STEP_M
    elbow_up = True

    _print_controls()
    print(f"Start pose: x={x * 1000:.1f} y={y * 1000:.1f} z={z * 1000:.1f} mm\n")

    with KeyReader() as keys:
        try:
            while True:
                t0 = time.time()
                tok = keys.get()
                nx, ny, nz, npitch = x, y, z, pitch

                if tok is not None:
                    if tok == "UP":
                        nx += step
                    elif tok == "DOWN":
                        nx -= step
                    elif tok == "LEFT":
                        ny += step
                    elif tok == "RIGHT":
                        ny -= step
                    elif tok == "w":
                        nz += step
                    elif tok == "s":
                        nz -= step
                    elif tok == "a":
                        npitch -= STEP_PITCH
                    elif tok == "d":
                        npitch += STEP_PITCH
                    elif tok == "o":
                        gripper += GRIPPER_STEP
                    elif tok == "c":
                        gripper -= GRIPPER_STEP
                    elif tok == "]":
                        step = min(0.02, step + 0.001)
                    elif tok == "[":
                        step = max(0.001, step - 0.001)
                    elif tok == "e":
                        elbow_up = not elbow_up
                    elif tok == "h":
                        nx, ny, nz, npitch = home
                    elif tok in ("x", "\x03"):
                        break

                # Only commit the new target if IK actually solves for it.
                status = ""
                try:
                    sol = so_kin.inverse(nx, ny, nz, npitch, elbow_up=elbow_up)
                    x, y, z, pitch = nx, ny, nz, npitch
                    action = {f"{n}.pos": v for n, v in sol.items()}
                    action["wrist_roll.pos"] = roll
                    action["gripper.pos"] = gripper
                    robot.send_action(action)
                except so_kin.Unreachable as e:
                    status = f" BLOCKED: {e}"
                except Exception as e:
                    status = f" send_action error: {e}"

                print(
                    f"\rx={x * 1000:7.1f} y={y * 1000:7.1f} z={z * 1000:7.1f} mm "
                    f"pitch={pitch:6.1f} grip={gripper:6.1f} step={step * 1000:.0f}mm "
                    f"elbow={'up' if elbow_up else 'down'}{status}   ",
                    end="",
                    flush=True,
                )
                time.sleep(max(0.0, 1.0 / CTRL_HZ - (time.time() - t0)))
        finally:
            print()
            robot.disconnect()


def main():
    p = argparse.ArgumentParser(description="Cartesian end-effector test for one arm")
    p.add_argument("--port", required=True, help="Serial port for the follower arm")
    p.add_argument("--id", required=True, help="Calibration id (e.g. sr101A)")
    p.add_argument(
        "--probe",
        action="store_true",
        help="Read-only: print joint angles and predicted tip pose, move nothing",
    )
    args = p.parse_args()

    print(f"Connecting follower on {args.port} (id={args.id}) ...")
    robot = connect_follower(args.port, args.id)
    print("Connected.")

    if args.probe:
        try:
            _probe(robot)
        finally:
            robot.disconnect()
    else:
        _run(robot)


if __name__ == "__main__":
    main()
