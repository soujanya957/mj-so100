"""
02-keyboard-teleop-two-robots.py - Keyboard teleop for two SO-100/101 follower arms (hardware only)
==============================================================================
Direct per-joint keyboard control of two real follower arms. No simulation.

Usage:
    python 02-keyboard-teleop-two-robots.py \
        --port1 COMx --port2 COMx

Find each port with `lerobot-find-port` (unplug/replug to identify).
See README.md for wiring, motor setup, and calibration.
"""

import argparse

from so_hw import connect_follower, run_keyboard_joint_teleop


def main():
    p = argparse.ArgumentParser(description="Hardware keyboard teleop for two arms")
    p.add_argument("--port1", required=True, help="Serial port for follower 1")
    p.add_argument("--port2", required=True, help="Serial port for follower 2")
    p.add_argument("--id1", default="follower1", help="Calibration id for follower 1")
    p.add_argument("--id2", default="follower2", help="Calibration id for follower 2")
    args = p.parse_args()

    ports = [args.port1, args.port2]
    ids = [args.id1, args.id2]
    labels = ["follower1", "follower2"]

    robots = []
    for port, robot_id, label in zip(ports, ids, labels):
        print(f"Connecting {label} on {port} (id={robot_id}) ...")
        robots.append(connect_follower(port, robot_id))
    print("All arms connected.")

    run_keyboard_joint_teleop(robots, labels)


if __name__ == "__main__":
    main()
