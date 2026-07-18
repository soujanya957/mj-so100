"""
01-keyboard-teleop-one-robot.py - Keyboard teleop for one SO-100/101 follower arm (hardware only)
==============================================================================
Direct per-joint keyboard control of one real follower arm. No simulation.

Usage:
    python 01-keyboard-teleop-one-robot.py \
        --port1 COMx

Find each port with `lerobot-find-port` (unplug/replug to identify).
See README.md for wiring, motor setup, and calibration.
"""

import argparse

from so_hw import connect_follower, run_keyboard_joint_teleop


def main():
    p = argparse.ArgumentParser(description="Hardware keyboard teleop for one arm")
    p.add_argument("--port1", required=True, help="Serial port for follower 1")
    p.add_argument("--id1", default="follower1", help="Calibration id for follower 1")
    args = p.parse_args()

    ports = [args.port1]
    ids = [args.id1]
    labels = ["follower1"]

    robots = []
    for port, robot_id, label in zip(ports, ids, labels):
        print(f"Connecting {label} on {port} (id={robot_id}) ...")
        robots.append(connect_follower(port, robot_id))
    print("All arm connected.")

    run_keyboard_joint_teleop(robots, labels)


if __name__ == "__main__":
    main()
