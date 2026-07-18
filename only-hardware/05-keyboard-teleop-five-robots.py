"""
05-keyboard-teleop-five-robots.py - Keyboard teleop for five SO-100/101 follower arms (hardware only)
==============================================================================
Direct per-joint keyboard control of five real follower arms. No simulation.

Usage:
    python 05-keyboard-teleop-five-robots.py \
        --port1 COMx --port2 COMx --port3 COMx --port4 COMx --port5 COMx

Find each port with `lerobot-find-port` (unplug/replug to identify).
See README.md for wiring, motor setup, and calibration.
"""

import argparse

from so_hw import connect_follower, run_keyboard_joint_teleop


def main():
    p = argparse.ArgumentParser(description="Hardware keyboard teleop for five arms")
    p.add_argument("--port1", required=True, help="Serial port for follower 1")
    p.add_argument("--port2", required=True, help="Serial port for follower 2")
    p.add_argument("--port3", required=True, help="Serial port for follower 3")
    p.add_argument("--port4", required=True, help="Serial port for follower 4")
    p.add_argument("--port5", required=True, help="Serial port for follower 5")
    p.add_argument("--id1", default="follower1", help="Calibration id for follower 1")
    p.add_argument("--id2", default="follower2", help="Calibration id for follower 2")
    p.add_argument("--id3", default="follower3", help="Calibration id for follower 3")
    p.add_argument("--id4", default="follower4", help="Calibration id for follower 4")
    p.add_argument("--id5", default="follower5", help="Calibration id for follower 5")
    args = p.parse_args()

    ports = [args.port1, args.port2, args.port3, args.port4, args.port5]
    ids = [args.id1, args.id2, args.id3, args.id4, args.id5]
    labels = ["follower1", "follower2", "follower3", "follower4", "follower5"]

    robots = []
    for port, robot_id, label in zip(ports, ids, labels):
        print(f"Connecting {label} on {port} (id={robot_id}) ...")
        robots.append(connect_follower(port, robot_id))
    print("All arms connected.")

    run_keyboard_joint_teleop(robots, labels)


if __name__ == "__main__":
    main()
