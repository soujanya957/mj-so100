"""
06-leader-follower-teleop.py - Leader arm drives one follower arm (hardware only)
=================================================================================
Physically move the leader arm; the follower mirrors it in real time.
No keyboard, no simulation — classic lerobot leader/follower teleop.

Usage:
    python 06-leader-follower-teleop.py \
        --leader-port COMx --follower-port COMy

Both arms must be calibrated first (`lerobot-calibrate`), using the ids you
pass below (default: "leader" and "follower"). Ctrl-C to stop.

See README.md for finding ports, motor setup, and calibration.
"""

import argparse
import time

from so_hw import CTRL_HZ, connect_follower, connect_leader


def main():
    p = argparse.ArgumentParser(description="Leader arm -> follower arm teleop")
    p.add_argument("--leader-port", required=True, help="Serial port for leader arm")
    p.add_argument("--follower-port", required=True, help="Serial port for follower arm")
    p.add_argument("--leader-id", default="leader", help="Leader calibration id")
    p.add_argument("--follower-id", default="follower", help="Follower calibration id")
    args = p.parse_args()

    print(f"Connecting leader on {args.leader_port} (id={args.leader_id}) ...")
    leader = connect_leader(args.leader_port, args.leader_id)
    print(f"Connecting follower on {args.follower_port} (id={args.follower_id}) ...")
    follower = connect_follower(args.follower_port, args.follower_id)
    print("Both arms connected. Move the leader — follower will follow. Ctrl-C to stop.\n")

    try:
        while True:
            t0 = time.time()
            # Leader reports target joint positions; follower executes them.
            action = leader.get_action()
            follower.send_action(action)

            first = next(iter(action), None)
            if first is not None:
                print(f"\r{first} = {action[first]:7.1f}   ", end="", flush=True)

            time.sleep(max(0.0, 1.0 / CTRL_HZ - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        for arm in (leader, follower):
            try:
                arm.disconnect()
            except Exception:
                pass
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
