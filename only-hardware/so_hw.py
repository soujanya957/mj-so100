"""
so_hw.py - Shared helpers for hardware-only SO-100/101 keyboard teleop
======================================================================
No MuJoCo, no simulation. Talks straight to the real motors via lerobot.

Provides:
  * connect_follower(port, robot_id)  -> connected follower arm
  * connect_leader(port, robot_id)    -> connected leader arm (teleoperator)
  * KeyReader                         -> non-blocking terminal key input
  * run_keyboard_joint_teleop(...)    -> the per-joint keyboard control loop

Imports are defensive: they work whether your installed lerobot uses the
`so_follower` / `so_leader` names or the mainline `so101_follower` /
`so101_leader` names.
"""

import sys
import time

# ── Motor / joint config ──
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# Letter keys that select each joint (index-aligned with JOINT_NAMES)
JOINT_SELECT_KEYS = ["q", "w", "e", "r", "t", "y"]

CTRL_HZ = 30
DEG_STEP = 2.0  # degrees moved per key press


# ── Robot connection (defensive imports) ──
def _load_follower_classes():
    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        return SOFollower, SOFollowerRobotConfig
    except Exception:
        pass
    try:
        from lerobot.robots.so101_follower import (
            SO101Follower,
            SO101FollowerConfig,
        )

        return SO101Follower, SO101FollowerConfig
    except Exception as e:
        raise ImportError(
            "Could not import an SO follower class from lerobot. Tried "
            "`lerobot.robots.so_follower` and `lerobot.robots.so101_follower`. "
            f"Underlying error: {e}"
        )


def _load_leader_classes():
    try:
        from lerobot.teleoperators.so_leader import SOLeader, SOLeaderConfig

        return SOLeader, SOLeaderConfig
    except Exception:
        pass
    try:
        from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

        return SO101Leader, SO101LeaderConfig
    except Exception as e:
        raise ImportError(
            "Could not import an SO leader class from lerobot. Tried "
            "`lerobot.teleoperators.so_leader` and "
            "`lerobot.teleoperators.so101_leader`. "
            f"Underlying error: {e}"
        )


def connect_follower(port, robot_id):
    """Connect to a real follower arm. `robot_id` must match its calibration id."""
    Follower, Config = _load_follower_classes()
    robot = Follower(Config(port=port, id=robot_id, use_degrees=True))
    robot.connect()
    return robot


def connect_leader(port, robot_id):
    """Connect to a real leader arm (teleoperator)."""
    Leader, Config = _load_leader_classes()
    leader = Leader(Config(port=port, id=robot_id))
    leader.connect()
    return leader


# ── Non-blocking keyboard input (macOS / Linux / Windows) ──
class KeyReader:
    """Context manager yielding key tokens without blocking.

    get() returns one of: 'UP', 'DOWN', 'LEFT', 'RIGHT', a single character,
    or None when no key is waiting.
    """

    def __enter__(self):
        self._win = sys.platform.startswith("win")
        if not self._win:
            import termios
            import tty

            self._termios = termios
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if not self._win:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old)

    def get(self):
        if self._win:
            return self._get_win()
        return self._get_unix()

    def _get_unix(self):
        import select

        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # escape sequence (arrow keys)
            r, _, _ = select.select([sys.stdin], [], [], 0.002)
            if not r:
                return "ESC"
            if sys.stdin.read(1) == "[":
                code = sys.stdin.read(1)
                return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(code)
            return "ESC"
        return ch

    def _get_win(self):
        import msvcrt

        if not msvcrt.kbhit():
            return None
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # arrow / special key prefix
            code = msvcrt.getwch()
            return {"H": "UP", "P": "DOWN", "M": "RIGHT", "K": "LEFT"}.get(code)
        return ch


# ── Teleop loop ──
def _read_joint_state(robot):
    """Return {joint_name: degrees} from the robot's current observation."""
    obs = robot.get_observation()
    return {n: float(obs[f"{n}.pos"]) for n in JOINT_NAMES if f"{n}.pos" in obs}


def run_keyboard_joint_teleop(robots, labels):
    """Per-joint keyboard teleop for one or more follower arms.

    robots : list of connected follower objects
    labels : list of human-readable names (same length as robots)
    """
    n_robots = len(robots)
    # Seed each robot's target from where it currently is, so nothing jerks.
    targets = [_read_joint_state(r) for r in robots]

    active_robot = 0
    active_joint = 0
    step = DEG_STEP

    _print_controls(n_robots)

    with KeyReader() as keys:
        try:
            while True:
                t0 = time.time()
                tok = keys.get()

                if tok is not None:
                    # Robot selection: number keys 1..N
                    if tok.isdigit() and 1 <= int(tok) <= n_robots:
                        active_robot = int(tok) - 1
                    # Joint selection: q w e r t y
                    elif tok in JOINT_SELECT_KEYS:
                        active_joint = JOINT_SELECT_KEYS.index(tok)
                    # Nudge selected joint
                    elif tok in ("UP", "k"):
                        _nudge(targets, active_robot, active_joint, +step)
                    elif tok in ("DOWN", "j"):
                        _nudge(targets, active_robot, active_joint, -step)
                    # Step size
                    elif tok == "]":
                        step = min(15.0, step + 1.0)
                    elif tok == "[":
                        step = max(0.5, step - 1.0)
                    # Home
                    elif tok == "h":
                        _home(targets, active_robot)
                    elif tok == "H":
                        for i in range(n_robots):
                            _home(targets, i)
                    # Quit
                    elif tok in ("x", "\x03"):  # x or Ctrl-C
                        break

                # Stream targets to every robot
                for i, robot in enumerate(robots):
                    action = {f"{n}.pos": targets[i][n] for n in targets[i]}
                    try:
                        robot.send_action(action)
                    except Exception as e:
                        print(f"\n[{labels[i]}] send_action error: {e}")

                _print_status(labels, targets, active_robot, active_joint, step)
                time.sleep(max(0.0, 1.0 / CTRL_HZ - (time.time() - t0)))
        finally:
            for robot in robots:
                try:
                    robot.disconnect()
                except Exception:
                    pass
            print("\nDisconnected.")


def _nudge(targets, ri, ji, delta):
    name = JOINT_NAMES[ji]
    if name in targets[ri]:
        targets[ri][name] += delta


def _home(targets, ri):
    for name in targets[ri]:
        targets[ri][name] = 0.0


def _print_controls(n_robots):
    print("\n=== Hardware keyboard teleop ===")
    if n_robots > 1:
        print(f"  1..{n_robots}    select active robot")
    print("  q w e r t y   select joint (pan, lift, elbow, wristflex, wristroll, grip)")
    print("  Up / Down     move selected joint +/-   (also k / j)")
    print("  [ / ]         smaller / bigger step")
    print("  h / H         home active robot / home all")
    print("  x             quit\n")


def _print_status(labels, targets, ri, ji, step):
    name = JOINT_NAMES[ji]
    val = targets[ri].get(name, float("nan"))
    print(
        f"\r[{labels[ri]}] joint={name:<13} {val:7.1f}°  step={step:.1f}°   ",
        end="",
        flush=True,
    )
