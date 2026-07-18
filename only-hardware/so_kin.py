"""
so_kin.py - Analytic kinematics for the SO-100/101 arm (hardware only)
======================================================================
No MuJoCo, no URDF. Closed-form FK/IK for the 5-DOF arm treated as a
yaw base carrying a 3-link planar chain.

    pan          rotates the whole arm about +z
    lift/elbow/  form a planar chain in the vertical plane at angle `pan`
      wrist_flex
    wrist_roll   spins the tool about its own axis (no effect on position)

!! LINK LENGTHS AND JOINT SIGNS ARE UNVERIFIED DEFAULTS !!
They are nominal SO-ARM100 values, not measured from your arm. Run
`07-ee-position-test.py --probe` and compare the printed FK against a
ruler before trusting any of this. See README notes in that script.
"""

import math

# ── Link lengths, metres. Measure these on your arm and correct them. ──
L0 = 0.0542  # base plate  -> shoulder_lift axis (vertical offset)
L1 = 0.1159  # shoulder_lift -> elbow_flex
L2 = 0.1350  # elbow_flex  -> wrist_flex
L3 = 0.0721  # wrist_flex  -> gripper tip

# ── Joint convention mapping ──
# lerobot reports calibrated degrees; this model wants angles measured with
# "arm straight up" = 0 and positive = folding forward. Per joint:
#     theta_model_deg = SIGN * (q_lerobot_deg - OFFSET)
# Flip a SIGN or shift an OFFSET here if --probe shows an axis moving the
# wrong way or reading a constant error.
CONV = {
    "shoulder_lift": {"sign": 1.0, "offset": 0.0},
    "elbow_flex": {"sign": 1.0, "offset": 0.0},
    "wrist_flex": {"sign": 1.0, "offset": 0.0},
    "shoulder_pan": {"sign": 1.0, "offset": 0.0},
}


def _to_model(name, q_deg):
    c = CONV[name]
    return math.radians(c["sign"] * (q_deg - c["offset"]))


def _to_lerobot(name, theta_rad):
    c = CONV[name]
    return math.degrees(theta_rad) / c["sign"] + c["offset"]


def forward(joints):
    """Joint dict {name: degrees} -> (x, y, z) in metres, plus tool pitch.

    Returns (x, y, z, pitch_deg) where pitch is the tool axis measured from
    vertical: 0 = pointing straight up, 90 = horizontal, 180 = straight down.
    """
    pan = _to_model("shoulder_pan", joints["shoulder_pan"])
    t1 = _to_model("shoulder_lift", joints["shoulder_lift"])
    t2 = _to_model("elbow_flex", joints["elbow_flex"])
    t3 = _to_model("wrist_flex", joints["wrist_flex"])

    a1, a2, a3 = t1, t1 + t2, t1 + t2 + t3
    r = L1 * math.sin(a1) + L2 * math.sin(a2) + L3 * math.sin(a3)
    z = L0 + L1 * math.cos(a1) + L2 * math.cos(a2) + L3 * math.cos(a3)

    # A folded-back arm gives r < 0, but inverse() recovers radius as hypot(x, y)
    # and can only ever see r >= 0. Normalise to that convention here: mirroring
    # the working plane (pan += 180) while negating the planar angles is an exact
    # identity - same tip, same tool axis - so nothing is lost.
    if r < 0.0:
        r = -r
        pan += math.pi
        a3 = -a3

    return r * math.cos(pan), r * math.sin(pan), z, math.degrees(a3)


class Unreachable(Exception):
    """Target lies outside the workspace for the requested tool pitch."""


def inverse(x, y, z, pitch_deg, elbow_up=True):
    """(x, y, z, pitch) -> joint dict {name: degrees}.

    Raises Unreachable if no solution exists. wrist_roll and gripper are not
    part of the position solution and are omitted from the result.
    """
    pan = math.atan2(y, x)
    r = math.hypot(x, y)
    pitch = math.radians(pitch_deg)

    # Back off along the tool axis to find the wrist_flex centre.
    rw = r - L3 * math.sin(pitch)
    zw = z - L0 - L3 * math.cos(pitch)

    reach = math.hypot(rw, zw)
    if reach > L1 + L2 or reach < abs(L1 - L2):
        raise Unreachable(
            f"wrist centre {reach * 1000:.0f}mm from shoulder; "
            f"arm spans {abs(L1 - L2) * 1000:.0f}-{(L1 + L2) * 1000:.0f}mm"
        )

    cos_t2 = (rw * rw + zw * zw - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    t2 = math.acos(max(-1.0, min(1.0, cos_t2)))
    if elbow_up:
        t2 = -t2

    t1 = math.atan2(rw, zw) - math.atan2(L2 * math.sin(t2), L1 + L2 * math.cos(t2))
    t3 = pitch - t1 - t2

    return {
        "shoulder_pan": _to_lerobot("shoulder_pan", pan),
        "shoulder_lift": _to_lerobot("shoulder_lift", t1),
        "elbow_flex": _to_lerobot("elbow_flex", t2),
        "wrist_flex": _to_lerobot("wrist_flex", t3),
    }
