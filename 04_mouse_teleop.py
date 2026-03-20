"""
04_mouse_teleop.py - Mouse-drag teleop, N SO100 arms
=====================================================
Usage:
    mjpython 04_mouse_teleop.py --urdf path/to/so100.urdf [--n-robots 3]

Drag the colored spheres to move each arm's end-effector.
MuJoCo handles mouse picking natively — just Ctrl+click and drag a sphere.

    --n-robots N   number of arms to spawn (default: 1, max: 6)

Scene: table + back wall + spotlight — same layout as the shadow art designer.
Shadow frames are written to /tmp/shadow_frame.npy for a shadow viewer.

Keys:
    Q   quit
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


IK_DAMPING = 1e-3
IK_ITERS = 40
IK_ALPHA = 0.3
CTRL_HZ = 60

ROBOT_SPACING = 0.28       # metres between adjacent arm bases

TABLE_Y   = 0.30
TABLE_HX  = 0.75           # half-length — auto-expands for more robots
TABLE_HY  = 0.42
TABLE_THK = 0.015
LEG_H     = 0.37

WALL_Y = -1.10
WALL_Z =  0.40

SHADOW_W = 640
SHADOW_H = 480
SHADOW_HZ = 20
SHADOW_FRAME = "/tmp/shadow_frame.npy"

# One colour per arm (up to 6)
MOCAP_COLORS = [
    "1.0 0.25 0.25 0.85",   # red
    "0.25 1.0 0.25 0.85",   # green
    "0.25 0.45 1.0  0.85",  # blue
    "1.0  0.80 0.0  0.85",  # yellow
    "1.0  0.25 1.0  0.85",  # magenta
    "0.0  1.0  0.90 0.85",  # cyan
]


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


def rename_bodies(xml, suffix):
    xml = re.sub(
        r'((?:body|joint|geom|site)\s+name=")([^"]+)(")',
        lambda m: m.group(1) + m.group(2) + suffix + m.group(3), xml,
    )
    xml = re.sub(
        r'(<(?:parent|child)\s+link=")([^"]+)(")',
        lambda m: m.group(1) + m.group(2) + suffix + m.group(3), xml,
    )
    return xml


def load_model(urdf_path, n_robots):
    abs_urdf = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(abs_urdf)

    with open(abs_urdf) as f:
        xml = patch_urdf(f.read())

    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "so100.urdf"), "w") as f:
        f.write(xml)
    for root, _, files in os.walk(urdf_dir):
        for fn in files:
            if fn.lower().endswith(".stl"):
                shutil.copy(os.path.join(root, fn), os.path.join(tmpdir, fn))

    orig = os.getcwd()
    os.chdir(tmpdir)
    try:
        m0 = mujoco.MjModel.from_xml_path("so100.urdf")
        mujoco.mj_saveLastXML("robot.xml", m0)
        with open("robot.xml") as f:
            robot_xml = f.read()

        asset_m = re.search(r"<asset>(.*?)</asset>", robot_xml, re.DOTALL)
        wb_m = re.search(r"<worldbody>(.*?)</worldbody>", robot_xml, re.DOTALL)
        assets = asset_m.group(1) if asset_m else ""
        wb = wb_m.group(1) if wb_m else ""

        suffixes = [f"_{chr(ord('a') + i)}" for i in range(n_robots)]
        total_width = (n_robots - 1) * ROBOT_SPACING
        xs = [i * ROBOT_SPACING - total_width / 2 for i in range(n_robots)]

        # Table wide enough to hold all robots
        table_hx = max(TABLE_HX, total_width / 2 + 0.22)

        robot_bodies = ""
        for x, suf in zip(xs, suffixes):
            renamed = rename_bodies(wb, suf)
            robot_bodies += (
                f'<body name="robot{suf}" pos="{x:.3f} 0 0">\n'
                f"  {renamed}\n</body>\n"
            )

        mocap_bodies = ""
        for i, (x, suf) in enumerate(zip(xs, suffixes)):
            col = MOCAP_COLORS[i % len(MOCAP_COLORS)]
            mocap_bodies += (
                f'<body name="ee{suf}_mocap"'
                f' pos="{x:.3f} {TABLE_Y - 0.10:.3f} 0.25" mocap="true">\n'
                f'  <geom type="sphere" size="0.030" rgba="{col}"/>\n'
                f'</body>\n'
            )

        shadow_cam_y = TABLE_Y + 0.90

        scene_xml = f"""<mujoco model="so100_mouse_teleop">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <headlight ambient="0.05 0.05 0.05" diffuse="0 0 0" specular="0 0 0"/>
    <global offheight="{SHADOW_H}" offwidth="{SHADOW_W}"/>
    <quality shadowsize="8192"/>
  </visual>
  <asset>{assets}</asset>
  <worldbody>

    <camera name="shadow_cam"
            pos="0 {shadow_cam_y:.3f} {WALL_Z:.3f}"
            xyaxes="1 0 0  0 0 -1"
            fovy="70"/>

    <light name="spot" pos="0 1.20 0.50" dir="0 -1 -0.10"
           diffuse="2.0 2.0 2.0" specular="0 0 0"
           castshadow="true" cutoff="60" exponent="2"/>

    <geom name="floor" type="plane" size="4 4 0.1"
          pos="0 0 -0.77" rgba="0.30 0.30 0.28 1"/>

    <geom name="backwall" type="box"
          size="{table_hx:.3f} 0.025 1.20"
          pos="0 {WALL_Y:.2f} {WALL_Z:.2f}" rgba="0.97 0.97 0.95 1"/>

    <body name="table" pos="0 {TABLE_Y:.2f} 0">
      <geom name="tabletop" type="box"
            size="{table_hx:.3f} {TABLE_HY} {TABLE_THK}"
            pos="0 0 -{TABLE_THK}" rgba="0.82 0.68 0.48 1"/>
      <geom name="leg_fl" type="box" size="0.025 0.025 {LEG_H}"
            pos=" {table_hx-0.06:.3f}  {TABLE_HY-0.06:.3f} -{LEG_H+TABLE_THK*2:.3f}"
            rgba="0.65 0.53 0.36 1"/>
      <geom name="leg_fr" type="box" size="0.025 0.025 {LEG_H}"
            pos=" {table_hx-0.06:.3f} -{TABLE_HY-0.06:.3f} -{LEG_H+TABLE_THK*2:.3f}"
            rgba="0.65 0.53 0.36 1"/>
      <geom name="leg_bl" type="box" size="0.025 0.025 {LEG_H}"
            pos="-{table_hx-0.06:.3f}  {TABLE_HY-0.06:.3f} -{LEG_H+TABLE_THK*2:.3f}"
            rgba="0.65 0.53 0.36 1"/>
      <geom name="leg_br" type="box" size="0.025 0.025 {LEG_H}"
            pos="-{table_hx-0.06:.3f} -{TABLE_HY-0.06:.3f} -{LEG_H+TABLE_THK*2:.3f}"
            rgba="0.65 0.53 0.36 1"/>
    </body>

    <body name="robots_root" pos="0 {TABLE_Y:.2f} 0">
      {robot_bodies}
    </body>

    {mocap_bodies}

  </worldbody>
</mujoco>"""

        with open("scene.xml", "w") as f:
            f.write(scene_xml)
        model = mujoco.MjModel.from_xml_path("scene.xml")
    finally:
        os.chdir(orig)

    shutil.rmtree(tmpdir, ignore_errors=True)
    model.dof_damping[:] = 10.0
    data = mujoco.MjData(model)
    print(f"[✓] {n_robots} arm(s) | {model.nbody} bodies | {model.nv} DOF")
    return model, data


def ik_solve(model, data, target_pos, target_quat, ee_id, qslice):
    for _ in range(IK_ITERS):
        mujoco.mj_fwdPosition(model, data)
        pos_err = target_pos - data.xpos[ee_id]
        quat_err = np.zeros(3)
        mujoco.mju_subQuat(quat_err, target_quat, data.xquat[ee_id])
        err = np.concatenate([pos_err, quat_err])
        if np.linalg.norm(err) < 1e-4:
            break
        J = np.zeros((6, model.nv))
        mujoco.mj_jacBody(model, data, J[:3], J[3:], ee_id)
        Jq = J[:, qslice]
        dq = IK_ALPHA * Jq.T @ np.linalg.solve(Jq @ Jq.T + IK_DAMPING * np.eye(6), err)
        data.qpos[qslice] += np.clip(dq, -0.1, 0.1)
        mujoco.mj_normalizeQuat(model, data.qpos)


def extract_shadow(rgb):
    gray = rgb.mean(axis=2).astype(np.float32)
    max_val = gray.max()
    if max_val < 10:
        return np.full(gray.shape[:2], 255, dtype=np.uint8)
    return np.where(gray < max_val * 0.75, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--n-robots", type=int, default=1,
                        help="Number of arms to spawn (1–6, default 1)")
    args = parser.parse_args()

    n = max(1, min(6, args.n_robots))
    suffixes = [f"_{chr(ord('a') + i)}" for i in range(n)]

    model, data = load_model(args.urdf, n)
    mujoco.mj_forward(model, data)

    dofs_per = model.nv // n
    slices = [list(range(i * dofs_per, (i + 1) * dofs_per)) for i in range(n)]

    ee_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"gripper{s}")
        for s in suffixes
    ]
    mocap_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ee{s}_mocap")
        for s in suffixes
    ]
    mocap_ids = [model.body_mocapid[b] for b in mocap_body_ids]

    # Park mocap handles at current EE positions
    for i, mid in enumerate(mocap_ids):
        data.mocap_pos[mid] = data.xpos[ee_ids[i]].copy()
    mujoco.mj_forward(model, data)

    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False
        print("[warn] opencv-python not found — shadow extraction disabled")

    renderer = mujoco.Renderer(model, height=SHADOW_H, width=SHADOW_W)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "shadow_cam")
    shadow_tick = 0
    shadow_every = max(1, CTRL_HZ // SHADOW_HZ)

    state = {"quit": False}

    def key_cb(k):
        if k == ord("Q"):
            state["quit"] = True

    print("Ctrl+click a colored sphere and drag to move that arm.")
    print("Q = quit\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.lookat[:] = [0.0, TABLE_Y, 0.25]
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15

        while viewer.is_running() and not state["quit"]:
            t0 = time.time()

            for i in range(n):
                ik_solve(
                    model, data,
                    data.mocap_pos[mocap_ids[i]],
                    data.mocap_quat[mocap_ids[i]],
                    ee_ids[i], slices[i],
                )

            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()

            if shadow_tick % shadow_every == 0:
                renderer.update_scene(data, camera=cam_id)
                rgb = renderer.render()
                shadow = extract_shadow(rgb)
                np.save(SHADOW_FRAME, shadow)

            shadow_tick += 1
            time.sleep(max(0, 1 / CTRL_HZ - (time.time() - t0)))

    print("Done.")


if __name__ == "__main__":
    main()
