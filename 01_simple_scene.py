"""
01_simple_scene.py - Minimal SO100 viewer
==========================================
Usage:
    mjpython 01_simple_scene.py --urdf path/to/so100.urdf

Opens an interactive MuJoCo viewer with the SO100 arm in its default pose.
Use the mouse to orbit, zoom, and pan.
"""

import argparse
import os
import re
import shutil
import tempfile

import mujoco
import mujoco.viewer


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

        scene_xml = f"""<mujoco model="so100_simple">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <asset>{assets}</asset>
  <worldbody>
    <light name="sun" pos="1 1 3" dir="-0.3 -0.3 -1" diffuse="1.1 1.1 1.0" castshadow="true"/>
    <light name="fill" pos="-1 0 2" dir="0.5 0 -1" diffuse="0.3 0.3 0.4" castshadow="false"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.38 0.38 0.38 1"/>
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
    model.dof_damping[:] = 5.0
    return model, mujoco.MjData(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True, help="Path to so100.urdf")
    args = parser.parse_args()

    model, data = load_model(args.urdf)
    mujoco.mj_forward(model, data)
    print(f"[✓] SO100 loaded: {model.nbody} bodies, {model.nv} DOF")
    print("    Mouse: left-drag=orbit  scroll=zoom  right-drag=pan  Q=quit\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=lambda k: None) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.2
        viewer.cam.lookat[:] = [0, 0, 0.2]

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
