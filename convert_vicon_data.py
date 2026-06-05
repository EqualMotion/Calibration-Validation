import os
from pathlib import Path
import bvh
import numpy as np
from scipy.spatial.transform import Rotation

TAKE_PATH = Path("vicon-data/Take03.bvh")

if not TAKE_PATH.is_absolute():
    TAKE_PATH = Path(__file__).parent / TAKE_PATH

# ENU→NWU frame correction: -90° about Z
# Remaps axes: X_nwu=Y_enu, Y_nwu=-X_enu, Z_nwu=Z_enu
_R_ENU2NWU = Rotation.from_quat([0.0, 0.0, -1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])

# def parent_name(mocap, joint_name):
#     node = mocap.joint_parent(joint_name)
#     return node.name if node is not None else None


def compute_global_quats(mocap):
    joints = mocap.get_joints_names()

    global_rots = {}

    for joint in joints:
        print(f"Processing Segment: {joint}")

        rotation_channels = [c for c in mocap.joint_channels(joint) if "rotation" in c.lower()] 

        axes = "".join(c[0].upper() for c in rotation_channels)  # e.g. 'XZY'

        euler = np.array(mocap.frames_joint_channels(joint, rotation_channels), dtype=float)

        local = Rotation.from_euler(axes, euler, degrees=True)

        node = mocap.joint_parent(joint)

        parent = node.name if node is not None else None

        if parent is None:
            global_rots[joint] = local
        else:
            global_rots[joint] = global_rots[parent] * local

    return global_rots


def write_output(mocap, global_rots, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    joints = mocap.get_joints_names()
    n_frames = mocap.nframes
    timestamps_us = (np.arange(n_frames) * mocap.frame_time * 1e6).astype(np.int64)

    print(f"\n{'Segment':<32} {'Frames':>8}  {'Duration (s)':>12}")
    print("-" * 56)

    for joint in joints:
        # Apply ENU→NWU and convert to [W, X, Y, Z]
        nwu = (_R_ENU2NWU * global_rots[joint]).as_quat()  # scipy: [x, y, z, w]
        quats_wxyz = np.column_stack([nwu[:, 3], nwu[:, 0], nwu[:, 1], nwu[:, 2]])

        out_path = os.path.join(output_dir, f"{joint}.csv")
        with open(out_path, "w") as f:
            f.write("Timestamp (us),W,X,Y,Z\n")
            lines = [f"{t},{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}" for t, q in zip(timestamps_us, quats_wxyz)]
            f.write("\n".join(lines) + "\n")

        duration = n_frames * mocap.frame_time
        print(f"{joint:<32} {n_frames:>8}  {duration:>12.3f}")

    print(f"\nOutput: {output_dir}/  ({len(joints)} files)")


output_dir = Path(__file__).resolve().parent / "converted-data" / TAKE_PATH.stem

print(f"Parsing {TAKE_PATH} ...")
with open(TAKE_PATH) as f:
    mocap = bvh.Bvh(f.read())

print(f"  {len(mocap.get_joints_names())} segments, {mocap.nframes} frames at {1.0 / mocap.frame_time:.1f} Hz ({mocap.nframes * mocap.frame_time:.2f} s)")

global_rots = compute_global_quats(mocap)

write_output(mocap, global_rots, output_dir)

segments_csv = output_dir / "Segments.csv"

with open(segments_csv, "w") as f:
    f.write("Segment,Length (m)\n")

    for joint in mocap.get_joints_names():
        offset = mocap.joint_offset(joint)

        length_m = np.linalg.norm(offset) / 100.0  # BVH offsets are in cm

        f.write(f"{joint},{length_m:.6f}\n")

print(f"Segment lengths written to {segments_csv}")
