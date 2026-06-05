import sys
import time
from pathlib import Path

import imumocap
import imumocap.file
import imumocap.solvers
import imumocap.viewer
import numpy as np

import time_series

DATA_DIR = Path(__file__).parent / "converted-data" / "Take03"
FPS = 30

dont_block = "dont_block" in sys.argv

# BVH segment name → imumocap link name, in topological order (parent before child).
# Omitted: Neck1 (extra neck), Left/RightHandMiddle metacarpals (no BVH equivalent).
# BVH uses: tiny ~3mm base node + 3 phalanges for Index/Ring/Pinky; 3 phalanges only
# for Middle (no metacarpal); 3 nodes for Thumb (maps to Metacarpal/Proximal/Distal).
# imumocap uses Roman numerals: I=thumb, II=index, III=middle, IV=ring, V=pinky.
BVH_TO_LINK = {
    # --- Body ---
    "Hips": "Pelvis",
    "Spine": "Lower Lumbar",
    "Spine1": "Upper Lumbar",
    "Spine2": "Lower Torso",
    "Spine3": "Upper Torso",
    "Neck": "Neck",
    "Head": "Head",
    # Left arm
    "LeftShoulder": "Left Shoulder",
    "LeftArm": "Left Upper Arm",
    "LeftForeArm": "Left Forearm",
    "LeftHand": "Left Carpus",
    # Left hand — thumb (I): 3 BVH nodes → 3 imumocap nodes
    "LeftHandThumb1": "Left I Metacarpal",
    "LeftHandThumb2": "Left I Proximal",
    "LeftHandThumb3": "Left I Distal",
    # Left hand — index (II): base + 3 phalanges → metacarpal + 3 phalanges
    "LeftHandIndex": "Left II Metacarpal",
    "LeftHandIndex1": "Left II Proximal",
    "LeftHandIndex2": "Left II Middle",
    "LeftHandIndex3": "Left II Distal",
    # Left hand — middle (III): 3 phalanges only, metacarpal left at rest
    "LeftHandMiddle1": "Left III Proximal",
    "LeftHandMiddle2": "Left III Middle",
    "LeftHandMiddle3": "Left III Distal",
    # Left hand — ring (IV): base + 3 phalanges → metacarpal + 3 phalanges
    "LeftHandRing": "Left IV Metacarpal",
    "LeftHandRing1": "Left IV Proximal",
    "LeftHandRing2": "Left IV Middle",
    "LeftHandRing3": "Left IV Distal",
    # Left hand — pinky (V): base + 3 phalanges → metacarpal + 3 phalanges
    "LeftHandPinky": "Left V Metacarpal",
    "LeftHandPinky1": "Left V Proximal",
    "LeftHandPinky2": "Left V Middle",
    "LeftHandPinky3": "Left V Distal",
    # Right arm
    "RightShoulder": "Right Shoulder",
    "RightArm": "Right Upper Arm",
    "RightForeArm": "Right Forearm",
    "RightHand": "Right Carpus",
    # Right hand — thumb (I)
    "RightHandThumb1": "Right I Metacarpal",
    "RightHandThumb2": "Right I Proximal",
    "RightHandThumb3": "Right I Distal",
    # Right hand — index (II)
    "RightHandIndex": "Right II Metacarpal",
    "RightHandIndex1": "Right II Proximal",
    "RightHandIndex2": "Right II Middle",
    "RightHandIndex3": "Right II Distal",
    # Right hand — middle (III)
    "RightHandMiddle1": "Right III Proximal",
    "RightHandMiddle2": "Right III Middle",
    "RightHandMiddle3": "Right III Distal",
    # Right hand — ring (IV)
    "RightHandRing": "Right IV Metacarpal",
    "RightHandRing1": "Right IV Proximal",
    "RightHandRing2": "Right IV Middle",
    "RightHandRing3": "Right IV Distal",
    # Right hand — pinky (V)
    "RightHandPinky": "Right V Metacarpal",
    "RightHandPinky1": "Right V Proximal",
    "RightHandPinky2": "Right V Middle",
    "RightHandPinky3": "Right V Distal",
    # --- Legs ---
    "LeftUpLeg": "Left Upper Leg",
    "LeftLeg": "Left Lower Leg",
    "LeftFoot": "Left Foot",
    "LeftToeBase": "Left Toe",
    "RightUpLeg": "Right Upper Leg",
    "RightLeg": "Right Lower Leg",
    "RightFoot": "Right Foot",
    "RightToeBase": "Right Toe",
}

# Load per-segment quaternion time series from converted CSVs
print("Loading BVH quaternion data...")
raw: dict[str, np.ndarray] = {}
seconds = None

for bvh_name in BVH_TO_LINK:
    data = np.loadtxt(DATA_DIR / f"{bvh_name}.csv", delimiter=",", skiprows=1)
    if seconds is None:
        seconds = data[:, 0] / 1e6  # microseconds → seconds
    raw[bvh_name] = data[:, 1:5]  # columns: W X Y Z

bvh_ts = time_series.QuaternionTimeSeries(seconds=seconds, names=tuple(BVH_TO_LINK.keys()), quaternion=raw)
print(f"  {len(bvh_ts)} frames at {bvh_ts.sample_rate:.1f} Hz ({bvh_ts.duration:.2f}s)")

# Subsample to target FPS
step = max(1, round(bvh_ts.sample_rate / FPS))
frame_indices = range(0, len(bvh_ts), step)

# Load model
model = imumocap.file.load_model("model.json")

# Build pose frames by driving links with BVH global quaternions.
# set_joint_world converts world-space rotation to each link's local joint frame.
# Parent links must be set before children so child origins are up-to-date.
print(f"Building {len(list(frame_indices))} pose frames at {FPS} fps...")
pose_frames: list[imumocap.Pose] = []

for i in frame_indices:
    bvh_frame = bvh_ts.get(index=i)  # {bvh_name: Matrix}
    for bvh_name, link_name in BVH_TO_LINK.items():
        model.links[link_name].set_joint_world(bvh_frame[bvh_name])

    imumocap.solvers.floor(model)
    pose_frames.append(model.get_pose())

print("Done.")

# Plot
imumocap.plot(model, pose_frames, block=not dont_block)

# Stream to IMU Mocap Viewer
viewer = imumocap.viewer.Connection()

while True:
    for pose_frame in pose_frames:
        time.sleep(1 / FPS)

        model.set_pose(pose_frame)
        
        viewer.send_frame(imumocap.viewer.model_to_primitives(model, mirror="Left"))

    if dont_block:
        break
