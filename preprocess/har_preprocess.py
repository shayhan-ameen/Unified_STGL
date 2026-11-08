"""NTU RGB+D skeleton preprocessing: ``.skeleton`` files -> joint and bone streams.

Pipeline (paper Sec. "Human Activity Recognition")
    1. parse every ``.skeleton`` file (samples listed in the missing-skeleton
       file are skipped), keeping the body with the largest motion energy,
    2. subtract the reference joint (joint 2, middle of the spine) from every
       joint in every frame,
    3. resample the sequence to T frames by linear interpolation,
    4. bone stream: displacement from each joint to its anatomical parent (zero
       for the reference joint),
    5. assign samples to the train / test partition of each protocol and save
       the arrays per protocol and split.

Usage
-----
    python preprocess/har_preprocess.py --input <nturgb+d_skeletons> --output data/processed/ntu60 \
        --benchmark ntu60 --missing <samples_with_missing_skeletons.txt> --num-frames 64
    python preprocess/har_preprocess.py --input <nturgbd+d_skeletons 120> --output data/processed/ntu120 --benchmark ntu120 ...

Sample names follow ``SsssCcccPpppRrrrAaaa`` (setup, camera, performer, replication, action).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.graphs import NTU_NUM_JOINTS, NTU_ROOT_JOINT, skeleton_parents  # noqa: E402

TRAIN_SUBJECTS_60 = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TRAIN_SUBJECTS_120 = TRAIN_SUBJECTS_60 + [45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82,
                                          83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
TRAIN_CAMERAS_XVIEW = [2, 3]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def read_skeleton(path: str) -> List[Dict[int, np.ndarray]]:
    """Per frame, a mapping body-id -> ``[25, 3]`` joint coordinates."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().split("\n")
    cursor = 0
    num_frames = int(lines[cursor]); cursor += 1
    frames: List[Dict[int, np.ndarray]] = []
    for _ in range(num_frames):
        bodies: Dict[int, np.ndarray] = {}
        num_bodies = int(lines[cursor]); cursor += 1
        for _ in range(num_bodies):
            info = lines[cursor].split(); cursor += 1
            body_id = int(info[0])
            num_joints = int(lines[cursor]); cursor += 1
            joints = np.zeros((NTU_NUM_JOINTS, 3), dtype=np.float32)
            for j in range(num_joints):
                values = lines[cursor].split(); cursor += 1
                if j < NTU_NUM_JOINTS:
                    joints[j] = [float(values[0]), float(values[1]), float(values[2])]
            bodies[body_id] = joints
        frames.append(bodies)
    return frames


def most_active_body(frames: List[Dict[int, np.ndarray]]) -> np.ndarray:
    """``[F, 25, 3]`` trajectory of the body with the largest motion energy."""
    ids = sorted({body_id for frame in frames for body_id in frame})
    if not ids:
        raise ValueError("No body in the recording")
    tracks = {}
    for body_id in ids:
        track = np.zeros((len(frames), NTU_NUM_JOINTS, 3), dtype=np.float32)
        present = np.zeros(len(frames), dtype=bool)
        for t, frame in enumerate(frames):
            if body_id in frame:
                track[t] = frame[body_id]
                present[t] = True
        tracks[body_id] = (track, present)
    def energy(item):
        track, present = item
        return float(track[present].std(axis=0).sum()) if present.sum() > 1 else 0.0
    best = max(ids, key=lambda b: energy(tracks[b]))
    track, present = tracks[best]
    return track[present]


# --------------------------------------------------------------------------- #
# Normalisation and streams
# --------------------------------------------------------------------------- #


def resample(sequence: np.ndarray, num_frames: int) -> np.ndarray:
    """Linear interpolation of ``[F, N, 3]`` to ``[num_frames, N, 3]`` preserving the temporal order."""
    source = np.linspace(0.0, 1.0, num=sequence.shape[0])
    target = np.linspace(0.0, 1.0, num=num_frames)
    flat = sequence.reshape(sequence.shape[0], -1)
    out = np.stack([np.interp(target, source, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape(num_frames, *sequence.shape[1:]).astype(np.float32)


def joint_stream(sequence: np.ndarray) -> np.ndarray:
    """Coordinates relative to the reference joint, per frame."""
    return sequence - sequence[:, NTU_ROOT_JOINT : NTU_ROOT_JOINT + 1]


def bone_stream(joints: np.ndarray) -> np.ndarray:
    """Displacement from every joint to its parent; the root joint gets zeros."""
    parents = skeleton_parents()
    bones = joints - joints[:, parents]
    bones[:, NTU_ROOT_JOINT] = 0.0
    return bones


def parse_name(name: str) -> Tuple[int, int, int, int]:
    """(setup, camera, performer, action) from ``SsssCcccPpppRrrrAaaa``."""
    return int(name[1:4]), int(name[5:8]), int(name[9:12]), int(name[17:20])


def is_training(name: str, protocol: str, benchmark: str) -> bool:
    setup, camera, performer, _ = parse_name(name)
    if protocol == "xsub":
        return performer in (TRAIN_SUBJECTS_60 if benchmark == "ntu60" else TRAIN_SUBJECTS_120)
    if protocol == "xview":
        return camera in TRAIN_CAMERAS_XVIEW
    if protocol == "xset":
        return setup % 2 == 0
    raise ValueError(f"Unknown protocol '{protocol}'")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="folder with the .skeleton files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", choices=["ntu60", "ntu120"], default="ntu60")
    parser.add_argument("--missing", default=None, help="samples_with_missing_skeletons.txt")
    parser.add_argument("--num-frames", type=int, default=64)
    args = parser.parse_args()

    missing = set()
    if args.missing and os.path.isfile(args.missing):
        with open(args.missing, encoding="utf-8") as handle:
            missing = {line.strip() for line in handle if line.strip()}
    max_action = 60 if args.benchmark == "ntu60" else 120
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(args.input) if f.endswith(".skeleton"))
    names = [n for n in names if n not in missing and parse_name(n)[3] <= max_action]
    protocols = ["xsub", "xview"] if args.benchmark == "ntu60" else ["xsub", "xset"]

    joints_all, bones_all, labels_all = [], [], []
    for k, name in enumerate(names, start=1):
        frames = read_skeleton(os.path.join(args.input, name + ".skeleton"))
        joints = joint_stream(resample(most_active_body(frames), args.num_frames))
        joints_all.append(joints)
        bones_all.append(bone_stream(joints))
        labels_all.append(parse_name(name)[3] - 1)
        if k % 1000 == 0 or k == len(names):
            print(f"[{k}/{len(names)}] parsed")
    joints_all, bones_all = np.stack(joints_all), np.stack(bones_all)
    labels_all = np.array(labels_all, dtype=np.int64)

    for protocol in protocols:
        folder = os.path.join(args.output, protocol)
        os.makedirs(folder, exist_ok=True)
        train_mask = np.array([is_training(n, protocol, args.benchmark) for n in names])
        for split, mask in (("train", train_mask), ("test", ~train_mask)):
            np.save(os.path.join(folder, f"{split}_joint.npy"), joints_all[mask])
            np.save(os.path.join(folder, f"{split}_bone.npy"), bones_all[mask])
            np.save(os.path.join(folder, f"{split}_labels.npy"), labels_all[mask])
            with open(os.path.join(folder, f"{split}_names.txt"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(n for n, m in zip(names, mask) if m) + "\n")
            print(f"{protocol}/{split}: {int(mask.sum())} samples")
    print(f"Done: cached under {args.output}")


if __name__ == "__main__":
    main()
