"""Checks of the preprocessing building blocks on small synthetic inputs."""

import numpy as np
import pytest

from preprocess.esd_preprocess import filter_signals, parse_chbmit_summary, segment_starts, windows, zscore
from preprocess.fer_preprocess import extract_patches, graph_sequence, select_keyframes
from preprocess.har_preprocess import bone_stream, is_training, joint_stream, parse_name, resample


def test_select_keyframes_keeps_order_and_reference_frame():
    rng = np.random.default_rng(0)
    landmarks = rng.normal(size=(20, 51, 2)).cumsum(axis=0)
    keyframes = select_keyframes(landmarks, 9)
    assert len(keyframes) == 9 and keyframes[0] == 0
    assert np.all(np.diff(keyframes) > 0)
    short = select_keyframes(landmarks[:4], 9)
    assert len(short) == 9 and short[-1] == 3


def test_extract_patches_handles_borders():
    magnitude = np.arange(100, dtype=np.float32).reshape(10, 10)
    patches = extract_patches(magnitude, np.array([[0.0, 0.0], [9.0, 9.0], [5.0, 4.0]]), 7)
    assert patches.shape == (3, 7, 7)
    assert patches[2, 3, 3] == magnitude[4, 5]


def test_graph_sequence_shapes():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    frames = (rng.uniform(0, 255, (12, 64, 64))).astype(np.uint8)
    landmarks = rng.uniform(10, 50, (12, 51, 2)).astype(np.float32)
    out = graph_sequence(frames, landmarks, num_keyframes=8, patch_size=7)
    assert out["patches"].shape == (8, 51, 7, 7) and out["landmarks"].shape == (9, 51, 2)
    assert (out["patches"] >= 0).all()


def test_chbmit_summary_parsing(tmp_path):
    text = """File Name: chb01_03.edf
File Start Time: 13:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb01_04.edf
Number of Seizures in File: 0
"""
    path = tmp_path / "chb01-summary.txt"
    path.write_text(text)
    seizures = parse_chbmit_summary(str(path))
    assert seizures == {"chb01_03.edf": [(2996.0, 3036.0)], "chb01_04.edf": []}


def test_segmentation_and_windows():
    ictal, inter = segment_starts(num_samples=60 * 256, seizures=[(20.0, 30.0)], segment_sec=4.0, ictal_step=1.0)
    assert ictal == [s * 256 for s in range(20, 27)]
    assert all(start + 4 * 256 <= 20 * 256 or start >= 30 * 256 for start in inter)
    segment = np.random.default_rng(0).normal(size=(18, 4 * 256))
    assert windows(segment).shape == (7, 18, 256)
    filtered = zscore(filter_signals(segment, 256))
    assert filtered.shape == segment.shape and np.allclose(filtered.mean(axis=1), 0, atol=1e-6)


def test_skeleton_streams_and_splits():
    rng = np.random.default_rng(0)
    sequence = rng.normal(size=(30, 25, 3)).astype(np.float32)
    joints = joint_stream(resample(sequence, 64))
    assert joints.shape == (64, 25, 3) and np.allclose(joints[:, 1], 0)
    bones = bone_stream(joints)
    assert bones.shape == joints.shape and np.allclose(bones[:, 1], 0)
    assert parse_name("S001C002P003R001A045") == (1, 2, 3, 45)
    assert is_training("S001C002P001R001A001", "xsub", "ntu60") and not is_training("S001C002P003R001A001", "xsub", "ntu60")
    assert is_training("S001C002P003R001A001", "xview", "ntu60") and not is_training("S001C001P003R001A001", "xview", "ntu60")
    assert is_training("S002C001P003R001A001", "xset", "ntu120") and not is_training("S001C001P003R001A001", "xset", "ntu120")
