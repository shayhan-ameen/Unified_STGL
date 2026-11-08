"""Facial expression preprocessing: video -> landmark-motion graph sequence.

Pipeline (paper Sec. "Facial Expression Recognition")
    1. read the frames of a clip,
    2. detect 68 facial landmarks with dlib, align the face to 224 x 224 using
       the eye centres, keep the 51 inner-face landmarks,
    3. rank frames by the summed landmark displacement between consecutive
       frames and keep the T + 1 highest-ranked frames in temporal order,
    4. compute dense Farneback optical flow between successive keyframes and
       take its magnitude,
    5. cut a Sz x Sz magnitude patch around every landmark,
    6. save ``patches [T, 51, Sz, Sz]`` and the aligned ``landmarks [T + 1, 51, 2]``.

Usage
-----
    python preprocess/fer_preprocess.py --dataset afew    --input <AFEW root>  --output data/processed/afew \
        --predictor shape_predictor_68_face_landmarks.dat --num-keyframes 8
    python preprocess/fer_preprocess.py --dataset oulu_vl --input <Oulu-CASIA root> --output data/processed/oulu_vl ...
    python preprocess/fer_preprocess.py --dataset smic_hs --input <SMIC root> --output data/processed/smic_hs ...
    python preprocess/fer_preprocess.py --dataset casme3  --annotations casme3_clips.csv --output data/processed/casme3 ...

``--annotations`` accepts a CSV with the columns ``id, subject, label, split,
frames`` (``frames`` is a video file or a directory of images) and optional
``onset, offset`` frame indices; it works for every dataset and is the way to
feed CAS(ME)^3, CK+, and CASME II, whose official annotation tables differ.

Expected raw layouts for the built-in listers
    afew     <root>/Train/<Emotion>/<clip>.avi, <root>/Val/<Emotion>/<clip>.avi
    oulu_vl  <root>/OriginalImg/VL/Strong/P<subject>/<Emotion>/<frame>.jpeg
    smic_hs  <root>/HS/s<subject>/micro/<positive|negative|surprise>/<clip>/<frame>.bmp
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")
INNER_LANDMARKS = slice(17, 68)   # 51 landmarks without the jaw line


# --------------------------------------------------------------------------- #
# Frame reading
# --------------------------------------------------------------------------- #


def read_frames(source: str, onset: Optional[int] = None, offset: Optional[int] = None) -> List[np.ndarray]:
    """Frames (BGR) of a video file or an image directory, optionally cut to [onset, offset]."""
    frames: List[np.ndarray] = []
    if os.path.isdir(source):
        files = sorted(f for f in os.listdir(source) if f.lower().endswith(IMAGE_EXTENSIONS))
        for name in files:
            image = cv2.imread(os.path.join(source, name))
            if image is not None:
                frames.append(image)
    else:
        capture = cv2.VideoCapture(source)
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frames.append(image)
        capture.release()
    if onset is not None or offset is not None:
        lo = max(int(onset or 0), 0)
        hi = int(offset) + 1 if offset is not None else len(frames)
        frames = frames[lo:hi]
    return frames


# --------------------------------------------------------------------------- #
# Landmarks and alignment
# --------------------------------------------------------------------------- #


class LandmarkDetector:
    """dlib HOG face detector + 68-point shape predictor."""

    def __init__(self, predictor_path: str) -> None:
        import dlib  # imported here so the rest of the module works without dlib

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def __call__(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray, face)
        return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)


def alignment_transform(landmarks68: np.ndarray, size: int = 224, eye_y: float = 0.38, eye_dist: float = 0.32) -> np.ndarray:
    """Similarity transform that maps the eye centres to canonical positions in a ``size x size`` image."""
    left_eye = landmarks68[36:42].mean(axis=0)
    right_eye = landmarks68[42:48].mean(axis=0)
    dx, dy = right_eye - left_eye
    angle = np.degrees(np.arctan2(dy, dx))
    scale = (eye_dist * size) / max(np.hypot(dx, dy), 1e-6)
    centre = (left_eye + right_eye) / 2.0
    matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), float(angle), float(scale))
    matrix[0, 2] += size / 2.0 - centre[0]
    matrix[1, 2] += eye_y * size - centre[1]
    return matrix


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return homogeneous @ matrix.T


def detect_and_align(frames: Sequence[np.ndarray], detector: LandmarkDetector, size: int = 224) -> Tuple[np.ndarray, np.ndarray]:
    """Aligned grayscale frames ``[F, size, size]`` and inner landmarks ``[F, 51, 2]``.

    Frames without a detected face reuse the landmarks of the previous frame
    (or the next detected frame for a leading gap).
    """
    raw: List[Optional[np.ndarray]] = [detector(frame) for frame in frames]
    if all(lm is None for lm in raw):
        raise RuntimeError("No face detected in any frame of the clip")
    first = next(lm for lm in raw if lm is not None)
    filled = []
    previous = first
    for lm in raw:
        previous = lm if lm is not None else previous
        filled.append(previous)
    aligned_frames, aligned_landmarks = [], []
    for frame, lm in zip(frames, filled):
        matrix = alignment_transform(lm, size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aligned_frames.append(cv2.warpAffine(gray, matrix, (size, size), flags=cv2.INTER_LINEAR))
        aligned_landmarks.append(apply_transform(lm, matrix)[INNER_LANDMARKS])
    return np.stack(aligned_frames), np.stack(aligned_landmarks).astype(np.float32)


# --------------------------------------------------------------------------- #
# Keyframes, optical flow, patches
# --------------------------------------------------------------------------- #


def select_keyframes(landmarks: np.ndarray, num_keyframes: int) -> np.ndarray:
    """Indices of the ``num_keyframes`` frames with the largest landmark displacement, in temporal order.

    The displacement of frame t is the summed Euclidean motion of all landmarks
    from frame t - 1; the first frame is always kept as the reference.  Clips
    shorter than ``num_keyframes`` repeat their last frame.
    """
    count = landmarks.shape[0]
    if count <= num_keyframes:
        return np.concatenate([np.arange(count), np.full(num_keyframes - count, count - 1)])
    motion = np.linalg.norm(landmarks[1:] - landmarks[:-1], axis=-1).sum(axis=1)   # [F - 1]
    ranked = np.argsort(-motion, kind="stable")[: num_keyframes - 1] + 1
    return np.sort(np.concatenate([[0], ranked]))


def flow_magnitude(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Dense Farneback optical flow magnitude between two grayscale frames."""
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)


def extract_patches(magnitude: np.ndarray, landmarks: np.ndarray, patch_size: int) -> np.ndarray:
    """``[N, Sz, Sz]`` patches centred at the landmark positions (border replicated)."""
    half = patch_size // 2
    padded = np.pad(magnitude, half, mode="edge")
    patches = np.zeros((landmarks.shape[0], patch_size, patch_size), dtype=np.float32)
    height, width = magnitude.shape
    for k, (x, y) in enumerate(landmarks):
        cx = int(np.clip(round(float(x)), 0, width - 1)) + half
        cy = int(np.clip(round(float(y)), 0, height - 1)) + half
        patches[k] = padded[cy - half : cy + half + 1, cx - half : cx + half + 1]
    return patches


def graph_sequence(frames: np.ndarray, landmarks: np.ndarray, num_keyframes: int, patch_size: int) -> Dict[str, np.ndarray]:
    """From aligned frames/landmarks to the cached arrays (``num_keyframes`` + 1 frames, T intervals)."""
    keyframes = select_keyframes(landmarks, num_keyframes + 1)
    patches = []
    for a, b in zip(keyframes[:-1], keyframes[1:]):
        magnitude = flow_magnitude(frames[a], frames[b])
        patches.append(extract_patches(magnitude, landmarks[a], patch_size))
    return {
        "patches": np.stack(patches).astype(np.float32),          # [T, 51, Sz, Sz]
        "landmarks": landmarks[keyframes].astype(np.float32),      # [T + 1, 51, 2]
        "keyframes": keyframes.astype(np.int64),
    }


# --------------------------------------------------------------------------- #
# Clip listing
# --------------------------------------------------------------------------- #


def list_afew(root: str) -> List[Dict[str, str]]:
    clips = []
    for split_dir, split in (("Train", "train"), ("Val", "val")):
        for emotion in sorted(os.listdir(os.path.join(root, split_dir))):
            for path in sorted(glob.glob(os.path.join(root, split_dir, emotion, "*"))):
                if path.lower().endswith(VIDEO_EXTENSIONS) or os.path.isdir(path):
                    clip_id = os.path.splitext(os.path.basename(path))[0]
                    clips.append({"id": f"{split}_{clip_id}", "subject": clip_id, "label": emotion, "split": split, "frames": path})
    return clips


def list_oulu_vl(root: str) -> List[Dict[str, str]]:
    base = os.path.join(root, "OriginalImg", "VL", "Strong")
    clips = []
    for subject in sorted(os.listdir(base)):
        for emotion in sorted(os.listdir(os.path.join(base, subject))):
            frames = os.path.join(base, subject, emotion)
            if os.path.isdir(frames):
                clips.append({"id": f"{subject}_{emotion}", "subject": subject, "label": emotion, "split": "", "frames": frames})
    return clips


def list_smic_hs(root: str) -> List[Dict[str, str]]:
    base = os.path.join(root, "HS")
    clips = []
    for subject in sorted(os.listdir(base)):
        for emotion in ("positive", "negative", "surprise"):
            for frames in sorted(glob.glob(os.path.join(base, subject, "micro", emotion, "*"))):
                if os.path.isdir(frames):
                    clip_id = os.path.basename(frames)
                    clips.append({"id": f"{subject}_{clip_id}", "subject": subject, "label": emotion, "split": "", "frames": frames})
    return clips


def list_from_annotations(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "subject", "label", "frames"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"Annotation file lacks the columns {sorted(missing)}")
    return rows


LISTERS = {"afew": list_afew, "oulu_vl": list_oulu_vl, "smic_hs": list_smic_hs}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def process_clip(clip: Dict[str, str], detector: LandmarkDetector, num_keyframes: int, patch_size: int, image_size: int) -> Dict[str, np.ndarray]:
    onset = int(clip["onset"]) if clip.get("onset") not in (None, "") else None
    offset = int(clip["offset"]) if clip.get("offset") not in (None, "") else None
    frames = read_frames(clip["frames"], onset, offset)
    if len(frames) < 2:
        raise RuntimeError(f"Clip {clip['id']} has fewer than two frames")
    aligned, landmarks = detect_and_align(frames, detector, image_size)
    return graph_sequence(aligned, landmarks, num_keyframes, patch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["afew", "oulu_vl", "smic_hs", "casme3", "ck_plus", "casme2"])
    parser.add_argument("--input", help="raw dataset root for the built-in listers")
    parser.add_argument("--annotations", help="CSV with id, subject, label, split, frames[, onset, offset]")
    parser.add_argument("--output", required=True, help="cache directory, e.g. data/processed/afew")
    parser.add_argument("--predictor", required=True, help="dlib shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--num-keyframes", type=int, default=8, help="T optical-flow intervals (T + 1 keyframes)")
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    if args.annotations:
        clips = list_from_annotations(args.annotations)
    elif args.input and args.dataset in LISTERS:
        clips = LISTERS[args.dataset](args.input)
    else:
        parser.error("Give --annotations, or --input for afew / oulu_vl / smic_hs")

    detector = LandmarkDetector(args.predictor)
    os.makedirs(os.path.join(args.output, "clips"), exist_ok=True)
    index_path = os.path.join(args.output, "index.csv")
    written, failed = 0, []
    with open(index_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "path", "subject", "label", "split"])
        writer.writeheader()
        for k, clip in enumerate(clips, start=1):
            try:
                arrays = process_clip(clip, detector, args.num_keyframes, args.patch_size, args.image_size)
            except Exception as err:  # keep going; report at the end
                failed.append((clip["id"], str(err)))
                continue
            rel_path = os.path.join("clips", f"{clip['id']}.npz")
            np.savez_compressed(os.path.join(args.output, rel_path), **arrays)
            writer.writerow({"id": clip["id"], "path": rel_path, "subject": clip["subject"], "label": clip["label"], "split": clip.get("split", "")})
            written += 1
            if k % 50 == 0 or k == len(clips):
                print(f"[{k}/{len(clips)}] written {written}, failed {len(failed)}")
    print(f"Done: {written} clips cached under {args.output}")
    for clip_id, reason in failed:
        print(f"  skipped {clip_id}: {reason}")


if __name__ == "__main__":
    main()
