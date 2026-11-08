"""EEG preprocessing: EDF recordings -> windowed graph segments.

Pipeline (paper Sec. "Epileptic Seizure Detection")
    1. read every EDF file of a patient and derive the bipolar channels
       (18 for CHB-MIT, 16 for Siena; Siena is resampled from 512 to 256 Hz),
    2. third-order Butterworth band-pass 0.5-70 Hz, notch filters at 50 and
       60 Hz, FastICA artifact removal (components with excess kurtosis are
       zeroed), per-channel z-score standardisation,
    3. cut 4 s segments: ictal segments inside the annotated seizures with a
       1 s step (75 % overlap), interictal segments outside seizures without
       overlap (a random subset per file keeps the cache small),
    4. split every segment into T = 7 windows of 1 s with a 0.5 s step,
       giving ``[T, N, 256]`` per segment,
    5. save one ``.npz`` per recording with ``x [S, T, N, 256]`` and ``y [S]``.

Usage
-----
    python preprocess/esd_preprocess.py --dataset chbmit --input <CHB-MIT root> --output data/processed/chbmit
    python preprocess/esd_preprocess.py --dataset siena  --input <Siena root>  --output data/processed/siena

Expected raw layouts
    chbmit  <root>/chbNN/chbNN_XX.edf and <root>/chbNN/chbNN-summary.txt
    siena   <root>/PNNN/PNNN-K.edf and <root>/PNNN/Seizures-list-PNNN.txt
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from scipy.stats import kurtosis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.graphs import CHBMIT_CHANNELS, SIENA_CHANNELS  # noqa: E402

TARGET_RATE = 256


# --------------------------------------------------------------------------- #
# Annotation parsing
# --------------------------------------------------------------------------- #


def parse_chbmit_summary(path: str) -> Dict[str, List[Tuple[float, float]]]:
    """Seizure intervals (seconds) per EDF file name from a ``chbNN-summary.txt``."""
    seizures: Dict[str, List[Tuple[float, float]]] = {}
    current = None
    starts: List[float] = []
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("File Name:"):
                current = line.split(":", 1)[1].strip()
                seizures[current] = []
                starts = []
            elif current and re.match(r"Seizure( \d+)? Start Time:", line):
                starts.append(float(re.search(r"(\d+)\s*seconds", line).group(1)))
            elif current and re.match(r"Seizure( \d+)? End Time:", line):
                end = float(re.search(r"(\d+)\s*seconds", line).group(1))
                seizures[current].append((starts[len(seizures[current])], end))
    return seizures


def _parse_clock(text: str) -> datetime:
    return datetime.strptime(text.strip().replace(".", ":"), "%H:%M:%S")


def parse_siena_list(path: str) -> Dict[str, List[Tuple[float, float]]]:
    """Seizure intervals (seconds from the registration start) per EDF file from ``Seizures-list-PNNN.txt``."""
    seizures: Dict[str, List[Tuple[float, float]]] = {}
    current, registration = None, None
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line.lower().startswith("file name:"):
                current = line.split(":", 1)[1].strip()
                seizures.setdefault(current, [])
            elif line.lower().startswith("registration start time:"):
                registration = _parse_clock(line.split(":", 1)[1])
            elif line.lower().startswith("seizure start time:") and registration is not None:
                start = (_parse_clock(line.split(":", 1)[1]) - registration)
                start = start + timedelta(days=1) if start.total_seconds() < 0 else start
                seizures[current].append((start.total_seconds(), None))
            elif line.lower().startswith("seizure end time:") and current in seizures and seizures[current]:
                end = (_parse_clock(line.split(":", 1)[1]) - registration)
                end = end + timedelta(days=1) if end.total_seconds() < 0 else end
                s, _ = seizures[current][-1]
                seizures[current][-1] = (s, end.total_seconds())
    return {k: [(s, e) for s, e in v if e is not None] for k, v in seizures.items()}


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #


def _normalise_name(name: str) -> str:
    return re.sub(r"^(EEG\s+)", "", name.strip().upper()).replace("T3", "T7").replace("T4", "T8").replace("T5", "P7").replace("T6", "P8")


def bipolar_signals(raw, channels: Sequence[str]) -> Optional[np.ndarray]:
    """``[N, samples]`` bipolar signals, built from bipolar or monopolar EDF channels; None if any is missing."""
    names = {_normalise_name(n): i for i, n in enumerate(raw.ch_names)}
    data = raw.get_data()
    rows = []
    for channel in channels:
        if channel in names:                                     # already bipolar (CHB-MIT)
            rows.append(data[names[channel]])
            continue
        first, second = channel.split("-")
        if first in names and second in names:                  # monopolar (Siena)
            rows.append(data[names[first]] - data[names[second]])
        else:
            return None
    return np.stack(rows).astype(np.float64)


def filter_signals(x: np.ndarray, rate: int) -> np.ndarray:
    b, a = butter(3, [0.5, 70.0], btype="bandpass", fs=rate)
    x = filtfilt(b, a, x, axis=-1)
    for freq in (50.0, 60.0):
        bn, an = iirnotch(freq, Q=30.0, fs=rate)
        x = filtfilt(bn, an, x, axis=-1)
    return x


def remove_artifacts_ica(x: np.ndarray, kurtosis_threshold: float = 5.0, seed: int = 0) -> np.ndarray:
    """FastICA; components whose excess kurtosis exceeds the threshold (ocular / muscle spikes) are removed."""
    from sklearn.decomposition import FastICA

    ica = FastICA(n_components=x.shape[0], random_state=seed, max_iter=500)
    sources = ica.fit_transform(x.T)                            # [samples, components]
    keep = np.abs(kurtosis(sources, axis=0)) <= kurtosis_threshold
    sources[:, ~keep] = 0.0
    return ica.inverse_transform(sources).T


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)


def preprocess_recording(x: np.ndarray, rate: int, use_ica: bool = True) -> np.ndarray:
    if rate != TARGET_RATE:
        x = resample_poly(x, TARGET_RATE, rate, axis=-1)
    x = filter_signals(x, TARGET_RATE)
    if use_ica:
        x = remove_artifacts_ica(x)
    return zscore(x).astype(np.float32)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


def segment_starts(num_samples: int, seizures: Sequence[Tuple[float, float]], segment_sec: float, ictal_step: float,
                   rate: int = TARGET_RATE) -> Tuple[List[int], List[int]]:
    """Start samples of ictal (overlapping) and interictal (non-overlapping) segments."""
    seg = int(segment_sec * rate)
    ictal: List[int] = []
    for start_sec, end_sec in seizures:
        s, e = int(start_sec * rate), int(end_sec * rate)
        ictal.extend(range(s, max(s, e - seg) + 1, int(ictal_step * rate)))
    inter: List[int] = []
    for start in range(0, num_samples - seg + 1, seg):
        if all(start + seg <= int(s * rate) or start >= int(e * rate) for s, e in seizures):
            inter.append(start)
    return ictal, inter


def windows(segment: np.ndarray, window_sec: float = 1.0, step_sec: float = 0.5, rate: int = TARGET_RATE) -> np.ndarray:
    """``[N, seg]`` -> ``[T, N, n_w]`` overlapping windows."""
    n_w, step = int(window_sec * rate), int(step_sec * rate)
    starts = range(0, segment.shape[-1] - n_w + 1, step)
    return np.stack([segment[:, s : s + n_w] for s in starts]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def list_recordings(dataset: str, root: str) -> List[Tuple[str, str, List[Tuple[float, float]]]]:
    """(patient, edf path, seizure intervals) for every recording."""
    recordings = []
    if dataset == "chbmit":
        for summary in sorted(glob.glob(os.path.join(root, "chb*", "chb*-summary.txt"))):
            patient = os.path.basename(os.path.dirname(summary))
            seizures = parse_chbmit_summary(summary)
            for edf in sorted(glob.glob(os.path.join(os.path.dirname(summary), "*.edf"))):
                recordings.append((patient, edf, seizures.get(os.path.basename(edf), [])))
    else:
        for listing in sorted(glob.glob(os.path.join(root, "PN*", "Seizures-list-PN*.txt"))):
            patient = os.path.basename(os.path.dirname(listing))
            seizures = parse_siena_list(listing)
            for edf in sorted(glob.glob(os.path.join(os.path.dirname(listing), "*.edf"))):
                recordings.append((patient, edf, seizures.get(os.path.basename(edf), [])))
    return recordings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["chbmit", "siena"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--segment-sec", type=float, default=4.0)
    parser.add_argument("--ictal-step-sec", type=float, default=1.0)
    parser.add_argument("--max-interictal-per-file", type=int, default=200, help="random subset of interictal segments kept per recording")
    parser.add_argument("--no-ica", action="store_true", help="skip FastICA artifact removal")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import mne

    channels = CHBMIT_CHANNELS if args.dataset == "chbmit" else SIENA_CHANNELS
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.output, "recordings"), exist_ok=True)
    index_rows, skipped = [], []
    recordings = list_recordings(args.dataset, args.input)
    for k, (patient, edf, seizures) in enumerate(recordings, start=1):
        name = os.path.splitext(os.path.basename(edf))[0]
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
        signals = bipolar_signals(raw, channels)
        if signals is None:
            skipped.append((name, "incomplete electrode configuration"))
            continue
        x = preprocess_recording(signals, int(round(raw.info["sfreq"])), use_ica=not args.no_ica)
        ictal_starts, inter_starts = segment_starts(x.shape[-1], seizures, args.segment_sec, args.ictal_step_sec)
        if len(inter_starts) > args.max_interictal_per_file:
            inter_starts = sorted(rng.choice(inter_starts, args.max_interictal_per_file, replace=False).tolist())
        seg = int(args.segment_sec * TARGET_RATE)
        segments = [windows(x[:, s : s + seg]) for s in ictal_starts + inter_starts]
        if not segments:
            skipped.append((name, "no segments"))
            continue
        labels = np.array([1] * len(ictal_starts) + [0] * len(inter_starts), dtype=np.int64)
        rel_path = os.path.join("recordings", f"{name}.npz")
        np.savez_compressed(os.path.join(args.output, rel_path), x=np.stack(segments), y=labels)
        index_rows.append({"file": name, "path": rel_path, "patient": patient, "num_segments": len(labels)})
        print(f"[{k}/{len(recordings)}] {name}: {len(ictal_starts)} ictal, {len(inter_starts)} interictal")

    with open(os.path.join(args.output, "index.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "path", "patient", "num_segments"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Done: {len(index_rows)} recordings cached under {args.output}")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")


if __name__ == "__main__":
    main()
