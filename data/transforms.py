"""Data augmentation for facial graph sequences.

Operates on the preprocessed sample (optical-flow magnitude patches and
aligned landmark coordinates) so that no optical flow has to be recomputed:

* random scaling in [0.9, 1.1]  : flow magnitudes and landmark coordinates
                                   scale with the face
* horizontal flip (p = 0.5)      : patches are mirrored and nodes are
                                   permuted with their left/right counterparts
* random rotation in [-10, 10]  : every patch is rotated about its centre and
                                   the landmarks about the image centre
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from scipy.ndimage import rotate as ndimage_rotate

from models.graphs import face_mirror_index


class FaceAugmentation:
    """Random scale, horizontal flip, and rotation of a facial sample."""

    def __init__(
        self,
        scale_range: Sequence[float] = (0.9, 1.1),
        flip_prob: float = 0.5,
        rotation_degrees: float = 10.0,
        image_size: int = 224,
        seed: Optional[int] = None,
    ) -> None:
        self.scale_range = tuple(scale_range)
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees
        self.image_size = image_size
        self.mirror = face_mirror_index()
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        patches = sample["x"].numpy().copy()                     # [T, N, Sz, Sz]
        landmarks = sample["landmarks"].numpy().copy() if "landmarks" in sample else None

        scale = self.rng.uniform(*self.scale_range)
        patches *= scale
        if landmarks is not None:
            centre = self.image_size / 2.0
            landmarks = (landmarks - centre) * scale + centre

        if self.rng.random() < self.flip_prob:
            patches = patches[:, self.mirror][..., ::-1]
            if landmarks is not None:
                landmarks = landmarks[:, self.mirror]
                landmarks[..., 0] = self.image_size - landmarks[..., 0]

        angle = self.rng.uniform(-self.rotation_degrees, self.rotation_degrees)
        if abs(angle) > 1e-3:
            length, nodes, size, _ = patches.shape
            flat = patches.reshape(length * nodes, size, size)
            flat = ndimage_rotate(flat, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
            patches = flat.reshape(length, nodes, size, size)
            if landmarks is not None:
                theta = math.radians(angle)
                rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
                centre = self.image_size / 2.0
                landmarks = (landmarks - centre) @ rot.T + centre

        out = dict(sample)
        out["x"] = torch.from_numpy(np.ascontiguousarray(patches, dtype=np.float32))
        if landmarks is not None:
            out["landmarks"] = torch.from_numpy(np.ascontiguousarray(landmarks, dtype=np.float32))
        return out
