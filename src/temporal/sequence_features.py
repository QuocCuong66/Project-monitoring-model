"""Window-level skeleton normalization shared by training and live inference.

Per-frame root-centering (``normalize_keypoints``) erases whole-body motion: a person
dropping to the floor looks the same as one bending in place. The temporal models
therefore consume a window normalized once with a single origin and scale, so the
downward trajectory of a fall, a stagger's sway or a collapse stays visible.

Both ``training/train_action.py`` and ``DecoupledPipeline`` call ``build_model_sequence``;
keep them on this one function so the model never sees a different distribution live.
"""

from typing import Optional

import numpy as np

CONF_THRESH = 0.2
COORD_CLIP = 8.0
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def _midpoint(kp: np.ndarray, a: int, b: int) -> Optional[np.ndarray]:
    ca, cb = kp[a, 2] > CONF_THRESH, kp[b, 2] > CONF_THRESH
    if ca and cb:
        return (kp[a, :2] + kp[b, :2]) / 2.0
    if ca:
        return kp[a, :2]
    if cb:
        return kp[b, :2]
    return None


def _window_scale(raw: np.ndarray) -> float:
    """Median torso length over the window (falls back to a third of the body extent)."""
    lengths: list[float] = []
    extents: list[float] = []
    for kp in raw:
        hip = _midpoint(kp, L_HIP, R_HIP)
        sh = _midpoint(kp, L_SHOULDER, R_SHOULDER)
        if hip is not None and sh is not None:
            lengths.append(float(np.linalg.norm(sh - hip)))
        valid = kp[:, 2] > CONF_THRESH
        if valid.sum() >= 2:
            pts = kp[valid, :2]
            extents.append(float(np.max(np.ptp(pts, axis=0))))
    if lengths:
        scale = float(np.median(lengths))
        if scale > 1e-3:
            return scale
    if extents:
        return max(1e-3, float(np.median(extents)) / 3.0)
    return 1.0


def _window_origin(raw: np.ndarray) -> np.ndarray:
    """Mid-hip of the earliest frame that has one (else the mean of all visible joints)."""
    for kp in raw:
        hip = _midpoint(kp, L_HIP, R_HIP)
        if hip is not None:
            return hip
    valid = raw[:, :, 2] > CONF_THRESH
    if np.any(valid):
        return raw[:, :, :2][valid].mean(axis=0)
    return np.zeros(2, dtype=np.float32)


def build_model_sequence(raw_keypoints: np.ndarray, window_size: int) -> np.ndarray:
    """Turn raw pixel keypoints of one track into the temporal-model input.

    Args:
        raw_keypoints: (N, 17, 3) pixel-space keypoints (x, y, conf), oldest first,
            sampled at the temporal rate (15 Hz).
        window_size: Model window length T.

    Returns:
        (window_size, 17, 3) float32: x, y relative to the window origin in torso
        lengths (+y down), confidence kept; joints below CONF_THRESH are zeroed.
        Shorter histories are front-padded with their first frame.
    """
    raw = np.asarray(raw_keypoints, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] == 0:
        return np.zeros((window_size, 17, 3), dtype=np.float32)
    if raw.shape[0] > window_size:
        raw = raw[-window_size:]
    elif raw.shape[0] < window_size:
        pad = np.repeat(raw[:1], window_size - raw.shape[0], axis=0)
        raw = np.concatenate([pad, raw], axis=0)

    origin = _window_origin(raw)
    scale = _window_scale(raw)

    seq = np.empty_like(raw)
    seq[:, :, :2] = np.clip((raw[:, :, :2] - origin) / scale, -COORD_CLIP, COORD_CLIP)
    seq[:, :, 2] = raw[:, :, 2]
    seq[raw[:, :, 2] <= CONF_THRESH] = 0.0
    return seq
