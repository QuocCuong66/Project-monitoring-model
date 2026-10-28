"""Build temporal training windows from the NTU RGB+D 60 2D skeleton release.

Source: MMAction2's ``ntu60_2d.pkl`` (HRNet COCO-17 keypoints extracted from the NTU RGB
videos), downloaded from https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl
NTU RGB+D is licensed for non-commercial research only; models trained on it inherit that.

NTU classes are mapped onto the system's ACTION_CLASSES. Medical classes that are visible
warning signs of stroke / acute illness (staggering, clutching the head, chest pain,
nausea/vomiting) are kept as their own signals; NTU has no lying / immobile class, so those
are built from the final frames of real fall clips and from body-rotated daily poses.

Output: data/processed/ntu_windows.npz with raw pixel-space windows (N, T, 17, 3) at 15 Hz.
Normalization happens at training time through ``build_model_sequence`` so that the
augmentations (rotation, top-down squash, flips) act on raw geometry exactly as cameras do.
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.action.action_classifier import ACTION_CLASSES  # noqa: E402

CLS = {name: i for i, name in enumerate(ACTION_CLASSES)}

# NTU60 labels are 0-based (A001 -> 0)
NTU_FALL = 42            # A43 falling down
NTU_STAGGER = 41         # A42 staggering
NTU_DISTRESS = {43, 44, 47}  # A44 headache, A45 chest pain, A48 nausea / vomiting
NTU_STAND_UP = 8         # A09 stand up (from sitting)
NTU_BEND = {5, 15, 16}   # A06 pick up, A16 wear shoe, A17 take off shoe
NTU_WALK = {58, 59}      # A59 walking towards, A60 walking apart (two people)
NTU_SKIP = {7, 45, 46}   # A08 sit down (ambiguous transition), A46 back pain, A47 neck pain
NTU_MUTUAL_FIRST = 49    # A50+ are two-person interactions (only walking is used)

SRC_FPS = 30
STRIDE = 2               # 30 fps -> 15 Hz temporal rate used live
CONF = 0.2


def person_track(ann: dict, m: int) -> np.ndarray:
    kp = ann["keypoint"][m]                      # (T, 17, 2)
    sc = ann["keypoint_score"][m][..., None]     # (T, 17, 1)
    return np.concatenate([kp, sc], axis=-1).astype(np.float32)[::STRIDE]


def windows(track: np.ndarray, t: int, n_random: int, rng: np.random.Generator,
            tail_only: bool = False) -> list[np.ndarray]:
    """End-aligned window plus random windows (tail_only keeps windows in the last half)."""
    n = len(track)
    if n <= t:
        return [track]
    out = [track[n - t:]]
    lo = n // 2 - t if tail_only else 0
    lo = max(0, min(lo, n - t))
    for _ in range(n_random):
        s = int(rng.integers(lo, n - t + 1))
        out.append(track[s:s + t])
    return out


def leg_extension(track: np.ndarray) -> float:
    """Median (ankle_y - hip_y) / torso length: ~2 standing, ~1 sitting on a chair."""
    vals = []
    for kp in track:
        if kp[[5, 6, 11, 12, 15, 16], 2].min() < CONF:
            continue
        hip = kp[[11, 12], :2].mean(0)
        sh = kp[[5, 6], :2].mean(0)
        ank = kp[[15, 16], :2].mean(0)
        torso = np.linalg.norm(sh - hip)
        if torso > 1e-3:
            vals.append((ank[1] - hip[1]) / torso)
    return float(np.median(vals)) if vals else float("nan")


def rotate(win: np.ndarray, deg: float) -> np.ndarray:
    out = win.copy()
    pts = out[..., :2]
    c = pts[out[..., 2] > CONF].mean(0) if np.any(out[..., 2] > CONF) else pts.reshape(-1, 2).mean(0)
    th = np.deg2rad(deg)
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    out[..., :2] = (pts - c) @ r.T + c
    return out


def still(frame: np.ndarray, t: int, rng: np.random.Generator, jitter_px: float) -> np.ndarray:
    win = np.repeat(frame[None], t, axis=0).copy()
    win[..., :2] += rng.normal(0, jitter_px, size=win[..., :2].shape).astype(np.float32)
    return win


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=str(ROOT / "data/raw/ntu/ntu60_2d.pkl"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/ntu_windows.npz"))
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t = args.window

    data = pickle.load(open(args.pkl, "rb"))
    train_ids = set(data["split"]["xsub_train"])

    xs: list[np.ndarray] = []
    ys: list[int] = []
    split: list[int] = []      # 0 = train, 1 = val (NTU cross-subject protocol)
    src: list[int] = []        # NTU label (or -1 synthetic)

    def add(ws: list[np.ndarray], label: int, is_val: int, ntu: int) -> None:
        for w in ws:
            if w.shape[0] < t:
                w = np.concatenate([np.repeat(w[:1], t - w.shape[0], 0), w], 0)
            xs.append(w.astype(np.float16))
            ys.append(label)
            split.append(is_val)
            src.append(ntu)

    fall_tails: list[tuple[np.ndarray, int]] = []
    daily: list[tuple[np.ndarray, int]] = []

    for ann in data["annotations"]:
        lab = int(ann["label"])
        is_val = 0 if ann["frame_dir"] in train_ids else 1
        if lab in NTU_SKIP:
            continue
        if lab in NTU_WALK:
            for m in range(ann["keypoint"].shape[0]):
                add(windows(person_track(ann, m), t, 1, rng), CLS["walking"], is_val, lab)
            continue
        if lab >= NTU_MUTUAL_FIRST:
            continue
        tr = person_track(ann, 0)
        if lab == NTU_FALL:
            add(windows(tr, t, 3, rng, tail_only=True), CLS["falling"], is_val, lab)
            fall_tails.append((tr[-6:], is_val))
        elif lab == NTU_STAGGER:
            add(windows(tr, t, 3, rng), CLS["stumbling"], is_val, lab)
        elif lab in NTU_DISTRESS:
            add(windows(tr, t, 1, rng), CLS["abnormal_movement"], is_val, lab)
        elif lab == NTU_STAND_UP:
            add(windows(tr, t, 2, rng), CLS["getting_up"], is_val, lab)
        elif lab in NTU_BEND:
            add(windows(tr, t, 1, rng), CLS["bending"], is_val, lab)
        else:
            ext = leg_extension(tr)
            if np.isnan(ext):
                continue
            posture = "standing" if ext >= 1.4 else ("sitting" if ext <= 1.15 else None)
            if posture is None:
                continue
            w = windows(tr, t, 0, rng)[0]
            daily.append((w, is_val))
            add([w], CLS[posture], is_val, lab)

    # Lying / immobile: real post-fall floor poses + daily poses rotated to horizontal.
    for tail, is_val in fall_tails:
        add([still(tail[-1], t, rng, 0.6)], CLS["immobile"], is_val, -1)
        moving = np.concatenate([tail, tail[::-1]] * 3, 0)[:t].copy()
        moving[..., :2] += rng.normal(0, 2.5, size=moving[..., :2].shape).astype(np.float32)
        add([moving], CLS["lying"], is_val, -1)
    picks = rng.choice(len(daily), size=min(len(daily), 1500), replace=False)
    for i in picks:
        w, is_val = daily[i]
        deg = float(rng.choice([-1, 1]) * rng.uniform(70, 110))
        lying = rotate(w, deg)
        if rng.random() < 0.5:
            add([still(lying[int(rng.integers(0, len(lying)))], t, rng, 0.6)], CLS["immobile"], is_val, -1)
        else:
            add([lying], CLS["lying"], is_val, -1)

    x = np.stack(xs)
    y = np.array(ys, dtype=np.int64)
    sp = np.array(split, dtype=np.int8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, y=y, split=sp, src=np.array(src, dtype=np.int16))
    for name, s in (("train", 0), ("val", 1)):
        c = Counter(y[sp == s].tolist())
        print(name, {ACTION_CLASSES[k]: c[k] for k in sorted(c)})
    print("saved", args.out, x.shape)


if __name__ == "__main__":
    main()
