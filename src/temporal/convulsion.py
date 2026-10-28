"""Convulsion (seizure-like jerking) detector on normalized 2D skeletons.

A convulsion shows as rhythmic jerking of several body parts at once (clonic jerks, roughly 2-6 Hz) that
keeps going for seconds. Normal movements look different:

* lying down / turning over - one slow whole-body movement that settles again (below the jerk band)
* phone use                 - one wrist moving near the face (a single body part)
* walking / exercising      - limbs swing at ~1-1.5 Hz (below the jerk band)
* clapping / waving         - arms only, the legs stay quiet
* keypoint jitter           - small, irregular, not in the same rhythm across body parts

Decision support only: rhythmic jerking is a visible warning sign, never a diagnosis of epilepsy.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

CONVULSION = "convulsion"

# Body parts as (joints) in COCO-17 order; each joint is tracked relative to the hips (normalized skeleton)
BODY_PARTS: dict[str, tuple[int, ...]] = {
    "head": (0,),
    "left_arm": (7, 9),
    "right_arm": (8, 10),
    "left_leg": (13, 15),
    "right_leg": (14, 16),
}
ARMS = ("left_arm", "right_arm")
LEGS = ("left_leg", "right_leg")


@dataclass
class ConvulsionConfig:
    """Thresholds for the convulsion detector (configs/thresholds.yaml -> convulsion)."""
    enabled: bool = True
    window_seconds: float = 4.0         # spectral window
    min_hz: float = 1.8                 # walking / exercise swing is slower than this
    max_hz: float = 6.0
    min_amplitude: float = 0.05         # in-band RMS (torso lengths): keypoint jitter is smaller
    min_band_share: float = 0.65        # in-band power / all motion power (slow trend removed)
    min_peak_share: float = 0.35        # power at the dominant frequency (+-1 bin) / in-band power
    min_parts: int = 3                  # body parts jerking together (an arm AND a leg among them)
    max_frequency_spread: float = 1.0   # Hz: the parts jerk in the same rhythm
    min_seconds: float = 5.0            # sustained rhythmic jerking this long = convulsion
    gap_seconds: float = 1.0            # brief misses (occlusion, a pause between jerks) keep the run going
    min_positive_share: float = 0.8     # ...but most windows of the run must show the rhythm (heavy keypoint
                                        # jitter gives scattered hits, ~60% at most)
    onset_min_seconds: float = 1.5      # jerking this long right before going down = convulsion onset
    onset_lookback_seconds: float = 5.0
    min_sample_hz: float = 8.0          # below this the jerk rhythm cannot be measured
    min_visible_share: float = 0.7      # a joint must be visible this share of the window

    @classmethod
    def from_dict(cls, cfg: Optional[dict[str, Any]]) -> "ConvulsionConfig":
        cfg = dict(cfg or {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class Rhythm:
    """Body parts jerking in a shared rhythm over one window."""
    parts: list[str] = field(default_factory=list)
    frequency: float = 0.0
    positive: bool = False


def analyze_window(times: np.ndarray, keypoints: np.ndarray, cfg: ConvulsionConfig) -> Rhythm:
    """Find body parts jerking rhythmically in the jerk band.

    Args:
        times: (N,) sample times in seconds, increasing.
        keypoints: (N, 17, 3) normalized keypoints (hip centre = origin, torso length = 1).
        cfg: Detector thresholds.
    """
    n = len(times)
    if n < 8:
        return Rhythm()
    duration = float(times[-1] - times[0])
    if duration < 0.75 * cfg.window_seconds:
        return Rhythm()
    fs = (n - 1) / duration
    if fs < cfg.min_sample_hz:
        return Rhythm()

    grid = np.linspace(times[0], times[-1], n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= cfg.min_hz) & (freqs <= min(cfg.max_hz, 0.45 * fs))
    if int(band.sum()) < 2:
        return Rhythm()
    taper = np.hanning(n)

    part_freq: dict[str, float] = {}
    for part, joints in BODY_PARTS.items():
        best: Optional[tuple[float, float]] = None                  # (in-band RMS, frequency)
        for j in joints:
            if np.mean(keypoints[:, j, 2] >= 0.3) < cfg.min_visible_share:
                continue
            for axis in (0, 1):
                sig = np.interp(grid, times, keypoints[:, j, axis])
                sig = sig - np.polyval(np.polyfit(grid, sig, 1), grid)     # remove the slow drift
                spec = np.abs(np.fft.rfft(sig * taper)) ** 2
                total = float(spec[1:].sum())
                band_power = float(spec[band].sum())
                if total <= 1e-12 or band_power / total < cfg.min_band_share:
                    continue
                k = int(np.argmax(np.where(band, spec, 0.0)))
                if float(spec[max(k - 1, 0):k + 2].sum()) / band_power < cfg.min_peak_share:
                    continue                                        # no clear rhythm: irregular jitter
                f_full = np.fft.rfft(sig)
                f_full[~band] = 0.0
                rms = float(np.sqrt(np.mean(np.fft.irfft(f_full, n) ** 2)))
                if rms >= cfg.min_amplitude and (best is None or rms > best[0]):
                    best = (rms, float(freqs[k]))
        if best is not None:
            part_freq[part] = best[1]

    if len(part_freq) < cfg.min_parts:
        return Rhythm(list(part_freq))
    centre = float(np.median(list(part_freq.values())))
    together = [p for p, f in part_freq.items() if abs(f - centre) <= cfg.max_frequency_spread]
    positive = (len(together) >= cfg.min_parts and any(p in ARMS for p in together)
                and any(p in LEGS for p in together))
    return Rhythm(together, centre, positive)


@dataclass
class ConvulsionTracker:
    """Per-person run of rhythmic-jerking windows (brief gaps tolerated, most windows must be positive)."""
    run_start: Optional[float] = None
    last_positive: Optional[float] = None
    frequency: float = 0.0
    run_ticks: int = 0                  # windows analysed from the run start up to the last positive one
    run_hits: int = 0                   # ... of which positive
    _ticks: int = 0

    def update(self, t: float, rhythm: Rhythm, cfg: ConvulsionConfig) -> None:
        if rhythm.positive and (self.last_positive is None or t - self.last_positive > cfg.gap_seconds):
            self.run_start, self.run_ticks, self.run_hits, self._ticks = t, 0, 0, 0
        if self.last_positive is not None and t - self.last_positive > cfg.gap_seconds and not rhythm.positive:
            return                                          # run over: keep its numbers for jerked_recently
        self._ticks += 1
        if rhythm.positive:
            self.last_positive = t
            self.frequency = rhythm.frequency
            self.run_hits += 1
            self.run_ticks = self._ticks

    def run_seconds(self) -> float:
        if self.run_start is None or self.last_positive is None:
            return 0.0
        return self.last_positive - self.run_start

    def _steady(self, cfg: ConvulsionConfig) -> bool:
        return self.run_ticks > 0 and self.run_hits / self.run_ticks >= cfg.min_positive_share

    def convulsing(self, t: float, cfg: ConvulsionConfig) -> bool:
        """Rhythmic jerking sustained for min_seconds and still going on."""
        return (self.last_positive is not None and t - self.last_positive <= cfg.gap_seconds
                and self.run_seconds() >= cfg.min_seconds and self._steady(cfg))

    def jerked_recently(self, t: float, cfg: ConvulsionConfig) -> bool:
        """Rhythmic jerking (>= onset_min_seconds) in the last few seconds: how a convulsion begins."""
        return (self.last_positive is not None and t - self.last_positive <= cfg.onset_lookback_seconds
                and self.run_seconds() >= cfg.onset_min_seconds and self._steady(cfg))
