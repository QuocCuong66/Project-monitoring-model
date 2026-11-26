"""One Euro Filter for smoothing skeletal keypoints and preventing jitter without phase lag.

Reference:
    Casiez, G., Roussel, N., & Vogel, D. (2012).
    1 € filter: a simple speed-based low-pass filter for noisy input in interactive systems.
    Proceedings of the SIGCHI Conference on Human Factors in Computing Systems.
"""

import math
import time
from typing import Optional, Tuple
import numpy as np


class LowPassFilter:
    """Standard first-order exponential low-pass filter."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.y: Optional[float] = None
        self.s: Optional[float] = None

    def set_alpha(self, alpha: float) -> None:
        self.alpha = np.clip(alpha, 0.0, 1.0)

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.set_alpha(alpha)
        if self.y is None:
            s = value
        else:
            s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        self.s = s
        return s

    def reset(self) -> None:
        self.y = None
        self.s = None


class OneEuroFilter1D:
    """1D implementation of the 1 Euro Filter."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0 else 0.01
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        t = timestamp if timestamp is not None else time.time()

        if self.last_time is None or t <= self.last_time:
            self.last_time = t
            return self.x_filter.filter(x, 1.0)

        rate = 1.0 / max(1e-5, (t - self.last_time))
        self.last_time = t

        # Filter derivative to determine speed
        prev_x = self.x_filter.s if self.x_filter.s is not None else x
        dx = (x - prev_x) * rate
        edx = self.dx_filter.filter(dx, self._alpha(rate, self.d_cutoff))

        # Dynamically adjust cutoff frequency: higher speed -> higher cutoff -> zero lag
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter.filter(x, self._alpha(rate, cutoff))

    def reset(self) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class KeypointOneEuroFilter:
    """Multi-joint One Euro Filter for 17 COCO skeletal keypoints.
    
    Each keypoint (x, y) has independent 1D filters that suppress low-speed jitter
    (standing, sitting) while reacting instantly without lag during fast falls.
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        self.num_keypoints = num_keypoints
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.filters_x = [
            OneEuroFilter1D(min_cutoff, beta, d_cutoff) for _ in range(num_keypoints)
        ]
        self.filters_y = [
            OneEuroFilter1D(min_cutoff, beta, d_cutoff) for _ in range(num_keypoints)
        ]

    def filter(self, keypoints: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """Filter a (17, 3) keypoints array [x, y, conf].
        
        Args:
            keypoints: Array of shape (17, 3) where columns are (x, y, conf).
            timestamp: Observation timestamp in seconds.
            
        Returns:
            Smoothed keypoints array of shape (17, 3) preserving confidences.
        """
        if keypoints is None or len(keypoints) == 0:
            return keypoints

        out = np.copy(keypoints)
        for i in range(min(self.num_keypoints, len(keypoints))):
            conf = keypoints[i, 2]
            # Only filter keypoints with reasonable detection confidence
            if conf > 0.15:
                out[i, 0] = self.filters_x[i].filter(float(keypoints[i, 0]), timestamp)
                out[i, 1] = self.filters_y[i].filter(float(keypoints[i, 1]), timestamp)
            else:
                out[i, 0] = keypoints[i, 0]
                out[i, 1] = keypoints[i, 1]
        return out

    def reset(self) -> None:
        """Reset internal filter states."""
        for fx, fy in zip(self.filters_x, self.filters_y):
            fx.reset()
            fy.reset()
