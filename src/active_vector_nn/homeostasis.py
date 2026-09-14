from __future__ import annotations

import math

import numpy as np


class LocalRMSHomeostat:
    """Slow local persistence adaptation driven only by state RMS error."""

    def __init__(
        self,
        target_rms: float,
        ema_rate: float,
        adaptation_rate: float,
        min_alpha: float,
        max_alpha: float,
    ):
        if target_rms <= 0.0 or not math.isfinite(target_rms):
            raise ValueError("target_rms must be positive and finite")
        if not 0.0 < ema_rate <= 1.0:
            raise ValueError("ema_rate must be in (0, 1]")
        if adaptation_rate < 0.0 or not math.isfinite(adaptation_rate):
            raise ValueError("adaptation_rate must be nonnegative and finite")
        if not 0.0 < min_alpha < max_alpha < 1.0:
            raise ValueError("alpha bounds must satisfy 0 < min_alpha < max_alpha < 1")

        self.target_rms = float(target_rms)
        self.ema_rate = float(ema_rate)
        self.adaptation_rate = float(adaptation_rate)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self._power_ema = self.target_rms * self.target_rms
        self._min_logit = self._logit(self.min_alpha)
        self._max_logit = self._logit(self.max_alpha)

    @staticmethod
    def _logit(value: float | np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        return np.log(array) - np.log1p(-array)

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-value))

    @property
    def local_rms(self) -> float:
        return math.sqrt(max(self._power_ema, 0.0))

    def reset(self) -> None:
        self._power_ema = self.target_rms * self.target_rms

    def update(self, state: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, float]:
        state_vector = np.asarray(state, dtype=float)
        alpha_vector = np.asarray(alpha, dtype=float)
        if state_vector.ndim != 1 or alpha_vector.shape != state_vector.shape:
            raise ValueError("state and alpha must be one-dimensional arrays with matching shapes")
        if np.any(~np.isfinite(state_vector)):
            raise ValueError("state must be finite")
        if np.any(~np.isfinite(alpha_vector)) or np.any(alpha_vector <= 0.0) or np.any(alpha_vector >= 1.0):
            raise ValueError("alpha values must be finite and strictly between 0 and 1")

        instant_power = float(np.mean(state_vector * state_vector))
        self._power_ema = (
            (1.0 - self.ema_rate) * self._power_ema
            + self.ema_rate * instant_power
        )
        rms = self.local_rms

        error = rms / self.target_rms - 1.0
        logits = self._logit(alpha_vector) + self.adaptation_rate * error
        logits = np.clip(logits, self._min_logit, self._max_logit)
        return self._sigmoid(logits), rms
