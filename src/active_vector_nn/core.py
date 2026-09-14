from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepResult:
    """Observable result of one active-vector update."""

    state: np.ndarray
    receiver_state: np.ndarray
    innovation: np.ndarray
    emitted: bool
    max_error: float


class ActiveVectorCell:
    """Persistent vector state with thresholded innovation communication."""

    def __init__(self, alpha: float | np.ndarray, threshold: float, dim: int):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if threshold < 0:
            raise ValueError("threshold must be nonnegative")

        self.dim = int(dim)
        self.threshold = float(threshold)
        self.alpha = self._coerce_alpha(alpha)
        self.state = np.zeros(self.dim, dtype=float)
        self.receiver_state = np.zeros(self.dim, dtype=float)

    def _coerce_alpha(self, alpha: float | np.ndarray) -> np.ndarray:
        values = np.asarray(alpha, dtype=float)
        if values.ndim == 0:
            values = np.full(self.dim, float(values), dtype=float)
        elif values.shape != (self.dim,):
            raise ValueError(f"alpha must be scalar or shape ({self.dim},)")
        else:
            values = values.astype(float, copy=True)

        if np.any(~np.isfinite(values)) or np.any(values <= 0.0) or np.any(values >= 1.0):
            raise ValueError("alpha values must be finite and strictly between 0 and 1")
        return values

    def reset(self) -> None:
        self.state.fill(0.0)
        self.receiver_state.fill(0.0)

    def step(self, u: np.ndarray) -> StepResult:
        input_vector = np.asarray(u, dtype=float)
        if input_vector.shape != (self.dim,):
            raise ValueError(f"input must have shape ({self.dim},)")
        if np.any(~np.isfinite(input_vector)):
            raise ValueError("input must be finite")

        self.state = self.alpha * self.state + (1.0 - self.alpha) * input_vector

        receiver_prediction = self.alpha * self.receiver_state
        innovation = self.state - receiver_prediction
        emitted = bool(np.max(np.abs(innovation)) > self.threshold)

        if emitted:
            self.receiver_state = receiver_prediction + innovation
        else:
            self.receiver_state = receiver_prediction

        max_error = float(np.max(np.abs(self.state - self.receiver_state)))
        return StepResult(
            state=self.state.copy(),
            receiver_state=self.receiver_state.copy(),
            innovation=innovation.copy(),
            emitted=emitted,
            max_error=max_error,
        )
