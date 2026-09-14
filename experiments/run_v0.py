from __future__ import annotations

import json
import math

import numpy as np

from active_vector_nn.core import ActiveVectorCell
from active_vector_nn.homeostasis import LocalRMSHomeostat
from active_vector_nn.streams import AR1Stream


REGIMES = ((0.0, 1000), (0.2, 1000), (0.6, 1000), (0.4, 1000), (0.0, 1000))
EVENT_THRESHOLDS = (0.4, 0.6, 0.8, 1.0)


def _piecewise_ar1(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = np.zeros(dim, dtype=float)
    rows: list[np.ndarray] = []
    for rho, steps in REGIMES:
        innovation_scale = math.sqrt(1.0 - rho * rho)
        for _ in range(steps):
            state = rho * state + innovation_scale * rng.normal(size=dim)
            rows.append(state.copy())
    return np.asarray(rows)


def _run_channel(inputs: np.ndarray, threshold: float, adaptive: bool) -> dict[str, float]:
    dim = int(inputs.shape[1])
    cell = ActiveVectorCell(alpha=0.6, threshold=threshold, dim=dim)
    homeostat = None
    if adaptive:
        homeostat = LocalRMSHomeostat(
            target_rms=0.5,
            ema_rate=0.01,
            adaptation_rate=0.002,
            min_alpha=0.05,
            max_alpha=0.995,
        )

    emitted = 0
    squared_error_sum = 0.0
    scalar_error_count = 0
    max_error = 0.0
    final_local_rms = float(np.sqrt(np.mean(cell.state * cell.state)))

    for vector in inputs:
        result = cell.step(vector)
        emitted += int(result.emitted)
        diff = result.state - result.receiver_state
        squared_error_sum += float(np.sum(diff * diff))
        scalar_error_count += diff.size
        max_error = max(max_error, result.max_error)

        if homeostat is not None:
            updated_alpha, final_local_rms = homeostat.update(result.state, cell.alpha)
            cell.set_alpha(updated_alpha)
        else:
            final_local_rms = float(np.sqrt(np.mean(result.state * result.state)))

    return {
        "threshold": float(threshold),
        "event_fraction": float(emitted / len(inputs)),
        "rmse": float(math.sqrt(squared_error_sum / scalar_error_count)),
        "max_error": float(max_error),
        "final_alpha": float(np.mean(cell.alpha)),
        "final_local_rms": float(final_local_rms),
    }


def _run_homeostatic_differentiation(seed: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for rho in (0.0, 0.2, 0.4, 0.6):
        dim = 64
        stream = AR1Stream(rho=rho, dim=dim, seed=seed + 17)
        cell = ActiveVectorCell(alpha=0.6, threshold=10.0, dim=dim)
        homeostat = LocalRMSHomeostat(
            target_rms=0.5,
            ema_rate=0.01,
            adaptation_rate=0.002,
            min_alpha=0.05,
            max_alpha=0.995,
        )

        final_rms = 0.5
        for _ in range(12000):
            result = cell.step(stream.step())
            updated_alpha, final_rms = homeostat.update(result.state, cell.alpha)
            cell.set_alpha(updated_alpha)

        rows.append(
            {
                "rho": float(rho),
                "final_alpha": float(np.mean(cell.alpha)),
                "final_rms": float(final_rms),
            }
        )
    return rows


def run_experiment(seed: int = 0) -> dict[str, object]:
    inputs = _piecewise_ar1(seed=seed, dim=8)
    dense = _run_channel(inputs, threshold=0.0, adaptive=False)
    fixed_curve = [
        _run_channel(inputs, threshold=threshold, adaptive=False)
        for threshold in EVENT_THRESHOLDS
    ]
    adaptive_curve = [
        _run_channel(inputs, threshold=threshold, adaptive=True)
        for threshold in EVENT_THRESHOLDS
    ]

    return {
        "seed": int(seed),
        "config": {
            "dim": 8,
            "initial_alpha": 0.6,
            "homeostatic_target_rms": 0.5,
            "regimes": [[float(rho), int(steps)] for rho, steps in REGIMES],
            "event_thresholds": [float(value) for value in EVENT_THRESHOLDS],
        },
        "dense": dense,
        "fixed_curve": fixed_curve,
        "adaptive_curve": adaptive_curve,
        "homeostatic_differentiation": _run_homeostatic_differentiation(seed),
    }


def main() -> None:
    print(json.dumps(run_experiment(seed=0), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
