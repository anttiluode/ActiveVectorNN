from __future__ import annotations

import json
import math

import numpy as np

from active_vector_nn.core import ActiveVectorCell
from active_vector_nn.homeostasis import LocalRMSHomeostat
from active_vector_nn.task import (
    NoisyLatentARStream,
    apply_readout,
    fit_ridge_readout,
)


REGIMES = ((0.0, 1500), (0.3, 1500), (0.6, 1500), (0.85, 1500))
EVENT_THRESHOLDS = (0.8, 1.0, 1.2, 1.5)
DIM = 8
NOISE_STD = 1.5
INITIAL_ALPHA = 0.6
RIDGE = 1e-6


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    error = np.asarray(prediction) - np.asarray(target)
    return float(np.sqrt(np.mean(error * error)))


def _homeostatic_target_rms() -> float:
    observation_variance = 1.0 + NOISE_STD * NOISE_STD
    white_filter_variance = (
        (1.0 - INITIAL_ALPHA) / (1.0 + INITIAL_ALPHA) * observation_variance
    )
    return math.sqrt(white_filter_variance)


def _collect_stream(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stream = NoisyLatentARStream(
        regimes=REGIMES,
        dim=DIM,
        noise_std=NOISE_STD,
        seed=seed,
    )
    observations: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    rhos: list[float] = []
    for _ in range(stream.total_steps):
        observation, latent, rho = stream.step()
        observations.append(observation)
        latents.append(latent)
        rhos.append(rho)
    return np.asarray(observations), np.asarray(latents), np.asarray(rhos)


def _make_homeostat() -> LocalRMSHomeostat:
    return LocalRMSHomeostat(
        target_rms=_homeostatic_target_rms(),
        ema_rate=0.01,
        adaptation_rate=0.002,
        min_alpha=0.05,
        max_alpha=0.995,
    )


def _collect_sender_states(
    observations: np.ndarray, adaptive: bool
) -> tuple[np.ndarray, float]:
    cell = ActiveVectorCell(alpha=INITIAL_ALPHA, threshold=0.0, dim=DIM)
    homeostat = _make_homeostat() if adaptive else None
    states: list[np.ndarray] = []
    for observation in observations:
        result = cell.step(observation)
        states.append(result.state)
        if homeostat is not None:
            new_alpha, _ = homeostat.update(result.state, cell.alpha)
            cell.set_alpha(new_alpha)
    return np.asarray(states), float(np.mean(cell.alpha))


def _run_event_channel(
    observations: np.ndarray,
    latents: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    adaptive: bool,
) -> dict[str, float]:
    cell = ActiveVectorCell(alpha=INITIAL_ALPHA, threshold=threshold, dim=DIM)
    homeostat = _make_homeostat() if adaptive else None
    sender_states: list[np.ndarray] = []
    receiver_states: list[np.ndarray] = []
    emitted = 0
    max_state_error = 0.0

    for observation in observations:
        result = cell.step(observation)
        sender_states.append(result.state)
        receiver_states.append(result.receiver_state)
        emitted += int(result.emitted)
        max_state_error = max(max_state_error, result.max_error)
        if homeostat is not None:
            new_alpha, _ = homeostat.update(result.state, cell.alpha)
            cell.set_alpha(new_alpha)

    sender = np.asarray(sender_states)
    receiver = np.asarray(receiver_states)
    sender_prediction = apply_readout(sender, weights)
    receiver_prediction = apply_readout(receiver, weights)

    return {
        "threshold": float(threshold),
        "event_fraction": float(emitted / len(observations)),
        "sender_task_rmse": _rmse(sender_prediction, latents),
        "receiver_task_rmse": _rmse(receiver_prediction, latents),
        "receiver_state_rmse": _rmse(receiver, sender),
        "max_state_error": float(max_state_error),
        "final_alpha": float(np.mean(cell.alpha)),
    }


def run_experiment(seed: int = 0) -> dict[str, object]:
    train_seed = int(seed)
    test_seed = int(seed + 1)
    train_observations, train_latents, _ = _collect_stream(train_seed)
    test_observations, test_latents, _ = _collect_stream(test_seed)

    fixed_train_states, _ = _collect_sender_states(train_observations, adaptive=False)
    adaptive_train_states, _ = _collect_sender_states(train_observations, adaptive=True)

    raw_weights = fit_ridge_readout(train_observations, train_latents, ridge=RIDGE)
    fixed_weights = fit_ridge_readout(fixed_train_states, train_latents, ridge=RIDGE)
    adaptive_weights = fit_ridge_readout(adaptive_train_states, train_latents, ridge=RIDGE)

    fixed_test_states, fixed_final_alpha = _collect_sender_states(
        test_observations, adaptive=False
    )
    adaptive_test_states, adaptive_final_alpha = _collect_sender_states(
        test_observations, adaptive=True
    )

    raw_task_rmse = _rmse(apply_readout(test_observations, raw_weights), test_latents)
    fixed_sender_task_rmse = _rmse(
        apply_readout(fixed_test_states, fixed_weights), test_latents
    )
    adaptive_sender_task_rmse = _rmse(
        apply_readout(adaptive_test_states, adaptive_weights), test_latents
    )

    fixed_curve = [
        _run_event_channel(
            test_observations,
            test_latents,
            fixed_weights,
            threshold=threshold,
            adaptive=False,
        )
        for threshold in EVENT_THRESHOLDS
    ]
    adaptive_curve = [
        _run_event_channel(
            test_observations,
            test_latents,
            adaptive_weights,
            threshold=threshold,
            adaptive=True,
        )
        for threshold in EVENT_THRESHOLDS
    ]

    max_event_fraction = 0.80
    max_relative_task_rmse = 1.05
    passing_thresholds = [
        row["threshold"]
        for row in fixed_curve
        if row["event_fraction"] <= max_event_fraction
        and row["receiver_task_rmse"]
        <= max_relative_task_rmse * fixed_sender_task_rmse
    ]

    return {
        "seed": int(seed),
        "config": {
            "train_seed": train_seed,
            "test_seed": test_seed,
            "dim": DIM,
            "noise_std": NOISE_STD,
            "initial_alpha": INITIAL_ALPHA,
            "homeostatic_target_rms": _homeostatic_target_rms(),
            "ridge": RIDGE,
            "regimes": [[float(rho), int(steps)] for rho, steps in REGIMES],
            "event_thresholds": [float(value) for value in EVENT_THRESHOLDS],
        },
        "raw_task_rmse": raw_task_rmse,
        "fixed_sender_task_rmse": fixed_sender_task_rmse,
        "adaptive_sender_task_rmse": adaptive_sender_task_rmse,
        "fixed_final_alpha": fixed_final_alpha,
        "adaptive_final_alpha": adaptive_final_alpha,
        "fixed_event_curve": fixed_curve,
        "adaptive_event_curve": adaptive_curve,
        "fixed_task_value_gate": {
            "max_event_fraction": max_event_fraction,
            "max_relative_task_rmse": max_relative_task_rmse,
            "passed": bool(passing_thresholds),
            "passing_thresholds": passing_thresholds,
        },
    }


def main() -> None:
    print(json.dumps(run_experiment(seed=0), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
