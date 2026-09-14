import numpy as np

from active_vector_nn.core import ActiveVectorCell
from active_vector_nn.homeostasis import LocalRMSHomeostat
from active_vector_nn.streams import AR1Stream


def test_homeostat_increases_persistence_when_local_rms_is_high():
    homeostat = LocalRMSHomeostat(
        target_rms=0.5,
        ema_rate=1.0,
        adaptation_rate=0.1,
        min_alpha=0.05,
        max_alpha=0.95,
    )
    alpha = np.array([0.6, 0.6])

    updated, rms = homeostat.update(np.array([1.0, 1.0]), alpha)

    assert rms == 1.0
    assert np.all(updated > alpha)


def test_homeostat_decreases_persistence_when_local_rms_is_low():
    homeostat = LocalRMSHomeostat(
        target_rms=0.5,
        ema_rate=1.0,
        adaptation_rate=0.1,
        min_alpha=0.05,
        max_alpha=0.95,
    )
    alpha = np.array([0.6, 0.6])

    updated, rms = homeostat.update(np.zeros(2), alpha)

    assert rms == 0.0
    assert np.all(updated < alpha)


def test_homeostat_keeps_alpha_within_configured_bounds():
    high = LocalRMSHomeostat(0.5, 1.0, 100.0, 0.2, 0.8)
    low = LocalRMSHomeostat(0.5, 1.0, 100.0, 0.2, 0.8)

    alpha_high, _ = high.update(np.ones(4) * 100.0, np.ones(4) * 0.6)
    alpha_low, _ = low.update(np.zeros(4), np.ones(4) * 0.6)

    assert np.all(alpha_high <= 0.8)
    assert np.all(alpha_low >= 0.2)


def _adapt_to_rho(rho: float) -> tuple[float, float]:
    dim = 64
    stream = AR1Stream(rho=rho, dim=dim, seed=17)
    cell = ActiveVectorCell(alpha=0.6, threshold=10.0, dim=dim)
    homeostat = LocalRMSHomeostat(
        target_rms=0.5,
        ema_rate=0.01,
        adaptation_rate=0.002,
        min_alpha=0.05,
        max_alpha=0.995,
    )

    rms = 0.0
    for _ in range(12000):
        result = cell.step(stream.step())
        updated_alpha, rms = homeostat.update(result.state, cell.alpha)
        cell.set_alpha(updated_alpha)

    return float(np.mean(cell.alpha)), rms


def test_equal_power_temporal_statistics_create_ordered_persistence():
    results = [_adapt_to_rho(rho) for rho in (0.0, 0.2, 0.4, 0.6)]
    alphas = [alpha for alpha, _ in results]
    rms_values = [rms for _, rms in results]

    assert alphas[0] < alphas[1] < alphas[2] < alphas[3]
    assert max(abs(rms - 0.5) for rms in rms_values) < 0.02
