import numpy as np
import pytest

from active_vector_nn.task import (
    NoisyLatentARStream,
    apply_readout,
    fit_ridge_readout,
)


def test_noisy_latent_stream_is_reproducible_and_reports_regime():
    regimes = ((0.0, 3), (0.6, 2))
    left = NoisyLatentARStream(regimes=regimes, dim=4, noise_std=1.5, seed=13)
    right = NoisyLatentARStream(regimes=regimes, dim=4, noise_std=1.5, seed=13)

    seen_rho = []
    for _ in range(5):
        obs_l, latent_l, rho_l = left.step()
        obs_r, latent_r, rho_r = right.step()
        assert obs_l.shape == latent_l.shape == (4,)
        np.testing.assert_allclose(obs_l, obs_r)
        np.testing.assert_allclose(latent_l, latent_r)
        assert rho_l == rho_r
        seen_rho.append(rho_l)

    assert seen_rho == [0.0, 0.0, 0.0, 0.6, 0.6]
    with pytest.raises(StopIteration):
        left.step()


def test_latent_marginal_variance_is_near_one_for_each_rho():
    for rho in (0.0, 0.3, 0.6, 0.85):
        stream = NoisyLatentARStream(
            regimes=((rho, 6000),), dim=16, noise_std=0.0, seed=21
        )
        rows = []
        for index in range(6000):
            _, latent, _ = stream.step()
            if index >= 500:
                rows.append(latent)
        variance = float(np.var(np.asarray(rows)))
        assert 0.9 < variance < 1.1


def test_ridge_readout_recovers_exact_linear_mapping():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(200, 3))
    truth = np.array([[2.0, -1.0], [0.5, 3.0], [-2.0, 0.25]])
    targets = features @ truth

    weights = fit_ridge_readout(features, targets, ridge=1e-10)
    predictions = apply_readout(features, weights)

    np.testing.assert_allclose(weights, truth, atol=1e-8)
    np.testing.assert_allclose(predictions, targets, atol=1e-8)


def test_readout_rejects_mismatched_rows():
    with pytest.raises(ValueError):
        fit_ridge_readout(np.zeros((4, 2)), np.zeros((3, 1)), ridge=1e-6)
