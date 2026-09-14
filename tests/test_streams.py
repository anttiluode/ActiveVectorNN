import numpy as np
import pytest

from active_vector_nn.streams import AR1Stream


def test_ar1_stream_is_reproducible_and_vector_shaped():
    left = AR1Stream(rho=0.6, dim=5, seed=123)
    right = AR1Stream(rho=0.6, dim=5, seed=123)

    for _ in range(20):
        a = left.step()
        b = right.step()
        assert a.shape == (5,)
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("rho", [0.0, 0.2, 0.6, -0.5])
def test_ar1_stream_keeps_approximately_unit_marginal_variance(rho):
    stream = AR1Stream(rho=rho, dim=16, seed=9)
    samples = []
    for step in range(5000):
        value = stream.step()
        if step >= 500:
            samples.append(value.copy())

    variance = float(np.var(np.asarray(samples)))
    assert 0.9 < variance < 1.1


def test_ar1_rejects_nonstationary_rho():
    with pytest.raises(ValueError):
        AR1Stream(rho=1.0, dim=2, seed=0)
    with pytest.raises(ValueError):
        AR1Stream(rho=-1.0, dim=2, seed=0)
