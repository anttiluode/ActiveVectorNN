import numpy as np
import pytest

from active_vector_nn.core import ActiveVectorCell


def test_state_is_persistent_leaky_vector():
    cell = ActiveVectorCell(alpha=0.5, threshold=10.0, dim=2)

    first = cell.step(np.array([1.0, -1.0]))
    second = cell.step(np.array([1.0, -1.0]))

    np.testing.assert_allclose(first.state, [0.5, -0.5])
    np.testing.assert_allclose(second.state, [0.75, -0.75])


def test_threshold_zero_matches_dense_receiver_state():
    rng = np.random.default_rng(7)
    cell = ActiveVectorCell(alpha=0.8, threshold=0.0, dim=4)

    for _ in range(50):
        result = cell.step(rng.normal(size=4))
        np.testing.assert_allclose(result.receiver_state, result.state, atol=1e-12)


def test_silent_or_emitted_steps_never_leave_receiver_beyond_threshold():
    rng = np.random.default_rng(11)
    threshold = 0.12
    cell = ActiveVectorCell(alpha=np.array([0.3, 0.6, 0.8]), threshold=threshold, dim=3)

    saw_silent = False
    saw_event = False
    for _ in range(300):
        result = cell.step(rng.normal(scale=0.35, size=3))
        saw_silent |= not result.emitted
        saw_event |= result.emitted
        assert result.max_error <= threshold + 1e-12
        assert np.max(np.abs(result.state - result.receiver_state)) <= threshold + 1e-12

    assert saw_silent
    assert saw_event


def test_reset_clears_sender_and_receiver_state():
    cell = ActiveVectorCell(alpha=0.7, threshold=0.1, dim=2)
    cell.step(np.array([1.0, 2.0]))

    cell.reset()
    result = cell.step(np.zeros(2))

    np.testing.assert_allclose(result.state, np.zeros(2))
    np.testing.assert_allclose(result.receiver_state, np.zeros(2))
    assert not result.emitted


@pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0], np.ones((2, 2))])
def test_step_rejects_wrong_input_shape(bad):
    cell = ActiveVectorCell(alpha=0.5, threshold=0.1, dim=2)
    with pytest.raises(ValueError):
        cell.step(np.asarray(bad, dtype=float))


def test_alpha_must_be_inside_open_unit_interval():
    with pytest.raises(ValueError):
        ActiveVectorCell(alpha=1.0, threshold=0.1, dim=1)
    with pytest.raises(ValueError):
        ActiveVectorCell(alpha=0.0, threshold=0.1, dim=1)


def test_threshold_must_be_nonnegative():
    with pytest.raises(ValueError):
        ActiveVectorCell(alpha=0.5, threshold=-0.1, dim=1)
