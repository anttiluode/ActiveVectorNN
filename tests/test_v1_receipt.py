import math

from experiments.run_v1 import run_experiment


def test_v1_receipt_is_deterministic_and_uses_held_out_stream():
    receipt = run_experiment(seed=0)
    assert receipt == run_experiment(seed=0)
    assert receipt["config"]["train_seed"] != receipt["config"]["test_seed"]
    assert receipt["config"]["dim"] == 8
    assert receipt["config"]["event_thresholds"] == [0.8, 1.0, 1.2, 1.5]

    for key in ("raw_task_rmse", "fixed_sender_task_rmse", "adaptive_sender_task_rmse"):
        assert math.isfinite(receipt[key])
        assert receipt[key] >= 0.0


def test_v1_event_curves_preserve_sender_accounting_and_error_bounds():
    receipt = run_experiment(seed=0)

    for family, sender_key in (
        ("fixed_event_curve", "fixed_sender_task_rmse"),
        ("adaptive_event_curve", "adaptive_sender_task_rmse"),
    ):
        rows = receipt[family]
        assert [row["threshold"] for row in rows] == [0.8, 1.0, 1.2, 1.5]
        for row in rows:
            assert row["sender_task_rmse"] == receipt[sender_key]
            assert 0.0 <= row["event_fraction"] <= 1.0
            assert row["max_state_error"] <= row["threshold"] + 1e-12
            assert row["receiver_task_rmse"] >= 0.0
            assert row["receiver_state_rmse"] >= 0.0


def test_v1_scientific_gate_is_recorded_not_enforced():
    receipt = run_experiment(seed=0)
    gate = receipt["fixed_task_value_gate"]

    assert isinstance(gate["passed"], bool)
    assert gate["max_event_fraction"] == 0.8
    assert gate["max_relative_task_rmse"] == 1.05
    assert all(value in [0.8, 1.0, 1.2, 1.5] for value in gate["passing_thresholds"])
