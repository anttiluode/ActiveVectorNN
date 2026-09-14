from experiments.run_v0 import run_experiment


def test_v0_receipt_is_deterministic_and_respects_event_bounds():
    receipt = run_experiment(seed=0)
    assert receipt == run_experiment(seed=0)

    dense = receipt["dense"]
    assert dense["threshold"] == 0.0
    assert dense["event_fraction"] == 1.0
    assert dense["rmse"] < 1e-12
    assert dense["max_error"] < 1e-12

    for family in ("fixed_curve", "adaptive_curve"):
        rows = receipt[family]
        assert [row["threshold"] for row in rows] == [0.4, 0.6, 0.8, 1.0]
        for row in rows:
            assert 0.0 <= row["event_fraction"] <= 1.0
            assert row["max_error"] <= row["threshold"] + 1e-12
            assert row["rmse"] >= 0.0
        assert rows[2]["event_fraction"] < 1.0


def test_v0_receipt_keeps_homeostatic_differentiation_visible():
    receipt = run_experiment(seed=0)
    rows = receipt["homeostatic_differentiation"]

    assert [row["rho"] for row in rows] == [0.0, 0.2, 0.4, 0.6]
    alphas = [row["final_alpha"] for row in rows]
    rms_values = [row["final_rms"] for row in rows]

    assert alphas[0] < alphas[1] < alphas[2] < alphas[3]
    assert max(abs(rms - 0.5) for rms in rms_values) < 0.02
