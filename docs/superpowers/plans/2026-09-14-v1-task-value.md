# ActiveVectorNN v1 Task-Value Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure task quality versus event communication while holding sender representation and state budget fixed, then separately record whether v0 local homeostasis helps or hurts the same task.

**Architecture:** Add a deterministic noisy latent-stream task and a tiny ridge readout utility. A frozen v1 experiment trains readouts on independent training streams and evaluates dense sender state plus sparse receiver copies on held-out streams.

**Tech Stack:** Python 3.11/3.12, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-v1-task-value-design.md`

## Global Constraints

- Eight recurrent state scalars for both fixed and adaptive models.
- No gradient or task loss changes recurrent dynamics.
- Readouts train only on the training stream.
- Threshold curve is frozen at `[0.8, 1.0, 1.2, 1.5]`.
- Scientific success/failure is recorded, never used to make CI red.

---

### Task 1: RED task/readout contracts

**Files:**
- Create: `tests/test_task.py`

**Interfaces:**
- `NoisyLatentARStream(regimes, dim, noise_std, seed).step() -> (observation, latent, rho)`
- `fit_ridge_readout(features, targets, ridge) -> ndarray`
- `apply_readout(features, weights) -> ndarray`

- [ ] Write tests for deterministic train/test stream behavior, constant latent marginal variance, observation shape, and readout recovery of an exact linear mapping.
- [ ] Run `pytest tests/test_task.py -q` and verify RED from missing module.

### Task 2: Minimal task utilities

**Files:**
- Create: `src/active_vector_nn/task.py`

- [ ] Implement the piecewise latent stream with stationary AR(1) innovation scaling and additive observation noise.
- [ ] Implement closed-form ridge regression with explicit shape validation.
- [ ] Run focused tests and verify GREEN.

### Task 3: RED frozen v1 receipt

**Files:**
- Create: `tests/test_v1_receipt.py`

**Interfaces:**
- `experiments.run_v1.run_experiment(seed=0) -> dict`

- [ ] Require deterministic keys and finite metrics.
- [ ] Require fixed sender metrics to be identical across event thresholds.
- [ ] Require state max error to respect each threshold.
- [ ] Require train/test seed separation metadata.
- [ ] Do **not** assert the scientific success criterion.
- [ ] Run focused receipt test and verify RED from missing experiment.

### Task 4: Implement same-task experiment

**Files:**
- Create: `experiments/run_v1.py`

- [ ] Generate independent training and held-out test streams using the frozen task parameters.
- [ ] Collect fixed and adaptive sender states.
- [ ] Fit separate but identically specified ridge readouts on training sender states.
- [ ] Evaluate raw observation, dense fixed sender, dense adaptive sender, fixed event curve, and adaptive event curve on held-out data.
- [ ] Compute the pre-registered task-value gate as a boolean/result object without using it as an exception or test condition.
- [ ] Run full tests and experiment.

### Task 5: Results, README, CI

**Files:**
- Create: `RESULTS_V1.md`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`

- [ ] Record exact v1 metrics and whether the pre-registered gate passed.
- [ ] State the adaptive result even if negative.
- [ ] Add `python -m experiments.run_v1` to CI.
- [ ] Run full local suite available in the container, then open a PR for fresh Python 3.11/3.12 verification.
