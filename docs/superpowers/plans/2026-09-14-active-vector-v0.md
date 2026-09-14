# ActiveVectorNN v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a minimal event-driven recurrent vector cell whose persistent state is locally maintained, whose downstream communication carries only innovations, and whose persistence can adapt using only local RMS homeostasis.

**Architecture:** A NumPy-only package separates fast state evolution/event emission from slow local persistence adaptation. Deterministic AR(1) streams provide equal-variance temporal environments. A frozen experiment compares dense, fixed-event, and adaptive-event channels without training or parameter search.

**Tech Stack:** Python 3.11/3.12, NumPy, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-14-active-vector-v0-design.md`

## Global Constraints

- No backpropagation or task loss in v0.
- The homeostat may observe only local state RMS and one common target.
- Threshold-zero event communication is a required dense-control equivalence.
- Scientific outcomes are recorded even when they are negative.
- Dependencies are limited to NumPy for runtime and pytest for tests.

---

### Task 1: Project test harness and RED core contract

**Files:**
- Create: `pyproject.toml`
- Create: `tests/test_core.py`

**Interfaces:**
- Consumes: none.
- Produces: required API `ActiveVectorCell(alpha, threshold)`, `step(u) -> StepResult`, and `reset()`.

- [ ] **Step 1: Add package/test configuration**

Configure setuptools with `src/` layout, NumPy runtime dependency, pytest test dependency, and pytest `pythonpath = ["src"]`.

- [ ] **Step 2: Write failing event-channel tests**

Tests must require: shape validation; persistent leaky state; threshold-zero dense equivalence; and post-step receiver max error bounded by threshold.

- [ ] **Step 3: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_core.py -q`
Expected: import failure because `active_vector_nn.core` does not exist.

- [ ] **Step 4: Commit RED tests**

Commit only configuration and tests before production code.

### Task 2: Minimal active-vector event channel

**Files:**
- Create: `src/active_vector_nn/__init__.py`
- Create: `src/active_vector_nn/core.py`

**Interfaces:**
- Produces `StepResult(state, receiver_state, innovation, emitted, max_error)` and `ActiveVectorCell`.

- [ ] **Step 1: Implement the minimal recurrence**

Use elementwise `x = alpha*x + (1-alpha)*u`. Receiver predicts `alpha*xhat`; innovation is sender state minus prediction. Emit the full correction only when max absolute innovation exceeds threshold.

- [ ] **Step 2: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_core.py -q`
Expected: all core tests pass.

- [ ] **Step 3: Refactor names/docstrings without changing behavior**

- [ ] **Step 4: Run focused tests again and commit**

### Task 3: RED local homeostasis contract

**Files:**
- Create: `tests/test_homeostasis.py`
- Create: `tests/test_streams.py`

**Interfaces:**
- Requires `AR1Stream(rho, dim, seed)` and `LocalRMSHomeostat(target_rms, ema_rate, adaptation_rate, min_alpha, max_alpha)`.
- Homeostat API: `update(state, alpha) -> (new_alpha, local_rms)`.

- [ ] **Step 1: Write deterministic AR(1) tests**

Require reproducibility, finite output, correct vector shape, and approximately equal marginal variance across several `rho` values after burn-in.

- [ ] **Step 2: Write homeostat tests**

Require alpha to increase when local RMS is above target, decrease when below target, stay bounded, and use no external labels.

- [ ] **Step 3: Write differentiation test**

Run four equal-variance streams (`rho=0,.2,.4,.6`) through identical cells with one shared target. Require strictly ordered final alpha and final RMS values near the target.

- [ ] **Step 4: Run focused tests and verify RED**

Expected: import failures for the missing modules.

### Task 4: Implement streams and slow local adaptation

**Files:**
- Create: `src/active_vector_nn/streams.py`
- Create: `src/active_vector_nn/homeostasis.py`
- Modify: `src/active_vector_nn/core.py`

**Interfaces:**
- `AR1Stream.step() -> np.ndarray`
- `LocalRMSHomeostat.update(state, alpha) -> tuple[np.ndarray, float]`
- `ActiveVectorCell.set_alpha(alpha)` validates/broadcasts updated persistence.

- [ ] **Step 1: Implement unit-variance AR(1)**

Use `u_t = rho*u_{t-1} + sqrt(1-rho^2)*epsilon_t` so marginal variance is one for `|rho|<1`.

- [ ] **Step 2: Implement EMA RMS and logit-space alpha adaptation**

Track scalar local power from mean squared state. Adapt alpha upward when RMS exceeds target and downward when below target, clamp only through configured alpha bounds.

- [ ] **Step 3: Run focused tests and tune only numerical stability constants if required by the tests**

No search over scientific stream parameters is allowed.

- [ ] **Step 4: Run full tests and commit**

### Task 5: Frozen v0 experiment and scientific receipt

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/run_v0.py`
- Create: `tests/test_v0_receipt.py`
- Create: `RESULTS_V0.md`

**Interfaces:**
- `run_experiment(seed=0) -> dict` returns dense/fixed/adaptive metrics and homeostatic differentiation metrics.

- [ ] **Step 1: Write receipt test first**

Require deterministic keys/finite values, dense event fraction `1.0`, fixed/adaptive post-step max error within threshold tolerance, and the four-rho alpha ordering.

- [ ] **Step 2: Run receipt test and verify RED**

Expected: missing `experiments.run_v0`.

- [ ] **Step 3: Implement one frozen experiment**

Use an 8-D piecewise AR(1) stream with fixed seed and predetermined regimes. Compare dense threshold zero, fixed event threshold, and adaptive event threshold. Record RMSE, max error, event fraction, final mean alpha, and differentiation table.

- [ ] **Step 4: Generate and commit `RESULTS_V0.md` from the measured output**

State negative/mixed results explicitly.

### Task 6: Documentation and CI

**Files:**
- Create: `README.md`
- Create: `.github/workflows/ci.yml`

**Interfaces:** none.

- [ ] **Step 1: Document the state/event/structure model and exact equations**

- [ ] **Step 2: Add Python 3.11 and 3.12 CI**

Install package with test extras, run `pytest -q`, then run `python -m experiments.run_v0`.

- [ ] **Step 3: Perform fresh local verification**

Run all tests and the experiment from a clean clone of the branch.

- [ ] **Step 4: Open a PR only after verification is green**

Do not merge until the PR head checks are confirmed green.
