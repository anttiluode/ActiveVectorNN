# ActiveVectorNN v0 Design

## Purpose

Build the smallest falsifiable AI system around the idea that a unit's persistent internal vector is the maintained present, while communication carries only changes that downstream state cannot already predict.

This is not a neuroscience simulator. The neuroscience lineage motivates three separated timescales, but the implementation is an AI mechanism with explicit invariants and controls.

## Core object

Each cell owns a vector state `x_t` and a per-dimension persistence parameter `alpha`:

`x_t = alpha * x_{t-1} + (1 - alpha) * u_t`

`x_t` is the local maintained state. A downstream receiver keeps its own estimate `xhat_t`. Before receiving anything at step `t`, it predicts the sender by applying the same autonomous persistence:

`xhat^-_t = alpha * xhat_{t-1}`

The sender forms the innovation:

`delta_t = x_t - xhat^-_t`

If `max(abs(delta_t)) > threshold`, the sender transmits that correction and the receiver applies it exactly. Otherwise nothing is transmitted. Therefore the receiver's post-step error must never exceed the configured threshold in max norm.

At threshold zero, the event channel should reduce to dense communication up to floating-point zero cases. This is a required control.

## Slow local adaptation

The optional homeostat sees only the cell's own state magnitude. It tracks an exponential moving estimate of local state power. If local RMS is above a common target, it increases persistence; if local RMS is below target, it decreases persistence.

The update is performed in persistence-logit space to keep `alpha` bounded. The rule may use only `(local_rms, target_rms, adaptation_rate)`; it may not observe the input correlation label, a task loss, gradients, desired final alpha, or receiver error.

Equal-variance but differently correlated AR(1) streams are the first scientific probe. The preregistered qualitative prediction is that more positively correlated streams require larger learned persistence to hold local RMS near one shared target.

## v0 experiments

### Gate A: event-channel invariant

For deterministic vector streams and several thresholds, verify that receiver max-norm error after each step is at most the threshold plus numerical tolerance.

### Gate B: dense control

At threshold zero, verify event-driven receiver state matches dense receiver state exactly within numerical tolerance.

### Gate C: local timescale differentiation

Generate equal-variance AR(1) streams with `rho = [0.0, 0.2, 0.4, 0.6]`. Start all cells from the same persistence. Use one shared RMS target and local homeostat. Record final persistence and final local RMS. Required result for claiming success: final persistence is strictly ordered with `rho`, while final local RMS values converge near the shared target.

### Gate D: communication tradeoff

On a piecewise-correlated vector stream, compare dense communication, fixed-persistence event communication, and adaptive-persistence event communication. Record receiver RMSE, max error, event fraction, and final persistence. No claim is made that adaptation must beat fixed persistence in v0; adaptation is retained if it satisfies its local rule and produces interpretable behavior.

## Non-goals

- no backpropagation
- no trainable output head
- no claim that biological spikes are mathematical derivatives
- no claim that dendrites literally implement this recurrence
- no benchmark-chasing
- no hidden parameter search after seeing experiment results

## Repository shape

- `src/active_vector_nn/core.py`: state, receiver prediction, event emission
- `src/active_vector_nn/homeostasis.py`: local RMS tracker and persistence adaptation
- `src/active_vector_nn/streams.py`: deterministic equal-variance AR(1) generators
- `experiments/run_v0.py`: frozen scientific receipt
- `tests/`: unit and scientific invariant tests
- `README.md`: concept, equations, controls, and current result
- `.github/workflows/ci.yml`: Python 3.11/3.12 tests and frozen experiment smoke run

## Success boundary

v0 succeeds only if it demonstrates a real state/communication separation: persistent local vector state can remain accurate downstream under sparse corrections, and the slow local rule can retune persistence from stream statistics without gradients or task labels. Anything beyond that waits for later gates.
