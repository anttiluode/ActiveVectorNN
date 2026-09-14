# ActiveVectorNN v0 results

Frozen experiment: `python -m experiments.run_v0`

The first gate asks only whether three distinct objects can coexist cleanly:

1. a persistent local vector state,
2. sparse communication of innovations rather than the whole state,
3. a slow local rule that changes persistence from temporal statistics without gradients or task labels.

## Local temporal differentiation

All four streams have the same unit marginal variance. They differ only in AR(1) temporal correlation. Every cell starts with `alpha = 0.6`, and every homeostat has the same RMS target `0.5`.

| rho | final alpha | final local RMS |
| ---: | ---: | ---: |
| 0.0 | 0.599182 | 0.505746 |
| 0.2 | 0.679573 | 0.507009 |
| 0.4 | 0.764523 | 0.508256 |
| 0.6 | 0.848789 | 0.510178 |

The ordering is strict. The rule never receives `rho`, a desired alpha, a task loss, a gradient, or receiver error. It sees only its own maintained-state RMS relative to one shared target.

## Event communication curve

The sender maintains

`x_t = alpha * x_(t-1) + (1-alpha) * u_t`.

The receiver predicts the sender's next state from its previous estimate. The sender emits a correction only when the max absolute innovation exceeds the threshold. Consequently the receiver max error is bounded by that threshold after every step.

The stream is 8-dimensional and switches through correlations `0.0 -> 0.2 -> 0.6 -> 0.4 -> 0.0`, 1000 steps each.

| threshold | fixed event fraction | fixed RMSE | adaptive event fraction | adaptive RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0.4 | 0.9550 | 0.04635 | 0.8124 | 0.08720 |
| 0.6 | 0.7310 | 0.15604 | 0.5204 | 0.19108 |
| 0.8 | 0.4636 | 0.27076 | 0.3068 | 0.28376 |
| 1.0 | 0.2768 | 0.36449 | 0.1662 | 0.36897 |

The threshold-zero dense control emits on every step and reconstructs sender state with zero measured error.

## What v0 establishes

- Persistent local state and communicated events are separable objects.
- Sparse innovation communication can maintain a bounded approximation of that state.
- A gradient-free local homeostat can retune persistence from temporal structure while returning its observed RMS toward a shared target.
- On this particular piecewise stream, the adaptive state requires fewer correction events than the fixed state at all four recorded thresholds.

## What v0 does **not** establish

The fixed and adaptive cells do **not** maintain the same representation. The homeostat changes `alpha`, so it changes the state itself. Therefore the lower adaptive event fraction is not yet evidence for equal task performance at lower communication cost.

That is the next gate. v1 should attach the same downstream task to fixed and adaptive cells and compare task quality against event count under an equal state budget. If adaptation merely smooths away useful information, v1 should expose it.
