# ActiveVectorNN v1 results: same task, fewer state transmissions

Frozen experiment: `python -m experiments.run_v1`

v0 had a confound: adaptive persistence changed the sender representation, so fewer communication events could not be interpreted as the same computation done more sparsely.

v1 removes that confound for its primary gate. The fixed dense sender and fixed event receiver use **the same sender dynamics, the same 8-dimensional state, and the same trained linear readout**. Only state synchronization changes.

## Task

An 8-D clean latent AR(1) signal switches through correlations

`0.0 -> 0.3 -> 0.6 -> 0.85`

for 1500 steps each. Every regime has unit latent marginal variance. The observation adds independent Gaussian noise with standard deviation `1.5`.

A ridge-linear readout is fit on seed 0 and evaluated on the independent seed-1 stream. The task is current clean-latent reconstruction.

## Dense representation quality

| representation | held-out task RMSE |
| --- | ---: |
| raw noisy observation | 0.832815 |
| fixed persistent state, alpha=0.6 | **0.821867** |
| locally adaptive persistent state | 0.826684 |

The fixed recurrent state is modestly better than the raw observation on this deliberately noisy task.

The adaptive homeostat finishes at mean `alpha = 0.755354`. It remains better than the raw observation here, but it is slightly worse than the fixed `alpha=0.6` sender. That is a useful negative: local RMS homeostasis is a genuine self-tuning rule, but v1 gives no evidence that it is task-optimal.

## Fixed sender: communication/task curve

| threshold | full-vector event fraction | receiver task RMSE | increase vs dense sender |
| ---: | ---: | ---: | ---: |
| 0.8 | 0.9188 | 0.826110 | +0.52% |
| 1.0 | 0.7817 | 0.837643 | +1.92% |
| 1.2 | 0.6152 | 0.856456 | +4.21% |
| 1.5 | 0.3985 | 0.887862 | +8.03% |

The pre-registered gate required at least one frozen operating point with:

- event fraction <= 0.80, and
- receiver task RMSE <= 1.05 times the dense fixed-sender RMSE.

**Gate result: PASS.** Thresholds `1.0` and `1.2` both pass.

At threshold `1.2`, the receiver gets only **61.52%** as many full-vector transmissions as dense communication while its task RMSE rises by **4.21%** relative to the dense sender.

This is the first result in the repository where the phrase "less communication for nearly the same task" is justified, because the sender state and readout are held fixed.

## Adaptive sender: secondary result

| threshold | full-vector event fraction | receiver task RMSE | increase vs adaptive sender |
| ---: | ---: | ---: | ---: |
| 0.8 | 0.8318 | 0.836599 | +1.20% |
| 1.0 | 0.6723 | 0.852978 | +3.18% |
| 1.2 | 0.5137 | 0.873998 | +5.72% |
| 1.5 | 0.3238 | 0.906796 | +9.69% |

The adaptive state is quieter and therefore communicates less at the same thresholds, but its dense representation already starts slightly worse. v1 therefore does **not** count the adaptive curve as a win over the fixed architecture.

## What changed conceptually

v0 established:

`persistent state != communication`

v1 adds:

`useful task readout can tolerate an approximate downstream copy of that state`

The current implementation still sends an entire 8-D correction whenever any component crosses threshold. That is deliberately crude. The obvious next experiment is coordinate/mode-level events: let only the changed components travel. If that preserves the same hard per-component error bound, it should attack the largest remaining communication inefficiency without changing the sender computation.
