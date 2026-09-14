# ActiveVectorNN

What if a network unit does **not** continuously retransmit its whole activation?

ActiveVectorNN starts from a simple split:

- **state** = what the unit is currently maintaining,
- **event** = what changed enough that downstream state needs correction,
- **structure** = the slower persistence rule that determines how the state evolves.

The neuroscience lineage is AnttisNeuron / GrowingAnttisNeuron, but this repository is deliberately an **AI architecture experiment**, not a claim that biological dendrites literally implement these equations.

## Core recurrence

A cell keeps a vector state

```text
x_t = alpha * x_(t-1) + (1 - alpha) * u_t
```

A downstream receiver keeps its own estimate and predicts

```text
xhat^-_t = alpha * xhat_(t-1)
```

The sender computes the innovation

```text
delta_t = x_t - xhat^-_t
```

and transmits a correction only when

```text
max(abs(delta_t)) > threshold
```

If no event is sent, the receiver continues from its prediction. If an event is sent, the full innovation is applied. Therefore

```text
post-step max receiver error <= threshold
```

At threshold zero this collapses to dense state communication.

## v0: signal statistics can retune persistence locally

The optional homeostat never sees a loss, gradient, task label, desired persistence, or temporal-correlation label. It only tracks the RMS of its own maintained state.

With equal-variance AR(1) streams, changing only temporal correlation produced:

| temporal correlation rho | final persistence alpha | final local RMS |
| ---: | ---: | ---: |
| 0.0 | 0.5992 | 0.5057 |
| 0.2 | 0.6796 | 0.5070 |
| 0.4 | 0.7645 | 0.5083 |
| 0.6 | 0.8488 | 0.5102 |

So the same starting unit plus the same local rule can acquire a different effective memory from temporal statistics alone.

v0 also showed a communication/error tradeoff, but it had an important confound: adaptive persistence changes the sender representation itself. Fewer adaptive events therefore did **not** yet mean the same useful computation for less communication. See [`RESULTS_V0.md`](RESULTS_V0.md).

## v1: same task, same sender, less communication

v1 removes that confound for the primary gate.

An 8-D persistent state is trained only through a tiny linear readout to reconstruct a clean latent process from heavily noisy observations. The dense sender and event receiver use **exactly the same recurrent state and the same readout**; only communication changes.

Held-out task RMSE:

| representation | RMSE |
| --- | ---: |
| raw noisy observation | 0.832815 |
| fixed persistent state | **0.821867** |
| locally adaptive persistent state | 0.826684 |

For the fixed sender, the full-vector event tradeoff is:

| threshold | event fraction | receiver task RMSE | task increase vs dense sender |
| ---: | ---: | ---: | ---: |
| 0.8 | 91.88% | 0.826110 | +0.52% |
| 1.0 | 78.17% | 0.837643 | +1.92% |
| 1.2 | 61.52% | 0.856456 | +4.21% |
| 1.5 | 39.85% | 0.887862 | +8.03% |

The pre-registered v1 gate required an operating point with no more than 80% of dense transmissions and no more than 5% relative task-RMSE increase. **Thresholds 1.0 and 1.2 pass.**

That is the first result here that supports the narrow statement:

```text
useful maintained state can be synchronized approximately
with fewer transmissions while preserving most task quality
```

See [`RESULTS_V1.md`](RESULTS_V1.md) for the exact setup and limits.

## The adaptive result is deliberately mixed

The local RMS homeostat finishes with a different persistence and communicates less at the same thresholds, but its dense sender RMSE is slightly worse than the fixed sender (`0.826684` vs `0.821867`).

So v1 does **not** support the stronger story that local homeostatic self-tuning is automatically task-optimal. It is a real unsupervised adaptation mechanism; whether its objective is useful depends on the task.

## What the architecture currently is

```text
fast:   innovation / event
medium: maintained vector state
slow:   local persistence adaptation
```

The useful part so far is the separation between maintained state and communication. The slow adaptation remains an open research direction rather than a proven advantage.

## Next gate

The current event is crude: if **any** one of the 8 components exceeds threshold, the entire 8-D correction is transmitted.

The obvious next experiment is mode/coordinate-level events:

```text
only send the state components whose innovations crossed threshold
```

That should preserve the same per-component error bound while attacking the largest remaining communication inefficiency. It is also closer to the original intuition that several simultaneously active components/modes make up the current state, while only changed components need to announce themselves.

## Run it

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_v0
python -m experiments.run_v1
```

## Repository map

- `src/active_vector_nn/core.py` — persistent vector state and event channel
- `src/active_vector_nn/homeostasis.py` — slow local RMS homeostat
- `src/active_vector_nn/streams.py` — equal-variance AR(1) streams
- `src/active_vector_nn/task.py` — v1 latent-stream task and ridge readout
- `experiments/run_v0.py` — mechanism receipt
- `experiments/run_v1.py` — same-task communication receipt
- `tests/` — behavioral and scientific invariant tests
- `RESULTS_V0.md`, `RESULTS_V1.md` — measured results and claim boundaries
- `docs/superpowers/` — frozen designs and implementation plans
