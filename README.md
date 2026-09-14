# ActiveVectorNN

What if a network unit does **not** continuously retransmit its whole activation?

ActiveVectorNN starts from a simple split:

- **state** = what the unit is currently maintaining,
- **event** = what changed enough that downstream state needs correction,
- **structure** = the slower persistence rule that determines how the state evolves.

The neuroscience lineage is AnttisNeuron / GrowingAnttisNeuron, but this repository is deliberately an **AI architecture experiment**, not a claim that biological dendrites literally implement these equations.

## v0: maintained vector + innovation events

A cell keeps a vector state

```text
x_t = alpha * x_(t-1) + (1 - alpha) * u_t
```

A downstream receiver keeps its own estimate. Before communication it predicts

```text
xhat^-_t = alpha * xhat_(t-1)
```

The sender computes

```text
delta_t = x_t - xhat^-_t
```

and emits a correction only when

```text
max(abs(delta_t)) > threshold
```

If no event is sent, the receiver continues from its prediction. If an event is sent, the full innovation is applied. That gives a useful hard invariant:

```text
post-step max receiver error <= threshold
```

At threshold zero this collapses to dense communication.

## Slow local self-tuning

The optional homeostat never sees a loss, gradient, task label, desired persistence, or the temporal-correlation label. It only tracks the RMS of its own maintained state.

If local RMS is above one shared target, persistence increases. If RMS is below target, persistence decreases. The update happens slowly in persistence-logit space.

With equal-variance AR(1) streams, changing only temporal correlation produced different final persistence values:

| temporal correlation rho | final persistence alpha | final local RMS |
| ---: | ---: | ---: |
| 0.0 | 0.5992 | 0.5057 |
| 0.2 | 0.6796 | 0.5070 |
| 0.4 | 0.7645 | 0.5083 |
| 0.6 | 0.8488 | 0.5102 |

So the same starting unit plus the same local rule can acquire a different effective memory from the statistics of the stream alone.

## Communication result

On the frozen 8-D piecewise-correlated stream, v0 records a whole threshold curve rather than selecting one flattering operating point.

| threshold | fixed events | adaptive events |
| ---: | ---: | ---: |
| 0.4 | 95.50% | 81.24% |
| 0.6 | 73.10% | 52.04% |
| 0.8 | 46.36% | 30.68% |
| 1.0 | 27.68% | 16.62% |

The receiver-error bound holds at every step. See [`RESULTS_V0.md`](RESULTS_V0.md) for RMSE values and the exact interpretation.

## The important limitation

The adaptive cell changes `alpha`, which means it changes **what state it maintains**. Therefore fewer events do **not** yet mean "same useful computation for less communication." It may simply be smoothing away information.

That is intentional. v0 isolates the mechanism first.

The next meaningful gate is task-facing:

```text
same state budget
+ same downstream task
+ fixed vs locally adaptive persistence
+ dense vs event communication
```

Then compare task quality against communicated events. If the adaptive/event model preserves useful performance while transmitting less, the architecture starts to have a real AI claim. If it loses task information, the idea gets narrowed or rejected.

## Run it

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_v0
```

## Repository map

- `src/active_vector_nn/core.py` — persistent vector state and event channel
- `src/active_vector_nn/homeostasis.py` — slow local RMS homeostat
- `src/active_vector_nn/streams.py` — equal-variance AR(1) streams
- `experiments/run_v0.py` — frozen v0 receipt
- `tests/` — behavioral and scientific invariant tests
- `RESULTS_V0.md` — measured v0 result and limits
- `docs/superpowers/` — design and implementation plan

## Current claim boundary

The interesting object here is not "a better neuron." It is a possible **locally self-tuning event-driven state-space primitive**:

```text
fast:   innovation / event
medium: maintained vector state
slow:   local persistence adaptation
```

v0 shows those three timescales can be made explicit, deterministic, and testable. It does not yet show that they outperform a conventional recurrent or state-space model on a task.
