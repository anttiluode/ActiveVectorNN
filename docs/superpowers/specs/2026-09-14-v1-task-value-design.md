# ActiveVectorNN v1 Task-Value Design

## Question

v0 proved that maintained vector state, sparse innovation events, and slow local persistence adaptation can coexist. It did **not** prove that the maintained state is useful for a task, because adaptive persistence changes the representation itself.

v1 asks the missing question:

> Under the same state budget and the same downstream task, how much event communication can be removed before useful task performance materially degrades?

The primary test isolates communication by keeping the sender dynamics fixed. Adaptive persistence is reported separately so it cannot hide behind a changed representation.

## Task

Use an 8-dimensional latent AR(1) process whose temporal correlation switches through four regimes:

`rho = 0.0 -> 0.3 -> 0.6 -> 0.85`, 1500 steps each.

The latent process has unit marginal variance in every regime. The observed input is

`u_t = z_t + sigma * epsilon_t`

with `sigma = 1.5` and independent unit Gaussian noise. The downstream task is to estimate the current clean latent vector `z_t`.

This is deliberately simple: a useful persistent state should be able to act as a low-pass denoising representation, while sparse event synchronization should reveal the cost of keeping a downstream copy of that representation current.

## Models and fairness

All recurrent models have exactly 8 maintained state scalars.

### Raw observation baseline

A ridge-linear readout maps the current noisy observation directly to the clean latent target.

### Fixed dense sender

`x_t = 0.6 * x_(t-1) + 0.4 * u_t`

A ridge-linear readout is trained on the sender state and clean latent target.

### Fixed event receiver

The sender state and readout are exactly the same as the fixed dense sender. Only communication changes: the receiver predicts its next state and receives a full-vector innovation only when the max innovation exceeds a threshold.

Because the representation and readout are identical, task degradation here can be attributed to sparse synchronization rather than a changed sender computation.

### Adaptive sender / receiver

The v0 local RMS homeostat changes persistence online. It uses the same 8 state scalars and no task loss or gradient. Its RMS target is not tuned from task results; it is analytically calibrated to the expected RMS of the initial `alpha=0.6` filter under the white (`rho=0`) noisy observation variance:

`target_rms = sqrt((1-alpha)/(1+alpha) * (1 + sigma^2)) = sqrt(0.8125)`.

The adaptive representation gets its own ridge readout trained by the same procedure. Adaptive results are secondary: v1 does not assume the homeostat is task-optimal.

## Train/test separation

Generate independent deterministic training and test streams from different seeds with the same regime schedule. Fit readouts on sender states from the training stream only. Report all metrics on the held-out test stream.

## Event thresholds

Record the full-vector event tradeoff at thresholds:

`[0.8, 1.0, 1.2, 1.5]`.

No single threshold is selected after seeing results.

## Metrics

For raw, fixed sender, and adaptive sender:

- held-out task RMSE.

For each event threshold and architecture:

- event fraction (fraction of time steps transmitting a full vector),
- receiver task RMSE,
- receiver-vs-sender state RMSE,
- maximum receiver state error.

## Pre-registered interpretation

The primary v1 gate is considered promising if **any recorded fixed-event operating point** achieves both:

- event fraction `<= 0.80`, and
- receiver task RMSE `<= 1.05 * fixed dense sender task RMSE`.

This criterion is recorded as a scientific result, not used as a CI pass/fail condition.

The adaptive result has no positive success criterion. If its sender task RMSE is worse than the fixed sender, v1 records that as evidence that local RMS homeostasis is a self-tuning mechanism but not automatically a useful learning rule.

## Non-goals

- no backpropagation through recurrent state,
- no learned recurrence,
- no threshold search beyond the frozen four-point curve,
- no claim that the synthetic denoising task represents general AI,
- no claim that reduced event fraction equals reduced wall-clock compute on current dense hardware.

## Next decision

If fixed event synchronization passes the task-value gate, the next engineering question is finer-grained communication: coordinate/mode-level events rather than sending a full vector whenever any component crosses threshold.

If it fails, the event idea needs a different receiver predictor or a different state representation before more biological machinery is added.
