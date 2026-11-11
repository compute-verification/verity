"""The routes: each module here trains a start one way, and a config names one (`training_slowdown.config`).

A route module states:

- `VERSION`, an integer in every config's digest: a change to what the route computes bumps it;
- `DIALS`, each dial's name and default (whose type is the dial's);
- `build(shape, dials)`, the model θ loads into (a start's `shape`, `training_slowdown.model`) and its optimizers;
- `step(model, rows, optimizers, dials, t, n)`, training step `t` of `n` on `rows`, a batch of windows of the start's
  data, (batch, ctx + 1) token ids;
- `plan(dials)`, the `training_slowdown.codecs.Plan` the output model's E is encoded with against θ; it raises
  `ValueError` on a dial value the route has no meaning for, which a config's load turns into its refusal.

The harness owns the loop around `step` (`training_slowdown.run`): the counter, the data, the checkpoints, their
evaluation and E, which it encodes itself from the weights that differ from θ, so no route code runs uncounted. A
route is screened like any code the counter runs (`training_slowdown.screen`): it reaches the harness only through these
functions' arguments, and imports from `training_slowdown` only `codecs` and `model`.

A job may stop after any checkpoint and continue in another process, which calls `build` again and then gets back
what the job kept (`run.resume_state`): the model's `state_dict`, each optimizer's `state_dict`, and torch's, numpy's
and Python's global generators. So a route carries from one step to the next only those. Anything else it would
keep between steps (a module attribute outside the state dict, a generator of its own, a value in a closure) starts
again as `build` leaves it, and the job then continues otherwise than an unbroken one runs.
"""
