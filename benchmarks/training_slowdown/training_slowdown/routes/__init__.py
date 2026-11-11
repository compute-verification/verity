"""The routes: each module here trains a start one way, and a config names one (`training_slowdown.config`).

A route module states:

- `VERSION`, an integer in every config's digest: a change to what the route computes bumps it;
- `DIALS`, each dial's name and default (whose type is the dial's);
- `build(shape, dials)`, the model θ loads into (a start's `shape`, `training_slowdown.model`) and its optimizers;
- `step(model, rows, optimizers, dials, t, n)`, training step `t` of `n` on `rows`, a batch of windows of the start's
  data, (batch, ctx + 1) token ids;
- `plan(dials)`, the `training_slowdown.codecs.Plan` the output model's E is encoded with against θ; it raises
  `ValueError` on a dial value the route has no meaning for, which a config's load turns into its refusal;
- optionally `output(model, dials)`, the output model's state dict under θ's names, where it isn't the trained
  model's own (an average of the weights, adapters merged into theirs). The harness calls it at each checkpoint. It is
  no free place to compute: a run counts what it computes, and the trace takes it as the end of every step, so its
  products are charged with the step's. A route that trained a weight of θ's in a codec's own form (the codes of a
  quantizer) may give that weight as the tuple of its trained parts, the tensors its plan's codec packs
  (`Codec.encode_trained`): the plan's codec packs them as they are, and the weight scored is the one that codec
  decodes from those bytes on θ, never one the route computed. No route packs or decodes: a rounding the library
  lacks is added to `training_slowdown.codecs`.

The harness owns the loop around `step` (`training_slowdown.run`): the counter, the data, the checkpoints, their
evaluation and E, which it encodes itself from the weights that differ from θ, so no route code runs uncounted. The
trace runs `step` on a copy of the model and of the optimizers, so `step` reaches every tensor it trains through those
two arguments: a tensor it reaches around them (a global, a closure) is the run's own, and the trace refuses it. A
route is screened like any code the counter runs (`run.screen`): it reaches the harness only through these
functions' arguments, and imports from `training_slowdown` only `codecs` and `model`, and a module beside it here
(`from ._optimizers import Lion`, `from . import subset`), which is held to the same screen. A module whose name
starts with `_` is no route: no config names it.
"""
