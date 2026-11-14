"""The routes: each module here trains a start one way, and a config names one (`training_slowdown.config`).

A route module states:

- `VERSION`, an integer in every config's digest: a change to what the route computes bumps it;
- `DIALS`, each dial's name and default (whose type is the dial's);
- optionally `LATER`, the names of the dials whose default is the step the route takes without the dial. A config
  that leaves one there digests without it, so a route gains a dial this way and no config written before it changes
  its digest, or what a record under that digest means. A dial whose default would change the step is not one: that
  is a new `VERSION`;
- optionally `EITHER`, groups of dials that say one thing in two ways, of which the route reads one. A config that
  writes two dials of a group is refused, whatever their values;
- `build(shape, dials)`, the model θ loads into (a start's `shape`, `training_slowdown.model`) and its optimizers;
- optionally `origin(shape, dials)`, where the model `build` returns doesn't start as θ (a deeper one, `grow`): each
  tensor of it that doesn't start as θ's tensor of its own name, mapped to θ's tensor it starts as, or to None for
  zeros, from the dials alone. The harness builds that start θ′ from θ (`run.start_state`, which refuses a map that
  doesn't cover the model or θ, or that starts a tensor as one of another shape) and loads it, and θ′ stands where θ
  does in the job: in the counter, so a product with a copied tensor not yet written is θ's kind and one with a
  zeroed tensor a constant's, and as E's base. θ stays the registered start, whose digest the record states, and the
  record states the map, θ′'s digest and the output's blocks (`origin`). No route reads θ;
- `step(model, rows, optimizers, dials, t, n)`, training step `t` of `n` on `rows`, a batch of windows of the start's
  data, (batch, ctx + 1) token ids;
- `plan(dials)`, the `training_slowdown.codecs.Plan` the output model's E is encoded with against θ (θ′ where the
  route states an `origin`); it raises
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
two arguments: a tensor it reaches around them (a global, a closure) is the run's own, and the trace refuses it.
Which windows `rows` holds is the harness's too: every window of the start's data in the job's seed's order, or,
where the config asks (`windows`, `training_slowdown.config`), only those the start's own training did not read. No
dial chooses them, and a route sees a batch and never the stream. A
route is screened like any code the counter runs (`training_slowdown.screen`): it reaches the harness only through these
functions' arguments, and imports from `training_slowdown` only `codecs` and `model`, and a module beside it here
(`from ._optimizers import Lion`, `from . import subset`), which is held to the same screen. A module whose name
starts with `_` is no route: no config names it.

A step copies a state dict, never the module: where it needs the weights as they were, it takes
`{k: v.detach().clone() for k, v in model.state_dict().items()}`. A job's trace runs `step` on tensors without data
(`training_slowdown.capture`), which have nothing a deep copy can copy, so the capture refuses a step that deep-copies
the model, one of its parameters or an optimizer (`copy.deepcopy`), naming the line: the job stays valid and has no
trace, so no ρ.

A job may stop after any checkpoint and continue in another process, which calls `build` again and then gets back
what the job kept (`run.resume_state`): the model's `state_dict`, its buffers outside the state dict (those `build`
registered, by name), each optimizer's `state_dict`, and torch's, numpy's and Python's global generators. So a route
carries from one step to the next only those: what its `output` reads beside the weights (an average of them, an
adapter it merges away) is a parameter or a buffer of the model, and the job keeps the model the route trains, not
the output a checkpoint scores. Anything else a route would keep between steps (a module attribute that is neither,
a generator of its own, a value in a closure) starts again as `build` leaves it, and the job then continues
otherwise than an unbroken one runs.
"""
