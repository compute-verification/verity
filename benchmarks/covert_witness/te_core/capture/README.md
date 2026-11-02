# te_core.capture: a training step's circuit structure, read from its PyTorch code

A search agent writes ordinary PyTorch training code. `capture_step` traces one whole training step of it (forward,
autograd's backward and the optimizer's update) into an ATen graph with shapes and dtypes and no values, and reads from
that graph what te_core scores: every multiplication's MACs, the role of each GEMM, its token dims, and the updated MACs
upstream of it. A run's circuit C is the sum of its distinct step graphs (`capture_run`), so the size of C is exact in
MACs without building a gate. The trace is `make_fx(tracing_mode="fake")` under `FakeTensorMode`, on a fake copy of the
model and a fresh optimizer of the same class and groups, so the caller's objects are never touched and nothing runs.

~~~python
from te_core.capture import capture_step, capture_run, schedule_updated
from te_core.capture.lower import lower

g = capture_step(model, torch.optim.AdamW(model.parameters()), loss_fn, (ids, labels), updated={"blocks.1.mlp.up.weight"})
g.macs, g.c_macs              # the GEMMs' MACs; the step's share of the size of C as te_core.flops counts it
g.role_macs("using")          # using / producing / neither (structure.py's docstring has the definitions)
g.phi(D)                      # the share of the step's MACs in using GEMMs with at least D updated MACs per token upstream
run = capture_run([(1, g0), (999, g1)])       # (steps, StepGraph or a thunk that captures it)
annotate(run.record())        # te_core.annotate reads a captured run as it reads a counted one
lower(g)                      # optional: gates per Definition and the share of MACs lowered
~~~

`capture_fn(step_fn, model, batch, optimizers=...)` traces any step function (MeZO's two forward passes, a hand-written
update); `phase("name")` labels a stretch of it. `updated` is the harness's set of parameters that differ from θ when the
step reads them (default: the ones the step trains); `schedule_updated` gives it for each step of a schedule, and
`StepGraph.with_updated` reclassifies a graph without tracing it again.

## Against te_core.flops

`RunGraph.record()` is in `te_core.flops.Counter.record()`'s form, and both count the op sets `te_core.flops` names
(`ELEMENTWISE_NAMES`, `SCALED_NAMES`, `FUSED_OPTIMIZER_MACS`, `LINALG_NAMES`). On the toolsmith's tiny tied-head model and the tests'
two-layer decoder (full training, top layer only, LoRA with its adapters in θ or new), the capture's record and the
Counter's runtime record of the same three steps agree on every step's tokens, GEMM and elementwise MACs, updated MACs,
updated MACs per token, depth D, depth profile, bits and φ. A capture has no values, so the records' weight digests,
which counterexample checks read, come from the Counter alone.

## Refusals

`CaptureRefused(reason, op, message, where)` names the op and the user's source line:

| reason | the step |
|---|---|
| `dynamic_control_flow` | reads a tensor's value into Python (`if x.sum() > 0`, `.item()`), or calls `torch.cond`/`while_loop` |
| `data_dependent_shape` | makes a shape from values (`x[mask]`, `nonzero`) |
| `custom_kernel` | calls an op with no fake kernel or no autograd formula, or leaves a non-ATen op in the graph |
| `uncounted_op` | runs a product this package doesn't count (a fused RNN or attention kernel) |
| `batch_dependent` | has a graph that changes with the batch or sequence size (a loop over examples) |
| `trace` | fails to trace for any other reason, with torch's message |

## Limits

- Upstream counts are per GEMM node: exact where a norm, softmax or contraction mixes a token's row between two using
  GEMMs, an upper bound otherwise (an up projection feeding its own layer's down projection).
- The traced optimizer step is its first, with its state created in the graph; later steps run the same ops.
- Attention is traced on SDPA's math backend, so its scalings show as elementwise multiplications, which a flash kernel
  doesn't run.
- Token axes are found by tracing again with each axis resized. By default every dim of an integer input is a token
  axis, and every dim of a float input but its last (its features); give `token_dims` for anything else (an image's
  channels and pixels).
