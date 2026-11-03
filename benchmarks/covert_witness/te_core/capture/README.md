# te_core.capture: a training step's circuit structure, read from its PyTorch code

A search agent writes ordinary PyTorch training code. `capture_step` traces one whole training step of it (forward,
autograd's backward and the optimizer's update) into an ATen graph with shapes and dtypes and no values, and reads from
that graph what te_core scores: every multiplication's MACs, the role of each GEMM, its token dims, the updated MACs
upstream of it, and the training formalism's quantities measured on the step as a circuit of values and gates
(`circuit.py`: σ and the affected share, the tokens consumed, τ, a, the profiles u_in(x) and u(x) with the no-update
share t, and Δ_grad).
Updated means update-dependent: computed, through any path, from a value the run produced and
carries across steps (the trained weights, optimizer state, any buffer it writes from them), so a model scores the same
however its circuit is written (LoRA merged or not, a tied lookup, a cache of θ). A run's circuit C is the sum of its distinct step graphs (`capture_run`), so the size of C is exact in
MACs without building a gate. The trace is `make_fx(tracing_mode="fake")` under `FakeTensorMode`, on a fake copy of the
model and a fresh optimizer of the same class and groups, so the caller's objects are never touched and nothing runs.

~~~python
from te_core.capture import capture_step, capture_run, schedule_updated
from te_core.capture.lower import lower

g = capture_step(model, torch.optim.AdamW(model.parameters()), loss_fn, (ids, labels), updated={"blocks.1.mlp.up.weight"})
g.macs, g.c_macs              # the GEMMs' MACs; the step's share of the size of C as te_core.flops counts it
g.role_macs("updated")        # updated / producing / neither (structure.py's docstring has the definitions)
g.phi(D)                      # the share of the step's |C| in updated MACs with at least D updated MACs per token upstream
g.cut_bits                    # the narrowest cut in bits between the gradient side and the values carried to the next step
g.formalism                   # a recorded program's measured step: te_core.capture.circuit.measure's quantities
run = capture_run([(1, g0), (999, g1)])       # (steps, StepGraph or a thunk that captures it)
annotate(run.record())        # te_core.annotate reads a captured run as it reads a counted one
lower(g)                      # optional: gates per Definition and the share of MACs lowered
~~~

`capture_fn(step_fn, model, batch, optimizers=...)` traces any step function (MeZO's two forward passes, a hand-written
update); `phase("name")` labels a stretch of it. `updated` names the parameters that differ from θ when the step reads
them (default: the ones the step writes, and every buffer written from them); `schedule_updated` gives it for each step
of a schedule, and `StepGraph.with_updated` reclassifies a graph without tracing it again.

## Against te_core.flops

`RunGraph.record()` is in `te_core.flops.Counter.record()`'s form, and both count the op sets `te_core.flops` names
(`ELEMENTWISE_NAMES`, `SCALED_NAMES`, `FUSED_OPTIMIZER_MACS`, `LINALG_NAMES`). On the tests' two-layer decoder (full
training, top layer only, LoRA with its adapters in θ or new), the capture's record and the Counter's runtime record of
the same three steps agree on every step's tokens, GEMM and elementwise MACs. The structure te_core 4.0 scores (updated
MACs, depth, the formalism's quantities) is the capture's: the Counter's rows keep the weight-operand rule it had before
4.0. A capture has no values, so the records' weight digests, which counterexample checks read, come from the Counter
alone.

## The measured circuit

`te_core.capture.script` traces a recorded program's measured step and builds it as a `StepCircuit`: one vertex per
value of the step (and every value it carries from the step before), its gates priced by `te_core.gates`, its phase,
whether it depends on θ' (the carried values) and on the step's tokens, and the positions of each token input it
reads. `circuit.measure` reads the formalism's quantities from it, each defined in the module's docstring; `script
--circuit F.npz` saves it, and `python -m te_core.capture.circuit F.npz` measures a saved one again.

The measured step's quantities also keep its matrix products as `te_core.io_bound`'s bound reads them
(`formalism["products"]`, which a saved circuit keeps; `products.rows`, which reads a step captured by `capture_step`
or `capture_fn` too): each
GEMM's dims and token dims, whether each operand depends on θ′, the width it is passed at, the bits a block pays an
entry of it at the console's k and the chain that makes one whole, identical rows merged with their count, and under
`at` the same rows at each other k of the console's dial (`products.rows_on_dial`, `products.DIAL_K`), since the rates,
kinds and chains move with k. A recorded
program reads a weight through the version the previous step's update wrote, so these rates and chains are keyed by
the capture's values, not by graph nodes as `io_bound`'s are. The summation order is `unknown`: a GEMM is one ATen op,
summed as its kernel sums it. `"capture"` names the version of these rows (`products.CAPTURE`);
`te_core.formalism.products_of_trace` reads them.

A Python number the program reads from a tensor has no value in the trace, so whatever Python decides from it is the
stand-in's. The recording lists each such read-back by its source line and steps, in `formalism["data_dependent"]` (a
comparison, a truth test or an integer conversion of a read-back float; an integer or boolean read-back; `numpy()` of a
tensor) and `formalism["carried_scalars"]` (an op reached by read-backs from two or more earlier steps: a Python float
folded over steps, whose Δ_grad grows with the step), and a saved circuit keeps both lists. te_core's formalism refuses a
trace with either list nonempty, or without the lists, and never scores it. A program keeps a running value as a device
tensor (a 0-dim fp32 tensor updated in place) and branches with `torch.where` or `clamp`, which the graph carries.
The recorder's session has `current_step()` and `progress_bucket()`, as the harness's does.

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

- Upstream counts are per MAC node: exact where a norm, softmax or contraction mixes a token's row between two updated
  nodes, an upper bound otherwise (an up projection feeding its own layer's down projection).
- Dependence follows the graph's dataflow: a value computed from θ, constants and token data alone is a cache, even
  when the program carries it from step to step. A Python scalar read from a tensor carries its tensor's dependence only
  in a recorded program (`te_core.capture.script`), and is listed there wherever Python decides anything from it.
- The cut prices a value at its dtype's width: a sign kept as a float costs 32 bits per element, as a bool one.
- The traced optimizer step is its first, with its state created in the graph; later steps run the same ops.
- Attention is traced on SDPA's math backend, so its scalings show as elementwise multiplications, which a flash kernel
  doesn't run.
- Token axes are found by tracing again with each axis resized. By default every dim of an integer input is a token
  axis, and every dim of a float input but its last (its features); give `token_dims` for anything else (an image's
  channels and pixels).
