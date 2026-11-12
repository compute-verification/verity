# circuit_properties

Structural properties of a circuit's counted operations 𝒥, measured exactly on declared Boolean circuits: the fan-out
a_out, the fan-in a_in, the count |𝒥|, the smallest product subcircuit s_×, and which products a partition computes
wholly inside one block. Each is defined for an arbitrary Boolean circuit and a designation of its counted operations.

## The circuit format

A circuit is a Program in `verity-ir/descriptor/v1` (`verity/circuits/PROTOCOL.md`) in the Boolean profile (§9),
read through its canonical layout (§6): a `verity.circuits.program.Program`, or `codec.decode_program`'s result. It is
the format the inference-only policy's check reads gate by gate
(`verity/protocols/inference_only/verifier/nci/lean/NciCheck/Check.lean`) and `Nci.Cover.Circuit` formalizes (`security/Definitions/Nci/Cover.lean`, `BitView.lean`), and the one the served workloads lower
to on bits. A word circuit has no per-product gates (one tensor-core step gate computes 16 products), so s_× and
whole-in-block are defined only on the Boolean lowering; `ops.NotSeparable` refuses a word op.

## The designation and the properties

`ops.Designation(sites, entries)`: a `Site(op, x, w, lanes, within)` names a Definition (regex over its id) whose
activations compute counted products, lane k multiplying activation operand `x[k]` by weight operand `w[k]`, counted
only under an ancestor matching `within`. `entries` names the Definitions whose returned leaves are the outputs
fan-in is read at. `references.GEMM_PRODUCTS` is the GEMMs' products: each lane of `AmpereBF16TcDot16_v3` under a
`Gemm`, fan-in at each `GemmCoordinate`.

- m(p), product p's subcircuit: the gates of its activation that read lane k's operands and nothing else
  (`ops.OpShape`). They are disjoint, nonempty and computed, as `Nci.Cover.MulBlocks` requires.
- |𝒥| (`products`), s_× (`s_x`): the products, and the fewest gates in any m(p).
- a_out: the most counted products any one activation value (a gate) feeds. Constants are counted apart
  (`constant_leaves`).
- a_in: over the returned leaves of every entry activation, the counted products inside it with a gate of m(p) the leaf
  is reached from; the least over leaves some product reaches.
- `whole(program, designation, Blocks)`: the products whose m(p) lies inside one block, per block, and those split.
  `Blocks(runs)` is a partition as runs of gate indices (`Blocks.from_owner` from each gate's block, `Blocks.by_root`
  one block per root activation).

`measure` computes all of them hierarchically, once per Definition, so a served layer of 10^9 to 10^12 gates measures
in seconds. `flat` computes them gate by gate from the definitions; the tests hold the two equal on synthetic circuits
covering every node form, the Boolean GEMM, and b1's layer at tiny shapes. `measure.Ambiguous` refuses the one case the
summaries can't count exactly (one value bound to two activation leaves above the op); `flat` measures it.

## The min-cut bound's work

`work.read(program, k)` reads off a word Program what `min_cut.bound` takes of a circuit at a policy's k: |C| in
Boolean gates (each primitive at its registered Boolean version's, `work.boolean_gates`), every product with its dims,
each operand's kind, rate, source bits and β, the charged products' MACs N and output wires O, and the arrays they
multiply, which give F. The rules are the training benchmark's reading of a trace (`capture.products`' rates,
`training_slowdown.work.he_of_rows`' β), each premise a Program doesn't state taken on the side of the smaller bound.
The products are a reader-side table of registered Definitions (`work.PRODUCTS`: the reference registry's `LinearRef` and `DotRef`),
and the lookups another (`work.LOOKUPS`, the gathers). The update θ′ is a scan's carried words that a fresh input
reaches; every input is free like θ (the carry is iterated to a fixed point, so a loop's iterations cost what its
distinct contexts do). A product is charged where both operands depend on θ′ in every word, each is specific or a weight
(`work.may_repeat`: an operand that reads the current inputs and is not specific may repeat its rows, however the
Program writes them), the two are no one array and not two activations; `Work.left_out` holds the rest by why. Nothing is stated
beside the circuit. A gather's rows by the batch stay a gather's rows until one rule makes them specific: a softmax over
a constant whose rows (or columns) differ, as a causal mask's do, through one-to-one ops and products by distinct rows.
The rule rests on one premise, `SHARED_PREFIX`'s: past each row's first position, no two positions of the batch hold
equal words. Its limits: a Program with no scan has no bound; the rule doesn't reach a softmax's backward or a table
the Program doesn't state (a rope table), so dQ·Wq and dK·Wk stay gathered; and a dense step, a batch of values into
linear layers, is charged nothing, since two rows of values may be equal, as two tokens may, and a product of its rows
by a shared weight would then read one pair of values twice. Two more premises hold the reading below the work, each
held as a fact by a test: every carried word is a parameter's or the optimizer's, so rows carried into the next
iteration are charged as weights, and θ′ is by data flow, so an update that changes nothing still makes θ′.
`work.bound(work, c)` is the most blocks over its
sub-works, through the calculator's `min_cut` at each sub-work's least β, none from `HEAD_K` on. The module docstring
has the rules and which way each moves the bound; the training benchmark's `tests/test_program_reading.py` holds a
training loop's reading beside its trace's, never above it.

## Interface

```python
from circuit_properties.measure import measure, whole, Blocks
from circuit_properties.references import GEMM_PRODUCTS, layer, configs

m = measure(layer(configs()["qwen25-1b5"][0]), GEMM_PRODUCTS)
m.products, m.s_x, m.a_out, m.a_in        # 46792704, 681, 17920, 1536
```

## Reference circuits

`references.layer(C, T)` is one b1 layer (σ(`LayerPre_v1`), the Boolean `Attention_v3`, σ(`LayerPost_v1`)) at T
positions; `references.configs()` lists the served configurations and two shapes with no served workload (Qwen2.5-7B,
Llama-3.1-405B), labeled as constructed, whose shapes are verity/protocols/inference_only/rho's (its `models.toml`, read through
`rho.instances.models`). The circuits are forward layers: the Programs declare no training (backward, optimizer)
circuit.

    uv run --extra torch-cpu verity bench circuit_properties [--models NAME,...] [--T T] [--out DIR]

measures each and sets a_out, a_in and |𝒥| per position beside verity/protocols/inference_only/rho's `fanout`, `legs` and layer products
(its `models.toml` row, and `derived_fanout` / `derived_legs` of the config's matrices), checks whole-in-block under
`Blocks.by_root` and a cut at the middle gate, and writes `properties.json`.

## What it reads, what it writes and what loads it

The package reads `verity.circuits` (Programs, the canonical layout) and `verity.definitions` (the Boolean GEMM and
the served b1 Definitions on bits). For a model's shapes it loads the slowdown calculator where it sits, with
`verity/protocols/inference_only/rho` put on `sys.path`: `rho.instances`, and `rho.params` for the registry, which
loads the inference-only policy's calibrations (`verity/protocols/inference_only/nci/calibrations.json`). The suite
names what it reads of both among its `inputs`.

The runner prints one JSON line a circuit and, with `--out DIR` (or `$OUT`), writes `DIR/properties.json`; without
either it writes no file.

The training benchmark's suite loads `work` by path, to hold it beside its own reading. What it measured of a training step (`s_×`, `a_out`, `a_in`) sits in
`verity/protocols/inference_only/rho/data/circuits.toml` as data, each number with its source.

The suite: `uv run tools/verity/check/suites.py benchmarks/circuit_properties`.
