# circuit_properties

Structural properties of a circuit's counted operations 𝒥, measured exactly on declared Boolean circuits: the fan-out
a_out, the fan-in a_in, the count |𝒥|, the smallest product subcircuit s_×, and which products a partition computes
wholly inside one block. Each is defined for an arbitrary Boolean circuit and a designation of its counted operations.

## The circuit format

A circuit is a Program in `verity-ir/descriptor/v1` (`verity/ml/circuits/PROTOCOL.md`) in the Boolean profile (§9),
read through its canonical layout (§6): a `verity.ml.circuits.program.Program`, or `codec.decode_program`'s result. It is
the format the inference-only policy's check reads gate by gate
(`verity/core/protocols/network/nci/lean/NciCheck/Check.lean`) and `Nci.Cover.Circuit` formalizes (`Security/Definitions/Nci/Cover.lean`, `BitView.lean`), and the one the served workloads lower
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

`measure` computes all of them hierarchically, once per Definition, so a served layer of 10^10 to 10^12 gates measures
in seconds. `flat` computes them gate by gate from the definitions; the tests hold the two equal on synthetic circuits
covering every node form, the Boolean GEMM, and b1's layer at tiny shapes. `measure.Ambiguous` refuses the one case the
summaries can't count exactly (one value bound to two activation leaves above the op); `flat` measures it.

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
Llama-3.1-405B), labeled as constructed. They are forward layers; the training step is `TrainStep_v1` (below).

    uv run --extra torch-cpu benchmarks/circuit_properties/run.py [--models NAME,...] [--T T] [--out DIR]

measures each and sets a_out, a_in and |𝒥| per position beside benchmarks/rho's `fanout`, `legs` and layer products
(its `models.toml` row, and `derived_fanout` / `derived_legs` of the config's matrices), checks whole-in-block under
`Blocks.by_root` and a cut at the middle gate, and writes `properties.json`.

## The upstream certificate

`Nci.Capability.Consumers.Upstream` (#6245's Lean, not yet on `main`) says every required gate of a
consumer reaches, inside its region, a gate each of its finishing sets reads: a block that finishes it imports from its
region or recomputes its required gates, which gives the local cap h_w·(k + c/g), g the fewest required gates.

- `upstream`: the certificate on an explicit wiring, as the Lean states it (`Consumers` with `G_sub` and `A_θ`,
  `of_uses`, `of_outputs`, `MulBlocks`, `holds`, overlap weights), and `required`, the largest certified G of a region.
- `regions.required_size`: |`required`| for a region that is a union of whole activations, along the call hierarchy,
  for regions of 10^9 gates and more; the tests hold it equal to `required` on the explicit wiring.
- `certcheck`: a training step's consumers. Uses (`Consumers.ofUses`) are one token's input to one layer (`LAYER`: its
  forward GEMMs' products, or its input-gradient GEMMs'), or, finer, one GEMM's input (`GEMM`); outputs
  (`Consumers.ofOutputs`) are a weight gradient coordinate's bits. Each consumer's region, classes and verdict (|A|, |G|,
  and why G is empty), the regions' overlap weights, and where the step's counted products are.
- `training`: `TrainStep_v1` as a Program, at smollm2-135m's shape or at a scaled shape the registry refuses in five
  places, which `training.scaled` lifts.

      uv run benchmarks/circuit_properties/run_certcheck.py [--shape scaled|real] [--tokens ..] [--layers ..] [--out DIR]

runs every consumer of the scaled step (or the given tokens and layers of the real one), the explicit cross-checks
(`upstream.holds` on a region listed gate by gate), and writes `certcheck.json`.
