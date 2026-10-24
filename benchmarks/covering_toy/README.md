# Toy coverings of a real training graph

The cheapest covering of a tiny model's actual SGD graph by inference-like blocks, against the covering bound
(`Security/Models/Nci/Guarantees/CoveringTwoCosts.lean`) and the production bound (section 4 of
note:console/20261024T1810Z-finding-production-bound-red-team). It is worker C2 of the training-vs-inference program
(note:console/20261024T1900Z-draft-training-vs-inference-program), and the instances and solvers are meant for C3's
value-splitting attacks.

Nothing in `verity` reads this directory.

## The model

A gate is one word. The graph is `steps` SGD steps of an MLP with layer widths `dims` on `batch` fresh tokens a step,
squared error, every multiply, add, subtract, ReLU and ReLU-derivative its own gate (`toygraph.mlp`). The step-1
weights and the learning rate are registered (`θ`); data are unregistered sources; updated weights are computed gates.
`Graph.evaluate` reproduces numpy SGD exactly (`tests/test_toygraph.py`).

Lean's vocabulary, word for word (`Security/Definitions/Nci/Covering.lean`):

| Lean | here |
|---|---|
| a gate of `C` | one word: `G.op[g]`, `G.preds[g]`, `G.tag[g]` |
| `θ` (registered) | `G.theta[g]`: the step-1 weights and `lr` |
| `in(C)` | sources: `θ` and the data (`G.op[g] == "data"`) |
| `(s, k)`-inference block | `cover.Block(computed, inputs)`, `S = computed ∪ inputs`, `|S| <= s` (registered inputs count), `|inputs \ θ| <= k` |
| `Covers` | `cover.validate(G, blocks, k, s)` |
| `m_{i,j}`, `P` | `G.mul[(slot, token)]`: the forward product of updated slot `i = (layer, row, col)` with token `j = (step, b)`, every step after the first |
| `T_j` | `G.token_gates[token]` |
| activation-chaining at `(D, c_a = 1)` | `bounds.upstream` (an activation is one gate, so its own cut and `c_a = 1`) |
| parameter-hardness `c_w(s)` | `sweep.cw_measured` over `cover.min_inputs_for_slots` |

`G.values` are the production bound's counted values: every forward pre-activation (the toy's maps are square, so
`min(d_in, d_out)` per token per map is all of them).

`toygraph.inference(dims, tokens)` is the baseline: the same model's forward pass alone, every weight registered, so a
training covering's `rho` can be read against inference's at the same `k` and `s`.

The graph is word level only. A bit-level variant (each word a fixed-point adder and multiplier circuit) is 30-50 times
larger, past what the exact solvers close; build it with `Graph.add` as for splitting attacks below.

## Solvers (`cover.py`)

- `greedy`, `lns`: heuristics with recomputation; `partition_topo`, `partition_bisect` (recursive Kernighan-Lin, a
  METIS-style partitioner): partitions, as baselines.
- `colgen`: column generation for the set-cover LP over every `(s, k)`-block, with CP-SAT pricing. Its `lb` is a
  proven lower bound on the optimum, and its integer master over the generated columns gives a covering.
- `exact`: the compact CP-SAT covering model (slower than `colgen` on these instances; kept for cross-checks).
- `block_capacity`: the most target gates one block computes, a valid per-block cap.

Every solver's output goes through `validate`. Each takes `targets` to cover only a subset (the counted values, `M*`).

The bounds (`bounds.py`): `lean_two_costs` (`CoveringBoundTwoCosts`: `|M*| / ((k / c_w)(k / c_a) + k)`),
`lean_additive` (`CoveringBoundAdditive`: `(k / c_w)(k / c_a) / 4 + k` per block), `production` (`beta |V| / k`) and
`trivial` (`|inner| / s`).

## Running

~~~sh
uv run --no-project --with ortools --with scipy --with numpy --with networkx \
    python benchmarks/covering_toy/sweep.py --out OUT [--instance NAME ...] [--jobs J --workers W --budget S]
~~~

`sweep.INSTANCES` names the instances. Per instance, `k` and gate budget (`eta1`: matched to honest inference on `k`
words; `eta2`: twice that; `inf`: none), it solves the whole covering, the covering of `V` and of `M*`, and writes
`OUT/<instance>.json` (each solution's blocks as `[computed, inputs]` gate lists, so a covering can be reloaded) and
`OUT/table.md`. `M_ONLY` instances (the deep, wide one, where the Lean bound isn't vacuous) solve only `M*`. `--ks`,
`--modes` and `--problems full,V,M` narrow a sweep; `--merge DIR ...` combines several sweeps' outputs into `OUT`, each
bracket tightened only by sound arguments (`sweep.merge`); `--caps` runs only the per-block cap test (`sweep.caps`), and
`--inference` only the inference baseline (`sweep.inference_baseline`). On a
pod: `research run --on vy-nebius-cpu-5 --pool research --as LANE --project verity --source . --cwd source -- bash -c
'cd benchmarks/covering_toy && uv run ... sweep.py --out "$OUT"'`.

The table's flags name the section-4 assumption a covering breaks: `P-total` (fewer blocks than `|V| / k`), `P-block`
(one block computes more than `k` values), `L-c_w=1` (fewer `M*` blocks than the Lean bound with `c_w = 1`, which only a
`c_w < 1` may do), `L-measured` (fewer than the Lean bound at the measured `c_w(s)`: a solver or statement bug).
`sweep.why` reads the densest such block's gates and says which assumption it breaks.

## For splitting attacks (C3)

The 24 Oct sweep over every instance, merged, is
`art:2f1eae928df7e08433733ba23a4ca2a409b2c02b5fbc2db42ecb90dc64a4d1a5` (its `README.md` says what each part is).

`sweep.build(name)` builds an instance; a sweep's `OUT/<instance>.json` holds every point's best coverings, and
`sweep.load_cover(G, r["full"], r["k"], r["s"])` turns one back into validated `cover.Block`s (`r["V"]` and `r["M"]`
cover `G.values` and `M*`: pass those as `targets`).

At word level a value is one gate, so a block holds it whole and locality can't fail: a splitting attack needs a graph
in which a value is several gates. Build it with `Graph.add` (low and high limbs, or additive shares, with the gadget
that recombines them where the recipe reads the value); `validate`, the solvers and `bounds` work on any `Graph` whose
gates have at most two predecessors and whose `mul`, `values` and `token_gates` say what the bounds count. The attack
beats the bound when its covering of a refined graph needs fewer blocks than the word-level optimum at the same `k`,
counted in the same units.
