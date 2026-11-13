# The experts' key wires and the tables' declarations: growth and forms

Rank 0, GLM-5.3's Build at TP8 (`benchmarks/compile_cost/run.py` `registered`), `--block w13=1 --block w2=32`, eager,
one request. All runs are from source `30dac672d2bb682dd3a81c80bcf4915f2577fd2e` (the joint branch's head), by
`p3_keycost.py`.

## Growth (item 2)

### Measured at 4 layers (one MoE layer, three dense)

Run `r20261113-041114-3cb4`.

| prompt ids, outputs | steps | key wires | by step | radix | tables | roots | openings | description |
|---|---|---|---|---|---|---|---|---|
| 43, 2 | 2 | **12,096** | 6,048 / 6,048 | (43,) ×6,048, (1,) ×6,048 | 756 | 1,512 | 756 | 4,417,654 B |
| 43, 3 | 3 | **18,144** | 6,048 / 6,048 / 6,048 | (43,) ×6,048, (1,) ×12,096 | 756 | 2,268 | 1,134 | 6,291,540 B |
| 86, 2 | 2 | **12,096** | 6,048 / 6,048 | (86,) ×6,048, (1,) ×6,048 | 756 | 1,512 | 756 | 10,098,369 B |

These are the number of `TableRead` (roots) and `TableOpening` (openings) entries. The counts per expert table are:
- 4 roots at 2 steps and 6 at 3 steps, so two a step: the root and the domain id;
- one opening a step;
- every table read in every step.

Bytes of the key wires: 2,721,147 at (43, 2), 4,083,283 at (43, 3) and 2,724,619 at (86, 2).
- A decode step adds 1,362,136 B.
- The prefill step costs 1,359,011 B at 43 ids and 1,362,483 B at 86 ids. The only difference is the radix's digits in
  the text.

### The count, as a formula

key wires = L_moe × S × TOPK × Σ_r B_r

- L_moe is the number of MoE layers: 75 of 78, and 1 of the 4-layer Program.
- S is the number of steps in the window.
- TOPK is the number of slots: 8.
- Σ_r B_r is the sum, over an MoE layer's four read templates, of each one's blocks:
  - w13: N/n = 512/1 = 512;
  - w13's scales: N/G = 512/128 = 4;
  - w2: N/n = 6,144/32 = 192;
  - w2's scales: N/G = 6,144/128 = 48.

  The sum is 756, which is also the MoE layer's number of tables.

A read is unit `(b TOPK + k) T + t`, and the run of each (block b, slot k) is one wire of radix (T,). T, the step's tokens,
is a digit of the wire, not a count of wires. So the count is 6,048 × L_moe × S, which is 453,600 a step at 78 layers.

The tables' declarations:
- tables = 756 × L_moe, **once a window**: 756 at 2 and at 3 steps;
- roots = 2 × 756 × L_moe × S;
- openings = 756 × L_moe × S.

`_table_reads` declares the `Table` once (`setdefault` on `(lo, H, block)`), and its two `TableRead`s and one
`TableOpening` once a (table, node). A table has one reader node a step, its layer's MoE node, so they repeat **every
step**.

### The two sentences

**"Counted by step, not by token": it holds for the experts.** The key wires, roots and openings are equal at 43 and at
86 prompt ids, and grow by exactly a third from 2 steps to 3.

It does not hold for the whole window. At 86 prompt ids the description is 10.1 MB against 4.4 MB at 43. Wire patterns
go 13,238 → 36,100, and before `coalesce` 19,158 → 737,762, all of the growth in two of my other families:
- BatchedGemm: 4 → 16,387 patterns;
- SparseMlaSplit: 479 → 6,599 patterns.

The cause, from a probe of those wires (`p3_tokwires.py`, run `r20261113-042057-0248`):
- **BatchedGemm.** At 86 ids, BatchedGemm{K=512}'s `x` reads its prefill tokens from two writers:
  - tokens 0–63 from `AllGather_v1{WORLD=1,N=512}`;
  - tokens 64–85 from `AllGather_v1{WORLD=8,N=64}`, at base 688.

  The read's units are token-fastest, so each of its 8,192 runs at 4 layers (2,048 a layer, all in the prefill step)
  needs one wire a writer: 8,192 + 8,193 patterns. At 43 ids one writer covers every token and the runs coalesce into
  one wire. Each run's source is the same, a fan-out.
- **SparseMlaSplit.** At 86 ids it takes its split path: `SparseMlaCombine_v1{NSPLIT=2}` reading `SparseMlaSplitEndCols_v1`
  per column group C, 92 patterns a group (88 of them in the prefill step).

Both families are mine, and nearly all of the growth is in the prefill step. They are the non-expert half of the window
and outside items 2 and 3. Scaled ×78 layers, the BatchedGemm part alone would be about 320,000 patterns at 86 ids. I
haven't measured it there. TOKWIRES2

**"About 454,000 key-wire patterns a step a rank": right** (453,600).

**"About 143 MB of description a step a rank (104 + 39)": wrong in the 39, right as an average over a 2-step window.**
- The 104 MB of key wires a step holds: 207,920,793 B / 2 steps at 78 layers, and the prefill and decode steps differ by
  0.2 % at 4 layers.
- The 39 MB is half of the 77.8 MB of declarations at 2 steps. But 12.0 MB of those are the 56,700 expert tables, once a
  window. What repeats each step is the roots and openings: 113,400 roots (20.8 MB) and 56,700 openings (12.1 MB), so
  32.9 MB a step.

The figure to give is **137 MB a step a rank plus 12 MB once a window**: 137 S + 12 MB for an S-step window, which is
286 MB at S = 2 as measured. The bytes per entry are 78-layer measurements:
- a key wire 229.2 B;
- a root 183.8 B;
- an opening 212.8 B;
- a table 211.2 B.

The 4-layer per-step deltas × 75 give 134.5 MB a step. Entries at 78 layers are about 2 % longer (layer indices).

## Forms that cut them (item 3), on paper

Fixed for every row: 78 layers, 75 MoE layers, S = 2 (the census's window), and the bytes per entry above. Each row
gives the experts' key wires and declarations only. The window's other 38,122 wire patterns (10.8 MB at 78 layers) are
the same in every row and left out.

| form | key wires | roots / openings / tables | bytes at 78 layers, S = 2 | bytes a step added | whose file |
|---|---|---|---|---|---|
| today: table-major `(b TOPK + k) T + t` | 907,200 | 226,800 / 113,400 / 56,700 | 285.7 MB | 136.9 MB | |
| A. slot-major `(k B + b) T + t` | 4,800 | 1,814,400 / 907,200 / 56,700 | 539.7 MB | 263.8 MB | ours (`node_window_sparse.py`) |
| B. one declaration a table for every step's node | 907,200 | 113,400 / 56,700 / 56,700 | ≈ 255.6 MB | ≈ 106.7 MB | vllm's (`interface.py`), then ours |
| C. a table family picked by a digit, with slot-major order | 4,800 | 1,200 / 600 / 300 families | ≈ 1.6 MB | ≈ 0.75 MB | vllm's (`interface.py`), then ours |

### A. Slot-major order, nothing else

**What changes.** Only our `expert_gemm` (`node_window_sparse.py`): the read is unit `(k B + b) T + t`, and the GEMM's
wire from the read's `o0` takes the new strides. That wire is still one wire a node: the strides over (t, k, b) go from
(1, T, TOPK T) to (1, B T, T).

**The key wire.** Slot k's run is B T units, contiguous, reading router output k at `(B, T)` steps `(0, step)`. A step of
0 is a fan-out, which `_Pattern` already allows. That gives TOPK wires a read template a node: 4,800.

**The declarations.** Table b's units in a node are now TOPK runs of T apart, and a `TableRead` or `TableOpening` covers
one contiguous run. So the roots go to 2 TOPK and the openings to TOPK a (table, node): 8× today.

**Checks:** none changes. `check_wires`, `check_inputs`, `coalesce` and `ordered` take it as it is.

**Verdict.** It trades 208 MB of key wires for 527 MB of declarations. Not on its own.

### B. One declaration a table, shared by every step's node

A table has exactly one reader node in a step, its layer's MoE node. So "a step's nodes share one declaration a table"
can only share across steps; this is that form.

**What changes, in `interface.py` (vllm's).** `TableRead` and `TableOpening` take a list of unit runs `[(at, units), ...]`
instead of one run. A list, not a stride: a node's units per table are M = TOPK T, and T differs between the prefill
step and the decode steps.

**What changes, ours.** `_table_reads` appends a run to the table's existing declarations instead of new ones.

**The cost.** One root pair and one opening a table, whatever S. Each added step adds one run (about 16 B) to each of
170,100 declarations, about 2.7 MB. The key wires are untouched: 104 MB a step.

**Checks.** `check_inputs` (each run's cover, its disjointness and its key fed), `input_count`, `registered_reads` (SPOT's
check of a registration), `listed_reads`, and `declarations` / `window_of_record` (parsing). `check_wires`, `coalesce`
and `ordered` don't change.

**The Lean.** `NciCheck.RecordText.inputsOf` makes one `rect` a run, so its quadratic `pairwiseDisjoint` sees as many
entries as today. Only the bytes drop.

### C. A table family picked by a digit, with slot-major order

**What changes, in `interface.py` (vllm's).**
- `Table` gains `blocks`: B tables, member b the parameter `weights/<label>@<H>/b<b>/table`. These are the same B
  registered parameters as today, so `expert_tables`, the registration and serving's committer don't change.
- `TableRead` and `TableOpening` gain `radix` and `digit`: unit u of `[at, at + units)` reads member d[digit] of u − at in
  mixed radix `radix`.

**What changes, ours.**
- The read's order is slot-major, `(k B + b) T + t`, with radix (TOPK, B, T) and digit 1.
- A read template a node declares one root pair and one opening, over all B TOPK T units.
- The key wires are TOPK a read template a node, at radix (B, T) and steps (0, step), as in A.

**The cost at 78 layers, S = 2:**
- 4,800 key wires;
- 1,200 roots and 600 openings;
- 300 families (4 a MoE layer).

That is about 1.6 MB. A step adds 2,400 key wires, 600 roots and 300 openings, about 0.75 MB.

**Checks.**
- `check_inputs`: a declaration's member varies across its units; its cover and its key's feed don't change.
- `input_count`: a family member is pinned or free by its own root.
- `registered_reads`: SPOT's check of a registration resolves a member's parameter by its digit.
- `listed_reads`, and `declarations` / `window_of_record`.
- `check_wires`, `coalesce` and `ordered` don't change: a zero step is already a fan-out.

**The Lean.** `NciCheck.RecordText` (`Table`, `Opening` and `fill := .table` naming a member by digit),
`security/Proofs/Properties/Nci/Record{Inputs,Kinds,Fresh}.lean`, and the Lean side of SPOT's registered check.

**Set aside:** one table a stack keyed b E + e. It needs a new Definition, a registration change and 8 more levels of
path in every opening.

### What each form costs the readers of a window's description on main

These are the readers found by search.

| reader | what it walks | today | A | B | C |
|---|---|---|---|---|---|
| `Spot.Wires.check` (Lean; `pairwiseDisjoint`, Wires.lean:192, quadratic in wires) | wires | 945k wires, ≈ 4.5×10^11 pairs | 43k, ≈ 9×10^8 | as today | 43k, ≈ 9×10^8 |
| `NciCheck.RecordText.inputsOf` (Lean, the inference-only policy's check; `pairwiseDisjoint` over declarations, RecordText.lean:369) | wires + reads + each root + each opening's 3 ports | ≈ 1.51M, ≈ 1.1×10^12 pairs | ≈ 4.58M, ≈ 1.0×10^13 | ≈ 1.51M (one `rect` a run) | ≈ 46k, ≈ 1.1×10^9 |
| `registered_reads` (SPOT's check of a registration), `input_count` (the policy's count: `block_counts.py`, `inference_blocks.py`, `prove_path`), `listed_reads`, `of_records` / `window_of_record` (Python, linear) | roots and openings | 340k | 2.7M | 170k | 1.8k |
| `check_wires`, `check_inputs`, `ordered`, `coalesce` (Python, at build) | wires, declarations | 6.8 s / 3.7 s / 3.1 s / 9.2 s | wires ÷ 22, declarations × 8 | declarations ÷ 2 | wires ÷ 22, declarations ÷ 190 |
| compose (`remote/verifier/composition`) | no wires or openings | no change | no change | no change | no change |

All "pairs" counts are n²/2, the quadratic check's tests at 78 layers, S = 2. The Lean readers and their proofs change only
under B and C, and only in how they parse and expand a declaration.

### Which one

**C.** It is the only form that cuts both the key wires and the declarations: 286 MB → 1.6 MB, and 137 → 0.75 MB a step.
It also takes the quadratic Lean readers from about 10^12 tests to about 10^9.

It costs one field each on `Table`, `TableRead` and `TableOpening` in vllm's `interface.py`, and their readers. Its
other half, slot-major order, is ours alone. A shows that slot-major order without C's declarations makes the
declarations 8× worse, so the two land together.
