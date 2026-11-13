# The experts' key wires and the inference blocks: today's, form C's, and how `NciCheck.Record` decides

Rank 0's window, 4 layers (one MoE layer), 43 prompt ids, the families frame, dense and sparse (run
`r20261113-041114-3cb4`'s settings), built at the join head `eee9c706eec20d77f343039f7042263b3895fce5`. Marks: [run] with
its id and the file read, [read] with file, line and commit, [inferred].

## 1. The descriptions, stored

Run `r20261113-161238-fb5b` (`p4_describe.py`, `run_describe.sh`; exit 0, 2 min 44 s, 3.0 GiB peak). Each file is
`Circuit.description()` (templates, wires, cuts, registered, order) plus the record's `blocks` (`Circuit.blocks_record()`),
as canonical JSON (`interface.circuit_bytes`) [run: `out/describe.json`, and the files fetched with `research fetch --all`]:

| file | bytes | SHA-256 | steps | templates | units | wires | inference entries |
|---|---|---|---|---|---|---|---|
| `description_43_2.json` | 4,437,972 | `ca6705d47ec707df61245a6f185fbb30bc3b66db448716e6c52c6f7360f32f5b` | 2 | 105 | 5,137,384 | 13,238 | 141 |
| `description_43_3.json` | 6,312,499 | `db4c18024f9571e5b79058c336b0aa91a5700dedaff2abf724bb64d5e8747d64` | 3 | 106 | 5,273,728 | 19,414 | 142 |

Without `blocks`, the descriptions are 4,417,654 and 6,291,540 B, with SHA-512 `61b9f8d8516fe67f…` and `5c63819831adf4b9…`:
byte for byte run `3cb4`'s [run: `describe.json` against `3cb4`'s `keycost_l4*.json`]. The blocks add 20,320 and
20,961 B. The class has one slot (`slots: 1`).

## 2. Today's key wires by block

`p4_blocks.py` reads the two files alone [run locally, 23 s, 167 MB; output `p4_out/blocks_43_{2,3}.json`]. Each unit's
block (slot, step) comes from the inference entries (`Blocks.slots_steps`'s rule). Every reader index of every pattern is
visited, and each (writer unit, reader unit) pair gets both units' blocks. A key wire is a pattern whose reader port is
`tok` of an `EmbeddingRead_v1` template: 12,096 and 18,144 of them, every one written by
`MoeRouterSigmoidBiasSlots_v1{E=256,TOPK=8,…}`.

| | key wires | inside one block | between two blocks |
|---|---|---|---|
| 2 steps | 12,096 | 12,096 | 0 |
| 3 steps | 18,144 | 18,144 | 0 |

By reader template, each step separately, every wire is inside:

| reader template | B | 2 steps: wires, by step | 3 steps: wires, by step |
|---|---|---|---|
| `EmbeddingRead_v1{V=256,H=3072}` | 512 | 8,192: 4,096 / 4,096 | 12,288: 4,096 ×3 |
| `EmbeddingRead_v1{V=256,H=4096}` | 192 | 3,072: 1,536 / 1,536 | 4,608: 1,536 ×3 |
| `EmbeddingRead_v1{V=256,H=4}` | 48 | 768: 384 / 384 | 1,152: 384 ×3 |
| `EmbeddingRead_v1{V=256,H=96}` | 4 | 64: 32 / 32 | 96: 32 ×3 |

The index pairs join writer step s to reader step s only: 260,064 in step 0 and 6,048 in each decode step.

**All wires, as a check:** at 2 steps, 13,237 inside and 1 between; at 3 steps, 19,411 inside and 3 between. The ones
between are attention reading earlier steps' latent rows: `SparseMlaHeadBlock_v1{…NVIS=44|45,FIRST=true…}`'s `kv` from
`DsaLatentRow_v1`'s `o0`, writer step 0 to reader step 1, 0 to 2, and 1 or 2 to 2 [run: `wires_between_blocks`].

## 3. Form C's key wires, and how the record check decides

### The patterns

These are written from today's patterns in the stored files by `p4_formc.py` [run: `p4_out/keywires_43_{2,3}.json`, which
list every key wire today and C's with each one's reader blocks and writer blocks]. C itself is not built.

Today a node's read units are `(b TOPK + k) T + t`, and slot k reads router output `o<8+k>`. The router has one unit a
token: 0–42 in step 0, then 43 in step 1 and 44 in step 2. C keeps each node's allocation and puts it in slot-major order,
`(k B + b) T + t`, one pattern a slot. With R one of the four read templates, B its tables, and the writer
`MoeRouterSigmoidBiasSlots_v1{E=256,TOPK=8,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"}`, each pattern holds:

| step | node's first unit | pattern k = 0..7: `at`, `radix`, `base`, `step`; `out` | written at |
|---|---|---|---|
| 0 (prefill, T = 43) | 0 | `at` = 43 B k, `radix` [B, 43], `base` 0, `step` [0, 1]; `out` `o<8+k>` | 2 and 3 steps |
| 1 (T = 1) | 344 B | `at` = 344 B + B k, `radix` [B, 1], `base` 43, `step` [0, 1]; `out` `o<8+k>` | 2 and 3 steps |
| 2 (T = 1) | 352 B | `at` = 352 B + B k, `radix` [B, 1], `base` 44, `step` [0, 1]; `out` `o<8+k>` | 3 steps |

Each pattern also holds `reader` R, `port` `tok` and the writer. That gives 8 a read template a step: 64 patterns at
2 steps and 96 at 3. They cover the same 266,112 and 272,160 reader units as today's 12,096 and 18,144.

- **`coalesce` merges none of them** [inferred from `interface.py:558-586` at `eee9c706eec2`]. It merges only patterns
  with one reader port, writer, `out`, cut, radix and step whose reader units follow on. A node's eight patterns differ in
  `out`, and slot k's patterns in two decode steps are 8 B units apart, not B.
- **The reader's entries do not change** [inferred from `window_builder.inference`, lines 313-335, and the stored
  entries]. Slot-major order permutes units inside one node's allocation, which stays one run at one step. The entries
  name only a template, a first unit, a radix, axes and a step, so they stay as stored:
  - at 2 steps, for R = `H=3072`: `{at 0, radix [1, 176128], axes [slot, -], step 0}` and
    `{at 176128, radix [1, 4096], axes [slot, -], step 1}`;
  - at 3 steps, the two decode steps are one entry, `{at 176128, radix [2, 1, 4096], axes [step, slot, -], step 1}`.

**Per unit pair, every C pattern is inside one (slot, step) block:** 64 of 64 and 96 of 96 [run: `p4_formc.py`, every
reader index visited]. No C pattern spans two steps: each covers one node, which is one step.

### How `NciCheck.Record` decides inside or crossing

Read at `origin/main` `5eb1231b25581a1c9e87191d28cd5cf591f6033b`, files
`verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/{Record,RecordText}.lean`.

- **A block is a slot, not a (slot, step).**
  - `Record.lean:121-125`: "A block is one slot of the class, as the record's own `blocks.inference` divides the units".
  - `RecordText.gridOf` (`RecordText.lean:384-395`) reads an entry's `template`, `at`, `radix` and `axes` only. It never
    reads `step` or the `step` digit. Its grid is (start, units, q), with q the product of the radices after the slot digit.
  - `slotIn` (`RecordText.lean:422-424`) gives a unit's slot as `(u - start) / q % S`. No line of `Record.lean` or
    `RecordText.lean` reads an entry's step [read: `rg "step"` over both files finds only the registered reads' and
    ingress's own `step` fields].
- **Per pattern, from its fields, by a bounded search.**
  - `finish` (`Record.lean:733`) takes one Boolean a wire pattern: `sameSlot S (gridsAt w.reader) (gridsAt w.writer) w`.
  - `kindOf` (`Record.lean:660-662`) classes every row that pattern fills as `inside` or `crossWires` by that one Boolean.
  - `sameSlot` (`Record.lean:551-552`) is `S == 1 || search S gr gw FUEL [box of the whole pattern]`.
  - `search` (`Record.lean:523-546`) tests a box of the pattern's digits with `leaf` (`Record.lean:507-518`), a closed
    form over the grids, the box's first reader unit and writer, and each digit's radix, reader stride and writer step.
    Where `leaf` cannot show a box, `search` splits it on its widest digit, down to single rows if needed. After
    `FUEL` = 4,096 visits (`Record.lean:471`) the pattern counts as crossing.
  - So it is the second reading, pattern-whole, but exact wherever the fuel suffices: a pattern counts as crossing only if
    some row it covers is read across slots, or the fuel runs out.
- **What it reports for our window: every wire inside, today's and C's.** The record states `slots: 1`, so
  `sameSlot` is true at `S == 1` (`Record.lean:552`; the docstring at line 550: "A class of one slot has every wire
  inside"). That covers all 12,096 / 18,144 key wires today, all 64 / 96 of C's, and the 1 / 3 wires that cross steps
  [read, applied to the stored `slots`].

The open point does not arise in this code [inferred from the lines above]: it never asks whether a wire spans steps,
and C's patterns don't span steps anyway.

**Not established:**
- What `search` gives C's patterns in a class of several slots. If a request's units are one run of q under the slot
  digit, as `inference` lays them, each C pattern's reader units and its writer units would sit inside one slot's run, and
  `leaf` would show the whole pattern in one visit [inferred, not run].
- Whether `read` accepts this record up to `finish`. Network-accounting's comment says no run has read a window through
  it.
