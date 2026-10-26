# commit_slots: the serve-time commit call, one subtree per sequence slot

The call that commits a serving step's proof-unit outputs (CMP-95 item 3, Linear HASH-13): every unit's outputs as
`hm96-sha512/row/v2` rows under hiding leaves in `frame-v3-sha512` trees, with one subtree per sequence slot, so a drawn
unit opens by replaying its own request. `slots.py` is the CPU reference and this file's executable form;
`slots_gpu.py` runs the same call on the GPU (or on its host twin) on HASH-20's SHA-512 engine
(`benchmarks/torch_gpu/sha512_rows/`): the engine hashes rows into hm96 leaves and builds frame-v3 levels, and
`slots_gpu.py` lays them out as one subtree per slot. `fixtures/commit-slots-v1.json` is the interface's fixture.

## The call

```python
import slots as S, slots_gpu as G

# reference (CPU): outputs[i][j] is group i's tensor j, (slots, ...) with each slot's bytes contiguous
S.commit(context, layout, step, salt_key, outputs, slots) -> S.StepCommit(root, slot_roots)

# GPU, the same signature with a plan in place of (context, layout)
plan = G.Plan(context, layout)                     # once a run: domains, pad tails, midstates, hm96's table on the device;
                                                   # temp_bytes= bounds a step's temporaries (default 1 GiB), leaf_minb=
                                                   # runs the leaf kernel at 1 or 2 blocks an SM (same bytes)
plan = G.Plan(context, layout, sms=32)             # the side stream on a green context of 32 SMs (G.SmPartition; plan.sms
                                                   # is the count the driver gave), the leaf kernel's grid filling those
                                                   # alone: work on the other SMs runs beside it (same bytes)
G.commit(plan, step, salt_key, outputs, slots) -> S.StepCommit(root, slot_roots)

# GPU, streaming as a step's groups (e.g. layers) come out of the forward pass
sc = G.StepCommitter(plan, step, slots, salt_key)
sc.add(tensors)                                    # each group, in order; enqueued on a side stream, no host sync
pending = sc.finish()                              # pending.result() -> S.StepCommit; pending.ready() polls

# each add as one CUDA graph launch (same roots; one such step open on a plan at a time)
sc = G.StepCommitter(plan, step, slots, salt_key, graphs=True)

# several layers a launch: the same units, rows, digest and roots (the group's tensors are its layers', in order)
plan = G.Plan(context, G.regroup(layout, 4))
```

- `context`: 64 bytes naming the run (the model and serving instance); every domain binds it.
- `layout`: the step's public structure (`S.Layout`): groups (what one `add` commits) of units of rows, each row the
  concatenation of byte ranges (`S.Part(source, offset, nbytes)`) of a slot's row of the group's tensors. Every slot has
  the same layout; a unit holds at most 262,144 bits (its size class). The engine takes at most two parts a row.
  Groups are the prover's plumbing (one launch each): the layout's digest binds only its units' rows.
  A plan is linear in the layout's units: the layout digest is computed once (`Layout.digest`), every unit's domain id in
  one pass (`S.unit_domain_ids`, two SHA-512s a unit), and a group the layout repeats (its layers) is laid out once, so
  `add` rebinds that group's row table to the step's tensors without a Python step a row.
  A step's memory is its slots' frontiers and one pass's temporaries. It holds no slot's leaves: a slot's tree is its
  frontier, one node a level (the left sibling still waiting for its right one), 24 nodes or 1.5 KB a slot at six
  million units. `add` folds a group's unit leaves into the frontiers a level at a time (the engine's `level` at any
  first index), and `finish` closes each with pad(units) and the subtrees of pads alone, which the plan hashes once. A
  group goes through in passes of as many slots as keep its temporaries within `temp_bytes`: about four (rows, slots)
  digest buffers, so a group of 117,400 rows takes 35 slots a pass at 1 GiB. A pass's salts are its slots' own
  (nonce `nonce0 + s`), so passes change no root.
  The leaf kernel writes a one-row unit's slot-tree leaf itself, from the unit's frame table (the engine's `frame_table`,
  one compression a unit an add), so such a unit's row-slot costs nine SHA-512 compressions: its row digest one, hm96's c
  and h two each (h's padding block over a constant schedule), its frame leaf one, its unit's slot-tree leaf one, and
  about one slot-tree node, two; with three ChaCha20 blocks and the hm96 mask (`tests/test_cost.py` counts them).
- `step`: the step's index; `slots`: how many sequence slots it has.
- `salt_key`: 32 bytes, `S.fresh_salt_key()` (the OS generator), fresh each step and the prover's alone, kept with the
  step's record to open from.
- Returns `StepCommit(root, slot_roots)`: the 64-byte step root, and every slot's 64-byte root in slot order (the
  prover keeps them; only `root` leaves the prover).

The by-table fallback (`commit_fb`, beside this directory) matches the call by taking the same arguments and returning a
`StepCommit` with a 64-byte `root`; its root differs, since its tree is laid out by table.

## Rows and leaves

Row g of slot s (rows numbered in layout order within the slot), rank r in its unit u:

```text
x    = SHA-512(P(l) || row),  P(l) = ("verity/sha512-row/v2\0" || u64be l) zero-padded to 128 B, l = 8 len(row)
y    = ChaCha20(salt_key, nonce s) blocks 3g, 3g+1, 3g+2               192 bytes: C-Flock's hm96_salt(salt_key, s, g)
c    = SHA-512(salt_prefix || y);  b = x xor M_key y                    hm96-sha512/v1, the pinned key
leaf = SHA-512(leaf_prefix(key) || b || c), framed as rank r of unit_domain(u), schema hm96-sha512/row/v2
```

`b || c` (128 bytes) is the row's public string; `x` and the row stay private, and an opening shows only `b || c` and the
Merkle paths unless the proof that consumes it reads the row and its salt inside zero knowledge.

## The tree

Three levels, each a `frame-v3-sha512` tree padded to a power of two, leaves in index order:

| tree | leaves | leaf | domain (identity_digest_sha512 of tag, manifest) |
| --- | --- | --- | --- |
| unit u (in every slot) | its rows, rank r | `hm96-sha512/row/v2` | `verity/commit-slots/unit/v1`, {context, layout, unit u, rows, schema} |
| slot s | its units' roots, index u | `frame-v3-sha512/root/v1` (`row_table.root_leaf`) | `verity/commit-slots/slot/v1`, {context, layout, units, schema} |
| step | its slots' roots, index s | `frame-v3-sha512/root/v1` | `verity/commit-slots/step/v1`, {context, layout, step, slots, schema} |

`layout` in a manifest is the layout's digest, `identity_digest_sha512("verity/commit-slots/layout/v1", {name, units:
[[row bits, ...], ...]})`. Unit and slot domains name neither the slot nor the step: the step tree's leaf index fixes the
slot and its domain the step, so a slot's subtree is the same function of its rows in any step position and a run's unit
and slot pad tails are computed once.

## Where the salt comes from

The salt key is the prover's private randomness, from the operating system's generator, never derived from anything
public or from the transcript: no Fiat–Shamir. The verifier's only coins are live: the step's root goes to the challenger
before the step's live coin is drawn, and the coin picks which (slot, unit) pairs open. A slot's salts are a function of
the key and the slot index alone, so replaying a slot's request regenerates its rows and their salts.

## Opening a unit

`S.open_unit(context, layout, step, salt_key, slot_roots, slot, unit, rows)` rebuilds the slot's subtree from that slot's
rows alone (its request's replay), refuses if its root isn't the committed one, and returns a `UnitOpening`: the unit's
`b || c` strings, the unit root's path in the slot tree and the slot root's path in the step tree (`RootOpening`s).
`S.verify_unit(step_root, context, layout, step, slots, opening, rows=None)` checks it, and with `rows` (rank to
`(row, salt)`) that those are exactly the unit's rows.

## The frozen H200 layout

`S.h200_layout()`: MQA, width 16384, 52 layers, 256 query heads of 64, 1 KV head, MLP 65536, every op's per-token output
committed, one group a layer, each output cut into one-row units of at most 32 KiB: attention RMSNorm, its s8 quantizer
(codes, then the f32 scale, one two-part row), QKV (2 units), scores (f32, 16 context rows standing in for the mean t + 1
at a context of 32), softmax, PV, quantizer, o_proj, residual, MLP RMSNorm, quantizer, up (4 units), ReLU (4), down's
quantizer (2 units of codes and one of the scale), down and residual.

| | per slot | per step at 20,480 slots |
| --- | --- | --- |
| units | 25 a layer, 1,300 | 26,624,000 |
| committed bytes | 663,824 a layer, 34,518,848 | 706,946,007,040 |

## Measured on the H200

Run `r20261026-101500-2636` (vy-dc-h200, one H200, the engine with its conflict-free mask and the leaf kernel at 1 block
an SM, in a tree merged with #4612 for FAMILIES' layout: `gpu.sh`'s tests and check, and one FAMILIES step); the tables
alone and beside the GEMMs are run `r20261026-092158-634c`'s, at 1a452ff289, before that mask:

- **Bit-exact.** The tests passed (64, the device against the host twin included), and so did every `check.py` check,
  each root equal to 1a452ff289's:
  - the fixture's root;
  - the edge layout at 1 to 300 slots against the reference, and at 4,097 and 20,480 against the host twin;
  - two and three H200 layers, and three layers in two groups;
  - the frozen layout at 4,096, 16,384 and 20,480 slots (141, 566 and 707 GB), each with six sampled slots rebuilt by
    the reference and the step root from the device's slot roots.
- **FAMILIES at 4,096 slots** (#4612's layout: 6,102,102 units of 6,102,832 rows, 41 MB a slot, 168 GB a step). A step
  holds 0.54 GiB beyond its plan (2.07 GiB) and inputs, 6 MiB of it frontiers, and takes 71.9 s (69.4 s and 2.05 GiB
  with `temp_bytes` at 4 GiB), against 83.9 s with the mask's bank conflicts (run r20261026-092158-634c) and 108.2 s at
  ten compressions a row-slot (run r20261026-082348-7996). The split: hiding leaves with their frame leaves and the
  one-row units' slot-tree leaves 51.0 s (was 62.9 s), the multi-row units' roots and slot-tree leaves 0.35 s, the fold
  13.6 s and row digests 7.0 s. Its rows average 6.7 bytes, so the step is the hashing of 25.0G row-slots, not their
  bytes: nine SHA-512 compressions, three ChaCha20 blocks and the hm96 mask each (The call, above). In G compressions/s
  against the engine's 4.79: leaves 2.94 (the ChaCha20 blocks and the mask not counted), row digests 3.72, the fold 3.67,
  the step 3.13. The roots of six slots
  (0, 1, 3,321, 3,322, 4,094 and 4,095) equal the leaf-matrix path's, and the step root equals the reference's step
  tree.
  Over the same random unit leaves at 35 slots (one pass), the fold and close take 0.153 s and 0.37 GiB, against the
  leaf matrix's `engine.root` at 0.160 s and 26.4 GiB, with equal roots: 6.1M nodes a slot against 8.4M, since the
  matrix padded every slot to 2^23 leaves (run r20261026-083415-a945).
- **Alone,** every op committed, 52 layers. The share is of a step whose rest (the int8 GEMMs, attention and the
  elementwise ops) takes 5.33 s at 16,384 slots and scales with the slots:

  | slots | layers a group | s a step | GB/s | rows | leaves | trees | share of the step |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 16,384 | 1 | 1.033 | 548 | 0.968 s | 50 ms | 15 ms | 16% |
  | 20,480 | 1 | 1.282 | 551 | 1.201 s | 63 ms | 18 ms | 16% |
  | 12,288 | 1 | 0.784 | 541 | 0.731 s | 39 ms | 14 ms | 16% |
  | 4,096 | 4 | 0.260 | 544 | 0.243 s | 12 ms | 5 ms | 16% |
  | 4,096 | 1 | 0.332 | 426 | 0.308 s | 16 ms | 8 ms | 20% |
  | 1,024 | 1 | 0.132 | 267 | 0.119 s | 6 ms | 7 ms | 28% |

  At 16,384 slots that's 4.30 G SHA-512 compressions/s of row data. Rows are the engine's `rows` and `row_digests`;
  leaves include each one-row unit's slot-tree leaf; trees are the multi-row units' trees, the slot trees' fold and
  close, and the step tree. The leaf matrix's trees (run r20261026-014030-c476) took 34 ms at 16,384 slots and 4 ms at
  1,024: a fold at a few slots a group is launch-bound.
- **Beside ALLP-H200's served step** (`overlap.py`: `../../e2e_h200/`'s driver at the frozen shape, 4,096 slots in
  micro-batches of 1,024, the demo weights resident; circuits' shared-row layout, 101,718 units in 102,448 rows, 27.4 MB
  a slot). Serving alone takes 3.25 s a step. Today's loop hides 2% of the commit: each group is streamed to the side
  stream, but the step's end waits for its roots and the driver's ticks synchronize the whole device. Overlapped, with
  the step's close in flight into the next step, the side stream at most `LAG` groups behind and the ticks synchronizing
  the forward's stream only:
  - On main, with 1.43 s of commit alone (run r20261026-161847-6d05), 5% hides at low priority and 4% at high.
  - On #4724, with 1.33 s alone (no sync in `add`, the leaf kernel at 2 blocks an SM; run r20261026-164041-ba43), 14%
    hides at low priority (a step of 4.40 s, 1.15 s over serving) and 8% at high. Issuing the commit from a host thread
    of its own changes nothing.

  The rest doesn't hide because both fill every SM and take turns. SM clocks stay at 1,980 MHz, at 566 to 588 W against
  the 700 W cap, with no throttle reason. HBM peaks at 146.3 GB, 0.04 GB over serving alone. With FAMILIES' layout
  (6,102,102 units, main) the commit alone takes 71.18 s, and a step overlapped at low priority takes 74.94 s.
- **Beside a layer's four int8 GEMMs** (`torch._int_mm`, M = 16,384, 939 TOPS): the GEMMs take 93.7 ms alone and
  107.1 ms beside the commit (+14%). Overlapped, a layer takes 107.1 ms, against 114.5 ms run one after the other, so
  overlap saves little. Over a step, the commit costs 22% of the GEMMs' 4.87 s.
- **CIR-113's shared rows at 4,096 slots** (run r20261026-161032-934f, vy-dc-h200-3, `gpu.sh`'s tests, check and step).
  `step.py export --shared` writes FAMILIES' layout with the gathered tables' producers dropped, circuits' layout: its
  digest is theirs (`27dfdf843d4b6944…`), 101,718 units of 102,448 rows, 27.4 MB a slot. A row-slot costs 11.03
  compressions (row digests 3.03, most rows being 256 B; leaves 5.99; the fold 1.99), 4.63G a step, 0.97 s at 4.79 G/s.
  A step takes 1.330 s and 0.51 GiB beyond its plan and inputs (2.36 GiB): hiding leaves 0.820 s at 3.06 G
  compressions/s (64% of peak, the ChaCha20 blocks and the mask not counted), row digests 0.287 s at 4.42 (92%), the
  fold 0.219 s at 3.81 (80%), the multi-row units 7 ms. That is with the leaf kernel at 2 blocks an SM (1.401 s at 1)
  and `add` no longer synchronizing the side stream (1.429 s before, the GPU idle 51 ms of it, against 9 ms); passes of
  4 GiB take 1.318 s. Six sampled slots' roots equal the reference's, and the step root the reference's step tree. The
  host still spends about 1.07 s in a step's `add`s, with no upload or sync left in them: it waits on the launch queue
  (5,108 launches a step, 1,379 of them the fold's levels).
- **Each add as one CUDA graph** (run r20261026-165818-b616, vy-dc-h200-3, `gpu.sh`'s tests and `step.py time --graphs
  0 1`). Under nsys (run r20261026-164519-8119, `launches.py`), a step's 5,421 enqueue calls take a median 4.4 us until
  about the 1,285th, and from there 517 of them block, 0.931 s together: each waits for room in the launch queue. With
  `graphs=True`, circuits' layout at 4,096 slots enqueues a step in 0.012 s, against 0.887 s eager (1.059 s in 1 GiB
  passes), and the step takes 1.313 s on the GPU, against 1.320 s; the first step, which captures each group's graph,
  1.324 s. The 1 KiB-row variant: 0.018 s against 0.527 s, and 0.803 s against 0.807 s. Every step on graphs gives the
  eager step's roots, under a second salt key too. The graphs' pool keeps a pass's temporaries reserved between steps
  (0.57 GiB more reserved in 1 GiB passes).
- **The commit on N of the 132 SMs** (`Plan(sms=N)`, the side stream on a green context of N SMs and the leaf kernel's
  grid filling them; `overlap.py`'s `MODE:shared:STEPS:N`, eager, shared rows at 4,096 slots, leaf kernel at 2 CTAs an
  SM; runs r20261026-171704-923c and r20261026-173759-4d04, vy-dc-h200-3). Alone it takes 166 to 173 SM·s a step at every N
  from 8 to 88 (20.73 s on 8 SMs, 4.18 on 40, 2.62 on 64), 176 on all 132, its roots the uncapped commit's. Beside the
  served step on every SM, the least overhead is the commit at high priority on 64 SMs: a step of 4.343 s, +1.046 s
  over serving alone (3.298 s in that run), 1.10 times CMP-149's 0.95 s, against +1.095 s for the uncapped commit at low
  priority in the same run; three later runs repeat it at +1.071, +1.076 and +1.099 s. From 56 to 88 SMs it stays at
  +1.05 to +1.10 s. Below 48 SMs the commit alone outlasts the
  forward. SM clocks stay at 1,980 MHz with no clock event reason in every run; the high and equal runs from 56 to 88
  SMs draw 619 to 641 W median against the 700 W limit, and no sample exceeds 697.9 W, so power never binds.
- **The forward on the other SMs** (`:Nr`) is worse at every N (run r20261026-181159-8f26; `serve:shared:3:Nr` is the
  served step alone there), for two reasons. The forward slows on fewer SMs: 3.534 s on 124, 4.182 on 92, 5.201 on 68,
  against 3.274 s on all 132. And when the commit lags, torch's allocator, at 146.3 of 150.1 GB, frees its cache for
  the next group's tables (2.08 GB at 4,096 slots), waiting on every stream: at 40 SMs (the commit alone 4.173 s, the
  forward alone 4.182 s) a step takes 6.984 s with 92 such retries in 3 steps, and at 8 SMs 23.288 s with 69. At 64,
  with none, it takes 5.297 s, the forward alone's time and 0.10 s. The forward's host waiting on the side stream's reads
  before it allocates (`--host-lag`, run r20261026-182023-2b8d) leaves 84 to 100 retries (6.878 s at 40 SMs): a table
  `record_stream`'d to the side stream is released at that stream's tail when the forward frees it, not at its read, so
  only freeing it after the read (or reusing a ring of buffers) avoids them. The partitions themselves run at once
  (`green_probe.py`, runs r20261026-174052-c188 and r20261026-180440-47aa): the commit on 40 or 64 SMs and int8 GEMMs
  on the others overlap 0.91 to 0.93 (1 at once, 0 in turns), 0.88 to 0.90 coupled as today's loop couples them, and
  0.08 sharing every SM in one context. A plan on N SMs runs on graphs too: a graph captured on its green context's
  stream keeps to those SMs when replayed, rekeyed with that context current (`SmPartition.current`; run
  r20261026-181335-1b14, a `%smid` probe on 16 SMs).

## Files

- `slots.py`: the reference, `edge_layout()`, `h200_layout()`, and `python slots.py fixture` (writes the fixture).
- `slots_gpu.py`: `Plan`, `StepCommitter`, `commit` and `regroup` on the engine (`../sha512_rows/engine.py`, its builds
  cached under `~/.cache/verity/sha512_rows`). Its uploads go through pinned memory, so no call waits on the side stream.
- `check.py`: the GPU's roots against the reference's (edge layouts, and the H200 layout at 4,096 to 20,480 slots on
  sampled slots); `bench.py`: GB/s and the share of a serving step at the frozen shape, alone and beside the layer's int8
  GEMMs; `step.py`: one step at a layout file (`slots._layout_json`'s form) timed by phase against the engine's peak,
  with its peak memory, the leaf kernel alone, a profile and sampled slots against the reference, and `export --shared`,
  CIR-113's layout (`--row-bytes B` cuts its shared rows to B bytes), `count`, a layout's compressions a row-slot on the
  host twin, and `time --graphs 0 1`, steps eager and on graphs with the host's enqueue time; `launches.py`: an nsys
  report's enqueue calls a step (how long each held the thread, and from which call they block); `gpu.sh`: these, with
  the tests, under one GPU lease (its header has the `research run` line; `STEP_WRAP` runs `step.py` under a profiler).
- `tests/`: the reference against the fixture and C-Flock's salts and an independent tree build, openings and their
  refusals, the host twin against the reference (regrouped layouts included), the streamed slot trees against the leaf
  matrix they replaced and the reference (unit counts around powers of two, groups cut anywhere, one-slot passes), a
  step's memory (counted: its frontier and one pass), a row-slot's compressions (counted), planning at six million
  units, and (with a GPU) the device against the host twin.
- `overlap.py`: the commit beside `../../e2e_h200/`'s served step on one GPU: serving alone, the commit alone, today's
  loop, and overlapped at low, high and equal priority (from the forward's host thread or one of its own), each with the
  commit on every SM or on a green context of N (`MODE:LAYOUT:STEPS:N`, the forward on every SM, or `:Nr`, on the other
  SMs' green context; `serve:LAYOUT:STEPS:Nr`, the served step alone there; `--host-lag`, the forward's host waiting on
  the side stream's reads before it allocates), with nvidia-smi's clocks, power and HBM, torch's allocator counters,
  each overlapped or capped step's roots against today's loop's and a `VERDICT` line a run; `overlap.sh`: it under one
  GPU lease (its header has the `research run` line). `green_probe.py`: the commit on a green context of N SMs beside
  int8 GEMMs on the other SMs or every SM, uncoupled or coupled as today's loop couples them (no model).
