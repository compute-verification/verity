# commit_slots: the serve-time commit call, one subtree per sequence slot

The call that commits a serving step's proof-block outputs: every proof block's outputs as
`hm96-sha512/row/v2` rows under hiding leaves in `frame-v3-sha512` trees, with one subtree per sequence slot, so a drawn
proof block opens by replaying its own request. `slots.py` is the CPU reference and this file's executable form;
`slots_gpu.py` runs the same call on the GPU (or on its host twin) on the SHA-512 engine
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
                                                   # runs the leaf kernel at 1 or 2 CTAs an SM (same bytes)
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

# several layers a launch: the same proof blocks, rows, digest and roots (the group's tensors are its layers', in order)
plan = G.Plan(context, G.regroup(layout, 4))
```

- `context`: 64 bytes naming the run (the model and serving instance); every domain binds it.
- `layout`: the step's public structure (`S.CommitShape`): groups (what one `add` commits) of units of rows, each row the
  concatenation of byte ranges (`S.Part(source, offset, nbytes)`) of a slot's row of the group's tensors. Every slot has
  the same layout; a unit holds at most 262,144 bits (its size class). The engine takes at most two parts a row.
  Parts, rows, groups and their checks are `verity.ml.commitments.layout`'s; `S.load(path)` reads a layout file through
  it, in the form `S._layout_json` writes.
  Groups are the prover's plumbing (one launch each): the layout's digest binds only its units' rows.
  A plan is linear in the layout's units: the layout digest is computed once (`CommitShape.digest`), every unit's domain id in
  one pass (`S.unit_domain_ids`, two SHA-512s a unit), and a group the layout repeats (its layers) is laid out once, so
  `add` rebinds that group's row table to the step's tensors without a Python step a row.
  A step's memory is its slots' frontiers and one pass's temporaries. It holds no slot's leaves: a slot's tree is its
  frontier, one node a level (the left sibling still waiting for its right one), 24 nodes or 1.5 KB a slot at six
  million units. `add` folds a group's proof-block leaves into the frontiers a level at a time (the engine's `level` at any
  first index), and `finish` closes each with pad(units) and the subtrees of pads alone, which the plan hashes once. A
  group goes through in passes of as many slots as keep its temporaries within `temp_bytes`: about four (rows, slots)
  digest buffers, so a group of 117,400 rows takes 35 slots a pass at 1 GiB. A pass's salts are its slots' own
  (nonce `nonce0 + s`), so passes change no root.
  The leaf kernel writes a one-row proof block's slot-tree leaf itself, from its frame table (the engine's `frame_table`,
  one compression a proof block an add), so such a proof block's row-slot costs nine SHA-512 compressions: its row digest
  one, hm96's c and h two each (h's padding chunk over a constant schedule), its frame leaf one, its slot-tree leaf one,
  and about one slot-tree node, two; with three ChaCha20 chunks and the hm96 mask (`tests/test_cost.py` counts them).
- `step`: the step's index; `slots`: how many sequence slots it has.
- `salt_key`: 32 bytes, `S.fresh_salt_key()` (the OS generator), fresh each step and the prover's alone, kept with the
  step's record to open from.
- Returns `StepCommit(root, slot_roots)`: the 64-byte step root, and every slot's 64-byte root in slot order (the
  prover keeps them; only `root` leaves the prover).

## Rows and leaves

Row g of slot s (rows numbered in layout order within the slot), rank r in its proof block u:

```text
x    = SHA-512(P(l) || row),  P(l) = ("verity/sha512-row/v2\0" || u64be l) zero-padded to 128 B, l = 8 len(row)
y    = ChaCha20(salt_key, nonce s) chunks 3g, 3g+1, 3g+2               192 bytes: C-Flock's hm96_salt(salt_key, s, g)
c    = SHA-512(salt_prefix || y);  b = x xor M_key y                    hm96-sha512/v1, the pinned key
leaf = SHA-512(leaf_prefix(key) || b || c), framed as rank r of unit_domain(u), schema hm96-sha512/row/v2
```

`b || c` (128 bytes) is the row's public string; `x` and the row stay private, and an opening shows only `b || c` and the
Merkle paths unless the proof that consumes it reads the row and its salt inside zero knowledge.

## The tree

Three levels, each a `frame-v3-sha512` tree padded to a power of two, leaves in index order:

| tree | leaves | leaf | domain (identity_digest_sha512 of tag, manifest) |
| --- | --- | --- | --- |
| proof block u (in every slot) | its rows, rank r | `hm96-sha512/row/v2` | `verity/commit-slots/unit/v1`, {context, layout, unit u, rows, schema} |
| slot s | its proof blocks' roots, index u | `frame-v3-sha512/root/v1` (`row_table.root_leaf`) | `verity/commit-slots/slot/v1`, {context, layout, units, schema} |
| step | its slots' roots, index s | `frame-v3-sha512/root/v1` | `verity/commit-slots/step/v1`, {context, layout, step, slots, schema} |

`layout` in a manifest is the layout's digest, `identity_digest_sha512("verity/commit-slots/layout/v1", {name, units:
[[row bits, ...], ...]})`. The proof-block and slot domains name neither the slot nor the step: the step tree's leaf index fixes the
slot and its domain the step, so a slot's subtree is the same function of its rows in any step position and a run's proof-block
and slot pad tails are computed once.

## Where the salt comes from

The salt key is the prover's private randomness, from the operating system's generator, never derived from anything
public or from the transcript: no Fiat–Shamir. The verifier's only coins are live: the step's root goes to the challenger
before the step's live coin is drawn, and the coin picks which (slot, proof block) pairs open. A slot's salts are a function of
the key and the slot index alone, so replaying a slot's request regenerates its rows and their salts.

## Opening a proof block

`S.open_unit(context, layout, step, salt_key, slot_roots, slot, unit, rows)` rebuilds the slot's subtree from that slot's
rows alone (its request's replay), refuses if its root isn't the committed one, and returns a `UnitOpening`: the proof block's
`b || c` strings, its root's path in the slot tree and the slot root's path in the step tree (`RootOpening`s).
`S.verify_unit(step_root, context, layout, step, slots, opening, rows=None)` checks it, and with `rows` (rank to
`(row, salt)`) that those are exactly the proof block's rows.

## The frozen H200 layout

`S.h200_layout()`: MQA, width 16384, 52 layers, 256 query heads of 64, 1 KV head, MLP 65536, every op's per-token output
committed, one group a layer, each output cut into one-row proof blocks (at most 32 KiB of outputs each): attention RMSNorm, its s8 quantizer
(codes, then the f32 scale, one two-part row), QKV (2 proof blocks), scores (f32, 16 context rows standing in for the mean t + 1
at a context of 32), softmax, PV, quantizer, o_proj, residual, MLP RMSNorm, quantizer, up (4 proof blocks), ReLU (4), down's
quantizer (2 proof blocks of codes and one of the scale), down and residual.

| | per slot | per step at 20,480 slots |
| --- | --- | --- |
| proof blocks | 25 a layer, 1,300 | 26,624,000 |
| committed bytes | 663,824 a layer, 34,518,848 | 706,946,007,040 |

## Files

- `slots.py`: the reference, `edge_layout()`, `h200_layout()`, `load()` (a layout file, through `commitments.layout`), and
  `python slots.py fixture` (writes the fixture).
- `slots_gpu.py`: `Plan`, `StepCommitter`, `commit` and `regroup` on the engine (`../sha512_rows/engine.py`, its builds
  cached under `~/.cache/verity/sha512_rows`). Its uploads go through pinned memory, so no call waits on the side stream.
- `check.py`: the GPU's roots against the reference's (edge layouts, and the H200 layout at 4,096 to 20,480 slots on
  sampled slots); `bench.py`: GB/s and the share of a serving step at the frozen shape, alone and beside the layer's int8
  GEMMs; `step.py`: one step at a layout file (`slots._layout_json`'s form) timed by phase against the engine's peak,
  with its peak memory, the leaf kernel alone, a profile and sampled slots against the reference; `count`, a layout's
  compressions a row-slot on the host twin; and `time --graphs 0 1`, steps eager and on graphs with the host's enqueue
  time. A file of FAMILIES' shared-row layout at B-byte rows is `e2e_h200.window.shared_layout(FROZEN, B)` written by
  `S._layout_json`. `launches.py`: an nsys report's enqueue calls a step (how long each held the thread, and from which
  call they block); `gpu.sh`: these, with the tests, under one GPU lease (`OUT=DIR bash gpu.sh` from the tree's root;
  `STEP_WRAP` runs `step.py` under a profiler).
- `tests/`: the reference against the fixture and C-Flock's salts and an independent tree build, openings and their
  refusals, the edge layout and FAMILIES' 1 KiB layout written and loaded back to the same layout, digests and roots,
  the host twin against the reference (regrouped layouts included), the streamed slot trees against the leaf
  matrix they replaced and the reference (unit counts around powers of two, groups cut anywhere, one-slot passes), a
  step's memory (counted: its frontier and one pass), a row-slot's compressions (counted), planning at six million
  units, the order a plan drops added tables in, and (with a GPU) the device against the host twin, at a near-full HBM
  no allocator retry while the side stream lags its reads, and a freed table's memory back at its read.
- `overlap.py`: the commit beside `../../../examples/h200/`'s served step on one GPU: serving alone, the commit alone,
  the serving loop, and overlapped at low, high and equal priority (from the forward's host thread or one of its own),
  each with the commit on every SM or on a green context of N (`MODE:LAYOUT:STEPS:N`, the forward on every SM, or
  `:Nr`, on the other SMs' green context; `serve:LAYOUT:STEPS:Nr`, the served step alone there; `--host-lag`, the
  forward's host waiting on the side stream's reads before it allocates), with nvidia-smi's clocks, power and HBM,
  torch's allocator counters, each overlapped or capped step's roots against the serving loop's and a `VERDICT` line a
  run; `overlap.sh`: it under one GPU lease on an H200 (its header has the command). `green_probe.py`: the commit on a
  green context of N SMs beside int8 GEMMs on the other SMs or every SM, uncoupled or coupled as the serving loop
  couples them (no model).
