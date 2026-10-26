# commit_slots: the serve-time commit call, one subtree per sequence slot

The call that commits a serving step's proof-unit outputs (CMP-95 item 3, Linear HASH-13): every unit's outputs as
`hm96-sha512/row/v2` rows under hiding leaves in `frame-v3-sha512` trees, with one subtree per sequence slot, so a drawn
unit opens by replaying its own request. `slots.py` is the CPU reference and this file's executable form;
`slots_gpu.py` runs the same call on the GPU (or on its host twin), on top of `benchmarks/pouw/commit_gpu/sha512.cuh`.
`fixtures/commit-slots-v1.json` is the interface's fixture.

## The call

```python
import slots as S, slots_gpu as G

# reference (CPU): outputs[i][j] is group i's tensor j, (slots, ...) with each slot's bytes contiguous
S.commit(context, layout, step, salt_key, outputs, slots) -> S.StepCommit(root, slot_roots)

# GPU, the same signature with a plan in place of (context, layout)
plan = G.Plan(context, layout)                     # once a run: domains, pad tails, midstates, hm96's table on the device
G.commit(plan, step, salt_key, outputs, slots) -> S.StepCommit(root, slot_roots)

# GPU, streaming as a step's groups (e.g. layers) come out of the forward pass
sc = G.StepCommitter(plan, step, slots, salt_key)
sc.add(tensors)                                    # each group, in order; enqueued on a side stream, no host sync
pending = sc.finish()                              # pending.result() -> S.StepCommit; pending.ready() polls
```

- `context`: 64 bytes naming the run (the model and serving instance); every domain binds it.
- `layout`: the step's public structure (`S.Layout`): groups (what one `add` commits) of units of rows, each row the
  concatenation of byte ranges (`S.Part(source, offset, nbytes)`) of a slot's row of the group's tensors. Every slot has
  the same layout; a unit holds at most 262,144 bits (its size class). The GPU takes at most two parts a row, 24
  tensors a group, 128 rows a unit and 2^16 units a slot.
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

Run `r20261026-002612-bfed` (vy-dc-h200, one H200, commit 2eb946b41c, `gpu.sh`):

- **Bit-exact.** Every `check.py` check passed. That covers the fixture's root, the edge layout at 1 to 300 slots against
  the reference and at 4,097 and 20,480 against the host twin, and two and three H200 layers. It also covers the frozen
  layout at 20,480 slots (707 GB): six sampled slots rebuilt by the reference, and the step root from the device's slot
  roots.
- **Alone,** every op committed, 52 layers:

  | slots | s a step | GB/s | rows kernel | fold |
  | --- | --- | --- | --- | --- |
  | 20,480 | 1.464 | 483 | 1.409 s | 38 ms |
  | 4,096 | 0.395 | 358 | 0.352 s | 37 ms |
  | 1,024 | 0.176 | 201 | 0.137 s | 37 ms |

  At 20,480 slots that's 3.79 G SHA-512 compressions/s of row data. It's 40% of a 3.7 s serving step and 20% of a
  7.5 s one.
- **Beside a layer's four int8 GEMMs** (`torch._int_mm`, M = 20,480, 945 TOPS): the GEMMs take 116.5 ms alone and
  145 ms beside the commit (+24%). Overlapped, a layer takes 145 ms, against 150 ms run one after the other. Over a
  step, the commit costs 20% of the GEMMs' 7.54 s.

## Files

- `slots.py`: the reference, `edge_layout()`, `h200_layout()`, and `python slots.py fixture` (writes the fixture).
- `commit_slots.cuh`: the hashing shared by the kernels and the host twin (rows, salts, hm96 leaves, frame nodes, the
  per-slot fold and the step tree); `commit_slots.cu` the kernels; `commit_slots_host.cpp` the host twin.
- `slots_gpu.py`: `Plan`, `StepCommitter`, `commit`; builds cached under `~/.cache/verity/commit_slots`.
- `check.py`: the GPU's roots against the reference's (edge layouts, and the H200 layout at 20,480 slots on sampled
  slots); `bench.py`: GB/s and the share of a serving step at the frozen shape, alone and beside the layer's int8 GEMMs;
  `gpu.sh`: both, with the tests, under one GPU lease (its header has the `research run` line).
- `tests/`: the reference against the fixture and C-Flock's salts and an independent tree build, openings and their
  refusals, the host twin against the reference, and (with a GPU) the kernels against the host twin.
