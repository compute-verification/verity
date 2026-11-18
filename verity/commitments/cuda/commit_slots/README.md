# commit_slots: the serve-time commit call, one subtree per sequence slot

The call that commits a serving step's proof-block outputs: every proof block's outputs as `hm96-sha512/row/v2` rows
under hiding leaves in `frame-v3-sha512` trees, with one subtree per sequence slot, so a sampled proof block opens by
replaying its own request. `slots.py` is the CPU reference, and its docstring states the format: a row's leaf, the three
trees (proof block, slot and step) and their domains. `slots_gpu.py` runs the same call on the GPU (or on its host twin)
on the SHA-512 engine (`verity/commitments/cuda/sha512_rows/`).
`tests/verity/commitments/fixtures/commit-slots-v1.json` is the interface's fixture.

## The call

```python
from verity.commitments.cuda.commit_slots import slots as S, slots_gpu as G

# reference (CPU): outputs[i][j] is group i's tensor j, (slots, ...) with each slot's bytes contiguous
salts = S.fresh_salts(layout, slots)                # (slots, rows, 192) uint8: one draw from the OS a row of a slot
S.commit(context, layout, step, salts, outputs, slots) -> S.StepCommit(root, slot_roots)

# GPU, the same signature with a plan in place of (context, layout); on a CUDA device, a Plan subclass whose ENGINE's
# runtime() is torch (verity imports none; the benchmarks' benchmarks/torch_gpu/commit_slots/torch_plan.Plan)
plan = G.Plan(context, layout, "cuda", build=DIR)  # once a run; temp_bytes= bounds a step's temporaries
plan = G.Plan(context, layout, "cuda", sms=32, build=DIR)  # the side stream on a green context of 32 SMs (same bytes)
G.commit(plan, step, salts, outputs, slots) -> S.StepCommit(root, slot_roots)   # salts on the plan's device

# GPU, streaming as a step's groups (e.g. layers) come out of the forward pass
sc = G.StepCommitter(plan, step, slots)
sc.add(tensors, salts[:, g0:g0 + rows])            # each group, in order, with its rows' salts (g0: the layout's
                                                   # group_row_base); enqueued on a side stream, no host sync
pending = sc.finish()                              # pending.result() -> S.StepCommit; pending.ready() polls

# each add as one CUDA graph launch (same roots; one such step open on a plan at a time)
sc = G.StepCommitter(plan, step, slots, graphs=True)

# several layers a launch: the same proof blocks, rows, digest and roots
plan = G.Plan(context, G.regroup(layout, 4), "cuda", build=DIR)
```

- `context`: 64 bytes naming the run (the model and serving instance); every domain binds it.
- `layout`: the step's public structure (`S.CommitShape`): groups (what one `add` commits) of proof blocks of rows, each
  row the concatenation of byte ranges (`S.Part(source, offset, nbytes)`) of a slot's row of the group's tensors. Every
  slot has the same layout, and a proof block holds at most 262,144 bits (its size class). Parts, rows and groups, with
  their checks, are `verity.commitments.layout`'s; `S.load(path)` reads a layout file in the form `S._layout_json`
  writes.
- `step`: the step's index; `slots`: how many sequence slots it has.
- `salts`: every row's hiding salt, `(slots, layout.row_count, 192)` uint8, slot s's row g's at `[s, g]`:
  `S.fresh_salts(layout, slots)`, one `hm96.SHA512.fresh_salt()` (the operating system's generator) a row of a slot, put
  on the plan's device by the caller and read there in place. Nothing derives a salt and no key for one exists, so the
  bytes themselves are what the prover keeps to open from: the prover's alone, as secret as the rows, in its private
  files and never under a run's output directory. A step's salts are 192 bytes a row-slot beside the rows (the H200
  layout's 1,300 rows a slot: 250 KB a slot).
- Returns `StepCommit(root, slot_roots)`: the 64-byte step root, the one value that leaves the prover, and every slot's
  root in slot order.

## Opening a proof block

`S.open_unit(context, layout, step, salts, slot_roots, slot, unit, rows)` rebuilds the slot's subtree from that slot's
rows and their salts (`salts`: the slot's, `(rows, 192)`) alone (its request's replay), refuses if its root isn't the committed one, and returns a `UnitOpening`: the proof
block's public strings, its root's path in the slot tree and the slot root's path in the step tree.

## Layouts

`S.edge_layout()` is a small layout over SHA-512's chunk edges, the fixture's. `S.h200_layout()` is the H200 demo's
frozen decoder's: every op's per-token output committed, one group a layer, each output cut into one-row proof
blocks.

## Files

- `slots.py`: the reference, the layouts, `load()`, and `python slots.py fixture` (writes the fixture).
- `slots_gpu.py`: `Plan`, `StepCommitter`, `commit` and `regroup` on the engine (`../sha512_rows/engine.py`).

What measures it is in `benchmarks/torch_gpu/commit_slots/`: `check.py`, the GPU's roots against the reference's;
`bench.py`, the call's throughput at the H200 example's layout, alone and beside the layer's int8 GEMMs; `step.py`, one
streamed step at a layout file, timed by phase (`time`), and a layout's compressions a row-slot on the host twin
(`count`); and `gpu.sh`, these with the tests under one GPU lease (`OUT=DIR bash benchmarks/torch_gpu/commit_slots/gpu.sh`
from the tree's root; its header has the options).

## Tests

`tests/` is the directory's suite: the reference against the fixture and an independent tree build from hm96's own leaf under the same salt bytes;
openings and their refusals; layouts written and loaded back; the host twin against the reference; the streamed slot
trees; a step's memory and a row-slot's compressions, counted; planning at scale; and, with a GPU, the device against the
host twin. Two say why they skip where they can't run: the FAMILIES layout's round trip where its fetched fixture is
absent (`research data fetch-fixtures tests/verity/commitments/fixtures/layouts`; the suites' runner and `gpu.sh` fetch
it), and the plan on some SMs where this Python's torch has no driver bindings (`slots_gpu.driver_bindings`).
