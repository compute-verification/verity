# commit_slots: the serve-time commit call, one subtree per sequence slot

Each serving step, the developer commits the outputs of every proof block before the step's live coin, and later
opens the ones an audit samples. This call does the committing: every proof block's outputs as `hm96-sha512/row/v2`
rows under hiding leaves in `frame-v3-sha512` trees, with one subtree per sequence slot, so a sampled proof block opens
by replaying its own request alone. `slots.py` is the CPU reference, and its docstring states the format: a row's leaf,
the three trees (proof block, slot and step) and their domains. `slots_gpu.py` runs the same call on the GPU (or on
its host twin) on the SHA-512 engine (`verity/commitments/cuda/sha512_rows/`).
`tests/verity/commitments/fixtures/commit-slots-v1.json` is the interface's fixture.

## Try it

~~~sh
uv run tools/check/suites.py verity/commitments/cuda/commit_slots
~~~

It holds the reference to the fixture and the host twin to the reference on the CPU, in under a minute; the tests that
need a GPU skip. [`DESIGN.md`](DESIGN.md) has what each test covers and what measures the call on a GPU.

## The call

```python
from verity.commitments.cuda.commit_slots import slots as S, slots_gpu as G

# reference (CPU): outputs[i][j] is group i's tensor j, (slots, ...) with each slot's bytes contiguous
S.commit(context, layout, step, salt_key, outputs, slots) -> S.StepCommit(root, slot_roots)

# GPU, the same signature with a plan in place of (context, layout); on a CUDA device, a Plan subclass whose ENGINE's
# runtime() is torch (verity imports none; the benchmarks' benchmarks/torch_gpu/commit_slots/torch_plan.Plan)
plan = G.Plan(context, layout, "cuda", build=DIR)  # once a run; temp_bytes= bounds a step's temporaries
plan = G.Plan(context, layout, "cuda", sms=32, build=DIR)  # the side stream on a green context of 32 SMs (same bytes)
G.commit(plan, step, salt_key, outputs, slots) -> S.StepCommit(root, slot_roots)

# GPU, streaming as a step's groups (e.g. layers) come out of the forward pass
sc = G.StepCommitter(plan, step, slots, salt_key)
sc.add(tensors)                                    # each group, in order; enqueued on a side stream, no host sync
pending = sc.finish()                              # pending.result() -> S.StepCommit; pending.ready() polls

# each add as one CUDA graph launch (same roots; one such step open on a plan at a time)
sc = G.StepCommitter(plan, step, slots, salt_key, graphs=True)

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
- `salt_key`: 32 bytes, `S.fresh_salt_key()` (the operating system's generator), fresh each step and the prover's alone,
  kept with the step's record to open from; never derived from anything public.
- Returns `StepCommit(root, slot_roots)`: the 64-byte step root, the one value that leaves the prover, and every slot's
  root in slot order.

## Opening a proof block

`S.open_unit(context, layout, step, salt_key, slot_roots, slot, unit, rows)` rebuilds the slot's subtree from that slot's
rows alone (its request's replay), refuses if its root isn't the committed one, and returns a `UnitOpening`: the proof
block's public strings, its root's path in the slot tree and the slot root's path in the step tree.
`S.verify_unit(step_root, context, layout, step, slots, opening, rows=None)` checks it, and with `rows` (rank to
`(row, salt)`) that those are exactly the proof block's rows.

## Layouts

`S.edge_layout()` is a small layout over SHA-512's chunk edges, the fixture's. `S.h200_layout()` is the H200 demo's
frozen decoder's: every op's per-token output committed, one group a layer, each output cut into one-row proof
blocks.

