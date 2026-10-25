# commit_fb: the fallback serve-time commit call

If KCOMMIT's per-sequence-slot layout (Linear HASH-13) isn't ready, the walking skeleton
commits its unit outputs through vLLM's existing row-table committer, laid out by table (Linear HASH-17).
`commit_fb.py` imports that committer and copies none of it. It only lays unit outputs out the way the committer takes them.

## The call

~~~python
from commit_fb import SaltSource, commit_step, commit_step_async, open_rows, segment_of

salt = SaltSource(key=row_key_32_bytes, segment=segment_of(engine_step, layer, 53), context=coin_root_64_bytes)
c = commit_step(unit_outputs, salt)       # c.root: 64 bytes; c.tables: ((op_path, root), ...); c.ranks[b]: slot b's leaf
p, rows, ranks = commit_step_async(unit_outputs, salt)   # the device path's Pending: ready(), result(), kernel_ms()
~~~

- `unit_outputs[op_path]` is either a tensor with one row per sequence or a quantizer output `(codes (B, K) int8, scales (B,)
  float32)`. A tensor can have any dtype (bf16, i32, ...): its row bytes are the committed codes, with scale +0.0f. Every
  table has the batch's B rows.
- CUDA tensors go to `verity_vllm.commit.row_tables_device.step_roots` (`row_table.cu`, on a side stream). Host arrays go to
  its CPU reference, `row_tables.step_roots`.
- `rows` (optional) holds the row ids `(request, token row)`. By default slot b is `("%08d" % b, 0)`, so a table's leaf
  order is slot order.

KCOMMIT's fixture for CMP-95 item 3 wasn't posted when this was written, so the signature is the obvious one,
`commit_step(unit_outputs, salt_source) -> root`, with the root on `StepCommit.root`.

## What it commits

- **Rows and leaves.** Each unit output (`op_path`) is one table, and each of its rows is one sequence's output, committed
  as `row_table.s8_row`: the codes, then the scale. Each row is one `hm96-sha512/row-seg/v1` hiding leaf. These are not the
  `row/v2` rows (u16 words) that the drawn-unit replay bundle and C-Flock's row ports read. A prover that opens these
  leaves reads row-seg/v1.
- **Trees.** A table's leaves form a `frame-v3-sha512` tree under `row_table.table_domain`. The segment's root is the tree
  over its tables' roots (`row_table.step_tree`).
- **Salts.** Salts come from ChaCha20 under the run's row key, with the segment as nonce
  (`ml.windows.rows.segment_salts`).

At shape H's planned batch (20,480 sequences) a step's unit outputs are 717 GB, or 490 GB without compute-accounting's fused
cuts (`shape_h`, pinned in the test). That can't all be resident. So the loop commits each layer as it finishes, as its own
segment (`segment_of`: 52 layers and the head make 53 segments a step), and records each segment's root with the challenger.
This is how vLLM's engine steps are committed already: a window's `row-table` port is the tree over its segments' roots.

## Opening

`StepCommit` keeps every table's root. `open_rows(commit, op_path, table, salt, batch_rows)` rebuilds that one table on the
CPU and opens chosen ranks against the segment's root (`row_table.StepOpening`, checked by `row_table.verify_step`).

The by-table layout needs that table's rows for **every** sequence in the batch, so opening one drawn unit means replaying
the whole batch at that step. The slot layout (HASH-13) fixes this: a drawn unit opens from its own request's replay.

## Checks and the bench

- `tests/test_commit_fb.py` runs on the CPU in the suite. It checks that the call's root is
  `verity.ml.commitments.row_table.commit_step`'s from the stated rows, salts and domains, that openings verify, and it pins
  shape H's bytes.
- `check_roots.py` runs on the GPU. It checks that the device roots equal `row_tables.step_roots`, the C++ twin `host_step`,
  and (in the small cases) `row_table.commit_step`, on random unit outputs at shape H's widths. It also checks that openings
  verify against the device root.
- `bench.py` runs on the GPU. It measures one layer's GB/s, a whole step streamed as 53 segments (every op, then fused), and
  the hashing beside int8 GEMMs. It reports each step's share of compute-accounting's shape-H step (7.5 s staged, 3.7 s
  overlapped).
- `run.sh` runs both under one `gpu-lease`, with the command in its header.
