# pouw_decode

Pearl-C -h3's hashing per decode call (`pearl-c-sm120-v1-h3`, vLLM's `pouw` option) on one RTX PRO 6000, measured
against Pearl-C's own launches, and pd's launches that make the same bytes in fewer, shallower launches. Nothing here is
served: pd is a candidate for compute accounting, whose code (`benchmarks/pouw/pearl_c_sm120`, `benchmarks/pouw/pearl_c`,
`verity/core/protocols/pouw`, vLLM's `pouw_pearl_c_*`) this directory reads, builds and imports, and never edits.

- `pd.cuh` (`pd.cu`): the launches.
  - `pd_commit` makes A's commitment in one launch, one CTA per row. In parallel it hashes the row's segments, its hm96
    salt (across a warp), its frame leaf's first block and the tree's node templates. It then folds the row digest, makes
    the hm96 leaf, the frame leaf, the row seed, its line key and the row's E_A line (the served call's `lines_ea`). In
    mode 0 the last CTA to finish also builds the tree's levels (kept in shared memory), seed_A, line_a and E_A's filler
    lines. Paired, a CTA counts itself finished once its leaf is made, so the last CTA's tail starts while its row seed
    chain, which is a compression longer than the leaf and which the tail doesn't read, is still running.
  - `pd_tree` is mode 0's tail as a launch of its own, for a side lane.
  - `pd_tiles` makes the tile leaves, one CTA per tile.
  - `pd_probe` measures one warp's cycles per compression, warm and on its code's first fetch.
  - Every launch is built once per variant (`PD_VARIANT_LIST`: `pd_commit_V` and so on), each with its own code and
    registers. The bits: `PD_ROUNDS` (1), the rounds-only fold, where the folding warp's lanes compute every block's
    schedule first; `PD_ROLLED` (2), compressions with one 16-round body run five times, a fifth of the code;
    `PD_CALL` (4), one copy of each compression per kernel, called from every site instead of inlined at each;
    `PD_PREFETCH` (8), each segment's lines prefetched into L1 before its first block; `PD_PAIR` (16), every serial
    compression on the critical path paired, a producer warp computing its message schedule's round inputs a chunk ahead
    while the consumer warp runs the rounds (named barriers, 48 KB of dynamic shared memory); `PD_STAMP` (32; variant
    51 is 19 with it), the global timer and the SM clock at each phase's end of `pd_commit` and `pd_tree`, written to
    p13; `PD_WHOLE` (64, a launch setting alone: variant 83 is 19 with it), the paired `pd_commit`
    and `pd_tree` launched with all the dynamic shared memory a CTA may take, so that no other CTA (a side lane's tile
    leaves) shares their SMs and slows their one-warp chains; `PD_GROUP` (128, with
    `PD_PAIR`; variant 211 is 83 with it), `pd_tiles` with four tiles a CTA, their folds together on one warp's lanes,
    paired, so a fold's lone lane no longer holds a warp's issue slots for its 9 or 17 compressions; `PD_LOOP` (256,
    with `PD_PAIR`; variants 307, 339 and 467 are 51, 83 and 211 with it), the hm96 leaf and frame leaf, a seed and its
    line key, and a node's two blocks each through one paired site in a rolled loop, and the producer warp's loop one
    called copy, so the last CTA's tail fetches 5 paired sites cold on its SM instead of 10 (an inlined paired site is
    about 7 KB, the producer's 23 KB). The counter's fences are acquire-release (`pd_fence`), not `__threadfence`'s
    sequentially consistent one. Pearl-C's unrolled
    compression is about 3,500 instructions (56 KB), and a call's serial compressions are each at a site of their own,
    so inlined they are each fetched cold. On its thread the schedule is a third of a compression's cycles.
- `pd_host.cpp`: the same source run on the CPU (`pd_run`), with Pearl-C's own hash kernels, for `host_test.py`.
- `refs.py`: the Python references (`pearl_c.row_digest`, `row_seed`, `unit_seed_a`, `line_key`, `sample_line` and
  `leaf`, and `audit.HM96`'s tree).
- `host_test.py`: pd's launches against Pearl-C's kernels and the Python references on the CPU: pads, odd segment counts,
  one and two segment warps, both modes and every variant, every output poisoned first.
- `bench.py`: per Llama-3.1-8B linear and decode's m16, on the GPU:
  - the check: pd in every variant against Pearl-C byte for byte, with Python references, a poison control and a
    stray-write control;
  - every captured schedule's own bytes (`graph_check`);
  - graphs of 8 calls per schedule, timed interleaved rep by rep with the SM clock;
  - for a `PD_STAMP` variant, its serial graph once more and the last call's stamps (`stamps` records), which
    `report.py` turns into the tail CTA's phases: microseconds per phase and per compression;
  - `--trace` for nsys.
- `pd_host.py`: pd's launches on a `run.Pipeline`'s buffers, and the checks.
- `trace.py`: the nsys trace per call and launch: duration, idle gap, overlap, grid, depth and lane.
- `report.py`: a run's tables: hashing per call before and after, the per-launch critical path, the projected decode
  step.
- `run.sh`: all of the above on a node, outputs under `$OUT`; with `NCU=` a regex of launch names, bench.py at one
  shape under ncu too, in the same lease after the timing and the trace.

~~~bash
research run --on vy-nebius-2 --project verity --campaign hash-loop --source . --cwd source \
  --env GPU_LEASE_WHO=kernels-pouw-decode -- bash benchmarks/hash/pouw_decode/run.sh
~~~

Schedules (`bench.Shape.schedules`): `ref_serial` is the served call, `ref_nohash` the call without hashing, and
`ref_deferred` the served launches with the tile leaves on a side lane. `pd_serial_V` is pd's call on one stream in
variant V, for every variant. For the variants in `PD_SIDE` (default 19, 83, 211, 339 and 467), `pd_deferred_V` puts pd's tile leaves on a
side lane, and `pd_side_tree_V` also puts A's tree and seed_A on a side lane; its bytes are unchanged only when m16 is a
multiple of 64. `pd_side_tree_V_proposal` is the same schedule at other m16, where filler rows read line_a, so it is
timed but its E_A lines are not served bytes.
