# pouw_decode

Pearl-C -h3's hashing per decode call (`pearl-c-sm120-v1-h3`, vLLM's `pouw` option) on one RTX PRO 6000, measured
against Pearl-C's own launches, and pd's launches that make the same bytes in fewer, shallower launches. Nothing here is
served: pd is a candidate for compute accounting, whose code (`benchmarks/pouw/pearl_c_sm120`, `benchmarks/pouw/pearl_c`,
`verity/core/protocols/pouw`, vLLM's `pouw_pearl_c_*`) this directory reads, builds and imports, and never edits.

- `pd.cuh` (`pd.cu`): the launches.
  - `pd_commit` makes A's commitment in one launch, one CTA per row. In parallel it hashes the row's segments, its hm96
    salt, its frame leaf's first block and the tree's node templates. It then folds the row digest, makes the hm96 leaf,
    the frame leaf, the row seed and the line key. In mode 0 the last CTA to finish also builds the tree's levels, seed_A
    and line_a.
  - `pd_tree` is mode 0's tail as a launch of its own, for a side lane.
  - `pd_tiles` makes the tile leaves, one CTA per tile.
  - `pd_probe` measures one warp's cycles per compression.
- `pd_host.cpp`: the same source run on the CPU (`pd_run`), with Pearl-C's own hash kernels, for `host_test.py`.
- `refs.py`: the Python references (`pearl_c.row_digest`, `row_seed`, `unit_seed_a`, `line_key` and `leaf`, and
  `audit.HM96`'s tree).
- `host_test.py`: pd's launches against Pearl-C's kernels and the Python references on the CPU: pads, odd segment counts,
  both modes and both folds, every output poisoned first.
- `bench.py`: per Llama-3.1-8B linear and decode's m16, on the GPU:
  - the check: pd against Pearl-C byte for byte, with Python references, a poison control and a stray-write control;
  - every captured schedule's own bytes (`graph_check`);
  - graphs of 8 calls per schedule, timed interleaved rep by rep with the SM clock;
  - `--trace` for nsys.
- `pd_host.py`: pd's launches on a `run.Pipeline`'s buffers, and the checks.
- `trace.py`: the nsys trace per call and launch: duration, idle gap, overlap, grid and depth.
- `run.sh`: all of the above on a node, outputs under `$OUT`.

~~~bash
research run --on vy-nebius-2 --project verity --campaign hash-loop --source . --cwd source \
  --env GPU_LEASE_WHO=kernels-pouw-decode -- bash benchmarks/hash/pouw_decode/run.sh
~~~

Schedules (`bench.Shape.schedules`): `ref_serial` is the served call, `ref_nohash` the call without hashing, and
`ref_deferred` the served launches with the tile leaves on a side lane. `pd_serial` and `pd_serial_rs_fold` are pd's
call on one stream, with the two folds. `pd_deferred` puts pd's tile leaves on a side lane. `pd_side_tree` also puts A's
tree and seed_A on a side lane; its bytes are unchanged only when m16 is a multiple of 64. `pd_side_tree_proposal` is
the same schedule at other m16, where filler rows read line_a, so it is timed but its E_A lines are not served bytes.
