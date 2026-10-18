# dedup

What Pearl-C's -h3 step (`pearl-c-sm120-v1-h3`, vLLM's `pouw` option) would save if A's commitment came from Commit's
tree of the same activation instead of Pearl-C's own: the measured upper bound, on one RTX PRO 6000. It's a what-if:
nothing here is served and no protocol changes. The dedup's design belongs to compute accounting, with circuits. This
directory reads, builds and imports their code (`benchmarks/pouw/pearl_c_sm120`, `verity/core/protocols/pouw`, vLLM's
`pouw_pearl_c_*`), and never edits it.

- `step.py`: Llama-3.1-8B's 128 linear calls a step through the served executor (`PearlCDeviceRun`, subclassed), on
  synthetic weights. Each call's input is the glue of the calls before it. Decode (m = 32, 128) is one whole-forward
  CUDA graph, prefill (m = 2560, 8192) runs eagerly. The arms are today's call (`h3`) and the dedup variants:
  - A's own row digests and tree omitted, with A's root and digests taken from a Commit-style tree, either made before
    the step (`pre`) or made in the call (`inline`);
  - only the tree taken from Commit (`tree`);
  - `pre` and `inline` on o and down alone (`-od`);
  - no hashing (`nohash`).

  The checks run before timing:
  - `pre` against `inline`, bit for bit;
  - the Python references;
  - a control with the Commit buffers poisoned;
  - each arm's graph replay against its eager pass.

  Then the arms are timed interleaved, rep by rep, with the SM clock read each rep.
- `run.sh`: the node job, which builds the ship off the GPU, then runs Pearl-C's own kernel check and `step.py` in one
  lease.
- `table.py`: the upper-bound table from a run's `dedup.json`: each arm's median step against today's -h3, and the saving
  as a share of the step and of -h3's hashing.
- `time_branch.sh`: times another branch's -h3 arm (compute accounting's A-from-Commit) against today's on the PoUW workload,
  with STEP-SPLIT's windows (`served_profile.sh`'s GPU part, Llama-3.1-8B). It runs from that branch's tree, both arms of
  a window back to back in one lease, in ABBA order across windows.
