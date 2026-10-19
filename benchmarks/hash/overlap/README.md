# Hashing a prefill step under its GEMMs

Can the hm96-sha512 hash of a prefill step's committed tensors (frame-v3-sha512 trees, 4 KiB row leaves, salts read
ahead, per-Call fused units) run beside the step's GEMMs on the same GPU and hide there? `overlap.cu` times, at the
Qwen3-8B prefill shapes (M 2,560: qkv, o, gate_up, down):

- cuBLASLt alone, the baseline, and CUTLASS sm90/sm100 GEMMs (`gemm_cutlass.cu`, configs from `GEMM_CFGS`) that leave a hash
  CTA's registers and shared memory free;
- the hash beside each GEMM on other streams: per shape, and as a chain of 4 layers with each call on its own stream (parN)
  against the separate pass at its throughput (hN);
- the hash server: one persistent CTA (or two) an SM pulling 128-leaf chunks as each unit's ready flag is set;
- a memory-bound stand-in for the step's non-GEMM kernels beside the separate pass.

Every hash arm's levels are checked on the device against `benchmarks/pouw/commit_gpu/`'s Committer, and its dumps go
through `benchmarks/pouw/commit_gpu/reference.py verify` (`verity.ml.commitments`), before anything is timed.

Run on a node, from a pushed commit:

~~~text
research run --on vy-dc-b200 --project verity --campaign hash --source . --cwd source --timeout 2h \
  --env GPU_LEASE_WHO=NAME --env 'GEMM_CFGS=u2_256x128:100:1:256:128:64:2:1:0:0:0:0 ...' -- bash benchmarks/hash/overlap/gpu.sh
~~~

`gpu.sh` fetches a pinned CUTLASS into scratch, builds, gates, times and writes `summary.md` (`summary.py`) under `$OUT`.
A `GEMM_CFGS` entry is `NAME:ARCH:KIND:TM:TN:TK:CM:CN:STAGES:MAXNREG:LOADR:MMAR`. `prefill_nsys.sh` and
`prefill_breakdown.py` split a served prefill step's GPU time by kernel class from an nsys trace.
