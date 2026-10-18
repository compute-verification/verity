# hash/structure: structural levers for hm96-sha512 commitments, SHA-512 kept

The hash lane's measurements of what structure buys today's commitment (hm96-sha512 tile leaves in frame-v3-sha512 trees):
fusing the leaf hash with the GEMM that produces the values, salts generated ahead of the step, leaf size and tree arity.
Results and verdicts are in the hash lane's notes (`lanes/hash/*-draft-structure.md`); runs are in the evidence store.

- `fuse.cu`, one binary with these commands:
  - `check`, `bench`, `diag`: a persistent s8 GEMM (mma.sync m16n8k32, 128 x 128 x 64 tiles, 8 MMA warps, one CTA an SM)
    hashed three ways while it runs:
    - 4 or 8 hash warps in its own CTAs;
    - a persistent `hash_kernel` beside it, consuming tiles as their flags turn;
    - the separate leaf kernel on a second stream.

    `diag` also takes the in-CTA kernel apart and runs cuBLASLt's GEMM beside the separate pass. The committed operand
    is C's bf16 byte image under commit_gpu's tile layout. The leaves and the tree are
    `benchmarks/pouw/commit_gpu/commit_gpu.cuh`'s own code, built from that path, unchanged.
  - `probe`: tensor-core warps (mma.sync; on sm_90a also wgmma) beside SHA-512 warps on one SM, registers only. It shows
    how much the pipes overlap, apart from any GEMM.
  - `tree`: a-ary trees (a changed node format, so timing only).
  - `chacha`: ChaCha20 salts.
- The jobs. Each runs its checks (every timed arm's levels against the separate pass and the committer of record,
  `commit_gpu/reference.py verify` against `verity.ml.commitments`, ChaCha20 against RFC 8439) before it times:
  - `run.sh`: everything.
  - `diag.sh`: the fused arms taken apart.
  - `probe.sh`: the pipe probe and the GEMM kernels' SASS opcode mix.
- `summary.py OUT`: `run.sh`'s tables.
- `rank.py OUT...`: the levers ranked by what each buys a Qwen3-8B prefill step.

~~~text
research run --on vy-nebius-2 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-structure \
  -- bash benchmarks/hash/structure/run.sh
~~~
