# hash/structure: structural levers for hm96-sha512 commitments, SHA-512 kept

The hash lane's measurements of what structure buys today's commitment (hm96-sha512 tile leaves in frame-v3-sha512 trees):
fusing the leaf hash into the GEMM that produces the values, salts generated ahead of the step, leaf size and tree arity.
Results and verdicts are in the hash lane's notes (`lanes/hash/*-draft-structure.md`); runs are in the evidence store.

- `fuse.cu`: a persistent s8 GEMM (mma.sync m16n8k32, 128 x 128 x 64 tiles, 8 MMA warps) whose CTAs also run 4 or 8 hash
  warps that hash each finished tile's leaves while the MMA warps compute the next tile. The committed operand is C's bf16
  byte image under commit_gpu's tile layout; the leaves and tree are `benchmarks/pouw/commit_gpu/commit_gpu.cuh`'s own
  code (built from that path, unchanged). Also a-ary tree timing (a changed node format: timing only) and ChaCha20 salts.
- `run.sh`: the job. Builds outside the lease; lease 1 writes check dumps; `commit_gpu/reference.py verify` holds every dump
  to `verity.ml.commitments`, and `check_chacha.py` holds the keystream to RFC 8439; lease 2 times, only if all passed.
- `summary.py OUT`: the tables.

~~~text
research run --on vy-nebius-2 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-structure \
  -- bash benchmarks/hash/structure/run.sh
~~~
