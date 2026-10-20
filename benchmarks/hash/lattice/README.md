# SIS commitment hashes on the tensor cores

What an SIS (Ajtai) hash h(x) = A x mod q costs natively on a GPU's integer tensor cores and in a Boolean circuit, beside
SHA-512 on the same die. A measurement only: nothing here is on the served path, where the commitment hash is SHA-512.

- `params.py`: Core-SVP parameters (M1 lattice reduction, M2 the Dilithium-style infinity-norm model, Wagner) for each
  encoding at 128, 96 and 80 bits, with MACs per byte, ANDs per byte, digest and key size. `--check` reproduces published
  numbers (ajtai-design, SWIFFT, Buchmann–Lindner) first. Not the lattice-estimator, which needs Sage.
- `sis_bench.cu`: the hash through cuBLASLt int8 GEMMs and through one fused `mma.sync` kernel, checked bit for bit before
  anything is timed, then timed rep by rep beside SHA-512 (`benchmarks/pouw/commit_gpu/sha512.cuh`, included read-only).
- `reference.py`: an independent numpy/hashlib reference run on every row set `sis_bench` dumps.
- `gpu.sh`: one `research run` job: builds `sis_bench` and commit_gpu for the GPU, runs every config under one
  `gpu-lease`, then `reference.py` and `summary.py`.

## The hash

A row is m bytes x_0..x_{m-1}. Its digits s_k are the bytes as int8 (encodings `q32`, `q16`; K = m) or its nibbles, low
first (`q8n`: s_2i = x_i & 15, s_2i+1 = x_i >> 4; K = 2m). A is L limbs of n x K int8 (L = 4, 2, 1 for q = 2^32, 2^16, 2^8),
read row-major from the SHA-512 counter stream of the tag `verity/hash-lattice/sis/v0|ENC|n|K` (block i = SHA-512(tag ‖
i as u64 big-endian)): limb l, row j, column k is stream byte (l n + j) K + k. Then

    h_j = sum_l 2^(8l) sum_k A_l[j][k] s_k  mod 2^(8L)

stored little-endian, L bytes a word: n L bytes a digest. Each int8 limb product is exact in int32 for K up to 2^17.

## Running

```text
research run --on vy-dc-h200 --project verity --campaign hash --source . --cwd source \
  --env GPU_LEASE_WHO=NAME [--env CONFIGS="q32:16384:256:fused q8n:16384:384:fused"] -- bash benchmarks/hash/lattice/gpu.sh
python3 benchmarks/hash/lattice/params.py --check
```

A config is `ENC:M_BYTES:N_WORDS[:tree][:fused]`. `tree` adds an all-SIS tree (each level's digests are the next level's
rows); `fused` adds the one-pass kernel, which needs L n a multiple of 128. `OUT/summary.md` has the table: SIS, fused and
GEMM-only GB/s of committed input, SHA-512 over the same rows, and the whole data phase of a commitment three ways
(all-SHA, SIS then SHA-512 over the digest, all-SIS).
