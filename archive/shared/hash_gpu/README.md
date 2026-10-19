# hash_gpu -- GPU Merkle commitments for the proof-internal commitment layer

Shared component (hash-gpu lane, 2026-09-22) for the A-GKR / B-Ligero GPU lanes: commit an `(R, N)` matrix of BabyBear
elements column-wise (Ligero layout: one leaf per codeword column, binary tree over the `N` leaves) with SHA-256, Blake3 or
Poseidon2/BabyBear on the device, and open columns with authentication paths. Every GPU hash is bit-exact against a CPU
reference. This is *not* Verity's external commitment hash (`verity.ml.commitments.merkle`, unchanged).

Measured rates (RTX 4090, run `r20260922-054241-b966`) are in `note:r20-proof/hash-gpu/20260922T0554Z-report-hash-gpu`; headline: full
tree over B-Ligero's 20.36 GB commitment in 45.6 ms with SHA-256 (446 GB/s), 56.6 ms with Blake3 (359 GB/s), 353 ms with
Poseidon2-16 (58 GB/s); the torch Poseidon2 variants (int64 elementwise, INT8 tensor-core MDS) are 0.3-0.4 GB/s.

## Layout

| file | what |
|---|---|
| `merkle.py` | `commit(matrix, hash) -> Tree`, `leaves`, `node_level`, `Tree.open`, `Tree.opening_bytes`, `digest_bytes` |
| `sha256_cuda.py` | cupy `RawKernel` SHA-256 (FIPS 180-4, structured after Brad Conte's public-domain `crypto-algorithms/sha256.c`); strided message layouts, `__funnelshift_r` rotates |
| `blake3_cuda.py` | cupy `RawKernel` Blake3 (from the spec / `reference_impl.rs`): 1 KiB chunks in parallel + parent kernel, so a column's leaf equals the `blake3` package |
| `poseidon2_ref.py`, `poseidon2_grain.py` | pure-Python Poseidon2 over BabyBear, widths 16 and 24, Plonky3 constants (Grain LFSR + seeded Xoroshiro128Plus) checked against Plonky3's test vectors |
| `poseidon2_cuda.py` | fused CUDA permutation, one thread per state, Montgomery arithmetic; `leaf_tree` / `merkle_level` |
| `poseidon2_torch.py` | torch permutation with three MDS strategies: `blocks` (int64 shift-add), `fp64` (float64 matmul), `int8` (`torch._int_mm`, 4 INT8 limbs per element, INT32 accumulate, deferred reduction) |
| `reference.py` | CPU leaf / node / tree / `verify_path` for all hashes (`hashlib`, `blake3`, `poseidon2_ref`) |
| `bench.py` | the recorded measurement (CUDA events, warm, median of `--reps`) |
| `reprice.py` | feeds measured rates into `explore.tensor.Priced.times` -> `notes-asset:campaigns/r20-proof/assets/hash-gpu/reports/hash_gpu.json` |
| `tests/test_hash_gpu.py` | reference vectors (CPU, run anywhere) + GPU-vs-CPU bit-exactness (skipped without CUDA) |

Digest conventions: every digest is `(n, 8)` `uint32`. SHA-256 words are big-endian, Blake3 little-endian (so `digest_bytes`
reproduces `hashlib.sha256(...).digest()` / `blake3(...).digest()`), Poseidon2 digests are 8 field elements. Leaf functions:
`sha256` hashes the column's bytes (elements as little-endian u32) sequentially in one thread; `sha256-chunked` hashes 1 KiB
sub-chunks in parallel and combines them with a binary SHA-256 tree (a different, labelled leaf function, 1.4x faster);
`blake3` is standard Blake3 of the column bytes; `poseidon2-16` / `poseidon2-24` build a 2:1 compression tree over the
column's 8-element blocks. Nodes: `H(left || right)` / `compress(left, right)`. An unpaired rightmost node is promoted.

## API

~~~python
import sys; sys.path.insert(0, "verity/ml/kernels")          # importable by path
import cupy as cp
from hash_gpu import merkle

m = cp.random.randint(0, 0x78000001, (48873, 16384), dtype=cp.uint32)   # (R, N), row-major, on device
t = merkle.commit(m, "sha256")                 # "sha256" | "sha256-chunked" | "blake3" | "poseidon2-16" | "poseidon2-24"
t.root                                         # (8,) uint32;  merkle.digest_bytes(t.root, "sha256") -> 32 bytes
t.levels                                       # [leaves (N, 8), ..., root (1, 8)]
paths = t.open([3, 17, 4095])                  # sibling digests per level, leaf level first (None where promoted)
t.opening_bytes(600)                           # bytes of 600 column openings (columns + paths)

t = merkle.commit(m, "poseidon2-16", backend="cuda")                 # fused kernel (default)
t = merkle.commit(m, "poseidon2-16", backend="torch", mds="int8")     # torch; mds in {"blocks", "fp64", "int8"}
~~~

Torch tensors (`int32`/`uint32`, and `int64` for Poseidon2) are accepted wherever cupy arrays are. The tree levels are
cupy arrays (torch for the torch Poseidon2 backend).

## Running

~~~bash
# CPU reference tests (seconds, laptop-safe); GPU tests run too where CUDA is present
uv run --frozen --package verity-numerical python -m pytest verity/ml/kernels/hash_gpu/tests -q

# the measurement, recorded on the pod (commit first; the snapshot is HEAD)
uv run --frozen research run --on vy-g2 --project verity --stage hash_gpu.bench --require-result -- \
  bash -c 'cd kernels && python3 -m hash_gpu.bench --reps 5 --out "$RESEARCH_RUN_DIR/bench.json"'
uv run --frozen research fetch <run-id> --all

# re-price explore.tensor with the fetched rates
uv run --frozen --package verity-numerical python verity/ml/kernels/hash_gpu/reprice.py \
  --bench ~/.research/runs/<run-id>/bench.json --run-id <run-id> --out backends/numerical/reports/hash_gpu.json
~~~

Pod requirements: `cupy-cuda12x`, `blake3`, torch with CUDA, `nvcc` on `PATH` (cupy compiles the kernels with NVRTC on first
use; `/usr/local/cuda-12.4/bin` on vy-g2).

## Results (RTX 4090, driver 580.159.04, CUDA 12.9 runtime, torch 2.4.1+cu124, cupy 14.2.0; run `r20260922-054241-b966`)

Ligero shapes at B = 4096, K = 1536, BabyBear, 2^-128: N = 16,384 columns; B-Ligero R = 310,602 rows (20.36 GB),
A-GKR R = 48,873 rows (3.20 GB). Full tree build, median of 5.

| hash | B GB/s | B ms | B us/VU | A GB/s | A ms | peak device mem (B shape) |
|---|---|---|---|---|---|---|
| sha256 | 446 | 45.6 | 11.1 | 425 | 7.5 | 19.6 GiB (incl. 19 GiB input) |
| sha256-chunked | 624 | 32.6 | 8.0 | 603 | 5.3 | 21.1 GiB |
| blake3 | 359 | 56.6 | 13.8 | 347 | 9.2 | 20.6 GiB |
| poseidon2-16 / cuda | 57.7 | 352.6 | 86 | 55.5 | 57.8 | 21.3 GiB |
| poseidon2-24 / cuda | 34.2 | 595.9 | 146 | 33.0 | 97.0 | 21.3 GiB |
| poseidon2-16 / torch blocks (int64) | 0.38 | 53,050 (x16 from a 1,024-col slice) | 12,950 | 0.36 | 8,974 | -- |
| poseidon2-16 / torch int8 (`_int_mm`) | 0.27 | 76,400 (x16 from a 1,024-col slice) | 18,660 | 0.25 | 12,640 | -- |

**A100 rows** (anchor-a100 lane, NVIDIA A100-SXM4-80GB, driver 580.126.16, torch 2.6.0+cu124, cupy 14.2.0; run
`r20260922-063811-e561`, same bench, `--reps 7`; full table and floors in `note:r20-proof/anchor-a100/20260922T0820Z-report-anchor-a100`):

| hash (A100) | B GB/s | B ms | B us/VU | A GB/s | A ms | vs 4090 |
|---|---|---|---|---|---|---|
| sha256 | 190 | 107.1 | 26.1 | 186 | 17.2 | 0.43x |
| sha256-chunked | 344 | 59.1 | 14.4 | 332 | 9.7 | 0.55x |
| blake3 | **377** | 54.0 | 13.2 | 356 | 9.0 | 1.05x |
| poseidon2-16 / cuda | 26.1 | 780.6 | 191 | 26.1 | 122.8 | 0.45x |
| poseidon2-24 / cuda | 13.3 | 1,528 | 373 | 13.3 | 240.5 | 0.39x |
| poseidon2-16 / torch blocks (int64) | 0.60 | 34,141 (x16) | 8,335 | 0.50 | 6,369 | 1.55x |
| poseidon2-16 / torch int8 | 0.41 | 49,664 (x16) | 12,125 | 0.37 | 8,733 | 1.54x |

The SIMT hashes do not scale with bandwidth: SHA-256 and Poseidon2 run at ~0.45x the 4090 (the A100's INT32 rate is 0.47x:
108 SMs x 64 lanes x 1.41 GHz vs 128 x 64 x 2.52 GHz), Blake3 holds its rate.  Blake3 is the fastest proof-internal hash on the
anchor; the "A100 = 4090 x 2.02" scaling rows below are superseded by these measurements.  Generic trees on the A100: SHA-256
2.2 G leaves/s, Blake3 2.7 G, Poseidon2-16 0.40 G, Poseidon2-24 0.21 G at 2^27; permutation w16 796 M perm/s, w24 397 M.

Generic trees, 32-byte leaves (tree-dominated), leaves/s at 2^20 / 2^23 / 2^25 / 2^27 leaves:

| hash | GB/s (2^27) | M leaves/s 2^20 | 2^23 | 2^25 | 2^27 | mem at 2^27 |
|---|---|---|---|---|---|---|
| sha256 | 146 | 2,592 | 4,474 | 4,565 | 4,577 | 14.5 GiB |
| blake3 | 87 | 2,202 | 2,658 | 2,720 | 2,734 | 14.5 GiB |
| poseidon2-16 / cuda | 24 | 787 | 858 | 838 | 750 | 14.5 GiB |
| poseidon2-24 / cuda | 15 | 461 | 536 | 463 | 468 | 14.5 GiB |

Poseidon2 permutation throughput (2^22 states, fused CUDA): width 16 1.68 G perm/s, width 24 0.97 G perm/s. Torch, 2^20
states, width 16: `blocks` 12.2 M perm/s (MDS-only 4.4 ms), `fp64` 16.4 M (1.9 ms), `int8` 8.4 M (8.6 ms) -- the tensor-core
INT8-limb MDS does **not** beat the int64 elementwise MDS; the limb split, recombination and deferred reduction cost more than
the 16x16 multiply they remove.

Re-pricing (`reprice.py`, utilisation 0.172): with the measured SHA-256 rate the hash bucket is 9% of B-Ligero's time on the
4090 (the model's assumed 50 GB/s made it 70%); B becomes matmul-bound. With Poseidon2-16 it is 44%. A100/H100 rows in
`notes-asset:campaigns/r20-proof/assets/hash-gpu/reports/hash_gpu.json` scale the 4090 rate by the bandwidth ratio (x2.02 / x3.32) and are labelled as a SCALING -- an upper bound,
since the SHA-256/Blake3 kernels are compute-bound at 1 TB/s.

Recorded runs: `r20260922-054241-b966` (full measurement, this table). Earlier exploratory runs on vy-g2 during development
(shape/OOM fixes) are superseded by it.
