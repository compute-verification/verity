# hash_gpu -- GPU Merkle commitments for the proof-internal commitment layer

Shared component (hash-gpu lane, 2026-09-22) for the Candidate A / B GPU lanes: commit an `(R, N)` matrix of BabyBear
elements column-wise (Ligero layout: one leaf per codeword column, binary tree over the `N` leaves) with SHA-256, Blake3 or
Poseidon2/BabyBear on the device, and open columns with authentication paths. Every GPU hash is bit-exact against a CPU
reference. This is *not* Verity's external commitment hash (`verity.commitments.merkle`, unchanged).

Measured rates (RTX 4090, run `r20260922-054241-b966`) are in `backends/numerical/reports/hash_gpu.md`; headline: full
tree over Candidate B's 20.36 GB commitment in 45.6 ms with SHA-256 (446 GB/s), 56.6 ms with Blake3 (359 GB/s), 353 ms with
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
| `reprice.py` | feeds measured rates into `explore.tensor.Priced.times` -> `backends/numerical/reports/hash_gpu.json` |
| `tests/test_hash_gpu.py` | reference vectors (CPU, run anywhere) + GPU-vs-CPU bit-exactness (skipped without CUDA) |

Digest conventions: every digest is `(n, 8)` `uint32`. SHA-256 words are big-endian, Blake3 little-endian (so `digest_bytes`
reproduces `hashlib.sha256(...).digest()` / `blake3(...).digest()`), Poseidon2 digests are 8 field elements. Leaf functions:
`sha256` hashes the column's bytes (elements as little-endian u32) sequentially in one thread; `sha256-chunked` hashes 1 KiB
sub-chunks in parallel and combines them with a binary SHA-256 tree (a different, labelled leaf function, 1.4x faster);
`blake3` is standard Blake3 of the column bytes; `poseidon2-16` / `poseidon2-24` build a 2:1 compression tree over the
column's 8-element blocks. Nodes: `H(left || right)` / `compress(left, right)`. An unpaired rightmost node is promoted.

## API

~~~python
import sys; sys.path.insert(0, "backends/shared")          # importable by path
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
uv run --frozen --package verity-numerical python -m pytest backends/shared/hash_gpu/tests -q

# the measurement, recorded on the pod (commit first; the snapshot is HEAD)
uv run --frozen research run --on vy-g2 --project verity --stage hash_gpu.bench --require-result -- \
  bash -c 'cd backends/shared && python3 -m hash_gpu.bench --reps 5 --out "$RESEARCH_RUN_DIR/bench.json"'
uv run --frozen research fetch <run-id> --all

# re-price explore.tensor with the fetched rates
uv run --frozen --package verity-numerical python backends/shared/hash_gpu/reprice.py \
  --bench ~/.research/runs/<run-id>/bench.json --run-id <run-id> --out backends/numerical/reports/hash_gpu.json
~~~

Pod requirements: `cupy-cuda12x`, `blake3`, torch with CUDA, `nvcc` on `PATH` (cupy compiles the kernels with NVRTC on first
use; `/usr/local/cuda-12.4/bin` on vy-g2).

## Results (RTX 4090, driver 580.159.04, CUDA 12.9 runtime, torch 2.4.1+cu124, cupy 14.2.0; run `r20260922-054241-b966`)

Ligero shapes at B = 4096, K = 1536, BabyBear, 2^-128: N = 16,384 columns; Candidate B R = 310,602 rows (20.36 GB),
Candidate A R = 48,873 rows (3.20 GB). Full tree build, median of 5.

| hash | B GB/s | B ms | B us/VU | A GB/s | A ms | peak device mem (B shape) |
|---|---|---|---|---|---|---|
| sha256 | 446 | 45.6 | 11.1 | 425 | 7.5 | 19.6 GiB (incl. 19 GiB input) |
| sha256-chunked | 624 | 32.6 | 8.0 | 603 | 5.3 | 21.1 GiB |
| blake3 | 359 | 56.6 | 13.8 | 347 | 9.2 | 20.6 GiB |
| poseidon2-16 / cuda | 57.7 | 352.6 | 86 | 55.5 | 57.8 | 21.3 GiB |
| poseidon2-24 / cuda | 34.2 | 595.9 | 146 | 33.0 | 97.0 | 21.3 GiB |
| poseidon2-16 / torch blocks (int64) | 0.38 | 53,050 (x16 from a 1,024-col slice) | 12,950 | 0.36 | 8,974 | -- |
| poseidon2-16 / torch int8 (`_int_mm`) | 0.27 | 76,400 (x16 from a 1,024-col slice) | 18,660 | 0.25 | 12,640 | -- |

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

Re-pricing (`reprice.py`, utilisation 0.172): with the measured SHA-256 rate the hash bucket is 9% of Candidate B's time on the
4090 (the model's assumed 50 GB/s made it 70%); B becomes matmul-bound. With Poseidon2-16 it is 44%. A100/H100 rows in
`hash_gpu.json` scale the 4090 rate by the bandwidth ratio (x2.02 / x3.32) and are labelled as a SCALING -- an upper bound,
since the SHA-256/Blake3 kernels are compute-bound at 1 TB/s.

Recorded runs: `r20260922-054241-b966` (full measurement, this table). Earlier exploratory runs on vy-g2 during development
(shape/OOM fixes) are superseded by it.
