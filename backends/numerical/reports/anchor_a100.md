# anchor-a100: every campaign GPU kernel measured on the A100 (the anchor), 2026-09-22

**Scope: component measurements** (kernel time on random/patterned data, bit-exact against each lane's CPU reference), then one `ARITHMETIC_DIAGNOSTIC` re-price of A and B with those numbers.  Nothing here is a proof.  Every recorded 4090 / H100 number is the lane's own (its README / report, run id in the lane's table); every A100 number is a `research run --on vy-a100b --exclusive` of the same bench, same flags, CUDA events, warm, median of >= 7, bit-exact checks on.  `overhead.vs_native_peak` is defined against *this* card's dense BF16 peak (`contract.NATIVE_REFERENCE`, 9.85 ps per K=1536 VU), so the last section is the first A-vs-B comparison where the prover and the reference are the same machine.

Device: NVIDIA A100-SXM4-80GB, 580.126.16, 1410 MHz, 1593 MHz, 81920 MiB; torch 2.6.0+cu124, CUDA 12.4, 108 SMs (sm_80).  Recorded runs: microbench `r20260922-063608-a304`, hash_gpu `r20260922-063811-e561`.

Bandwidth ratios the earlier reports *scaled* by: A100/4090 = 2,039/1,008 = 2.02; A100/H100 = 2,039/3,350 = 0.61.  Speed ratios below are recorded-time / A100-time (> 1 = the A100 is faster); a departure from the bandwidth ratio by > 1.5x either way is a finding about the kernel and is flagged.

## 0. The anchor's measured roofline inputs

| microbench (CUDA events, median of 9) | A100 measured | datasheet | fraction |
|---|---|---|---|
| HBM copy (read+write), 1 GiB | 1754 GB/s | 2,039 GB/s | 0.86 |
| HBM copy, 4 GiB | 1761 GB/s | 2,039 GB/s | 0.86 |
| HBM write only (`fill_`) | 1820 GB/s | 2,039 GB/s | 0.89 |
| HBM read only (`sum`, int32 -> int64: a torch reduction, not a bandwidth number) | 332 GB/s | 2,039 GB/s | 0.16 |
| `_int_mm` 4096x4096x4096 | 29.2 T MAC/s (2.354 ms) | 312 T MAC/s | 0.094 |
| `_int_mm` 8192x8192x8192 | 37.8 T MAC/s (14.538 ms) | 312 T MAC/s | 0.121 |
| `_int_mm` 16384x16384x16384 | 38.0 T MAC/s (115.872 ms) | 312 T MAC/s | 0.122 |
| `_int_mm` 48x32768x96 | 0.1 T MAC/s (1.541 ms) | 312 T MAC/s | 0.000 |
| `_int_mm` 2097152x48x48 | 6.3 T MAC/s (0.764 ms) | 312 T MAC/s | 0.020 |
| BF16 matmul 8192^3 | 292 TFLOPS (3.77 ms) | 312 TFLOPS | 0.94 |
| BF16 matmul 16384^3 | 264 TFLOPS (33.30 ms) | 312 TFLOPS | 0.85 |

The floors below use the measured copy bandwidth **1754 GB/s** (0.86 of the datasheet's 2,039) and the datasheet INT8 peak 312 T MAC/s; cuBLAS `_int_mm` reaches only 38 T MAC/s (0.12 of it) on this card (the 4090 got 0.253 of its peak), while BF16 reaches 292 TFLOPS (0.94 of the peak that defines the native reference).  The native reference is therefore ~6% conservative on this card, not more.

## 1. `backends/shared/hash_gpu` -- Merkle commitment hashing (4090 run `r20260922-054241-b966`)

| hash (leaf function) | shape | RTX 4090 recorded | A100 measured | speed ratio A100/4090 | A100 floor | fraction of floor |
|---|---|---|---|---|---|---|
| sha256 | B | 446.4 GB/s (45.6 ms) | 190.1 GB/s (107.1 ms) | 0.43x | 11.6 ms (read once at 1754 GB/s) | 0.11 |
| sha256 | A | 424.6 GB/s (7.5 ms) | 186.2 GB/s (17.2 ms) | 0.44x | 1.8 ms (read once at 1754 GB/s) | 0.11 |
| sha256-chunked | B | 623.8 GB/s (32.6 ms) | 344.3 GB/s (59.1 ms) | 0.55x | 11.6 ms (read once at 1754 GB/s) | 0.20 |
| sha256-chunked | A | 603.4 GB/s (5.3 ms) | 331.8 GB/s (9.7 ms) | 0.55x | 1.8 ms (read once at 1754 GB/s) | 0.19 |
| blake3 | B | 359.4 GB/s (56.6 ms) | 377.1 GB/s (54.0 ms) | 1.05x | 11.6 ms (read once at 1754 GB/s) | 0.22 |
| blake3 | A | 346.9 GB/s (9.2 ms) | 355.6 GB/s (9.0 ms) | 1.03x | 1.8 ms (read once at 1754 GB/s) | 0.20 |
| poseidon2-16 / cuda | B | 57.7 GB/s (352.6 ms) | 26.1 GB/s (780.6 ms) | 0.45x | 11.6 ms (read once at 1754 GB/s) | 0.01 |
| poseidon2-16 / cuda | A | 55.5 GB/s (57.8 ms) | 26.1 GB/s (122.8 ms) | 0.47x | 1.8 ms (read once at 1754 GB/s) | 0.01 |
| poseidon2-24 / cuda | B | 34.2 GB/s (595.9 ms) | 13.3 GB/s (1,528.1 ms) | 0.39x | 11.6 ms (read once at 1754 GB/s) | 0.01 |
| poseidon2-24 / cuda | A | 33.0 GB/s (97.0 ms) | 13.3 GB/s (240.5 ms) | 0.40x | 1.8 ms (read once at 1754 GB/s) | 0.01 |
| poseidon2-16 / torch blocks | B | 0.4 GB/s (53,050.4 ms) | 0.6 GB/s (34,140.6 ms) | 1.55x | 11.6 ms (read once at 1754 GB/s) | 0.00 |
| poseidon2-16 / torch blocks | A | 0.4 GB/s (8,974.3 ms) | 0.5 GB/s (6,368.7 ms) | 1.41x | 1.8 ms (read once at 1754 GB/s) | 0.00 |
| poseidon2-16 / torch int8 | B | 0.3 GB/s (76,402.3 ms) | 0.4 GB/s (49,663.5 ms) | 1.54x | 11.6 ms (read once at 1754 GB/s) | 0.00 |
| poseidon2-16 / torch int8 | A | 0.3 GB/s (12,640.2 ms) | 0.4 GB/s (8,733.2 ms) | 1.45x | 1.8 ms (read once at 1754 GB/s) | 0.00 |

Generic trees (32-byte leaves, tree-dominated):

| leaves | hash | RTX 4090 | A100 | ratio |
|---|---|---|---|---|
| 2^20 | sha256 | 2,592 M leaves/s | 1,548 M leaves/s | 0.60x |
| 2^20 | blake3 | 2,202 M leaves/s | 2,004 M leaves/s | 0.91x |
| 2^20 | poseidon2-16 | 787 M leaves/s | 346 M leaves/s | 0.44x |
| 2^20 | poseidon2-24 | 461 M leaves/s | 184 M leaves/s | 0.40x |
| 2^23 | sha256 | 4,474 M leaves/s | 2,071 M leaves/s | 0.46x |
| 2^23 | blake3 | 2,658 M leaves/s | 2,645 M leaves/s | 1.00x |
| 2^23 | poseidon2-16 | 858 M leaves/s | 398 M leaves/s | 0.46x |
| 2^23 | poseidon2-24 | 536 M leaves/s | 205 M leaves/s | 0.38x |
| 2^25 | sha256 | 4,565 M leaves/s | 2,161 M leaves/s | 0.47x |
| 2^25 | blake3 | 2,720 M leaves/s | 2,758 M leaves/s | 1.01x |
| 2^25 | poseidon2-16 | 838 M leaves/s | 404 M leaves/s | 0.48x |
| 2^25 | poseidon2-24 | 463 M leaves/s | 207 M leaves/s | 0.45x |
| 2^27 | sha256 | 4,577 M leaves/s | 2,187 M leaves/s | 0.48x |
| 2^27 | blake3 | 2,734 M leaves/s | 2,734 M leaves/s | 1.00x |
| 2^27 | poseidon2-16 | 750 M leaves/s | 404 M leaves/s | 0.54x |
| 2^27 | poseidon2-24 | 468 M leaves/s | 207 M leaves/s | 0.44x |

Poseidon2 permutation throughput:

| variant | RTX 4090 | A100 | ratio |
|---|---|---|---|
| w16 cuda-fused | 1,681.4 M perm/s | 795.8 M perm/s | 0.47x |
| w16 blocks | 12.2 M perm/s | 19.1 M perm/s | 1.56x |
| w16 fp64 | 16.4 M perm/s | 28.7 M perm/s | 1.75x |
| w16 int8 | 8.4 M perm/s | 13.2 M perm/s | 1.56x |
| w24 cuda-fused | 972.5 M perm/s | 396.8 M perm/s | 0.41x |
| w24 blocks | 7.1 M perm/s | 11.6 M perm/s | 1.63x |
| w24 fp64 | 8.9 M perm/s | 16.2 M perm/s | 1.81x |
| w24 int8 | 5.1 M perm/s | 7.8 M perm/s | 1.52x |

**Finding (flagged).**  The SIMT hash kernels do *not* scale with bandwidth: SHA-256 runs at 190 GB/s on the A100 vs 446 on the 4090 (0.43x, where the bandwidth scaling assumed 2.02x -- a 4.7x departure), Poseidon2-16 0.45x, Blake3 1.05x.  They are INT32-ALU-bound: the A100 has 108 SMs x 64 INT32 lanes at 1.41 GHz = 9.7 T int32 op/s against the 4090's 128 x 64 x 2.52 GHz = 20.6 T (0.47x), and SHA-256 / Poseidon2 land on that ratio.  Blake3 holds its rate (its chunk-parallel layout was latency-bound on the 4090).  On the anchor the best proof-internal hash is **blake3 at 377 GB/s** (54.0 ms per B batch, 13.2 us/VU); the hash_gpu.md / encode_gpu.json A100 rows (SHA-256 x2.02 = 901 GB/s) overstate the A100's hash rate by 2.4x.

## A vs B on the A100, same batch, same numbers

`explore.tensor` designs A and B (BabyBear, degree-6 challenge, 2^-128, B = 4096, K = 1536, HVZK layer, authentication excluded) priced on the A100 with the A100-measured buckets substituted; the still-modelled buckets are at 0.172 of the A100 INT8 peak / the model's SIMT rate and are listed as `omitted`.  Labels: `ARITHMETIC_DIAGNOSTIC`, scope `vu`, hardware NVIDIA A100 SXM4 80GB.

| design | model only (A100 peaks @0.172, Blake3 50 GB/s) | **A100-measured buckets** overhead (t.total) | buckets | measured buckets |
|---|---|---|---|---|
| B | 2.38e+07 (962 ms) | 1.51e+07 (608 ms) | encoding 437.0 ms; hash 54.0 ms; matmul_modelled 447.1 ms; simt_field_modelled 102.4 ms; hint_gen_modelled 1.6 ms; prg_modelled 3.2 ms | encoding: False; hash: True |
| A | 3.17e+07 (1278 ms) | 3.03e+07 (1222 ms) | encoding 68.8 ms; hash 9.0 ms; matmul_modelled 574.2 ms; simt_field_modelled 637.1 ms; hint_gen_modelled 1.6 ms; prg_modelled 0.5 ms; gkr_checker_modelled 850.1 ms; logup_modelled 248.8 ms | encoding: False; hash: True |

Measured inputs: `hash_rate_B` = 3.77e+11; `hash_rate_A` = 3.56e+11; `hash_B_kernel` = blake3; `hash_A_kernel` = blake3

* **B**: overhead **1.51e+07** (608 ms per batch, 148 us/VU).  Measured: encoding = False; hash = True.  Omitted (modelled): witness/hint generation (modelled: 2e4 int ops per unit at 25% of the A100 FP32 peak); Ligero row combination (modelled matmul @0.172 of the A100 INT8 peak); openings and transcript serialisation (not priced); prover randomness (modelled PRG at 100 GB/s); linear/quadratic tests, constraint combination, helper inversions (modelled SIMT at the A100 rate).
* **A**: overhead **3.03e+07** (1222 ms per batch, 298 us/VU).  Measured: encoding = False; hash = True.  Omitted (modelled): witness/hint generation (modelled: 2e4 int ops per unit at 25% of the A100 FP32 peak); Ligero row combination (modelled matmul @0.172 of the A100 INT8 peak); openings and transcript serialisation (not priced); prover randomness (modelled PRG at 100 GB/s); checker GKR sumcheck (modelled @0.172); logUp fractional GKR (modelled @0.172).
