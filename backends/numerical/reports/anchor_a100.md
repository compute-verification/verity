# anchor-a100: every campaign GPU kernel measured on the A100 (the anchor), 2026-09-22

**Scope: component measurements** (kernel time on random/patterned data, bit-exact against each lane's CPU reference), then one `ARITHMETIC_DIAGNOSTIC` re-price of A and B with those numbers.  Nothing here is a proof.  Every recorded 4090 / H100 number is the lane's own (its README / report, run id in the lane's table); every A100 number is a `research run --on vy-a100b --exclusive` of the same bench, same flags, CUDA events, warm, median of >= 7, bit-exact checks on.  `overhead.vs_native_peak` is defined against *this* card's dense BF16 peak (`contract.NATIVE_REFERENCE`, 9.85 ps per K=1536 VU), so the last section is the first A-vs-B comparison where the prover and the reference are the same machine.

Device: NVIDIA A100-SXM4-80GB, 580.126.16, 1410 MHz, 1593 MHz, 81920 MiB; torch 2.6.0+cu124, CUDA 12.4, 108 SMs (sm_80).  Recorded runs: microbench `r20260922-063608-a304`, hash_gpu `r20260922-063811-e561`, encode `r20260922-064413-0508`, fused `r20260922-064953-f9bb`, fused_logup `r20260922-071602-522d`, packed `r20260922-073828-153f`, packed_killed_mislaunch `r20260922-073540-f5b5`, tensor `r20260922-074359-964b`, wave2_gkr (packed Montgomery 2^26, direct 2^28, logUp GKR full-VU) `r20260922-075147-91a5`.

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

## 2. `backends/direct/encode` -- Reed-Solomon encoders (4090 run `r20260922-062016-6bdf`)

| encoder | shape | RTX 4090 recorded | A100 measured | speed ratio A100/4090 | A100 floor | fraction of floor |
|---|---|---|---|---|---|---|
| ntt_simt_v2_t512 | B | 32.2 ms (0.104 us/row) | 60.7 ms (0.195 us/row) | 0.53x | 14.5 ms HBM@1754 | 0.24 |
| ntt_simt_v2_t512_nosys | B | 31.6 ms (0.102 us/row) | 60.6 ms (0.195 us/row) | 0.52x | 11.6 ms HBM@1754 | 0.19 |
| ntt_simt_v1_t512 | B | 38.6 ms (0.124 us/row) | 73.2 ms (0.236 us/row) | 0.53x | 14.5 ms HBM@1754 | 0.20 |
| ntt_simt_v2_t128 | B | 34.8 ms (0.112 us/row) | 71.6 ms (0.230 us/row) | 0.49x | 14.5 ms HBM@1754 | 0.20 |
| ntt_simt_v2_t256 | B | 39.6 ms (0.128 us/row) | 76.3 ms (0.246 us/row) | 0.52x | 14.5 ms HBM@1754 | 0.19 |
| ntt_simt_v2_t1024 | B | 32.7 ms (0.105 us/row) | 57.4 ms (0.185 us/row) | 0.57x | 14.5 ms HBM@1754 | 0.25 |
| matdft_int8_triton | B | 171.7 ms (0.553 us/row) | 607.1 ms (1.955 us/row) | 0.28x | 33.4 ms TC@312T (HBM 14.5) | 0.06; 0.055 of INT8 peak |
| matdft_int8_cublas | B | 15,224.9 ms (49.017 us/row) | 9,146.2 ms (29.447 us/row) | 1.66x | 33.4 ms TC@312T (HBM 14.5) | 0.00; 0.004 of INT8 peak |
| ntt_simt_v2_t512 | A | 5.3 ms (0.107 us/row) | 11.6 ms (0.238 us/row) | 0.45x | 2.3 ms HBM@1754 | 0.20 |
| ntt_simt_v2_t512_nosys | A | 5.2 ms (0.106 us/row) | 11.6 ms (0.237 us/row) | 0.45x | 1.8 ms HBM@1754 | 0.16 |
| ntt_simt_v1_t512 | A | 6.0 ms (0.122 us/row) | 13.8 ms (0.282 us/row) | 0.43x | 2.3 ms HBM@1754 | 0.17 |
| ntt_simt_v2_t128 | A | 5.9 ms (0.120 us/row) | 13.7 ms (0.280 us/row) | 0.43x | 2.3 ms HBM@1754 | 0.17 |
| ntt_simt_v2_t256 | A | 6.3 ms (0.128 us/row) | 14.6 ms (0.299 us/row) | 0.43x | 2.3 ms HBM@1754 | 0.16 |
| ntt_simt_v2_t1024 | A | 5.2 ms (0.106 us/row) | 11.0 ms (0.225 us/row) | 0.47x | 2.3 ms HBM@1754 | 0.21 |
| matdft_int8_triton | A | 26.8 ms (0.549 us/row) | 94.6 ms (1.936 us/row) | 0.28x | 5.3 ms TC@312T (HBM 2.3) | 0.06; 0.056 of INT8 peak |
| matdft_int8_cublas | A | 2,395.7 ms (49.019 us/row) | 1,439.3 ms (29.450 us/row) | 1.66x | 5.3 ms TC@312T (HBM 2.3) | 0.00; 0.004 of INT8 peak |

Bit-exact checks on the A100 (64 full rows vs the numpy reference): `{'simt_v1_t256': True, 'simt_v1_t512': True, 'simt_v2_t128': True, 'simt_v2_t256': True, 'simt_v2_t512': True, 'simt_v2_t1024': True, 'cublas': True, 'triton': True}`.

The SIMT NTT (v2, 512 threads) encodes the B batch in **60.7 ms** on the A100 (0.53x the 4090; bandwidth ratio 2.02), 0.24 of the measured-bandwidth HBM floor (14.5 ms) and 0.21 of the datasheet floor.  The fused Triton INT8 matrix DFT takes 607 ms (0.28x the 4090, 0.055 of the A100 INT8 peak): the SIMT NTT beats it by 10.0x on the anchor too.

## 3. `backends/gkr/tensor/fused` -- fused checker sumcheck round (4090 run `r20260922-060339-740b`)

| fused checker round | RTX 4090 recorded | A100 measured | speed ratio A100/4090 | A100 HBM floor (measured BW) | fraction of floor |
|---|---|---|---|---|---|
| simt round, 2^23, ext input, fold+eval | 1,014 us (894 GB/s) | 1,871 us (484 GB/s) | 0.54x | 517 us | 0.28 |
| simt round, 2^22, ext input, fold+eval | 520 us (871 GB/s) | 935 us (484 GB/s) | 0.56x | 258 us | 0.28 |
| simt round, 2^22, base input, fold+eval | 256 us (786 GB/s) | 607 us (332 GB/s) | 0.42x | 115 us | 0.19 |
| simt round, 2^21, ext input, fold+eval | 273 us (828 GB/s) | 474 us (478 GB/s) | 0.58x | 129 us | 0.27 |
| simt round, 2^20, ext input, fold+eval | 153 us (742 GB/s) | 244 us (464 GB/s) | 0.62x | 65 us | 0.26 |
| simt round, 2^20, base input, fold+eval | 82 us (614 GB/s) | 159 us (317 GB/s) | 0.52x | 29 us | 0.18 |
| simt round, 2^18, ext input, fold+eval | 50 us (564 GB/s) | 78 us (362 GB/s) | 0.64x | 16 us | 0.21 |
| tc round, 2^23, ext input, fold+eval | 1,960 us (462 GB/s) | 6,246 us (145 GB/s) | 0.31x | 517 us | 0.08 |
| tc round, 2^22, ext input, fold+eval | 988 us (458 GB/s) | 3,131 us (145 GB/s) | 0.32x | 258 us | 0.08 |
| tc round, 2^21, ext input, fold+eval | 506 us (448 GB/s) | 1,569 us (144 GB/s) | 0.32x | 129 us | 0.08 |
| tc round, 2^20, ext input, fold+eval | 263 us (430 GB/s) | 796 us (142 GB/s) | 0.33x | 65 us | 0.08 |
| tc round, 2^18, ext input, fold+eval | 77 us (369 GB/s) | 210 us (135 GB/s) | 0.37x | 16 us | 0.08 |
| simt 6-layer / 130-round layer model, graphed | 6.20 ms | 12.35 ms | 0.50x | 2.11 ms | 0.17 |
| simt layer model, eager | 8.72 ms | 13.40 ms | 0.65x |  |  |
| tc 6-layer / 130-round layer model, graphed | 8.48 ms | 20.85 ms | 0.41x | 2.11 ms | 0.10 |
| tc layer model, eager | 10.11 ms | 22.39 ms | 0.45x |  |  |

Bit-exact tests on the A100 (`TC_DEVICE=cuda pytest tensor/fused/tests`): 25 passed in 1188.44s (0:19:48)

The SIMT fused round at 2^21 (ext input, cold L2) runs **474 us** on the A100 (0.58x the 4090's 273 us; 478 GB/s = 0.27 of the measured-bandwidth floor).  The 130-round layer model: 12.35 ms graphed (0.50x the 4090's 6.20 ms).

## 3b. `backends/gkr/tensor/fused` -- fused logUp fractional-GKR round (4090 report `fused_logup_4090.json`)

| fused logUp round (fold+eval) | RTX 4090 recorded | A100 measured | speed ratio A100/4090 | A100 HBM floor (measured BW) | fraction of floor |
|---|---|---|---|---|---|
| logUp round 2^21, block 256 x 4 warps | 600 us (629 GB/s) | 1,469 us (257 GB/s) | 0.41x | 215 us | 0.15 |
| logUp round 2^21, block 128 x 4 warps | 755 us (500 GB/s) | 1,636 us (231 GB/s) | 0.46x | 215 us | 0.13 |
| logUp round 2^21, block 512 x 8 warps | 681 us (554 GB/s) | 1,633 us (231 GB/s) | 0.42x | 215 us | 0.13 |
| logUp round 2^20, block 256 x 4 warps | 344 us (549 GB/s) | 867 us (218 GB/s) | 0.40x | 108 us | 0.12 |
| logUp round 2^20, block 128 x 4 warps | 402 us (469 GB/s) | 806 us (234 GB/s) | 0.50x | 108 us | 0.13 |
| logUp round 2^20, block 512 x 8 warps | 385 us (490 GB/s) | 878 us (215 GB/s) | 0.44x | 108 us | 0.12 |
| logUp round 2^18, block 256 x 4 warps | 118 us (401 GB/s) | 245 us (193 GB/s) | 0.48x | 27 us | 0.11 |
| logUp round 2^18, block 128 x 4 warps | 137 us (344 GB/s) | 287 us (164 GB/s) | 0.48x | 27 us | 0.09 |
| logUp round 2^18, block 512 x 8 warps | 124 us (381 GB/s) | 311 us (152 GB/s) | 0.40x | 27 us | 0.09 |
| logUp round 2^16, block 256 x 4 warps | 81 us (146 GB/s) | 112 us (105 GB/s) | 0.72x | 7 us | 0.06 |
| logUp round 2^16, block 128 x 4 warps | 78 us (152 GB/s) | 110 us (107 GB/s) | 0.71x | 7 us | 0.06 |
| logUp round 2^16, block 512 x 8 warps | 82 us (144 GB/s) | 118 us (100 GB/s) | 0.69x | 7 us | 0.06 |
| logUp round 2^14, block 256 x 4 warps | 78 us (38 GB/s) | 111 us (26 GB/s) | 0.70x | 2 us | 0.02 |
| logUp round 2^14, block 128 x 4 warps | 73 us (41 GB/s) | 99 us (30 GB/s) | 0.73x | 2 us | 0.02 |
| logUp round 2^14, block 512 x 8 warps | 84 us (35 GB/s) | 119 us (25 GB/s) | 0.71x | 2 us | 0.01 |
| logUp round 2^12, block 256 x 4 warps | 80 us (9 GB/s) | 111 us (7 GB/s) | 0.72x | 0 us | 0.00 |
| logUp round 2^12, block 128 x 4 warps | 73 us (10 GB/s) | 98 us (8 GB/s) | 0.74x | 0 us | 0.00 |
| logUp round 2^12, block 512 x 8 warps | 82 us (9 GB/s) | 118 us (6 GB/s) | 0.69x | 0 us | 0.00 |

## 4. `backends/gkr/packed` -- base-field-first packed sumcheck (H100 run `r20260922-062108-4b25`)

| packed sumcheck (whole sumcheck, all rounds, CUDA-graphed) | H100 recorded | A100 measured | speed ratio A100/H100 | A100 TC floor | fraction of INT8 peak | A100 peak mem |
|---|---|---|---|---|---|---|
| 2^26, k=0 | 30.87 ms (1.39% of H100 INT8 peak) | 50.22 ms (2.71% of A100 INT8 peak) | 0.61x | 1.36 ms TC@312T | 0.0271 | 34.1 GB |
| 2^26, k=1 | 18.47 ms (1.36% of H100 INT8 peak) | 30.73 ms (2.59% of A100 INT8 peak) | 0.60x | 0.79 ms TC@312T | 0.0259 | 17.2 GB |
| 2^26, k=2 | 11.73 ms (1.39% of H100 INT8 peak) | 20.21 ms (2.55% of A100 INT8 peak) | 0.58x | 0.52 ms TC@312T | 0.0255 | 8.9 GB |
| 2^26, k=3 | 9.08 ms (1.63% of H100 INT8 peak) | 16.58 ms (2.83% of A100 INT8 peak) | 0.55x | 0.47 ms TC@312T | 0.0283 | 4.5 GB |
| 2^26, k=4 | 23.37 ms (0.85% of H100 INT8 peak) | 64.95 ms (0.97% of A100 INT8 peak) | 0.36x | 0.63 ms TC@312T | 0.0097 | 4.1 GB |
| 2^24, k=0 | 9.20 ms (1.17% of H100 INT8 peak) | 16.15 ms (2.11% of A100 INT8 peak) | 0.57x | 0.34 ms TC@312T | 0.0211 | 8.7 GB |
| 2^24, k=1 | 6.06 ms (1.03% of H100 INT8 peak) | 10.63 ms (1.87% of A100 INT8 peak) | 0.57x | 0.20 ms TC@312T | 0.0187 | 4.5 GB |
| 2^24, k=2 | 4.35 ms (0.93% of H100 INT8 peak) | 7.42 ms (1.74% of A100 INT8 peak) | 0.59x | 0.13 ms TC@312T | 0.0174 | 1.8 GB |
| 2^24, k=3 | 3.71 ms (1.00% of H100 INT8 peak) | 6.96 ms (1.69% of A100 INT8 peak) | 0.53x | 0.12 ms TC@312T | 0.0169 | 1.0 GB |
| 2^24, k=4 | 7.30 ms (0.68% of H100 INT8 peak) | 18.27 ms (0.87% of A100 INT8 peak) | 0.40x | 0.16 ms TC@312T | 0.0087 | 1.2 GB |
| 2^22, k=0 | 3.57 ms (0.75% of H100 INT8 peak) | 6.00 ms (1.42% of A100 INT8 peak) | 0.60x | 0.09 ms TC@312T | 0.0142 | 1.7 GB |
| 2^22, k=1 | 2.77 ms (0.56% of H100 INT8 peak) | 4.87 ms (1.02% of A100 INT8 peak) | 0.57x | 0.05 ms TC@312T | 0.0102 | 1.0 GB |
| 2^22, k=2 | 2.29 ms (0.44% of H100 INT8 peak) | 3.96 ms (0.81% of A100 INT8 peak) | 0.58x | 0.03 ms TC@312T | 0.0081 | 0.6 GB |
| 2^22, k=3 | 2.15 ms (0.43% of H100 INT8 peak) | 3.66 ms (0.80% of A100 INT8 peak) | 0.59x | 0.03 ms TC@312T | 0.0080 | 0.4 GB |
| 2^22, k=4 | 3.00 ms (0.42% of H100 INT8 peak) | 6.76 ms (0.58% of A100 INT8 peak) | 0.44x | 0.04 ms TC@312T | 0.0058 | 0.5 GB |
| 2^20, k=0 | 2.04 ms (0.33% of H100 INT8 peak) | 3.35 ms (0.64% of A100 INT8 peak) | 0.61x | 0.02 ms TC@312T | 0.0064 | 0.4 GB |
| 2^20, k=1 | 1.76 ms (0.22% of H100 INT8 peak) | 2.88 ms (0.43% of A100 INT8 peak) | 0.61x | 0.01 ms TC@312T | 0.0043 | 0.3 GB |
| 2^20, k=2 | 1.56 ms (0.16% of H100 INT8 peak) | 2.61 ms (0.31% of A100 INT8 peak) | 0.60x | 0.01 ms TC@312T | 0.0031 | 0.2 GB |
| 2^20, k=3 | 1.55 ms (0.15% of H100 INT8 peak) | 2.54 ms (0.29% of A100 INT8 peak) | 0.61x | 0.01 ms TC@312T | 0.0029 | 0.2 GB |
| 2^20, k=4 | 1.80 ms (0.17% of H100 INT8 peak) | 3.49 ms (0.28% of A100 INT8 peak) | 0.51x | 0.01 ms TC@312T | 0.0028 | 0.3 GB |

Six-layer batch model (linear fit over the measured sizes; padded / un-padded widths):

| k | H100 per batch | H100 per VU | A100 per batch | A100 per VU | speed ratio A100/H100 | A100 fit |
|---|---|---|---|---|---|---|
| k=0 | 973 / 581 ms | 237.6 us | 1578 / 943 ms | 385.2 us | 0.62x | 0.704 ns/elem + 3.25 ms |
| k=1 | 566 / 339 ms | 138.1 us | 937 / 563 ms | 228.9 us | 0.60x | 0.415 ns/elem + 3.03 ms |
| k=2 | 345 / 209 ms | 84.3 us | 596 / 360 ms | 145.6 us | 0.58x | 0.262 ns/elem + 2.71 ms |
| k=3 | 257 / 156 ms | 62.7 us | 476 / 289 ms | 116.3 us | 0.54x | 0.208 ns/elem + 2.81 ms |
| k=4 | 729 / 436 ms | 177.9 us | 2071 / 1234 ms | 505.6 us | 0.35x | 0.928 ns/elem + 2.69 ms |

Agreement test on the A100: 36 passed in 61.24s (0:01:01)

Best k on the A100: **k=3**, 476 ms per six-layer batch = 116.3 us/VU (H100: k=3, 257 ms, 62.7 us/VU).  At 2^26 k=3: 16.58 ms vs 9.08 on the H100 (0.55x; bandwidth ratio 0.61, INT8-peak ratio 0.32).

## 4b. `backends/gkr/packed` second pass (a-packed2, landed on main 07:4xZ): 32-bit Montgomery packed sumcheck, direct 2^28, logUp fractional GKR (H100 runs `r20260922-064917-c293`, `r20260922-065412-73be`, `r20260922-073439-1f15`; A100 run `r20260922-075147-91a5`)

| a-packed2 kernels (whole sumcheck, CUDA-graphed rounds) | H100 recorded | A100 measured | speed ratio A100/H100 | A100 TC floor | fraction of INT8 peak | A100 peak mem |
|---|---|---|---|---|---|---|
| Montgomery packed sumcheck 2^26, k=0 | 12.56 ms | 21.39 ms (6.37% of A100 INT8 peak) | 0.59x | 1.36 ms TC@312T | 0.0637 | 34.1 GB |
| Montgomery packed sumcheck 2^26, k=1 | 10.68 ms | 17.80 ms (4.47% of A100 INT8 peak) | 0.60x | 0.79 ms TC@312T | 0.0447 | 17.2 GB |
| Montgomery packed sumcheck 2^26, k=2 | 6.43 ms | 11.17 ms (4.62% of A100 INT8 peak) | 0.58x | 0.52 ms TC@312T | 0.0462 | 8.9 GB |
| Montgomery packed sumcheck 2^26, k=3 | 5.97 ms | 11.54 ms (4.07% of A100 INT8 peak) | 0.52x | 0.47 ms TC@312T | 0.0407 | 4.5 GB |
| Montgomery packed sumcheck 2^26, k=4 | 9.25 ms | 15.19 ms (4.16% of A100 INT8 peak) | 0.61x | 0.63 ms TC@312T | 0.0416 | 2.4 GB |
| Montgomery packed sumcheck 2^28, k=2 | 19.78 ms | 34.33 ms (6.01% of A100 INT8 peak) | 0.58x | 2.06 ms TC@312T | 0.0601 | 34.4 GB |
| Montgomery packed sumcheck 2^28, k=3 | 17.69 ms | 31.93 ms (5.88% of A100 INT8 peak) | 0.55x | 1.88 ms TC@312T | 0.0588 | 17.8 GB |
| Montgomery packed sumcheck 2^28, k=4 | 30.96 ms | 43.80 ms (5.78% of A100 INT8 peak) | 0.71x | 2.53 ms TC@312T | 0.0578 | 10.6 GB |
| six-layer batch from the 2^26 fit, k=0 (padded / un-padded) | 369 / 223 ms = 90.1 us/VU | 638 / 385 ms = **155.8 us/VU** | 0.58x |  |  |  |
| six-layer batch from the 2^26 fit, k=1 (padded / un-padded) | 308 / 187 ms = 75.2 us/VU | 519 / 314 ms = **126.7 us/VU** | 0.59x |  |  |  |
| six-layer batch from the 2^26 fit, k=2 (padded / un-padded) | 169 / 104 ms = 41.4 us/VU | 303 / 186 ms = **74.0 us/VU** | 0.56x |  |  |  |
| six-layer batch from the 2^26 fit, k=3 (padded / un-padded) | 154 / 95 ms = 37.5 us/VU | 316 / 194 ms = **77.2 us/VU** | 0.49x |  |  |  |
| six-layer batch from the 2^26 fit, k=4 (padded / un-padded) | 263 / 160 ms = 64.2 us/VU | 432 / 263 ms = **105.5 us/VU** | 0.61x |  |  |  |
| six-layer batch from the 2^28 fit, k=2 (padded / un-padded) | 163 / 97 ms = 39.8 us/VU | 283 / 168 ms = **69.1 us/VU** | 0.58x |  |  |  |
| six-layer batch from the 2^28 fit, k=3 (padded / un-padded) | 146 / 86 ms = 35.6 us/VU | 263 / 156 ms = **64.3 us/VU** | 0.55x |  |  |  |
| six-layer batch from the 2^28 fit, k=4 (padded / un-padded) | 255 / 151 ms = 62.4 us/VU | 361 / 214 ms = **88.2 us/VU** | 0.71x |  |  |  |
| logUp fractional GKR, full-VU shape (393,216 x 306 lookups, 2^27 leaves), k=3 | 427 ms = 104 us/VU | **412 ms = 101 us/VU** (20 smallest levels 144 ms) | 1.04x |  |  | 12.4 GB |
| logUp fractional GKR, full-VU shape (393,216 x 306 lookups, 2^27 leaves), k=0 | 447 ms = 109 us/VU | **478 ms = 117 us/VU** (20 smallest levels 142 ms) | 0.94x |  |  | 24.8 GB |

Agreement test on the A100 (Montgomery kernels): 46 passed in 405.11s (0:06:45)

The Montgomery kernels scale with bandwidth on the A100: 0.52-0.61x the H100 at 2^26 (bandwidth ratio 0.61), k=3 direct 2^28 **31.9 ms** (0.55x the H100's 17.7; 5.9% of the A100 INT8 peak) = **64.3 us/VU, 263 ms per six-layer batch** padded (156 un-padded) -- 1.8x better than the first-pass packed k=3 on the same card (476 ms).  k=2 and k=3 are within 4% of each other on the A100 (the base x base Hadamard share is INT32 work at 0.47x the H100's rate).  **Finding (flagged).**  The logUp fractional GKR runs at **412 ms = 101 us/VU** on the A100 vs 427 on the H100 (1.04x: a 1.7x departure from the bandwidth ratio) -- it is host-launch-bound (~1 ms x 351 rounds; the 20 smallest levels are 144 of the 412 ms on both cards), so the GPU does not matter yet; the device work is ~0.1-0.2 s.  Fixing the launch loop (CUDA graphs per level) is worth ~2x on either card.

## 5. `backends/gkr/tensor` -- the original INT8-limb round (4090 run `r20260922-051730-15d4`)

| INT8-limb round (tensor lane) | RTX 4090 recorded | A100 measured | speed ratio A100/4090 | A100 TC floor | fraction of INT8 peak |
|---|---|---|---|---|---|
| t=2^21 deg 6 round (graphed) | 4,636 us | 4,894 us | 0.95x | 38.7 us TC@312T | 0.0079 |
| t=2^20 deg 6 round (graphed) | 2,220 us | 2,626 us | 0.85x | 19.4 us TC@312T | 0.0074 |
| t=2^19 deg 6 round (graphed) | 1,050 us | 1,717 us | 0.61x | 9.7 us TC@312T | 0.0056 |
| t=2^18 deg 6 round (graphed) | 602 us | 1,044 us | 0.58x | 4.8 us TC@312T | 0.0046 |
| t=2^17 deg 6 round (graphed) | 383 us | 713 us | 0.54x | 2.4 us TC@312T | 0.0034 |
| t=2^16 deg 6 round (graphed) | 298 us | 534 us | 0.56x | 1.2 us TC@312T | 0.0023 |
| t=2^15 deg 6 round (graphed) | 278 us | 454 us | 0.61x | 0.6 us TC@312T | 0.0013 |
| t=2^14 deg 6 round (graphed) | 254 us | 404 us | 0.63x | 0.3 us TC@312T | 0.0007 |
| t=2^13 deg 6 round (graphed) | 259 us | 391 us | 0.66x | 0.2 us TC@312T | 0.0004 |
| t=2^12 deg 6 round (graphed) | 257 us | 378 us | 0.68x | 0.1 us TC@312T | 0.0002 |
| t=2^22 deg 1 round (graphed) | 3,879 us | 3,706 us | 1.05x | 8.6 us TC@312T | 0.0023 |
| t=2^20 deg 1 round (graphed) | 957 us | 989 us | 0.97x | 2.2 us TC@312T | 0.0022 |
| layer model graphed wall (130 rounds) | 91.4 ms | 115.6 ms | 0.79x |  |  |
| layer model, CUDA-event segments (launch gaps included) | 472.4 ms | 519.7 ms | 0.91x |  |  |
|   of which matmul kernels (events) | 80.5 ms | 98.2 ms | 0.82x |  |  |
|   tensor bucket (events) | 371.0 ms | 414.7 ms | 0.89x |  |  |
| layer model GPU kernel time (profiler) | 87.4 ms (report) | 115.5 ms | 0.76x |  |  |
| `_int_mm` (8192, 8192, 8192) | 6,585 us (0.2528 of 4090 peak) | 14,536 us (0.1212 of A100 peak) | 0.45x |  |  |
| `_int_mm` (4096, 4096, 4096) | 846 us (0.2459 of 4090 peak) | 1,940 us (0.1136 of A100 peak) | 0.44x |  |  |
| `_int_mm` (48, 32768, 96) | 821 us (0.0006 of 4090 peak) | 1,547 us (0.0003 of A100 peak) | 0.53x |  |  |
| `_int_mm` (48, 131072, 96) | 3,204 us (0.0006 of 4090 peak) | 6,107 us (0.0003 of A100 peak) | 0.52x |  |  |
| `_int_mm` (2097152, 48, 48) | 583 us (0.0251 of 4090 peak) | 777 us (0.0199 of A100 peak) | 0.75x |  |  |
| `_int_mm` (2097152, 48, 96) | 1,070 us (0.0273 of 4090 peak) | 787 us (0.0394 of A100 peak) | 1.36x |  |  |
| `_int_mm` (24, 32768, 24) | 820 us (0.0001 of 4090 peak) | 1,545 us (0.0000 of A100 peak) | 0.53x |  |  |

Agreement test on the A100: 22 passed in 29.04s

## A vs B on the A100, same batch, same numbers

`explore.tensor` designs A and B (BabyBear, degree-6 challenge, 2^-128, B = 4096, K = 1536, HVZK layer, authentication excluded) priced on the A100 with the A100-measured buckets substituted; the still-modelled buckets are at 0.172 of the A100 INT8 peak / the model's SIMT rate and are listed as `omitted`.  Labels: `ARITHMETIC_DIAGNOSTIC`, scope `vu`, hardware NVIDIA A100 SXM4 80GB.

| design | model only (A100 peaks @0.172, Blake3 50 GB/s) | **A100-measured buckets**: overhead (t.total per batch) | buckets, ms (measured in bold) |
|---|---|---|---|
| **B** | 2.38e+07 (962 ms) | **4.88e+06** (197 ms; 48 us/VU) | encoding **57.4**; hash **54.0**; row_combination 10.1; linear/quadratic tests + helpers (SIMT) 70.6; hint_generation 1.6; prg 3.2 |
| **A** | 3.17e+07 (1278 ms) | **1.83e+07** (736 ms; 180 us/VU) | encoding **11.0**; hash **9.0**; row_combination 1.4; other SIMT (logUp tree build, ...) 37.2; gkr_checker_sumcheck **263.4**; logup_fractional_gkr **412.2**; hint_generation 1.6; prg 0.5 |

Measured inputs: `hash_rate_B` = 3.77e+11; `hash_rate_A` = 3.56e+11; `hash_B_kernel` = blake3; `hash_A_kernel` = blake3; `encoder_seconds_per_row_B` = 1.85e-07; `encoder_B_kernel` = ntt_simt_v2_t1024; `encoder_seconds_per_row_A` = 2.25e-07; `encoder_A_kernel` = ntt_simt_v2_t1024; `fused_layer_graphed_s` = 0.0123; `fused_checker_round_2p21_us` = 473.7920; `logup_round_2p21_us` = 1,468.6720; `logup_over_checker_round` = 3.0998; `packed_k0_batch_s_padded` = 1.5777; `packed_k0_per_vu_s` = 0.000385; `packed_k3_batch_s_padded` = 0.4764; `packed_k3_per_vu_s` = 0.000116; `packed2_best_batch_s_padded` = 0.2634; `packed2_best_batch_s_unpadded` = 0.1561; `packed2_best_label` = Montgomery packed k=3 from the 2^28 fit; `logup_gkr_full_vu_s` = 0.4122

* **B**: overhead **4.88e+06** (197 ms per batch, 48 us/VU).

  * encoding: 57.4 ms (29%) -- measured: ntt_simt_v2_t1024 0.185 us/row x 310,602 rows
  * hash: 54.0 ms (27%) -- measured: blake3 377 GB/s over 20.36 GB
  * row_combination: 10.1 ms (5%) -- modelled matmul @0.172
  * linear/quadratic tests + helpers (SIMT): 70.6 ms (36%) -- modelled SIMT
  * hint_generation: 1.6 ms (1%) -- modelled
  * prg: 3.2 ms (2%) -- modelled

  Omitted (still modelled): witness/hint generation (modelled: 2e4 int ops per unit at 25% of the A100 FP32 peak); Ligero row combination (modelled matmul @0.172 of the A100 INT8 peak); openings and transcript serialisation (not priced); prover randomness (modelled PRG at 100 GB/s); linear/quadratic tests, constraint combination, helper inversions (modelled SIMT at the A100 rate).

* **A**: overhead **1.83e+07** (736 ms per batch, 180 us/VU).

  * encoding: 11.0 ms (1%) -- measured: ntt_simt_v2_t1024 0.225 us/row x 48,873 rows
  * hash: 9.0 ms (1%) -- measured: blake3 356 GB/s over 3.21 GB
  * row_combination: 1.4 ms (0%) -- modelled matmul @0.172
  * other SIMT (logUp tree build, ...): 37.2 ms (5%) -- modelled SIMT
  * gkr_checker_sumcheck: 263.4 ms (36%) -- measured: Montgomery packed k=3 from the 2^28 fit six-layer batch (padded) (model @0.172: 850 ms; alternatives fused SIMT layer model x96 (graphed) 1185 ms, packed k=3 six-layer batch (padded) 476 ms, plain k=0 six-layer batch (padded) 1578 ms, Montgomery packed k=3 from the 2^28 fit six-layer batch (padded) 263 ms)
  * logup_fractional_gkr: 412.2 ms (56%) -- measured: a-packed2 logUp fractional GKR at the full-VU shape (393,216 units x 306 lookups, 65,536-row table), k=3, whole bucket on the A100 (model @0.172: 249 ms; launch-bound: ~0.1-0.2 s of device work)
  * hint_generation: 1.6 ms (0%) -- modelled
  * prg: 0.5 ms (0%) -- modelled

  Omitted (still modelled): witness/hint generation (modelled: 2e4 int ops per unit at 25% of the A100 FP32 peak); Ligero row combination (modelled matmul @0.172 of the A100 INT8 peak); openings and transcript serialisation (not priced); prover randomness (modelled PRG at 100 GB/s).

