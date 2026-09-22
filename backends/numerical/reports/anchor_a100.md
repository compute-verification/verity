# anchor-a100: every campaign GPU kernel measured on the A100 (the anchor), 2026-09-22

**Scope: component measurements** (kernel time on random/patterned data, bit-exact against each lane's CPU reference), then one `ARITHMETIC_DIAGNOSTIC` re-price of A and B with those numbers.  Nothing here is a proof.  Every recorded 4090 / H100 number is the lane's own (its README / report, run id in the lane's table); every A100 number is a `research run --on vy-a100b --exclusive` of the same bench, same flags, CUDA events, warm, median of >= 7, bit-exact checks on.  `overhead.vs_native_peak` is defined against *this* card's dense BF16 peak (`contract.NATIVE_REFERENCE`, 9.85 ps per K=1536 VU), so the last section is the first A-vs-B comparison where the prover and the reference are the same machine.

Device: NVIDIA A100-SXM4-80GB, 580.126.16, 1410 MHz, 1593 MHz, 81920 MiB; torch 2.6.0+cu124, CUDA 12.4, 108 SMs (sm_80).  Recorded runs: microbench `r20260922-063608-a304`, hash_gpu `r20260922-063811-e561`, encode `r20260922-064413-0508`, fused `r20260922-064953-f9bb`.

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

## A vs B on the A100, same batch, same numbers

`explore.tensor` designs A and B (BabyBear, degree-6 challenge, 2^-128, B = 4096, K = 1536, HVZK layer, authentication excluded) priced on the A100 with the A100-measured buckets substituted; the still-modelled buckets are at 0.172 of the A100 INT8 peak / the model's SIMT rate and are listed as `omitted`.  Labels: `ARITHMETIC_DIAGNOSTIC`, scope `vu`, hardware NVIDIA A100 SXM4 80GB.

| design | model only (A100 peaks @0.172, Blake3 50 GB/s) | **A100-measured buckets** overhead (t.total) | buckets | measured buckets |
|---|---|---|---|---|
| B | 2.38e+07 (962 ms) | 4.88e+06 (197 ms) | encoding 57.4 ms; hash 54.0 ms; matmul_modelled 10.1 ms; simt_field_modelled 70.6 ms; hint_gen_modelled 1.6 ms; prg_modelled 3.2 ms | encoding: True; hash: True |
| A | 3.17e+07 (1278 ms) | 3.71e+07 (1495 ms) | encoding 11.0 ms; hash 9.0 ms; matmul_modelled 505.4 ms; simt_field_modelled 632.1 ms; hint_gen_modelled 1.6 ms; prg_modelled 0.5 ms; gkr_checker_modelled 850.1 ms; logup_modelled 248.8 ms; gkr_checker_measured 1185.4 ms | encoding: True; hash: True; gkr_checker: fused SIMT layer model x96 (graphed) |

Measured inputs: `hash_rate_B` = 3.77e+11; `hash_rate_A` = 3.56e+11; `hash_B_kernel` = blake3; `hash_A_kernel` = blake3; `encoder_seconds_per_row_B` = 1.85e-07; `encoder_B_kernel` = ntt_simt_v2_t1024; `encoder_seconds_per_row_A` = 2.25e-07; `encoder_A_kernel` = ntt_simt_v2_t1024; `fused_layer_graphed_s` = 0.0123; `fused_checker_round_2p21_us` = 473.7920

* **B**: overhead **4.88e+06** (197 ms per batch, 48 us/VU).  Measured: encoding = True; hash = True.  Omitted (modelled): witness/hint generation (modelled: 2e4 int ops per unit at 25% of the A100 FP32 peak); Ligero row combination (modelled matmul @0.172 of the A100 INT8 peak); openings and transcript serialisation (not priced); prover randomness (modelled PRG at 100 GB/s); linear/quadratic tests, constraint combination, helper inversions (modelled SIMT at the A100 rate).
* **A**: overhead **3.71e+07** (1495 ms per batch, 365 us/VU).  Measured: encoding = True; hash = True; gkr_checker = fused SIMT layer model x96 (graphed).  Omitted (modelled): witness/hint generation (modelled: 2e4 int ops per unit at 25% of the A100 FP32 peak); Ligero row combination (modelled matmul @0.172 of the A100 INT8 peak); openings and transcript serialisation (not priced); prover randomness (modelled PRG at 100 GB/s); logUp fractional GKR (modelled @0.172).
