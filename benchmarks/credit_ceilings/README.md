# Credit ceilings (sm_120)

The best sustained FP8-equivalent MAC rate each non-tensor route of the GPU model's credit table reaches on one RTX PRO
6000 SM, as a share of the FP8 tensor rate (1,024 MACs a SM a clock). The model adopts each row as an upper bound that no
real kernel exceeds; these kernels are the strongest attempts at exceeding them. CMP-12, worker W10; the results and
their reading are in the compute-accounting lane's notes (`*-draft-credit-ceilings.md`).

Every route is an exact E4M3 GEMM, C = A * B in int64 units of 2^-18, checked bit for bit against a CPU int64 reference:
every output of a small instance, and 4,096 sampled outputs of each timed instance. The E4M3 decoder is first checked
against the hardware conversion (`__nv_cvt_fp8_to_halfraw`) on all 254 finite codes.

Inputs come in two distributions:

- **generic**: every finite E4M3 code, uniformly. This is the protocol's input in general.
- **narrow**: exponent field 4..7 and zero, the subset that is exactly int8 at one scale. It is the most favorable input
  an integer route can get.

| route | kernel | per 4 MACs |
|---|---|---|
| `dp4a_int8` | narrow, one int8 per value, 8x8 register tile | 1 dp4a |
| `dp4a_karatsuba3` | generic, 3 signed 6-bit digits per value, Karatsuba's 6 digit-pair products | 6 dp4a |
| `ffma_fp32` | narrow as FP32 (products exact, sums exact below 2^24, flushed to int64 every 1,024 k) | 4 FFMA |
| `imad_swar2` | narrow, two rows in 16-bit lanes of one IMAD, lanes split every 2 k's | 2 IMAD + split |
| `smem_lut` | generic, a product table per k in shared memory (hi/lo 18-bit halves), table build timed | 2 LDS.128 |
| `combo_dp4a_ffma` | dp4a and FFMA tiles in the same warps, narrow | |

`k_probe` measures dp4a beside IMAD, FFMA and LOP3 (warp instructions a SM a clock), and `ffma2_probe.cu` whether
packed FP32 FMA (`fma.rn.f32x2`) builds and runs on sm_120. Raw pipe caps are the census's
(`benchmarks/pouw/census_sm120/` on `cursor/census-sm120-ea24`), not repeated here.

Run it on vy-nebius-2 from a clean commit (`run.sh` has the full command); outputs go to `$OUT` and
`summarize.py OUT` renders `summary.md`.
