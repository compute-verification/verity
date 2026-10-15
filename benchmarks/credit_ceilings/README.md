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
| `imad_wide` | generic as int32 value * 2^9, one `mad.wide.s32` (IMAD.WIDE) into an int64 sum | 4 IMAD.WIDE |
| `imad_swar2` | narrow, two rows in 16-bit lanes of one IMAD, lanes split every 2 k's | 2 IMAD + split |
| `smem_lut` | generic, a product table per k in shared memory (hi/lo 18-bit halves), table build timed | 2 LDS.128 |
| `smem_lut8` | as `smem_lut`, 8 k's a stage in 64 KB of dynamic shared memory, double buffered | 2 LDS.128 |

`corun_dp4a_ffma` runs the dp4a and FFMA GEMMs at once on two streams, one block of each a SM, with the dp4a GEMM's K
calibrated so both end together. `fused_dp4a_ffma` runs both in one inner loop: a 128 x 128 dp4a tile over K = 4 KW
and a 128 x 16 TNF FFMA tile over K = KW, each checked against its own reference. `k_probe1` is the first build's
probe, kept because ptxas alternated IDP.4A and FFMA one by one in it. `k_probe` measures instruction mixes of dp4a,
IMAD, FFMA, LOP3, packed FFMA2 (`fma.rn.f32x2`) and IMAD.WIDE in warp instructions a SM a clock, to say which pipes the
routes share; `k_split` runs the dp4a and FFMA chains in different warps of each sub-partition, so the warp
schedulers, not the SASS order, pick the mix. Raw pipe caps are the sm_120 census's
(note:20261015T0608Z-draft-census-sm120), not repeated here.

Run it on vy-nebius-2 from a clean commit (`run.sh` has the full command); outputs go to `$OUT` and
`summarize.py OUT` renders `summary.md`.
