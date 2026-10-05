# Fused sumcheck rounds for A-GKR on the GPU (a-fused lane)

One Triton kernel per sumcheck round: load the tables' u32 limbs for a tile, fold with the previous challenge, evaluate the
degree-3 round polynomial at 4 points, write the folded tables -- nothing else touches HBM.  Measured on an RTX 4090
(vy-sp1, torch 2.6 cu124, Triton 3.2); report: `note:r20-proof/a-fused/20260922T0630Z-report-fused-sumcheck-4090` (+ `.json`,
`fused_logup_4090.json`), roofline: `note:r20-proof/a-fused/20260922T0536Z-report-roofline`.

| file | what |
|---|---|
| `field.py` | BabyBear Montgomery arithmetic (R = 2^32) and the BabyBear^6 (`X^6 = W`) product, emulated in Python; constants for the kernels |
| `round_simt.py` | fused checker round `sum eq . f . (g o h)` on SIMT INT32 (3 tables; ext or base input; fold / eval / fold+eval) |
| `round_tc.py` | same fusion with the fold on `tl.dot` int8 tiles (in-kernel limb decomposition, INT32 accumulate, Montgomery recomposition in registers) |
| `round_logup.py` | fused logUp fractional-GKR round `sum eq (p_l q_r + p_r q_l + lambda q_l q_r)` on SIMT (5 tables) |
| `bench.py`, `bench_logup.py` | pod harnesses: per-size kernels (eager / graphed / cold L2), 6-layer 130-round layer model, launch latency |
| `reprice.py` | laptop: substitute the measured sequence into `explore.tensor`'s A row |
| `tests/` | CPU: arithmetic + emulated round vs `tensor/reference.py`; pod (`TC_DEVICE=cuda`): every kernel mode bit-exact vs the reference |

## Result in one table (n = 2^21 input elements per table, cold L2)

| round | HBM floor | SIMT fused | TC fused | torch composition (previous lane) |
|---|---|---|---|---|
| checker, ext input | 225 us | **273 us** (828 GB/s, 1.22x) | 506 us (2.25x) | ~4-5 ms equivalent (0.012 utilisation) |
| logUp, ext input | 374 us | **600 us** (629 GB/s, 1.62x) | not built | modelled |
| 6-layer / 130-round checker model | 3.67 ms | **6.20 ms** graphed | 8.48 ms | 91.4 ms graphed |

**A100 rows** (anchor-a100 lane, NVIDIA A100-SXM4-80GB, torch 2.6 cu124, Triton 3.2; runs `r20260922-064953-f9bb` (checker,
25 tests bit-exact on CUDA) and `r20260922-071602-522d` (logUp); floors at the A100's measured copy bandwidth 1,754 GB/s;
full table `note:r20-proof/anchor-a100/20260922T0820Z-report-anchor-a100`):

| round (A100) | HBM floor | SIMT fused | TC fused | vs 4090 (SIMT) |
|---|---|---|---|---|
| checker, ext input, 2^21 | 129 us | **474 us** (478 GB/s, 3.7x floor; SIMT model 0.62 of the A100 INT32 peak) | 1,569 us | 0.58x |
| checker, base input, 2^22 | 115 us | 607 us (332 GB/s) | -- | 0.42x |
| logUp, ext input, 2^21 | 215 us | **1,469 us** (257 GB/s) | not built | 0.41x |
| 6-layer / 130-round checker model | 2.11 ms | **12.35 ms** graphed (13.40 eager; tail 32 us/round graphed) | 20.85 ms | 0.50x |

On the A100 the fused round is **not** HBM-bound: it runs at 0.27 of the card's HBM floor and ~0.6 of its INT32 peak (the A100
has 108 SMs x 64 INT32 lanes at 1.41 GHz = 9.7 T op/s, 0.47x the 4090's 20.6 T), so it lands at 0.5-0.6x the 4090 where
the bandwidth ratio would have predicted 2.0x.  Karatsuba / fewer Montgomery reductions (recommendation 4's "not needed
while memory-bound") is the lever on the anchor; the TC variant loses by 3.3x there.

SIMT wins.  The checker round is HBM-bound on this card: the SIMT arithmetic (0.5 of INT32 peak) hides under the memory
time and the kernel runs at 0.94-0.98 of the achievable copy bandwidth.  Tensor cores cannot shorten a memory-bound round,
and the MMA layout conversions plus the occupancy the int8 tiles cost make the TC variant 1.8-2.2x slower.  The logUp
round (7.5 extension products per element instead of 3.5) is the first one that turns compute-bound on SIMT (0.62 of
its HBM floor), so it is the only place a `tl.dot` product path could still pay -- if it beats the conversion cost measured
in `round_tc.py`.

## Recommended layout for the A-gpu prover

1. **Coefficient-major u32 tables** (`[6, n]` int32, one row per BabyBear^6 coefficient, Montgomery form): each of the six
   loads per element is coalesced, no AoS shuffles, and the even/odd pair split is a stride-2 gather inside a tile.  Never
   materialise int8 limbs in HBM: the previous lane's 4-8x byte blow-up is where the 0.012 came from.
2. **Fold fused into the next round's evaluation** (`FOLD + EVAL`): the folded table is written once and read by the next
   kernel; the round's 4 sums come out of the same tile.  Per input element this is 108 bytes (checker) / 180 bytes (logUp),
   about 40% less than an unfused fold -> eval pipeline.
3. **Keep the trace in the base field as long as the protocol allows** (`round_simt` `DEG_IN=1`): the first two rounds of
   a layer read 4-byte elements, 6x fewer bytes than the extension rounds; the base -> extension fold is 1.28x its floor.
4. **SIMT Montgomery arithmetic, R = 2^32**: one `mul.lo` + one `umulhi` + conditional add per base product; the extension
   product is 36 base products + 6 reductions of the `X^6 = W` wrap (schoolbook; Karatsuba is not needed while the round is
   memory-bound).  Challenges are pre-scaled by R so no extra scaling appears in the kernel; the block sums are `int64`
   partials fixed up on the host (`FusedRound.finish_s`).
5. **Block = 256 folded elements, 4 warps** for n >= 2^18; `tl.dot` is not on the critical path anywhere.
6. **CUDA-graph the layer** (a 130-kernel graph runs 6.20 ms vs 8.72 ms eager); the 90 tail rounds (n <= 2^15) cost a fixed
   ~16 us each *executing* -- a single-CTA dependent chain, not launch latency.  Binding several `c` variables per verifier
   message (PROTOCOL.md 5.4) or interleaving the six layers' tails on one stream is the lever; fusing tail rounds into one
   kernel cannot remove a per-round network round trip.
7. **Next factor is bytes, not kernel efficiency**: generate `eq` in-kernel from the challenge (saves 1/3 of the checker
   round's traffic), keep base-field tables longer, and consider A100/H100 for bandwidth (2.0x / 3.3x), not tensor peak.

## Running

~~~bash
# laptop (CPU, < 30 s)
cd backends/gkr && python -m pytest tensor/fused/tests -q
# pod
cd /workspace/afused/backends/gkr && TC_DEVICE=cuda /workspace/tc/venv/bin/python -m pytest tensor/fused/tests -q
/workspace/tc/venv/bin/python -m tensor.fused.bench --out results/fused.json
/workspace/tc/venv/bin/python -m tensor.fused.bench_logup --out results/fused_logup.json
# laptop: re-price A
cd backends/numerical && uv run python ../gkr/tensor/fused/reprice.py
~~~

Triton notes: module-level Python ints must be wrapped as `tl.constexpr` inside `@triton.jit` bodies; `tl.dot` int8 on
sm_89 needs M, N >= 16 and K >= 16 with both operands in shared-memory-friendly layouts, and its int32 result carries an
MMA layout whose conversion back to the element layout is the cost that sank `round_tc` (see the report); first
compilation of the logUp kernel (5 tables x 4 points fully unrolled) takes ~10 minutes.
