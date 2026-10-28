# fp_exact: are sm_120's FP8 and FP4 MMAs exact integer arithmetic on restricted inputs?

A chain of `K / k_step` MMAs of one format runs into one FP32 accumulator from the word C0. Every output is judged
bit for bit against the exact sum, which the device computes in int64 beside the chain. A partial sum is
*representable* when FP32 holds it exactly (its odd part is under 2^24). For each format and grid, the run measures
three things:

- whether every output whose MMA-boundary partials were all representable is the exact sum's word (`bad_rep` = 0);
- whether that holds whatever the accumulation order (`order_diff` = 0 over three orders of the K chunks);
- whether zero always comes out as +0 (`negzero` = 0).

| Format (`FMTS`) | PTX | SASS |
|---|---|---|
| `e4m3`, `e4m3_kind` | `mma.sync m16n8k32 .e4m3.e4m3.f32` (plain, and `kind::f8f6f4`) | `QMMA.16832.F32.E4M3.E4M3` |
| `mxf4`, `mxf4nv` | `kind::mxf4` and `kind::mxf4nvf4`, `scale_vec::2X`, ue8m0 = 0x7F (2^0) | `OMMA.SF.16864.F32.E2M1.E2M1.E8` |
| `nvf4` | `kind::mxf4nvf4`, `scale_vec::4X`, ue4m3 = 0x38 (1.0) | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` |
| `sp_mxf4`, `sp_nvf4` | `mma.sp::ordered_metadata m16n8k128`, the same scales | `OMMA.SF.SP.168128.F32.E2M1.E2M1.{E8,UE4M3.4X}` |

The grids (`GRIDS`, which `fp_exact.cu`'s `grids()` mirrors):

- E4M3 has the integers in [−16, 16] and their halves and quarters (`e4m3_int16`, `e4m3_half8`, `e4m3_quarter4`).
- E2M1 has the grid ×2 = {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12} halves (`e2m1_x2`) and its integer subset (`e2m1_int`).

The per-block scales (`SCALES`, which `fp_exact.cu`'s `make_scale()` mirrors; both operands sample from the one set, a
scale a 16-element block for NVFP4 / a 32-element block for MXFP4):

- `unit`: 1.0 on every block (E4M3 and the 2:4-sparse route, whose MMA takes unit scales, use only this).
- `lo`, `hi`: one fixed scale a block — NVFP4's UE4M3 3 / 15, MXFP4's UE8M0 2^−3 / 2^3.
- `mix`: a scale sampled per block and per operand — NVFP4 the odd integers 1..15 (Theorem K's 2-adic form keeps c = 1
  under odd-integer scales), MXFP4 the powers 2^k, k in [−3, 3] (a cheater's range stretch).
- `bind`, `bind26`: MXFP4 blocks alternate 2^7 and 2^−7 (scale-exponent spread 28, one past the 27-bit frac anchor) /
  2^7 and 2^−6 (spread 26). They probe whether a wide intra-firing spread makes the align-add's 27-bit anchor truncate
  a product beyond what FP32's 24-bit significand already drops.

The reference keeps the integer dot in units of 2^−(2s + sh), where `sh` carries the blocks' most negative scale
exponent so every shift is non-negative; the binade C0 is matched to that scaled resolution. The K-limit table
(`limits`) reports, per format, grid and scale, the largest K exact for every sample and the first K that rounds, with
the max scale product.

The distributions (`DISTS`) are:

- uniform;
- all-maximal;
- maximal with signs;
- a ramp: maximal products first, then ±minimal ones, which sum to an even number in every firing;
- pairwise cancellation;
- an edge: maximal products first, then 0 or the minimal product, so a firing's sum is odd about half the time. This
  is the sample that lands a partial on an odd value just past 2^24, or past C0's binade.

K runs over every power of two from one firing to 262144.

Each value of C0 (`c0_words`) has a role:

- `zero`.
- `binade`: NOBITS' 0x4B400000 generalized, so its ulp is one product unit (`note:20261016T0703Z-draft-fp4-nobits-probe`).
- `nobits`: NOBITS' own word, whose ulp of 1 is coarser than a dyadic grid's units.
- `over`: one binade higher, so FP32 must round.
- `negzero`, on the cancelling samples.

| File | What |
|---|---|
| `fp_exact.cu` | One kernel a format: 3 accumulation orders of the chain, the int64 reference and its representability, the counts, the first tiles' words (`gpu_kit/bench.cuh`'s 0xA5 poisoning and no-write control). `--dump` writes a config's operands from the device's generator, without a GPU |
| `fp_exact.py` | `configs` (the sweep over formats, grids, scales, distributions, C0 and K), `run` (the SASS gate, then the sweep and its control), `summarize` (the host's own regeneration and recheck of the recorded tiles, a replay of them through `verity.ml.circuits.silicon`'s models, the verdicts), `limits` (the K-limit table per format, grid and scale), `rates` (`../opcode_probe/probes/mma.cu`'s rows as MACs per SM per clock against IMMA, each format's `_data` probe against its fixed one to rule out a data-dependent clock) |
| `run.sh` | The job: build and gate outside the lease, then under one `gpu-lease` the sweep and the MMA rates, then the summary and the K-limit table |

~~~text
research run --on vy-nebius-2 --project verity --cwd source -- bash benchmarks/pouw/fp_exact/run.sh [--quick]
~~~

The host refuses a config in each of these cases:

- the device's counts are unwritten or short;
- a recorded word was left poisoned;
- the device's C0 isn't the config's;
- the device's int64 sums disagree with the host's.

Every config of the no-write control must be refused. The 2:4-sparse firing has no silicon model of its own, so the
replay tests a hypothesis: the dense adder over its 128 logical elements, with one align-add and one RZ per firing.
The tests are `../tests/test_fp_exact.py`.
