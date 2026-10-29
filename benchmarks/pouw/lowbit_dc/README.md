# lowbit_dc: Dither's FP8 and FP4 routes on datacenter GPUs

Dither for FP4 and FP8 on datacenter GPUs, in place of B300. Two questions on B200 (sm_100a, tcgen05) and H200 (sm_90a, wgmma), with sm_120's
answers in `../emu8/`:

- Are the FP8 and FP4 tensor kinds exact integer arithmetic, on which grids, and from which accumulator start
  (`fpx.cu`)?
- Does int8 Dither run bit-exact with int8 emulated in FP8 limbs, and at what time against native int8 in the same
  kernel (`egemm.cu`)?

| File | What |
|---|---|
| `fpx.cu` | Chains of tcgen05 kind::f8f6f4 E4M3, kind::mxf4 and kind::mxf4nvf4 (B200) or wgmma E4M3 (H200) on exact grids from four accumulator starts, each output against the int64 sum and against a round-toward-zero chain; the adder's alignment window; emu8's limb routes and the two Hopper-exact ones (k3h, e4r), every k32 prefix word (`fpx run OUT`) |
| `egemm.cu` | dgemm's Dither GEMM (`../dc/dgemm.cu`: the same receipt definition and single-stage skeleton) with int8 emulated in E4M3: k3z and k4s on tcgen05, k3h on wgmma, beside native int8 at the same tile (`i8`, and on B200 `i8r`, int8 in k4s's shape). Modes `selftest` (host: every limb identity), `gate` (every arm against the host reference, 0xA5 poison, no-write control, the device limb split against the host's) and `bench` (streamed, resident, and on B200 pipelined) |
| `pgemm.cu` | The k4s Dither GEMM pipelined on B200: a loader warp (cp.async.bulk of pre-tiled stages into a 6–8-stage ring on mbarriers), an MMA warp (tcgen05 into 4 TMEM word buffers), fold warps (tcgen05.ld x32, double-buffered). Routes k4s, k4p (k4s with a public per-word offset in the receipt, so its extraction can go to the IMAD pipe), k4b (k4s's receipt from a raw running sum, the public offset removed before the fold), k4t (k4s with the running sum in tensor memory as i8c carries it: a 128 × 256 CTA, each 128-column tile's accumulator holding 1.5·2^23 + T − T_base and its base' = T_base − 1.5·2^23 in the other half of TMEM, the word one IMAD-pipe add bits + base', a rebase every 7 words at L = 32 and 3 at L = 64 that keeps the accumulator in [2^23, 2^24); receipt tile 128, arms named fw1), int8 Dither in the same skeleton (i8c: running sum in TMEM, duo alone a word; i8r: fresh accumulator, T in registers), plain FP8/int8 and cuBLASLt; time-split modes (EXT, LD, MMA, FOLDONLY, and MMARES/FULLRES on resident operands); each arm's SM clock and median CTA clocks. Modes `selftest`, `gate`, `bench` |
| `analyze.py` | fpx's records into the exactness tables; `analyze.py gemm` egemm's gates and time ratios; `analyze.py pgemm` pgemm's γ per spacing, time split, binding facet and the word loop's issue cost from SASS |
| `run.sh` | The job for one GPU, the family read from the device: `run.sh` (fpx), `run.sh gemm` (egemm and dgemm with cuBLASLt) or `run.sh pgemm` (B200) |

The tests are `../tests/test_lowbit_dc.py`: each emulated route's k32 word step in FP32 against the int64 word under the
adders fpx measured, analyze.py's tables on synthetic rows, and egemm's and pgemm's nvcc builds and selftests (pgemm's
also its spill-free FW 2 kernels and each route's word-loop issue cost).

## The routes

Limbs are emu8's: x = 16h + l with h = (x + 8) >> 4 ∈ [−8, 8], l ∈ [−8, 7], m = h + l ∈ [−15, 14], every plane value an
E4M3 integer. A route's word step adds one k32 prefix word to T, which stays in registers:

| Route | Family | MMAs a step | Accumulators | Word step |
|---|---|---|---|---|
| k3z | B200 | 3 kind::f8f6f4 | (4h)(4h′) + (−l)(l′), mm′ from +0 | 2 FFMA into the 1.5·2^23 binade + IADD3 |
| k4s | B200 | 4 kind::f8f6f4 | (16h + l)(16h′ + l′) onto 1.5·2^23, restored by tcgen05.st | 1 IADD3 |
| k3h | H200 | 3 wgmma E4M3 | hh′, mm′, ll′ from +0 | 3 FFMA + IADD3 |
| e4r | H200, B200 | 1 | the E4M3 integer grid from +0 | FFMA + IADD3 (fpx only) |

Hopper's FP8 adder keeps a 14-bit window under the largest term of an instruction, the accumulator included, so routes
that start at the binade (k3b, k4b) or whose k32 sums pass 2^13 next to small terms (k3z, k4z) are wrong there; k3h's
terms and partial sums stay below 2^13. tcgen05's FP32 adder is exact whenever the result is representable and rounds
toward zero per MMA past it.

## Run

~~~text
research run --on vy-dc-b200 --project verity --source <tree> --cwd source --timeout 40m \
  --env GPU_LEASE_WHO=lowbit-dc -- bash benchmarks/pouw/lowbit_dc/run.sh [gemm|pgemm]
~~~

Each run builds and dumps SASS before it takes the lease (one GPU, `gpu-lease 1 --max-min 15`), and runs for under a
minute on the GPU. `run.sh gemm` gates three shapes × two code distributions (uniform, limb extremes) and benches only if
every gate is exact. The time ratios are same-kernel ones: the skeleton is 14–75× slower than cuBLASLt int8.

On a B300 (sm_103a, no kind::i8) `run.sh` runs fpx as on the B200 (the same tcgen05 kinds; its NVF4 form is
UTCOMMA.BLOCK16 there); that run is what k4s rests on there: B200's adder (exact whenever representable, round toward
zero per MMA) holding on sm_103. `run.sh gemm` refuses on sm_103a, since its baselines are native int8.

`run.sh pgemm` gates every route, spacing (L = 32, 64) and receipt tile width on three shapes × three distributions
(uniform, edge, and max: every code −128, so k4t's accumulator moves at its bound; k4t only where N is a multiple of 256), and
benches only if every gate is exact. Its tile is 128 × 128 on one CTA, so plain FP8 there is operand-feed bound (L2) below
cuBLASLt; γ is taken against cuBLASLt, and FULLRES gives it with the feed removed. Sent DC's pipelined int8 kernel
(`dc/pgemm.cu` and `dc.cuh` as `--send dc-pgemm.cu --send dc.cuh`, with
`--env DC_PGEMM_COMMIT=SHA`), it also runs that kernel's `rate` under the same lease, and analyze.py sets each route's
words a SM a clock of its median CTA (DC's metric) against DC's fastest honest configuration at each L.

Not covered: FP4 emulation of int8 (emu8's FP4 route is counted in `../emu8/accounting.py`, not built), and pgemm on a
CTA pair (cta_group::2, 256-wide tiles), which a production kernel would use for its operand feed.
