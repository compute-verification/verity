# lowbit_dc: Dither's FP8 and FP4 routes on datacenter GPUs

Decision 78 (instead of B300, test Dither for FP4 and FP8 on datacenter GPUs and make sure it works;
note:20261017T0643Z-draft-lowbit-dc). Two questions on B200 (sm_100a, tcgen05) and H200 (sm_90a, wgmma), with sm_120's
answers in `../fp_exact/`, `../fp4_nobits/` and `../emu8/`:

- Are the FP8 and FP4 tensor kinds exact integer arithmetic, on which grids, and from which accumulator start
  (`fpx.cu`)?
- Does int8 Dither run bit-exact with int8 emulated in FP8 limbs, and at what time against native int8 in the same
  kernel (`egemm.cu`)?

| File | What |
|---|---|
| `fpx.cu` | Chains of tcgen05 kind::f8f6f4 E4M3, kind::mxf4 and kind::mxf4nvf4 (B200) or wgmma E4M3 (H200) on exact grids from four accumulator starts, each output against the int64 sum and against a round-toward-zero chain; the adder's alignment window; emu8's limb routes and the two Hopper-exact ones (k3h, e4r), every k32 prefix word (`fpx run OUT`) |
| `egemm.cu` | dgemm's Dither GEMM (`../dc/dgemm.cu`: the same receipt definition and single-stage skeleton) with int8 emulated in E4M3: k3z and k4s on tcgen05, k3h on wgmma, beside native int8 at the same tile (`i8`, and on B200 `i8r`, int8 in k4s's shape). Modes `selftest` (host: every limb identity), `gate` (every arm against the host reference, 0xA5 poison, no-write control, the device limb split against the host's) and `bench` (streamed, resident, and on B200 pipelined) |
| `analyze.py` | fpx's records into the exactness tables; `analyze.py gemm` egemm's gates and time ratios |
| `run.sh` | The job for one GPU, the family read from the device: `run.sh` (fpx) or `run.sh gemm` (egemm and dgemm with cuBLASLt) |

The tests are `../tests/test_lowbit_dc.py`: each emulated route's k32 word step in FP32 against the int64 word under the
adders fpx measured, analyze.py's tables on synthetic rows, and egemm's nvcc build and selftest.

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
research run --on vy-dc-b200 --project verity --campaign cmp-13 --source <tree> --cwd source --timeout 40m \
  --env GPU_LEASE_WHO=lowbit-dc -- bash benchmarks/pouw/lowbit_dc/run.sh [gemm]
~~~

Each run builds and dumps SASS before it takes the lease (one GPU, `gpu-lease 1 --max-min 15`), and runs for under a
minute on the GPU. `run.sh gemm` gates three shapes × two code distributions (uniform, limb extremes) and benches only if
every gate is exact. The time ratios are same-kernel ones: the skeleton is 14–75× slower than cuBLASLt int8.

On a B300 (sm_103a, no kind::i8) `run.sh` runs fpx as on the B200 (the same tcgen05 kinds; its NVF4 form is
UTCOMMA.BLOCK16 there); that run is what k4s rests on there: B200's adder (exact whenever representable, round toward
zero per MMA) holding on sm_103. `run.sh gemm` refuses on sm_103a, since its baselines are native int8.

Not covered: a production-grade (TMA, multi-stage, warp-specialized) emulated kernel, and FP4 emulation of int8 (emu8's
FP4 route is counted in `../emu8/accounting.py`, not built).
