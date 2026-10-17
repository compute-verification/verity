# dc: the epsilon program's measurements on Hopper and datacenter Blackwell

TENSOR's M3 (sm_90a: H100, H200) and M4 (sm_100a: B200) as gated opcode probes, the scalar pipes that bound the fold,
and the honest Dither GEMM with its duo fold on wgmma and on tcgen05 (note:20261017T0321Z-draft-eps1-tensor §5,
note:20261017T0310Z-report-eps1-program). Everything here builds with CUDA 13.0 on any machine and passes the SASS gate
there; the rates and the GEMM's exactness need the GPU (`run.sh`).

| File | What |
|---|---|
| `probes/sm90_wgmma.cu` | M3: wgmma alone (s8 m64nNk32 for N = 8, 16, 32, 64, 128, 192, 256, SS and RS, u8×s8, E4M3 N 128/256, E5M2, F16, BF16 N 128/256, TF32, b1 AND.POPC; wait_group 0 and 1; zero-data twins), wgmma s8 n256 beside a lane (warp-specialized `wgspec_` and interleaved `wgmix_`: mix, IMAD, LOP3, SHFL, duo), mma.sync s8 k16/k32, F16, BF16, TF32, b1, E4M3 |
| `probes/sm100_tcgen05.cu` | M4: tcgen05.mma kind::i8 M 64/128 × N 32–256, f8f6f4 E4M3 N 128/256, f16 BF16 (one thread issues, 4 a loop, into TMEM); tcgen05.ld 32x32b.x{1,4,16,32}, 16x64b.x16, 16x128b.x8, 16x256b.x4 alone; tcgen05.ld beside the saturated MMA (4 and 8 loader warps); mma.sync alone and beside it; lanes beside it |
| `probes/scalar.cu` | Every family: IADD3, LOP3, IMAD, FFMA, SHF, LEA.HI, SHFL, mix and duo alone, and the pairs that assign them to pipes (M3's INT32-pipe question, M4d) |
| `probes/lanes.cuh` | The lane ops all three share |
| `dc.cuh` | The K-major no-swizzle operand layout and its descriptors (wgmma and tcgen05), the instruction descriptor, mbarrier and tcgen05 wrappers, and duo (floored_gemm's) |
| `gen_wgmma.py`, `wgmma.cuh` | The wgmma wrappers (up to 128 accumulator registers an operand list), generated |
| `dgemm.cu` | The honest Dither GEMM, s8 C = X Wᵀ with every 32-step s32 prefix word folded by duo (3 ops a word): `dg90` (wgmma m64n128k32, accumulators in registers) and `dg100` (tcgen05.mma 128×256×32 into TMEM, tcgen05.ld to registers). Modes `cpuref`, `gate` (0xA5 poisoning, no-write control, a mismatch rerun with LBO/SBO exchanged) and `bench` (fold, no fold, cuBLASLt int8) |
| `dgemm_ref.py` | The receipt's numpy twin: C and every receipt word from X, W and the seed |
| `fp8acc.cu` | M4e for E4M3 (sm_100a): emu8's k3 and k4 limb planes through tcgen05.mma kind::f8f6f4 from TMEM preloaded with the binade constants, on five adversarial operand classes; counts accumulators that differ from the exact C + T and k32 words that differ from the s8 dot product. `selftest` runs the same checks with ideal accumulators on the CPU (all exact) |
| `analyze.py` | rows.jsonl → slots a SM a clock, every loop's issue rate, a (issue cycles a tensor instruction takes from the lane beside it), M3's and M4's pass lines, the fold's words a clock |
| `run.sh` | The job for one family (below); `DC_BUILD_ONLY=1` builds and gates without a GPU |

The tests are `../tests/test_dc.py` (wgmma.cuh current, the twin against a per-word loop and against dgemm.cu's host
reference, analyze's arithmetic). A probe's rate is opcode_probe's: warp instructions a SM a clock, timed only after its
SASS gate, its poisoned sink and its no-write control pass (`../opcode_probe/README.md`).

## Run plan

~~~text
research run --on <h200-node> --project verity --campaign pouw --source <tree> --cwd source --custody-ttl 12h \
  --env GPU_LEASE_WHO=dc -- bash benchmarks/pouw/dc/run.sh sm90
research run --on <b200-node> ... -- bash benchmarks/pouw/dc/run.sh sm100
~~~

The node needs CUDA 13.0 (`/usr/local/cuda-13.0`, driver ≥ 580) and the deployed tree (`research deploy install`). Each
run builds and SASS-gates everything before it touches the GPU, then under one lease: the GEMM gate at three shapes
(each dump rechecked by `dgemm_ref.py`), the bench only if all three are exact, the probes with nvidia-smi sampled every
200 ms, and `analyze.py` (`verdict.json`).

| Run | Measures | Gates | Pass line | GPU-hours |
|---|---|---|---|---|
| H200 `run.sh sm90` | ρ_T over 17 wgmma and 7 mma.sync forms; wgmma beside 5 lanes (two schedules); scalar alone and pairs; dg90 exact, fold vs no-fold vs cuBLASLt | 85 SASS gates, poisoned sinks, no-write controls; dgemm 0xA5 poison + control + twin | M3: no form above wgmma s8 m64n256k32 × 1.02 | 0.5 a run |
| B200 `run.sh sm100` | ρ_T5 (kind::i8 128×256), 10 MMA shapes; ρ_ld over 7 load geometries alone and beside the MMA; ρ_sync alone and beside; lanes beside; scalar; dg100 exact and timed; fp8acc (is kind::f8f6f4 exact in emu8's binades) | 51 SASS gates; as above; fp8acc's CPU selftest at build | M4: ρ_ld ≥ ρ_T5 and ρ_sync ≤ ρ_T5 at 2% | 0.5 a run |

A run is about 10 minutes of GPU time (builds ~1 min, the GEMM gate and bench ~2, the probes ~5); the half hour covers
deploy, sync and the build on the node. The request is **2 GPU-hours on 1× H200 and 2 GPU-hours on 1× B200 (4 in all,
each node at `--max-hours 2`)**: a first run, one fix-and-rerun if the GEMM gate is not exact (the descriptors' LBO/SBO
reading is the one layout field no CPU check reaches; the gate reports whether exchanging them is exact), and a repeat
run whose rates must agree within 2%.

Not covered yet: sparse wgmma and tcgen05 (metadata operands), the N sweep for the FP kinds (s8 only), cta_group::2 (a
CTA pair), tcgen05.ld x64/x128, M4e beyond fp8acc's binade question (kind::f16/tf32/f8f6f4 accumulation against the
silicon models in general), and the
pipelined (TMA, multi-stage) GEMMs whose receipts must equal these kernels'. On sm_90 CUDA 13.0 has no native FP8 or s4
mma.sync: e4m3 m16n8k32 is 2 HMMA.16816.F32 + 4 FADD (timed as that), s4 m16n8k64 a CALL (not timed).

## B300 (sm_103a)

sm_103 has no tcgen05 kind::i8: ptxas refuses it (`Feature '.kind::i8' not supported on .target 'sm_103a'`; CUDA
13.0), so `dg100` and the i8 probes don't build there (`scalar.cu` does). The honest kernel needs an exact route through
an FP kind, which the protocol states as its own T1 table (TENSOR §5: an FP design or kind::ti16, sm_107f only):

- **k3 on kind::f8f6f4 (E4M3)**, emu8's Karatsuba (its own branch, `cursor/emu8-ea24`, not on main): balanced
  base-16 limbs, xy = 240hh′ + 16mm′ − 15ll′; three MMAs a k32 step into two accumulators started at binade constants
  (C1 ulp 1, C2 ulp 1/8), and the word grows by 15·bits(acc1) + 2·bits(acc2), two IMADs. k4 (four MMAs, one C0) is the
  fallback. Both are exact only if the tcgen05 FP32 adder keeps C + T exact in that binade; TENSOR's table lists that
  adder as unmodeled, so `fp8acc.cu` measures it in the B200 run (the same kind) before any B300 time is asked.
- **NVFP4** (kind::mxf4nvf4, K 64, twice the FP8 rate): emu8's `accounting.py` `fp4_route` writes U as two E2M1 terms
  under per-block scales; its adder matches the existing silicon model (SIL BLACKWELL_SM100_NVF4/MXF4).

Cost against B200's kind::i8 kernel, at the rates TENSOR's table gives every 32-deep kind (256 slots a SM a clock):

| | B200 kind::i8 | B300 k3 (E4M3) |
|---|---|---|
| MMAs a k32 step a tile | 1 | 3 |
| committed words a SM a clock, tensor-bound | 256 | 85.3 |
| TMEM words loaded a committed word | 1 | 2 (acc1, acc2) |
| fold ops a word | 3 (duo) | 5 (2 IMAD + duo) |
| honest time at L = 32 (fold-bound on both) | 1 | ≈ 5/3 a SM (scaled by SM count and lane rate) |
| honest time where the tensor pipe binds | 1 | 3 |

So at L = 32, where TENSOR's table has both parts fold-bound, the missing int8 costs B300 the two recombination IMADs (≈
1.67× the fold's time a SM); where the tensor pipe binds, k3 costs 3× (k4 4×). The B300 measurements: `fp8acc` built
for sm_103a (its guard and device check name sm_100a today) and extended from one k32 step to emu8's K up to 16384,
then sm100_tcgen05.cu's f8f6f4, f16 and tcgen05.ld probes (its i8 forms removed) and `scalar.cu`: about 2 GPU-hours on
a 1× B300, asked only once fp8acc is exact on the B200.
