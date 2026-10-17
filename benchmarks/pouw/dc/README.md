# dc: the epsilon program's measurements on Hopper and datacenter Blackwell

TENSOR's M3 (sm_90a: H100, H200) and M4 (sm_100a: B200) as gated opcode probes, the scalar pipes that bound the fold,
and the honest Dither GEMM with its duo fold on wgmma and on tcgen05 (note:20261017T0321Z-draft-eps1-tensor §5,
note:20261017T0310Z-report-eps1-program). Everything here builds with CUDA 13.0 on any machine and passes the SASS gate
there; the rates and the GEMM's exactness need the GPU (`run.sh`).

| File | What |
|---|---|
| `probes/sm90_wgmma.cu` | M3: wgmma alone (s8 m64nNk32 for N = 8, 16, 32, 64, 128, 192, 256, SS and RS, u8×s8, E4M3 N 128/256, E5M2, F16, BF16 N 128/256, TF32, b1 AND.POPC; wait_group 0 and 1; zero-data twins), wgmma s8 n256 beside a lane (warp-specialized `wgspec_` and interleaved `wgmix_`: mix, IMAD, LOP3, SHFL, duo; `wgU_`: the lanes or mma.sync run until the wgmma ends), mma.sync s8 k16/k32, F16, BF16, TF32, b1, E4M3 |
| `probes/sm100_tcgen05.cu` | M4: tcgen05.mma kind::i8 M 64/128 × N 32–256, f8f6f4 E4M3 N 128/256, f16 BF16 (one thread issues, 4 a loop, into TMEM); tcgen05.ld 32x32b.x{1,4,16,32,64,128}, 16x64b.x16, 16x128b.x8, 16x256b.x4 alone; tcgen05.ld, mma.sync and lanes beside the saturated MMA, block-timed and exact (`ldU_`, `mmaU_`, `syncU_`, `tclaneU_`: one role fixed, the other until it ends) |
| `probes/scalar.cu` | Every family: IADD3, LOP3, IMAD, FFMA, SHF, LEA.HI, SHFL, mix and duo alone, and the pairs that assign them to pipes (M3's INT32-pipe question, M4d) |
| `probes/lanes.cuh` | The lane ops all three share |
| `dc.cuh` | The K-major no-swizzle operand layout and its descriptors (wgmma and tcgen05), the instruction descriptor, mbarrier and tcgen05 wrappers, and duo (floored_gemm's) |
| `gen_wgmma.py`, `wgmma.cuh` | The wgmma wrappers (up to 128 accumulator registers an operand list), generated |
| `dgemm.cu` | The honest Dither GEMM, s8 C = X Wᵀ with every 32-step s32 prefix word folded by duo (3 ops a word): `dg90` (wgmma m64n128k32, accumulators in registers) and `dg100` (tcgen05.mma 128×256×32 into TMEM, tcgen05.ld to registers). Modes `cpuref`, `gate` (0xA5 poisoning, no-write control, a mismatch rerun with LBO/SBO exchanged) and `bench` (fold, no fold, cuBLASLt int8) |
| `dgemm_ref.py` | The receipt's numpy twin: C and every receipt word from X, W and the seed |
| `fp8acc.cu` | M4e for E4M3 (sm_100a): emu8's k3 and k4 limb planes through tcgen05.mma kind::f8f6f4 from TMEM preloaded with the binade constants, on five adversarial operand classes; counts accumulators that differ from the exact C + T and k32 words that differ from the s8 dot product. `selftest` runs the same checks with ideal accumulators on the CPU (all exact) |
| `analyze.py` | rows.jsonl → slots a SM a clock, every loop's issue rate, a (issue cycles a tensor instruction takes from the lane beside it), M3's and M4's pass lines, sm_100's TMEM model (honest at L = 32, the L that saturates the tensor, the two-stage LP's ε), the fold's words a clock; `agree A B` checks a repeat |
| `census.py` | Runs → machine_model's census lines, a device directory a family (`../machine_model/data/devices/sm_90/`, `sm_100/`: census.csv and rows.csv), and the tables: T1 a form, ε_tensor at L = 32, the joint issue rows, q*, L_bal |
| `run.sh` | The job for one family (below); `DC_BUILD_ONLY=1` builds and gates without a GPU |

The tests are `../tests/test_dc.py` (wgmma.cuh current, the twin against a per-word loop and against dgemm.cu's host
reference, analyze's arithmetic, census.py's facets on a synthetic run, the device directories' form). A probe's rate is
opcode_probe's: warp instructions a SM a clock, timed only after its SASS gate, its poisoned sink and its no-write
control pass (`../opcode_probe/README.md`).

## Running

~~~text
research run --on vy-dc-h200 --project verity --campaign cmp-13 --source /workspace --cwd source --custody-ttl 12h \
  --timeout 60m --env GPU_LEASE_WHO=dc -- bash benchmarks/pouw/dc/run.sh sm90
research run --on vy-dc-b200 ... -- bash benchmarks/pouw/dc/run.sh sm100
python3 benchmarks/pouw/dc/analyze.py agree ~/.research/runs/RUN_A/out ~/.research/runs/RUN_B/out
python3 benchmarks/pouw/dc/census.py sm90 ~/.research/runs/RUN/out... --write benchmarks/pouw/machine_model/data/devices/sm_90
~~~

The node needs CUDA 13.0 (`/usr/local/cuda-13.0`, driver ≥ 580). Each run builds and SASS-gates everything before it
touches the GPU (91 gates on sm_90a, 68 on sm_100a), then under one lease: the GEMM gate at three shapes (each dump
rechecked by `dgemm_ref.py`), the bench only if all three are exact, fp8acc (sm100), the probes with nvidia-smi sampled
every 200 ms, and `analyze.py` (`verdict.json`). A run is about 6 minutes on the GPU.

## Results (1× H200, 1× B200; driver 580.173.02, CUDA 13.0.88)

The census is `../machine_model/data/devices/sm_90/` (runs r20261017-060118-0230, r20261017-060725-d7ca,
r20261017-065542-af0a, r20261017-065542-af19) and `sm_100/` (r20261017-062812-a7a7, r20261017-063727-4f9a,
r20261017-063734-4b49); every repeat agrees with its twin within 2% (worst 1.0008). f_max at default clocks: H200 1980
MHz, B200 1965 MHz.

- **The Dither GEMM is exact on both** (three shapes each, 0xA5 poison, the no-write control rejected, the twin agrees;
  the descriptors as written, no LBO/SBO exchange). These kernels aren't pipelined: at 8192³ the fold takes 7.42 ms
  (H200) and 4.86 ms (B200) against cuBLASLt's 0.74 and 0.32.
- **sm_90 (M3 passes).** Every wgmma kind at N ≥ 64 writes 127.99 slots/SM/clk (N = 8/16/32: 28.4/51.2/85.3);
  mma.sync s8 k16 124.7, the other forms 84.0. mma.sync shares the tensor pipe (beside saturated wgmma: tensor 0.94,
  mma.sync 1.2 slots/SM/clk). LOP3, SHF and LEA.HI share one half-rate pipe (1.97), IMAD another (2.0); FFMA is issue-
  bound alone. Beside saturated wgmma every lane pipe keeps about half: the issue facet is IGMMA.64x256x32 62.7 + every
  other instruction 0.98 SMSP-cycles, and the IMAD and ALU pipes' facets cost the IGMMA about 63 too.
- **sm_100 (M4 fails: the TMEM load binds).** tcgen05.mma i8, f8 and f16 write 256.05 slots/SM/clk at M = 128, N ≥ 128
  (M = 64: 128); mma.sync 64.0, on the same pipe. tcgen05.ld alone reaches 244.5 words/SM/clk (x128; x32 242.7), 233.3
  beside the saturated MMA. Honest at L = 32 loads every slot it writes and balances at 0.914 of ρ_T5 (234.1 words/SM/
  clk); the tensor saturates from L = 64; the two-stage LP gives ε_tensor ≤ 2.3% at L = 32. A tcgen05.mma costs one
  issue slot (UTCIMMA 1.09 + other 0.98), so lanes beside it keep their rates.
- **q*** (the issue facet's tensor coefficient over the words a lane gets): sm_90 0.49, sm_100 0.001 (0.047 with the
  tcgen05.ld.x32 issue that brings the words to registers), against sm_120's 2.03. The fold binds at L = 32 on both
  (duo 31.3 and 31.5 words/SM/clk alone: L_bal 131 and 260).

Not covered yet: sparse wgmma and tcgen05 (metadata operands), the N sweep for the FP kinds (s8 only), cta_group::2 (a
CTA pair), M4e beyond fp8acc's binade question (kind::f16/tf32/f8f6f4 accumulation against the silicon models in
general), lanes beside wgmma with more than one lane warp a SMSP (`wgU_` reaches only 0.3–0.4 of alone, under the
block-timed points the sm_90 facets rest on), and the pipelined (TMA, multi-stage) GEMMs whose receipts must equal these
kernels'. On sm_90 CUDA 13.0 has no native FP8 or s4 mma.sync: e4m3 m16n8k32 is 2 HMMA.16816.F32 + 4 FADD (timed as
that), s4 m16n8k64 a CALL (not timed).

## B300 (sm_103a)

sm_103 has no tcgen05 kind::i8: ptxas refuses it (`Feature '.kind::i8' not supported on .target 'sm_103a'`; CUDA
13.0), so `dg100` and the i8 probes don't build there (`scalar.cu` does). The honest kernel needs an exact route through
an FP kind, which the protocol states as its own T1 table (TENSOR §5: an FP design or kind::ti16, sm_107f only):

- **k3 on kind::f8f6f4 (E4M3)**, emu8's Karatsuba (its own branch, `cursor/emu8-ea24`, not on main): balanced
  base-16 limbs, xy = 240hh′ + 16mm′ − 15ll′; three MMAs a k32 step into two accumulators started at binade constants
  (C1 ulp 1, C2 ulp 1/8), and the word grows by 15·bits(acc1) + 2·bits(acc2), two IMADs. k4 (four MMAs, one C0) is the
  fallback. Both are exact only if the tcgen05 FP32 adder keeps C + T exact in that binade. **On the B200 it does**:
  k3 and k4 on all five adversarial classes, 0 bad accumulators and 0 bad k32 words (r20261017-060121-661e,
  r20261017-063727-4f9a), so the E4M3 route holds for a k32 step.
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
a 1× B300.

`../lowbit_dc/` has since run both routes end to end on the B200 in dgemm's kernel and receipts, bit-exact: k4s, k4 with
the accumulator restored to its binade by tcgen05.st after each word leaves, costs one IADD3 a word (no IMAD) and is
1.24–1.45× native int8 in the same pipelined shape, while k3 from +0 (2 FFMA + IADD3) is 1.9–2.2×. Its `fpx` (every
tcgen05 FP8 and FP4 kind to K = 2^17, and the adder's window) builds for sm_103a, and is the B300 exactness run.
