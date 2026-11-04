# dc: the epsilon program's measurements on Hopper and datacenter Blackwell

The tensor measurements M3 (sm_90a: H100, H200) and M4 (sm_100a: B200) as gated opcode probes, the scalar pipes that
bound the fold, and the honest Dither GEMM with its duo fold on wgmma and on tcgen05. Everything here builds with CUDA
13.0 on any machine and passes the SASS gate there; the rates and the GEMM's exactness need the GPU (`run.sh`).

| File | What |
|---|---|
| `probes/sm90_wgmma.cu` | M3: wgmma alone (s8 m64nNk32 for N = 8, 16, 32, 64, 128, 192, 256, SS and RS, u8×s8, E4M3 N 128/256, E5M2, F16, BF16 N 128/256, TF32, b1 AND.POPC; wait_group 0 and 1; zero-data twins), wgmma s8 n256 beside a lane (warp-specialized `wgspec_` and interleaved `wgmix_`: mix, IMAD, LOP3, SHFL, redux, reduce, duo; `wgU_`: the lanes or mma.sync run until the wgmma ends), mma.sync s8 k16/k32, F16, BF16, TF32, b1, E4M3; `wgli_` the honest LI loop beside the wgmma with no checkpoint, with LI-CP-W's redux and with its reduce, and `wgli_fill` the redux loop with ffill's filler |
| `ffill.cuh`, `ffill.cu`, `ffill_ref.py`, `ffill.sh` | ffill, the H200's FFMA filler (`dither-2s-licpw-redux-l32`'s `[filler]`): an IMAD and two `fma.rn.f32` on 3 of a thread's 128 accumulator words a MMA, two chains from the round's coins. `ffill_ref.py` is the receipt in exact arithmetic (F32FmaV2, rounded once) and checks `ffill.cu`'s dumps (`fma`: specials, subnormals, ties and seeded cases; `chain`: the kernel's chains); `ffill.sh` is the H200 job: build, SASS check (no `.FTZ`, the chain an IMAD and two FFMA a step), conformance, and the probes for `wgli_fill` against `wgli_redux` |
| `probes/sm100_tcgen05.cu` | M4: tcgen05.mma kind::i8 M 64/128 × N 32–256, f8f6f4 E4M3 N 128/256, f16 BF16 (one thread issues, 4 a loop, into TMEM); tcgen05.ld 32x32b.x{1,4,16,32,64,128}, 16x64b.x16, 16x128b.x8, 16x256b.x4 alone; tcgen05.ld, mma.sync and lanes (mix, IMAD, LOP3, duo, redux, reduce) beside the saturated MMA, block-timed and exact (`ldU_`, `mmaU_`, `syncU_`, `tclaneU_`: one role fixed, the other until it ends); `tcliU_` the honest LI loop (TMEM load then LI fold a segment) with no checkpoint, with LI-CP-W's redux reduction, and with its 5-shuffle fallback, so the ratio is what the checkpoint costs the latency-bound loop; `tcliU_floor` the floor program of `dither-2s-licpw-redux-ldtm0-l32` (LI-CP-W redux on words already in registers, ♭'s 544 : 512 : 24 a UTCIMMA, 12 lane warps), so its issue is the most the floor's own mix gets beside the tensor |
| `probes/scalar.cu` | Every family: IADD3, LOP3, IMAD, FFMA, SHF, LEA.HI, SHFL, mix, duo, redux (one `redux.sync.xor.b32`) and reduce (the same by 5 butterfly shuffles + 5 XORs) alone, and the pairs that assign them to pipes (M3's INT32-pipe question, M4d). redux/reduce are LI-CP-W's per-segment checkpoint reduction |
| `probes/lanes.cuh` | The lane ops all three share |
| `dc.cuh` | The K-major no-swizzle operand layout and its descriptors (wgmma and tcgen05), the instruction descriptor, mbarrier and tcgen05 wrappers, and duo (floored_gemm's) |
| `gen_wgmma.py`, `wgmma.cuh` | The wgmma wrappers (up to 128 accumulator registers an operand list), generated |
| `dgemm.cu` | The honest Dither GEMM, s8 C = X Wᵀ with every 32-step s32 prefix word folded by duo (3 ops a word): `dg90` (wgmma m64n128k32, accumulators in registers) and `dg100` (tcgen05.mma 128×256×32 into TMEM, tcgen05.ld to registers). Modes `cpuref`, `gate` (0xA5 poisoning, no-write control, a mismatch rerun with LBO/SBO exchanged) and `bench` (fold, no fold, cuBLASLt int8) |
| `dgemm_ref.py` | The receipt's numpy twin: C and every receipt word from X, W and the seed, at meta.json's commit spacing L |
| `pgemm.cu` | The honest Dither GEMM on sm_100a, pipelined: one CTA a SM holds a 128 × 512 block as NB = 2 or 4 TMEM tiles in flight (all 512 columns); MMA warps (one, or one a tile: MI) issue lm tcgen05.mma kind::i8 into a tile and tcgen05.commit to its full barrier; LW loader warpgroups wait on it, tcgen05.ld 32x32b.x32 the rows, hand the tile back on its empty barrier after the last load and fold with duo; L = 32 lm. Arms: 0 pipeline (loaders hand back only), 1 TMEM stage (load, no fold), 2 honest, 3 MMA self-wait (no loaders). Operands are resident (K-periodic inputs), so the rate leaves out operand streaming. Modes `gate` (MODE 2, 0xA5 poison, host reference, dumps for `dgemm_ref.py`) and `rate` |
| `pipe.py`, `pipe.sh` | pgemm's job and its analysis: the build gate (no spill, launchable registers, UTCIMMA and LDTM.x32 present), the rate per arm and spacing against ρ_T5, the honest loader's issue a word from SASS, q*, ε_tensor at L = 32, and `agree` for a repeat |
| `fp8acc.cu` | M4e for E4M3 (sm_100a): emu8's k3 and k4 limb planes through tcgen05.mma kind::f8f6f4 from TMEM preloaded with the binade constants, on five adversarial operand classes; counts accumulators that differ from the exact C + T and k32 words that differ from the s8 dot product. `selftest` runs the same checks with ideal accumulators on the CPU (all exact) |
| `analyze.py` | rows.jsonl → slots a SM a clock, every loop's issue rate, a (issue cycles a tensor instruction takes from the lane beside it), M3's and M4's pass lines, sm_100's TMEM model (honest at L = 32, the L that saturates the tensor, the two-stage LP's ε), the fold's words a clock; `agree A B` checks a repeat |
| `census.py` | Runs → machine_model's census lines, a device directory a family (`../machine_model/data/devices/sm_90/`, `sm_100/`: census.csv and rows.csv), and the tables: T1 a form, ε_tensor at L = 32, the joint issue rows, q*, L_bal |
| `run.sh` | The job for one family (below); `DC_BUILD_ONLY=1` builds and gates without a GPU |
| `probes/uniform.cuh`, `probes/sm90_uniform.cu`, `probes/sm100_uniform.cu`, `probes/sm90_issue.cu`, `probes/sm100_issue.cu`, `probes/sm90_sat.cu`, `probes/sm100_sat.cu` | The uniform datapath: 8 warp-uniform chains of UIMAD + ULOP3, ULOP3, UIADD3 + ULOP3, USHF (funnel shift), UPRMT and ULEA/USHF alone (`uz_`), and beside the honest step (8 IMMA m16n8k32 + duo reading the accumulators: `uh_`), beside it with 64 FFMA (`ug_`) or 32 LOP3 (`ul_`), beside 32 IMMA a fold (`um_`), beside it with 256 FFMA (`ub_`, the `issue` tables), and beside issue-saturated streams (the `sat` tables: the honest step with FFMA or FADD, IADD3 and LOP3 on independent accumulators, `us_`, `uv_`, `uw_`, `ux_`, at 16 to 128 uniform instructions an iteration); each family's table pins every loop's opcodes as the gate saw them, since ptxas keeps the chains uniform only beside an MMA whose accumulators the lanes read (facet_break) |
| `probes/sm100_ldtm.cu`, `ldtm.py` | The TMEM extraction's price: tcgen05.ld.32x32b.x128 (LDTM.x128, 4096 words a warp instruction) at 1 to 16 an iteration beside uniform.cuh's issue-saturated streams (`l`: the honest step and UVec, K ≤ 2; `n`: UVec alone, 8 stream warps of 288 threads) and beside a saturated tcgen05.mma (`m`: `n` with warp 8 issuing UTCIMMA 128x256x32 until the stream ends); `ldtm.py analyze` prices the load as the least slowdown it adds to a stream that issues at least 0.95, less 2%, with every other added instruction (the wait's NOP, the next address's R2UR and LOP3) charged a slot first, and `tables` rewrites the probe table from the gate's JSON; `uni.sh` runs it under `UNI_TABLES=ldtm` |
| `probes/uchain.cuh`, `probes/sm90_async.cu`, `probes/sm100_async.cu` | The same 8 warp-uniform chains (now in `uchain.cuh`, shared with `uniform.cuh`) beside a *saturated async* tensor op: on sm_90a, 4 wgmma s8 m64n256k32 a group, bare (`uwg_`), with an in-loop accumulator fold (`uwgf_`), and with the honest mma.sync step (`uwgh_`); on sm_100a, 4 tcgen05.mma i8 128×256×32 a group from one warp, bare (`utc_`) and with the honest step (`utch_`). K sweeps 0, 32, 64, 128; each loop's full SASS is pinned, so a chain that moves off its datapath refuses (`UNI_TABLES=async`) |
| `probes/sm100_a2u.cu`, `a2u.py` | A2u where `utc_` can't test it: warps 1–31 (8 a SMSP, as `scalar.cu`'s saturating streams) saturate every SMSP's issue with `L_MIX` (OP_UNTIL) while warp 0 keeps 4 tcgen05.mma i8 128×256×32 a group in flight and adds K uniform chain instructions (`a2_tc_i0`–`i96`, `x96`; `a2_mix` the lanes alone). `a2u.py` counts both loops in full from the SASS and the per-block trips and stop cycles: warp instructions an SMSP-cycle above 1 (NOP excluded) falsify A2u, and `uniform_price` is what each added uniform instruction cost the lanes in issue slots (`UNI_TABLES=a2u`) |
| `probes/imad_forms.cuh`, `probes/sm90_imad.cu`, `probes/sm100_imad.cu` | The IMAD forms of duo's ALU work: IMAD.HI.U32 alone, beside duo against IMAD and LOP3, and duo with its LEA.HI as IMAD.HI + IMAD on every word (`f_duoi`) and on one in three (`f_duom`) |
| `uni.sh`, `uniform.py` | Their job for one family and its analysis: UP (the uniform pipe), UNI's lines (UP; an issue slot on the issue rows; no other row, each row's limit recorded), every facet checked at every point, and the IMAD forms; `census --write` puts UNI's lines in the device directory, which census.py's `--write` keeps |
| `probes/s8_mma.cu`, `s8_mma.py`, `s8_mma.sh` | The integer step's semantics on each card, one instruction a tile in eight forms (s8/u8 × s8/u8, with and without `.satfinite`): mma.sync m16n8k32 on both, wgmma m64n8k32 and m64nNk32 for N = 64, 128, 256 (the N an s8 GEMM's warpgroup tile issues: families `wg64`, `wg128`, `wg256`) on sm_90a, tcgen05.mma kind::i8 128×32×32 on sm_100a (idesc bit 3 saturates), each against `verity.ml.circuits.silicon.integer` on sm_120's IMMA probe's families at its tile, with the clamp tiles; `s8_mma.py evidence` writes an instruction's `tc-evidence/v1`, which pinned `sm90.*.s8` and `sm100.*.s8`. `s8_mma.sh build` compiles on a CPU node, and `S8_MMA_BIN` runs that binary on the card |
| `probes/sm90_emp.cu`, `emp.sh`, `emp.py` | The machine model's combined routes on sm_90a, each beside a saturated wgmma: the uniform chain with the honest step and an FFMA filler (`emp_pack`, `emp_pack3` at 3 warps a SMSP), 2:4 sparse wgmma against its dense twin with A in shared memory or registers (`wgsp_`, `wgsprs_`), LDS.128 beside SS wgmma (`emp_smem`), setmaxnreg producer/consumer splits (`emp_reg*`); `emp.sh` builds and gates on a CPU node and times on the H200, `emp.py` prices every timed loop's whole SASS on DS-90's facets (issue, tensor, alu, imad, shfl; a facet above 1 is a rate above the sheet) and reads each probe's event clock beside NVML's |

The tests are `../tests/test_dc.py` (wgmma.cuh current, the twin against a per-word loop at L = 32 and 64 and against
dgemm.cu's host reference, analyze's arithmetic, census.py's facets on a synthetic run, the device directories' form,
pipe.py's SASS profile and q*, uniform.py's tables, fit and writer, emp.py's facet pricing and event clock) and `../tests/test_ffill.py` (ffill's reference
against the integer FP32 pipe, its coins, chains and pinned receipts, its dump checks, the header's host build, the
design that prices it). A probe's rate is
opcode_probe's: warp instructions a SM a clock, timed only after its SASS gate, its poisoned sink and its no-write
control pass (`../opcode_probe/README.md`).

## Running

~~~text
research run --on vy-dc-h200 --project verity --source /workspace --cwd source --custody-ttl 12h \
  --timeout 60m --env GPU_LEASE_WHO=dc -- bash benchmarks/pouw/dc/run.sh sm90
research run --on vy-dc-b200 ... -- bash benchmarks/pouw/dc/run.sh sm100
research run --on vy-dc-b200 ... --timeout 40m ... -- bash benchmarks/pouw/dc/pipe.sh
python3 benchmarks/pouw/dc/analyze.py agree ~/.research/runs/RUN_A/out ~/.research/runs/RUN_B/out
python3 benchmarks/pouw/dc/pipe.py analyze ~/.research/runs/RUN/out --census SM100_RUN_OUT...
python3 benchmarks/pouw/dc/pipe.py agree ~/.research/runs/RUN_A/out ~/.research/runs/RUN_B/out
python3 benchmarks/pouw/dc/census.py sm90 ~/.research/runs/RUN/out... --write benchmarks/pouw/machine_model/data/devices/sm_90
research run --on vy-dc-b200 ... --timeout 30m ... -- bash benchmarks/pouw/dc/uni.sh sm100     # or sm90 on the H200
research run --on vy-dc-b200 ... --env UNI_TABLES=sat ... -- bash benchmarks/pouw/dc/uni.sh sm100   # issue, sat
research run --on vy-dc-b200 ... --env UNI_TABLES=ldtm ... -- bash benchmarks/pouw/dc/uni.sh sm100  # LDTM.x128's price
research run --on vy-dc-b200 ... --env UNI_TABLES=async ... -- bash benchmarks/pouw/dc/uni.sh sm100   # uniform beside a saturated async tcgen05/wgmma
research run --on vy-dc-b200 ... --env UNI_TABLES=a2u ... -- bash benchmarks/pouw/dc/uni.sh sm100     # A2u with every SMSP's issue saturated
python3 benchmarks/pouw/dc/uniform.py census ~/.research/runs/RUN/out ~/.research/runs/SAT_RUN/out... --write
research run --on vy-dc-h200 ... --timeout 40m ... -- bash benchmarks/pouw/dc/ffill.sh     # ffill's conformance and cost
research run --on vy-dc-h200 ... -- bash benchmarks/pouw/dc/s8_mma.sh sm90     # sm100 on vy-dc-b200: the integer step
research run --on vy-nebius-cpu-1 ... -- bash benchmarks/pouw/dc/s8_mma.sh build sm90   # CUDA 13.0 there (through slot.py); then on the card (--send):
research run --on vy-dc-h200 ... --send BIN --send BIN.nvcc --env S8_MMA_BIN=inputs/s8_mma_90a -- bash benchmarks/pouw/dc/s8_mma.sh sm90 wg64 wg128 wg256
~~~

The node needs CUDA 13.0 (`/usr/local/cuda-13.0`, driver ≥ 580). Each run builds and SASS-gates everything before it
touches the GPU (103 gates on sm_90a, 68 on sm_100a), then under one lease: the GEMM gate at three shapes (each dump
rechecked by `dgemm_ref.py`), the bench only if all three are exact, fp8acc (sm100), the probes with nvidia-smi sampled
every 200 ms, and `analyze.py` (`verdict.json`). A run is about 6 minutes on the GPU.

## Results (1× H200, 1× B200; driver 580.173.02, CUDA 13.0.88)

The census is `../machine_model/data/devices/sm_90/` and `sm_100/`; every repeat agrees with its twin within 2% (worst
1.0008). f_max at default clocks: H200 1980 MHz, B200 1965 MHz.

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
  beside the saturated MMA. On load bandwidth alone, honest at L = 32 would balance at 0.914 of ρ_T5 (234.1 words/SM/
  clk; the two-stage LP's ε_tensor ≤ 2.3%), but the pipelined kernel below measures 0.489: at L = 32 the TMEM round
  trip binds, not the load. A tcgen05.mma costs one issue slot (UTCIMMA 1.09 + other 0.98), so lanes beside it keep
  their rates.
- **q*** (the issue facet's tensor coefficient over the words a lane gets): sm_90 0.49, sm_100 0.001 (0.032 with the
  tcgen05.ld.x32 that bring the words to registers; ptxas emits no instruction for tcgen05.wait::ld), against sm_120's
  2.03. The fold binds at L = 32 on both (duo 31.3 and 31.5 words/SM/clk alone: L_bal 131 and 260).

## The pipelined honest GEMM on sm_100 (`pgemm.cu`, 1× B200)

A run and its repeat (152 rate points, all within 2%, worst 1.0064). All 32 gates are exact in both: (NB, LW) in (2,
1), (2, 2), (4, 1), (4, 2), one MMA warp or one a tile, at L = 32 and 64, on 128 × 512 × 256 and 256 × 1024 × 1024, each dump rechecked by `dgemm_ref.py`. Rates are slots/SM/clk (each
point's best configuration; committed words = slots / lm), clock64 a CTA on 148 CTAs:

| L | honest (fold) | TMEM stage | pipeline | MMA self-wait |
|---|---|---|---|---|
| 32 | 29.6 (0.116 ρ_T5) | 125.3 (0.489) | 147.6 (0.576) | 149.9 (0.585) |
| 64 | 58.1 (0.227), 29.1 words | 218.3 (0.853), 109.2 words | 255.8 (0.999) | 180.5 (0.705) |
| 128 | 114.9 (0.449), 28.7 words | 255.9 (1.000) | 256.0 | 234.3 |
| 256 | 230.1 (0.899), 28.8 words | 256.3 (1.000) | 256.1 | 255.7 |

- **The fold sets honest's rate at every spacing**: 29.6, 29.1, 28.7 and 28.8 committed words/SM/clk at L = 32 … 256
  (duo alone 31.5). Its loader issues 3.03 lane instructions a word (ALU 1.94, IMAD 0.98, LDTM.x32 1/32), so at L = 32
  it fills the ALU pipe to 0.90, the IMAD pipe to 0.45 and the SM's issue to 0.69. The MMA's share is 0.001 SMSP-cycles
  a lane-word. q* is 0.0011 (MMA only) and 0.032 (with the loads) at L = 32, and 0.0021 and 0.033 at L = 64.
- **At L = 32 the TMEM round trip binds the tensor.** Each running sum must leave TMEM before the next k-step lands on
  it, so no more than TMEM's 65,536 words are in flight. The tile's round trip (MMA, commit, loaders, hand-back) is at
  least 437 clocks in these kernels, which caps the tensor at 0.585 of ρ_T5 even with no reads (self-wait). One MMA warp
  issuing for every tile serializes the commits (at most 0.36 at L = 32), so a tile needs its own issuer. Four tiles of
  128 columns take as long a round trip as two of 256.
- **ε_tensor at L = 32**, honest's TMEM stage at 0.489 of ρ_T5: T1 gives 51.1%. The tensor table's two-stage LP (mma.sync fills
  the shared pipe's slack) gives 47.7% against an adversary at the probes' bandwidth balance (0.914) and 29.0% against one
  held to the same round trip (0.585). The 2.3% above assumed load bandwidth alone.
- **The commit spacing need not be 64.** At L = 64 the TMEM stage commits 109.2 words/SM/clk, 0.426 of ρ_T5, so T1 gives
  57.4% against 51.1% at L = 32 (even a TMEM stage at ρ_T5 would give 50% at L = 64), and the fold-bound honest kernel
  is no faster (29.1 against 29.6 words/SM/clk). L = 32 stays sm_100's spacing.

Not covered yet: sparse wgmma and tcgen05 (metadata operands), the N sweep for the FP kinds (s8 only), cta_group::2 (a
CTA pair), M4e beyond fp8acc's binade question (kind::f16/tf32/f8f6f4 accumulation against the silicon models in
general), lanes beside wgmma with more than one lane warp a SMSP (`wgU_` reaches only 0.3–0.4 of alone, under the
block-timed points the sm_90 facets rest on), operand streaming (TMA) in `pgemm.cu`, and a pipelined sm_90 GEMM. On sm_90 CUDA 13.0 has no native FP8 or s4 mma.sync: e4m3 m16n8k32 is 2 HMMA.16816.F32 + 4 FADD (timed as
that), s4 m16n8k64 a CALL (not timed).

## The uniform datapath and the IMAD forms (`uni.sh`, 1× H200, 1× B200)

Five runs on each of the H200 and the B200, 115 gates each, all passing:
every uniform chain stays uniform alone and beside the honest step, FFMA, LOP3, the MMA-heavy stream and every
issue-loaded one. Repeats agree within 0.002 SMSP-cycles a uniform instruction.

- **UP: one uniform instruction every 2 SMSP-cycles on both** (0.500 a SMSP-cycle for ULOP3, UIADD3, USHF, UPRMT, UIMAD
  and ULEA alike; 1.957 and 1.961 SMSP-cycles with the 2% margin on sm_100 and sm_90).
- **Beside the honest step the uniform pipe runs nearly free**: up to 0.48 (B200) and 0.46 (H200) uniform instructions
  a SMSP-cycle beside the step and its lanes, and no point has issue (vector plus uniform) above 0.94. No point loads
  a lane pipe or the tensor pipe to its limit beside the chains (the honest step is latency-bound; the MMA-heavy
  stream reaches 0.74 of M on the B200 and 0.61 on the H200, with mma.sync, not tcgen05 or wgmma), so UNI holds UP
  and its issue price, and no other row.
- **A uniform instruction costs 0.92 (B200) and 0.98 (H200) of an issue slot**, its least slowdown beside an
  issue-saturated stream, less 2%. The streams beside the honest step and the `ub_` stream leave 6–28% of issue idle,
  and their slack absorbs the chains (0.10 to 1.08 a uniform instruction). The `sat` streams don't: on the B200 `uv_`,
  `uw_` and `ux_` issue 0.958, 0.976 and 0.981 of their slots without chains, and each uniform instruction added slows
  them 0.94 to 1.03 SMSP-cycles (least 0.943, `uv_64x`). On the H200 only `ux_` (FADD by an immediate: one register
  read) reaches 0.974; FFMA streams stop at 0.92–0.93, where the chains slow them by −0.13 to 0.85. Beside `ux_` each
  uniform instruction costs 1.00 to 1.12 (least 1.003, `ux_128x`). Every count is the loop's full SASS, not the table's
  eight opcodes. The joint issue rows scale the price as they scale the ALU's slot (JI100_0 0.906, JI90_0 0.965).
- **Beside a saturated async tensor op a co-located chain fills idle issue slots; with the port saturated it takes one**.
  The placement is pinned (`probes/sm90_async.cu`, `sm100_async.cu`,
  `UNI_TABLES=async`): ptxas 13.0 keeps a uniform chain on the uniform datapath beside a saturated **tcgen05.mma** (the
  honest kernel issues the whole 128×256 tile from one warp, a uniform context, so even the bare `utc` is uniform beside
  UTCIMMA), but lowers it to the vector lanes beside a saturated **wgmma** unless the honest mma.sync is in the loop
  (`uwg`/`uwgf` land on IMAD/LOP3; only `uwgh`, the already-priced F_M structure, stays on UIMAD/ULOP3). Measured on
  silicon (H200 and B200), the bare chain's K-sweep is flat: the async
  group's cycles a iteration don't grow with the chain (H200 `uwg` 0.0002, B200 `utc` 0.0009 SMSP-cycles a instruction),
  because the chain fills the async op's idle issue slots (the 4 wgmma / 4 tcgen05 hold the SMSP ~1024 / ~512 cycles,
  far more than the K ≤ 128 the sweep adds; a realistic word adds only a handful). That flat slope is idle slots filled,
  not dual-issue: one warp's 4 tensor ops leave its SMSP issuing ≤ 0.27 a cycle. With every SMSP's issue saturated by
  lanes beside the chain and a saturated tcgen05.mma (`probes/sm100_a2u.cu`, `UNI_TABLES=a2u`, `a2u.py`), the B200 shows
  none. The block issues at most one warp instruction an SMSP-cycle at every K (0.9996 at K = 32 to 135; the lanes alone
  1.00004), and each added uniform instruction costs the lanes 0.95 to 1.10 of a slot. The tensor starves as K grows
  (1.00 to 0.34) because the chain takes SMSP 0's slots. So A2u holds beside async on
  the B200, and its joint row JI100_0 0.906 is below the measured ≈ 1, the conservative side. sm_90 is MEASURED near
  saturation, no dual-issue (`dc/probes/sm90_emp.cu` `emp_pack3`: the chain uniform beside 4
  wgmma s8 n128 a group, issue 0.87, each uniform instruction 1.08-1.19 SMSP-cycles). Pricing the H200's
  uniform-beside-async route free anyway moves no headline (`eps`, `credit floor`, `operands free`, q* 1.22,
  `dither-2s-li-l32`): the H200 rises 0.131% → 0.198%. The B200 stays 0.905% either way (its headline binds the issue
  row, which sits above the async joint row). It breaches 1% only if the issue ceiling *itself* is discounted below
  ~0.012 of a slot, an >80× discount that the census's issue pricing precludes and `a2u` measures absent.
- **The IMAD form of duo's LEA.HI is slower on both.** IMAD.HI.U32 costs 4.19 (B200) and 4.18 (H200) SMSP-cycles,
  twice an IMAD, so rotl(u, r) + y′ as IMAD.HI + IMAD costs about 6 cycles of the IMAD pipe against LEA.HI's 2 of
  the ALU's. Duo with 29% (B200) and 33% (H200) of its LEA.HI in that form takes 17% and 21% longer a word than duo.
- **ε** (`machine-model eps --credit floor --operands floor --q 2.03`, Dither-1S / 2S / C, core ε): the B200 at L = 32
  is 12.990 / 0.978 / 1.062% (0.59% of words on the uniform route; the cheat binds the ALU pipe A and issue), at L = 64
  22.589 / 1.342 / 1.429%. Without the uniform route ε_machine is 0, since A holds no tensor term: 12.395 / 0.383 /
  0.467% and 21.995 / 0.748 / 0.835%. The census already charges a uniform instruction its issue slot, so `--uni-issue
  1` changes nothing. The H200 has no uniform route (16.709 / 4.686 / 4.741% at L = 32, 31.128 / 9.758 / 9.824% at
  L = 64): its binding joint issue row JI90_0 charges a uniform instruction 0.965, which closes it. (The B200 has no
  joint ALU row bounding duo: a row fitted to the block-timed tclane_lop3_r128 would, but the exact tclaneU_lop3 keeps
  the tensor at 0.999 and LOP3 at 0.991 beside each other, so census.py writes no such row.)

## Combined routes beside wgmma (`emp.sh`, 1× H200)

`probes/sm90_emp.cu` runs the routes the machine model's red team named beside
a saturated wgmma, and `emp.py` prices each timed loop's whole SASS on DS-90's facets. Three rounds, 69 loops, none
above a facet:

- **A uniform instruction takes an issue slot beside a saturated wgmma.** `emp_pack3_i_r1024_k{0,64,128}` runs 3 warps a
  SMSP, each with 4 wgmma s8 n128 a group, the honest step's 8 IMMA, 1024 FFMA and K uniform instructions an iteration.
  The chain stays uniform, and the loop's SASS adds only uniform opcodes as K grows. It issues 0.870 / 0.869 / 0.871 a
  SMSP-cycle, and each added uniform instruction costs 1.19 (K 0 → 64) and 1.08 (K 64 → 128) SMSP-cycles. So A2u holds
  beside async wgmma on the H200, and JI90_0's 0.965 sits below the measured price, the conservative side.
- **2:4 sparse wgmma reaches A3's sparse line only with A in registers.** `wgsprs_s8` (m64n256k64.sp) runs at its dense
  twin's instruction rate, 4095.6 physical (8191 logical) MACs a SM-clock. With A in shared memory (`wgsp_s8`, `_e4m3`,
  `_f16`) it runs at 0.889 of it: 18 KB of operands an instruction at the dense rate is 144 B a SM-clock against the
  128 B shared port. B is always in shared memory, so the port caps a 2:4 form at 4096/b physical MACs a SM-clock for
  b-byte operands, each type's dense peak.
- **SS wgmma's operand reads share the shared-memory port with LDS.** `emp_smem`: wgmma n128 SS reads 94.3 B a SM-clock
  and LDS.128 beside it 28.7, 123.0 of 128; LDS.128 alone 127.96.
- **setmaxnreg changes no rate.** ptxas drops a release it can't honor (C7507, no USETMAXREG: 232/40 and 208/64 at 384
  threads) and keeps 184/96 (`emp_reg_d96`). All three run tensor 0.9996 beside IMAD 0.789 a SM-clock.
- **The clock under a saturated wgmma is under the 1830 MHz tensor clock.** NVML's sample reads 1905–1965 MHz there, but
  the event clock (`emp.py`'s evMHz: the longest block's cycles over the event time a call) is 1518–1797 MHz on every
  loop at tensor ≥ 0.99. A second, the H200 does at most 1942.7 dense s8 TOPS (1.8% under its
  sheet's 1979), 899.7 f16 TFLOPS and 473.8 tf32. Light loops (scalar, shared memory) read the same on both clocks.
- **b1**: BGMMA.64x256x256 runs at s8 n256's instruction rate (0.03125 a SM-clock, 8× its MACs: k 256 against 32) and BMMA.168256 at
  0.656, as IMMA.16832; DS-90 prices b1 on issue alone.

## B300 (sm_103a)

sm_103 has no tcgen05 kind::i8: ptxas refuses it (`Feature '.kind::i8' not supported on .target 'sm_103a'`; CUDA
13.0), so `dg100` and the i8 probes don't build there (`scalar.cu` does). The honest kernel needs an exact route through
an FP kind, which the protocol states as its own T1 table (an FP design or kind::ti16, sm_107f only):

- **k3 on kind::f8f6f4 (E4M3)**, emu8's Karatsuba (`../emu8/`): balanced
  base-16 limbs, xy = 240hh′ + 16mm′ − 15ll′; three MMAs a k32 step into two accumulators started at binade constants
  (C1 ulp 1, C2 ulp 1/8), and the word grows by 15·bits(acc1) + 2·bits(acc2), two IMADs. k4 (four MMAs, one C0) is the
  fallback. Both are exact only if the tcgen05 FP32 adder keeps C + T exact in that binade. **On the B200 it does**:
  k3 and k4 on all five adversarial classes, 0 bad accumulators and 0 bad k32 words in two runs, so the E4M3 route
  holds for a k32 step.
- **NVFP4** (kind::mxf4nvf4, K 64, twice the FP8 rate): emu8's `accounting.py` `fp4_route` writes U as two E2M1 terms
  under per-block scales; its adder matches the existing silicon model (SIL BLACKWELL_SM100_NVF4/MXF4).

Cost against B200's kind::i8 kernel, at the rates the tensor table gives every 32-deep kind (256 slots a SM a clock):

| | B200 kind::i8 | B300 k3 (E4M3) |
|---|---|---|
| MMAs a k32 step a tile | 1 | 3 |
| committed words a SM a clock, tensor-bound | 256 | 85.3 |
| TMEM words loaded a committed word | 1 | 2 (acc1, acc2) |
| fold ops a word | 3 (duo) | 5 (2 IMAD + duo) |
| honest time at L = 32 (fold-bound on both) | 1 | ≈ 5/3 a SM (scaled by SM count and lane rate) |
| honest time where the tensor pipe binds | 1 | 3 |

So at L = 32, where the tensor table has both parts fold-bound, the missing int8 costs B300 the two recombination IMADs (≈
1.67× the fold's time a SM); where the tensor pipe binds, k3 costs 3× (k4 4×). The B300 measurements: `fp8acc` built
for sm_103a (its guard and device check name sm_100a today) and extended from one k32 step to emu8's K up to 16384,
then sm100_tcgen05.cu's f8f6f4, f16 and tcgen05.ld probes (its i8 forms removed) and `scalar.cu`: about 2 GPU-hours on
a 1× B300.

`../lowbit_dc/` has since run both routes end to end on the B200 in dgemm's kernel and receipts, bit-exact: k4s, k4 with
the accumulator restored to its binade by tcgen05.st after each word leaves, costs one IADD3 a word (no IMAD) and is
1.24–1.45× native int8 in the same pipelined shape, while k3 from +0 (2 FFMA + IADD3) is 1.9–2.2×. Its `fpx` (every
tcgen05 FP8 and FP4 kind to K = 2^17, and the adder's window) builds for sm_103a, and is the B300 exactness run.
