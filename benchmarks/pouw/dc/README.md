# dc: the epsilon program's measurements on Hopper and datacenter Blackwell

What one H200 (sm_90a) and one B200 (sm_100a) do a SM a clock, as gated opcode probes: the tensor measurements M3
(wgmma) and M4 (tcgen05), the scalar pipes that bound the fold, the uniform datapath, and the honest Dither GEMM with
its duo fold on wgmma and on tcgen05, checked word for word. The ACCEL model's census lines for both cards
(`../../machine_model/data/devices/sm_90/`, `sm_100/`) are written from these runs. Everything here builds with CUDA 13.0
on any machine and passes the SASS gate there; the rates and the GEMM's exactness need the GPU.

## Running

Each job is the command of a `research run` from a pushed commit, on a node with the card, CUDA 13.0
(`/usr/local/cuda-13.0`) and driver ≥ 580. `run.sh` is the runner, the job for one family:

~~~text
research run --on H200 --project verity --source /workspace --cwd source --custody-ttl 12h \
  --timeout 60m --env GPU_LEASE_WHO=dc -- bash benchmarks/pouw/dc/run.sh sm90
research run --on B200 ... -- bash benchmarks/pouw/dc/run.sh sm100
research run --on H200 ... -- bash benchmarks/pouw/dc/run.sh sm90jb   # the stress of the H200's rows: probes only
~~~

H200 and B200 name a node with that card, and CPU_NODE below a CPU node. A run builds and SASS-gates everything before
it touches the GPU (103 gates on sm_90a, 68 on sm_100a; `gate-build.txt`), then under one lease: the GEMM gate at three
shapes, each dump rechecked by `dgemm_ref.py` (`dgemm/gate-<shape>/`), the bench only if all three are exact
(`dgemm/bench.jsonl`), fp8acc (sm100, `fp8acc/`), the probes with nvidia-smi sampled every 200 ms (`probes/`, `scalar/`:
`rows.jsonl`, `gate.json`, `times.jsonl`; `env/smi.csv`), and `analyze.py` (`analysis.json`, `verdict.json`), all under
the run's `out/`. A run is about 6 minutes on the GPU. `DC_BUILD_ONLY=1` builds and gates without a GPU. `sm90jb`
builds and gates `probes/sm90_jbreak.cu` in place of `sm90_wgmma.cu` and times only its `jb` probes (`DC_ONLY`, `^jb`
unless set), with no GEMM gate, no bench and no `analyze.py`: `gate-build.txt`, `sass/`, `env/` and `probes/`.

The other five jobs are each a script of their own, started the same way; a script's header gives its outputs and exit
codes:

~~~text
research run --on B200 ... --timeout 30m ... -- bash benchmarks/pouw/dc/uni.sh sm100     # or sm90 on the H200
research run --on B200 ... --env UNI_TABLES=sat ... -- bash benchmarks/pouw/dc/uni.sh sm100   # issue, sat
research run --on B200 ... --env UNI_TABLES=ldtm ... -- bash benchmarks/pouw/dc/uni.sh sm100  # LDTM.x128's price
research run --on B200 ... --env UNI_TABLES=async ... -- bash benchmarks/pouw/dc/uni.sh sm100   # uniform beside a saturated async tcgen05/wgmma
research run --on B200 ... --env UNI_TABLES=a2u ... -- bash benchmarks/pouw/dc/uni.sh sm100     # A2u with every SMSP's issue saturated
research run --on B200 ... --timeout 40m ... -- bash benchmarks/pouw/dc/pipe.sh
research run --on H200 ... --timeout 40m ... -- bash benchmarks/pouw/dc/ffill.sh     # ffill's conformance and cost
research run --on H200 ... -- bash benchmarks/pouw/dc/s8_mma.sh sm90     # sm100 on a B200: the integer step
research run --on CPU_NODE ... -- bash benchmarks/pouw/dc/s8_mma.sh build sm90   # CUDA 13.0 there (through slot.py); then on the card (--send):
research run --on H200 ... --send BIN --send BIN.nvcc --env S8_MMA_BIN=inputs/s8_mma_90a -- bash benchmarks/pouw/dc/s8_mma.sh sm90 wg64 wg128 wg256
research run --on CPU_NODE ... -- bash benchmarks/pouw/dc/emp.sh build SRC...   # sm_90a probes built and gated; then on the card (--send):
research run --on H200 ... --send bins.tgz ... -- bash benchmarks/pouw/dc/emp.sh time [REPS]
~~~

| Job | Measures | Writes under `out/` |
|---|---|---|
| `uni.sh` | The uniform datapath and the IMAD forms, the tables `UNI_TABLES` picks | `gate-build.txt`, `probes/`, `env/smi.csv`, and `uniform.json` (`ldtm.json`, `a2u.json` for those tables) with its `.txt` |
| `pipe.sh` | The pipelined Dither GEMM on the B200: its gates, then its rate in each arm | `gate-build.txt`, `sass/pgemm.sass`, `gate/<shape>/`, `rate.jsonl`, `pipe.json`, `pipe.txt` |
| `ffill.sh` | The FFMA filler on the H200: conformance bit for bit, and its cost beside the LI-CP-W GEMM | `sass/ffill-check.txt`, `gate-build.txt`, `conformance/`, `probes/` |
| `s8_mma.sh` | The integer step's words on each card against `verity.circuits.silicon.integer` | `check-<family>.json` and `.txt`, `structured-<family>.json`, `meta-<family>.json`, `sass.txt`, `inputs.sha256`, `status.txt` |
| `emp.sh` | The combined routes on the H200, each beside a saturated wgmma | `build`: `bins.tgz`, `sass/`, `gate-build.txt`, `gate-build.json`; `time`: a directory a binary, `env/` |

The analyses read a finished run's `out/`:

~~~text
python3 benchmarks/pouw/dc/analyze.py agree ~/.research/runs/RUN_A/out ~/.research/runs/RUN_B/out
python3 benchmarks/pouw/dc/pipe.py analyze ~/.research/runs/RUN/out --census SM100_RUN_OUT...
python3 benchmarks/pouw/dc/pipe.py agree ~/.research/runs/RUN_A/out ~/.research/runs/RUN_B/out
python3 benchmarks/pouw/dc/census.py sm90 ~/.research/runs/RUN/out... --write benchmarks/machine_model/data/devices/sm_90
python3 benchmarks/pouw/dc/uniform.py census ~/.research/runs/RUN/out ~/.research/runs/SAT_RUN/out... --write
uv run python benchmarks/pouw/dc/emp.py TIME_RUN_OUT BUILD_RUN_OUT/sass [--json OUT]
~~~

## Files


| File | What |
|---|---|
| `probes/sm90_wgmma.cu` | M3: wgmma alone (s8 m64nNk32 for N = 8, 16, 32, 64, 128, 192, 256, SS and RS, u8×s8, E4M3 N 128/256, E5M2, F16, BF16 N 128/256, TF32, b1 AND.POPC; wait_group 0 and 1; zero-data twins), wgmma s8 n256 beside a lane (warp-specialized `wgspec_` and interleaved `wgmix_`: mix, IMAD, LOP3, SHFL, redux, reduce, duo; `wgU_`: the lanes or mma.sync run until the wgmma ends), mma.sync s8 k16/k32, F16, BF16, TF32, b1, E4M3; `wgli_` the honest LI loop beside the wgmma with no checkpoint, with LI-CP-W's redux and with its reduce, and `wgli_fill` the redux loop with ffill's filler |
| `ffill.cuh`, `ffill.cu`, `ffill_ref.py`, `ffill.sh` | ffill, the H200's FFMA filler (`dither-2s-licpw-redux-l32`'s `[filler]`): an IMAD and two `fma.rn.f32` on 3 of a thread's 128 accumulator words a MMA, two chains from the round's challenges. `ffill_ref.py` is the receipt in exact arithmetic (F32FmaV2, rounded once) and checks `ffill.cu`'s dumps (`fma`: specials, subnormals, ties and seeded cases; `chain`: the kernel's chains); `ffill.sh` is the H200 job: build, SASS check (no `.FTZ`, the chain an IMAD and two FFMA a step), conformance, and the probes for `wgli_fill` against `wgli_redux` |
| `probes/sm100_tcgen05.cu` | M4: tcgen05.mma kind::i8 M 64/128 × N 32–256, f8f6f4 E4M3 N 128/256, f16 BF16 (one thread issues, 4 a loop, into TMEM); tcgen05.ld 32x32b.x{1,4,16,32,64,128}, 16x64b.x16, 16x128b.x8, 16x256b.x4 alone; tcgen05.ld, mma.sync and lanes (mix, IMAD, LOP3, duo, redux, reduce) beside the saturated MMA, block-timed and exact (`ldU_`, `mmaU_`, `syncU_`, `tclaneU_`: one role fixed, the other until it ends); `tcliU_` the honest LI loop (TMEM load then LI fold a segment) with no checkpoint, with LI-CP-W's redux reduction, and with its 5-shuffle fallback, so the ratio is what the checkpoint costs the latency-bound loop; `tcliU_floor` the floor program of `dither-2s-licpw-redux-ldtm0-l32` (LI-CP-W redux on words already in registers, ♭'s 544 : 512 : 24 a UTCIMMA, 12 lane warps), so its issue is the most the floor's own mix gets beside the tensor |
| `probes/scalar.cu` | Every family: IADD3, LOP3, IMAD, FFMA, SHF, LEA.HI, SHFL, mix, duo, redux (one `redux.sync.xor.b32`) and reduce (the same by 5 butterfly shuffles + 5 XORs) alone, and the pairs that assign them to pipes (M3's INT32-pipe question, M4d). redux/reduce are LI-CP-W's per-segment checkpoint reduction |
| `probes/lanes.cuh` | The lane ops all three share |
| `probes/wg_common.cuh` | The sm_90a probes' shared pieces (`sm90_wgmma.cu`, `sm90_jbreak.cu`): one shared-memory operand tile, the wgmma forms as types, and the commit group |
| `probes/sm90_jbreak.cu`, `probes/sm90_jbreak.inc`, `gen_jbreak.py` | The stress of the rows the H200 design reads (JI90_0–2, JPA90_0/1, JPI90_0/1, issue, M): facet_break's sm_120 break (few accumulators, one-read lanes by an immediate, 2 warps a SMSP) moved beside wgmma (`jb_`: each warpgroup 4 wgmma a commit group into one of NS accumulator sets, then R lane ops), its exact points (`jbU_`: lane warps run until the wgmma warpgroups end), the lanes alone (`jba_`), and the wgmma kinds the design's menu clause reads (f16, tf32, s8 RS) in honest's shape (`jbk_`, exact `jbkU_`). `sm90_jbreak.inc` is the probe list, one line a probe with its loop's declared opcode counts, which `gen_jbreak.py` writes; `run.sh sm90jb` times it |
| `dc.cuh` | The K-major no-swizzle operand layout and its descriptors (wgmma and tcgen05), the instruction descriptor, mbarrier and tcgen05 wrappers, and duo (floored_gemm's) |
| `gen_wgmma.py`, `wgmma.cuh` | The wgmma wrappers (up to 128 accumulator registers an operand list), generated |
| `dgemm.cu` | The honest Dither GEMM, s8 C = X Wᵀ with every 32-step s32 prefix word folded by duo (3 ops a word): `dg90` (wgmma m64n128k32, accumulators in registers) and `dg100` (tcgen05.mma 128×256×32 into TMEM, tcgen05.ld to registers). Modes `cpuref`, `gate` (0xA5 poisoning, no-write control, a mismatch rerun with LBO/SBO exchanged) and `bench` (fold, no fold, cuBLASLt int8) |
| `dgemm_ref.py` | The receipt's numpy twin: C and every receipt word from X, W and the seed, at meta.json's commit spacing L |
| `pgemm.cu` | The honest Dither GEMM on sm_100a, pipelined: one CTA a SM holds a 128 × 512 block as NB = 2 or 4 TMEM tiles in flight (all 512 columns); MMA warps (one, or one a tile: MI) issue lm tcgen05.mma kind::i8 into a tile and tcgen05.commit to its full barrier; LW loader warpgroups wait on it, tcgen05.ld 32x32b.x32 the rows, hand the tile back on its empty barrier after the last load and fold with duo; L = 32 lm. Arms: 0 pipeline (loaders hand back only), 1 TMEM stage (load, no fold), 2 honest, 3 MMA self-wait (no loaders). Operands are resident (K-periodic inputs), so the rate leaves out operand streaming. Modes `gate` (MODE 2, 0xA5 poison, host reference, dumps for `dgemm_ref.py`) and `rate` |
| `pipe.py`, `pipe.sh` | pgemm's job and its analysis: the build gate (no spill, launchable registers, UTCIMMA and LDTM.x32 present), the rate per arm and spacing against ρ_T5, the honest loader's issue a word from SASS, q*, ε_tensor at L = 32, and `agree` for a repeat |
| `fp8acc.cu` | M4e for E4M3 (sm_100a): emu8's k3 and k4 limb planes through tcgen05.mma kind::f8f6f4 from TMEM preloaded with the binade constants, on five adversarial operand classes; counts accumulators that differ from the exact C + T and k32 words that differ from the s8 dot product. `selftest` runs the same checks with ideal accumulators on the CPU (all exact) |
| `analyze.py` | rows.jsonl → slots a SM a clock, every loop's issue rate, a (issue cycles a tensor instruction takes from the lane beside it), M3's and M4's pass lines, sm_100's TMEM model (honest at L = 32, the L that saturates the tensor, the two-stage LP's ε), the fold's words a clock; `agree A B` checks a repeat |
| `census.py` | Runs → machine_model's census lines, a device directory a family (`../../machine_model/data/devices/sm_90/`, `sm_100/`: census.csv and rows.csv), and the tables: T1 a form, ε_tensor at L = 32, the joint issue rows, q*, L_bal |
| `run.sh` | The job for one family (below); `DC_BUILD_ONLY=1` builds and gates without a GPU |
| `probes/uniform.cuh`, `probes/sm90_uniform.cu`, `probes/sm100_uniform.cu`, `probes/sm90_issue.cu`, `probes/sm100_issue.cu`, `probes/sm90_sat.cu`, `probes/sm100_sat.cu` | The uniform datapath: 8 warp-uniform chains of UIMAD + ULOP3, ULOP3, UIADD3 + ULOP3, USHF (funnel shift), UPRMT and ULEA/USHF alone (`uz_`), and beside the honest step (8 IMMA m16n8k32 + duo reading the accumulators: `uh_`), beside it with 64 FFMA (`ug_`) or 32 LOP3 (`ul_`), beside 32 IMMA a fold (`um_`), beside it with 256 FFMA (`ub_`, the `issue` tables), and beside issue-saturated streams (the `sat` tables: the honest step with FFMA or FADD, IADD3 and LOP3 on independent accumulators, `us_`, `uv_`, `uw_`, `ux_`, at 16 to 128 uniform instructions an iteration); each family's table pins every loop's opcodes as the gate saw them, since ptxas keeps the chains uniform only beside an MMA whose accumulators the lanes read (facet_break) |
| `probes/sm100_ldtm.cu`, `ldtm.py` | The TMEM extraction's price: tcgen05.ld.32x32b.x128 (LDTM.x128, 4096 words a warp instruction) at 1 to 16 an iteration beside uniform.cuh's issue-saturated streams (`l`: the honest step and UVec, K ≤ 2; `n`: UVec alone, 8 stream warps of 288 threads) and beside a saturated tcgen05.mma (`m`: `n` with warp 8 issuing UTCIMMA 128x256x32 until the stream ends); `ldtm.py analyze` prices the load as the least slowdown it adds to a stream that issues at least 0.95, less 2%, with every other added instruction (the wait's NOP, the next address's R2UR and LOP3) charged a slot first, and `tables` rewrites the probe table from the gate's JSON; `uni.sh` runs it under `UNI_TABLES=ldtm` |
| `probes/uchain.cuh`, `probes/sm90_async.cu`, `probes/sm100_async.cu` | The same 8 warp-uniform chains (now in `uchain.cuh`, shared with `uniform.cuh`) beside a *saturated async* tensor op: on sm_90a, 4 wgmma s8 m64n256k32 a group, bare (`uwg_`), with an in-loop accumulator fold (`uwgf_`), and with the honest mma.sync step (`uwgh_`); on sm_100a, 4 tcgen05.mma i8 128×256×32 a group from one warp, bare (`utc_`) and with the honest step (`utch_`). K sweeps 0, 32, 64, 128; each loop's full SASS is pinned, so a chain that moves off its datapath refuses (`UNI_TABLES=async`) |
| `probes/sm100_a2u.cu`, `a2u.py` | A2u where `utc_` can't test it: warps 1–31 (8 a SMSP, as `scalar.cu`'s saturating streams) saturate every SMSP's issue with `L_MIX` (OP_UNTIL) while warp 0 keeps 4 tcgen05.mma i8 128×256×32 a group in flight and adds K uniform chain instructions (`a2_tc_i0`–`i96`, `x96`; `a2_mix` the lanes alone). `a2u.py` counts both loops in full from the SASS and the per-block trips and stop cycles: warp instructions an SMSP-cycle above 1 (NOP excluded) falsify A2u, and `uniform_price` is what each added uniform instruction cost the lanes in issue slots (`UNI_TABLES=a2u`) |
| `probes/imad_forms.cuh`, `probes/sm90_imad.cu`, `probes/sm100_imad.cu` | The IMAD forms of duo's ALU work: IMAD.HI.U32 alone, beside duo against IMAD and LOP3, and duo with its LEA.HI as IMAD.HI + IMAD on every word (`f_duoi`) and on one in three (`f_duom`) |
| `uni.sh`, `uniform.py` | Their job for one family and its analysis: UP (the uniform pipe), UNI's lines (UP; an issue slot on the issue rows; no other row, each row's limit recorded), every facet checked at every point, and the IMAD forms; `census --write` puts UNI's lines in the device directory, which census.py's `--write` keeps |
| `probes/s8_mma.cu`, `s8_mma.py`, `s8_mma.sh` | The integer step's semantics on each card, one instruction a tile in eight forms (s8/u8 × s8/u8, with and without `.satfinite`): mma.sync m16n8k32 on both, wgmma m64n8k32 and m64nNk32 for N = 64, 128, 256 (the N an s8 GEMM's warpgroup tile issues: families `wg64`, `wg128`, `wg256`) on sm_90a, tcgen05.mma kind::i8 128×32×32 on sm_100a (idesc bit 3 saturates), each against `verity.circuits.silicon.integer` on sm_120's IMMA probe's families at its tile, with the clamp tiles; `s8_mma.py evidence` writes an instruction's `tc-evidence/v1`, which pinned `sm90.*.s8` and `sm100.*.s8`. `s8_mma.sh build` compiles on a CPU node, and `S8_MMA_BIN` runs that binary on the card |
| `probes/sm90_emp.cu`, `emp.sh`, `emp.py` | The ACCEL model's combined routes on sm_90a, each beside a saturated wgmma: the uniform chain with the honest step and an FFMA filler (`emp_pack`, `emp_pack3` at 3 warps a SMSP), 2:4 sparse wgmma against its dense twin with A in shared memory or registers (`wgsp_`, `wgsprs_`), LDS.128 beside SS wgmma (`emp_smem`), setmaxnreg producer/consumer splits (`emp_reg*`); `emp.sh` builds and gates on a CPU node and times on the H200, `emp.py` prices every timed loop's whole SASS on DS-90's facets (issue, tensor, alu, imad, shfl; a facet above 1 is a rate above the sheet) and reads each probe's event clock beside NVML's |
| `probes/h32.cuh`, `probes/h32_mma.cuh`, `probes/sm90_h32.cu`, `probes/sm90_h32_tc.cu`, `probes/sm90_h32_sp.cu`, `probes/sm90_h32_mem.cu`, `probes/sm90_h32_pu.cu`, `h32.py` | H32 on the H200: every register writer of MH100's classes, alone and co-issued across pipes. The scalar forms of ffma, int32, fp64 and other, each at 512 an iteration and 64 (`x_`), and mixes in one warp (`m_`); wgmma of every `Shape100` shape at N 8 to 256, FP16 accumulate included (`wa_`), legacy mma.sync, DMMA and mma.sp at 128 a trip (`ms_`, `sp_`), each followed by a scalar round (`mm_`), and wgmma with lane warps beside it (`wu_`, `wu256_`); lookups whose address is the last one's result, and atomics (`d_`, `a_`); the carry writers, add.cc and addc chains of a 64-bit and a 128-bit add, priced under int32 by the general registers a lane writes (`c_`). Each probe's note is its class and the registers one warp instruction writes a lane. `emp.sh` builds, gates and times them (`emp_cuda.sh RELEASE` builds at a named CUDA release's nvcc, from NVIDIA's redistributable archives, for a node without that toolkit or with an older libc; `glibc_compat.h`); `h32.py` prices rows.jsonl in MH100's unit (4,096 over the registers a SM writes a clock): the `Prices100` table by Round 7's rule, H32 and FloorTc, the co-issue minimum and the table as a Lean literal |

The tests are `../tests/test_dc.py` (wgmma.cuh current, the twin against a per-word loop at L = 32 and 64 and against
dgemm.cu's host reference, analyze's arithmetic, census.py's facets on a synthetic run, the device directories' form,
pipe.py's SASS profile and q*, uniform.py's tables, fit and writer, emp.py's facet pricing and event clock, h32.py's
prices and table),
`../tests/test_dc_jbreak.py` (sm90_jbreak.inc current),
`../tests/test_dc_pipe.py` (the build gate's register limit, ε_tensor at L = 32 under the balance and under the round
trip), `../tests/test_dc_uniform.py` (the table writer),
`../tests/test_dc_s8_mma.py` (the SASS reader, and the word check on the sm_120 step's own words),
`../tests/test_dc_device_gate.py` (each job's check of the card it runs on, with stand-ins for nvidia-smi and nvcc) and
`../tests/test_ffill.py` (ffill's reference against the integer FP32 pipe, its `coins`, chains and pinned receipts, its
dump checks, the header's host build, the design that prices it). A probe's rate is opcode_probe's: warp instructions a
SM a clock, timed only after its SASS gate, its poisoned sink and its no-write control pass
(`../opcode_probe/README.md`).

The directory is scripts run by their paths, and no package: nothing outside `benchmarks/pouw/` imports it. Two
modules are loaded by file by another, `analyze.py` by `census.py` and `census.py` by `pipe.py`; each one's `__all__`
lists what its readers and the tests take, and every other name is the script's own. No Python here starts a process
or touches a GPU: the `.sh` jobs do, and the Python reads what a run wrote and writes only where its command says
(`--json`, `--write`, a probe table's header).

## Results

The results of `run.sh`, `pipe.sh` and `uni.sh` are not kept here: the analyses print them from a run's `out/`.
`analyze.py` gives a family's pass line (M3 on sm_90, M4 on sm_100), and its `agree` holds a repeat to 2%; `census.py`
the tensor, load and scalar tables, T1 and ε_tensor at L = 32, the joint rows, q* and L_bal; `pipe.py` the pipelined
GEMM's gates, its rate in each arm and ε_tensor at L = 32; `uniform.py census`, on a card's `uni.sh` runs together, UP,
a uniform instruction's issue price and the IMAD forms; and `a2u.py` A2u's test. A run's own `analysis.json`,
`pipe.txt`, `uniform.txt` and `a2u.txt` are the analysis as the run's commit had it. `uniform.py` reads the default
tables' alone points (`uz_`), so a run of `issue`, `sat`, `async` or `a2u` alone writes an error to its `uniform.txt`.
No analysis reads the `async` tables: their K-sweep is the rates in `probes/rows.jsonl`. Each device directory's
`census.csv` and `rows.csv` name the runs a line was written from, and
`uv run machine-model eps --device sm_100 --credit floor --operands floor --q Q --table` (or `sm_90`) prints each
design's ε on those lines, with a column at q* = Q.

Not covered yet: sparse wgmma and tcgen05 (metadata operands), the N sweep for the FP kinds (s8 only), cta_group::2 (a
CTA pair), M4e beyond fp8acc's binade question (kind::f16/tf32/f8f6f4 accumulation against the silicon models in
general), lanes beside wgmma with more than one lane warp a SMSP (`wgU_`'s lanes stay well under their rate alone,
and under the block-timed points the sm_90 facets rest on), operand streaming (TMA) in `pgemm.cu`, and a pipelined
sm_90 GEMM. On sm_90 CUDA 13.0 has no native FP8 or s4 mma.sync: e4m3 m16n8k32 is 2 HMMA.16816.F32 + 4 FADD (timed as
that), s4 m16n8k64 a CALL (not timed).

## Measured results with no run in the evidence store

These stay here until a rerun's are in the evidence store. `emp.py` prices a timing run on its build run's `sass/`: a
timing run holds none, and the tree names no build run. No B300 has run, and the probe `fpx` is not in the tree.

### Combined routes beside wgmma (`emp.sh`, 1× H200)

`probes/sm90_emp.cu` runs the routes the ACCEL model's red team named beside
a saturated wgmma, and `emp.py` prices each timed loop's whole SASS on DS-90's facets. Three rounds, 69 loops, none
above a facet:

- **A uniform instruction takes an issue slot beside a saturated wgmma.** `emp_pack3_i_r1024_k{0,64,128}` runs 3 warps a
  SMSP, each with 4 wgmma s8 n128 a group, the honest step's 8 IMMA, 1024 FFMA and K uniform instructions an iteration.
  The chain stays uniform, and the loop's SASS adds only uniform opcodes as K grows. It issues 0.870 / 0.869 / 0.871 a
  SMSP-cycle, and each added uniform instruction costs 1.19 (K 0 → 64) and 1.08 (K 64 → 128) SMSP-cycles. So A2u holds
  beside async wgmma on the H200, and JI90_0's 0.965 sits below the measured price, the conservative side. Beside a
  saturated async tensor op a co-located uniform chain fills idle issue slots, and with the SMSP's issue saturated it
  takes one (`UNI_TABLES=async` and `a2u`, and `emp_pack3` on sm_90).
- **2:4 sparse wgmma reaches A3's sparse line only with A in registers.** `wgsprs_s8` (m64n256k64.sp) runs at its dense
  twin's instruction rate, 4095.6 physical (8191 logical) MACs a SM-clock. With A in shared memory (`wgsp_s8`, `_e4m3`,
  `_f16`) it runs at 0.889 of it: 18 KB of operands an instruction at the dense rate is 144 B a SM-clock against the
  128 B shared port. B is always in shared memory, so the port limits a 2:4 form to 4096/b physical MACs a SM-clock for
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

### B300 (sm_103a)

sm_103 has no tcgen05 kind::i8: ptxas refuses it (`Feature '.kind::i8' not supported on .target 'sm_103a'`; CUDA
13.0), so `dg100` and the i8 probes don't build there (`scalar.cu` does). The honest kernel needs an exact route through
an FP kind, which the protocol states as its own T1 table (an FP design or kind::ti16, sm_107f only):

- **k3 on kind::f8f6f4 (E4M3)**, emu8's Karatsuba: balanced
  base-16 limbs, xy = 240hh′ + 16mm′ − 15ll′; three MMAs a k32 step into two accumulators started at binade constants
  (C1 ulp 1, C2 ulp 1/8), and the word grows by 15·bits(acc1) + 2·bits(acc2), two IMADs. k4 (four MMAs, one C0) is the
  fallback. Both are exact only if the tcgen05 FP32 adder keeps C + T exact in that binade. **On the B200 it does**:
  k3 and k4 on all five adversarial classes, 0 bad accumulators and 0 bad k32 words in two runs, so the E4M3 route
  holds for a k32 step.
- **NVFP4** (kind::mxf4nvf4, K 64, twice the FP8 rate): emu8's FP4 route writes U as two E2M1 terms
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

Both routes have since run end to end on the B200 in dgemm's kernel and receipts, bit-exact: k4s, k4 with
the accumulator restored to its binade by tcgen05.st after each word leaves, costs one IADD3 a word (no IMAD) and is
1.24–1.45× native int8 in the same pipelined shape, while k3 from +0 (2 FFMA + IADD3) is 1.9–2.2×. The probe that ran them
(`fpx`: every tcgen05 FP8 and FP4 kind to K = 2^17, and the adder's window) also built for sm_103a.
