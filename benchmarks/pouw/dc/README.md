# dc: the machine model's measurements on Hopper and datacenter Blackwell

The tensor forms of sm_90a (H100, H200) and sm_100a (B200) as gated opcode probes, the scalar pipes that bound the fold,
and the honest Dither GEMM with its duo fold on wgmma and on tcgen05. Everything here builds with CUDA 13.0 on any
machine and passes the SASS gate there; the rates and the GEMM's exactness need the GPU (`run.sh`). The census these
runs write is `../machine_model/data/devices/sm_90/` and `sm_100/`.

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
| `probes/uniform.cuh`, `probes/sm90_uniform.cu`, `probes/sm100_uniform.cu`, `probes/sm90_issue.cu`, `probes/sm100_issue.cu`, `probes/sm90_sat.cu`, `probes/sm100_sat.cu` | The uniform datapath: 8 warp-uniform chains of UIMAD + ULOP3, ULOP3, UIADD3 + ULOP3, USHF (funnel shift), UPRMT and ULEA/USHF alone (`uz_`), and beside the honest step (8 IMMA m16n8k32 + duo reading the accumulators: `uh_`), beside it with 64 FFMA (`ug_`) or 32 LOP3 (`ul_`), beside 32 IMMA a fold (`um_`), beside it with 256 FFMA (`ub_`, the `issue` tables), and beside issue-saturated streams (the `sat` tables: the honest step with FFMA or FADD, IADD3 and LOP3 on independent accumulators, `us_`, `uv_`, `uw_`, `ux_`, at 16 to 128 uniform instructions an iteration); each family's table pins every loop's opcodes as the gate saw them, since ptxas keeps the chains uniform only beside an MMA whose accumulators the lanes read (FACET-BREAK) |
| `probes/sm100_ldtm.cu`, `ldtm.py` | TMEM-EXTRACT's price: tcgen05.ld.32x32b.x128 (LDTM.x128, 4096 words a warp instruction) at 1 to 16 an iteration beside uniform.cuh's issue-saturated streams (`l`: the honest step and UVec, K ≤ 2; `n`: UVec alone, 8 stream warps of 288 threads) and beside a saturated tcgen05.mma (`m`: `n` with warp 8 issuing UTCIMMA 128x256x32 until the stream ends); `ldtm.py analyze` prices the load as the least slowdown it adds to a stream that issues at least 0.95, less 2%, with every other added instruction (the wait's NOP, the next address's R2UR and LOP3) charged a slot first, and `tables` rewrites the probe table from the gate's JSON; `uni.sh` runs it under `UNI_TABLES=ldtm` |
| `probes/uchain.cuh`, `probes/sm90_async.cu`, `probes/sm100_async.cu` | The same 8 warp-uniform chains (now in `uchain.cuh`, shared with `uniform.cuh`) beside a *saturated async* tensor op (RT-ISA round 2 WATCH): on sm_90a, 4 wgmma s8 m64n256k32 a group, bare (`uwg_`), with an in-loop accumulator fold (`uwgf_`), and with the honest mma.sync step (`uwgh_`); on sm_100a, 4 tcgen05.mma i8 128×256×32 a group from one warp, bare (`utc_`) and with the honest step (`utch_`). K sweeps 0, 32, 64, 128; each loop's full SASS is pinned, so a chain that moves off its datapath refuses (`UNI_TABLES=async`) |
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
research run --on vy-dc-h200 --project verity --campaign pouw --source /workspace --cwd source --custody-ttl 12h \
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
every 200 ms, and `analyze.py` (`verdict.json`).

## Not covered

Sparse wgmma and tcgen05 (metadata operands), the N sweep for the FP kinds (s8 only), cta_group::2 (a CTA pair), M4e
beyond fp8acc's binade question (kind::f16/tf32/f8f6f4 accumulation against the silicon models in general), lanes beside
wgmma with more than one lane warp a SMSP, operand streaming (TMA) in `pgemm.cu`, and a pipelined sm_90 GEMM. On sm_90
CUDA 13.0 has no native FP8 or s4 mma.sync: e4m3 m16n8k32 is 2 HMMA.16816.F32 + 4 FADD (timed as that), s4 m16n8k64 a
CALL (not timed).

## B300 (sm_103a)

sm_103 has no tcgen05 kind::i8: ptxas refuses it (`Feature '.kind::i8' not supported on .target 'sm_103a'`; CUDA 13.0),
so `dg100` and the i8 probes don't build there (`scalar.cu` does). The honest kernel there needs an exact route through
an FP kind.
