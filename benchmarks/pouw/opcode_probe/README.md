# opcode_probe: an instruction's rate as a price row, gated first

A probe times one instruction class in one loop on sm_120 and writes its price row in the machine model's census schema
(`../machine_model/`). Every probe's SASS passes `../gpu_kit/sass_gate.py`
before anything is timed, and a probe that doesn't pass writes no row. A probe declared known-invalid must be refused,
by the check it names, or nothing is timed at all.

| File | What |
|---|---|
| `probe.cuh` | `OP_KERNEL`, `OP_BEGIN`/`OP_LOOP`/`OP_END`, `OP_SINK`, the `OpProbe` table entry and `op_main` (`--list`: the table as JSON, host only; `--time`: the sink's poisoned pass and control, then `gpu_kit/bench.cuh`'s interleaved items) |
| `probe.py` | `build` (nvcc; `--arch` sm_120a, or sm_90a / sm_100a for `../dc/`'s probes, the host side seeing it as `PROBE_ARCH` 120, 90 or 100), `gate` (CPU only), `run` (gate, time, write the rows) |
| `rows.py` | The census line (`kind,row,rate,noise,cycles,u,since,until,runs`), machine_model's `u_of`, and MACs a word |
| `run.sh` | The job: build and gate outside the lease, then gate and time under one `gpu-lease` |
| `isa_accept.sh` | CPU only: a one-instruction PTX kernel for each form A7's menu has no verdict for (add with an immediate, f32x2 FMA, f64 MMA, s4 and b1 MMA, `kind::f8f6f4` at each type, tcgen05, redux, 64-bit XOR, add and shift, `mad.wide.u32`), assembled by ptxas for a target; whether ptxas accepts it and the SASS it becomes |
| `probes/lm.cu` | LDSM and STSM by register count, LDS.128, STS.128, and the two known-invalid round-7 forms |
| `probes/issue.cu` | IMMA alone, and k_spec/k_mixu: DADD, DFMA or RED.64 beside IMMA (joint points, no census line) |
| `probes/atom.cu` | ATOMS, ATOMG, REDG and ATOM.E, 32- and 64-bit (ATOME64: generic EXCH.64 and CAS.64 to shared); each global form on 16 SMs (its row) and on every SM (L2red) |
| `probes/cl.cu` | DSMEM atomics and reductions in a 2-CTA cluster, to the own CTA (ATOME32; ATOME64's EXCH.64 and CAS.64, and the known-invalid red.add.u64) and the peer (joint) |
| `probes/mma.cu` | IMMA s8, QMMA E4M3, OMMA MXFP4 and NVFP4, dense and 2:4-sparse, at unit scales, and a `_data` probe a format on the sweep's drawn operands and mix scales (a data-dependent-clock check; joint points, no census line) |
| `probes/a2u.cu` | A2u at saturated issue: uniform.cu's uniform chains beside an issue-bound vector stream (2 IMMA with 64 FFMA, 16 LOP3 and 16 IMAD a loop, or with 16, 40 and 40) at 8 warps a SMSP, where ptxas moves one chain pair of three or four to the uniform datapath, each beside a 2-pair control with the same lane instructions and none uniform, up to 128 uniform instructions a loop, more than the stream's idle slots (joint points, no census line) |
| `probes/mma_types.cu` | `kind::f8f6f4` QMMA at each of E4M3, E5M2, E3M2, E2M3 and E2M1, which `loads.py` prices as one class, and the f64 DMMA, which it doesn't price, each beside the IMMA s8 reference (joint points, no census line) |
| `probes/frontier.cu` | The joint frontiers: IMMA beside IMAD, HFMA2, LOP3 and SHFL.BFLY at 2 to 128 ops per 8 IMMA in one stream, warp-specialized points, each op alone (joint points, no census line) |
| `probes/side.cu` | The side units, each alone (`sa_*`) and beside the honest kernel (`hs_*`, an `until` loop): REDG .32/.64/v4, ATOMG.32, LDGSTS, UBLKRED and UBLKCP (one lane), DSMEM stores and reductions to the peer CTA, TLD and filtered TEX, RED beside UBLKRED; deep roles (more fetches in flight a warp) and 12-warp roles for the latency-bound TLD, TEX and ATOMG |
| `probes/uniform_up.cu` | facet_break's UP search: ULOP3, UIADD3 + ULOP3 and 64-bit (UIADD3.64, UMOV.64) uniform chains inside the honest step and beside 64 FFMA, each loop's uniform count gated exactly (joint points, no census line) |
| `probes/match.cu` | MATCH.ANY (32-bit with the lanes in 32, 4, 2 or 1 classes; .U64 in 32 or 1), alone and beside LOP3 and FFMA, for the coverage table's MATCH rows (`machine_model/data/cover/`; joint points, no census line) |
| `probes/collective.cu`, `collective_verdict.py` | REDUX (.SUM, .MIN, .XOR) and VOTE (.ANY's ballot, .ALL), alone and beside LOP3, and REDUX beside MATCH.ANY, for the coverage table's REDUX and VOTE rows; built for sm_120a or sm_90a (`PROBE_ARCH` picks the chain counts each ptxas lowers to). `collective_verdict.py` reads its rows and `match.cu`'s: whether MATCH.ANY and REDUX share one unit and VOTE spends at most 16 alu lanes (joint points, no census line) |
| `probes/pipes.cu`, `pipes_verdict.py` | The lane opcodes the coverage table places on a pipe no run had placed them on: PLOP3, SGXT and BMSK alone and beside LOP3 (alu), the 32-bit conversion forms (I2F/F2I in u32, s32, s16, u16 and s8, FRND, I2I and I2IP) alone and beside MUFU.EX2 (sfu), the 64-bit ones (F2F.F64/F32, I2F/F2I to and from f64 and 64-bit integers, FRND.F64) alone and beside DFMA (f64) and MUFU.EX2, BAR.RED.OR beside LOP3 (and on a named barrier beside a LOP3 stream in other warps), and IADD.64 and IABS alone and beside LOP3; `pipes_verdict.py` holds each to its pipe's ρ (joint points, no census line) |
| `probes/smem_side.cu` | LDS.128 (broadcast and line) beside a bulk unit that reads shared memory (4 KB UBLKCP, 512 B UBLKRED), for R6r's one read pool (facet_break; joint points, no census line) |
| `probes/honest.cuh`, `probes/power.cu` | The honest kernel (8 IMMA on drawn int8 operands, m16n8k32 or m16n8k16, then the duo fold of their words), and the clock kernels, which `power.sh` (`power_lease.sh` inside the lease) sustains at the node's lock and, given clock permission, at the default clocks |
| `clock_ceiling.sh`, `clock_ceiling_lease.sh`, `clocks.py` | The clock-ceiling run: every clock and power knob the driver offers a prover with root (`-lgc` at and above the max, `-ac`, `-cc`, `-pl`, NVML's GPC and memory VF offsets and locked-clock limits through `clocks.py`, the vboost slider), each followed by power.cu's `--calib` launches (clock64 cycles over the host's steady_clock, a clock that reads no timer on the GPU), then the clock kernels sustained at the settings that clock highest; the node's lock, power limit, vboost and offsets put back on exit; `eps1.py ceiling` reads it |
| `sm89/clock.sh` | power.cu on the L40S: clocks reset (`sudo nvidia-smi -rgc`), every kernel sustained at the power limit as found and at the board's default limit, then `eps1.py clock` |
| `sm89/scalar.cu`, `sm89/run.sh` | The sm_89 (L40S) census: every scalar pipe alone (IADD3, LOP3, SHF, PRMT, IMAD, IDP.4A, FFMA, HFMA2, IMNMX, SHFL, POPC, FLO), LDSM, LDS and STS.128, IMMA beside eight scalar classes (mixes and warp-specialized points), and the honest step at m16n8k32 and m16n8k16 with its MMAs alone; built for sm_89 by its own `run.sh`, census lines `since` sm89 |
| `sm89/census.py` | `machine_model/data/devices/sm_89/` from a `sm89/run.sh` run and a tensor_kinds `TK_ARCH=89` run: each pipe row at its pipe's rate (the loop's instructions on the pipe counted), the tensor pipe and joint issue row of every MMA form, and each MMA + X family's facets with every non-MMA loop instruction counted as X |
| `eps1.py` | The readings: `facets` (the upper hull of frontier.cu's points with the 2% margin; `--prior` adds machine_model's K rows' vertices, `--emit` prints rows and census lines), `side` (words a clock alone and beside honest, MMA kept), `clock`, `ceiling` |

## Write a probe

1. **The kernel.** Set up outside the timed region, then put one iteration of the instructions in `OP_LOOP` and store
   what they computed with `OP_SINK`:

   ~~~cpp
   #include "../probe.cuh"

   OP_KERNEL(shfl_bfly, 1024) {
     unsigned v[8];
   #pragma unroll
     for (int u = 0; u < 8; u++) v[u] = threadIdx.x * 8 + u;
     OP_BEGIN
     OP_LOOP(it) {
   #pragma unroll
       for (int u = 0; u < 8; u++) v[u] = __shfl_xor_sync(~0u, v[u], 1);
     }
     OP_END
     OP_SINK(v[0] ^ v[1] ^ v[2] ^ v[3] ^ v[4] ^ v[5] ^ v[6] ^ v[7])
   }
   ~~~

   `OP_LOOP` is `#pragma unroll 1`: the gate counts one iteration's instructions, so ptxas must not unroll the loop. Give
   every instruction of an iteration its own register chain (8 here, so the rate is the pipe's and not one chain's
   latency) and its own address or input, so ptxas can't merge two. A shared-memory
   address comes from a function `(slot, lane, warp, it) -> byte offset` that the host can call too
   (`__host__ __device__`), so the bank check reads the kernel's own addresses.

   ptxas wraps some uniform ops (UBLKRED) in an ELECT loop that runs once a lane: issue them from one lane, and the gate
   counts that loop as one trip. A SHFL in a warp-divergent role gets a BRA.DIV fallback past the closing clock read, so
   the gate refuses its loop (frontier.cu has no warp-specialized SHFL point).

2. **The table entry.** One `OpProbe` a probe, in a `PROBES` array, and `main` returns `op_main(argc, argv, PROBES)`:

   ~~~cpp
   static const OpProbe PROBES[] = {
       {.name = "shfl_bfly", .kernel = (const void*)shfl_bfly, .tpb = 1024, .kind = "SHFL", .row = "ROW_SHFLM",
        .expect = "SHFL.BFLY=8", .words = 32},
   };
   int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
   ~~~

   - `expect`: each timed loop's exact opcode counts, `OP=N,OP=N` (fnmatch on the dotted SASS opcode), loops separated
     by `;`. The rate is the `counted` pattern's (the first by default).
   - `kind` and `row`: the census class and row (`row` defaults to `ROW_<kind>`). They must be machine_model's own
     names, from its `data/census/` and `data/rows.csv`. With no `kind` the probe is a joint point: `rows.jsonl` only.
   - `words`: the 4-byte words a warp instruction of the counted class moves, for MACs a word.
   - `smem_ops` and `smem`: the loop's shared accesses in order (`LDSM.16.M88*8`) and the offset function.
   - `warps`, `iters`, `dyn_smem`, `cluster`, `blocks` (0: one block on every SM), `fn` (a second probe on one kernel).
   - `until`: a joint point whose second role runs only while the first does (`OP_BEGIN_STOP`, `OP_DONE`, `OP_UNTIL`,
     `OP_TRIPS`, `OP_END_STOP`; `probe.cuh` says how). That loop's rate is its trips over the stop's cycles, and the row
     records `stop_gap`, the share of the kernel the first role's warps had stopped for.
   - The kernel's `OpArgs` carries `g` as two texture objects, `tex1` (1D u32, TLD) and `tex2` (pitch-2D float4,
     linear filtering, TEX).
   - `refuse` (`loops`, `smem`, `sink` or `forbid`): a known-invalid probe, kept as the gate's regression test.

3. **Gate it on the CPU**, then run it on a GPU from a pushed commit:

   ~~~bash
   python probe.py build --out /tmp/p probes/my.cu && python probe.py gate /tmp/p/my
   research run --on MACHINE --project verity --source <tree> --cwd source \
     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/opcode_probe/run.sh benchmarks/pouw/opcode_probe/probes/my.cu
   ~~~

## What a run writes

Under `$OUT`:

- `gate.json`: every probe's verdict, reasons and SASS hash.
- `census.csv`: one line for each (kind, row) a timed probe prices, the fastest of its probes (the class costs what its
  cheapest form does), `noise` 0.02 and `runs` the run id. machine_model refuses two lines for one kind on one row in
  an epoch, so this is the form a new epoch's lines take (`--since EPOCH`).
- `rows.jsonl`: a line a timed probe: class, rate, MACs a word, run id, SASS hash, binary hash, the census line it would
  write and which probe's line was kept (`census_by`), the clock, every loop's rate, and tags (`throttled`, a rep
  spread over 0.5%, fewer distinct SMs than blocks).
- `times.jsonl`: the binaries' own lines (the sink's gate, then every block's cycles for every rep).

The rate is warp instructions an SM a clock: the loop's count of the counted class × `iters` × the warps running it,
over the block's `clock64` cycles from `OP_BEGIN` to `OP_END`; the median block, then the median rep. MACs a word is
`4 cycle_u / (rate MAC_U) / words`, with cycle_u 300.08 and MAC_U 0.15922 (VF3e's MAC price).

## The known-invalid probes

- `bad_ldsm_merged`: round 7's LDSM loop, one address set every iteration and no
  `#pragma unroll 1`. ptxas unrolled it, so the gate refuses it by `loops`: 3 loops between the clock reads, 1 declared.
- `bad_stsm_conflict`: round 7's STSM addresses, a 128-B lane stride, so each 8-row phase lands on 4 banks. The gate
  refuses it by `smem`: 32 of 32 accesses conflict, 8-way.
- `dsm_red_add64_self`: VF3h's ROW_ATOME64 form, `red.shared::cluster.add.u64` at the own rank.
  Every 64-bit ATOM.E but EXCH and CAS carries a QSPC + ATOMS.CAST.SPIN.64 fallback loop, so the gate refuses it by
  `loops`: 8 loops nested in the timed loop. ATOME64 is timed by its EXCH.64 and CAS.64 forms.

The tests are `../tests/test_opcode_probe.py`.
