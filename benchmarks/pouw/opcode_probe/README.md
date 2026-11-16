# opcode_probe: an instruction's rate as a price row, gated first

How fast can an sm_120 GPU run one kind of instruction? PoUW's ε for a card is computed from the ACCEL model's census
of that card ([`machine_model/`](../../machine_model/README.md)), which prices each class of instruction. A probe here
times one instruction class in one loop and writes its price row in that census's form.

A rate is only as good as the loop it timed, so nothing is timed before the SASS gate
(`verity/protocols/resource_accounting/compute/pouw/prover/kernels/gpu_kit/sass_gate.py`) has read the probe's SASS
and found exactly the instructions the probe declares. A probe that does not pass writes no row. The probes this
directory keeps as known-invalid must each be refused by the check it names, or nothing is timed at all.

The probes in `../dc/`, `../facet_break/` and `../sass_census/` are written and run with this directory's harness.

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

`probe.py --help` lists the harness's commands. [`DESIGN.md`](DESIGN.md) says what each probe source measures,
what a run writes and how a rate is computed, why each known-invalid probe is refused, and what other directories read
from here.
