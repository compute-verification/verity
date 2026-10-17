# opcode_probe: an instruction's rate as a price row, gated first

A probe times one instruction class in one loop on sm_120 and writes its price row in the machine model's census schema
(RS-W′'s `machine_model`, branch `cursor/machine-model-ea24`). Every probe's SASS passes `../gpu_kit/sass_gate.py`
before anything is timed, and a probe that doesn't pass writes no row. A probe declared known-invalid must be refused,
by the check it names, or nothing is timed at all.

| File | What |
|---|---|
| `probe.cuh` | `OP_KERNEL`, `OP_BEGIN`/`OP_LOOP`/`OP_END`, `OP_SINK`, the `OpProbe` table entry and `op_main` (`--list`: the table as JSON, host only; `--time`: the sink's poisoned pass and control, then `gpu_kit/bench.cuh`'s interleaved items) |
| `probe.py` | `build` (nvcc, sm_120a), `gate` (CPU only), `run` (gate, time, write the rows) |
| `rows.py` | The census line (`kind,row,rate,noise,cycles,u,since,until,runs`), machine_model's `u_of`, and MACs a word |
| `run.sh` | The job: build and gate outside the lease, then gate and time under one `gpu-lease` |
| `probes/lm.cu` | LDSM and STSM by register count, LDS.128, STS.128, and the two known-invalid round-7 forms |
| `probes/issue.cu` | IMMA alone, and k_spec/k_mixu: DADD, DFMA or RED.64 beside IMMA (joint points, no census line) |
| `probes/atom.cu` | ATOMS, ATOMG, REDG and ATOM.E, 32- and 64-bit; each global form on 16 SMs (its row) and on every SM (L2red) |
| `probes/cl.cu` | DSMEM atomics and reductions in a 2-CTA cluster, to the own CTA (ATOME32) and the peer (joint) |
| `probes/mma.cu` | IMMA s8, QMMA E4M3, OMMA MXFP4 and NVFP4, dense and 2:4-sparse, at unit scales (joint points, no census line; `../fp_exact/` reads them) |

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
     names, from its `data/census.csv` and `data/rows.csv`. With no `kind` the probe is a joint point: `rows.jsonl` only.
   - `words`: the 4-byte words a warp instruction of the counted class moves, for MACs a word.
   - `smem_ops` and `smem`: the loop's shared accesses in order (`LDSM.16.M88*8`) and the offset function.
   - `warps`, `iters`, `dyn_smem`, `cluster`, `blocks` (0: one block on every SM), `fn` (a second probe on one kernel).
   - `refuse` (`loops`, `smem`, `sink` or `forbid`): a known-invalid probe, kept as the gate's regression test.

3. **Gate it on the CPU**, then run it on a GPU from a pushed commit:

   ~~~bash
   python probe.py build --out /tmp/p probes/my.cu && python probe.py gate /tmp/p/my
   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd source \
     --env GPU_LEASE_WHO=<lane> -- bash benchmarks/pouw/opcode_probe/run.sh benchmarks/pouw/opcode_probe/probes/my.cu
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

- `bad_ldsm_merged`: round 7's LDSM loop (r20261016-112703-0297), one address set every iteration and no
  `#pragma unroll 1`. ptxas unrolled it, so the gate refuses it by `loops`: 3 loops between the clock reads, 1 declared.
- `bad_stsm_conflict`: round 7's STSM addresses, a 128-B lane stride, so each 8-row phase lands on 4 banks. The gate
  refuses it by `smem`: 32 of 32 accesses conflict, 8-way.

The tests are `../tests/test_opcode_probe.py`.
