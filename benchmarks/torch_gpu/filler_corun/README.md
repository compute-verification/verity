# filler_corun: the filler's exact int8 GEMM with SHA-512 in separate CTAs on the same SMs (H200, sm_90a)

The filler GEMM must hold the tensor peak while the commitment's SHA-512 runs beside it. A wgmma CTA can't hold enough hash
warps: ptxas sizes each wgmma by the launch's registers a thread, so two m64n256k32 warpgroups leave room for about 11. Here
the hash runs as its own CTAs, one beside each GEMM CTA on every SM.

**Roles.** There are two launches on two streams. One launch would give every CTA the GEMM's shared memory and registers,
and no second CTA would fit on an SM.

- `gemm_1` / `gemm_2` is one persistent launch of 132 CTAs, one an SM, held there by its shared memory.
- `hash_rows` (or `hash_chain`) is a persistent launch of 132 CTAs on a second stream. Its CTAs are sized so that exactly
  one fits beside each GEMM CTA.
- `gate_k` holds the hash stream until every GEMM CTA has started. The hash stops on a flag that the GEMM's last CTA sets.
- Every CTA records its `%smid`. An item counts only with one GEMM CTA and one hash CTA on every SM.

| an SM (65,536 registers, 233,472 B shared) | threads | registers | shared memory |
|---|---|---|---|
| GEMM CTA: 2 MMA warpgroups, thread 0 the TMA producer | 256 | 256 x 160 = 40,960 | 4 stages x 48 KB + 2 KB, +1 KB reserved |
| hash CTA: sx0000, 8 warps | 256 | 256 x 80 = 20,480 | 1 KB reserved |

The registers are four sub-partitions of 16,384, and each warp's come from one of them. Eight GEMM warps put two warps
(10,240 registers) on each, which leaves two hash warps of up to 96 registers on every one. A ninth, producer warp would
put 15,360 on one sub-partition, and no hash CTA of more than six warps could sit beside it.

**The GEMM.** O = A B^T, with s32 O compared against `torch._int_mm` word for word. It is s8_linear's sm_90a kernel
(`../s8_linear/s8_wgmma.cu`) with these changes:

- the 128 x 256 tile's s32 sums are stored as they are;
- there is no producer warpgroup: thread 0 issues each TMA between its own wgmmas, one stream of k-blocks across tiles;
- the kernel is capped at 160 registers a thread with no setmaxnreg;
- it is persistent over `passes` passes, so a 2 s item has no wave tail.

`gemm_2` pairs M-adjacent CTAs, which share each B box by TMA multicast. The operands are u2: int8 bytes uniform in
[0, 3], so every sum is at most 9 K and exact. The default shape is M 8,448, N 8,192, K 8,192. That is 2,112 tiles, 16 a
CTA a pass, and A and B together exceed L2.

**The hash.** `hash_rows` computes SHA-512 of the 1 KiB messages of O (nine compressions each) with
`../sha512_rows/sha512x.cuh`'s `compress<0, 1>` (sx0000). It writes big-endian digest words, and the gate checks them
against hashlib. `hash_chain` is the bar's own form: compressions chained in registers.

**Timing** (`corun.py`). The gates run first in the same process: O poisoned with 0xA5, then the GEMM against
`torch._int_mm`, then every digest against hashlib. After that come 2 s items, interleaved over rounds:

- the GEMM alone;
- the hash alone, in the same 8-warp CTA held to one an SM, plus the 16-warp ruler (two CTAs an SM);
- the co-run;
- `torch._int_mm` back to back.

Clocks come from clock64 over `%globaltimer` in each CTA. The other numbers are:

- MACs an SM-clock against 4,096;
- θ against 989.4 T MAC/s;
- the hash's compressions an SM-clock against the bar (0.01803: 4.355 G/s at 1,830 MHz on 132 SMs);
- κ, a co-run's GEMM rate over the same GEMM's alone;
- NVML's SM clock and clock event reasons every 10 ms, beside the stamps' clock.

`summary.py` holds the arithmetic, and `tests/` holds it on synthetic stamps.

`run.sh` builds and gates the cubin on an H200 node and runs `corun.py` under one GPU lease, from the tree's root:
`CORUN_KS="8192 32768" bash benchmarks/torch_gpu/filler_corun/run.sh [--rounds 3]`, one corun.py process a K.

| file | what |
|---|---|
| `corun.cu` | the kernels: `gemm_1`, `gemm_2`, `hash_rows`, `hash_chain`, `gate_k`, `corun_layout` |
| `corun.py` | the cubin from Python (s8_linear's `wgmma.py` driver table), the gates, the items, `table.md` |
| `summary.py` | rates from stamps and logs |
| `run.sh` | the build on the node, its gate (no spill, local memory or ptxas warning, the IGMMA atom), one lease |
