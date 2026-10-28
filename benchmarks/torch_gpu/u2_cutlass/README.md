# u2_cutlass: the exact s8 GEMM on 2-bit operands, CUTLASS's sm90 kernels, sustained on the H200

D = A·Bᵀ in s32: A is M×K and B is N×K, both int8 row-major (K-major, as wgmma s8 needs), with values in a 2-bit range
(default the unsigned {0 .. 3}). D equals `torch._int_mm(A, B.t())` bit for bit. The kernels are CUTLASS 3.x's sm90
warp-specialized persistent GEMMs, cooperative or ping-pong. CUTLASS is fetched at a pinned commit in the build and never
vendored.

`u2_gemm.cu` launches each one from its own entry, which stamps every consumer warpgroup's `clock64` and `%globaltimer`.
A timed call therefore carries the SM clock it ran at, against wgmma's 4,096 s8 MACs per SM clock.

## The configurations

`build.sh`'s `CONFIGS` table lists them. Each has a tile, a cluster (the operand shared across it is multicast by TMA),
the mainloop's stages and the epilogue: shared memory and TMA, or `NoSmemWarpSpecialized`'s stores from registers.

- **Register budget.** Stock CUTLASS launches its 384 threads at 168 registers, the whole register file, so no CTA of
  another kernel fits beside it. The build patches the kernel headers' `setmaxnreg` constants into `U2_LOAD_REGS` and
  `U2_MMA_REGS`. ptxas compiles the whole kernel within the launch's registers, and a consumer's wgmma m64n256 holds
  128 accumulators, so a launch needs at least 160 registers a thread.
- **Beside, in a CTA of its own** (`_r160`): the GEMM is launched at 160 registers (producer 40, consumers 168). That
  leaves 4,096 registers for one `hash.cu` CTA per SM: form 0 is one warp at 128 registers, and form 1 is two warps at
  64. The 4 stages leave the hash CTA its shared memory.
- **Fused** (`_hk`): the cooperative kernel's producer warpgroup has three warps that a static scheduler and a void C
  leave idle. The build adds a hook after the producer's warp roles, and these warps chain the same compressions at 120
  registers while the consumers run at 192 (120 + 2 × 192 = 3 × 168). They stop when a consumer warpgroup of their CTA
  ends its tiles. The hook is off unless `bench.py` switches it on, so the same kernel is also timed alone.
- **The hash** is sha512_rows' `sx::compress<0, 1>`, chained per thread. Forms 2 to 4 (128, 256 and 512 threads) are
  the hash alone.

## Running it

- `build.sh OUT [NAME ...]` builds the ship on a CPU node: `libu2gemm.so` with every configuration and the hash, plus
  `build.json` with each one's ptxas registers, spill and SASS MMA forms.
- `run.sh [bench.py arguments]` runs `bench.py` from that ship (sent as `u2-ship.tar`) under one GPU lease.
- `bench.py` does the work:
  - it gates every configuration on every shape (a 0xA5-poisoned D, two calls, every CTA stamped);
  - it times 2 s items interleaved with cuBLASLt (`torch._int_mm`): each configuration alone, the fused items, the hash
    alone, and the beside items;
  - it reads nvidia-smi's board power at 100 ms;
  - it writes `items.jsonl`, `gates.json` and `summary.md`. The table reports MACs per SM clock over 4,096, θ over
    989.4 T MAC/s, watts, pJ per MAC, the hash over the 4.355 G/s bar, and the gaps to 0.985 and 0.97.
