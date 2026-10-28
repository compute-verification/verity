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
  another kernel fits beside it. The `_r128` and `_r120` configurations launch at 128 or 120 registers a thread instead.
  The build patches the kernel headers' `setmaxnreg` constants into `U2_LOAD_REGS` and `U2_MMA_REGS`: producer 40 or
  24, consumers 168. That leaves 16,384 or 19,456 registers for `hash.cu`'s SHA-512 CTAs.
- **The hash forms.** Each is a CTA size of sha512_rows' `sx::compress<0, 1>` chained per thread: 192 or 224 threads at
  80 registers, or 256 at 64.

## Running it

- `build.sh OUT [NAME ...]` builds the ship on a CPU node: `libu2gemm.so` with every configuration and the hash, plus
  `build.json` with each one's ptxas registers, spill and SASS MMA forms.
- `run.sh [bench.py arguments]` runs `bench.py` from that ship (sent as `u2-ship.tar`) under one GPU lease.
- `bench.py` does the work:
  - it gates every configuration on every shape (a 0xA5-poisoned D, two calls, every CTA stamped);
  - it times 2 s items interleaved with cuBLASLt (`torch._int_mm`), each configuration alone and the beside items with
    the hash;
  - it reads nvidia-smi's board power at 100 ms;
  - it writes `items.jsonl`, `gates.json` and `summary.md`. The table reports MACs per SM clock over 4,096, θ over
    989.4 T MAC/s, watts, pJ per MAC, the hash over the 4.355 G/s bar, and the gaps to 0.985 and 0.97.
