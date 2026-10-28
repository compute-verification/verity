# u2_gemm: the filler's exact int8 GEMM on streamed 2-bit operands, at the H200's power cap

`O = A W^T`, s32, from A (M x K) and W (N x K) row-major int8 whose bytes the filler draws in a 2-bit range (`u2`: {0, 1,
2, 3}; `s2`: {-2, ..., 1}), the layout `torch._int_mm(A, W.t())` takes. Nothing in the kernel depends on the range but the
sums' size: one s32 accumulator over all of K is exact while |sum| < 2^31 (K < 2^17 for any int8). O equals
`torch._int_mm`'s output bit for bit, which `bench.py` checks before it times an arm and again after.

`u2_gemm.cu` (sm_90a) streams both operands from global memory on every call: TMA boxes of 128 bytes of depth with
SWIZZLE_128B into a ring of 4 or 5 stages, `wgmma m64n256k32 .s32.s8.s8` from shared memory, and an epilogue that stores
each accumulator pair as one 8-byte word. Persistent CTAs raster over the tiles in groups of tile rows. The schedules:

| kernel | tile | who issues the TMA | registers |
|---|---|---|---|
| `u2mm_ws_m<CL>` | 128 x 256 | a producer warpgroup (setmaxnreg 40 / 232) | the whole file |
| `u2mm_ci<NWG>_m<CL>` | 64 NWG x 256, NWG 1 to 3 | consumer thread 0, refilling the stage just released | ci1, ci2: at most 168 a thread, room for a 256-thread `hash_k` CTA beside them; ci3: the whole file |
| `u2mm_ci<NWG>l_m<CL>` | as ci | the same, each refill one stage behind the release | as ci |

`_m2` runs clusters of two M-adjacent CTAs, each loading half of the shared W box and multicasting it to both. M must be a
multiple of 64 NWG CL, N of 256 and K of 128. The default shape, 25,344 x 16,384 x 32,768, gives every arm's CTAs the same
number of tiles on 132 SMs (25,344 = 384 x 66).

`hash_k` runs SHA-512 compressions (`../sha512_rows/sha512x.cuh`'s `sx::compress`) on CTAs of their own, on the GEMM's
SMs. `stamp_k` stamps each SM's `%clock64` and `%globaltimer`, so the SM clock over a span of launches is the difference of
two stamps.

## Running it

`run.sh` builds the cubin with nvcc, holds it to `sass_gate.py`, then runs `bench.py` on one GPU under `gpu-lease`, with
nvidia-smi sampled beside it. `sass_gate.py` refuses:

- a ptxas warning, a spill or local memory;
- any MMA but the s8 atom, or a GEMM kernel without TMA loads;
- ci1, ci2 or ci2l over 168 registers, or `hash_k` over 80.

~~~text
OUT=DIR bash benchmarks/torch_gpu/u2_gemm/run.sh --arms lt,ws_m2,ci3_m2,ci2_m2+h1 --shapes 25344x16384x32768 --ops u2
~~~

Arms:

- `lt`: cuBLASLt, as `torch._int_mm` runs it.
- A kernel's arm (`ws_m1`, `ci3l_m2`, ...): that kernel alone.
- `ARM+h1` or `ARM+h2`: that kernel with 1 or 2 `hash_k` CTAs an SM beside it, launched once its first call's CTAs are
  resident.

Each item runs an arm back to back for `--secs` (an item beside the hash runs for twice as long). It reports the segments
that start at least a second in, where segments are about a quarter second each:

- MACs per SM clock against 4,096, the clocks taken from the stamps;
- the rate against θ = 4,096 x 132 x 1.83 GHz;
- the board's mean watts (NVML's energy counter), its largest instantaneous sample, and pJ a MAC;
- beside the hash, the hash's rate against 4.355 G/s, the GEMM's rate over the segments the hash spans, and the watts.

`bench.py` writes `gates.jsonl`, `items.jsonl` and `summary.md` under `--out`.

`tests/` checks the host side without a GPU: `Params` and the arms against the kernel, every tile's shared memory, the
default shape's balance, the clock and power arithmetic, and the SASS gate.
