# s8_linear: the exact W8A8 linear at the large model's shapes, on the H200 (sm_90a) and sm_120

The linear `ScaledMmS8Coordinate_v1{K, DOT}` defines, DOT the card's s8 step (`GEMM_SM90_S8` on the H100 and H200,
`GEMM_SM120_S8` on the RTX PRO 6000): the s32 sum of s8 x s8 over K (exact below 2^17), then
`cvt.rn.bf16.f32((cvt.rn.f32.s32(acc) * s_a) * s_w)`, the token scale first, no FTZ, no FMA, every NaN 0x7FFF.

Three exact backends, each with that epilogue in that order (run.py's `--kernel`), all held to one gate (`run.py`):

- **sm_90a: `s8_wgmma.cu`** (`wgmma.py` launches it on torch tensors; `wgmma.linear(x_s8, sx_f32, w_s8, sw_f32)` is
  `int_mm.linear`'s signature). Pearl-C's sm_90a plain GEMM (`benchmarks/pouw/pearl_c_sm90`) on the s8
  atom: persistent and warp-specialized (one TMA producer warpgroup, one or two consumer warpgroups), `wgmma
  m64nNk32 .s32.s8.s8` (N = 64, 128 or 256, the tile's columns) into one s32 accumulator over all of K, the epilogue
  after the main loop. Its grid is a set of SMs (`Cfg` cN: at most N CTAs), the producer its own warpgroup role, and the
  main loop issues no integer instructions but counters, for decode warps that share its SMs. Two variants change only where
  the operands and sums travel: `:m2` runs clusters of two CTAs on M-adjacent tiles that share each W box by TMA
  multicast (a third less L2 traffic at 128 x 256), and `:kN` splits K where the tiles leave SMs idle (the narrow
  linears, kv and head, at decode), each split's s32 sums added into a workspace (integer addition: exact in any order)
  and the epilogue run once on the total. On sm_90 it is preferred to int8_fused, whose mma.sync runs below wgmma's rate.
- **sm_120: `verity_vllm.engine.int8_fused.scaled_mm`**, as vLLM's int8 profile serves it: one Triton GEMM on mma.sync
  m16n8k32; on sm_90 a second exact kernel beside wgmma.
- **any card: `int_mm.py`** (`--kernel int_mm`): `torch._int_mm`, cuBLASLt's s32 GEMM, then the epilogue as its own
  launch, `"fused"` (one Triton kernel of the Definition's PTX, held to `int_mm.ptx_audit`) or `"eager"` (four torch
  kernels). `int_mm.linear` is the plain GEMM that `p2_stage` and `j10r_sm90` call.

Set aside: vLLM's CUTLASS int8 GEMM (its epilogue applies s_w first; on the transposed product it is the Definition's
order but needs M a multiple of 16 and writes y^T), a baseline in `bench.py`.

The shapes are `fixture.MODELS`: `mqa16k52` (the H200's: width 16384, 52 layers, MQA, MLP 65536,
context 32, ~20,480 sequences), `mqa16k` and `mha8k` (the RTX's), plus `d<width>` for any fused-QKV 4x layer. `realw.py`
gates the kernel on the demo's seeded weights at the H200 shape (`e2e_h200.served_weights`).

| file | what |
|---|---|
| `s8_wgmma.cu`, `s8_acc.h` | the sm_90a kernel |
| `build.sh`, `sass_gate.py` | its build on a CPU node (nvcc, -fmad=false) and the SASS gate it must pass: the s8 atom alone, no spill, the epilogue's I2F, FMUL, F2FP.BF16, no FFMA, no .FTZ but integer division's; the ship `run.sh` takes |
| `int_mm.py` | the int_mm backend: `torch._int_mm`'s s32 sums (also the gate's), the fused and eager epilogues, the PTX audit |
| `wgmma.py` | the cubin from Python: tensor maps, configs, launches, the cubin's identity held to its gate and MANIFEST |
| `fixture.py` | the models, the tables a layer commits, the gate's inputs and edges, and the CPU twin of the Definition's epilogue (numpy) |
| `run.py` | the gate: every cell of every case against the twin, s32 sums against exact float64 ones, poison, second call, graph replay, the kernel's checks, the s32 edge at K = 131040 |
| `realw.py` | the same gate on one layer's seeded weights and the head, every row |
| `configs.py` | the configs per kernel, arch and (N, K), `tune.py`'s gated winners |
| `tune.py` | the config sweep (kbench's interleaved timing, each config gated before it is timed) |
| `bench.py`, `lt.py`, `roofline.py` | time against M for the exact kernels, cuBLASLt int8 autotuned and vLLM's CUTLASS, against the s8 peak (measured and data-sheet readings) and the measured DRAM bandwidth |
| `commit.py` | a layer's op outputs as hm96-sha512 row tables, timed against the layer's linears |
| `run.sh` | steps of these scripts on one leased GPU of an H200, a lease each, from build.sh's ship |

`tests/` holds the twin to the Definition's evaluator for both steps, `int_mm.linear` on the CPU and its PTX audit to the
twin and small listings, wgmma.py's host side and the SASS gate to the kernel's source and small listings, and the
roofline analysis to synthetic curves (CPU).

~~~text
# from the tree's root, on a CPU node with CUDA 13's nvcc
uv sync --all-packages --extra torch-cpu && bash benchmarks/torch_gpu/s8_linear/build.sh /tmp/s8-linear-ship
tar -C /tmp/s8-linear-ship -cf /tmp/s8-ship.tar .
# on an H200 with CUDA torch, vLLM and `gpu-lease`
OUT=DIR S8_SHIP=/tmp/s8-ship.tar bash benchmarks/torch_gpu/s8_linear/run.sh "run.py" "tune.py" \
  "bench.py --configs @OUT@/2-tune/best.json"
~~~
