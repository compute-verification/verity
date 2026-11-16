# s8_linear: the exact W8A8 linear

This is a W8A8 linear layer written so that a GPU computes exactly what its Definition says, at the H200 example's
shapes, and timed against cuBLASLt's int8 GEMM and vLLM's CUTLASS one. The linear is the one
`ScaledMmS8Coordinate_v1{K, DOT}` defines, DOT the card's s8 step (`GEMM_SM90_S8` on the H100 and H200,
`GEMM_SM120_S8` on the RTX PRO 6000): the s32 sum of s8 x s8 over K (exact below 2^17), then
`cvt.rn.bf16.f32((cvt.rn.f32.s32(acc) * s_a) * s_w)`, the token scale first, no FTZ, no FMA, every NaN 0x7FFF.

It comes in three backends, each with that epilogue in that order, all held to one gate (`run.py`): a wgmma kernel of
its own for sm_90a, the vLLM adapter's Triton GEMM on sm_120, and `torch._int_mm` with the epilogue as its own launch
on any card. The gate holds every output word of every case to a CPU twin of the Definition, with the output buffers
poisoned first.

## Try it

Build the sm_90a kernel on a CPU node, then run the gate, the config sweep and the timing on an H200:

~~~text
# from the tree's root, on a CPU node with CUDA 13's nvcc
bash benchmarks/kernels/s8_linear/build.sh /tmp/s8-linear-ship
tar -C /tmp/s8-linear-ship -cf /tmp/s8-ship.tar .
# on an H200 with CUDA torch, vLLM and `gpu-lease`
OUT=DIR S8_SHIP=/tmp/s8-ship.tar bash benchmarks/kernels/s8_linear/run.sh "run.py" "tune.py" \
  "bench.py --configs @OUT@/2-tune/best.json"
~~~

`build.sh` refuses a kernel that fails the SASS gate. `run.sh` runs each step under a lease of its own and writes it
into `$OUT/<i>-<script>`; its header says what else a step may be. The CPU tests are a suite,
`benchmarks/kernels/s8_linear`.

The shapes are `fixture.MODELS`: `mqa16k52` (the H200's), `mqa16k` and `mha8k` (the RTX's), plus `d<width>` for any
fused-QKV 4x layer. [`DESIGN.md`](DESIGN.md) says how each backend computes the linear and what each file does.
