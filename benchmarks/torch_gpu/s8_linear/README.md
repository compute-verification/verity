# s8_linear: the exact W8A8 linear at the large model's shapes, on the H200 (sm_90a) and sm_120

The linear `ScaledMmS8Coordinate_v1{K, DOT}` defines, DOT the card's s8 step (`GEMM_SM90_S8` on the H100 and H200,
`GEMM_SM120_S8` on the RTX PRO 6000): the s32 sum of s8 x s8 over K (exact below 2^17), then
`cvt.rn.bf16.f32((cvt.rn.f32.s32(acc) * s_a) * s_w)`, the token scale first, no FTZ, no FMA, every NaN 0x7FFF.

Two exact kernels, each with that epilogue fused in that order (run.py's `--kernel`):

- **sm_90a: `s8_wgmma.cu`** (`wgmma.py` launches it on torch tensors; `wgmma.linear(x_s8, sx_f32, w_s8, sw_f32)` is
  the signature of KS8-SIMPLE's simple path, HASH-11). Pearl-C's sm_90a plain GEMM (`benchmarks/pouw/pearl_c_sm90`) on the s8
  atom: persistent and warp-specialized (one TMA producer warpgroup, one or two consumer warpgroups), `wgmma
  m64nNk32 .s32.s8.s8` into one s32 accumulator over all of K, the epilogue after the main loop. Its grid is a set of SMs
  (`Cfg` cN: at most N CTAs), the producer its own warpgroup role, and the main loop issues no integer work but counters,
  for KP2-DEC's decode warps beside it. Adopted over int8_fused on this card because Triton reached about 47% of the
  H200's 1,942.7 s8 TOPS (r20261019-183823-8641) and mma.sync runs at 0.656 of the wgmma rate on sm_90.
- **sm_120: `verity_vllm.engine.int8_fused.scaled_mm`**, as vLLM's int8 profile serves it: one Triton GEMM on mma.sync
  m16n8k32 (about 93% of the RTX's 807.5 TOPS at M = 8192, r20261019-183925-ef93); on sm_90 a second exact kernel beside
  wgmma.

Set aside: torch's `_int_mm` plus a torch epilogue (separate launches; kept as the gate's independent s32 sums) and vLLM's
CUTLASS int8 GEMM (its epilogue applies s_w first; on the transposed product it is the Definition's order but needs M a
multiple of 16 and writes y^T), both baselines in `bench.py`.

The shapes are `fixture.MODELS`, compute-accounting's: `mqa16k52` (the H200's: width 16384, 52 layers, MQA, MLP 65536,
context 32, ~20,480 sequences), `mqa16k` and `mha8k` (the RTX's), plus `d<width>` for any fused-QKV 4x layer. `realw.py`
gates the kernel on the demo's seeded weights at the H200 shape (`verity_torch_loop.weights`).

| file | what |
|---|---|
| `s8_wgmma.cu`, `s8_acc.h` | the sm_90a kernel |
| `build.sh`, `sass_gate.py` | its build on a CPU node (nvcc, -fmad=false) and the SASS gate it must pass: the s8 atom alone, no spill, the epilogue's I2F, FMUL, F2FP.BF16, no FFMA, no .FTZ but integer division's; the ship `run.sh` takes |
| `wgmma.py` | the cubin from Python: tensor maps, configs, launches, the cubin's identity held to its gate and MANIFEST |
| `fixture.py` | the models, the units a layer commits, the gate's inputs and edges, and the CPU twin of the Definition's epilogue (numpy) |
| `run.py` | the gate: every cell of every case against the twin, s32 sums against exact float64 ones, poison, second call, graph replay, the kernel's checks, the s32 edge at K = 131040 |
| `realw.py` | the same gate on one layer's seeded weights and the head, every row |
| `configs.py` | the configs per kernel, arch and (N, K), `tune.py`'s gated winners |
| `tune.py` | the config sweep (kbench's interleaved timing, each config gated before it is timed) |
| `bench.py`, `lt.py`, `roofline.py` | time against M for the exact kernels, cuBLASLt int8 autotuned and vLLM's CUTLASS, against the s8 peak (measured and data-sheet readings) and the measured DRAM bandwidth |
| `commit.py` | a layer's unit outputs as hm96-sha512 row tables against the layer's linears (now KCOMMIT's to measure) |
| `run.sh` | steps of these scripts on one leased GPU of vy-dc-h200, a lease each, from build.sh's ship |

`tests/` holds the twin to the Definition's evaluator for both steps, wgmma.py's host side and the SASS gate to the kernel's
source and small listings, and the roofline analysis to synthetic curves (CPU).

~~~text
research run --on auto --project verity --campaign e2e-demo --source . --cwd source -- \
  bash -lc 'uv sync -q --all-packages --extra torch-cpu && bash benchmarks/torch_gpu/s8_linear/build.sh "$OUT/ship"'
tar -C BUILD_RUN/out/ship -cf /tmp/s8-ship.tar .
research run --on vy-dc-h200 --project verity --campaign e2e-demo --source . --cwd source --timeout 90m \
  --env GPU_LEASE_WHO=LANE --send /tmp/s8-ship.tar -- bash benchmarks/torch_gpu/s8_linear/run.sh "run.py" "tune.py" \
  "bench.py --configs @OUT@/2-tune/best.json"
~~~
