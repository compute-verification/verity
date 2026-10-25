# s8_linear: the exact W8A8 linear at the large model's shapes on sm_120

The linear `ScaledMmS8Coordinate_v1{K, BlackwellS8ImmaDot32_v1}` defines (`GEMM_SM120_S8`): the s32 sum of s8 x s8 over K
(exact below 2^17), then `cvt.rn.bf16.f32((cvt.rn.f32.s32(acc) * s_a) * s_w)`, the token scale first, no FTZ, no FMA, every
NaN 0x7FFF. The kernel is `verity_vllm.engine.int8_fused.scaled_mm`, adopted as vLLM's int8 profile serves it: one Triton
GEMM on the s8 tensor cores (mma.sync m16n8k32) with that epilogue fused in that order. Set aside: torch's `_int_mm` plus a
torch epilogue (separate launches; kept as the gate's independent s32 sums), vLLM's CUTLASS int8 GEMM (its epilogue applies
s_w first; on the transposed product it is the Definition's order but needs M a multiple of 16 and writes y^T, so it is a
baseline here), and a new CUDA kernel (no measurement showed one beating int8_fused; `bench.py` measures how near each gets
to the s8 peak).

The shapes are `fixture.MODELS`, compute-accounting's: `mqa16k` (width 16384, 35 layers, MQA, MLP 65536, ~4096 sequences)
and `mha8k` (width 8192, 70 layers, MLP 32768, 1023 sequences), plus `d<width>` for any fused-QKV 4x layer.

| file | what |
|---|---|
| `fixture.py` | the models, the units a layer commits, the gate's inputs and edges, and the CPU twin of the Definition's epilogue (numpy) |
| `run.py` | the gate: every cell of every case against the twin, s32 sums against exact float64 ones, poison, second call, graph replay, PTX and SASS (IMMA, no FFMA, no .FTZ, no local memory), the s32 edge at K = 131040 |
| `configs.py` | the configs the gate and the benchmarks use per (N, K), `tune.py`'s gated winners |
| `tune.py` | the config sweep (kbench's interleaved timing, each config gated before it is timed) |
| `bench.py`, `lt.py`, `roofline.py` | time against M for the exact kernel, cuBLASLt int8 autotuned and vLLM's CUTLASS, against the s8 peak and the measured DRAM bandwidth: the crossing M |
| `commit.py` | a layer's unit outputs as hm96-sha512 row tables (vLLM's GPU row-table committer) against the layer's linears, and a whole step table-major and slot-major |
| `run.sh` | one of the scripts on one leased GPU of a node, from the shipped tree |

`tests/` holds the twin to the Definition's evaluator and the roofline analysis to synthetic curves (CPU). The kernel is
compiled by Triton in the process that runs it, so its PTX and SASS are checked there, by `run.py`, per config.

~~~text
research run --on vy-nebius-2 --project verity --campaign e2e-demo --source . --cwd source --env GPU_LEASE_WHO=LANE \
  [--env S8_SCRIPT=tune.py|bench.py|commit.py] [--env S8_MAX_MIN=30] -- bash benchmarks/torch_gpu/s8_linear/run.sh [ARGS]
~~~
