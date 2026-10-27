# HIGHUTIL: a GPU at full batch on useful matmuls, certified

Llama-3-8B's four linear layers (QKV 6144 x 4096, O 4096 x 4096, gate-and-up 28672 x 4096, down 4096 x 14336) at
M = 16,384, s8 x s8 with int32 accumulation, called back to back on one RTX PRO 6000 (sm_120). Each call goes through
Dither's no-fold receipt (CMP-172; v31.py's `Fill`, its shapes and commit phase per `verity/core/protocols/pouw/accel/PROTOCOL.md` v3.2) on V31-GPU's prover, with a live
challenger session a period. The report is the certified share: credited MACs over θ's INT8 tensor peak times the wall
time, and γ against plain cuBLASLt on the same calls, interleaved.

- `period.py`: the workload, the registration, the coins, the picks, the prover's answers and the verifier (its
  docstring has the protocol and the deviations from v3.2). `period.py run CFG ref OUT` runs it all on the CPU at the
  small shapes `prep --small` writes, through v31.py's reference prover.
- The period's GPU driver (`period.GpuDriver`) runs each layer's calls through V31-GPU's binding (`v31lib.Prover` over
  `libv31.so`, one Prover a layer, all driven by one thread since the binding's contexts share their streams, each
  call's commit beside the next call's compute), with both operands of every transformer block on the GPU. L is 16,384,
  so every call has one segment (S = 1), and under `V31_LTW=1` (run.sh's default) a call at S = 1 takes cuBLASLt's int32
  X W'^T in place of v31's GEMM. Dither's argument covers a segment of at most 8 L' (`T_MAX`), and v31 takes L' <= 1024,
  so a call takes the least L' from 512 to `--lp-max` that covers its K (`period.lp_of`), and down (K = 14,336, which
  would need L' >= 1,792) runs as K-slices (`period.calls`) whose outputs sum to its own: four of 3,584 at L' 512 (t = 7,
  run.sh's default), or two of 7,168 at L' 1,024 (`LP_MAX=1024`, t = 7), which measured 1.7% more GPU time a step.
  QKV, O and gate-and-up are whole at L' = 512 (t = 8). The plain arm is cuBLASLt (`torch._int_mm`) on the same linears, each
  whole, so the slicing's cost counts against the protocol.
- `hu_lib.cu` builds `libv31.so` from `v31.cu` with two functions more, so the host never waits on the device inside a
  block: `hu_point_b` points a call's weights at the block's device tensor (no `v31_set_b` copy and wait), and
  `hu_levels_async` copies each call's tree to page-locked memory behind its commit (no `v31_root` and `v31_levels`
  wait), for the roots and the openings after the check coin.
- `run.sh`: the build, the config and the periods under a GPU lease (`research run`; its header has the command), and
  with `NSYS=1` each arm's kernel timeline (`timeline.py`: GEMM share, busy share, idle gaps), with `NCU=1` each kernel's
  tensor-pipe active share and INT8 MACs from Nsight Compute (`pipes.py`), and `terms.py`: where θ x wall goes.
- `v31.cu`, `wg_proto.cuh`, `v31.py` and `v31lib.py` are vendored unmodified from kernels' no-fold fast path on the H200
  (#5202, branch `cursor/kernels-nofold-fast-0d88` at b7475b2ff8, its accel_v31 directory, which carries #5118's sm_90a
  port of compute-research's V31-GPU: leaf pos of a Prover's c-th call salted prf_salt(Ks, c leaves + pos); its seeds are
  main's `verity.core.protocols.pouw.circuit.dither`), sha256 ea70c51549d6..., 2a830ea0dc44..., ea4119e95b7c... and
  1735dcd9fb4b...; `run.sh` records the hashes. One copy builds for both cards: sm_120a, and sm_90a with `-DV31_WG=1`
  (run.sh's header), where a filler call's prefix-sum segments run on the warpgroup GEMM (`V31_WGW`, default 1) and
  `V31_SCHED 2`, `V31_LMINB 4` and `V31_SAHEAD 1` are #5202's timed knobs (its RP 4, `V31_HCHUNK` and `V31_GEMM` are the
  served call's, which the period doesn't make). Fixes belong upstream there, then a fresh copy here.
