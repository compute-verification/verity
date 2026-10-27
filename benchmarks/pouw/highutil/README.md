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
  so every layer has one segment (S = 1), and under `V31_LTW=1` (run.sh's default) a call at S = 1 takes cuBLASLt's int32
  X W'^T in place of v31's GEMM; the plain arm is cuBLASLt (`torch._int_mm`) on the same calls.
- `hu_lib.cu` builds `libv31.so` from `v31.cu` with two functions more, so the host never waits on the device inside a
  block: `hu_point_b` points a call's weights at the block's device tensor (no `v31_set_b` copy and wait), and
  `hu_levels_async` copies each call's tree to page-locked memory behind its commit (no `v31_root` and `v31_levels`
  wait), for the roots and the openings after the check coin.
- `run.sh`: the build, the config and the periods under a GPU lease (`research run`; its header has the command), and
  with `NSYS=1` each arm's kernel timeline (`timeline.py`: GEMM share, busy share, idle gaps).
- `v31.cu`, `v31.py` and `v31lib.py` are vendored unmodified from compute-research's V31-GPU, branch
  `cursor/compute-research-v31-gpu-ea24` at 17a5bac0e8 (its accel_v31 directory), sha256 7ad2ea0d37c1..., 7288206a8e11...
  and fd7b48ed743a...; `run.sh` records the hashes. Fixes belong upstream there, then a fresh copy here.
