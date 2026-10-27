# HIGHUTIL: a GPU at full batch on useful matmuls, certified

Llama-3-8B's four linear layers (QKV 6144 x 4096, O 4096 x 4096, gate-and-up 28672 x 4096, down 4096 x 14336) at
M = 16,384, s8 x s8 with int32 accumulation, called back to back on one RTX PRO 6000 (sm_120). Each call goes through
Dither's no-fold receipt (`verity/core/protocols/pouw/accel/PROTOCOL.md`, v3.2) on V31-GPU's prover, with a live
challenger session a period. The report is the certified share: credited MACs over θ's INT8 tensor peak times the wall
time, and γ against plain cuBLASLt on the same calls, interleaved.

- `period.py`: the workload, the registration, the coins, the picks, the prover's answers and the verifier (its
  docstring has the protocol and the deviations from v3.2). `period.py run CFG ref OUT` runs it all on the CPU at the
  small shapes `prep --small` writes, through v31.py's reference prover.
- `hu.cu`: the period driver. It `#include`s `v31.cu` and runs each layer's calls through v31's `Ctx` (compute on one
  stream, each call's commit beside the next call's GEMMs on another), with a fresh salt key a call, each call's tree
  copied to pinned host memory for the openings, and the activations generated beside the GEMMs.
- `run.sh`: the build, the config and the periods under a GPU lease (`research run`; its header has the command).
- `v31.cu` and `v31.py` are vendored unmodified from compute-research's V31-GPU, branch
  `cursor/compute-research-v31-gpu-ea24` at d251532e04 (its accel_v31 directory), sha256 383381f2df1e... and
  dbfd18690e5f...; `run.sh` records both hashes. Fixes belong upstream there, then a fresh copy here.
