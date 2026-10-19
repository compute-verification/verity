"""kernels-dither-rtx's sbench.py on the H200 and the B200, unchanged but for one more arm:

    stock-cutlass  sbench's stock arm without its disable, so vLLM keeps its first int8 choice, CutlassInt8ScaledMMLinearKernel.
                   It runs on sm_90a only (it refuses int8 on sm_100a and sm_120). It applies the weight scale before the
                   activation scale, the reverse of Triton and of the int8-exact path (CA's H1-HB, r20261019-064629-9e21),
                   so Dither's served row pairs with `stock` (Triton) and this arm is reported beside it.

sbench's arms run in spawned worker processes, which import this file as their main module before unpickling
`sbench.worker`, so the change below applies in every worker too. Run by hb_served.sh with HB_DITHER_DIR, the source tree's
benchmarks/hash/dither.

    python hb_sbench.py --out OUT [sbench.py's flags]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ["HB_DITHER_DIR"])
import sbench  # noqa: E402

_build = sbench.build


def build(arm: str, scratch, gpu_mem: float):
    if arm != "stock-cutlass":
        return _build(arm, scratch, gpu_mem)
    from vllm import LLM
    os.environ.pop("VLLM_DISABLED_KERNELS", None)
    return LLM(model=str(sbench.stock_checkpoint(sbench.PIN, scratch)), **sbench.engine_args(gpu_mem))


sbench.build = build

if __name__ == "__main__":
    sys.exit(sbench.main())
