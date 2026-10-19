"""kernels-dither-rtx's sbench.py on the H200 and the B200, unchanged but for the stock arm on sm_90a: with HB_STOCK_CUTLASS=1
the arm keeps vLLM's first int8 choice, CutlassInt8ScaledMMLinearKernel, which sbench disables for sm_120 (where it refuses
int8). On sm_100a cutlass refuses int8 too, so sbench's disable stays and vLLM takes Triton's kernel.

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

if os.environ.get("HB_STOCK_CUTLASS") == "1":
    sbench.DISABLED.pop("stock", None)

if __name__ == "__main__":
    sys.exit(sbench.main())
