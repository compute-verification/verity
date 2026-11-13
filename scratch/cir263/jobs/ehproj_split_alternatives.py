"""How many of a split launch's kept words differ under each modeled split-K hypothesis: `python3 ehproj_split_alternatives.py
RUN_DIR M [SPLITS [ROW0 ROW1]]`, from `RUN_DIR/out/eh_proj/words/` (the weight's words and launch M's operands and output), on a
CPU; ROW0 and ROW1 take the launch's rows ROW0 to ROW1 - 1 alone."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from verity.definitions.registry import bf16_gemm_check as BC

run, m = Path(sys.argv[1]), int(sys.argv[2])
splits = int(sys.argv[3]) if len(sys.argv) > 3 else 2
words = run / "out" / "eh_proj" / "words"
w = np.ascontiguousarray(np.load(words / "w.npy"))
x, out = np.load(words / f"M{m}" / "x.npy"), np.load(words / f"M{m}" / "out.npy")
lo, hi = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, x.shape[0])
x, out = x[lo:hi], out[lo:hi]
t, differ = time.time(), {}
for r0 in range(0, x.shape[0], 8):
    models = BC.cublas_splitk_hypotheses(np.ascontiguousarray(x[r0:r0 + 8]), w, splits, out="bf16")
    for name, v in models.items():
        differ[name] = differ.get(name, 0) + int((v != out[r0:r0 + 8]).sum())
print(json.dumps({"run": run.name, "M": m, "splits": splits, "rows": [lo, hi], "words": int(out.size), "differ": differ,
                  "seconds": round(time.time() - t, 1)}))
