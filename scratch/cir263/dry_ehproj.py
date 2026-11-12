"""CPU dry run of the eh_proj capture's planning, bindings and one compare job (the kernel's words stood in by the model)."""
import tempfile
import time
from pathlib import Path

import numpy as np

from tests.properties import glm53_bf16_gemm_capture_gpu as BG
from tests.properties import glm53_mtp_eh_proj_capture_gpu as M
from verity.definitions.registry import bf16_gemm_check as BC

routes = [[1, 64, 4], [65, 900, None], [901, 3000, 2], [3001, 8192, None]]
print("launch_ms", M.launch_ms(routes))
print("spans 8192", M.row_spans(8192)[:3], len(M.row_spans(8192)), "spans 100", M.row_spans(100))
t = time.time()
print("bind", M.bind_check([2, 3, 4]), round(time.time() - t, 1), "s")
rng = np.random.default_rng(0)
f = lambda s: ((rng.standard_normal(s).astype(np.float32)).view(np.uint32) >> 16).astype(np.uint16)  # noqa: E731
x, w = f((8, M.K)), f((M.N, M.K))
for split in (None, 3):
    out = BC.linear_rows(x, w) if split is None else BC.cublas_splitk_rows(x, w, split, out="bf16")
    d = Path(tempfile.mkdtemp())
    paths = BG._save(d, x=x, w=w, out=out)
    job = {"model": M.MODEL, "phase": M.PHASE, "case": M.NAME, "M": 8, "K": M.K, "N": M.N, "kernel": None, "paths": paths,
           "split_k": split, "probe_paths": None}
    (j,) = M.jobs_of(job)
    t = time.time()
    r = BG.compare(j)
    print(split, {k: r.get(k) for k in ("rows", "words", "mismatch", "reference_coords", "twin_mismatch", "error")},
          round(time.time() - t, 1), "s")
