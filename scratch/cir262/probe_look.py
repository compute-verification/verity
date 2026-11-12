import sys, json
import numpy as np
from pathlib import Path
from tests.properties import glm53_flashmla_fp8_capture_gpu as X
V = X.V
outdir = Path(sys.argv[1]) / "capture"
lay = {k: np.load(outdir / f"layout_{k}.npy") for k in ("lens", "table", "numpy")}
cache = np.load(outdir / "cache.npy", mmap_mode="r")
rb, blocks = np.load(outdir / "readback" / "rows.npy"), np.load(outdir / "readback" / "blocks.npy").tolist()
plant = len(lay["lens"]) - 1
planted = X.slots(lay, plant, np.arange(X.PLANT_ROWS)).tolist()
for case in sys.argv[2:]:
    a = X._case(outdir, case)
    for t in range(a["out"].shape[0]):
        g = np.asarray(a["gidx"][t]); slot = int(g[g != -1][0])
        pads = np.asarray(a["out"][t][X.NH:], dtype=np.uint16); got = pads[0]
        back = rb[blocks.index(slot // X.PBS) * X.PBS + slot % X.PBS]
        nope, scale, rope = X.rows_at(cache, [slot])
        staged = V.kv_rows(nope, scale, rope, [True], X.QT)[0, :X.DN]
        dense = X.dense_rows(nope, scale)[0]
        sp_n = bool(np.any((staged & 0x7F80) == 0x7F80)); sp_r = bool(np.any((rope[0] & 0x7F80) == 0x7F80))
        diff = got != back[:X.DN]
        zs = int(((got == 0) & (back[:X.DN] == 0x8000)).sum())
        d = (got.astype(np.int64) - back[:X.DN].astype(np.int64))[diff]
        r = int(a["req"][t]); w = ("planted#%d" % planted.index(slot)) if slot in planted else "numpy" if lay["numpy"][r] else "vllm"
        print(case, t, w, "scales", [hex(int(s)) for s in scale[0]], "differ", int(diff.sum()), "zero-sign", zs, "specialN", sp_n, "specialR", sp_r,
              "sparse_model", int((got != np.where(staged == 0x8000, 0, staged)).sum()), "dense_model", int((back[:X.DN] != dense).sum()),
              "ulp", dict(zip(*[x.tolist() for x in np.unique(d, return_counts=True)])) if 0 < d.size and len(np.unique(d)) < 8 else d.size)
