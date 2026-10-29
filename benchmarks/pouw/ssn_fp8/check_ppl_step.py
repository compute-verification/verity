import sys
import numpy as np
import torch
sys.path.insert(0, "benchmarks/ppl")
import steps as ST
from benchmarks.pouw.ssn_fp8 import design as D, sim, study as Stu

bad = 0
for dtype, K in [("e4m3", 1024), ("e4m3", 300), ("bf16", 512), ("bf16", 100)]:
    A, B, *_ = Stu.data(dtype, 16, 12, K, np.random.default_rng(3))
    f = D.form(dtype, A, B, "pin", np.random.default_rng(4), bound="partial")
    w, badw = D.run(dtype, f)
    core = sim.fp32_value(w[:, :, -1]) - f.u[:, None] * f.noise[:, :, -1] - f.bias[:, :, 0]
    xv = torch.tensor(D.decode(dtype, A), dtype=torch.float64)
    wv = torch.tensor(D.decode(dtype, B), dtype=torch.float64)
    # the design's own bound (every prefix) and noise bound, so p agrees
    prod = xv[:, None, :] * wv[None, :, :]
    dmax = torch.maximum(prod.abs().amax((1, 2)), prod.cumsum(-1).abs().amax((1, 2)))
    s, _ = ST.ssnfp_layout(dtype, K)
    data, u, p, unp = ST.ssnfp_product(dtype, xv, wv, dmax, 24 + s)
    same_p = np.array_equal(p.numpy(), f.p)
    eq = np.array_equal(data.numpy(), core)
    b = ST.ssnfp_bound(dtype, xv, wv, "partial")
    sound = bool((b["partial"] <= dmax + 1e-9).all() and (b["l1"] >= b["partial"]).all() and (b["cs"] >= b["partial"] - 1e-6).all()
                 and (b["holder"] >= b["partial"]).all())
    # a separable bound's grid per dot product: column j equals the per-row path at dmax_i 2^b_j
    cols = torch.exp2(torch.arange(wv.shape[0], dtype=torch.float64) % 5 - 2) * 0.75
    sdata, _, sp, _ = ST.ssnfp_product(dtype, xv, wv, (dmax, cols), 24 + s)
    bj = torch.ceil(torch.log2(cols))
    col = [ST.ssnfp_product(dtype, xv, wv[j:j + 1], dmax * 2 ** bj[j], 24 + s) for j in range(wv.shape[0])]
    live = dmax > 0
    sep = all(torch.equal(sdata[:, j], c[0][:, 0]) and torch.equal(sp[live, j], c[2][live]) for j, c in enumerate(col))
    print(dtype, K, "steps", s, f.steps, "same p", same_p, "data part equal", eq, "bounds sound", sound,
          "separable grid", sep, "max |diff|", float(np.abs(data.numpy() - core).max()))
    bad += (not same_p) + (not eq) + (not sound) + (s != f.steps) + (not sep)
sys.exit(1 if bad else 0)
