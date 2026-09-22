"""Per-coordinate probe of the extension product primitives (lane a-kernel-a100): a-gpu ``kernels.ext_mmul`` vs the
packed ``_ext_mmul6_j`` vs the Python oracle, with the intermediate ``W`` terms ``w_j = mmul(b_j, WR)`` stored too.

    python -m gpu.v2.tools.extmul_probe [--n 4096] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import triton
import triton.language as tl

from ... import kernels as ak
from ...field import P, W
from ...kernels import WR_C as AK_WR
from ...kernels import ext_mmul as ak_ext_mmul
from ...kernels import mmul as ak_mmul
from packed.kernels_triton import _ext_mmul6_j as pk_ext_mmul
from packed.kernels_triton import mmul as pk_mmul

R = 1 << 32
MU = pow(P, -1, R)
WR = W * R % P
RINV = pow(R % P, P - 2, P)


def mont_py(a: int, b: int) -> int:
    t = a * b
    lo, hi = t & (R - 1), t >> 32
    m = (lo * MU) & (R - 1)
    u = (m * P) >> 32
    r = hi - u
    return r + P if r < 0 else r


def ext_mmul_py(a, b):
    """``a b R^-1`` coordinate-wise exactly as the kernels compute it."""
    w = [None] + [mont_py(b[j], WR) for j in range(1, 6)]
    c = []
    for k in range(6):
        s = 0
        for i in range(6):
            j = k - i
            term = mont_py(a[i], b[j]) if j >= 0 else mont_py(a[i], w[j + 6])
            s = (s + term) % P
        c.append(s)
    return c, w[1:]


@triton.jit
def _k_agpu(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    ba = i * 6
    a0 = tl.load(a_ptr + ba + 0, mask=m, other=0).to(tl.int32)
    a1 = tl.load(a_ptr + ba + 1, mask=m, other=0).to(tl.int32)
    a2 = tl.load(a_ptr + ba + 2, mask=m, other=0).to(tl.int32)
    a3 = tl.load(a_ptr + ba + 3, mask=m, other=0).to(tl.int32)
    a4 = tl.load(a_ptr + ba + 4, mask=m, other=0).to(tl.int32)
    a5 = tl.load(a_ptr + ba + 5, mask=m, other=0).to(tl.int32)
    b0 = tl.load(b_ptr + ba + 0, mask=m, other=0).to(tl.int32)
    b1 = tl.load(b_ptr + ba + 1, mask=m, other=0).to(tl.int32)
    b2 = tl.load(b_ptr + ba + 2, mask=m, other=0).to(tl.int32)
    b3 = tl.load(b_ptr + ba + 3, mask=m, other=0).to(tl.int32)
    b4 = tl.load(b_ptr + ba + 4, mask=m, other=0).to(tl.int32)
    b5 = tl.load(b_ptr + ba + 5, mask=m, other=0).to(tl.int32)
    c0, c1, c2, c3, c4, c5 = ak_ext_mmul(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5)
    ob = i * 11
    tl.store(o_ptr + ob + 0, c0, mask=m)
    tl.store(o_ptr + ob + 1, c1, mask=m)
    tl.store(o_ptr + ob + 2, c2, mask=m)
    tl.store(o_ptr + ob + 3, c3, mask=m)
    tl.store(o_ptr + ob + 4, c4, mask=m)
    tl.store(o_ptr + ob + 5, c5, mask=m)
    tl.store(o_ptr + ob + 6, ak_mmul(b1, AK_WR), mask=m)
    tl.store(o_ptr + ob + 7, ak_mmul(b2, AK_WR), mask=m)
    tl.store(o_ptr + ob + 8, ak_mmul(b3, AK_WR), mask=m)
    tl.store(o_ptr + ob + 9, ak_mmul(b4, AK_WR), mask=m)
    tl.store(o_ptr + ob + 10, ak_mmul(b5, AK_WR), mask=m)


@triton.jit
def _k_packed(a_ptr, b_ptr, o_ptr, n, WRc: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    ba = i * 6
    a0 = tl.load(a_ptr + ba + 0, mask=m, other=0)
    a1 = tl.load(a_ptr + ba + 1, mask=m, other=0)
    a2 = tl.load(a_ptr + ba + 2, mask=m, other=0)
    a3 = tl.load(a_ptr + ba + 3, mask=m, other=0)
    a4 = tl.load(a_ptr + ba + 4, mask=m, other=0)
    a5 = tl.load(a_ptr + ba + 5, mask=m, other=0)
    b0 = tl.load(b_ptr + ba + 0, mask=m, other=0)
    b1 = tl.load(b_ptr + ba + 1, mask=m, other=0)
    b2 = tl.load(b_ptr + ba + 2, mask=m, other=0)
    b3 = tl.load(b_ptr + ba + 3, mask=m, other=0)
    b4 = tl.load(b_ptr + ba + 4, mask=m, other=0)
    b5 = tl.load(b_ptr + ba + 5, mask=m, other=0)
    c0, c1, c2, c3, c4, c5 = pk_ext_mmul(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5)
    ob = i * 11
    tl.store(o_ptr + ob + 0, c0, mask=m)
    tl.store(o_ptr + ob + 1, c1, mask=m)
    tl.store(o_ptr + ob + 2, c2, mask=m)
    tl.store(o_ptr + ob + 3, c3, mask=m)
    tl.store(o_ptr + ob + 4, c4, mask=m)
    tl.store(o_ptr + ob + 5, c5, mask=m)
    tl.store(o_ptr + ob + 6, pk_mmul(b1, WRc), mask=m)
    tl.store(o_ptr + ob + 7, pk_mmul(b2, WRc), mask=m)
    tl.store(o_ptr + ob + 8, pk_mmul(b3, WRc), mask=m)
    tl.store(o_ptr + ob + 9, pk_mmul(b4, WRc), mask=m)
    tl.store(o_ptr + ob + 10, pk_mmul(b5, WRc), mask=m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(args.seed)
    n = args.n
    a = torch.randint(0, P, (n, 6), generator=g, dtype=torch.int64)
    b = torch.randint(0, P, (n, 6), generator=g, dtype=torch.int64)
    want = []
    for i in range(n):
        c, w = ext_mmul_py(a[i].tolist(), b[i].tolist())
        want.append(c + w)
    rep = {"gpu": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()), "torch": torch.__version__,
           "triton": triton.__version__, "n": n, "WR": WR, "W": W, "agpu_WR_C": int(ak.WR), "agpu_MU_I32": int(ak.MU_I32),
           "agpu_R2": int(ak.R2_MOD_P)}
    a32, b32 = a.to(torch.int32).to(dev).contiguous(), b.to(torch.int32).to(dev).contiguous()
    names = ["c0", "c1", "c2", "c3", "c4", "c5", "w1", "w2", "w3", "w4", "w5"]
    for name, kern, extra in (("agpu", _k_agpu, {}), ("packed", _k_packed, {"WRc": WR})):
        o = torch.zeros((n, 11), dtype=torch.int32, device=dev)
        kern[(triton.cdiv(n, 128),)](a32.view(-1), b32.view(-1), o.view(-1), n, BLOCK=128, **extra)
        torch.cuda.synchronize()
        got = o.to(torch.int64).cpu().tolist()
        row = {}
        for k, nm in enumerate(names):
            bad = [i for i in range(n) if got[i][k] % R != want[i][k] % R]
            e = {"mismatches": len(bad)}
            if bad:
                i = bad[0]
                gv, wv = got[i][k] % R, want[i][k]
                e["first"] = {"i": i, "got": gv, "want": wv}
                # characterise: is got == want * R^-1 / * R / == want computed with W=1 (WR -> R) ?
                if k >= 6:
                    bj = int(b[i][k - 5])
                    e["got_is_bj"] = gv == bj                            # W-term dropped (w_j = b_j)
                    e["got_is_W_bj_Rinv"] = gv == (W * bj * RINV) % P
                    e["got_is_bj_Rinv"] = gv == (bj * RINV) % P          # mmul(b_j, 1)
                    e["got_is_W_bj"] = gv == (W * bj) % P
            row[nm] = e
        rep[name] = row
        print(name, json.dumps(row), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
