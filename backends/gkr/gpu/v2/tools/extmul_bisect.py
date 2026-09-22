"""Bisect which feature of a-gpu's ``ext_mul_kernel`` makes it disagree with torch on this GPU (lane a-kernel-a100).

    python -m gpu.v2.tools.extmul_bisect [--n 8192] [--json OUT]

One parametrised copy of ``gpu/kernels.py::ext_mul_kernel`` with constexpr switches: ``TO32`` (the ``.to(tl.int32)``
on the loads), ``R2FIX`` (the trailing ``mmul(c, R^2)`` on device vs ``* R`` on the host), ``PK`` (the packed module's
``_ext_mmul6_j``/``mmul`` instead of a-gpu's), ``IDX64`` (int64 element offsets), plus ``num_warps`` and the input dtype.
Each variant is compared coordinate-wise with the torch ``ext_mul`` (kernels disabled).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import torch
import triton
import triton.language as tl

from ... import field as _field
from ... import kernels as ak
from ...field import P
from ...kernels import R2_C as AK_R2
from ...kernels import ext_mmul as ak_ext_mmul
from ...kernels import mmul as ak_mmul
from packed.kernels_triton import _ext_mmul6_j as pk_ext_mmul
from packed.kernels_triton import mmul as pk_mmul

R = 1 << 32


@triton.jit
def _k(a_ptr, b_ptr, out_ptr, n, TO32: tl.constexpr, R2FIX: tl.constexpr, PK: tl.constexpr, IDX64: tl.constexpr,
       R2c: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    if IDX64:
        base = idx.to(tl.int64) * 6
    else:
        base = idx * 6
    if TO32:
        a0 = tl.load(a_ptr + base + 0, mask=m, other=0).to(tl.int32)
        a1 = tl.load(a_ptr + base + 1, mask=m, other=0).to(tl.int32)
        a2 = tl.load(a_ptr + base + 2, mask=m, other=0).to(tl.int32)
        a3 = tl.load(a_ptr + base + 3, mask=m, other=0).to(tl.int32)
        a4 = tl.load(a_ptr + base + 4, mask=m, other=0).to(tl.int32)
        a5 = tl.load(a_ptr + base + 5, mask=m, other=0).to(tl.int32)
        b0 = tl.load(b_ptr + base + 0, mask=m, other=0).to(tl.int32)
        b1 = tl.load(b_ptr + base + 1, mask=m, other=0).to(tl.int32)
        b2 = tl.load(b_ptr + base + 2, mask=m, other=0).to(tl.int32)
        b3 = tl.load(b_ptr + base + 3, mask=m, other=0).to(tl.int32)
        b4 = tl.load(b_ptr + base + 4, mask=m, other=0).to(tl.int32)
        b5 = tl.load(b_ptr + base + 5, mask=m, other=0).to(tl.int32)
    else:
        a0 = tl.load(a_ptr + base + 0, mask=m, other=0)
        a1 = tl.load(a_ptr + base + 1, mask=m, other=0)
        a2 = tl.load(a_ptr + base + 2, mask=m, other=0)
        a3 = tl.load(a_ptr + base + 3, mask=m, other=0)
        a4 = tl.load(a_ptr + base + 4, mask=m, other=0)
        a5 = tl.load(a_ptr + base + 5, mask=m, other=0)
        b0 = tl.load(b_ptr + base + 0, mask=m, other=0)
        b1 = tl.load(b_ptr + base + 1, mask=m, other=0)
        b2 = tl.load(b_ptr + base + 2, mask=m, other=0)
        b3 = tl.load(b_ptr + base + 3, mask=m, other=0)
        b4 = tl.load(b_ptr + base + 4, mask=m, other=0)
        b5 = tl.load(b_ptr + base + 5, mask=m, other=0)
    if PK:
        c0, c1, c2, c3, c4, c5 = pk_ext_mmul(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5)
        if R2FIX:
            c0 = pk_mmul(c0, R2c)
            c1 = pk_mmul(c1, R2c)
            c2 = pk_mmul(c2, R2c)
            c3 = pk_mmul(c3, R2c)
            c4 = pk_mmul(c4, R2c)
            c5 = pk_mmul(c5, R2c)
    else:
        c0, c1, c2, c3, c4, c5 = ak_ext_mmul(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5)
        if R2FIX:
            c0 = ak_mmul(c0, R2c)
            c1 = ak_mmul(c1, R2c)
            c2 = ak_mmul(c2, R2c)
            c3 = ak_mmul(c3, R2c)
            c4 = ak_mmul(c4, R2c)
            c5 = ak_mmul(c5, R2c)
    tl.store(out_ptr + base + 0, c0, mask=m)
    tl.store(out_ptr + base + 1, c1, mask=m)
    tl.store(out_ptr + base + 2, c2, mask=m)
    tl.store(out_ptr + base + 3, c3, mask=m)
    tl.store(out_ptr + base + 4, c4, mask=m)
    tl.store(out_ptr + base + 5, c5, mask=m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(args.seed)
    n = args.n
    a = torch.randint(0, P, (n, 6), generator=g, dtype=torch.int64).to(dev)
    b = torch.randint(0, P, (n, 6), generator=g, dtype=torch.int64).to(dev)
    saved = _field._fast_ext_mul
    _field._fast_ext_mul = None
    try:
        want = _field.ext_mul(a, b)
    finally:
        _field._fast_ext_mul = saved
    rep = {"gpu": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()), "torch": torch.__version__,
           "triton": triton.__version__, "n": n, "variants": []}
    # the real a-gpu wrapper first
    got = ak.ext_mul(a, b)
    bad = (got % P != want).sum(0).tolist()
    rep["agpu.ext_mul(int64 in)"] = bad
    print("agpu.ext_mul int64 in, per-coordinate mismatches:", bad, flush=True)
    got = ak.ext_mul(a.to(torch.int32), b.to(torch.int32))
    bad = (got % P != want).sum(0).tolist()
    rep["agpu.ext_mul(int32 in)"] = bad
    print("agpu.ext_mul int32 in, per-coordinate mismatches:", bad, flush=True)
    for to32, r2fix, pk, idx64, nw, in_dtype, out_dtype in itertools.product((1, 0), (1, 0), (0, 1), (0, 1), (4, 1, 8),
                                                                              ("i64", "i32"), ("i64", "i32")):
        if pk and (to32 or idx64 or nw != 4):
            continue                                                  # keep the PK arm small
        ai = a if in_dtype == "i64" else a.to(torch.int32)
        bi = b if in_dtype == "i64" else b.to(torch.int32)
        out = torch.empty((n, 6), dtype=torch.int64 if out_dtype == "i64" else torch.int32, device=dev)
        try:
            _k[(triton.cdiv(n, 256),)](ai.contiguous().view(-1), bi.contiguous().view(-1), out.view(-1), n, TO32=to32,
                                       R2FIX=r2fix, PK=pk, IDX64=idx64, R2c=ak.R2_MOD_P, BLOCK=256, num_warps=nw)
            torch.cuda.synchronize()
            o = out.to(torch.int64) % P
            if not r2fix:
                o = (o * (R % P)) % P
            bad = (o != want).sum(0).tolist()
            err = None
        except Exception as e:  # noqa: BLE001
            bad, err = None, f"{type(e).__name__}: {e}"
        v = {"TO32": to32, "R2FIX": r2fix, "PK": pk, "IDX64": idx64, "num_warps": nw, "in": in_dtype, "out": out_dtype,
             "mismatch_per_coord": bad, "error": err}
        rep["variants"].append(v)
        print(json.dumps(v), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
