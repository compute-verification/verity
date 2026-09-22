"""Probe the two Triton Montgomery primitives (a-gpu ``gpu/kernels.py`` vs a-packed2 ``packed/kernels_triton.py``)
element by element against the Python oracle, and characterise a mismatch (lane a-kernel-a100).

    python -m gpu.v2.tools.mmul_probe [--n 65536] [--seed 1] [--json OUT]

Runs, on the same random canonical inputs: ``mmul`` of each module, ``tl.umulhi`` alone, the ``a * b`` low word, the
``ext_mul`` kernel of a-gpu vs the packed ``ext_mul_rows`` vs torch, and compares to candidate wrong models
(signed ``mul.hi.s32``, missing ``R^2`` fix-up, ...) so the report can say WHICH instruction differs.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import triton
import triton.language as tl

from ... import field as _field
from ... import kernels as ak
from ...kernels import R2_C as AK_R2
from ...kernels import mmul as ak_mmul
from packed.kernels_triton import mmul as pk_mmul
from ...field import P
from packed import kernels_triton as pk
from packed import sumcheck_packed as _sp

R = 1 << 32
MU = pow(P, -1, R)
R2 = R * R % P


def mont_py(a: int, b: int) -> int:
    t = a * b
    lo, hi = t & (R - 1), t >> 32
    m = (lo * MU) & (R - 1)
    u = (m * P) >> 32
    r = hi - u
    return r + P if r < 0 else r


def mont_py_signed_hi(a: int, b: int) -> int:
    """The same with ``umulhi`` replaced by a signed ``mul.hi.s32`` (candidate wrong model)."""
    def s32(x):
        x &= R - 1
        return x - R if x >= 1 << 31 else x
    def shi(x, y):
        return (s32(x) * s32(y)) >> 32
    lo = (a * b) & (R - 1)
    hi = shi(a, b)
    m = (lo * MU) & (R - 1)
    u = shi(m, P)
    r = hi - u
    r = s32(r)
    return r + P if r < 0 else r


@triton.jit
def _probe_agpu(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(a_ptr + i, mask=m, other=0).to(tl.int32)
    b = tl.load(b_ptr + i, mask=m, other=0).to(tl.int32)
    tl.store(o_ptr + i, ak_mmul(a, b), mask=m)
    tl.store(o_ptr + n + i, tl.umulhi(a, b), mask=m)
    tl.store(o_ptr + 2 * n + i, a * b, mask=m)
    tl.store(o_ptr + 3 * n + i, ak_mmul(ak_mmul(a, b), AK_R2), mask=m)


@triton.jit
def _probe_packed(a_ptr, b_ptr, o_ptr, n, R2c: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(a_ptr + i, mask=m, other=0)
    b = tl.load(b_ptr + i, mask=m, other=0)
    tl.store(o_ptr + i, pk_mmul(a, b), mask=m)
    tl.store(o_ptr + n + i, tl.umulhi(a, b), mask=m)
    tl.store(o_ptr + 2 * n + i, a * b, mask=m)
    tl.store(o_ptr + 3 * n + i, pk_mmul(pk_mmul(a, b), R2c), mask=m)


def _cmp(name, got: torch.Tensor, want: list[int], rep: dict, models: dict | None = None) -> None:
    g = got.to(torch.int64).cpu().tolist()
    bad = [i for i, (x, y) in enumerate(zip(g, want)) if (x % (1 << 32)) != (y % (1 << 32))]
    row = {"mismatches": len(bad), "of": len(want)}
    if bad:
        i = bad[0]
        row["first"] = {"i": i, "got": g[i], "want": want[i]}
        if models:
            row["matches_model"] = {k: sum(1 for i in bad if (g[i] % (1 << 32)) == (v[i] % (1 << 32))) for k, v in models.items()}
    rep[name] = row
    print(name, row, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(args.seed)
    n = args.n
    a = torch.randint(0, P, (n,), generator=g, dtype=torch.int64)
    b = torch.randint(0, P, (n,), generator=g, dtype=torch.int64)
    al, bl = a.tolist(), b.tolist()
    want_m = [mont_py(x, y) for x, y in zip(al, bl)]
    want_hi = [(x * y) >> 32 for x, y in zip(al, bl)]
    want_lo = [((x * y) & (R - 1)) for x, y in zip(al, bl)]
    want_full = [x * y % P for x, y in zip(al, bl)]
    model_signed = [mont_py_signed_hi(x, y) for x, y in zip(al, bl)]
    model_no_r2 = want_m
    rep = {"gpu": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()), "torch": torch.__version__,
           "triton": triton.__version__, "n": n}
    a32, b32 = a.to(torch.int32).to(dev), b.to(torch.int32).to(dev)
    for name, kern, extra in (("agpu", _probe_agpu, {}), ("packed", _probe_packed, {"R2c": R2})):
        o = torch.zeros((4 * n,), dtype=torch.int32, device=dev)
        kern[(triton.cdiv(n, 256),)](a32, b32, o, n, BLOCK=256, **extra)
        torch.cuda.synchronize()
        _cmp(f"{name}.mmul", o[:n], want_m, rep, {"signed_mulhi": model_signed})
        _cmp(f"{name}.umulhi", o[n:2 * n], want_hi, rep)
        _cmp(f"{name}.mul_lo", o[2 * n:3 * n], want_lo, rep)
        _cmp(f"{name}.mmul_mmul_R2 (= a b mod p)", o[3 * n:], want_full, rep, {"no_R2_fixup": model_no_r2})
    # the real kernels on extension elements
    ae = torch.randint(0, P, (n, 6), generator=g, dtype=torch.int64)
    be = torch.randint(0, P, (n, 6), generator=g, dtype=torch.int64)
    saved = _field._fast_ext_mul
    _field._fast_ext_mul = None
    try:
        ref = _field.ext_mul(ae.to(dev), be.to(dev))
    finally:
        _field._fast_ext_mul = saved
    got_a = ak.ext_mul(ae.to(dev), be.to(dev))
    got_p = pk.ext_mul_rows(ae.to(torch.int32).to(dev), be.to(torch.int32).to(dev))
    _cmp("agpu.ext_mul", got_a.reshape(-1), ref.reshape(-1).tolist(), rep)
    _cmp("packed.ext_mul_rows", got_p.reshape(-1), ref.reshape(-1).tolist(), rep)
    # a-gpu combine_level and fold on the same data (these feed the LogUp tree / GKR)
    p = torch.randint(0, P, (2 * (n // 2),), generator=g, dtype=torch.int32).to(dev)
    q = ae.to(torch.int32).to(dev)
    saved = _field._fast_combine_level
    _field._fast_combine_level = None
    try:
        from ... import logup
        saved_mul = _field._fast_ext_mul
        _field._fast_ext_mul = None
        try:
            rp, rq = logup._combine_level(p, q)
        finally:
            _field._fast_ext_mul = saved_mul
    finally:
        _field._fast_combine_level = saved
    gp, gq = ak.combine_level(p, q)
    _cmp("agpu.combine_level.p", gp.reshape(-1), rp.reshape(-1).tolist(), rep)
    _cmp("agpu.combine_level.q", gq.reshape(-1), rq.reshape(-1).tolist(), rep)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
