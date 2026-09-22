"""Micro-benchmark of the constraint-test term (the "36%"): the quadratic test's P_0 = sum_q rho_q (A_q B_q - C_q)
on a (m x n) codeword matrix, naive (three apply_sparse of Q x n, product, combine) vs split (``witness.quad_p0``:
booleanity rows squared in one fused pass, the 7% general constraints through the sparse tables).  Random U (the
timing does not depend on the values); equality of the two forms is asserted.

    python -m backends.direct.ligero.tests_bench --device cuda --n 65536 --reps 5
"""
from __future__ import annotations

import argparse
import time

import torch
from verity_numerical.checker.params import REAL

from .compile import P
from .protocol import _combine
from .unit import compile_unit
from .witness import apply_sparse, quad_p0, tables


def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def naive(tb, rho, U):
    A = apply_sparse(tb.quad["a"][:3], tb.quad["a"][3], tb.Q, U)
    B = apply_sparse(tb.quad["b"][:3], tb.quad["b"][3], tb.Q, U)
    T = (A * B) % P
    del A, B
    C = apply_sparse(tb.quad["c"][:3], tb.quad["c"][3], tb.Q, U)
    T = (T - C) % P
    del C
    return _combine(rho, T)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--D", type=int, default=6)
    args = ap.parse_args(argv)
    dev = args.device
    sys_ = compile_unit(REAL, chain=True)
    tb = tables(sys_, dev)
    sp = tb.quad["split"]
    print(f"m={sys_.m} Q={tb.Q} booleanity={len(sp['bool_q'])} general={sp['Q_gen']} n={args.n} D={args.D}")
    g = torch.Generator(device=dev).manual_seed(1)
    U = torch.randint(0, P, (sys_.m + 30, args.n), dtype=torch.int64, device=dev, generator=g)
    rho = torch.randint(0, P, (args.D, tb.Q), dtype=torch.int64, device=dev, generator=g)
    ref = naive(tb, rho, U)
    new = quad_p0(tb, rho, U)
    assert torch.equal(ref, new), "split form != naive form"
    print("split == naive: OK")
    for name, fn in (("naive", naive), ("split", quad_p0)):
        fn(tb, rho, U)
        _sync(dev)
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            fn(tb, rho, U)
            _sync(dev)
            ts.append(time.perf_counter() - t0)
        print(f"{name:6s}: {min(ts) * 1e3:8.1f} ms (min of {args.reps}; median {sorted(ts)[len(ts) // 2] * 1e3:.1f})")
    # the pieces of the split form
    Ub = U.index_select(0, sp["bool_j"])
    from .witness import _bool_row
    from .field import compiled
    row = compiled(_bool_row, dev)
    rb = rho.index_select(1, sp["bool_q"])
    row(rb[0], Ub); _sync(dev)
    t0 = time.perf_counter()
    for d in range(args.D):
        row(rb[d], Ub)
    _sync(dev)
    print(f"  booleanity pass x{args.D}: {(time.perf_counter() - t0) * 1e3:.1f} ms  ({Ub.numel() * 8 / 1e9:.2f} GB read per pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
