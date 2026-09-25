"""Bit-exactness and timing of tests_fused against the torch paths (protocol._combine, witness.quad_p0)."""

from __future__ import annotations

import argparse
import json
import time

import torch

from . import protocol, tests_fused, witness
from .field import P


def _bench(fn, reps):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = torch.device(a.device)
    g = torch.Generator(device="cpu").manual_seed(7)
    rep = {"exact": {}, "timing": {}}
    D = 6
    f = tests_fused.fused_for(D, dev)
    assert f is not None
    for R, cols, dt in ((37, 1000, torch.int32), (3521, 16640, torch.int64), (2534, 65536, torch.int32)):
        X = torch.randint(0, P, (R, cols), generator=g).to(dt).to(dev)
        C = torch.randint(0, P, (D, R), generator=g).to(dev)
        witness.USE_FUSED = False
        ref = protocol._combine(C, X.to(torch.int64))
        refb = torch.stack([witness._bool_row(C[d], X.to(torch.int64)) for d in range(D)])
        witness.USE_FUSED = True
        got = f.lincomb(C, X)
        gotb = f.lincomb(C, X, square_minus=True)
        rep["exact"][f"lincomb_{R}x{cols}_{str(dt)[6:]}"] = torch.equal(ref, got)
        rep["exact"][f"boolcomb_{R}x{cols}_{str(dt)[6:]}"] = torch.equal(refb, gotb)
        # the booleanity rows read in place through a row table (a random subset), v4 and scalar kernels vs torch
        Rb = max(1, R // 3)
        rows = torch.randperm(R, generator=g)[:Rb].sort().values.to(dev)
        Cb = torch.randint(0, P, (D, Rb), generator=g).to(dev)
        refr = torch.stack([witness._bool_row(Cb[d], X.index_select(0, rows).to(torch.int64)) for d in range(D)])
        gotr = f.lincomb(Cb, X, square_minus=True, rows=rows)
        vec = tests_fused.VEC_COLS
        tests_fused.VEC_COLS = 1
        gots = f.lincomb(Cb, X, square_minus=True, rows=rows)
        tests_fused.VEC_COLS = vec
        rep["exact"][f"boolcomb_rows_{Rb}x{cols}_{str(dt)[6:]}"] = torch.equal(refr, gotr) and torch.equal(refr, gots)
        if R > 100:
            tests_fused.VEC_COLS = 1
            t_s = _bench(lambda C=C, X=X: f.lincomb(C, X, square_minus=True), a.reps)
            tests_fused.VEC_COLS = vec
            t_v = _bench(lambda C=C, X=X: f.lincomb(C, X, square_minus=True), a.reps)
            rep["timing"][f"boolcomb_{R}x{cols}"] = {"scalar_ms": t_s, "v4_ms": t_v}
            print(f"boolcomb {R}x{cols}: scalar {t_s:.3f} ms | v4 {t_v:.3f} ms")
        if R > 100:
            witness.USE_FUSED = False
            t_ref = _bench(lambda C=C, X=X: protocol._combine(C, X.to(torch.int64)), a.reps)
            witness.USE_FUSED = True
            t_new = _bench(lambda C=C, X=X: f.lincomb(C, X), a.reps)
            rep["timing"][f"lincomb_{R}x{cols}"] = {"torch_ms": t_ref, "fused_ms": t_new}
            print(f"lincomb {R}x{cols}: torch {t_ref:.2f} ms | fused {t_new:.2f} ms")
    # general quadratic constraints: random sparse tables, Q rows, <= 4 terms per side
    Q, cols, R = 853, 65536, 3521
    X = torch.randint(0, P, (R, cols), generator=g).to(torch.int32).to(dev)
    gen = {}
    for side in ("a", "b", "c"):
        nnz = torch.randint(1, 5, (Q,), generator=g)
        qq = torch.repeat_interleave(torch.arange(Q), nnz)
        jj = torch.randint(0, R, (qq.numel(),), generator=g)
        cc = torch.randint(0, P, (qq.numel(),), generator=g)
        consts = torch.randint(0, P, (Q,), generator=g)
        perm = torch.randperm(qq.numel(), generator=g)                          # unsorted, as _triples may give
        gen[side] = tuple(t.to(dev) for t in (qq[perm], jj[perm], cc[perm], consts))
    rho = torch.randint(0, P, (D, Q), generator=g).to(dev)
    A = witness.apply_sparse(gen["a"][:3], gen["a"][3], Q, X)
    B = witness.apply_sparse(gen["b"][:3], gen["b"][3], Q, X)
    Cc = witness.apply_sparse(gen["c"][:3], gen["c"][3], Q, X)
    T = ((A * B) % P - Cc) % P
    witness.USE_FUSED = False
    ref = protocol._combine(rho, T)
    witness.USE_FUSED = True
    csr = tests_fused.csr_tables(gen, Q, dev)
    got = f.quad_general(rho, X, csr)
    rep["exact"]["quad_general_853x65536"] = torch.equal(ref, got)

    def torch_quad():
        A = witness.apply_sparse(gen["a"][:3], gen["a"][3], Q, X)
        B = witness.apply_sparse(gen["b"][:3], gen["b"][3], Q, X)
        Cc = witness.apply_sparse(gen["c"][:3], gen["c"][3], Q, X)
        T = ((A * B) % P - Cc) % P
        witness.USE_FUSED = False
        out = protocol._combine(rho, T)
        witness.USE_FUSED = True
        return out
    t_ref = _bench(torch_quad, a.reps)
    t_new = _bench(lambda: f.quad_general(rho, X, csr), a.reps)
    rep["timing"]["quad_general_853x65536"] = {"torch_ms": t_ref, "fused_ms": t_new}
    print(f"quad_general 853x65536: torch {t_ref:.2f} ms | fused {t_new:.2f} ms")
    # quad_v4 (Montgomery, 4 columns per thread) against the scalar kernel: sides of 0..44 terms (fp8-ada-v3x4 has up to
    # 44 / 30 / 8), residues up to p - 1, every grid layout; and the partial-sum reduction kernel against torch's
    Q2, cols2, R2 = 301, 4096, 700
    X2 = torch.randint(0, P, (R2, cols2), generator=g).to(torch.int32)
    X2[:, :64] = P - 1
    X2 = X2.to(dev)
    gen2 = {}
    for side, hi in (("a", 45), ("b", 31), ("c", 9)):
        nnz = torch.randint(0, hi, (Q2,), generator=g)
        qq = torch.repeat_interleave(torch.arange(Q2), nnz)
        jj = torch.randint(0, R2, (qq.numel(),), generator=g)
        cc = torch.randint(0, P, (qq.numel(),), generator=g)
        cc[:16] = P - 1
        consts = torch.randint(0, P, (Q2,), generator=g)
        gen2[side] = tuple(t.to(dev) for t in (qq, jj, cc, consts))
    rho2 = torch.randint(0, P, (D, Q2), generator=g).to(dev)
    csr2 = tests_fused.csr_tables(gen2, Q2, dev)
    saved = (tests_fused.QUAD_V4, tests_fused.REDUCE_KERNEL, tests_fused.QUAD_SPLITS, tests_fused.QUAD_QFAST,
             tests_fused.QUAD_THREADS)
    tests_fused.QUAD_V4, tests_fused.REDUCE_KERNEL = False, False
    ref2 = f.quad_general(rho2, X2, csr2)
    tests_fused.QUAD_V4, tests_fused.REDUCE_KERNEL = True, True
    ok2 = True
    for thr in (128, 256):
        for qf in (False, True):
            for S in (1, 7, 128):
                tests_fused.QUAD_THREADS, tests_fused.QUAD_QFAST, tests_fused.QUAD_SPLITS = thr, qf, S
                ok2 &= torch.equal(f.quad_general(rho2, X2, csr2), ref2)
    (tests_fused.QUAD_V4, tests_fused.REDUCE_KERNEL, tests_fused.QUAD_SPLITS, tests_fused.QUAD_QFAST,
     tests_fused.QUAD_THREADS) = saved
    rep["exact"]["quad_v4_vs_scalar_301x4096"] = bool(ok2)
    part = torch.randint(0, P, (37, D, 1000), generator=g).to(torch.int32).to(dev)
    tests_fused.REDUCE_KERNEL = True
    rep["exact"]["reduce_partial"] = torch.equal(f._reduce(part), part.to(torch.int64).sum(0) % P)
    tests_fused.REDUCE_KERNEL = saved[1]
    # lincomb2 (w and v in one pass) against two lincomb calls: C1 a row-strided view with negative and >= p entries,
    # C2 over a row prefix, residues up to p - 1, several split counts
    Rl, Rv, colsl = 1037, 1001, 2176
    Xl = torch.randint(0, P, (Rl, colsl), generator=g).to(torch.int32)
    Xl[:, :32] = P - 1
    Xl = Xl.to(dev)
    big = torch.randint(-3 * P, 3 * P, (D, Rl + 40), generator=g, dtype=torch.int64)
    big[:, :8] = P - 1
    C1 = big.to(dev)[:, :Rl]
    C2 = torch.randint(0, P, (D, Rv), generator=g, dtype=torch.int64).to(dev)
    ref_w, ref_v = f.lincomb(C1 % P, Xl), f.lincomb(C2, Xl[:Rv])
    ok3 = f.lincomb2_ok(C1, C2, Xl)
    saved3 = (tests_fused.LINCOMB2_SPLITS, tests_fused.LINCOMB2_THREADS)
    for thr in (64, 128, 256):
        for S in (0, 1, 9, 200):
            tests_fused.LINCOMB2_THREADS, tests_fused.LINCOMB2_SPLITS = thr, S
            w2, v2 = f.lincomb2(C1, C2, Xl)
            ok3 &= torch.equal(w2, ref_w) and torch.equal(v2, ref_v)
    tests_fused.LINCOMB2_SPLITS, tests_fused.LINCOMB2_THREADS = saved3
    rep["exact"]["lincomb2_vs_lincomb_1037x2176"] = bool(ok3)
    # intt_scaled against field.intt * post, residues up to p - 1
    from .field import intt as field_intt
    ok4 = True
    for nn in (2, 8, 1024, 16384):
        xi = torch.randint(0, P, (D, nn), generator=g, dtype=torch.int64)
        xi[:, :2] = P - 1
        xi = xi.to(dev)
        post = torch.randint(0, P, (nn,), generator=g, dtype=torch.int64).to(dev)
        ok4 &= f.intt_ok(xi) and torch.equal(f.intt_scaled(xi, post), (field_intt(xi) * post) % P)
    rep["exact"]["intt_scaled_vs_field_intt"] = bool(ok4)
    ok = all(rep["exact"].values())
    print("exact:", rep["exact"])
    print("bit-exact:", "PASS" if ok else "FAIL")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=1)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
