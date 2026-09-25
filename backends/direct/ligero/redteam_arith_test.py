"""Red-team (lane red-team-arith) of lane/arith's tests_fused kernels (quad_v4 + reduce_partial, lincomb2_v4, intt_rows):
adversarial inputs against an independent torch reference (protocol._combine / field.intt with the fused kernels off)
and against the kernels they replace.

    python -m backends.direct.ligero.redteam_arith_test [--out report.json] [--big]

Every case is recorded as true (bit-exact), false (a counterexample: the inputs are summarised in ``detail``) or the
launch error.  Cases marked ``domain`` feed inputs outside the kernels' documented domain (X canonical); they document
behaviour and are not part of the verdict."""

from __future__ import annotations

import argparse
import json
import traceback

import torch

from . import protocol, tests_fused, witness
from .field import P, intt as field_intt

I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
EXTREMES = [I64_MIN, I64_MAX, I64_MIN + 1, I64_MAX - 1, -1, 0, 1, P - 1, P, P + 1, -P, -P + 1, -P - 1, 2 * P - 1,
            2 * P, 3 * P + 5, -(1 << 62), (1 << 62), (1 << 32) - 1, 1 << 32, -(1 << 32), 1 << 31, -(1 << 31)]


def _ref_combine(C: torch.Tensor, X: torch.Tensor, rows: int = 1024) -> torch.Tensor:
    """(D, cols) = (C mod p) @ X mod p in plain torch int64 (every product reduced; no fused kernel)."""
    Cm = C % P
    out = torch.zeros((C.shape[0], X.shape[1]), dtype=torch.int64, device=X.device)
    for j0 in range(0, X.shape[0], rows):
        Xc = X[j0:j0 + rows].to(torch.int64)
        for d in range(C.shape[0]):
            out[d] += ((Cm[d, j0:j0 + rows, None] * Xc) % P).sum(0)
            out[d] %= P
    return out


def _extreme_matrix(D: int, R: int, g: torch.Generator, dev) -> torch.Tensor:
    m = torch.randint(-(1 << 62), 1 << 62, (D, R), generator=g, dtype=torch.int64) * 2 + torch.randint(0, 2, (D, R), generator=g)
    ext = torch.tensor(EXTREMES, dtype=torch.int64)
    k = min(R, len(EXTREMES))
    for d in range(D):
        m[d, :k] = ext[torch.randperm(len(EXTREMES), generator=g)[:k]]
    return m.to(dev)


class Report:
    def __init__(self):
        self.cases: dict = {}

    def run(self, name, fn, domain=False):
        try:
            ok, detail = fn()
            self.cases[name] = {"ok": bool(ok), "detail": detail, "domain": domain}
        except Exception as e:  # noqa: BLE001
            self.cases[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400], "domain": domain,
                                "trace": traceback.format_exc()[-800:]}
        c = self.cases[name]
        print(f"{'PASS' if c['ok'] else ('ERROR' if 'error' in c else 'FAIL')}{' (domain)' if domain else ''} {name}"
              f" {c.get('detail') or c.get('error') or ''}", flush=True)


def _mismatch(a: torch.Tensor, b: torch.Tensor) -> dict | None:
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    ne = (a != b).nonzero()
    if ne.numel() == 0:
        return None
    i = tuple(ne[0].tolist())
    return {"n_diff": int(ne.shape[0]), "first": list(i), "got": int(a[i]), "want": int(b[i])}


def lincomb2_case(D, R, R1, R2, cols, xmode, g, dev, strided_view=True):
    def fn():
        f = tests_fused.fused_for(D, dev)
        if xmode == "dev":
            gd = torch.Generator(device=dev).manual_seed(R * 131 + cols)
            X = torch.randint(0, P, (R, cols), generator=gd, device=dev, dtype=torch.int32)
            X[:, :4] = P - 1
        elif xmode == "pm1":
            X = torch.full((R, cols), P - 1, dtype=torch.int32)
        elif xmode == "noncanon":
            X = torch.randint(P, (1 << 31) - 1, (R, cols), generator=g, dtype=torch.int64).to(torch.int32)
            X[:, :4] = (1 << 31) - 1
        else:
            X = torch.randint(0, P, (R, cols), generator=g, dtype=torch.int64).to(torch.int32)
            X[:, :4] = P - 1
            X[:, 4:8] = 0
        X = X.to(dev)
        if strided_view:     # like r = x[:, :M] of the challenge block: row stride > R
            wide = _extreme_matrix(D, R1 + 29, g, dev)
            C1 = wide[:, :R1]
        else:
            C1 = _extreme_matrix(D, R1, g, dev)
        C2 = _extreme_matrix(D, R2, g, dev) if R2 else torch.zeros((D, 0), dtype=torch.int64, device=dev)
        ok_path = f.lincomb2_ok(C1, C2, X)
        try:
            w, v = f.lincomb2(C1, C2, X)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 -- record whether the replaced kernels take the same inputs
            try:
                f.lincomb(C1[:, :R] % P, X), f.lincomb(C2 % P, X[:R2]) if R2 else None
                torch.cuda.synchronize()
                old = "old two-lincomb path OK on the same inputs"
            except Exception as e2:  # noqa: BLE001
                old = f"old path fails too: {type(e2).__name__}"
            raise RuntimeError(f"{type(e).__name__}: {e} [smem {2 * D * min(1024, -(-R // max(1, -(-R // 1024)))) * 4} B max; {old}]") from e
        rw = _ref_combine(C1[:, :R], X)
        rv = _ref_combine(C2, X[:R2]) if R2 else torch.zeros((D, cols), dtype=torch.int64, device=dev)
        mw, mv = _mismatch(w, rw), _mismatch(v, rv)
        detail = {"lincomb2_ok": ok_path}
        if mw:
            detail["w"] = mw
        if mv:
            detail["v"] = mv
        if xmode != "noncanon" and R2:   # the kernels it replaces (two lincomb calls) on the same inputs
            ow, ov = f.lincomb(C1[:, :R] % P, X), f.lincomb(C2 % P, X[:R2])
            detail["old_equal"] = bool(torch.equal(ow, w) and torch.equal(ov, v))
        return (ok_path and mw is None and mv is None), detail
    return fn


def quad_case(D, Q, R, cols, hi, mode, g, dev, splits=None):
    def fn():
        f = tests_fused.fused_for(D, dev)
        if mode == "pm1":
            X = torch.full((R, cols), P - 1, dtype=torch.int32)
        elif mode == "noncanon":
            X = torch.randint(P, (1 << 31) - 1, (R, cols), generator=g, dtype=torch.int64).to(torch.int32)
        else:
            X = torch.randint(0, P, (R, cols), generator=g, dtype=torch.int64).to(torch.int32)
        X = X.to(dev)
        gen = {}
        for side, h in zip("abc", hi):
            nnz = torch.randint(0, h + 1, (Q,), generator=g)
            nnz[:3] = torch.tensor([0, h, max(0, h - 1)])[: min(3, Q)]
            qq = torch.repeat_interleave(torch.arange(Q), nnz)
            jj = torch.randint(0, R, (qq.numel(),), generator=g)
            if mode == "pm1":
                cc = torch.full((qq.numel(),), P - 1, dtype=torch.int64)
                consts = torch.full((Q,), P - 1, dtype=torch.int64)
            else:
                cc = torch.randint(0, P, (qq.numel(),), generator=g)
                consts = torch.randint(0, P, (Q,), generator=g)
            perm = torch.randperm(qq.numel(), generator=g)
            gen[side] = tuple(t.to(dev) for t in (qq[perm], jj[perm], cc[perm], consts))
        rho = (torch.full((D, Q), P - 1, dtype=torch.int64) if mode == "pm1"
               else torch.randint(0, P, (D, Q), generator=g, dtype=torch.int64)).to(dev)
        csr = tests_fused.csr_tables(gen, Q, dev)
        saved = (tests_fused.QUAD_V4, tests_fused.REDUCE_KERNEL, tests_fused.QUAD_SPLITS)
        try:
            tests_fused.QUAD_V4, tests_fused.REDUCE_KERNEL = False, False
            old = f.quad_general(rho, X, csr)
            torch.cuda.synchronize()
            tests_fused.QUAD_V4, tests_fused.REDUCE_KERNEL = True, True
            if splits:
                tests_fused.QUAD_SPLITS = splits
            try:
                new = f.quad_general(rho, X, dict(csr))
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001 -- the scalar kernel (old) has just run on the same inputs
                chunk = -(-Q // max(1, min(Q, splits or tests_fused.QUAD_SPLITS)))
                raise RuntimeError(f"{type(e).__name__}: {e} [quad_v4 shared {D * chunk * 4} B; the scalar quad kernel ran OK on the same inputs]") from e
        finally:
            tests_fused.QUAD_V4, tests_fused.REDUCE_KERNEL, tests_fused.QUAD_SPLITS = saved
        Xl = X.to(torch.int64)
        A = witness.apply_sparse(gen["a"][:3], gen["a"][3], Q, Xl)
        B = witness.apply_sparse(gen["b"][:3], gen["b"][3], Q, Xl)
        Cc = witness.apply_sparse(gen["c"][:3], gen["c"][3], Q, Xl)
        T = ((A * B) % P - Cc) % P
        ref = _ref_combine(rho, T)
        m_new, m_old = _mismatch(new, ref), _mismatch(old, ref)
        detail = {"new_vs_ref": m_new, "old_vs_ref": m_old, "smem_bytes": D * (-(-Q // max(1, min(Q, splits or tests_fused.QUAD_SPLITS)))) * 4}
        return m_new is None, detail
    return fn


def intt_case(D, n, mode, g, dev):
    def fn():
        f = tests_fused.fused_for(D, dev)
        if mode == "pm1":
            x = torch.full((D, n), P - 1, dtype=torch.int64)
            post = torch.full((n,), P - 1, dtype=torch.int64)
        elif mode == "zero":
            x = torch.zeros((D, n), dtype=torch.int64)
            post = torch.randint(0, P, (n,), generator=g, dtype=torch.int64)
        else:
            x = torch.randint(0, P, (D, n), generator=g, dtype=torch.int64)
            post = torch.randint(0, P, (n,), generator=g, dtype=torch.int64)
        x, post = x.to(dev), post.to(dev)
        if not f.intt_ok(x):
            return True, {"intt_ok": False, "smem_optin": f._smem_optin}
        got = f.intt_scaled(x, post)
        want = (field_intt(x) * post) % P
        m = _mismatch(got, want)
        return m is None, {"intt_ok": True, "mismatch": m}
    return fn


def ginv_case(D, n, g, dev):
    """protocol._intt_ginv (the prover's call) against the pre-change expression, on the prover's own g^-i table."""
    def fn():
        x = torch.randint(0, P, (D, n), generator=g, dtype=torch.int64).to(dev)
        x[:, :3] = P - 1
        got = protocol._intt_ginv(lambda name, fn_, t: fn_(t), x, n, dev)
        want = (field_intt(x) * protocol._ginv_powers(n, dev)) % P
        return torch.equal(got, want), {"mismatch": _mismatch(got, want)}
    return fn


def reduce_case(D, S, cols, g, dev):
    def fn():
        f = tests_fused.fused_for(D, dev)
        part = torch.full((S, D, cols), P - 1, dtype=torch.int32, device=dev)
        part[:, :, cols // 2:] = torch.randint(0, P, (S, D, cols - cols // 2), generator=g, dtype=torch.int64).to(torch.int32).to(dev)
        saved = tests_fused.REDUCE_KERNEL
        tests_fused.REDUCE_KERNEL = True
        try:
            got = f._reduce(part)
        finally:
            tests_fused.REDUCE_KERNEL = saved
        want = part.to(torch.int64).sum(0) % P
        return torch.equal(got, want), {"mismatch": _mismatch(got, want)}
    return fn


def graph_case(D, R, cols, n, g, dev):
    """First use of lincomb2 / intt_scaled inside a CUDA-graph capture (fresh FusedTests, no cached tables), then replay:
    the replay against the reference.  The prover warms up twice eagerly before capturing; this probes the worst case."""
    def fn():
        tests_fused._FT.pop((D, str(dev)), None)
        f = tests_fused.fused_for(D, dev)
        X = torch.randint(0, P, (R, cols), generator=g, dtype=torch.int64).to(torch.int32).to(dev)
        C1 = torch.randint(-3 * P, 3 * P, (D, R), generator=g, dtype=torch.int64).to(dev)
        C2 = torch.randint(0, P, (D, R // 2), generator=g, dtype=torch.int64).to(dev)
        x = torch.randint(0, P, (D, n), generator=g, dtype=torch.int64).to(dev)
        post = protocol._ginv_powers(n, dev)
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            with torch.cuda.graph(gr, stream=s):
                w, v = f.lincomb2(C1, C2, X)
                h = f.intt_scaled(x, post)
        torch.cuda.current_stream(dev).wait_stream(s)
        gr.replay()
        torch.cuda.synchronize()
        ok = (torch.equal(w, _ref_combine(C1, X)) and torch.equal(v, _ref_combine(C2, X[:R // 2]))
              and torch.equal(h, (field_intt(x) * post) % P))
        return ok, {}
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    ap.add_argument("--big", action="store_true", help="also the prover-sized shapes (R ~ 23k rows, k up to 16384)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    g = torch.Generator(device="cpu").manual_seed(0x5EDA)
    rep = Report()
    props = torch.cuda.get_device_properties(dev)
    info = {"gpu": props.name, "cc": f"{props.major}.{props.minor}", "torch": torch.__version__}
    witness.USE_FUSED = True
    assert tests_fused.fused_for(6, dev) is not None, "no fused kernels on this device"
    info["smem_optin"] = tests_fused.fused_for(6, dev)._smem_optin
    print(info, flush=True)

    # lincomb2: int64 coefficients of any sign / size, row-strided C1, every R2 edge, awkward R and cols
    for D in (1, 2, 6):
        for R, cols in ((1, 4), (3, 12), (5, 8), (1023, 64), (1024, 132), (1025, 2176), (4097, 256)):
            for R2 in sorted({0, 1, max(1, R // 2), R - 1, R} - {0} if R > 1 else {0, 1}) + [0]:
                rep.run(f"lincomb2_D{D}_R{R}_R2{R2}_c{cols}", lincomb2_case(D, R, R + 7, R2, cols, "rand", g, dev))
        rep.run(f"lincomb2_D{D}_pm1_R1037_c2176", lincomb2_case(D, 1037, 1037, 1001, 2176, "pm1", g, dev))
        rep.run(f"lincomb2_D{D}_exactR1_R1000", lincomb2_case(D, 1000, 1000, 999, 1024, "rand", g, dev, strided_view=False))
    rep.run("lincomb2_D6_noncanonX_R517", lincomb2_case(6, 517, 517, 300, 256, "noncanon", g, dev), domain=True)
    # D = 7 is the Fiat-Shamir extension degree at 2^-128 (protocol.py): 2 D chunk 4 bytes of shared C
    rep.run("lincomb2_D7_R1037_c2176", lincomb2_case(7, 1037, 1037, 1001, 2176, "rand", g, dev))
    if a.big:
        for D in (6, 7):
            for R, cols in ((22730, 4096), (22730, 16384), (40000, 16384)):
                rep.run(f"lincomb2_big_D{D}_R{R}_c{cols}", lincomb2_case(D, R, R + 100, R // 3, cols, "dev", g, dev))

    # quad_v4 + reduce_partial: side lengths past the fp8-ada-v3x4 maxima, empty sides, all p - 1, ragged Q
    for D in (1, 6):
        for Q, R, cols, hi in ((1, 17, 4, (5, 5, 5)), (5, 33, 8, (0, 3, 1)), (127, 400, 64, (70, 45, 13)),
                               (129, 400, 4096, (44, 30, 8)), (4097, 900, 256, (9, 7, 5))):
            rep.run(f"quad_D{D}_Q{Q}_c{cols}_rand", quad_case(D, Q, R, cols, hi, "rand", g, dev))
            rep.run(f"quad_D{D}_Q{Q}_c{cols}_pm1", quad_case(D, Q, R, cols, hi, "pm1", g, dev))
        for S in (1, 3, 1000):
            rep.run(f"quad_D{D}_Q301_splits{S}", quad_case(D, 301, 700, 512, (45, 31, 9), "rand", g, dev, splits=S))
    rep.run("quad_D6_noncanonX", quad_case(6, 129, 400, 256, (44, 30, 8), "noncanon", g, dev), domain=True)
    rep.run("quad_D7_Q301", quad_case(7, 301, 700, 512, (45, 31, 9), "rand", g, dev))
    if a.big:
        for D, Q in ((6, 100_000), (6, 300_000), (7, 250_000)):
            rep.run(f"quad_big_D{D}_Q{Q}", quad_case(D, Q, 2000, 64, (3, 3, 2), "rand", g, dev))

    # intt_rows: every power of two the device's opt-in shared memory admits (and the first it does not)
    for D in (1, 6):
        n = 2
        while n <= (1 << 17):
            for mode in ("rand", "pm1") + (("zero",) if n <= 64 else ()):
                rep.run(f"intt_D{D}_n{n}_{mode}", intt_case(D, n, mode, g, dev))
            n <<= 1
    for n in (1024, 8192, 16384, 32768):
        rep.run(f"intt_ginv_protocol_D6_n{n}", ginv_case(6, n, g, dev))
    for S in (1, 2, 128, 512):
        rep.run(f"reduce_partial_S{S}", reduce_case(6, S, 1000, g, dev))
    rep.run("graph_first_use_in_capture", graph_case(6, 1037, 2176, 16384, g, dev))

    verdict = [k for k, c in rep.cases.items() if not c["ok"] and not c["domain"]]
    out = {"info": info, "cases": rep.cases, "failed": verdict, "verdict": "PASS" if not verdict else "FAIL"}
    print("failed:", verdict)
    print("red-team verdict:", out["verdict"])
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    return 0 if not verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
