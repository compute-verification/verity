"""The break-even surface: for each lever alone and combined, the beta at which a request-grain line's exfiltration
bound reaches GLM-5.3's FP8 and FP4 weights, its bound at the beta of `REPORT`, and its floor (proving at nothing),
each through total_cost._point as Figure 1 computes it.

    uv run python benchmarks/spot_figures/breakeven/sweep.py --out DIR [--workers N] [--scopes all] [--only NAME...]

A case is a base line ((A) `tokens`, or (A′) `per_expert`, optionally with each table read fused into its consumer
GEMM) and its levers: all proving scaled (`scale`, a faster prover or another h), the marginal scaled (`marginal`,
in-proof hashing cut at a share), each statement's cap in blocks multiplied (`cap`) or its fixed cost scaled
(`fixed`), the registration statements share within (`registration`), and the replay a sampled request costs in
GPU-seconds (`replay`; default the figure's, a machine's 600 s block).  `tax` is its serve tax, outside beta.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from math import exp, log
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levers import (MACHINE_GPUS, WEIGHTS, bound, floor_bits, fuse_reads, make_line, priced_strata,  # noqa: E402
                    scale_proving, tb)

from common import SCOPES, c, sc, tc  # noqa: E402

REPORT = (0.01, 0.05, 0.10)
#: GLM-5.3's serve taxes from the serve_tax card, and their parts: the certified tick, the edge snapshot, the top-k
TAXES = tc.taxes(tc.spot_cards.load("serve_tax", "GLM-5.3"), ("tokens", "per_expert"))
PARTS = {p["part"]: p["share"] for p in TAXES["tokens"]["parts"]}
EDGE_PAUSE = tc.EDGE_PAUSE[0]
WINDOW = c["window"].value
REPLAY_FACTOR = c["replay_factor"].value


@dataclass(frozen=True)
class Case:
    line: str = "tokens"
    fuse: bool = False
    fixed_sum: bool = False
    scale: float = 1.0
    marginal: float = 1.0
    cap: float = 1.0
    fixed: float = 1.0
    registration: str | None = None
    replay: float | None = None
    edge_s: float | None = None
    live: bool = False
    labels: tuple[str, ...] = field(default_factory=tuple)

    def tax(self) -> float:
        """The serve tax: the certified tick and the top-k always; the edge snapshot's pause over the replay block's
        length, or none when the replay reads the edge's state live."""
        edge = 0.0 if self.live else EDGE_PAUSE / (self.edge_s or WINDOW)
        return PARTS["certified-tick rule"] + PARTS["deterministic top-k"] + edge


def build(case: Case):
    strata = None
    if case.fuse or case.marginal != 1.0 or case.cap != 1.0 or case.fixed != 1.0:
        strata = priced_strata(case.line)
        if case.fuse:
            strata = fuse_reads(strata, fixed="sum" if case.fixed_sum else "read")
        strata = scale_proving(strata, marginal=case.marginal, fixed=case.fixed, cap=case.cap)
    return make_line(case.line, strata, replay_gpu_s=case.replay, registration=case.registration)


def solve(L, target: float, scope: str, scale: float, lo: float = 1e-5, hi: float = 1e4, rel: float = 0.005):
    """The least beta at which the bound reaches `target` bits: regula falsi (Illinois) on ln bound against ln beta,
    which is near-linear; None if the bound is above the target at `hi`."""
    def f(x: float) -> float:
        return log(max(bound(L, exp(x), scope, scale), 1.0)) - log(target)

    xa, fa = log(0.01), None
    fa = f(xa)
    xb = min(max(xa + fa, log(lo)), log(hi))
    fb = f(xb)
    tries = 0
    while fa * fb > 0 and tries < 8:
        slope = (fb - fa) / (xb - xa) if xb != xa else -1.0
        slope = min(slope, -0.05)
        step = max(min(-fb / slope, 4.0), -4.0)
        xa, fa = xb, fb
        xb = min(max(xb + step, log(lo)), log(hi))
        if xb == xa:
            break
        fb = f(xb)
        tries += 1
    if fa * fb > 0:
        return None if fb > 0 else exp(xb)
    side = 0
    while abs(xb - xa) > log(1 + rel):
        xm = xb - fb * (xb - xa) / (fb - fa)
        fm = f(xm)
        if fm * fb < 0:
            xa, fa = xb, fb
            side = 0
        else:
            fa = fa / 2 if side == 1 else fa
            side = 1
        xb, fb = xm, fm
        if abs(fm) < log(1 + rel / 4):
            break
    return exp(xb)


def run_case(job):
    name, case, scope = job
    L = build(case)
    out = {"case": name, "scope": scope, "tax": case.tax(), "h": L[2][2] * case.scale,
           "bound": {str(b): bound(L, b, scope, case.scale) for b in REPORT},
           "floor": floor_bits(L, 0.01, scope) if case.scale == 1.0 else bound(L, 0.01, scope, 1e-12),
           "beta": {w: solve(L, t, scope, case.scale) for w, t in WEIGHTS.items()}}
    return out


def replay_gpu_s(edge_s: float, lifetime: float) -> float:
    """A sampled request's replay: every machine block its lifetime spans, from the edge before it arrives, 1 + T/L
    blocks of L seconds on average, at the replay card's factor."""
    return REPLAY_FACTOR * MACHINE_GPUS * (edge_s + lifetime)


def cases() -> dict[str, Case]:
    """Each lever alone on (A), (A′) and (A′) fused, then combined."""
    out: dict[str, Case] = {}
    bases = {"A": Case("tokens"), "P": Case("per_expert"), "F": Case("per_expert", fuse=True)}
    for b, base in bases.items():
        out[b] = base
    out["F-fixedsum"] = Case("per_expert", fuse=True, fixed_sum=True)
    card = 5.53e8 / 6.869e8
    out["P-card"] = Case("per_expert", scale=card)
    out["F-card"] = Case("per_expert", fuse=True, scale=card)
    for b in ("A", "F"):
        base = bases[b]
        for k in (2, 4, 8):
            out[f"{b}-cap{k}"] = Case(**{**base.__dict__, "cap": float(k)})
            out[f"{b}-cap{k}-dc"] = Case(**{**base.__dict__, "cap": float(k), "registration": "datacenter"})
        out[f"{b}-nofixed"] = Case(**{**base.__dict__, "fixed": 0.0})
        out[f"{b}-dc"] = Case(**{**base.__dict__, "registration": "datacenter"})
        for s in (0.25, 0.5, 0.75):
            for f in (2, 4):
                out[f"{b}-hash{f}-s{int(s * 100)}"] = Case(**{**base.__dict__, "marginal": 1 - s * (1 - 1 / f)})
        for k in (2, 4):
            out[f"{b}-fast{k}"] = Case(**{**base.__dict__, "scale": 1 / k})
        for edge in (600, 60, 10):
            out[f"{b}-edge{edge}-fig"] = Case(**{**base.__dict__, "replay": REPLAY_FACTOR * MACHINE_GPUS * edge,
                                               "edge_s": float(edge)})
            for life in (30, 90, 300):
                out[f"{b}-edge{edge}-T{life}"] = Case(**{**base.__dict__, "replay": replay_gpu_s(edge, life),
                                                       "edge_s": float(edge)})
                out[f"{b}-edge{edge}-T{life}-live"] = Case(**{**base.__dict__, "replay": replay_gpu_s(edge, life),
                                                            "edge_s": float(edge), "live": True})
        for life in (30, 90, 300):
            out[f"{b}-edge0-T{life}-live"] = Case(**{**base.__dict__, "replay": replay_gpu_s(0.0, life),
                                                   "edge_s": 1.0, "live": True})
    # combinations, on the fused cut
    hash4 = 1 - 0.5 * (1 - 1 / 4)
    hash2 = 1 - 0.5 * (1 - 1 / 2)
    live10 = dict(replay=replay_gpu_s(10, 90), edge_s=10.0, live=True)
    combos = {
        "F+nofixed": dict(fixed=0.0),
        "F+cap4": dict(cap=4.0),
        "F+fast2": dict(scale=0.5),
        "F+fast4": dict(scale=0.25),
        "F+cap4+fast2": dict(cap=4.0, scale=0.5),
        "F+cap4+fast4": dict(cap=4.0, scale=0.25),
        "F+nofixed+fast2": dict(fixed=0.0, scale=0.5),
        "F+nofixed+fast4": dict(fixed=0.0, scale=0.25),
        "F+cap4+hash4": dict(cap=4.0, marginal=hash4),
        "F+cap4+fast2+hash2": dict(cap=4.0, scale=0.5, marginal=hash2),
        "F+cap4+fast2+hash4": dict(cap=4.0, scale=0.5, marginal=hash4),
        "F+cap4+fast4+hash4": dict(cap=4.0, scale=0.25, marginal=hash4),
        "F+nofixed+fast4+hash4": dict(fixed=0.0, scale=0.25, marginal=hash4),
        "F+cap4+fast4+hash4+live10": dict(cap=4.0, scale=0.25, marginal=hash4, **live10),
        "F+nofixed+fast4+hash4+live10": dict(fixed=0.0, scale=0.25, marginal=hash4, **live10),
        "F+cap8+fast4+hash4+live10": dict(cap=8.0, scale=0.25, marginal=hash4, **live10),
        "F+live10": dict(**live10),
        "F+fast4+live10": dict(scale=0.25, **live10),
    }
    for name, kw in combos.items():
        out[name] = Case(**{**bases["F"].__dict__, **kw})
    a_combos = {
        "A+nofixed": dict(fixed=0.0),
        "A+cap4+fast4": dict(cap=4.0, scale=0.25),
        "A+cap4+fast4+hash4": dict(cap=4.0, scale=0.25, marginal=hash4),
        "A+nofixed+fast4+hash4+live10": dict(fixed=0.0, scale=0.25, marginal=hash4, **live10),
    }
    for name, kw in a_combos.items():
        out[name] = Case(**{**bases["A"].__dict__, **kw})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scopes", nargs="*", default=["world week"])
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    scopes = list(SCOPES) if args.scopes == ["all"] else args.scopes
    every = cases()
    chosen = {n: k for n, k in every.items() if not args.only or n in args.only}
    jobs = [(n, k, s) for s in scopes for n, k in chosen.items()]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(run_case, jobs):
            rows.append(r)
            print(json.dumps(r), flush=True)
    meta = {"weights_bits": WEIGHTS, "report_betas": REPORT, "taxes": {k: v["tax"] for k, v in TAXES.items()},
            "parts": PARTS, "cases": {n: k.__dict__ | {"tax": k.tax()} for n, k in chosen.items()},
            "proof_overhead_card": sc["proof_overhead"].get("art")}
    (args.out / "breakeven_surface.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    lines = ["case | scope | h | tax | bound at 1% | at 5% | at 10% | floor | beta FP8 | beta FP4"]
    for r in rows:
        b = r["beta"]
        lines.append(" | ".join([r["case"], r["scope"], f"{r['h']:.3g}", f"{r['tax']:.2%}"]
                                + [f"{tb(r['bound'][str(x)]):.4g} TB" for x in REPORT] + [f"{tb(r['floor']) * 1e3:.4g} GB"]
                                + ["none" if b[w] is None else f"{b[w]:.4g}" for w in ("FP8", "FP4")]))
    (args.out / "breakeven_surface.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
