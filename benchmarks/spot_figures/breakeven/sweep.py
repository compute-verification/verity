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
from dataclasses import dataclass, field, replace
from math import exp, log
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levers import (MACHINE_GPUS, WEIGHTS, bound, floor_bits, fuse_reads, make_line, priced_strata,
                    scale_proving, tb)

from common import SCOPES, c, sc, tc
import reprice

REPORT = (0.01, 0.05, 0.10)
#: each cut's prover busy time against its end-to-end reading on the same sessions, (marginal, fixed a statement): the
#: per-expert cut's 3.29e8 of 5.53e8 marginal and 3.95e8 of 1.70e9 at full statements, the registered cut's 1.3e10 of
#: 1.4e10 and 2.4e10 of 1.3e11; the fixed ratio is (full - marginal) busy over (full - marginal) end to end
BUSY = {"per_expert": (3.29 / 5.53, (3.95 - 3.29) / (17.0 - 5.53)), "tokens": (1.3 / 1.4, (2.4 - 1.3) / (13.0 - 1.4))}
#: GLM-5.3's serve taxes from the serve_tax card, and their parts: the certified tick, the edge snapshot, the top-k
TAXES = tc.taxes(tc.spot_cards.load("serve_tax", "GLM-5.3"), ("tokens", "per_expert"))
PARTS = {p["part"]: p["share"] for p in TAXES["tokens"]["parts"]}
EDGE_PAUSE = tc.EDGE_PAUSE[0]
WINDOW = c["window"].value
REPLAY_FACTOR = c["replay_factor"].value
COMMIT_WRITE = reprice.COMMIT_WRITE
#: how the main line prices the fused per-expert blocks (`reprice.price_fused`)
FUSED = "h200"
#: the FLOPs the 128k plan's prefix-cache hits skip, over the FLOPs it computes, served with caching on
PREFIX_OFF = 0.1386
#: advice a machine's window when every prefix-cache hit is sent whole: the codec's hit term, a 2k served trace's
#: full-window mean
HITS_WINDOW_BITS = 5181.0
#: compiling one sampled request's Programs on 8 ranks, core-hours, by 135 s a Program a rank over a 256-step span
COMPILE_CORE_H = (3.1, 3.8)
TAX_ONE_STAGE = tc.taxes(tc.spot_cards.load("serve_tax", "GLM-5.3"), ("every_block",))["every_block"]["tax"]


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
    busy: bool = False
    labels: tuple[str, ...] = field(default_factory=tuple)
    cache: bool = False
    overlap: str = ""
    prover: float = 1.0
    full: bool = False
    host: float = 1.0
    plain: bool = False
    ctx: str = ""
    form: str = ""
    commit_write: bool = False
    fused: str = ""
    verify: float = 0.0
    verify_expert: float | None = None
    verify_other: float | None = None
    server: bool = False
    k4: bool = False
    front: bool = False
    quant_floor: bool = True
    column: str = "fast"
    tmpl: bool = False
    fused_busy: float | None = None
    advice_window_bits: float = 0.0
    prefix_off: bool = False
    compile_core_h: float = 0.0

    def tax(self) -> float:
        """The serve tax: the certified tick and the top-k always; the edge snapshot's pause over the replay block's
        length, or none when the replay reads the edge's state live; every full K/V page hashed when written if
        `commit_write`; the prefill a prefix-cache hit would have skipped if `prefix_off`.  The one-stage design's
        is the every-block line's, hashing every block's outputs."""
        if self.form == "one-stage":
            return TAX_ONE_STAGE
        edge = 0.0 if self.live else EDGE_PAUSE / (self.edge_s or WINDOW)
        return (PARTS["certified-tick rule"] + PARTS["deterministic top-k"] + edge
                + (COMMIT_WRITE if self.commit_write else 0.0) + (PREFIX_OFF if self.prefix_off else 0.0))


def build(case: Case):
    strata = None
    per_type = (case.cache or case.overlap or case.prover != 1.0 or case.host != 1.0 or case.plain or case.verify
                or case.server)
    if case.fuse or case.marginal != 1.0 or case.cap != 1.0 or case.fixed != 1.0 or case.busy or per_type:
        strata = priced_strata(case.line)
        if case.fuse:
            strata = fuse_reads(strata, fixed="sum" if case.fixed_sum or case.fused else "read")
            if case.fused:
                strata = reprice.price_fused(strata, case.fused)
        if case.server:
            strata = reprice.apply_server(strata, prover=case.prover, k4=case.k4, front=case.front,
                                          quant_floor=case.quant_floor, host=case.host, plain=case.plain,
                                          verify=case.verify, column=case.column, tmpl=case.tmpl,
                                          fused_busy=case.fused_busy, verify_other=case.verify_other)
        elif per_type:
            strata = reprice.apply(strata, cache=case.cache, overlap=case.overlap, prover=case.prover,
                                   full=case.full, host=case.host, plain=case.plain, verify=case.verify,
                                   verify_expert=case.verify_expert, verify_other=case.verify_other)
        bm, bf = BUSY[case.line] if case.busy else (1.0, 1.0)
        strata = scale_proving(strata, marginal=case.marginal * bm, fixed=case.fixed * bf, cap=case.cap)
    L = make_line(case.line, strata, replay_gpu_s=case.replay, registration=case.registration)
    if case.ctx:
        L = reprice.workload(L, {"c343": reprice.C343, "c343-served": reprice.C343_SERVED}[case.ctx])
    if case.form == "one-stage":
        L = (*L[:7], (L[7] if len(L) > 7 else {}) | {"point": "every_block"})
    return L


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
    """A case's bounds at REPORT's betas and its break-even betas, the target each weight's bits less the advice it
    charges over the scope's machine windows (none where the advice alone reaches it); with `compile_core_h`,
    compiling the requests the FP4 break-even's plan samples, a share of the scope's compute at CORE_PER_GPU."""
    name, case, scope = job
    L = build(case)
    s = SCOPES[scope]
    advice = case.advice_window_bits * s.gpu_seconds / (MACHINE_GPUS * WINDOW)
    out = {"case": name, "scope": scope, "tax": case.tax(), "h": L[2][2] * case.scale, "advice_bits": advice,
           "bound": {str(b): bound(L, b, scope, case.scale) for b in REPORT},
           "floor": floor_bits(L, 0.01, scope) if case.scale == 1.0 else bound(L, 0.01, scope, 1e-12),
           "beta": {w: solve(L, t - advice, scope, case.scale) if t > advice else None for w, t in WEIGHTS.items()}}
    b4 = out["beta"]["FP4"]
    if case.compile_core_h and b4 is not None:
        sampled = bound(L, b4, scope, case.scale, full=True)[1]["plan"]["replay_sample"]
        out["compile"] = {"sampled": sampled, "share": {str(h): sampled * h * 3600 / reprice.CORE_PER_GPU
                                                        / s.gpu_seconds for h in (*COMPILE_CORE_H, 14.0)}}
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
    busy = {
        "P-busy": ("P", {}), "A-busy": ("A", {}), "F-busy": ("F", {}),
        "F+busy+fast2": ("F", dict(scale=0.5)), "F+busy+fast4": ("F", dict(scale=0.25)),
        "F+busy+hash2": ("F", dict(marginal=hash2)), "F+busy+hash4": ("F", dict(marginal=hash4)),
        "F+busy+fast2+hash2": ("F", dict(scale=0.5, marginal=hash2)),
        "F+busy+fast2+hash4": ("F", dict(scale=0.5, marginal=hash4)),
        "F+busy+fast4+hash4": ("F", dict(scale=0.25, marginal=hash4)),
        "F+busy+live10": ("F", dict(**live10)),
        "F+busy+fast2+live10": ("F", dict(scale=0.5, **live10)),
        "F+busy+fast2+hash4+live10": ("F", dict(scale=0.5, marginal=hash4, **live10)),
        "F+busy+fast4+hash4+live10": ("F", dict(scale=0.25, marginal=hash4, **live10)),
        "A+busy+fast4": ("A", dict(scale=0.25)),
        "A+busy+fast4+hash4": ("A", dict(scale=0.25, marginal=hash4)),
        "A+busy+fast4+hash4+live10": ("A", dict(scale=0.25, marginal=hash4, **live10)),
    }
    for name, (b, kw) in busy.items():
        out[name] = Case(**{**bases[b].__dict__, "busy": True, **kw})
    hash4s75 = 1 - 0.75 * (1 - 1 / 4)
    life = dict(replay=replay_gpu_s(0.0, 90), edge_s=1.0, live=True)
    more = {
        "F+busy+life": dict(**life),
        "F+busy+fast2+hash2+life": dict(scale=0.5, marginal=hash2, **life),
        "F+busy+fast2+hash4+life": dict(scale=0.5, marginal=hash4, **life),
        "F+busy+fast4+life": dict(scale=0.25, **life),
        "F+busy+fast4+hash4+life": dict(scale=0.25, marginal=hash4, **life),
        "F+busy+fast4+hash4s75": dict(scale=0.25, marginal=hash4s75),
        "F+busy+fast4+hash4s75+life": dict(scale=0.25, marginal=hash4s75, **life),
        "F+busy+fast8": dict(scale=0.125),
        "F+busy+fast8+hash4": dict(scale=0.125, marginal=hash4),
        "F+busy+fast8+hash4+life": dict(scale=0.125, marginal=hash4, **life),
        "F+busy+fast8+hash4s75+life": dict(scale=0.125, marginal=hash4s75, **life),
    }
    for name, kw in more.items():
        out[name] = Case(**{**bases["F"].__dict__, "busy": True, **kw})
    return out | repriced()


def repriced() -> dict[str, Case]:
    """(A′) at 128k over vLLM's 128k serve plan, each measured or modeled lever added in turn (`R0`..`R6x4`, `R6` and
    `R6x4` adding the Lean verifier's CPU), the extras on the main line's last two steps, the one-stage design on the
    same blocks, and the plan as served beside it."""
    hash4 = 1 - 0.5 * (1 - 1 / 4)
    base = dict(line="per_expert", ctx="c343", replay=reprice.BLOCKS_128K)
    life = dict(replay=reprice.LIFETIME_128K, live=True, commit_write=True)
    fz = dict(fuse=True, fused=FUSED)
    steps = {"R0": {}, "R1": fz, "R2": fz | dict(cache=True), "R3": fz | dict(cache=True, overlap="experts")}
    steps["R4"] = steps["R3"] | dict(prover=3.5)
    steps["R4x4"] = steps["R3"] | dict(prover=4.0)
    steps["R5"] = steps["R4"] | life
    steps["R5x4"] = steps["R4x4"] | life
    steps["R6"] = steps["R5"] | dict(verify=1.0)
    steps["R6x4"] = steps["R5x4"] | dict(verify=1.0)
    extras = {"all": dict(overlap="all"), "4perH200": dict(full=True), "host": dict(host=3.5),
              "hash4": dict(marginal=hash4), "plain": dict(plain=True),
              "all+4perH200": dict(overlap="all", full=True),
              "all+4perH200+host": dict(overlap="all", full=True, host=3.5),
              "all+4perH200+host+plain": dict(overlap="all", full=True, host=3.5, plain=True),
              "all+4perH200+host+hash4": dict(overlap="all", full=True, host=3.5, marginal=hash4)}
    out = {n: Case(**(base | kw)) for n, kw in steps.items()}
    for s in ("R6", "R6x4"):
        for e, kw in extras.items():
            out[f"{s}+{e}"] = Case(**(base | steps[s] | kw))
    for s in ("R6x4+all+4perH200+host", "R6x4+all+4perH200+host+hash4"):
        out[f"{s}+vtmpl"] = replace(out[s], verify=reprice.VERIFY_TEMPLATED)
    for s in ("R6x4", "R6x4+all+4perH200"):
        out[f"{s}+vflat"] = replace(out[s], verify=0.4)
        out[f"{s}+vtmpl"] = replace(out[s], verify=reprice.VERIFY_TEMPLATED)
        out[f"{s}+vmedian"] = replace(out[s], verify_expert=reprice.VERIFY_DEFAULT)
        out[f"{s}+vfloor"] = replace(out[s], verify_other=reprice.VERIFY_FIXED)
    for s in ("R1", "R6x4"):
        out[f"{s}-1s"] = replace(out[s], form="one-stage")
        out[f"{s}-cpuratio"] = replace(out[s], fused="cpu")
    out["R6x4-served"] = replace(out["R6x4"], ctx="c343-served")
    return out | served_line()


def served_line() -> dict[str, Case]:
    """(A′) at 128k priced as the long-running proof server proves it (`N0`..`N5`): today's measured server prices,
    main's verifier and the quant K6144 at its floor, then each lever in turn, each type's overlap set by its device's
    busy share; the extras on `N5`, and the advice charged two ways."""
    hash4 = 1 - 0.5 * (1 - 1 / 4)
    life = dict(replay=reprice.LIFETIME_128K, live=True, commit_write=True)
    n0 = dict(line="per_expert", ctx="c343", replay=reprice.BLOCKS_128K, fuse=True, fused=FUSED, server=True,
              verify=1.0, column="today", compile_core_h=1.0)
    steps = {"N0": n0}
    steps["N1"] = steps["N0"] | dict(column="fast")
    steps["N2"] = steps["N1"] | life
    steps["N3"] = steps["N2"] | dict(k4=True)
    steps["N4"] = steps["N3"] | dict(front=True)
    steps["N5"] = steps["N4"] | dict(prover=4.0)
    out = {n: Case(**kw) for n, kw in steps.items()}
    extras = {"host": dict(host=3.5), "hash4": dict(marginal=hash4), "plain": dict(plain=True),
              "tmpl": dict(tmpl=True), "vfloor": dict(verify_other=reprice.VERIFY_COLUMNS["fast"][1]),
              "fusedfront": dict(fused_busy=0.55), "card": dict(quant_floor=False),
              "host+tmpl": dict(host=3.5, tmpl=True), "host+hash4+tmpl": dict(host=3.5, marginal=hash4, tmpl=True),
              "hits": dict(advice_window_bits=HITS_WINDOW_BITS), "prefixoff": dict(prefix_off=True),
              "host+tmpl+prefixoff": dict(host=3.5, tmpl=True, prefix_off=True)}
    for e, kw in extras.items():
        out[f"N5+{e}"] = Case(**(steps["N5"] | kw))
    for s in ("N2", "N3"):
        out[f"{s}+hits"] = Case(**(steps[s] | dict(advice_window_bits=HITS_WINDOW_BITS)))
        out[f"{s}+prefixoff"] = Case(**(steps[s] | dict(prefix_off=True)))
    out["N2+tmpl"] = Case(**(steps["N2"] | dict(tmpl=True)))
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
