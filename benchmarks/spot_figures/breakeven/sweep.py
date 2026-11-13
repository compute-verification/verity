"""The break-even surface: for each lever alone and combined, the beta at which a request-grain line's exfiltration
bound reaches GLM-5.3's FP8 and FP4 weights, its bound at the beta of `REPORT`, and its floor (proving at nothing),
each through total_cost._point as Figure 1 computes it.

    uv run python benchmarks/spot_figures/breakeven/sweep.py --out DIR [--workers N] [--scopes all] [--only NAME...]
                                                             [--steady ROWS.json]

A case is a base line ((A) `tokens`, or (A′) `per_expert`, optionally with each table read fused into its consumer
GEMM) and its levers: all proving scaled (`scale`, a faster prover or another h), the marginal scaled (`marginal`,
each block's seconds), each statement's cap in blocks multiplied (`cap`) or its fixed cost scaled
(`fixed`), the registration statements share within (`registration`), and the replay a sampled request costs in
GPU-seconds (`replay`; default the figure's, a machine's 600 s block).  `tax` is its serve tax, outside beta.

`--steady` prices each proof-server case's block types from the steady-state rows of a file where it holds one
(`reprice.steady`), in place of the server prices and the statement cache's factors, and each such case's row then
says what share of its proving each basis prices (`basis`).
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
                    scale_proving, split_cols, tb)

from common import SCOPES, c, sc, tc
import reprice

REPORT = (0.01, 0.05, 0.10)
#: each cut's prover busy time against its end-to-end reading on the same sessions, (marginal, fixed a statement): the
#: per-expert cut's 3.29e8 of 5.53e8 marginal and 3.95e8 of 1.70e9 at full statements, the registered cut's 1.3e10 of
#: 1.4e10 and 2.4e10 of 1.3e11; the fixed ratio is (full - marginal) busy over (full - marginal) end to end
BUSY = {"per_expert": (3.29 / 5.53, (3.95 - 3.29) / (17.0 - 5.53)), "tokens": (1.3 / 1.4, (2.4 - 1.3) / (13.0 - 1.4))}
#: GLM-5.3's serve taxes from the serve_tax card, and (A)'s parts by name
TAXES = tc.taxes(tc.spot_cards.load("serve_tax", "GLM-5.3"), ("tokens", "per_expert"))
PARTS = {p["part"]: p["share"] for p in TAXES["tokens"]["parts"]}
#: the parts every two-stage case pays; the edge snapshot (`tc.EDGE`) is the one a case can go without
SERVED = ("certified-tick rule", "step record", "two-stage boundary commitment", "history tables",
          "deterministic top-k")
if set(PARTS) != {*SERVED, tc.EDGE}:
    raise ValueError(f"the serve_tax card's (A) has the parts {sorted(PARTS)} and Case.tax reads "
                     f"{sorted({*SERVED, tc.EDGE})}: name each part of the card in SERVED (or as total_cost.EDGE), "
                     "so that a case's serve tax is the sum of the card's parts")
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
#: the per-window pool unit, its pool side (CacheStep_v3 on MemCheck_v1 and BlockKey_v1) and its compile window: a
#: window's ANDs at the chat class, which the busiest window sets, with every lookup walking and with the head-of-line
#: memo; as built, which pays for every position, with the ragged token row alone (estimated), and with the ragged row
#: and the request numbers carried in the slot row (`v5`, built)
POOL_WINDOW_ANDS = {"built walked": 3.05e11, "built memo": 2.95e11, "ragged walked": 2.25e11, "ragged memo": 2.15e11,
                    "v5 walked": 1.24e11, "v5 memo": 1.10e11}
#: the share of the FLOPs the 128k plan computes that a pool per window recomputes, the hits on blocks only an earlier
#: window computed, by window seconds: the expectation over a uniformly random phase from the 128k trace's reuse
#: distances, all 60-300 s and 189 s on average a saved FLOP, so 189 s / W of the 13.86% its hits save
POOL_WINDOW_LOST = {600: 0.04367, 1200: 0.02184, 1800: 0.01456, 3600: 0.00728}
#: the leverage factors priced (`Case.leverage`): how many times its charged bits every incorrect proof block tells a
#: decoder about the weights, and the total share the largest one is solved at
LEVERAGE = (1.0, 2.0, 4.0, 8.0, 16.0, 64.0)
LEVERAGE_SHARE = 0.10
#: quant K6144's cold session, 188.4 s at 2 and at 64 blocks with the process's start, less its warm fixed, 8.56 s: what
#: a prover process pays on its first statement of the type, the cache hitting every later one
COLD_6144 = 188.4 - 8.56
#: GPU-seconds an AND under --zk on the H200: 1.0e12 ANDs take a tenth of a machine's window
ZK_GPU_S_PER_AND = 0.10 * MACHINE_GPUS * WINDOW / 1.0e12
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
    quant6144: float | None = None
    k_gain: str = "measured"
    dense_k: int = 2
    verify_cut: bool = False
    cold: float = 0.0
    column: str = "fast"
    tmpl: bool | str = False
    fused_busy: float | None = None
    advice_window_bits: float = 0.0
    prefix_off: bool = False
    pool_window_s: int = 0
    compile_core_h: float = 0.0
    budget_form: str = ""
    split: tuple[tuple[str, int], ...] = ()
    leverage: tuple[float, ...] = ()
    steady: str = ""

    def tax(self) -> float:
        """The serve tax, a sum of the serve_tax card's named parts: the certified tick, the step record, the two-stage
        boundary commitment, the history tables and the top-k always (`SERVED`); the edge snapshot's pause over the
        replay block's length (the card's part at the census window), or none when the replay reads the edge's state
        live; every full K/V page hashed when written if `commit_write`; the prefill a prefix-cache hit would have
        skipped if `prefix_off`, or only the hits a pool per window of `pool_window_s` loses.  The one-stage design's
        is the every-block line's, hashing every block's outputs."""
        if self.form == "one-stage":
            return TAX_ONE_STAGE
        edge = 0.0 if self.live else (EDGE_PAUSE / self.edge_s if self.edge_s else PARTS[tc.EDGE])
        return (sum(PARTS[p] for p in SERVED) + edge
                + (COMMIT_WRITE if self.commit_write else 0.0) + (PREFIX_OFF if self.prefix_off else 0.0)
                + (POOL_WINDOW_LOST[self.pool_window_s] if self.pool_window_s else 0.0))


def build(case: Case):
    assert case.server or not case.steady, "a steady-state row is the proof server's price: the case needs `server`"
    strata = None
    per_type = (case.cache or case.overlap or case.prover != 1.0 or case.host != 1.0 or case.plain or case.verify
                or case.server)
    if (case.fuse or case.marginal != 1.0 or case.cap != 1.0 or case.fixed != 1.0 or case.busy or per_type
            or case.split):
        strata = priced_strata(case.line)
        if case.fuse:
            strata = fuse_reads(strata, fixed="sum" if case.fixed_sum or case.fused else "read")
            if case.fused:
                strata = reprice.price_fused(strata, case.fused)
        if case.server:
            strata = reprice.apply_server(strata, prover=case.prover, k4=case.k4, front=case.front,
                                          quant6144=case.quant6144, host=case.host, plain=case.plain,
                                          verify=case.verify, column=case.column, tmpl=case.tmpl,
                                          fused_busy=case.fused_busy, verify_other=case.verify_other,
                                          k_gain=case.k_gain, dense_k=case.dense_k, verify_cut=case.verify_cut,
                                          cold=case.cold, steady=reprice.steady(case.steady) if case.steady else None)
        elif per_type:
            strata = reprice.apply(strata, cache=case.cache, overlap=case.overlap, prover=case.prover,
                                   full=case.full, host=case.host, plain=case.plain, verify=case.verify,
                                   verify_expert=case.verify_expert, verify_other=case.verify_other)
        bm, bf = BUSY[case.line] if case.busy else (1.0, 1.0)
        strata = scale_proving(strata, marginal=case.marginal * bm, fixed=case.fixed * bf, cap=case.cap)
        for pattern, n in case.split:
            strata = split_cols(strata, pattern, n)
    L = make_line(case.line, strata, replay_gpu_s=case.replay, registration=case.registration)
    if case.ctx:
        L = reprice.workload(L, {"c343": reprice.C343, "c343-served": reprice.C343_SERVED}[case.ctx])
    if case.form == "one-stage":
        L = (*L[:7], (L[7] if len(L) > 7 else {}) | {"point": "every_block"})
    if case.budget_form:
        L = (*L[:7], (L[7] if len(L) > 7 else {}) | {"budget_form": case.budget_form})
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
    if case.server:
        out["basis"] = reprice.basis_shares(L[1]["partition"].value["strata"])
    b4 = out["beta"]["FP4"]
    if case.compile_core_h and b4 is not None:
        pt = bound(L, b4, scope, case.scale, full=True)[1]
        sampled = pt["plan"]["replay_sample"]
        out["compile"] = {"sampled": sampled, "share": {str(h): sampled * h * 3600 / reprice.CORE_PER_GPU
                                                        / s.gpu_seconds for h in (*COMPILE_CORE_H, 14.0)}}
        out["pool"] = pool(L, case, pt["registration_sample"])
        if case.leverage:
            walked = {k: v for k, v in out["pool"]["every"].items() if k.endswith("walked")}
            out["leverage"] = leverage(L, case, scope, WEIGHTS["FP4"] - advice, b4, walked)
    return out


def leverage(L, case, scope: str, target: float, beta1: float, pools: dict) -> dict:
    """FP4's break-even when every incorrect proof block's charged bits, its content and its location, are λ times
    as many, at each λ of `case.leverage`: the bound λ times larger, so the target over λ, with its compiling at the
    total's core-hours; and the largest λ at which the total (the serve tax, proving and verifying, compiling and the
    pool) is LEVERAGE_SHARE, with each of `pools` and with none: the target over the bound at the beta that leaves."""
    s = SCOPES[scope]
    h = COMPILE_CORE_H[-1]

    def compile_share(beta: float) -> float:
        sampled = bound(L, beta, scope, case.scale, full=True)[1]["plan"]["replay_sample"]
        return sampled * h * 3600 / reprice.CORE_PER_GPU / s.gpu_seconds

    betas = {str(x): beta1 if x == 1 else solve(L, target / x, scope, case.scale) for x in case.leverage}
    at = {}
    for name, p in {"none": 0.0, **pools}.items():
        left = beta = LEVERAGE_SHARE - case.tax() - p
        for _ in range(3):
            if beta <= 0:
                break
            beta = left - compile_share(beta)
        at[name] = target / bound(L, beta, scope, case.scale) if beta > 0 else None
    return {"share": LEVERAGE_SHARE, "compile_core_h": h, "beta": betas,
            "compile": {x: None if b is None else compile_share(b) for x, b in betas.items()}, "at": at}


def pool(L, case, windows_sampled: float) -> dict:
    """The memory-checked pool's share of compute: its ANDs at ZK_GPU_S_PER_AND plus the Lean verifier at the line's
    median measured core-seconds an AND, over a machine's window; proved every window, or only on the share of windows
    the plan samples."""
    v = reprice.verify_per_and(L[1]["partition"].value["strata"], case.column)
    every = {k: a * (ZK_GPU_S_PER_AND + v / reprice.CORE_PER_GPU) / (MACHINE_GPUS * WINDOW)
             for k, a in POOL_WINDOW_ANDS.items()}
    prove = {k: a * ZK_GPU_S_PER_AND / (MACHINE_GPUS * WINDOW) for k, a in POOL_WINDOW_ANDS.items()}
    return {"verify_core_s_per_and": v, "windows_sampled": windows_sampled, "prove": prove, "every": every,
            "sampled": {k: e * windows_sampled for k, e in every.items()}}


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
        "F+live10": dict(**live10),
        "F+fast4+live10": dict(scale=0.25, **live10),
    }
    for name, kw in combos.items():
        out[name] = Case(**{**bases["F"].__dict__, **kw})
    a_combos = {
        "A+nofixed": dict(fixed=0.0),
        "A+cap4+fast4": dict(cap=4.0, scale=0.25),
    }
    for name, kw in a_combos.items():
        out[name] = Case(**{**bases["A"].__dict__, **kw})
    busy = {
        "P-busy": ("P", {}), "A-busy": ("A", {}), "F-busy": ("F", {}),
        "F+busy+fast2": ("F", dict(scale=0.5)), "F+busy+fast4": ("F", dict(scale=0.25)),
        "F+busy+live10": ("F", dict(**live10)),
        "F+busy+fast2+live10": ("F", dict(scale=0.5, **live10)),
        "A+busy+fast4": ("A", dict(scale=0.25)),
    }
    for name, (b, kw) in busy.items():
        out[name] = Case(**{**bases[b].__dict__, "busy": True, **kw})
    life = dict(replay=replay_gpu_s(0.0, 90), edge_s=1.0, live=True)
    more = {
        "F+busy+life": dict(**life),
        "F+busy+fast4+life": dict(scale=0.25, **life),
        "F+busy+fast8": dict(scale=0.125),
    }
    for name, kw in more.items():
        out[name] = Case(**{**bases["F"].__dict__, "busy": True, **kw})
    return out | repriced()


def repriced() -> dict[str, Case]:
    """(A′) at 128k over vLLM's 128k serve plan, each measured or modeled lever added in turn (`R0`..`R6x4`, `R6` and
    `R6x4` adding the Lean verifier's CPU), the extras on the main line's last two steps, the one-stage design on the
    same blocks, and the plan as served beside it."""
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
              "plain": dict(plain=True),
              "all+4perH200": dict(overlap="all", full=True),
              "all+4perH200+host": dict(overlap="all", full=True, host=3.5),
              "all+4perH200+host+plain": dict(overlap="all", full=True, host=3.5, plain=True)}
    out = {n: Case(**(base | kw)) for n, kw in steps.items()}
    for s in ("R6", "R6x4"):
        for e, kw in extras.items():
            out[f"{s}+{e}"] = Case(**(base | steps[s] | kw))
    out["R6x4+all+4perH200+host+vtmpl"] = replace(out["R6x4+all+4perH200+host"], verify=reprice.VERIFY_TEMPLATED)
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


#: the column blocks: the all-reduces and MoeSum cut 32 ways, ResidualAdd 16
COLUMN_BLOCKS = ((r"^(AllReduce|MoeSum)", 32), (r"^ResidualAdd", 16))
#: kernels' marked prover PRs over main's prover, over old main's (2.088 s), and their modeled target over main's
PROVER_MARKED = reprice.PROVE_MAIN / reprice.PROVE_MARKED
PROVER_OLD_MAIN = reprice.PROVE_OLD_MAIN / reprice.PROVE_MARKED
PROVER_GOAL = reprice.PROVE_MAIN / reprice.PROVE_TARGET


def served_line() -> dict[str, Case]:
    """(A′) at 128k priced as the long-running proof server proves it (`N0`..`N5`): today's measured server prices,
    main's verifier and the quant card, then each lever in turn; the measured levers alone (`N5m`); the extras on `N5`,
    the path from it (`P1`..`P5`), a pool per window's lost hits on `N5`, `P1` and `P5`, quant K6144's cold session
    once a prover process a day and once a window on `P1`'s, the leverage factors on `P1`'s and `P5`'s, the
    per-window budget form on `N5` and `P5`, and the advice charged two ways."""
    life = dict(replay=reprice.LIFETIME_128K, live=True, commit_write=True)
    n0 = dict(line="per_expert", ctx="c343", replay=reprice.BLOCKS_128K, fuse=True, fused=FUSED, server=True,
              verify=1.0, column="today", compile_core_h=1.0)
    steps = {"N0": n0}
    steps["N1"] = steps["N0"] | dict(column="fast")
    steps["N2"] = steps["N1"] | life
    steps["N3"] = steps["N2"] | dict(k4=True, k_gain="mps_types")
    steps["N4"] = steps["N3"] | dict(front=True)
    steps["N5"] = steps["N4"] | dict(prover=PROVER_MARKED)
    steps["N5m"] = steps["N1"] | dict(k4=True, k_gain="mps_types", prover=PROVER_MARKED)
    steps["P1"] = steps["N5"] | dict(split=COLUMN_BLOCKS)
    steps["P2"] = steps["P1"] | dict(verify_cut=True)
    steps["P3"] = steps["P2"] | dict(dense_k=4)
    steps["P4"] = steps["P3"] | dict(prover=PROVER_GOAL)
    steps["P5"] = steps["P4"] | dict(tmpl="goal")
    out = {n: Case(**kw) for n, kw in steps.items()}
    extras = {"cols": dict(split=COLUMN_BLOCKS), "flatcut": dict(verify_cut=True),
              "dense4": dict(dense_k=4), "provergoal": dict(prover=PROVER_GOAL),
              "prover2.8": dict(prover=PROVER_OLD_MAIN), "tmpl": dict(tmpl="measured"), "tmplgoal": dict(tmpl="goal"),
              "host": dict(host=3.5), "plain": dict(plain=True), "q20": dict(quant6144=0.020),
              "hits": dict(advice_window_bits=HITS_WINDOW_BITS), "prefixoff": dict(prefix_off=True),
              "ktypes": dict(k_gain="types")}
    for e, kw in extras.items():
        out[f"N5+{e}"] = Case(**(steps["N5"] | kw))
    out["N5m+ktypes"] = Case(**(steps["N5m"] | dict(k_gain="types")))
    for s in ("N5", "P1", "P5"):
        for w in POOL_WINDOW_LOST:
            if s == "P5" or w == 600:
                out[f"{s}+poolwin{w}"] = Case(**(steps[s] | dict(pool_window_s=w, scale=1 + POOL_WINDOW_LOST[w])))
    for life, t in (("day", 86400.0), ("window", float(WINDOW))):
        out[f"P1+poolwin600+cold{life}"] = replace(out["P1+poolwin600"], cold=COLD_6144 / (t - COLD_6144))
    for n in ("P1+poolwin600", "P5", "P5+poolwin600", "P5+poolwin3600"):
        out[n] = replace(out[n], leverage=LEVERAGE)
    for s in ("N5", "P5"):
        out[f"{s}+zk"] = Case(**(steps[s] | dict(budget_form="zkstack")))
        out[f"{s}+zkmachine"] = Case(**(steps[s] | dict(budget_form="zkstack-machine")))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scopes", nargs="*", default=["world week"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--steady", type=Path, help="steady-state rows (reprice.steady) for the proof-server cases")
    args = ap.parse_args(argv)
    scopes = list(SCOPES) if args.scopes == ["all"] else args.scopes
    every = cases()
    chosen = {n: k for n, k in every.items() if not args.only or n in args.only}
    if args.steady:
        chosen = {n: replace(k, steady=str(args.steady)) if k.server else k for n, k in chosen.items()}
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
