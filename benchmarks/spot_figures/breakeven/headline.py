"""The figure of record: the honest rows' exfiltration bounds at pinned betas, the bits at which each scope's total
reaches 10% of its compute, and a row's sampling plan, from one entry point at a commit.

    python -m benchmarks.spot_figures.breakeven.headline rows --out DIR [--rows ROW ...] [--scopes SCOPE ...]
        [--betas BETA ...] [--workers N]      # DIR/rows.jsonl, one row a (row, scope, beta), and DIR/figure.json
        [--steady ROWS.json [--steady-map MAP.json]]
    python -m benchmarks.spot_figures.breakeven.headline read ROWS.jsonl ...
    python -m benchmarks.spot_figures.breakeven.headline plan --row ROW --scope SCOPE --beta BETA --out DIR
        [--types MIX.json]                    # DIR/mix.json: the row's plan, block by block
        [--steady ROWS.json [--steady-map MAP.json]]

from the repository's root. A row (`ROWS`) is the proof server's line at its last step
(`sweep.served_line()["P5+poolwin600"]`) with its case's levers at their measured values, over the 128k workload with
every sparse-MLA head's blocks (`volume`: the `workload` card's head blocks counted over the serve plan's requests,
`mla128`), its verification phase a datacenter's window. `price` prices its strata: the planner's per-type pricing, the quant types' host walk, the Lean verifier
as measured and the row's scale. `evaluate` charges it: the carried pool's pages recommitted at each edge
(a copy stratum and the kv arm's hash tax), a request cut at the edge recomputed, its prefix hits reproved, B128's
wait at paced egress, the pool's AND gates when only replays read it, and a replay block served at most trace128's
longest output.
`assemble` refuses a row whose request count isn't its own plan's, and takes its split from its own design.

`--steady ROWS.json` prices each block of the row's line that a steady-state row of the file measures at that row
(`reprice.steady`'s format, which `verity bench prove --steady-rows` writes by census block type), in place of the
planner's price: the statement's GPU-seconds over its blocks, with the Lean verifier as measured and no walk or scale
of the row's, which would rescale a measurement. A row prices the line's block of its own name. `--steady-map MAP.json`
says where the names differ, `{census block type: target}`: the line's block that the row measures; `[block, n]` where
the row's type is one of the n column blocks the line's block is cut into (n of its blocks a block, a statement of 1/n
as many); or null, a row this line has no block for, which is listed as unread. A row that prices no block of the line
and isn't mapped to null is refused by name, so no measurement is dropped unsaid. Each row then names the rows that
priced it and the ones unread (`steady`), and its split gives the share of the plan's proving each basis prices
(`split.basis`: `steady`, or the planner's table).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import ceil, exp, log
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

MODEL = "GLM-5.3"
LAMBDA = 1.05
#: the weights' share a row's bound may leave beside them: the target is the weights over lambda
SHARE = 1 - 1 / LAMBDA
RUN_S = 600
SCOPES = ("world week", "world month", "world year", "datacenter week", "datacenter month", "datacenter year")
BETAS = (0.05, 0.055, 0.06, 0.15, 0.4, 1.0, 2.5, 6.0, 15.0)
#: the most bits a replay block serves: trace128's longest output, 875 tokens, at 32 bits a token
SERVED_BITS = 875 * 32.0
SERVED_CASE = "trace128 max x32"
MIX, RULE = "B128", "paced at 10 tokens/s"
CHARGES = "certified hits, replays reprove hits"
STEP = "+ the verifier as measured"
#: the world the rows are priced in, the tree's reference fleet, by the name `verity bench exfiltration` takes a run of
#: record's rows by; it also reads a row's pinned beta as `b10`, and its step by the prefix "+ the verifier as measured"
WORLD = "ours"
#: the classes of gates a sampled proof block's reveals open, each whole, in a row's binding bound: assumed
CLASSES_A_PROOF_BLOCK = 4130
#: the symbols of one table's level-0 codewords, at most, in a row's binding bound: assumed
TABLE_SYMBOLS = 2 ** 23
#: what a binding bound summed over verification phases takes of the prover: assumed
ONE_SAMPLE = "one commitment, one sample; every committed window is presented"

#: the GPU prover at its measured 3.71 times, not the 4 times goal (1.101 s against 2.088 s over 3.71)
PROVER_MEASURED = 1.101 / (2.088 / 3.71)
#: every modelled lever of the P5 path at its measured value: the prover, the templated check as measured, two dense
#: provers a GPU, no flat-column verifier cut, no front run beside device work, and replay over every machine block,
#: not its lifetime alone; and the rows' verification phase, a datacenter's window, whose first stage samples a whole
#: count of its replay blocks
MEASURED = {"prover": PROVER_MEASURED, "tmpl": "measured", "dense_k": 2, "verify_cut": False, "front": False,
            "replay": 5258.0, "live": False, "commit_write": False, "registration": "datacenter"}
#: the served tokens' line, unfused
TOKENS = {"line": "tokens", "fuse": False, "fused": ""}
#: the types whose seconds a warm served block replaces in (iii): the sparse-MLA row's units as the card cuts it (its
#: head blocks, a head's end, a split's column ends, the combine, a whole row) and the families beside them
WARM_FAMILIES = ("SparseMlaHeadBlock_v1{", "SparseMlaHeadEnd_v1{", "SparseMlaSplitEndCols_v1{", "Fp8GroupQuantUe8m0_v1{",
                 "SparseMlaCombine_v1{", "BatchedGemm_v1{", "CublasSplitKGemmF32Out_v1{", "CublasSplitKGemm_v1{",
                 "DsaRMSNorm_v1{", "RmsNormMnnvl_v1{", "AllReduceResidualMnnvl_v1{", "AllReduceMnnvl_v1{",
                 "EmbeddingShard_v1{", "LlBf16DotprodGemm_v1{", "SparseMlaSplit_v1{", "Gemm_v2{")


def warm(f: float) -> dict:
    return {p: f for p in WARM_FAMILIES}


#: each row's case fields over the P5 line's, and the factors (seconds a block, fixed seconds) its types take
ROWS = {"(i) honest": {"case": TOKENS | MEASURED, "scale": {}},
        "(ii) honest": {"case": dict(MEASURED), "scale": {}},
        "(iii) honest": {"case": dict(MEASURED), "scale": warm(0.25)}}

RMS8 = "RMSNormFusedCudaVector8_v1{N=6144,"
#: the Lean verifier as measured, core-seconds a one-block statement and a block: RMSNorm v8's measured statement times
#: the two parse cuts measured since
V_MEASURED = {RMS8: (76.39 * (71.52 / 76.39) * (60.35 / 73.80), 6.8e-3)}
#: every other type: its planner seconds times the cuts' geometric mean over the two types they were measured on, and
#: the slope a block of the smaller of the two (assumed)
V_OTHER = (((25.59 / 26.77) * (23.57 / 26.46) * (71.52 / 76.39) * (60.35 / 73.80)) ** 0.5, 0.83e-3)
#: the quant types' host witness walk at width 8 on 12 vCPUs, seconds a block per K/2048: K2048's 2,048-slot walk took
#: 26.84 CPU-s at width 1 and 10.53 at width 8, one group a 32nd of the first; other K's in proportion (assumed)
QUANT = re.compile(r"Fp8GroupQuantUe8m0_v1\{K=(\d+),")
QUANT_WALK = (10.53 - 26.84 / 32) / 12 / 2048
#: a GPU's share of the node's host memory (2,163,991,326,720 bytes over eight GPUs), GiB
GPU_GIB = 2163991326720 / 2**30 / 8
#: each type's prover and verifier pair in host memory, GiB: quant at 291 GiB a K6144 pair, other K's in proportion
#: (assumed); the sparse-MLA head block's as a prover and verifier of a row's 64 keys took at load, a floor (assumed:
#: the head block's own pair is unmeasured, and at 64 keys it has that statement's MACs and k_log)
PAIR_GIB = {**{f"Fp8GroupQuantUe8m0_v1{{K={k},": 291.0 * k / 6144 for k in (6144, 2048, 1536, 256)},
            "SparseMlaHeadBlock_v1": 64.4 + 72.1}


def host_provers(pair: float, budget: float = GPU_GIB) -> int:
    """Provers a GPU a type's pair leaves room for in the GPU's share of host memory: four, two or one."""
    return 4 if 4 * pair <= budget else 2 if 2 * pair <= budget else 1


HOST_PROVERS = {p: host_provers(g) for p, g in PAIR_GIB.items()}

#: B128's carried cut at 600 s phases: the share of a phase's compute a request cut at the edge recomputes
CUT = 0.000441
#: GPU-seconds an edge holds under B128's cheapest legal rule, paced egress at 10 tokens/s
HOLD = 13.451791309265463
#: the pool's AND gates a window when only replays read it
REPLAY_ONLY = 4.9e9
#: serve_tax's kv arm: a 103.33 GB pool hashed in a median 547 ms at each 20 s edge, a tax of 4.0% against tokens
KV_ARM_GB, KV_ARM_S, KV_ARM_EDGE, KV_ARM_TAX = 103.33, 0.547, 20.0, 0.040
#: a GLM-5.3 TP8 rank's pool: 6,285 blocks of 64 tokens at 92,628 B a token
POOL_GB, POOL_BLOCKS, BLOCK_TOKENS, TOKEN_BYTES = 37.26, 6285, 64, 92628
CLASS_MAX = 262144
PAGE_ROWS = ceil(BLOCK_TOKENS * TOKEN_BYTES * 8 / CLASS_MAX)
#: one SHA-512 compression in Flock's circuit model; a row's hm96 commit is its 1,024-bit blocks, the padding block and
#: the salt's two, under the old leaf and the new
COMPRESSION_ANDS = 58120
COPY_ANDS = 2 * (CLASS_MAX // 1024 + 3) * COMPRESSION_ANDS
#: the compile's core-hours a sampled replay block
COMPILE_CORE_H = 3.8
#: the fused reads' types in a plan
FUSED = {"fr-w13": ("EmbeddingRead_v1{V=256,H=3072}", "MoeExpertFp8BlockGemm_v4"),
         "fr-w2": ("EmbeddingRead_v1{V=256,H=4096}", "MoeExpertFp8BlockGemmW_v4")}
LEFT_OUT = ("PageCopy[",)
CALL_KEYS = ("block", "kind", "per_token", "seconds_per_block", "fixed_seconds", "statement_blocks", "ands", "k_log",
             "priced", "macs", "basis")
#: what priced the carried pages' copy blocks, and a block whose pricing call names no basis
COPY_BASIS, NO_BASIS = "headline.copy_stratum", "none stated"
#: a row's request count may differ from its plan's by rounding only
REQUESTS_RTOL = 1e-9


def heads() -> int:
    """The sparse-MLA heads a token runs: GLM-5.3's `NH`."""
    from verity.definitions.registry.sparse_mla import SPARSE_MLA_GLM53
    return SPARSE_MLA_GLM53["NH"]


@lru_cache(maxsize=None)
def _volume(registered: int, n: int) -> tuple:
    for path in (HERE, HERE.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import cards
    import mla128
    return tuple(mla128.volume(cards.load("workload", MODEL)["value"], heads=n).items())


def volume(n: int | None = None) -> dict:
    """The 128k workload's sparse-MLA blocks a token over the `workload` card's, by block (`mla128.volume`), its rows
    counted at `n` heads (GLM-5.3's when not given)."""
    return dict(_volume(heads(), heads() if n is None else n))


def factors(x: dict, scale: dict) -> tuple | None:
    """The (seconds a block, fixed seconds) factors of the first prefix of `scale` that `x`'s block starts with."""
    for p, v in scale.items():
        if x["block"].startswith(p):
            return (v.get("marg", 1.0), v.get("fixed_x", 1.0)) if isinstance(v, dict) else (v, v)
    return None


def family(block: str) -> str:
    return block.split("{")[0]


def planner() -> SimpleNamespace:
    """The planner's modules, imported flat as they import each other."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import levers
    import reprice
    import sweep
    from common import f1, opt, tc

    import binding
    return SimpleNamespace(levers=levers, reprice=reprice, sweep=sweep, f1=f1, opt=opt, tc=tc, binding=binding)


def binding_bound(P: SimpleNamespace, design: dict, sampled: int, sessions: float, phases: float) -> dict:
    """A row's binding bound, modeled: the two-stage first-hit finder's bound by classes of gates
    (`binding.over_phases`) of one verification phase of the row's `design` and of the `phases` of its scope, each
    counted in calls and in compressions, with what Lean proves of one phase. The phase's `sessions` and its sampled
    proof blocks, `sampled` replay blocks' of them, are the design's. Its classes are `CLASSES_A_PROOF_BLOCK` a
    sampled proof block, and a session's reruns are those of its largest session, its tables of `TABLE_SYMBOLS`
    symbols each: both assumed, and listed (`assumed`) with what a sum over phases takes of the prover
    (`ONE_SAMPLE`)."""
    tables = max(1, ceil(design["tables"] - 1e-9))
    blocks = sampled * design["draws"] / design["replay_sample"]
    return {"what": "the two-stage first-hit finder's bound by classes of gates, every run at the stated calls of "
                    "SHA-512: of one verification phase of the row's design (`phase`) and of the row's scope of them, "
                    "their sum (`phases`), a finder's cost counted in calls, as the Lean counts, and in compressions, "
                    "every call charged as the longest; `lean` is what Lean proves at numbers of one phase's conflict "
                    "and first-hit part",
            "status": "modeled", "proof_blocks": blocks, "tables": tables,
            "assumed": [
                {"input": "classes of gates a sampled proof block", "value": CLASSES_A_PROOF_BLOCK, "unit": "class",
                 "source": "no count of the row's own blocks gives the classes one reveal opens whole"},
                {"input": "symbols of a table's level-0 codewords", "value": TABLE_SYMBOLS, "unit": "symbol",
                 "source": "the most a table's codewords hold at the sizes Lean's instances are stated at; a "
                           "session's reruns are counted at its tables times this"},
                {"input": ONE_SAMPLE, "value": None, "unit": "",
                 "source": "what a sum over verification phases takes of the prover; the row prices no check that "
                           "holds it to either"}],
            **P.binding.over_phases(ceil(sessions - 1e-9), CLASSES_A_PROOF_BLOCK * ceil(blocks - 1e-6),
                                    P.binding.queries(tables * TABLE_SYMBOLS), ceil(phases - 1e-9))}


def placed(rows: dict, where: dict) -> tuple[dict, list]:
    """Steady-state rows by the line's block each prices, and the census types left unread. `rows` is by census block
    type (`reprice.steady`) and `where` maps a type whose name isn't the line's block: to that block, to
    `[block, n]` (the type is one of n column blocks of it: n of the row's blocks make one, and a statement holds 1/n as
    many), or to None (unread). A map entry no row has, a statement that isn't whole blocks of the target and two rows
    for one block are refused."""
    if stale := sorted(set(where) - set(rows)):
        raise ValueError(f"the map names {len(stale)} block types no steady-state row measures: {stale}; remove them")
    by: dict = {}
    for name, r in rows.items():
        block, n = (name, 1) if name not in where else (where[name], 1) if not isinstance(where[name], list) else where[name]
        if block is None:
            continue
        if n < 1 or r["blocks"] % n:
            raise ValueError(f"`{name}`: a statement of {r['blocks']} blocks isn't whole blocks of `{block}` at {n} a block")
        if block in by:
            raise ValueError(f"`{block}` takes two steady-state rows, `{by[block]['type']}` and `{name}`: map one to null")
        by[block] = r | {"block": block, "type": name, "blocks": r["blocks"] // n, "per_block": n}
    return by, sorted(n for n in rows if n in where and where[n] is None)


def steady(rows_path, map_path=None, P: SimpleNamespace | None = None) -> dict:
    """What a row's spec takes as `steady`: a rows file's steady-state rows by the line's block (`placed`, under the
    map file's `{census block type: target}`), and the types the map leaves unread."""
    where = json.loads(Path(map_path).read_text()) if map_path else {}
    by, unread = placed((P or planner()).reprice.steady(rows_path), where)
    return {"rows": by, "unread": unread}


def walk(strata: list) -> list:
    """Each quant type's host witness walk added to its seconds a block; a steady-state row timed its own."""
    out = []
    for x in strata:
        if x.get("priced") != "steady" and (m := QUANT.match(x["block"])) and (add := QUANT_WALK * int(m.group(1)) / 2048):
            x = x | {"seconds_per_block": x["seconds_per_block"] + add}
        out.append(x)
    return out


def verifier(x: dict, bare: dict, core: float) -> dict:
    """`x`'s verifier seconds (its fixed seconds over `bare`'s, priced without the verifier) at the Lean verifier as
    measured: a measured type's own seconds a statement and slope, every other type's scaled by `V_OTHER`."""
    key = next((p for p in V_MEASURED if x["block"].startswith(p)), None)
    if key:
        v_new, slope = V_MEASURED[key][0] / core, V_MEASURED[key][1]
    else:
        v_new, slope = (x["fixed_seconds"] - bare["fixed_seconds"]) * V_OTHER[0], V_OTHER[1]
    return x | {"fixed_seconds": bare["fixed_seconds"] + v_new, "seconds_per_block": x["seconds_per_block"] + slope / core}


def pricing(planner_price, scale: dict, core: float, calls: list, rows: dict | None = None):
    """The row's `reprice.apply_server`: the planner's pricing, the quant walk, the measured verifier and the row's
    scale, each call recorded in `calls` as (verify, strata). A block with a steady-state row
    (`rows`, by the line's block) costs what the row measured, with the measured verifier and neither the walk nor the
    scale; a row that prices no block of the line is refused."""
    def price(strata, **kw):
        if rows:
            if lost := sorted(set(rows) - {x["block"] for x in strata}):
                raise ValueError(f"{len(lost)} steady-state rows price no block of this line: "
                                 + "; ".join(f"`{rows[b]['type']}`" + (f" (as `{b}`)" if b != rows[b]["type"] else "")
                                             for b in lost)
                                 + ". Map each to the line's block it measures, or to null to list it unread (--steady-map)")
            kw = kw | {"steady": rows}
        out = walk(planner_price(strata, **kw))
        if kw.get("verify"):
            bare = walk(planner_price(strata, **(kw | {"verify": 0.0})))
            out = [verifier(x, b, core) for x, b in zip(out, bare, strict=True)]
        out = [x if x.get("priced") == "steady" or (f := factors(x, scale)) is None else
               x | {"seconds_per_block": x["seconds_per_block"] * f[0], "fixed_seconds": x["fixed_seconds"] * f[1]}
               for x in out]
        calls.append((bool(kw.get("verify")), [{k: x.get(k) for k in CALL_KEYS} for x in out]))
        return out
    return price


def copy_stratum(strata: list, tokens_window: float) -> dict:
    """The carried pages recommitted at each edge: each page's rows a copy block, an identity whose proof opens the
    row under the old and the new leaves, priced at the measured AND types' median seconds an AND."""
    measured = [x for x in strata if x["priced"] == "measured" and x["ands"] > 0 and x["macs"] == 0]
    s_and = sorted(x["seconds_per_block"] / x["ands"] for x in measured)[len(measured) // 2]
    k27 = sorted(x["fixed_seconds"] for x in measured if x["k_log"] == 27) or [2.674]
    return dict(block=f"PageCopy[{CLASS_MAX} bits]", kind="PageCopy",
                per_token=POOL_BLOCKS * PAGE_ROWS / tokens_window, out_bits=CLASS_MAX, in_bits=CLASS_MAX,
                ands=COPY_ANDS, macs=0, seconds_per_block=COPY_ANDS * s_and, fixed_seconds=k27[len(k27) // 2],
                statement_blocks=256, table_blocks=256, k_log=27, priced="modeled", basis=COPY_BASIS)


def hash_tax(window: float, gpus: int) -> float:
    """The kv arm's serving tax for hashing the carried pages once an edge, split over the machine's ranks, a share
    of a window."""
    gpu = KV_ARM_S * POOL_GB / KV_ARM_GB / window / gpus
    return gpu * KV_ARM_TAX / (KV_ARM_S / KV_ARM_EDGE)


@dataclass
class Trace:
    """What a row's evaluation recorded: its pricing calls, its designs and each scope's model."""

    calls: list = field(default_factory=list)
    designs: list = field(default_factory=list)
    models: dict = field(default_factory=dict)
    extra: dict | None = None


@contextmanager
def installed(P: SimpleNamespace, spec: dict):
    """The planner set to the row's path for the block, restored after it."""
    reprice, sweep, f1, tc = P.reprice, P.sweep, P.f1, P.tc
    trace, saved = Trace(), []

    def put(obj, name, value):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    price, make_line, model, design = reprice.apply_server, sweep.make_line, f1.model, tc.two_stage_design
    vol = spec["volume"]

    def line(n, strata=None, **kw):
        strata = list(strata or P.levers.priced_strata(n)) + ([trace.extra] if trace.extra else [])
        strata = [x | {"per_token": x["per_token"] * next((f for p, f in vol.items() if x["block"].startswith(p)), 1.0)}
                  for x in strata]
        return make_line(n, strata, **kw)

    def capped(c_, part, s, overhead, fill):
        c_ = {**c_, "served_bits": replace(c_["served_bits"], value=spec["S"])}
        m = model(c_, part, s, overhead, fill)
        trace.models[s.name] = (m, [x for x in c_["partition"].value["strata"] if x["per_token"] > 0])
        return m

    def designed(cells, replay_blocks, delta, **kw):
        row = design(cells, replay_blocks, delta, **kw)
        trace.designs.append({"cells": list(cells), "row": row, "kw": kw})
        return row

    try:
        put(reprice, "K_MAX", HOST_PROVERS | {p: min(m, HOST_PROVERS.get(p, m)) for p, m in reprice.K_MAX.items()})
        put(reprice, "apply_server", pricing(price, spec["scale"], reprice.CORE_PER_GPU, trace.calls,
                                             (spec.get("steady") or {}).get("rows")))
        put(sweep, "make_line", line)
        put(f1, "model", capped)
        put(tc, "two_stage_design", designed)
        yield trace
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


def spec_of(row: str, scope: str, beta: float, n_heads: int | None = None, steady: dict | None = None) -> dict:
    """A job: `ROWS[row]` at `scope` and the pinned `beta`, over every sparse-MLA head, its blocks priced at the
    steady-state rows of `steady` (`steady()`) where it holds one."""
    r = ROWS[row]
    return {"row": row, "scope": scope, "beta": beta, "case": r["case"], "scale": r["scale"], "steady": steady,
            "volume": volume(n_heads), "heads": heads() if n_heads is None else n_heads, "S": SERVED_BITS,
            "share": SHARE, "step": STEP, "set": CHARGES, "S_case": SERVED_CASE, "mix": MIX, "rule": RULE,
            "run_s": RUN_S}


def evaluate(spec: dict, P: SimpleNamespace | None = None) -> tuple[dict, dict]:
    """The row's charges and bound at its pinned beta, and what its plan reads (its design, line and model).

    serve is the line's serving tax without the pool window's lost hits (the pool is carried), plus the kv arm's hash
    tax and the edge's wait; edge is the cut's recompute; proving is scaled by it and the reproved prefix hits; pool is
    the pool's AND gates a window when only replays read it, at the line's price an AND; compile is the sampled replay
    blocks' compile; total adds them to beta.  The row's verification phase is its case's (`MEASURED`: a datacenter's
    window), the first stage a whole count of its replay blocks, and the row carries it (`verification_phase`: its
    size, the count sampled, its sessions, the first limit of today's second-stage code the design meets,
    `total_cost.limits`, and its binding bound, `binding_bound`); a beta that pays for no sampled replay block of it
    has no design and is refused."""
    P = P or planner()
    sweep, levers, reprice = P.sweep, P.levers, P.reprice
    scope, t, b = spec["scope"], spec["run_s"], spec["beta"]
    with installed(P, spec) as trace:
        case = replace(sweep.served_line()["P5+poolwin600"], leverage=(), **spec["case"])
        h = HOLD / t
        edge, wait = CUT, h / (1 - h)
        serve = case.tax() - sweep.POOL_WINDOW_LOST[case.pool_window_s]
        case = replace(case, scale=1.0 + edge + sweep.PREFIX_OFF)
        L0 = sweep.build(case)
        part = L0[1]["partition"].value
        tokens = (sweep.MACHINE_GPUS * sweep.WINDOW * levers.c["mfu_serve"].value * levers.c["peak_flops"].value
                  / (2 * part["served_macs_per_token"]))
        trace.extra = copy_stratum(part["strata"], tokens * t / sweep.WINDOW)
        serve += hash_tax(sweep.WINDOW, sweep.MACHINE_GPUS) * sweep.WINDOW / t + wait
        L = sweep.build(case)
        s = L[7]["scopes"][scope]
        w = levers.WEIGHTS["FP4"]
        designs = len(trace.designs)
        y, pt = levers.bound(L, b, scope, case.scale, full=True)
        if len(trace.designs) != designs + 1:
            raise RuntimeError(f"{spec['row']}, {scope}: the bound made {len(trace.designs) - designs} designs, not one")
        p, gs = pt["plan"], sweep.SCOPES[scope].gpu_seconds
        if p is None:
            raise ValueError(f"{spec['row']}, {scope}: beta {b:g} pays for no sampled replay block of the row's "
                             f"verification phase (one costs {pt['sample_budgets']:.3g} of its budgets): no design")
        d = trace.designs[-1]
        phase = P.tc.registration_gpu_seconds(case.registration, levers.c)
        comp = p["replay_sample"] * COMPILE_CORE_H * 3600 / reprice.CORE_PER_GPU / gs
        pools = sweep.pool(L, case, pt["registration_sample"])
        pool = pools["every"]["v5 walked"] / sweep.POOL_WINDOW_ANDS["v5 walked"] * REPLAY_ONLY
        model, strata = trace.models[scope]
        total = serve + edge + b + comp + pool
        row = {"row": spec["row"], "scope": scope, "beta": b, "step": spec["step"], "S_case": spec["S_case"],
               "mix": spec["mix"], "run_s": t, "share": spec["share"], "S": spec["S"], "heads": spec["heads"],
               "serve": serve, "edge": edge, "wait": wait, "pool": pool, "compile": comp, "total": total,
               "prove": p["prove"] / gs, "replay": p["replay"] / gs, "replay_sample": p["replay_sample"],
               "q": p["replay_sample"] / s.requests, "bound_bits": y, "weights_bits": w,
               "target_bits": (1 - spec["share"]) * w, "delta_seen": model.delta, "S_seen": float(spec["S"]),
               "verification_phase": {"name": case.registration, "seconds": sweep.WINDOW,
                                      "machines": phase / (sweep.MACHINE_GPUS * sweep.WINDOW),
                                      "replay_blocks": d["kw"]["registration"],
                                      "sampled": pt["registration_sample"], "sessions": pt["sessions"],
                                      "limit": pt["limit"]},
               "served_bits": L[1]["served_bits"].value, "requests": s.requests, "flop_request": s.flop_request,
               "gpu_seconds": gs, "case": spec["case"], "scale": spec["scale"], "volume": spec["volume"],
               "census_scope": {k: getattr(sweep.SCOPES[scope], k) for k in ("requests", "flop_request")}}
        row["verification_phase"]["binding"] = binding_bound(P, d["row"], pt["registration_sample"], pt["sessions"],
                                                             gs / phase)
        if st := spec.get("steady"):
            row["steady"] = {"priced": [{k: r[k] for k in ("block", "type", "per_block", "provers", "record")}
                                        for r in st["rows"].values()], "unread": st["unread"]}
        verify = [c for c in trace.calls if c[0]]
        reads = {"design": trace.designs[-1], "strata": strata, "calls": list(trace.calls),
                 "pricing": (verify or trace.calls)[-1][1]}
    return row, reads


def pricing_split(strata: list) -> dict:
    """Each block family's share of the proving GPU-seconds a token, X, as the row's last pricing call priced it."""
    fam: dict = {}
    for x in strata:
        if not x["per_token"]:
            continue
        f = fam.setdefault(family(x["block"]), {"n": 0.0, "marg": 0.0, "fixed": 0.0, "strata": 0, "k_log": set(),
                                                 "priced": set()})
        f["n"] += x["per_token"]
        f["marg"] += x["per_token"] * x["seconds_per_block"]
        f["fixed"] += x["per_token"] * x["fixed_seconds"] / max(1, x["statement_blocks"])
        f["strata"] += 1
        f["k_log"].add(str(x.get("k_log")))
        f["priced"].add(str(x.get("priced")))
    X = sum(f["marg"] + f["fixed"] for f in fam.values())
    return {"X": X, "families": {k: v | {"k_log": sorted(v["k_log"]), "priced": sorted(v["priced"]),
                                         "share": (v["marg"] + v["fixed"]) / X if X else 0.0} for k, v in fam.items()}}


def type_of(block: str, known: dict) -> str | None:
    for t, (read, gemm) in FUSED.items():
        if block.startswith(f"Fused[{read} + {gemm}{{"):
            return t
    return known.get(block)


def plan(spec: dict, row: dict, reads: dict, known: dict, P: SimpleNamespace | None = None) -> dict:
    """The row's plan at its beta, from its own design: each block's cell, its share of a sampled replay block's
    proving and its staged type, the blocks left out of the staged types apart."""
    P = P or planner()
    opt, f1, sweep = P.opt, P.f1, P.sweep
    d, strata = reads["design"], reads["strata"]
    drow, cells = d["row"], d["cells"]
    if len(strata) != len(cells):
        raise ValueError(f"{spec['row']}, {spec['scope']}: {len(strata)} strata against {len(cells)} cells")
    split = f1.split_cells(cells, drow["merged"])
    st = opt.two_stage_statements(split, drow["k"], drow["registration_sample"], drow["form"])
    total = sum(x["prove"] for x in st)
    s = sweep.SCOPES[spec["scope"]]
    seconds = dict(f1.PERIODS)[s.period]
    out, left = [], []
    for x, c, t in zip(strata, split, st):
        if t["taken"] <= 0:
            continue
        cell = {"block": x["block"], "type": type_of(x["block"], known), "kind": x.get("kind"),
                "k_log": x["k_log"], "cap": c.cap, "table": c.table, "taken": t["taken"], "shared": t["shared"],
                "fixed_s": c.fixed, "block_s": c.prove, "prove_s": t["prove"], "share": t["prove"] / total}
        (left if x["block"].startswith(LEFT_OUT) else out).append(cell)
    out.sort(key=lambda c: -c["share"])
    return {"case": f"{spec['row']} | {spec['step']} | {spec['S_case']}", "scope": spec["scope"], "beta": spec["beta"],
            "q": drow["q"], "replay_sample": drow["replay_sample"], "period_s": seconds,
            "replay_blocks_s": drow["replay_sample"] / seconds, "gpus": s.gpu_seconds / seconds,
            "gpu_seconds": s.gpu_seconds, "prove_gpu_s": drow["prove"], "total_gpu_s": drow["total"],
            "covered": sum(c["share"] for c in out if c["type"]), "left_out": left,
            "left_out_share": sum(c["share"] for c in left), "designs": 1,
            "budget_share": d["kw"]["budget"] / s.gpu_seconds, "bound_bits": row["bound_bits"],
            "count_bits": drow.get("bound"), "row": spec["row"], "set": spec["set"], "step": spec["step"],
            "S_case": spec["S_case"], "run_s": spec["run_s"], "cells": out}


def plan_split(p: dict, calls: list, strata: list) -> dict:
    """Each block family's share of a sampled replay block's proving in the row's plan `p`, with how its blocks were
    priced (`priced`) and what priced them (`basis`: `steady`, a steady-state row, else the planner's table): by the
    row's own pricing calls, else by its model's strata."""
    priced = {x["block"]: str(x.get("priced")) for _v, xs in calls for x in xs}
    priced |= {x["block"]: str(x["priced"]) for x in strata if x.get("priced") and x["block"] not in priced}
    basis = {x["block"]: x.get("basis") for _v, xs in calls for x in xs}
    basis |= {x["block"]: x["basis"] for x in strata if x.get("basis") and x["block"] not in basis}
    fam: dict = {}
    ways: dict = {}
    bases: dict = {}
    for c in p["cells"] + p["left_out"]:
        f = fam.setdefault(family(c["block"]), {"share": 0.0, "prove_s": 0.0, "taken": 0.0, "cells": 0,
                                                 "priced": set()})
        way = priced.get(c["block"], "unknown")
        f["share"] += c["share"]
        f["prove_s"] += c["prove_s"]
        f["taken"] += c["taken"]
        f["cells"] += 1
        f["priced"].add(way)
        ways[way] = ways.get(way, 0.0) + c["share"]
        b = "steady" if way == "steady" else (basis.get(c["block"]) or NO_BASIS).split(", cut into")[0]
        bases[b] = bases.get(b, 0.0) + c["share"]
    return {"what": "the row's plan at its beta: each block family's share of a sampled replay block's proving, "
                    "priced as the row's pricing calls priced its blocks",
            "X": sum(f["prove_s"] for f in fam.values()), "q": p["q"], "replay_sample": p["replay_sample"],
            "priced": ways, "basis": dict(sorted(bases.items(), key=lambda kv: -kv[1])),
            "families": {k: v | {"priced": sorted(v["priced"])} for k, v in fam.items()}}


def assemble(row: dict, p: dict, calls: list, strata: list, pricing: list) -> dict:
    """The row with its own plan's split: refused when the plan isn't the row's at its beta, or the row's request
    count isn't its plan's (`replay_sample / q`)."""
    if (p["beta"], p["replay_sample"]) != (row["beta"], row["replay_sample"]):
        raise ValueError(f"{row['row']}, {row['scope']}: the plan at beta {p['beta']} samples {p['replay_sample']:.6g} "
                         f"replay blocks, not the row's {row['replay_sample']:.6g} at beta {row['beta']}")
    planned = p["replay_sample"] / p["q"]
    if abs(planned / row["requests"] - 1) > REQUESTS_RTOL:
        raise ValueError(f"{row['row']}, {row['scope']}: the row's {row['requests']:.6g} requests are not its plan's "
                         f"{planned:.6g} (replay_sample / q)")
    return row | {"world": WORLD, "b10": row["beta"], "pricing_split": pricing_split(pricing),
                  "split": plan_split(p, calls, strata)}


def run(spec: dict, known: dict | None = None, P: SimpleNamespace | None = None) -> tuple[dict, dict]:
    """A row with its own plan's request count and split, and the plan."""
    P = P or planner()
    row, reads = evaluate(spec, P)
    p = plan(spec, row, reads, known or {}, P)
    return assemble(row, p, reads["calls"], reads["strata"], reads["pricing"]), p


def _row(args: tuple) -> dict:
    return run(spec_of(*args))[0]


def rows(names=("(i) honest",), scopes=SCOPES, betas=BETAS, workers: int = 1, steady: dict | None = None):
    """Every (row, scope, beta)'s row, in that order, `workers` at a time over the planner imported once."""
    P = planner()
    jobs = [(r, s, b, None, steady) for r in names for s in scopes for b in betas]
    if workers <= 1:
        yield from (run(spec_of(*j), P=P)[0] for j in jobs)
        return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as ex:
        yield from ex.map(_row, jobs)


def fit(rs: list) -> tuple[float, float]:
    """The bound's least-squares slope in log beta, and the largest residual from the fit."""
    xs, ys = [log(r["beta"]) for r in rs], [log(r["bound_bits"]) for r in rs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    e = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return e, max(abs(my + e * (x - mx) - y) for x, y in zip(xs, ys))


def at(rs: list, target: float) -> tuple[float, float]:
    """The beta at which the bound reaches `target` bits and the scope's total there: log-log between the pinned rows
    that bracket it (or the end pair), compile in proportion to beta."""
    for a, b in zip(rs, rs[1:]):
        if b["bound_bits"] <= target <= a["bound_bits"]:
            break
    else:
        a, b = (rs[0], rs[1]) if target > rs[0]["bound_bits"] else (rs[-2], rs[-1])
    t = (log(target) - log(a["bound_bits"])) / (log(b["bound_bits"]) - log(a["bound_bits"]))
    beta = exp(log(a["beta"]) + t * (log(b["beta"]) - log(a["beta"])))
    c = a["compile"] / a["beta"] + (b["compile"] / b["beta"] - a["compile"] / a["beta"]) * t
    fixed = a["serve"] + a["edge"] + a["pool"]
    return beta, fixed + beta * (1 + c)


def gb10(rs: list) -> float:
    """The GB at which the scope's total is 10% of its compute: bisection in log bits on `at`."""
    small, large = rs[-1]["bound_bits"] / 10, rs[0]["bound_bits"] * 10
    for _ in range(200):
        mid = (small * large) ** 0.5
        _, tot = at(rs, mid)
        small, large = (mid, large) if tot > 0.10 else (small, mid)
    return (small * large) ** 0.5 / 8e9


def read(rows_: list) -> dict:
    """The figure, by "row | scope": the bound's slope in beta, the fixed charges, the requests a row's plan implies,
    the GB at the 10% point, and the beta and total at the FP4 weights at lambda 1 and 1.05 and at 300 GB at 1.05."""
    errors = [r for r in rows_ if r.get("error")]
    if errors:
        raise ValueError(f"rows with errors: {[(r['row'], r['scope'], r['error']) for r in errors[:3]]}")
    rows_ = list({(r["row"], r["scope"], r["step"], r["beta"]): r for r in rows_}.values())
    names = list(dict.fromkeys(r["row"] for r in rows_))
    out = {}
    for name in names:
        for scope in dict.fromkeys(r["scope"] for r in rows_ if r["row"] == name):
            rs = sorted((r for r in rows_ if r["row"] == name and r["scope"] == scope), key=lambda r: r["beta"])
            if len(rs) < 2:
                continue
            if len({r["step"] for r in rs}) > 1:
                raise ValueError(f"{name}, {scope}: rows of more than one step")
            w = rs[0]["weights_bits"]
            fixed = {round(r["serve"] + r["edge"] + r["pool"], 12) for r in rs}
            e, resid = fit(rs)
            rec = {"row": name, "scope": scope, "e": e, "resid": resid, "fixed": min(fixed),
                   "fixed_spread": max(fixed) - min(fixed),
                   "requests_line": sorted({round(r["replay_sample"] / r["q"] / 1e16, 6) for r in rs}),
                   "requests_field": rs[0]["requests"], "S": rs[0]["S"], "delta": rs[0]["delta_seen"],
                   "step": rs[0]["step"], "share": rs[0]["share"], "gpu_seconds": rs[0]["gpu_seconds"],
                   "gb10": gb10(rs)}
            for k, tb in {"FP4 l1": w, "FP4 l1.05": w / LAMBDA, "300GB l1.05": 300 * 8e9 / LAMBDA}.items():
                beta, tot = at(rs, tb)
                rec[k] = {"beta": beta, "total": tot}
            out[f"{name} | {scope}"] = rec
    return out


def show(figure: dict) -> str:
    lines = [f"row | scope | e | resid | fixed pt | GB@10% x{LAMBDA} | FP4 l1 | FP4 l1.05 | 300GB l1.05 (beta)"]
    for r in figure.values():
        lines.append(f"{r['row'][:44]:44s} | {r['scope']:16s} | {r['e']:.5f} | {r['resid']:.1e} | "
                     f"{r['fixed'] * 100:.3f} | {r['gb10'] * LAMBDA:9.1f} | {r['FP4 l1']['total'] * 100:7.2f} | "
                     f"{r['FP4 l1.05']['total'] * 100:7.2f} | {r['300GB l1.05']['total'] * 100:7.2f} "
                     f"({r['300GB l1.05']['beta'] * 100:.2f})")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.spot_figures.breakeven.headline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows", help="every (row, scope, beta)'s row and the figure from them")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--rows", nargs="+", default=["(i) honest"], choices=sorted(ROWS))
    r.add_argument("--scopes", nargs="+", default=list(SCOPES))
    r.add_argument("--betas", nargs="+", type=float, default=list(BETAS))
    r.add_argument("--workers", type=int, default=1)
    d = sub.add_parser("read", help="the figure from rows")
    d.add_argument("files", nargs="+", type=Path)
    p = sub.add_parser("plan", help="one row's plan, block by block")
    p.add_argument("--row", required=True, choices=sorted(ROWS))
    p.add_argument("--scope", required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--types", type=Path, help="a plan whose cells name each block's staged type")
    p.add_argument("--out", type=Path, required=True)
    for cmd in (r, p):
        cmd.add_argument("--steady", type=Path, metavar="ROWS.json",
                         help="steady-state rows by census block type (`verity bench prove --steady-rows`): a block "
                              "of the line with a row is priced at it")
        cmd.add_argument("--steady-map", type=Path, metavar="MAP.json",
                         help="{census block type: the line's block it measures, [block, n] for one of its n column "
                              "blocks, or null to list the row unread}, for the rows whose type isn't a block's name")
    a = ap.parse_args(argv)
    if a.cmd != "read" and a.steady_map and not a.steady:
        ap.error("--steady-map maps the rows of --steady")
    measured = steady(a.steady, a.steady_map) if a.cmd != "read" and a.steady else None
    if a.cmd == "rows":
        a.out.mkdir(parents=True, exist_ok=True)
        got = []
        with open(a.out / "rows.jsonl", "w") as f:
            for row in rows(a.rows, a.scopes, a.betas, a.workers, measured):
                f.write(json.dumps(row) + "\n")
                f.flush()
                got.append(row)
        figure = read(got)
        (a.out / "figure.json").write_text(json.dumps(figure, indent=1) + "\n")
        print(show(figure))
    elif a.cmd == "read":
        print(show(read([json.loads(x) for f in a.files for x in f.read_text().splitlines() if x.strip()])))
    else:
        known = ({c["block"]: c["type"] for c in json.loads(a.types.read_text())["cells"] if c.get("type")}
                 if a.types else {})
        row, mix = run(spec_of(a.row, a.scope, a.beta, steady=measured), known)
        tree = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "mix.json").write_text(json.dumps(mix | {"planner": tree.strip()}, indent=1, default=str) + "\n")
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
