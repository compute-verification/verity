"""z3 over F_p = BabyBear: every embedded gadget (``redteam.babybear``) asserted modulo p under a rule set, the solver
asked for an accepted witness whose integer semantics differ from the reference (a forgery).

Encoding.  Hints are field elements (Ints in ``[0, p)``); public inputs are integers in their contract ranges (words
as two 16-bit limbs); every ``eq`` is a congruence ``a - b = m p`` with a fresh ``m``; every ``rng`` pins a fresh
representative ``rep in [lo, hi)`` with ``x = rep (mod p)``; ``lt_pow`` in its product form is the product read mod p.
The forgery condition is ``exists output: (out - ref) mod p != 0``.

Case splits.  Every table lookup is split on its row (the key is then a congruence to a constant and the row's
columns are constants, so ``q * pow`` is linear) and every bit-valued hint or input on its two values, so each job
is (almost) linear integer arithmetic and the whole gadget is thousands of small independent jobs -- the wave-2 remedy
for UNKNOWN (split on selectors) is the default.  A gadget is UNSAT iff every job is UNSAT, SAT if any job is (its
model is a concrete forgery), UNKNOWN otherwise (the unknown jobs are listed).

Rule ablation.  ``--drops`` re-runs each gadget with one rule removed (``Embedding(ALL - {rule})``): SAT confirms the
rule is necessary; UNSAT means the rule is redundant for that gadget (reported: it saves columns).

    python -m verity_numerical.redteam.z3_babybear --procs 12 --timeout 120 --json out.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from verity_numerical.checker import REAL, Params, Reject
from verity_numerical.checker.gadgets import Bf16Decode, GroupMax, Product, SignedSum, StateOut
from verity_numerical.checker.search import SPECS, ModZ3Ctx, _lead_zref
from verity_numerical.checker.tables import align_table, pow_table

from .babybear import ALL_RULES, LIMB, RULES, Embedding, lead_table_bb, norm_table_bb

__all__ = ["BBSpec", "BBZ3Ctx", "GadgetVerdict", "JobResult", "SPECS_BB", "jobs_for", "main", "run_gadget", "run_job"]

P = 2**31 - 2**27 + 1


# ---------------------------------------------------------------------------------------------------------------------
# The context: ModZ3Ctx with rows pinned and canonical booleans
# ---------------------------------------------------------------------------------------------------------------------

class BBZ3Ctx(ModZ3Ctx):
    def __init__(self, pins: dict[str, int] | None = None):
        super().__init__(P)
        self.pins = pins or {}

    def lookup(self, name, table, key):
        if table.name in self.pins:
            k = self.pins[table.name]
            if isinstance(key, int):
                self._add(key == k)
            else:
                self._add(self._cong(key, k))
            return table.fn(k)
        return super().lookup(name, table, key)

    def boolean(self, name, b):
        if isinstance(b, int):
            self._add(b % self.modulus in (0, 1))
            return
        # a hint is canonical in [0, p): b = 0 or 1 exactly; an expression is a congruence
        if self.z3.is_const(b) and str(b).startswith("hint."):
            self._add(self.z3.Or(b == 0, b == 1))
            return
        super().boolean(name, b)


# ---------------------------------------------------------------------------------------------------------------------
# Specs over the embedding
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BBSpec:
    name: str
    contract: Callable[[Embedding, Params], object]
    call: Callable[..., tuple]                                   # (emb, ctx, p, x, h) -> outputs (field elements)
    zref: Callable[..., tuple]                                   # (z3, p, x[, emb]) -> reference outputs as integers
    tables: tuple[Callable[[Params], object], ...] = ()          # the tables to split on
    zadmissible: Callable[[object, Params, dict], object] | None = None
    note: str = ""


def _word(x, lo: str, hi: str):
    return x[hi] * 2**LIMB + x[lo]


def _fp32_call(emb, ctx, p, x, h):
    return emb.fp32_decode(ctx, p, (x["c_lo"], x["c_hi"]), h)


def _fp32_zref(z3, p, x):
    return SPECS["fp32_decode"].zref(z3, p, {"c": _word(x, "c_lo", "c_hi")})


def _pack_call(emb, ctx, p, x, h):
    G = len(p.groups)
    return emb.pack(ctx, p, (x["s"], x["e1"], x["M"], x["z"]), [(x[f"ovf[{g}]"], x[f"sgn[{g}]"]) for g in range(G)], h)


def _pack_zref(z3, p, x, emb):
    (y,) = SPECS["pack"].zref(z3, p, x)
    return (y % 2**LIMB, y / 2**LIMB) if emb.has("LIMB16") else (y, 0)


def _epi_call(emb, ctx, p, x, h):
    return (emb.epilogue(ctx, p, (x["y32_lo"], x["y32_hi"]), h),)


def _epi_zref(z3, p, x):
    return SPECS["epilogue"].zref(z3, p, {"y32": _word(x, "y32_lo", "y32_hi")})


def _lead_norm_contract(emb: Embedding, p: Params):
    """leading_bit then normalise on one ``v``: ``l`` is a hint (pinned by LEAD), E an input."""
    from verity_numerical.checker.gadgets import Contract

    n = emb.contract("normalise", p)
    return Contract("lead_norm", {"E": n.inputs["E"], "v": n.inputs["v"]},
                    {"l": emb.contract("leading_bit", p).hints["l"], **n.hints}, n.checked, n.outputs, ("LEAD", "NORM"))


def _lead_norm_call(emb, ctx, p, x, h):
    l = emb.leading_bit(ctx, p, x["v"], {"l": h["l"]})
    return emb.normalise(ctx, p, x["E"], l, x["v"], h)


def _lead_norm_zref(z3, p, x):
    (l,) = _lead_zref(z3, p, x)
    return SPECS["normalise"].zref(z3, p, {**x, "l": l})


def _plain(name: str, call):
    return BBSpec(name, lambda emb, p: emb.contract(name, p), call, SPECS[name].zref, zadmissible=SPECS[name].zadmissible)


SPECS_BB: dict[str, BBSpec] = {
    "bf16_decode": _plain("bf16_decode", lambda emb, ctx, p, x, h: Bf16Decode.check(ctx, p, x["w"], h)),
    "fp32_decode": BBSpec("fp32_decode", lambda emb, p: emb.contract("fp32_decode", p), _fp32_call, _fp32_zref),
    "product": _plain("product", lambda emb, ctx, p, x, h: Product.check(
        ctx, p, (x["s_a"], x["m_a"], x["e_a"]), (x["s_b"], x["m_b"], x["e_b"]), h)),
    "group_max": _plain("group_max", lambda emb, ctx, p, x, h: (lambda E_ds: (E_ds[0], *E_ds[1]))(GroupMax.check(
        ctx, p, [x[f"cand[{i}]"] for i in range(p.n + 1)], {"E": h["E"], "sel": [h[f"sel[{i}]"] for i in range(p.n + 2)]}))),
    "align_state": BBSpec("align_state", lambda emb, p: emb.contract("align_state", p),
                          lambda emb, ctx, p, x, h: (emb.align_state(ctx, p, x["M"], x["d"], h),),
                          SPECS["align_state"].zref, (pow_table,)),
    "align_product": BBSpec("align_product", lambda emb, p: emb.contract("align_product", p),
                            lambda emb, ctx, p, x, h: (emb.align_product(ctx, p, x["u"], x["d"], h),),
                            SPECS["align_product"].zref, (align_table,)),
    "signed_sum": _plain("signed_sum", lambda emb, ctx, p, x, h: SignedSum.check(
        ctx, p, [(x[f"s[{i}]"], x[f"q[{i}]"]) for i in range(p.n + 1)], h)),
    "leading_bit": BBSpec("leading_bit", lambda emb, p: emb.contract("leading_bit", p),
                          lambda emb, ctx, p, x, h: (emb.leading_bit(ctx, p, x["v"], h),),
                          SPECS["leading_bit"].zref, (lead_table_bb,)),
    "normalise": BBSpec("normalise", lambda emb, p: emb.contract("normalise", p),
                        lambda emb, ctx, p, x, h: emb.normalise(ctx, p, x["E"], x["l"], x["v"], h),
                        SPECS["normalise"].zref, (norm_table_bb,), zadmissible=SPECS["normalise"].zadmissible,
                        note="l = bitlength(v) assumed (leading_bit's guarantee)"),
    "lead_norm": BBSpec("lead_norm", _lead_norm_contract, _lead_norm_call, _lead_norm_zref, (lead_table_bb, norm_table_bb),
                        note="leading_bit and normalise composed; l a hint"),
    "state_out": _plain("state_out", lambda emb, ctx, p, x, h: StateOut.check(ctx, p, x["sgn"], x["e1"], x["M"], h)),
    "pack": BBSpec("pack", lambda emb, p: emb.contract("pack", p), _pack_call, _pack_zref,
                   zadmissible=SPECS["pack"].zadmissible),
    "epilogue": BBSpec("epilogue", lambda emb, p: emb.contract("epilogue", p), _epi_call, _epi_zref,
                       zadmissible=lambda z3, p, x: SPECS["epilogue"].zadmissible(z3, p, {"y32": _word(x, "y32_lo", "y32_hi")})),
}

#: Which gadgets each rule touches (from ``babybear.rules_report()['drop_one_rule']`` and the composition).
RULE_GADGETS: dict[str, tuple[str, ...]] = {
    "LIMB16": ("fp32_decode", "pack", "epilogue"),
    "LT_DIRECT": ("align_state", "align_product", "leading_bit", "lead_norm"),
    "QUOT_TIGHT": ("align_state", "align_product"),
    "NORM_MBND": ("normalise", "lead_norm"),
    "NORM_VBND": ("normalise", "lead_norm"),
    "V_PREMISE": ("leading_bit", "normalise", "lead_norm"),
}


# ---------------------------------------------------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Job:
    gadget: str
    rules: tuple[str, ...]
    pins: dict[str, int]              # table name -> row
    bits: dict[str, int]              # "in.<name>" / "hint.<name>" -> 0/1
    timeout_ms: int


@dataclass
class JobResult:
    gadget: str
    rules: tuple[str, ...]
    pins: dict
    bits: dict
    status: str
    seconds: float
    model: dict | None = None
    note: str = ""


def jobs_for(spec: BBSpec, emb: Embedding, p: Params, timeout_ms: int, *, split_bits: bool = True,
             max_jobs: int = 200_000) -> list[Job]:
    contract = spec.contract(emb, p)
    row_ranges = []
    for tf in spec.tables:
        t = tf(p)
        row_ranges.append([(t.name, k) for k in range(t.lo, t.hi)])
    bit_names = [f"in.{n}" for n, r in contract.inputs.items() if r == (0, 2)] + \
                [f"hint.{n}" for n, r in contract.hints.items() if r == (0, 2)]
    n_rows = 1
    for rr in row_ranges:
        n_rows *= len(rr)
    if not split_bits or n_rows * 2 ** len(bit_names) > max_jobs:
        bit_names = []
    out = []
    for rows in itertools.product(*row_ranges):
        for bits in itertools.product((0, 1), repeat=len(bit_names)):
            out.append(Job(spec.name, tuple(sorted(emb.rules)), dict(rows), dict(zip(bit_names, bits)), timeout_ms))
    return out


def run_job(job: Job, p: Params = REAL) -> JobResult:
    spec = SPECS_BB[job.gadget]
    emb = Embedding(frozenset(job.rules))
    ctx = BBZ3Ctx(job.pins)
    z3 = ctx.z3
    contract = spec.contract(emb, p)
    x, h = {}, {}
    for name, (lo, hi) in contract.inputs.items():
        if f"in.{name}" in job.bits:
            x[name] = job.bits[f"in.{name}"]
            continue
        v = ctx.fresh(f"in.{name}")
        ctx.solver.add(v >= lo, v < hi)
        x[name] = v
    for name in contract.hints:
        if f"hint.{name}" in job.bits:
            h[name] = job.bits[f"hint.{name}"]
            continue
        v = ctx.fresh(f"hint.{name}")
        ctx.solver.add(v >= 0, v < P)
        h[name] = v
    try:
        outs = spec.call(emb, ctx, p, x, h)
    except Reject as exc:
        return JobResult(job.gadget, job.rules, job.pins, job.bits, "rejected-statically", 0.0, note=str(exc))
    xz = {k: (z3.IntVal(v) if isinstance(v, int) else v) for k, v in x.items()}
    refs = spec.zref(z3, p, xz, emb) if spec.name == "pack" else spec.zref(z3, p, xz)
    if spec.zadmissible is not None:
        ctx.solver.add(spec.zadmissible(z3, p, xz))
    ctx.solver.add(z3.Or([(o - r) % P != 0 for o, r in zip(outs, refs)]))
    ctx.solver.set("timeout", job.timeout_ms)
    t0 = time.perf_counter()
    status = str(ctx.solver.check())
    seconds = time.perf_counter() - t0
    model = None
    if status == "sat":
        m = ctx.solver.model()

        def val(v):
            if isinstance(v, int):
                return v
            e = m.eval(v, model_completion=True)
            return e.as_long() if hasattr(e, "as_long") else str(e)

        model = {"inputs": {k: val(v) for k, v in x.items()}, "hints": {k: val(v) for k, v in h.items()},
                 "outputs": [val(o) for o in outs], "reference": [val(r) for r in refs], "pins": job.pins}
    return JobResult(job.gadget, job.rules, job.pins, job.bits, status, round(seconds, 4), model)


def _run_job_tuple(args) -> JobResult:
    return run_job(*args)


@dataclass
class GadgetVerdict:
    gadget: str
    rules: tuple[str, ...]
    dropped: str
    status: str                       # UNSAT | SAT | UNKNOWN
    jobs: int
    unsat: int
    sat: int
    unknown: int
    seconds_total: float
    seconds_max: float
    models: list = field(default_factory=list)
    unknown_jobs: list = field(default_factory=list)
    note: str = ""

    def row(self) -> str:
        return (f"| {self.gadget} | {self.dropped or 'all rules'} | **{self.status}** | {self.jobs} | {self.unsat}/{self.sat}/{self.unknown} "
                f"| {self.seconds_total:.1f} | {self.seconds_max:.2f} | {self.note} |")


def run_gadget(name: str, emb: Embedding, *, dropped: str = "", timeout_ms: int = 60_000, procs: int = 1,
               p: Params = REAL, pool=None, log=print) -> GadgetVerdict:
    spec = SPECS_BB[name]
    jobs = jobs_for(spec, emb, p, timeout_ms)
    t0 = time.perf_counter()
    if pool is not None and procs > 1:
        results = pool.map(_run_job_tuple, [(j, p) for j in jobs], chunksize=max(1, len(jobs) // (procs * 8)))
    else:
        results = [run_job(j, p) for j in jobs]
    wall = time.perf_counter() - t0
    st = {"unsat": 0, "sat": 0, "unknown": 0}
    models, unknown = [], []
    for r in results:
        if r.status == "rejected-statically":
            st["unsat"] += 1
            continue
        st[r.status] = st.get(r.status, 0) + 1
        if r.status == "sat" and len(models) < 5:
            models.append(r.model | {"bits": r.bits})
        if r.status == "unknown":
            unknown.append({"pins": r.pins, "bits": r.bits})
    status = "SAT" if st["sat"] else ("UNSAT" if not st["unknown"] else "UNKNOWN")
    v = GadgetVerdict(name, tuple(sorted(emb.rules)), dropped, status, len(jobs), st["unsat"], st["sat"], st["unknown"],
                      round(sum(r.seconds for r in results), 2), round(max((r.seconds for r in results), default=0), 3),
                      models, unknown[:20], spec.note + (f"; wall {wall:.1f}s" if wall else ""))
    log(v.row(), flush=True)
    return v


def render(rows: list[GadgetVerdict]) -> str:
    out = ["| gadget | rule dropped | verdict | jobs | unsat/sat/unknown | z3 s (sum) | z3 s (max) | note |",
           "|---|---|---|---|---|---|---|---|"]
    out += [r.row() for r in rows]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=60.0, help="z3 timeout per job, seconds")
    ap.add_argument("--only", default=None, help="comma-separated gadget names")
    ap.add_argument("--no-drops", action="store_true", help="skip the one-rule-dropped ablation")
    ap.add_argument("--drops-only", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    names = args.only.split(",") if args.only else list(SPECS_BB)
    T = int(args.timeout * 1000)
    rows: list[GadgetVerdict] = []
    pool = mp.Pool(args.procs) if args.procs > 1 else None
    try:
        if not args.drops_only:
            for n in names:
                rows.append(run_gadget(n, Embedding(ALL_RULES), timeout_ms=T, procs=args.procs, pool=pool))
        if not args.no_drops:
            for rule in RULES:
                for n in RULE_GADGETS[rule]:
                    if n in names:
                        rows.append(run_gadget(n, Embedding(ALL_RULES - {rule}), dropped=rule, timeout_ms=T,
                                               procs=args.procs, pool=pool))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(render(rows), flush=True)
    report = {"p": P, "params": repr(REAL), "timeout_s": args.timeout, "verdicts": [asdict(r) for r in rows]}
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    path = args.json or (os.path.join(run_dir, "z3_babybear.json") if run_dir else None)
    if path:
        with open(path, "w") as f:
            json.dump(report, f, indent=1, default=str)
        print(f"wrote {path}")
    if run_dir:
        full = [r for r in rows if not r.dropped]
        problems = [f"SAT under all rules: {r.gadget}" for r in full if r.status == "SAT"]
        redundant = [f"{r.gadget} without {r.dropped}" for r in rows if r.dropped and r.status == "UNSAT"]
        unknown = [f"{r.gadget} ({r.dropped or 'all'})" for r in rows if r.status == "UNKNOWN"]
        result = {
            "schema": "research/result/v0.1",
            "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
            "workload_fingerprint": {"params": repr(REAL), "p": P, "timeout_s": args.timeout, "rules": list(RULES)},
            "validation": {"status": "passed" if not problems else "failed",
                           "detail": "; ".join(problems) or "no forgery under the full rule set",
                           "unresolved": unknown, "redundant_rules": redundant},
            "measurements": [{"name": f"z3.{r.gadget}.{r.dropped or 'all'}", "value": r.seconds_total, "unit": "s"} for r in rows],
            "measurement_files": ["z3_babybear.json"],
            "artifacts": ["z3_babybear.json"],
        }
        with open(os.path.join(run_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
