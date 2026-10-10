"""Front F6's per-unit-type cost inventory: for every served workload in the census and every unit type in its template mix
(``vu_family`` + parameters), what one unit costs as a Boolean circuit, how many units an audit draws, and the unit's public
work weight.

    python -m verity_numerical.bench.unit_costs --out DIR [--exact-nnz 4e8] [--jobs 8] [--workload SUBSTRING]

**Gates and nonzeros.** A unit type's subcircuit is its census template bound at the census parameters
(``bench.templates.subcircuit``), and its circuit is that Definition lowered whole, piece by piece, by C-Flock's generic
lowering (``verity_flock.ir_lower.lower``) into ``verity.ml.circuits.boolean.forms.Circuit`` with CSE, the builder C-Flock's
units use. ``counts()`` gives the live cone's committed input bits, ANDs, output copies and ``nnz_AB`` (the nonzeros of A and
B); C = I adds one nonzero per AND row, so a unit's C-Flock nonzeros are ``nnz_AB + ands``. Table reads (``lookups``) are not
ANDs; their tables and widths are listed.

Every number says how it was obtained (``method``):

- ``exact``: the circuit built at the census parameters.
- ``affine``: built at four small values of one parameter (:data:`AFFINE`: GEMM-like K, an element-wise width, the argmax's
  V), every count checked to be exactly affine in it (fitted on the first two points, equal on the others), and evaluated at
  the census value. When the census value is within ``--exact-nnz`` the circuit is also built there and must agree.
- ``model`` (attention only): the head at the census D, BN and semantics built at key counts T within the first key block
  (:data:`ATTENTION_T`), checked affine in T on those points, and evaluated at the stratum's T. Across key blocks the
  per-block terms (the rescale of the running state, the q decode shared between blocks) are not verified, so these are
  not exact.
- ``missing``: no piece for a primitive of the Definition (the reason names it), or ``excluded`` (embedding-row is a
  commitment opening, not proved by a proof backend), or ``not built`` (a structured reduction whose census size is over
  ``--exact-nnz``).

A unit = one VU is ``subcircuits_per_vu`` subcircuits: its counts are that multiple of the subcircuit's, without the
sharing a VU-level circuit could make between its subcircuits (an operand they share decoded once), so an upper bound on
ANDs and nonzeros.

**Sampled instances.** The draw is uniform over all units, because a hidden partition cannot be stratified by type: K =
``record_k(ε, δ)`` units (``verity.core.protocols.pouw.circuit.plan``, 27,713 at the default ε = 0.1%, δ = 2⁻⁴⁰) without
replacement from the workload's N evaluable units, at each granularity. A type with n of the N units gets a hypergeometric
count: its mean ``K n / N`` and its upper ``1 − δ`` quantile, the least k with P(X > k) ≤ δ, summed exactly over the tail
in log space.

**Work weight.** The template's census ``work_model``: ``ops_per_input`` and ``bytes_per_input`` at the bound parameters
(``BITS`` the datatype's width; attention's T as above), times ``subcircuits_per_vu`` for a VU.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import multiprocessing
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import census

METRICS = ("inputs", "ands", "output_copies", "committed", "nnz_AB", "lookups")
EPSILON = Fraction(1, 1000)
DELTA = Fraction(1, 2 ** 40)

#: template -> (the parameter the circuit is affine in, its small build points given the other parameters)
AFFINE: dict[str, tuple[str, Callable[[Mapping[str, Any]], tuple[int, ...]]]] = {
    "gemm-coordinate": ("K", lambda p: (32, 48, 64, 80)),
    "moe-expert-coordinate": ("K", lambda p: (32, 48, 64, 80)),
    "moe-expert-coordinate-weighted": ("K", lambda p: (32, 48, 64, 80)),
    "scaled-mm-fp8-block-coordinate": ("K", lambda p: tuple(int(p["G"]) * m for m in (2, 3, 4, 5))),
    "rope-head": ("D", lambda p: (16, 32, 48, 64)),
    "silu-mul": ("I", lambda p: (16, 32, 48, 64)),
    "bias-add": ("N", lambda p: (16, 32, 48, 64)),
    "bf16-mul-scalar": ("N", lambda p: (16, 32, 48, 64)),
    "moe-sum": ("H", lambda p: (16, 32, 48, 64)),
    "token-select-greedy": ("V", lambda p: (64, 128, 192, 256)),
    "gumbel-top-p-token-select": ("V", lambda p: (64, 128, 192, 256)),
}
#: templates whose Definition has a structured reduction (chunks, thread blocks): built only at the census parameters
EXACT_ONLY = ("rmsnorm-fused-cuda", "rmsnorm-triton", "moe-router-topk")
#: a small instance of each exact-only template, to estimate the census size's nonzeros (linear in the width) before building it
PROBE = {"rmsnorm-fused-cuda": "N", "rmsnorm-triton": "N", "moe-router-topk": None}
EXCLUDED = {"embedding-row": "commitment opening (census `commitment_opening`): opened against the commitment, not proved"}
#: attention's key counts T, within the first key block, at which the head is built
ATTENTION_T = (16, 32, 48, 64)


# --- building --------------------------------------------------------------------------------------------------------------

def build(template: str, params: Mapping[str, Any], statics: Mapping[str, int] | None = None) -> dict[str, Any]:
    """The subcircuit's Definition lowered whole into a CSE'd forms Circuit: ``counts()`` plus the live table reads by table."""
    from verity.ml.circuits.evaluation.reference import resolve, signature_of
    from verity_flock import ir_lower
    from verity_flock.gf2 import Circuit

    from . import templates as TM

    d = resolve(TM.subcircuit(template, **params).definition(**dict(statics or {})))
    sig, _ = signature_of(d)
    C = Circuit(cse=True)
    ins = [C.input(f"in{j}", w) for j, w in enumerate(w for p in sig for w in p.leaf_widths())]
    for j, o in enumerate(ir_lower.lower(C, d, ins)):
        C.output(f"out{j}", o)
    out = dict(C.counts())
    live = C.live()
    groups = {C.data[v][0] for v in live if C.kind[v] == "lookup"}
    tables: dict[str, dict[str, int]] = {}
    for g in sorted(groups):
        name, idx, width = C.lookups[g]
        t = tables.setdefault(f"{name}/{len(idx)}->{width}", {"reads": 0, "index_bits": len(idx), "width": width})
        t["reads"] += 1
    out["tables"] = tables
    out["definition"] = d.id
    return out


def _build_job(job: tuple) -> tuple[tuple, dict[str, Any]]:
    template, params, statics = job
    try:
        return job, build(template, dict(params), dict(statics))
    except Exception as e:  # a missing piece (NoPiece) or an unsupported parameter: reported, never guessed around
        return job, {"error": f"{type(e).__name__}: {e}"}


def _key(template: str, params: Mapping[str, Any], statics: Mapping[str, int] | None = None) -> tuple:
    return (template, tuple(sorted(params.items())), tuple(sorted((statics or {}).items())))


def run_builds(jobs: Iterable[tuple], n_jobs: int) -> dict[tuple, dict[str, Any]]:
    """Each build in its own spawned process (a build's forms are only freed with its process); a worker that dies (the OOM
    killer) raises BrokenProcessPool rather than hanging."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = list(dict.fromkeys(jobs))
    if n_jobs <= 1:
        return dict(_build_job(j) for j in jobs)
    with ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as ex:
        return dict(ex.map(_build_job, jobs))


# --- exact affine fits -----------------------------------------------------------------------------------------------------

def affine(points: Sequence[tuple[int, Mapping[str, int]]], metrics: Sequence[str] = METRICS) -> dict[str, tuple[Fraction, Fraction]]:
    """Per metric ``(slope, intercept)`` through the first two points, checked equal on every other point (ValueError naming
    the metric and point otherwise). Needs at least three points."""
    if len(points) < 3:
        raise ValueError(f"an affine fit is checked on at least three points, not {len(points)}")
    (x0, c0), (x1, c1) = points[0], points[1]
    fit = {}
    for m in metrics:
        a = Fraction(c1[m] - c0[m], x1 - x0)
        b = c0[m] - a * x0
        for x, c in points[2:]:
            if a * x + b != c[m]:
                raise ValueError(f"{m} is not affine: {a} * {x} + {b} = {a * x + b}, the build gives {c[m]}")
        fit[m] = (a, b)
    return fit


def at(fit: Mapping[str, tuple[Fraction, Fraction]], x: Fraction | int) -> dict[str, Fraction]:
    return {m: a * x + b for m, (a, b) in fit.items()}


# --- the draw --------------------------------------------------------------------------------------------------------------

def sample_size(epsilon: Fraction = EPSILON, delta: Fraction = DELTA) -> int:
    from verity.core.protocols.pouw.circuit.plan import record_k

    return record_k(epsilon, delta)


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def draw_count(N: int, n: int, K: int, tail: float = float(DELTA)) -> tuple[float, int]:
    """A type's units in a uniform K-subset of N units, n of them the type's: ``(mean, q)``, q the least k with P(X > k) <= tail
    (hypergeometric, the upper tail summed term by term from the largest possible count down)."""
    if not 0 <= n <= N or not 0 <= K <= N:
        raise ValueError(f"N = {N}, n = {n}, K = {K}")
    lo, hi = max(0, K - (N - n)), min(n, K)
    base = _log_comb(N, K)
    acc = 0.0
    for k in range(hi, lo - 1, -1):
        p = math.exp(_log_comb(n, k) + _log_comb(N - n, K - k) - base)
        if acc + p > tail:
            return K * n / N, k
        acc += p
    return K * n / N, lo


# --- work weights ----------------------------------------------------------------------------------------------------------

def work(template: str, params: Mapping[str, Any], T: float | None = None) -> dict[str, float | None]:
    """``ops_per_input`` and ``bytes_per_input`` of the census work model at ``params`` (and T)."""
    wm = census.SUBCIRCUIT_TEMPLATES[template].work_model
    env = {k: v for k, v in params.items() if isinstance(v, int)}
    dt = params.get("datatype") or ("e4m3" if "e4m3" in str(params.get("semantics", "")) else "bf16")
    env["BITS"] = census.DATATYPES[dt].bits
    out: dict[str, float | None] = {}
    for name in ("ops_per_input", "bytes_per_input"):
        expr = getattr(wm, name, None)
        if expr is None:
            out[name] = None
        elif "T" in {n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name)}:
            # affine in T for every template that reads it (attention): exact at an integer T, the mean at the mean T
            lo = census.evaluate(expr, {**env, "T": 0})
            out[name] = lo + (census.evaluate(expr, {**env, "T": 1}) - lo) * (T if T is not None else float("nan"))
        else:
            out[name] = census.evaluate(expr, env)
    return out


# --- unit types ------------------------------------------------------------------------------------------------------------

@dataclass
class UnitType:
    workload: str
    family: str
    parameters: dict[str, Any]
    vus: int
    template: str
    params: dict[str, Any]
    per_vu: int
    T: dict[str, float] | None = None       # attention: min, max, mean of the stratum's key count

    @property
    def label(self) -> str:
        skip = {"DOT", "INV", "EPS", "KVH"}
        return self.family + "{" + ",".join(f"{k}={v}" for k, v in self.parameters.items() if k not in skip) + "}"


def _t_stats(entry: Mapping[str, Any], BN: int) -> dict[str, float]:
    NB, last = int(entry["parameters"]["NB"]), entry["parameters"]["last"]
    lo, hi = ((NB * BN, NB * BN) if last == "full" else ((NB - 1) * BN + 1, NB * BN - 1))
    runs = entry.get("variable_counts", {}).get("T", {}).get("runs")
    if runs:
        n = sum((r["max"] - r["min"] + 1) * r["vus_per_value"] for r in runs)
        s = sum((r["min"] + r["max"]) * (r["max"] - r["min"] + 1) / 2 * r["vus_per_value"] for r in runs)
        return {"min": min(r["min"] for r in runs), "max": max(r["max"] for r in runs), "mean": s / n, "basis": "variable_counts"}
    return {"min": lo, "max": hi, "mean": (lo + hi) / 2, "basis": "stratum range, midpoint"}


def unit_types(workloads: Sequence[Mapping[str, Any]]) -> list[UnitType]:
    out = []
    for w in workloads:
        for e in w["template_mix"]:
            p = dict(e["subcircuit_parameters"])
            T = _t_stats(e, int(p["BN"])) if e["subcircuit_template"] == "attention-head" else None
            out.append(UnitType(w["id"], e["vu_family"], dict(e["parameters"]), int(e["vus"]), e["subcircuit_template"], p,
                                int(e["subcircuits_per_vu"]), T))
    return out


# --- the inventory ---------------------------------------------------------------------------------------------------------

def _plain(c: Mapping[str, Any]) -> dict[str, int]:
    return {m: int(c[m]) for m in METRICS}


def _num(v: Fraction) -> int | float:
    return int(v) if v.denominator == 1 else float(v)


def subcircuit_costs(types: Sequence[UnitType], exact_nnz: float, n_jobs: int, attention_t: Sequence[int] = ATTENTION_T,
                     log: Callable[[str], None] = print) -> dict[tuple, dict[str, Any]]:
    """Per distinct (template, census parameters): the per-subcircuit counts and how they were obtained; attention per
    (D, BN, semantics) carries an affine model in T instead."""
    distinct: dict[tuple, tuple[str, dict[str, Any]]] = {}
    for t in types:
        distinct.setdefault(_key(t.template, t.params), (t.template, t.params))
    jobs: list[tuple] = []
    for template, params in distinct.values():
        if template in AFFINE:
            var, pts = AFFINE[template]
            jobs += [_key(template, {**params, var: x}) for x in pts(params)]
        elif template in EXACT_ONLY and PROBE[template]:
            jobs.append(_key(template, {**params, PROBE[template]: 128}))
        elif template == "attention-head":
            jobs += [_key(template, params, {"T": T}) for T in attention_t if T <= int(params["BN"])]
        elif template not in EXCLUDED:
            jobs.append(_key(template, params))
    log(f"stage 1: {len(set(jobs))} small builds")
    built = run_builds(jobs, n_jobs)

    out: dict[tuple, dict[str, Any]] = {}
    exact_jobs: dict[tuple, tuple] = {}
    for key, (template, params) in distinct.items():
        r: dict[str, Any] = {"template": template, "params": params}
        out[key] = r
        if template in EXCLUDED:
            r.update(method="excluded", reason=EXCLUDED[template])
            continue
        if template in AFFINE:
            var, pts = AFFINE[template]
            res = [(x, built[_key(template, {**params, var: x})]) for x in pts(params)]
            err = next((c["error"] for _, c in res if "error" in c), None)
            if err:
                r.update(method="missing", reason=err)
                continue
            try:
                fit = affine(res)
            except ValueError as e:
                r.update(method="missing", reason=f"not affine in {var} on {[x for x, _ in res]}: {e}")
                continue
            x = int(params[var])
            r.update(method="affine", variable=var, points=[x for x, _ in res], counts={m: _num(v) for m, v in at(fit, x).items()},
                     fit={m: [str(a), str(b)] for m, (a, b) in fit.items()}, tables=res[-1][1]["tables"],
                     definition=res[-1][1]["definition"])
            if x in r["points"]:
                r["method"] = "exact"
            elif r["counts"]["nnz_AB"] <= exact_nnz:
                exact_jobs[key] = _key(template, params)
        elif template in EXACT_ONLY:
            probe = PROBE[template]
            if probe:
                small = built[_key(template, {**params, probe: 128})]
                if "error" in small:
                    r.update(method="missing", reason=small["error"])
                    continue
                est = small["nnz_AB"] * int(params[probe]) / 128
                r["estimated_nnz_AB"] = est
                if int(params[probe]) == 128:
                    r.update(method="exact", counts=_plain(small), tables=small["tables"], definition=small["definition"])
                elif est <= exact_nnz:
                    exact_jobs[key] = _key(template, params)
                    r["method"] = "pending"
                else:
                    r.update(method="not built", reason=f"a structured reduction, built only at the census size; about {est:.2e} "
                                                         f"nonzeros is over --exact-nnz {exact_nnz:.2e}")
            else:
                c = built[_key(template, params)]
                if "error" in c:
                    r.update(method="missing", reason=c["error"])
                else:
                    r.update(method="exact", counts=_plain(c), tables=c["tables"], definition=c["definition"])
        elif template == "attention-head":
            res = [(T, built[_key(template, params, {"T": T})]) for T in attention_t if T <= int(params["BN"])]
            err = next((c["error"] for _, c in res if "error" in c), None)
            if err:
                r.update(method="missing", reason=err)
                continue
            try:
                fit = affine(res)
            except ValueError as e:
                r.update(method="missing", reason=f"not affine in T on {[T for T, _ in res]}: {e}")
                continue
            r.update(method="model", variable="T", points=[T for T, _ in res], fit={m: [str(a), str(b)] for m, (a, b) in fit.items()},
                     built={T: _plain(c) for T, c in res}, tables=res[-1][1]["tables"], definition=res[-1][1]["definition"])
            r["_fit"] = fit
        else:
            c = built[_key(template, params)]
            if "error" in c:
                r.update(method="missing", reason=c["error"])
            else:
                r.update(method="exact", counts=_plain(c), tables=c["tables"], definition=c["definition"])

    log(f"stage 2: {len(exact_jobs)} builds at the census parameters (nonzeros <= {exact_nnz:.2e})")
    big = run_builds(exact_jobs.values(), n_jobs)
    for key, job in exact_jobs.items():
        r, c = out[key], big[job]
        if "error" in c:
            r.setdefault("notes", []).append(f"the build at the census parameters failed: {c['error']}")
            if r.get("method") == "pending":
                r.update(method="missing", reason=c["error"])
            continue
        if r["method"] == "affine":
            agree = all(r["counts"][m] == c[m] for m in METRICS)
            r["exact_check"] = "agrees" if agree else {m: [r["counts"][m], c[m]] for m in METRICS}
            if not agree:
                r.setdefault("notes", []).append("the affine extrapolation disagrees with the exact build; the exact build is used")
        r.update(method="exact", counts=_plain(c), tables=c["tables"], definition=c["definition"])
    return out


def inventory(workloads: Sequence[Mapping[str, Any]], exact_nnz: float = 0, n_jobs: int = 1,
              attention_t: Sequence[int] = ATTENTION_T, log: Callable[[str], None] = print) -> dict[str, Any]:
    K = sample_size()
    types = unit_types(workloads)
    subs = subcircuit_costs(types, exact_nnz, n_jobs, attention_t, log)
    rows = []
    for w in workloads:
        mine = [t for t in types if t.workload == w["id"]]
        N_vu = sum(t.vus for t in mine)
        N_sub = sum(t.vus * t.per_vu for t in mine)
        for t in mine:
            s = subs[_key(t.template, t.params)]
            per_sub: dict[str, Any] | None = None
            if s["method"] == "model":
                per_sub = {m: float(v) for m, v in at(s["_fit"], Fraction(t.T["mean"])).items()}
                per_sub_range = {b: float(at(s["_fit"], int(t.T[b]))["nnz_AB"] + at(s["_fit"], int(t.T[b]))["ands"]) for b in ("min", "max")}
            elif "counts" in s:
                per_sub = dict(s["counts"])
                per_sub_range = None
            else:
                per_sub_range = None
            row: dict[str, Any] = {"workload": w["id"], "unit_type": t.label, "vu_family": t.family, "parameters": t.parameters,
                                   "template": t.template, "subcircuit_parameters": t.params, "subcircuits_per_vu": t.per_vu,
                                   "method": s["method"], "reason": s.get("reason"), "T": t.T}
            wk = work(t.template, t.params, t.T["mean"] if t.T else None)
            for gran, n, N, mult in (("vu", t.vus, N_vu, t.per_vu), ("subcircuit", t.vus * t.per_vu, N_sub, 1)):
                mean, q = draw_count(N, n, K)
                g: dict[str, Any] = {"units": n, "population": N, "share": n / N, "sampled_mean": mean, "sampled_q": q,
                                     "ops": None if wk["ops_per_input"] is None else wk["ops_per_input"] * mult,
                                     "bytes": None if wk["bytes_per_input"] is None else wk["bytes_per_input"] * mult}
                if per_sub is not None:
                    c = {m: per_sub[m] * mult for m in METRICS}
                    c["nnz_C"] = c["ands"]
                    c["nnz"] = c["nnz_AB"] + c["ands"]
                    g.update(per_unit=c, audit_nnz_mean=mean * c["nnz"], audit_nnz_q=q * c["nnz"])
                    if per_sub_range:
                        g["nnz_range"] = {b: v * mult for b, v in per_sub_range.items()}
                row[gran] = g
            rows.append(row)
    for s in subs.values():
        s.pop("_fit", None)
    return {"format": "verity/f6-unit-costs/v1", "epsilon": str(EPSILON), "delta": "2^-40", "K": K,
            "draw": "uniform K-subset of all units per workload and granularity (a hidden partition cannot stratify by type)",
            "quantile": "least k with P(X > k) <= 2^-40, hypergeometric", "exact_nnz": exact_nnz, "attention_t": list(attention_t),
            "subcircuits": [dict(v, params=v["params"]) for v in subs.values()], "rows": rows}


# --- rendering -------------------------------------------------------------------------------------------------------------

def _si(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    v = float(v)
    for unit, d in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(v) >= d:
            return f"{v / d:.3g}{unit}"
    return f"{v:.3g}"


METHOD_MARK = {"exact": "E", "affine": "A", "model": "M", "missing": "–", "excluded": "x", "not built": "nb"}


def render(inv: Mapping[str, Any]) -> str:
    lines = [f"# F6 unit costs: K = {inv['K']} draws (ε = {inv['epsilon']}, δ = {inv['delta']}), uniform over all units", "",
             "Method: E exact build at the census parameters, A exact affine extrapolation (checked on 4 builds), "
             "M attention model (affine in T within the first key block), – missing, x excluded, nb not built.", "",
             "nnz = nnz_AB + ANDs (C = I). q = the 1 − 2⁻⁴⁰ hypergeometric quantile of the sampled count. Ops and bytes are "
             "the census work model.", ""]
    for gran in ("vu", "subcircuit"):
        lines += [f"## Unit = one {'VU' if gran == 'vu' else 'subcircuit'}", ""]
        for wid in dict.fromkeys(r["workload"] for r in inv["rows"]):
            rows = sorted((r for r in inv["rows"] if r["workload"] == wid), key=lambda r: -(r[gran].get("audit_nnz_mean") or 0))
            g0 = rows[0][gran]
            tot = sum(r[gran].get("audit_nnz_mean") or 0 for r in rows)
            lines += [f"### {wid}", "", f"N = {g0['population']:,} units; sampled nonzeros per audit (mean, built types) "
                      f"{_si(tot)}", "",
                      "| unit type | m | units | share | ANDs | committed | lookups | nnz_AB | nnz | sampled mean | q | "
                      "audit nnz | ops | bytes |", "|" + "---|" * 14]
            for r in rows:
                g = r[gran]
                c = g.get("per_unit") or {}
                lines.append("| " + " | ".join([
                    r["unit_type"], METHOD_MARK.get(r["method"], r["method"]), f"{g['units']:,}", f"{g['share']:.2%}",
                    _si(c.get("ands")), _si(c.get("committed")), _si(c.get("lookups")), _si(c.get("nnz_AB")), _si(c.get("nnz")),
                    f"{g['sampled_mean']:.1f}", str(g["sampled_q"]), _si(g.get("audit_nnz_mean")), _si(g["ops"]), _si(g["bytes"])]) + " |")
            lines.append("")
    lines += ["## Subcircuits", "", "| template | parameters | method | definition | detail |", "|---|---|---|---|---|"]
    for s in inv["subcircuits"]:
        detail = s.get("reason") or ""
        if s.get("points"):
            detail = f"{s.get('variable')} at {s['points']}" + (f"; exact check {s['exact_check']}" if s.get("exact_check") else "")
        if s.get("tables"):
            detail += "; tables " + ", ".join(f"{k} x{v['reads']}" for k, v in s["tables"].items())
        lines.append(f"| {s['template']} | {json.dumps(s['params'], sort_keys=True)} | {s['method']} | {s.get('definition', '')} | {detail} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for unit-costs.json and unit-costs.md")
    ap.add_argument("--exact-nnz", type=float, default=0, help="also build at the census parameters when the nonzeros are at most this")
    ap.add_argument("--jobs", type=int, default=1, help="parallel builds (spawned processes)")
    ap.add_argument("--workload", action="append", default=[], help="only workloads whose id contains this (repeatable)")
    ap.add_argument("--attention-t", default=",".join(map(str, ATTENTION_T)), help="attention's key counts within the first block")
    a = ap.parse_args(argv)
    from pathlib import Path

    workloads = [w for w in census.raw("workloads") if not a.workload or any(s in w["id"] for s in a.workload)]
    inv = inventory(workloads, a.exact_nnz, a.jobs, tuple(int(x) for x in a.attention_t.split(",")), log=lambda m: print(m, flush=True))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "unit-costs.json").write_text(json.dumps(inv, indent=1, default=str) + "\n")
    (out / "unit-costs.md").write_text(render(inv))
    print(f"wrote {out / 'unit-costs.json'} and {out / 'unit-costs.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
