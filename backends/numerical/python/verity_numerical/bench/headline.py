"""The per-workload headline: for each served workload in the census and each backend, the share of the run's native work
that published, cleared cells cover, and the work-weighted overhead over that covered share.

A workload's template mix (``census/workloads.json``: VUs x subcircuits per VU, per bound subcircuit) gives the units; each
template's census work model gives the native seconds per unit on the served hardware line, under two bases:

* ``tensor-peak`` -- the GEMM-like templates (:data:`TENSOR_TEMPLATES`) at the line's dense tensor-core peak (``ops_per_input``
  over ``flop_per_second``), every other template memory-bound (``bytes_per_input`` over ``memory_bytes_per_second``);
* ``memory-bound`` -- every template memory-bound, a GEMM coordinate reading its weight row (the batch-1 decode reading).

A subcircuit is covered by a backend when the render holds an admissible result of that backend on exactly that subcircuit
(template and bound parameters, the semantics included), cleared (not provisional, no open finding), whose prover ran on the
served hardware.  The covered share is covered native seconds over all modelled native seconds; the overhead is the covered
units' proving seconds (units / P, the highest-P such result per subcircuit) over their native seconds.  An uncovered subcircuit
is listed with why; nothing is imputed.  A subcircuit whose template has no work model in a basis is listed and left out of
both sums in that basis.  Where a work model has a per-input variable known only to bounds (attention's key count ``T`` from
its VU's ``NB`` and ``last``), native seconds, shares and overheads are intervals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from verity_numerical.bench import census
from verity_numerical.bench import tables as T

#: Templates whose native N under the tensor-peak basis is the tensor-core peak (their arithmetic is tensor-core MMA).
TENSOR_TEMPLATES = frozenset({"gemm-coordinate", "moe-expert-coordinate", "moe-expert-coordinate-weighted", "scaled-mm-fp8-block-coordinate"})


@dataclass(frozen=True)
class Basis:
    id: str
    label: str
    description: str


BASES = (
    Basis("tensor-peak", "tensor-peak GEMM",
          "GEMM-like templates (" + ", ".join(sorted(TENSOR_TEMPLATES)) + ") at the served line's dense tensor-core peak "
          "(ops_per_input ÷ flop_per_second); every other template memory-bound (bytes_per_input ÷ memory_bytes_per_second)"),
    Basis("memory-bound", "all memory-bound",
          "every template memory-bound (bytes_per_input ÷ memory_bytes_per_second); a GEMM coordinate reads its weight row, "
          "the batch-1 decode reading"),
)

Interval = tuple[float, float]


def _env(params: Mapping[str, Any]) -> dict[str, int]:
    env = {k: v for k, v in params.items() if isinstance(v, int) and not isinstance(v, bool)}
    dt = census.DATATYPES.get(params.get("datatype")) if isinstance(params.get("datatype"), str) else None
    if dt is not None:
        env["BITS"] = dt.bits
    return env


def _eval_range(expr: str, env: Mapping[str, int], bounds: Mapping[str, tuple[int, int]]) -> tuple[int, int]:
    """``expr`` at the lower and at the upper end of every bounded variable (the work models are increasing in them)."""
    return (census.evaluate(expr, {**env, **{k: lo for k, (lo, _) in bounds.items()}}),
            census.evaluate(expr, {**env, **{k: hi for k, (_, hi) in bounds.items()}}))


def _bounds(wm: Any, vu_params: Mapping[str, Any]) -> dict[str, tuple[int, int]] | str:
    out = {}
    declared = getattr(wm, "variable_bounds", None)
    for var, vb in (vars(declared).items() if declared else ()):
        case = getattr(vb.cases, str(vu_params.get(vb.by)), None)
        if case is None:
            return f"no bound for {var} at {vb.by} = {vu_params.get(vb.by)!r}"
        env = _env(vu_params)
        out[var] = (census.evaluate(case.min, env), census.evaluate(case.max, env))
    return out


def _line_for(served: Any, params: Mapping[str, Any]) -> Any:
    """The served device's census line for the subcircuit's datatype (its own when the datatype has none there)."""
    dt = params.get("datatype")
    try:
        return census.line(served.device, dt) if isinstance(dt, str) else served
    except KeyError:
        return served


def seconds_per_unit(template: str | None, params: Mapping[str, Any], vu_params: Mapping[str, Any], served: Any, basis: str) -> Interval | str:
    """Native seconds of one unit (one subcircuit instance) on the served line under ``basis``, or why there is none."""
    t = census.SUBCIRCUIT_TEMPLATES.get(template) if template else None
    if t is None:
        return "no census template"
    wm = t.work_model
    bounds = _bounds(wm, vu_params)
    if isinstance(bounds, str):
        return bounds
    env = _env(params)
    if basis == "tensor-peak" and template in TENSOR_TEMPLATES:
        hw = _line_for(served, params)
        lo, hi = _eval_range(wm.ops_per_input, env, bounds)
        return lo / hw.flop_per_second, hi / hw.flop_per_second
    expr = getattr(wm, "bytes_per_input", None)
    if expr is None:
        return "no bytes model (bytes_per_input) in the census"
    bw = getattr(served, "memory_bytes_per_second", None)
    if not isinstance(bw, (int, float)):
        return f"no memory bandwidth for {served.id} in the census"
    lo, hi = _eval_range(expr, env, bounds)
    return lo / bw, hi / bw


def bound_subcircuit(template: str | None, params: Mapping[str, Any]) -> Any:
    """The census subcircuit of ``template`` bound to exactly ``params``, or ``None``."""
    return next((s for s in census.SUBCIRCUITS.values() if s.template == template and vars(s.parameters) == dict(params)), None)


def _display(template: str | None, params: Mapping[str, Any], vu_family: str) -> str:
    known = bound_subcircuit(template, params)
    if known is not None:
        return known.display
    if template is None:
        return f"{vu_family} (no census template)"
    return f"{census.SUBCIRCUIT_TEMPLATES[template].name} · " + ", ".join(f"{k}={v}" for k, v in params.items())


def groups(workload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The workload's units per subcircuit (template and bound parameters), with native seconds per basis (an interval, or why
    none), in the mix's order of first appearance."""
    served = census.HARDWARE[workload["serving"]["hardware"]]
    out: dict[str, dict[str, Any]] = {}
    for m in workload["template_mix"]:
        tpl, params = m["subcircuit_template"], m["subcircuit_parameters"] or {}
        key = json.dumps([tpl or m["vu_family"], params], sort_keys=True)
        units = m["vus"] * m["subcircuits_per_vu"]
        known = bound_subcircuit(tpl, params)
        g = out.setdefault(key, {"template": tpl, "parameters": dict(params), "subcircuit": known.id if known else None,
                                 "display": _display(tpl, params, m["vu_family"]), "vu_families": [], "vus": 0, "units": 0,
                                 "native_seconds": {b.id: (0.0, 0.0) for b in BASES}, "unmodelled": {}})
        if m["vu_family"] not in g["vu_families"]:
            g["vu_families"].append(m["vu_family"])
        g["vus"] += m["vus"]
        g["units"] += units
        for b in BASES:
            s = seconds_per_unit(tpl, params, m["parameters"], served, b.id)
            if isinstance(s, str):
                g["unmodelled"][b.id] = s
            else:
                lo, hi = g["native_seconds"][b.id]
                g["native_seconds"][b.id] = (lo + units * s[0], hi + units * s[1])
    for g in out.values():
        for bid in g["unmodelled"]:
            g["native_seconds"][bid] = None
    return list(out.values())


def _share(part: Interval, rest: Interval) -> Interval:
    lo = part[0] / (part[0] + rest[1]) if part[0] + rest[1] > 0 else 0.0
    hi = part[1] / (part[1] + rest[0]) if part[1] + rest[0] > 0 else 0.0
    return lo, hi


def _sum(xs: Sequence[Interval]) -> Interval:
    return sum(x[0] for x in xs), sum(x[1] for x in xs)


def fmt_pct(iv: Interval) -> str:
    a, b = f"{100 * iv[0]:.3g}", f"{100 * iv[1]:.3g}"
    return f"{a}%" if a == b else f"{a}–{b}%"


def fmt_x(iv: Interval | None) -> str:
    if iv is None:
        return T.EMPTY
    a, b = T.fmt_overhead(iv[0]), T.fmt_overhead(iv[1])
    return a if a == b else f"{a[:-1]}–{b}"


def fmt_s(iv: Interval) -> str:
    a, b = f"{iv[0]:.3g}", f"{iv[1]:.3g}"
    return f"{a} s" if a == b else f"{a}–{b} s"


def headline(V: Any, workloads: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One record per workload (census order): its subcircuits (units, native seconds and share of the run's native work per
    basis) and, per backend, what it covers (the cell used), what it does not (and why), the covered share and the
    work-weighted overhead per basis."""
    from verity_numerical.bench import views  # noqa: PLC0415  (views renders this module's records)

    workloads = census.raw("workloads") if workloads is None else workloads
    cfg_of = {c.id: c for c in V.configurations}
    backend_ids = [b["id"] for b in views.backends(views.configuration_records(V))]

    def sub_of(c: Any) -> str | None:
        if c.subcircuit:
            return c.subcircuit
        try:
            return census.subcircuit_of(c.line).id
        except StopIteration:
            return None

    def prover_line(c: Any) -> Any:
        gpu = V.prover_gpu.get(c.artifact)
        return next((h for h in census.HARDWARE.values() if T.same_sku(gpu, h.device)), None)

    cells = [(c, views.backend_of(cfg_of[c.configuration]) if c.configuration in cfg_of else None, sub_of(c)) for c in V.admissible]
    out = []
    for w in workloads:
        served = census.HARDWARE[w["serving"]["hardware"]]
        gs = groups(w)
        totals = {b.id: _sum([g["native_seconds"][b.id] for g in gs if g["native_seconds"][b.id] is not None]) for b in BASES}
        subs = []
        for g in gs:
            share = {}
            for b in BASES:
                s = g["native_seconds"][b.id]
                share[b.id] = None if s is None else _share(s, (totals[b.id][0] - s[0], totals[b.id][1] - s[1]))
            subs.append({**g, "share": share})
        rows = []
        for bid in backend_ids:
            covered, uncovered = [], []
            for g in subs:
                mine = [(c, sub) for c, b, sub in cells if b == bid and g["subcircuit"] is not None and sub == g["subcircuit"]]
                hits = [c for c, _ in mine if not c.provisional and not c.legacy.finding and T.same_sku(V.prover_gpu.get(c.artifact), served.device)]
                if hits:
                    c = min(hits, key=lambda x: (-x.throughput["instances_per_second"], x.artifact))
                    P = c.throughput["instances_per_second"]
                    covered.append({"subcircuit": g["subcircuit"], "display": g["display"], "units": g["units"], "artifact": c.artifact,
                                    "configuration": c.configuration, "scheme": c.scheme, "P": P, "proving_seconds": g["units"] / P})
                    continue
                elsewhere = []
                for c, _ in mine:
                    h = prover_line(c)
                    elsewhere.append({"artifact": c.artifact, "prover_hardware": h.id if h else None, "prover_gpu": V.prover_gpu.get(c.artifact),
                                      "on_served_hardware": T.same_sku(V.prover_gpu.get(c.artifact), served.device),
                                      "cleared": not c.provisional and not c.legacy.finding})
                if g["subcircuit"] is None:
                    why, code = "the census binds no subcircuit to these parameters, so no result can name it", "no census subcircuit"
                elif not elsewhere:
                    why, code = "no admissible result on this subcircuit", "no result"
                elif any(e["on_served_hardware"] for e in elsewhere):
                    why, code = "results on the served hardware are not cleared (provisional or an open finding)", "not cleared"
                else:
                    names = sorted({(census.HARDWARE[e["prover_hardware"]].name if e["prover_hardware"] else e["prover_gpu"] or "?") for e in elsewhere})
                    why, code = f"results only on {', '.join(names)}, not the served {served.name}", f"only on {', '.join(names)}"
                uncovered.append({"subcircuit": g["subcircuit"], "template": g["template"], "parameters": g["parameters"], "display": g["display"],
                                  "units": g["units"], "reason": code, "text": why, "results_elsewhere": elsewhere})
            per_basis = {}
            for b in BASES:
                cov = _sum([g["native_seconds"][b.id] for g in subs if g["native_seconds"][b.id] is not None
                            and any(x["subcircuit"] == g["subcircuit"] for x in covered)])
                prove = sum(x["proving_seconds"] for x in covered
                            if next(g for g in subs if g["subcircuit"] == x["subcircuit"])["native_seconds"][b.id] is not None)
                share = _share(cov, (totals[b.id][0] - cov[0], totals[b.id][1] - cov[1]))
                overhead = (prove / cov[1], prove / cov[0]) if cov[0] > 0 else None
                per_basis[b.id] = {"share": share, "overhead": overhead, "covered_native_seconds": cov, "proving_seconds": prove,
                                   "unmodelled": [g["display"] for g in subs if g["native_seconds"][b.id] is None]}
            rows.append({"backend": bid, "covered": covered, "uncovered": uncovered, "bases": per_basis})
        out.append({"workload": w["id"], "display": w["display"], "served_hardware": served.id, "served_display": f"{served.name} · "
                    f"{census.DATATYPES[served.dtype].display}", "vus": w["vus"], "not_evaluable_vus": w["not_evaluable_vus"],
                    "native_seconds": totals, "subcircuits": subs, "backends": rows})
    return out


def _j(iv: Interval | None) -> dict[str, float] | None:
    return None if iv is None else {"min": iv[0], "max": iv[1]}


def to_json(records: Sequence[Mapping[str, Any]], result_ids: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The headline records for the site: intervals as ``{min, max}`` beside their printed ``display``."""
    result_ids = result_ids or {}
    return {
        "bases": [{"id": b.id, "label": b.label, "description": b.description} for b in BASES],
        "tensor_templates": sorted(TENSOR_TEMPLATES),
        "workloads": [{
            "id": r["workload"], "display": r["display"], "served_hardware": r["served_hardware"], "served_display": r["served_display"],
            "vus": r["vus"], "not_evaluable_vus": r["not_evaluable_vus"],
            "native_seconds": {b: _j(v) for b, v in r["native_seconds"].items()},
            "subcircuits": [{"subcircuit": g["subcircuit"], "template": g["template"], "parameters": g["parameters"], "display": g["display"],
                             "vu_families": g["vu_families"], "vus": g["vus"], "units": g["units"],
                             "native_seconds": {b: _j(v) for b, v in g["native_seconds"].items()}, "unmodelled": g["unmodelled"],
                             "share": {b: (None if v is None else {**_j(v), "display": fmt_pct(v)}) for b, v in g["share"].items()}}
                            for g in r["subcircuits"]],
            "backends": [{"backend": row["backend"],
                          "covered": [{**x, "result": result_ids.get(x["artifact"])} for x in row["covered"]],
                          "uncovered": row["uncovered"],
                          "bases": {b: {"share": {**_j(v["share"]), "display": fmt_pct(v["share"])},
                                        "overhead": None if v["overhead"] is None else {**_j(v["overhead"]), "display": fmt_x(v["overhead"])},
                                        "covered_native_seconds": _j(v["covered_native_seconds"]), "proving_seconds": v["proving_seconds"],
                                        "unmodelled": v["unmodelled"]} for b, v in row["bases"].items()}}
                         for row in r["backends"]],
        } for r in records],
    }


def render_markdown(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The headline section's markdown lines."""
    tp, mb = BASES
    out = ["## Workload headline — share of a served run's native work covered, and its work-weighted overhead", "",
           "Per served workload (`census/workloads.json`) and backend. Units = VUs × subcircuits per VU, per bound subcircuit (the "
           "workload's template mix). Native work = units × the template's census work model on the served hardware line, in two bases: "
           f"**{tp.label}**: {tp.description}; **{mb.label}**: {mb.description}. A subcircuit is covered when the backend has an "
           "admissible result on exactly that subcircuit (template and bound parameters, the semantics included), cleared (not "
           "provisional, no open finding), proved on the served hardware. Share = covered native work ÷ all modelled native work; "
           "overhead = the covered units' proving time (units ÷ P of the highest-P such result, at the reference network) ÷ their "
           "native time. Uncovered subcircuits are listed with why and never imputed; a template without a work model in a basis is "
           "listed and left out of both sums. Ranges: attention's key count T is known only to its block (NB, last).", ""]
    for r in records:
        out += [f"### {r['display']} — served on {r['served_display']}", "",
                f"`{r['workload']}`: {r['vus']} VUs ({r['not_evaluable_vus']} not evaluable, outside the mix); native time "
                f"{fmt_s(r['native_seconds'][tp.id])} ({tp.label}), {fmt_s(r['native_seconds'][mb.id])} ({mb.label}).", ""]
        by_sub: dict[str, list[str]] = {}
        for row in r["backends"]:
            for x in row["covered"]:
                by_sub.setdefault(x["subcircuit"], []).append(row["backend"])
        rows = [[g["display"], str(g["units"]),
                 *[(fmt_pct(g["share"][b.id]) if g["share"][b.id] is not None else f"not modelled: {g['unmodelled'][b.id]}") for b in BASES],
                 ", ".join(by_sub.get(g["subcircuit"], [])) or "none"] for g in r["subcircuits"]]
        out += [T._md_table(("Subcircuit", "Units", f"Share of native work ({tp.label})", f"Share of native work ({mb.label})", "Covered by"), rows), ""]
        rows = []
        for row in r["backends"]:
            bt, bm = row["bases"][tp.id], row["bases"][mb.id]
            why: dict[str, list[str]] = {}
            for u in row["uncovered"]:
                why.setdefault(u["reason"], []).append(u["display"])
            unc = "; ".join(f"{reason}: {', '.join(subs)}" for reason, subs in why.items()) or "none"
            rows.append([row["backend"], f"{len(row['covered'])} of {len(r['subcircuits'])}", fmt_pct(bt["share"]), fmt_x(bt["overhead"]),
                         fmt_pct(bm["share"]), fmt_x(bm["overhead"]), unc])
        out += [T._md_table(("Backend", "Subcircuits covered", f"Share ({tp.label})", f"Overhead over covered ({tp.label})",
                             f"Share ({mb.label})", f"Overhead over covered ({mb.label})", "Uncovered (why)"), rows), ""]
    return out
