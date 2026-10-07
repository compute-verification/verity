"""The per-workload headline: for each served workload in the census and each backend, the share of the run's native work
that published, cleared cells cover, and the work-weighted overhead over that covered share.

A workload's template mix (``benchmarks/census/workloads.json``: VUs x subcircuits per VU, per bound subcircuit) gives the units; each
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
its VU's ``NB`` and ``last``), native seconds, shares and overheads are intervals.  A template the census marks a commitment
opening (``commitment_opening``: embedding) is uncovered for that reason: the ontology has no covered-by-commitment state, so its
native work stays in the denominator and is never counted as covered.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
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


def seconds_per_unit(template: str | None, params: Mapping[str, Any], vu_params: Mapping[str, Any], served: Any, basis: str,
                     fixed: Mapping[str, int] | None = None) -> Interval | str:
    """Native seconds of one unit (one subcircuit instance) on the served line under ``basis``, or why there is none; ``fixed``
    pins per-input variables (attention's T) instead of their VU bounds."""
    t = census.SUBCIRCUIT_TEMPLATES.get(template) if template else None
    if t is None:
        return "no census template"
    wm = t.work_model
    bounds = {k: (v, v) for k, v in fixed.items()} if fixed else _bounds(wm, vu_params)
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
                                 "native_seconds": {b.id: (0.0, 0.0) for b in BASES}, "unmodelled": {},
                                 "variable": variable_of(tpl), "values": {}, "values_complete": True})
        if m["vu_family"] not in g["vu_families"]:
            g["vu_families"].append(m["vu_family"])
        g["vus"] += m["vus"]
        g["units"] += units
        counts = (m.get("variable_counts") or {}).get(g["variable"]) if g["variable"] else None
        if g["variable"] and counts is None:
            g["values_complete"] = False
        for b in BASES:
            if counts is not None:      # the VUs at each value of the variable (attention's T): exact native work
                s = (0.0, 0.0)
                for r in counts["runs"]:
                    for v in range(r["min"], r["max"] + 1):
                        n = r["vus_per_value"] * m["subcircuits_per_vu"]
                        g["values"].setdefault(v, {"units": 0, "native_seconds": {}})
                        if b is BASES[0]:
                            g["values"][v]["units"] += n
                        one = seconds_per_unit(tpl, params, m["parameters"], served, b.id, {g["variable"]: v})
                        if isinstance(one, str):
                            s = one
                            break
                        g["values"][v]["native_seconds"][b.id] = g["values"][v]["native_seconds"].get(b.id, 0.0) + n * one[0]
                        s = (s[0] + n * one[0], s[1] + n * one[0])
                    if isinstance(s, str):
                        break
            else:
                s = seconds_per_unit(tpl, params, m["parameters"], served, b.id)
                if not isinstance(s, str):
                    s = (units * s[0], units * s[1])
            if isinstance(s, str):
                g["unmodelled"][b.id] = s
            else:
                lo, hi = g["native_seconds"][b.id]
                g["native_seconds"][b.id] = (lo + s[0], hi + s[1])
    for g in out.values():
        for bid in g["unmodelled"]:
            g["native_seconds"][bid] = None
        if not g["values_complete"]:
            g["values"] = {}
    return list(out.values())


def variable_of(template: str | None) -> str | None:
    """The per-input variable a template's work model depends on (attention's ``T``), or ``None``."""
    t = census.SUBCIRCUIT_TEMPLATES.get(template or "")
    wm = t.work_model if t else None
    used = [v for v in vars(getattr(wm, "variables", None) or SimpleNamespace())
            if v in str(wm.ops_per_input) + str(getattr(wm, "bytes_per_input", ""))] if wm else []
    return used[0] if used else None


def input_source(V: Any, c: Any) -> str | None:
    """The ``source`` its registered input set records for the cell's inputs (``synthetic`` for the spine generator's), or
    ``None`` when the set is not registered or records none."""
    inst = getattr(c, "instances", None) or {}
    sets = getattr(V, "input_sets", {}) or {}
    hit = next((sets[k] for k in (inst.get("set_art"), inst.get("content_digest"), inst.get("manifest_sha256"))
                if isinstance(k, str) and k in sets), None)
    return (hit[1].get("source") if hit else None) or None


def provenance(per_value: list[dict[str, Any]], var: str) -> dict[str, Any] | None:
    """The footnote a covered entry carries when some of its values are proved on synthetic inputs (red-team-flock-3's CP6:
    counted, like the FP8 spine cells, with their provenance shown)."""
    syn = [x[var] for x in per_value if x.get("source") == "synthetic"]
    if not syn:
        return None
    return {"synthetic_values": syn, "text": f"{len(syn)} of the {len(per_value)} covered {var} values are proved on synthetic inputs "
                                             f"(the spine generator, source: synthetic), the rest on captured inputs"}


def verifier_evaluated(template: str | None) -> Any:
    """The census's record of a template's verifier-evaluated parts (``verifier_evaluated``: what, how much, whether counted
    and on whose ruling), or ``None``."""
    return getattr(census.SUBCIRCUIT_TEMPLATES.get(template or ""), "verifier_evaluated", None)


def runs(values: Sequence[int]) -> str:
    """``1–4, 128–132, 287``: sorted integers as runs."""
    vs, out = sorted(values), []
    for v in vs:
        if out and v == out[-1][1] + 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return ", ".join(f"{a}" if a == b else f"{a}–{b}" for a, b in out)


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


def drawn_values(V: Any, subcircuit: str, var: str, workload: Mapping[str, Any]) -> set[int]:
    """The values of ``var`` the workload's registered input sets of ``subcircuit`` hold: each set's recorded value (its
    parameters, relation statics, name, or a ``<var>_values`` list), for sets recorded as the workload's row."""
    row = (workload.get("source") or {}).get("row")
    out: set[int] = set()
    for _art, meta in {a: m for a, m in getattr(V, "input_sets", {}).values()}.items():
        if meta.get("subcircuit") != subcircuit or meta.get("row") != row:
            continue
        for d in (meta.get("parameters"), (meta.get("relation") or {}).get("statics") if isinstance(meta.get("relation"), Mapping) else None):
            if isinstance(d, Mapping) and isinstance(d.get(var), int) and not isinstance(d.get(var), bool):
                out.add(d[var])
        m = re.search(rf"(?:^|[-_/.:]){var.lower()}(\d+)$", str(meta.get("set") or ""))
        if m:
            out.add(int(m.group(1)))
        out.update(v for v in meta.get(f"{var}_values") or () if isinstance(v, int))
    return out


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

    # a typed circuit cell (verity/flock-circuit/types) is compared with a flat one per instance only, never in one headline
    cells = [(c, views.backend_of(cfg_of[c.configuration]) if c.configuration in cfg_of else None, sub_of(c)) for c in V.admissible
             if c.configuration not in views.TYPED_CONFIGS]
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
                anywhere = getattr(V.rules, "headline_any_hardware", False)      # a PREVIEW (Daniel decides): N stays the served line's
                ve = verifier_evaluated(g["template"])
                count_ve = getattr(V.rules, "headline_count_omitted", False)     # the one switch for verifier-evaluated parts
                strict = getattr(V.rules, "headline_strict_verifier", False)      # Daniel 2026-09-26: the verifier evaluates nothing
                parts = ve.parts if ve is not None else None

                def counted(c: Any, ve: Any = ve, count_ve: bool = count_ve) -> bool:
                    """Verifier-evaluated parts (the template's census record, or the cell's ``omitted`` label) count where the census
                    records a ruling that counts them (attention: red-team-flock-3's grant), else only under the switch."""
                    left = getattr(V, "omitted_of", {}).get(c.artifact)
                    if strict:
                        return not (ve is not None or left)
                    if ve is None and not left:
                        return True
                    return count_ve or bool(ve is not None and ve.counted)

                hits = [c for c, _ in mine if not c.provisional and not c.legacy.finding
                        and (anywhere or T.same_sku(V.prover_gpu.get(c.artifact), served.device)) and counted(c)]
                footnote = ({"parts": ve.parts, "share": ve.share,
                             "ruling": ve.ruling if ve.counted else f"counted by the verifier-evaluated switch; otherwise {ve.ruling}"}
                            if ve is not None else None)
                var = g["variable"]
                if hits and var:                # a cell covers the units at its own value of the variable (attention: its T), no other
                    best: dict[int, Any] = {}
                    rate: dict[int, float] = {}
                    klass: dict[int, Any] = {}

                    def offer(v: int, c: Any, P: float, k: Any = None) -> None:
                        if v not in best or (-P, c.artifact) < (-rate[v], best[v].artifact):
                            best[v], rate[v], klass[v] = c, P, k

                    for c in hits:
                        kc = getattr(V, "key_class_of", {}).get(c.artifact)
                        if kc and var == "T":   # a class cell: each T it lists, at that T's own throughput (heads / e2e_s), no other
                            for v, x in kc["credit"].items():
                                offer(v, c, x["P"], kc["range"])
                            continue
                        v = (getattr(c, "variables", None) or {}).get(var)
                        if v is not None:
                            offer(v, c, c.throughput["instances_per_second"])
                    vals = g["values"]
                    at = sorted(v for v in best if v in vals)
                    other = sorted(v for v in best if v not in vals)
                    if not vals:
                        uncovered.append({"subcircuit": g["subcircuit"], "template": g["template"], "parameters": g["parameters"],
                                          "display": g["display"], "units": g["units"], "reason": f"no per-{var} counts",
                                          "text": (f"cleared results at {var} = {runs(best)} exist, but the census records no VU counts per {var} "
                                                   f"for this workload, so they are not counted (never extrapolated)"),
                                          "results_elsewhere": []})
                        continue
                    per_value = []
                    for v in at:
                        c, P = best[v], rate[v]
                        per_value.append({var: v, "units": vals[v]["units"], "artifact": c.artifact, "P": P,
                                          "proving_seconds": vals[v]["units"] / P, "native_seconds": dict(vals[v]["native_seconds"]),
                                          **({"class": klass[v]} if klass[v] else {}),
                                          **({"source": src} if (src := input_source(V, c)) else {})})
                    missing = [v for v in vals if v not in best]
                    drawn = sorted(drawn_values(V, g["subcircuit"], var, w))
                    if per_value:
                        covered.append({"subcircuit": g["subcircuit"], "display": g["display"], "units": sum(x["units"] for x in per_value),
                                        "artifact": per_value[0]["artifact"], "configuration": best[at[0]].configuration,
                                        "scheme": best[at[0]].scheme, "P": None, "proving_seconds": sum(x["proving_seconds"] for x in per_value),
                                        "native_seconds": {b.id: sum(x["native_seconds"].get(b.id, 0.0) for x in per_value) for b in BASES
                                                           if g["native_seconds"][b.id] is not None},
                                        "variable": var, "covered_values": at, "drawn_values": drawn, "population_values": len(vals),
                                        "values_elsewhere": other, "per_value": per_value, "verifier_evaluated": footnote,
                                        **({"provenance": pv} if (pv := provenance(per_value, var)) else {})})
                    if missing:
                        uncovered.append({"subcircuit": g["subcircuit"], "template": g["template"], "parameters": g["parameters"],
                                          "display": g["display"], "units": sum(vals[v]["units"] for v in missing),
                                          "reason": f"not at every {var}" if per_value else f"no result at a served {var}",
                                          "text": (f"results at {var} = {runs(at) or 'none'} ({len(at)} of the run's {len(vals)} values"
                                                   + (f"; drawn {runs(drawn)}" if drawn else "") + f"); the other {len(missing)} values' units are "
                                                   f"not covered: nothing is extrapolated across {var}"
                                                   + (f"; results at {var} = {runs(other)} the run does not serve" if other else "")),
                                          "missing_values": len(missing), "results_elsewhere": []})
                    continue
                if hits:
                    c = min(hits, key=lambda x: (-x.throughput["instances_per_second"], x.artifact))
                    P = c.throughput["instances_per_second"]
                    covered.append({"subcircuit": g["subcircuit"], "display": g["display"], "units": g["units"], "artifact": c.artifact,
                                    "configuration": c.configuration, "scheme": c.scheme, "P": P, "proving_seconds": g["units"] / P,
                                    "verifier_evaluated": footnote})
                    continue
                elsewhere = []
                for c, _ in mine:
                    h = prover_line(c)
                    elsewhere.append({"artifact": c.artifact, "prover_hardware": h.id if h else None, "prover_gpu": V.prover_gpu.get(c.artifact),
                                      "on_served_hardware": T.same_sku(V.prover_gpu.get(c.artifact), served.device),
                                      "cleared": not c.provisional and not c.legacy.finding,
                                      "omitted": getattr(V, "omitted_of", {}).get(c.artifact) or (parts if parts and not counted(c) else None)})
                opening = getattr(census.SUBCIRCUIT_TEMPLATES.get(g["template"] or ""), "commitment_opening", None)
                if opening is not None and not elsewhere:
                    why, code = f"a commitment opening: {opening.note}", "commitment opening"
                elif g["subcircuit"] is None:
                    why, code = "the census binds no subcircuit to these parameters, so no result can name it", "no census subcircuit"
                elif not elsewhere:
                    why, code = "no admissible result on this subcircuit", "no result"
                elif any(e["on_served_hardware"] and e["cleared"] and e.get("omitted") for e in elsewhere):
                    left = next(e["omitted"] for e in elsewhere if e["on_served_hardware"] and e.get("omitted"))
                    why = (f"the cleared result leaves out verifier-evaluated parts ({left if isinstance(left, str) else '; '.join(map(str, left))}"
                           + (f"; {ve.share}" if ve is not None else "") + "); whether that counts as covered is a pending decision")
                    code = "omits verifier-evaluated parts"
                elif any(e["on_served_hardware"] for e in elsewhere):
                    why, code = "results on the served hardware are not cleared (provisional or an open finding)", "not cleared"
                else:
                    names = sorted({(census.HARDWARE[e["prover_hardware"]].name if e["prover_hardware"] else e["prover_gpu"] or "?") for e in elsewhere})
                    why, code = f"results only on {', '.join(names)}, not the served {served.name}", f"only on {', '.join(names)}"
                uncovered.append({"subcircuit": g["subcircuit"], "template": g["template"], "parameters": g["parameters"], "display": g["display"],
                                  "units": g["units"], "reason": code, "text": why, "results_elsewhere": elsewhere})
            per_basis = {}
            for b in BASES:
                modelled = {g["subcircuit"]: g for g in subs if g["native_seconds"][b.id] is not None}
                parts = [((x["native_seconds"][b.id],) * 2 if "native_seconds" in x else modelled[x["subcircuit"]]["native_seconds"][b.id])
                         for x in covered if x["subcircuit"] in modelled]
                cov = _sum(parts)
                prove = sum(x["proving_seconds"] for x in covered if x["subcircuit"] in modelled)
                share = _share(cov, (totals[b.id][0] - cov[0], totals[b.id][1] - cov[1]))
                overhead = (prove / cov[1], prove / cov[0]) if cov[0] > 0 else None
                per_basis[b.id] = {"share": share, "overhead": overhead, "covered_native_seconds": cov, "proving_seconds": prove,
                                   "unmodelled": [g["display"] for g in subs if g["native_seconds"][b.id] is None]}
            rows.append({"backend": bid, "covered": covered, "uncovered": uncovered, "bases": per_basis})
        out.append({"workload": w["id"], "display": w["display"], "served_hardware": served.id, "served_display": f"{served.name} · "
                    f"{census.DATATYPES[served.dtype].display}", "vus": w["vus"], "not_evaluable_vus": w["not_evaluable_vus"],
                    "native_seconds": totals, "subcircuits": subs, "backends": rows,
                    # verified proofs under the 2^-128 bar on this workload's own inputs: shown, never counted as covered
                    "below_bar": [{"artifact": e["artifact"], "backend_family": e["backend_family"], "definition": e["definition"], "B": e["B"],
                                   "soundness_bits": min((m["soundness_bits"] for m in e["modes"].values() if m.get("soundness_bits")), default=None)}
                                  for e in getattr(V, "below_bar", []) if e.get("served_row") is not None
                                  and e["served_row"] == (w.get("source") or {}).get("row")]})
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
            "vus": r["vus"], "not_evaluable_vus": r["not_evaluable_vus"], "below_bar": r.get("below_bar", []),
            "native_seconds": {b: _j(v) for b, v in r["native_seconds"].items()},
            "subcircuits": [{"subcircuit": g["subcircuit"], "template": g["template"], "parameters": g["parameters"], "display": g["display"],
                             "vu_families": g["vu_families"], "vus": g["vus"], "units": g["units"],
                             "variable": g["variable"], "values": runs(g["values"]) if g["values"] else None,
                             "native_seconds": {b: _j(v) for b, v in g["native_seconds"].items()}, "unmodelled": g["unmodelled"],
                             "share": {b: (None if v is None else {**_j(v), "display": fmt_pct(v)}) for b, v in g["share"].items()}}
                            for g in r["subcircuits"]],
            "backends": [{"backend": row["backend"],
                          "covered": [{**x, "result": result_ids.get(x["artifact"]),
                                       **({"per_value": [{**pv, "result": result_ids.get(pv["artifact"])} for pv in x["per_value"]]}
                                          if x.get("per_value") else {})} for x in row["covered"]],
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
           "Per served workload (`benchmarks/census/workloads.json`) and backend. Units = VUs × subcircuits per VU, per bound subcircuit (the "
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
        if r.get("below_bar"):
            defs = sorted({x["definition"].split("{")[0] for x in r["below_bar"] if x.get("definition")})
            bits = [x["soundness_bits"] for x in r["below_bar"] if x.get("soundness_bits")]
            out += [f"Below the 128-bit bar, verified but not counted: {len(r['below_bar'])} proofs on this workload's inputs "
                    f"({', '.join(defs)}; {min(bits):.1f}–{max(bits):.1f} bits)." if bits else
                    f"Below the 128-bit bar, verified but not counted: {len(r['below_bar'])} proofs on this workload's inputs.", ""]
        by_sub: dict[str, list[str]] = {}
        notes: dict[str, str] = {}
        sources: dict[str, str] = {}
        for row in r["backends"]:
            for x in row["covered"]:
                mark = ""
                if x.get("verifier_evaluated"):
                    ve = x["verifier_evaluated"]
                    notes.setdefault(x["display"], f"{ve['parts']} ({ve['share']}) is checked natively by the verifier, not proven; "
                                                   f"counted: {ve['ruling']}")
                    mark = "†"
                if x.get("provenance"):
                    sources.setdefault(x["display"], x["provenance"]["text"])
                    mark += "‡"
                per = (f" at {x['variable']} = {runs(x['covered_values'])} ({len(x['covered_values'])} of {x['population_values']})"
                       if x.get("variable") else "")
                by_sub.setdefault(x["subcircuit"], []).append(row["backend"] + mark + per)
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
        for row in r["backends"]:
            for x in row["covered"]:
                if x.get("variable"):
                    v = x["variable"]
                    out += [f"{x['display']}, {row['backend']}, per {v}: covered {v} = {runs(x['covered_values'])} "
                            f"({len(x['covered_values'])} of the run's {x['population_values']} values, each weighted by its units in the "
                            f"template mix); drawn {v} = {runs(x['drawn_values']) if x['drawn_values'] else 'not recorded'}"
                            + (f"; {', '.join(map(str, sorted(set(x['drawn_values']) - set(x['covered_values']))))} drawn but not covered"
                               if set(x["drawn_values"]) - set(x["covered_values"]) else "")
                            + f". Units at any other {v} are not covered: nothing is extrapolated.", ""]
        out += [f"† Verifier-evaluated, {d}: {n}." for d, n in notes.items()] + ([""] if notes else [])
        out += [f"‡ Synthetic inputs, {d}: {n}." for d, n in sources.items()] + ([""] if sources else [])
    return out
