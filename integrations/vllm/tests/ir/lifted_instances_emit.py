"""Emit the LIFTED derived body of a real `build_request` dir in the Build's own `instances.json.gz` row schema (`cb-a/instance-sequence/v1`)
so reqval's RequiredValues traversal (`verity_capture.commit.required_values audit|diff --program <dir>`) can read Lift(P) exactly as it reads
P — reqval M-0591: the traversal unlifts `Lifted[<id>]_v1{statics}` before the family lookup, so RequiredValues(Lift(P), Q) must equal
RequiredValues(P, Q) row for row (the guard rows carry no identity; a lifted word is one value of w+1 bits, so no element count moves).

    python -m tests.ir.lifted_instances_emit --build-request <dir> --out <dir_lifted> --eos 0 --cap 32 [--workload-program <row>/build_workload/workload_program.json | --lag N]

Writes `<out>/instances.json.gz` (rows: the lifted body's nodes in order — every non-guard node carries the ORIGINAL row's `name` (op path)
and its lifted spec id; guard nodes are named `lifted_guard/<k>`; `args` are the lifted body's own refs in the rows' JSON form), `result.json`
(the original with `program.digest` = the lifted Program's and a `lifted` block naming the source) and `lifted_emit.json` (the counts and the
TRUE lifted widths, R17-6(b)).  R17-1: the LAG is the Build's DECLARED value, read from the row's `workload_program.json` through the same rule
the record tool and GP-01/GM-01 apply (`L.declared_lag_of_row`; one rule, two consumers); `--lag` is accepted only as an explicit declaration
for a synthetic source and must agree with the row's when the row declares one; NO default.  Lifted-padded: a Program-side artifact, no record fact."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from verity_vllm.lifting import record as R
from verity_vllm.ir import workload as W
from verity.ir.codec import _flat_parts, _spec_id
from verity.ir.defs import REGISTRY
from verity_vllm.registry import lifted as L


def _parts_json(refs, pmap=None) -> list:
    """The rows' `args` form: a list of `[space, idx, lo, hi]` runs (the Build writes every arg as such a list).  `pmap`: component
    parameter j (one Input batch per prescribed leaf, `request_component`) -> (wrapper param index, element offset of the leaf in it) --
    the Build's rows address the WRAPPER's parameters (`prompt`, `weights`) with element offsets into the flattened record."""
    out: list = []
    for run in _runs(refs):
        space, idx, lo, hi = run
        if space == "p" and pmap is not None:
            widx, off = pmap[idx]
            run = [space, widx, off + lo, off + hi]
        out.append(run)
    return out


def _runs(refs) -> list:
    out: list = []
    for r in _flat_parts(refs):
        j = r.to_json()
        if isinstance(j, dict) and "affine" in j:
            space, idx, base, count = j["affine"]
            out.append([space, idx, base, base + count])
        else:  # a non-affine part: enumerate its leaves as maximal runs (rare in derived bodies; exact, never approximated)
            run = None
            for space, idx, e in r:
                if run and run[0] == space and run[1] == idx and run[3] == e:
                    run[3] = e + 1
                else:
                    if run:
                        out.append(run)
                    run = [space, idx, e, e + 1]
            if run:
                out.append(run)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eos", required=True, help="comma-separated declared EOS ids")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--lag", type=int, default=None, help="R17-1: an EXPLICIT declaration (synthetic sources); must agree with the row's when it declares one; no default")
    ap.add_argument("--workload-program", default=None, help="the row's build_workload/workload_program.json (the Build's declared lag: lag.value / "
                    "stop_rule.lag / stop_rule.eos_lag_rows); looked for at <build-request>/../build_workload/workload_program.json when omitted")
    ap.add_argument("--boundary-prefix", default=L.BOUNDARY_PREFIX)
    a = ap.parse_args(argv)
    src, out = Path(a.build_request), Path(a.out)
    # R17-1: the declared lag, through the ONE rule (L.declared_lag_of_row -> GP-01 eos_lag_rows_declared); nothing defaulted, nothing inferred
    wp_path = Path(a.workload_program) if a.workload_program else src.parent / "build_workload" / "workload_program.json"
    declared_lag, declared_field = L.declared_lag_of_row(json.loads(wp_path.read_text())) if wp_path.exists() else (None, None)
    if declared_lag is None and a.lag is None:
        raise SystemExit(f"no declared LAG: {wp_path} {'declares none of ' + ', '.join(L.LAG_DECLARED_FIELDS) if wp_path.exists() else 'does not exist'} "
                         "and no --lag was given (R17-1: the lag is a declared specialization parameter; a pre-R17-1 Build declares none and is not a lifted "
                         "Program of record; nothing is defaulted)")
    if declared_lag is not None and a.lag is not None and a.lag != declared_lag:
        raise SystemExit(f"--lag {a.lag} contradicts the row's declared LAG={declared_lag} ({declared_field}): the Build's declaration rules")
    lag = L.check_lag(declared_lag if declared_lag is not None else a.lag)
    lag_source = f"{wp_path}:{declared_field}" if declared_lag is not None else "--lag (explicit declaration on the command line; no row declaration)"
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(src / "instances.json.gz", "rt") as f:
        inst = json.load(f)
    rows_u = inst["rows"]
    params = [(str(n), str(t)) for n, t in (inst.get("params") or [])] or None
    desc = R.load_descriptor(str(src))
    comp = W.request_component(desc, REGISTRY, params)
    eos = [int(x) for x in a.eos.split(",")]
    lifted = L.lift_request_component(comp.fn, eos, max_tokens=a.cap, lag=lag, boundary_prefix=a.boundary_prefix)
    b = lifted.body
    guards = set(lifted.guard_nodes)  # type: ignore[attr-defined]
    n_non_guard = len(b.nodes) - len(guards)
    if n_non_guard != len(rows_u):
        raise SystemExit(f"lifted body has {n_non_guard} non-guard nodes but the Build wrote {len(rows_u)} rows: no row correspondence, nothing emitted")
    # component param j -> (wrapper param index, element offset): the leaves of each wrapper group in order (request_component's tiling)
    pmap: dict[int, tuple[int, int]] = {}
    if params is not None:
        widx = {name: i for i, (name, _) in enumerate(params)}
        off: dict[str, int] = {}
        for j, (g, (_, t)) in enumerate(zip(comp.groups, comp.fn.body.params)):
            pmap[j] = (widx[g], off.get(g, 0))
            off[g] = off.get(g, 0) + t.leaves
    else:
        pmap = None  # type: ignore[assignment]
    rows_l, j = [], 0
    for k, n in enumerate(b.nodes):
        spec = _spec_id(n.fn)
        if k in guards:
            name = f"lifted_guard/{k}"
        else:
            u = rows_u[j]
            j += 1
            if L.unlifted_id(spec) != _spec_id(comp.fn.body.nodes[j - 1].fn):  # PAD-03 modulo the wrapper and the class vector (v0.7 A11 / G1)
                raise SystemExit(f"node {k}: lifted spec {spec} is not the lift of the Build's row {j - 1} ({u['spec']})")
            name = u["name"]
        rows_l.append({"i": k, "spec": spec, "form": n.form, "name": name, "args": [_parts_json(arg.refs, pmap) for arg in n.args]})
    body = {"schema": inst.get("schema", "cb-a/instance-sequence/v1"), "params": inst.get("params"), "rows": rows_l}
    text = json.dumps(body, separators=(",", ":"))
    body["digest"] = hashlib.sha256(text.encode()).hexdigest()
    with gzip.open(out / "instances.json.gz", "wt") as f:
        json.dump({"schema": body["schema"], "digest": body["digest"], "params": body["params"], "rows": rows_l}, f, separators=(",", ":"))
    res = json.load(open(src / "result.json")) if (src / "result.json").exists() else {"schema": "cb-a/derive_step/v3", "program": {}, "note": "source dir had no result.json"}
    res.setdefault("program", {})
    res["lifted"] = {"source_build_request": str(src), "source_program_digest": comp.program_digest, "lifted_id": lifted.id[:200],
                     "eos": eos, "cap": a.cap, "lag": lag, "lag_source": lag_source, "lag_declared_field": declared_field,
                     "guard_nodes": len(guards), "label": R.LABEL, "rules_version": L.LIFTED_RULES_VERSION, "program_rules_version": L.PADDED_RULES_VERSION,
                     "param_classes": list(getattr(lifted, "param_classes", ()))}  # G1: which prescribed inputs stay ordinary (unwidened)
    res["program"]["digest"] = R.lifted_program_digest(lifted)  # the ONE function the record tool admits in R3a / R7 / W5 (F-pad-30)
    res["program"]["unlifted_digest"] = comp.program_digest
    (out / "result.json").write_text(json.dumps(res, indent=1))
    # R17-6(b) (SPEC v0.5 A4.4): the TRUE lifted widths of the emitted rows -- Σ (w+1) over each node's output leaves, the payload total
    # beside it for information, the widest node output, and how many rows carry an output wider than the active-reference W_MAX
    widths_l = [sum(R._leaf_widths(n.out)) for n in b.nodes]
    widths_u = [sum(R._leaf_widths(n.out)) for n in comp.fn.body.nodes]
    widest = max(range(len(b.nodes)), key=lambda k: widths_l[k])
    non_guard_bits = sum(x for k, x in enumerate(widths_l) if k not in guards)
    non_guard_leaves = sum(len(R._leaf_widths(n.out)) for k, n in enumerate(b.nodes) if k not in guards)
    width_summary = {"kind": L.USE_KIND_LIFTED, "rules_version": L.LIFTED_RULES_VERSION, "rule": "true lifted width Σ(w+1); nothing split (R17-6(b))",
                     "lifted_bits_total": sum(widths_l), "lifted_bits_non_guard": non_guard_bits, "guard_bits": sum(widths_l) - non_guard_bits,
                     "unlifted_bits_total": sum(widths_u), "grown_by_bits": non_guard_bits - sum(widths_u),  # == one tag bit per leaf, row for row
                     "leaves_total": sum(len(R._leaf_widths(n.out)) for n in b.nodes), "leaves_non_guard": non_guard_leaves,
                     "widest_row": {"i": widest, "spec": rows_l[widest]["spec"][:80], "lifted_bits": widths_l[widest]},
                     "rows_with_output_over_w_max_active_reference": sum(1 for x in widths_l if x > L.W_MAX),
                     "w_max_active_reference": L.W_MAX, "w_max_lifted": L.W_MAX_LIFTED, "w_max_lifted_status": L.W_MAX_LIFTED_STATUS}
    summary = {"rows_unlifted": len(rows_u), "rows_lifted": len(rows_l), "guard_rows": len(guards), "unlifted_gates": comp.fn.body.gates,
               "lifted_gates": b.gates, "instances_digest": body["digest"], "source_program_digest": comp.program_digest,
               "lifted_program_digest": res["program"]["digest"], "label": R.LABEL, "rules_version": L.LIFTED_RULES_VERSION,
               "lag": lag, "lag_source": lag_source, "widths": width_summary}
    (out / "lifted_emit.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
