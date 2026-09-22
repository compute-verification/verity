"""Bound + prover-overhead record over a composed run's declared plan (lane INT-PAD, GUIDANCE §29).

    python -m e2e.bound_record --dir RECORD --out OUT [--adapter DIR] [--bound-steps 200000]
    python -m e2e.bound_record --dir RECORD --check OUT   # cheap re-check of a committed record

Two things, both derived from the record and the frozen machinery -- nothing here edits the fold
(`src/veritor/analysis/bound.py` / `src/veritor/plans/bound.py` are wave-frozen; this module only calls them
through the existing adapter `veritor.frontier.bound_adapter`):

* **Bound rows** -- the record's declared plan units (``plan/validator_report.json``; B0: the layer
  ``LayerPost{C}`` with partitions P1 / P2 from ``run_composed.unit_plans``) are rebuilt, re-validated, and
  folded: ``veritor.plans.bound.hierarchical_table`` gives the kind table, ``veritor.frontier.bound_adapter``
  wraps it (label ``exact``) and emits one ``veritor/bound-report/v1`` object per (unit, partition) at the
  record's declared policy point, plus a sweep of fold rows over ``delta`` in {1e-6, 1e-12} x ``q`` in
  {1, measured} x ``s`` in {declared, effective}.  The recorded bound (``validator_report.json``, computed by
  ``validate_unit`` at ``delta = 1/64``) is re-folded at its own point and must agree.  The ``capped`` flag is
  carried on every row: a capped ``U`` *is* the unit's interface width (``out_bits``), and the request-level
  ``W`` treatment of the padded output family is PAD-BOUND's, not this record's.
* **Prover-overhead record** -- GUIDANCE §29 rule 7: headline cost =
  ``(replay/preparation + commitment + proof generation + prover fixed) / T_native``.  Every component is a
  seconds value read from the record (``summary.json``; the prover's own ``adapter/selective/
  selective_prover.json`` timing block; ``adapter/production/timings.json`` recording-off passes for
  ``T_native``), each with a provenance string and a label following ``veritor.frontier.costs`` (``measured``
  for a recorded number, ``derived`` for arithmetic over measured numbers, ``unknown`` when the record does
  not carry it).  Verifier work is **reported separately and labelled diagnostic** (rule 6: verifier-side
  re-execution is a labelled diagnostic, not headline integrity support), as is the serving-profile overhead
  (recorded pass vs recording-off) and the second process's cost.

``--check`` mode re-reads a written record, re-folds every bound report's kind table with the frozen fold
(``check_report_against_fold``) and re-derives the overhead arithmetic from the record's own components --
cheap (no circuit rebuild), the green-loop stage of ``scripts/e2e_b0.sh``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from e2e.run_composed import static_configuration, unit_plans, write_json  # noqa: E402

SCHEMA = "verity-vllm/e2e-composed/bound-cost/v1"
RULE7 = ("GUIDANCE §29 rule 7: prover overhead = (replay/preparation + commitment + proof generation + prover fixed) "
         "/ T_native; verifier work and serving-profile overhead are reported separately")
RULE6 = ("GUIDANCE §29 rule 6: headline integrity results require plans that are both replay-supported and "
         "proof-supported; verifier-side re-execution stays as a labelled diagnostic only")
DELTAS = (Fraction(1, 10**6), Fraction(1, 10**12))


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _cell(value: float | None, unit: str, label: str, provenance: str, note: str | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {"value": value if value is None else round(float(value), 6), "unit": unit, "label": label, "provenance": provenance}
    if note:
        d["note"] = note
    return d


def _find_adapter(record: Path, adapter_arg: str | None) -> Path | None:
    """The adapter directory: the record's own, or a read-only auxiliary copy (--adapter, e.g. the archive)."""
    for c in ([Path(adapter_arg)] if adapter_arg else []) + [record / "adapter"]:
        if (c / "selective" / "selective_prover.json").exists() or (c / "production" / "timings.json").exists():
            return c
    return None


# -- prover overhead (rule 7) --------------------------------------------------------------------------------------


def overhead_record(record: Path, summary: dict[str, Any], adapter: Path | None) -> dict[str, Any]:
    stages = summary.get("stages") or {}
    prove = stages.get("prove") or {}
    verify = stages.get("verify") or {}
    proofs = prove.get("proofs") or {}

    # T_native: the recording-off passes of the production adapter (native serving of the whole workload)
    t_native = None
    t_native_cell = _cell(None, "s", "unknown", "adapter/production/timings.json not present in this record copy",
                          "the committed fixture omits adapter/; pass --adapter <archive copy> (READ-ONLY) to price T_native")
    serving_profile = None
    if adapter is not None and (adapter / "production" / "timings.json").exists():
        t = _load(adapter / "production" / "timings.json")
        offs = [float(x["wall_s"]) for x in (t.get("recording_off") or []) if x.get("wall_s")]
        rec_pass = float((t.get("recorded_pass") or {}).get("wall_s") or 0) or None
        if offs:
            t_native = sum(offs) / len(offs)
            t_native_cell = _cell(t_native, "s", "measured", f"{adapter}/production/timings.json recording_off[*].wall_s (mean of {len(offs)})",
                                  "native engine pass of the workload on the record's host, recording off (engine build excluded)")
        if rec_pass and t_native:
            serving_profile = {"recorded_pass": _cell(rec_pass, "s", "measured", "production/timings.json recorded_pass.wall_s"),
                               "recording_off": _cell(t_native, "s", "measured", "production/timings.json recording_off mean"),
                               "ratio": _cell(rec_pass / t_native, "x", "derived", "recorded_pass / recording_off"),
                               "note": "serving-profile overhead, reported separately per rule 7"}

    # prover-side seconds: RC's selective prover writes its own timing block; INT's full replay records stage seconds only
    pt: dict[str, float] = {}
    pt_src = ""
    if adapter is not None and (adapter / "selective" / "selective_prover.json").exists():
        pt = {k: float(v) for k, v in (_load(adapter / "selective" / "selective_prover.json").get("timing") or {}).items()}
        pt_src = "adapter/selective/selective_prover.json -> timing"

    def ptc(keys: list[str], what: str) -> dict[str, Any]:
        if pt and all(k in pt for k in keys):
            return _cell(sum(pt[k] for k in keys), "s", "measured", f"{pt_src}: {' + '.join(keys)}", what)
        return _cell(None, "s", "unknown", f"{pt_src or 'no prover timing block in this record copy'}", what)

    # proof generation and its one-time setup, from the prove stage's per-mode rows
    gen = sum(float((proofs.get(m) or {}).get("prove_seconds") or 0) for m in ("core", "compressed")) or None
    setup = sum(float((proofs.get(m) or {}).get("setup_seconds") or 0) for m in ("core", "compressed")) or None
    execute_s = None  # the guest execute pass is inside the stage wall but not recorded as its own number

    components = {
        "replay_preparation": ptc(["replay_run_s"], "the prover's replay of J (its own batch), capture on"),
        "commitment": ptc(["phase0_boundaries_s", "phase0_weights_write_s", "phase0_commit_s", "phase2_boundaries_s", "phase2_total_s"],
                          "phase-0 boundary + weights write/commit and the whole of phase 2 (interiors: capture write + trees); "
                          "phase2_total_s includes phase2_commit_s -- the residual is capture finalisation, kept on the commitment side"),
        "proof_generation": _cell((gen or 0) + (execute_s or 0) if gen else None, "s", "measured" if gen else "unknown",
                                  "summary.stages.prove proofs.{core,compressed}.prove_seconds",
                                  "both modes were generated and are both counted; the guest execute pass is inside the stage wall "
                                  "but not recorded as its own number and is not included"),
        "prover_fixed": _cell(((pt.get("engine_build_s") or 0) + (pt.get("warmup_s") or 0) + (setup or 0)) or None, "s",
                              "measured" if pt and setup else "unknown",
                              f"{pt_src}: engine_build_s + warmup_s; summary.stages.prove proofs.*.setup_seconds",
                              "engine build + warmup + SP1 setup (both modes); one-time per host/vk, included in the headline per rule 7 and broken out here"),
    }
    excluded = {"wait_r_q_s": _cell(pt.get("wait_r_q_s"), "s", "measured" if "wait_r_q_s" in pt else "unknown", pt_src,
                                    "protocol idle: the prover waiting for the verifier's r_q; not prover work"),
                "control_replay": _cell(float(((stages.get("preparation") or {}).get("control") or {}).get("seconds") or 0) or None, "s",
                                        "measured" if (stages.get("preparation") or {}).get("control") else "unknown",
                                        "summary.stages.preparation.control", "the full-replay control is a batch-invariance audit, not part of the protocol")}

    num = [c for c in components.values() if c["value"] is not None]
    numerator = sum(c["value"] for c in num) if len(num) == len(components) else None
    marginal = None
    if components["replay_preparation"]["value"] is not None and components["commitment"]["value"] is not None and gen:
        marginal = components["replay_preparation"]["value"] + components["commitment"]["value"] + gen + (execute_s or 0)

    doc: dict[str, Any] = {
        "definition": RULE7,
        "T_native": t_native_cell,
        "components": components,
        "excluded": excluded,
        "numerator": _cell(numerator, "s", "derived", "sum of the four components") if numerator else _cell(None, "s", "unknown", "a component is unknown"),
        "prover_overhead": _cell(numerator / t_native if (numerator and t_native) else None, "x", "derived",
                                 "numerator / T_native", "headline, per rule 7 (prover fixed included)"),
        "prover_overhead_marginal": _cell(marginal / t_native if (marginal and t_native) else None, "x", "derived",
                                          "(replay + commitment + proof generation) / T_native", "marginal: prover fixed (engine build, warmup, SP1 setup) excluded"),
        "q_note": ("the protocol replayed J only (q as recorded in summary.scope); the ratio prices the protocol as run, "
                   "not an extrapolation to q = 1"),
        "reported_separately": {
            "verifier_work": {**_cell(float(verify.get("seconds") or 0) or None, "s", "measured", "summary.stages.verify.seconds"),
                              "role": "diagnostic", "rule": RULE6,
                              "detail": "the verifier's re-execution of every sampled VU (same-runtime + independent relations); "
                                        "of the sampled families only GEMM coordinates are proof-supported in this record"},
            "serving_profile_overhead": serving_profile,
        },
    }
    return doc


# -- bound rows ----------------------------------------------------------------------------------------------------


def bound_rows(record: Path, manifest: dict[str, Any], summary: dict[str, Any], out: Path, bound_steps: int) -> dict[str, Any]:
    from veritor.core.policy import VerificationPolicy
    from veritor.frontier.bound_adapter import Table, kind_table_to_json
    from veritor.plans.bound import BoundLimits, hierarchical_table
    from veritor.plans.validate import validate_plan

    report = _load(record / "plan" / "validator_report.json")
    unit_reports: dict[str, dict[str, Any]] = report.get("units") or {"layer": {"partitions": report["partitions"], "plan_digest": report.get("plan_digest")}}
    unit_order: list[str] = list(report.get("unit_order") or unit_reports.keys())
    config = static_configuration(summary["case"])

    # the record's policy points
    cmd = str(manifest.get("command") or "")
    s_declared = Fraction("1/4096")  # run_composed default; --s on the command line overrides
    for tok, nxt in zip(cmd.split(), cmd.split()[1:]):
        if tok == "--s":
            s_declared = Fraction(nxt)
    vstage = (summary.get("stages") or {}).get("verify") or {}
    pop = vstage.get("population")
    population = int(pop.get("vus") if isinstance(pop, dict) else (pop or 0))
    checked = int(vstage.get("sample_union") or 0)
    if not checked:
        checked = int(vstage.get("certified_sample") or 0) + int(vstage.get("coverage_sample") or 0)
    s_effective = Fraction(checked, population) if population and checked else None
    q_measured = None
    qm = ((summary.get("stages") or {}).get("preparation") or {}).get("q_measured")
    if isinstance(qm, dict) and qm.get("positions_replayed") and qm.get("positions_total"):
        q_measured = Fraction(int(qm["positions_replayed"]), int(qm["positions_total"]))
    elif isinstance(qm, dict) and qm.get("fraction"):
        q_measured = Fraction(str(qm["fraction"])).limit_denominator(10**6)

    units_out: dict[str, Any] = {}
    reports_written: list[str] = []
    for unit in unit_order:
        rec_u = unit_reports.get(unit) or {}
        Q, c, partitions = unit_plans(config, unit)
        per_partition: dict[str, Any] = {}
        for tag, plan in partitions:
            v = validate_plan(c, plan)
            if v.status != "valid":
                per_partition[tag] = {"status": v.status, "error": "plan did not re-validate; no fold"}
                continue
            tres = hierarchical_table(c, plan, v, BoundLimits(max_reach_steps=bound_steps))
            table = Table(tres.table, Q.name, plan.name, "exact",
                          f"veritor.plans.bound.hierarchical_table over the record's declared plan (run_composed.unit_plans {unit!r}/{tag}); "
                          f"validated here; fold = frozen veritor.analysis.bound.bound via veritor.frontier.bound_adapter")
            # the canonical bound-report/v1 at the record's declared point (q = 1 as the schema's to_report emits)
            rep = table.to_report(s_declared, DELTAS[0], program_digest=str(manifest["inputs"].get("program_digest") or ""),
                                  plan_digest=str(rec_u.get("plan_digest") or report.get("plan_digest") or ""))
            rep_path = out / f"bound-report-{unit.replace(':', '_')}-{tag}.json"
            write_json(rep_path, rep)
            reports_written.append(rep_path.name)
            # the sweep: delta x q x s, each row the frozen fold's answer
            rows = []
            for delta in DELTAS:
                for q_name, q in (("1", Fraction(1)),) + (((f"measured {q_measured}", q_measured),) if q_measured else ()):
                    for s_name, s in (("declared " + str(s_declared), s_declared),) + (((f"effective {checked}/{population}", s_effective),) if s_effective else ()):
                        b = table.fold(s, delta, q=q)
                        rows.append({"delta": str(delta), "q": q_name, "s": s_name, "U_bits": float(b.bits), "capped": bool(b.capped),
                                     "out_bits": b.out_bits, "knapsack_bits": None if b.knapsack_bits == float("inf") else float(b.knapsack_bits),
                                     "laplace_bits": None if b.laplace_bits == float("inf") else float(b.laplace_bits)})
            # the recorded bound's own point (validate_unit: delta = 1/64, policy from the run) must reproduce
            recorded = ((rec_u.get("partitions") or {}).get(tag) or {}).get("bound") or {}
            recorded_check = None
            if recorded.get("bits") is not None and recorded.get("policy"):
                rb = table.fold(Fraction(recorded["policy"]["s"]), Fraction(1, 64), q=Fraction(recorded["policy"]["q"]))
                recorded_check = {"recorded_bits": recorded["bits"], "refolded_bits": float(rb.bits), "agrees": abs(float(rb.bits) - float(recorded["bits"])) < 1e-6,
                                  "point": {"delta": "1/64", **recorded["policy"]}}
            per_partition[tag] = {"status": "ok", "report": rep_path.name, "vus": len(plan.verification_units), "cus": len(plan.commitment_units),
                                  "vu_kinds": table.vu_kinds, "cu_kinds": table.cu_kinds, "out_bits": table.output_bits,
                                  "kind_table_digest": kind_table_to_json(tres.table)["digest"], "fallback_units": tres.fallback_units,
                                  "rows": rows, "recorded_bound_check": recorded_check}
        units_out[unit] = per_partition
        del Q, c, partitions

    return {"units": units_out, "reports": reports_written, "bound_steps": bound_steps,
            "policy_points": {"s_declared": str(s_declared), "s_effective": (str(s_effective) if s_effective else None),
                              "q_measured": (str(q_measured) if q_measured else None), "deltas": [str(d) for d in DELTAS]},
            "notes": [
                "RC's Bound is stated for Bernoulli(q); the record's selector is fixed-size (k requests), q recorded as measured, "
                "not substituted (e2e-composed.md §11.8)",
                "a capped row's U equals the unit's interface width (out_bits): the fold certifies nothing narrower than the layer "
                "interface at these points; the request-level W of the padded output family (n, y_0..y_{n-1}) is PAD-BOUND's and is "
                "not claimed here",
                "label 'exact' = RA's hierarchical table over the validated plan; the fold itself is the frozen bound (wave rule: "
                "nobody edits src/veritor/analysis/bound.py or src/veritor/plans/bound.py -- this module only calls them)",
            ]}


# -- check mode ----------------------------------------------------------------------------------------------------


def check(record: Path, out: Path) -> int:
    from veritor.frontier.bound_adapter import check_report_against_fold

    doc = _load(out / "bound_record.json")
    problems: list[str] = []
    for name in doc["bound"]["reports"]:
        rep = _load(out / name)
        reported, refolded = check_report_against_fold(rep)
        agree = abs(reported - refolded) < 1e-6
        print(f"[bound_record] check {name}: reported U={reported} refolded U={refolded} agree={agree}")
        if not agree:
            problems.append(f"{name}: reported {reported} != frozen fold {refolded}")
    # overhead arithmetic re-derived from the components stored in the record itself
    oh = doc["overhead"]
    comp = {k: v.get("value") for k, v in oh["components"].items()}
    if all(v is not None for v in comp.values()):
        num = sum(comp.values())
        if abs(num - (oh["numerator"]["value"] or 0)) > 1e-3:
            problems.append(f"numerator {oh['numerator']['value']} != recomputed {num}")
        tn = oh["T_native"]["value"]
        if tn and abs(num / tn - (oh["prover_overhead"]["value"] or 0)) > 1e-3:
            problems.append(f"prover_overhead {oh['prover_overhead']['value']} != recomputed {num / tn}")
    for row in ((doc["bound"].get("units") or {}).get("layer") or {}).values():
        rc = row.get("recorded_bound_check")
        if rc and not rc.get("agrees"):
            problems.append(f"recorded bound does not re-fold: {rc}")
    status = "ok" if not problems else "FAIL"
    print(f"[bound_record] check {status}" + (f": {problems[0]}" if problems else f" ({len(doc['bound']['reports'])} report(s) re-folded; overhead arithmetic re-derived)"))
    return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="a run_composed record directory (committed fixture or archive copy)")
    ap.add_argument("--out", required=True, help="where bound_record.json and the bound-report/v1 files go (never the record itself)")
    ap.add_argument("--adapter", default=None, help="read-only auxiliary adapter dir (e.g. the archive's) when the record copy omits adapter/")
    ap.add_argument("--bound-steps", type=int, default=200_000, help="reach cap for hierarchical_table (the recorded run's value)")
    ap.add_argument("--check", action="store_true", help="re-check an existing --out against the frozen fold instead of rebuilding")
    a = ap.parse_args(argv)
    record = Path(a.dir).resolve()
    out = Path(a.out).resolve()
    if a.check:
        return check(record, out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = _load(record / "manifest.json")
    summary = _load(record / "summary.json")
    adapter = _find_adapter(record, a.adapter)
    t0 = time.time()
    overhead = overhead_record(record, summary, adapter)
    bounds = bound_rows(record, manifest, summary, out, a.bound_steps)
    doc = {"schema": SCHEMA, "record": str(record), "case": summary.get("case"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "program_digest": manifest["inputs"].get("program_digest"), "plan_digest": manifest["inputs"].get("plan_digest"),
           "adapter_source": str(adapter) if adapter else None, "seconds": round(time.time() - t0, 1),
           "overhead": overhead, "bound": bounds,
           # OUT16 (GUIDANCE §32.5/.7, landed c4d52ad0): declare/enforce/account with declared_domain = 16
           # on the 16 token outputs of a B0 request -- W_out 512 -> 256 bits.  This record's plan predates
           # the declaration (VerificationPlan.declared_domains empty; the recorded description has no
           # declared_domain on sampler.select), so the narrowed number applies to the NEXT capture under
           # profile llama-eager-batch-invariant-v3, not retroactively to this one -- account without
           # enforce is the forbidden shortcut (§31.2) and is not taken here.
           "w_out": {"convention": "OUT16 declared_domain=16; kind tables charge min(wire, declared)",
                     "this_record_bits_per_request": 512,
                     "with_v3_profile_bits_per_request": 256,
                     "binding": "veritor/plan/v1 declared_domains + description sampler.select declared_domain "
                                "(profile llama-eager-batch-invariant-v3)"}}
    write_json(out / "bound_record.json", doc)
    po = overhead["prover_overhead"]["value"]
    print(f"[bound_record] {summary.get('case')}: prover overhead {po and round(po, 1)}x (T_native {overhead['T_native']['value']} s); "
          f"{len(bounds['reports'])} bound-report(s) written; verifier work reported separately as diagnostic -> {out / 'bound_record.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
