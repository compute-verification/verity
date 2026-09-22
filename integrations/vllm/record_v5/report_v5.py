#!/usr/bin/env python
"""[A9 contract 2026-09-10] ``REC/report_v5.md`` + ``report_v5.json`` -- the run's report per docs/PROTOCOL.md, Reporting requirements:
source revision / patch, circuit / partition / population identities, workload identity, hardware / numerics, actual q, s, B_ex, epoch,
randomness mode, counts (inputs / computed gates / VUs / RUs, widths, |J|, |F|, |Q|, coverage, verifier outcome), the eight cost classes
(MEASURED / UNAVAILABLE with reason; MODELED only in the predictions section, labelled), the certificate for exactly this configuration,
failed / aborted attempts (every phase with rc != 0 from logs/run.log), the conformance-table link.  Assembled from the record's artefacts
only; a missing artefact is written as such (never a borrowed number).

    python -m record_v5.report_v5 --config run.json --record REC --population population.v5.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from record_v5 import config as cfgmod
from record_v5.costlog_v5 import CLASSES, CLASS_TITLES

CONFORMANCE = "docs/PROTOCOL-CONFORMANCE.md (rows R14, R16-R19; data/contract/a9-runner/conformance-rows.md)"


def j(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--population", required=True)
    a = ap.parse_args(argv)
    cfg = cfgmod.load(a.config)
    rec = Path(a.record)
    pop = j(Path(a.population)) or {}
    fz, deps, inst = j(rec / "freeze.json") or {}, j(rec / "deps.json") or {}, j(rec / "instance" / "protocol_instance.json") or {}
    prot, ev, exreq = j(rec / "protocol.json"), j(rec / "protocol_events.json") or [], j(rec / "adapter" / "selective" / "exemption_request.json")
    serving, cov, expr = j(rec / "serving_v5.json"), j(rec / "coverage_v5.json"), j(rec / "expressibility_v5.json") or {}
    cost, ver, nonfin = j(rec / "costlog_v5.json") or {}, j(rec / "verify" / "verify_v5.json") or {}, j(rec / "nonfinite_v5.json")
    obl, unsup = j(rec / "obligations" / "obligations.json"), j(rec / "obligations" / "unsupported_selected.json")
    cert = j(rec / "verify" / "certificate_v2.json")
    run_log = (rec / "logs" / "run.log").read_text() if (rec / "logs" / "run.log").exists() else ""
    failed = [m.group(0) for m in re.finditer(r"=== \[[^\]]+\] \S+ rc=([1-9]\d*) \d+s", run_log)]
    refused = next((e for e in ev if str(e.get("event", "")).endswith("_REFUSED")), None)

    J = (prot or {}).get("J") if prot else next((e.get("J") for e in ev if e.get("event") == "r_q"), None)
    F = (prot or {}).get("F") if prot else (exreq or {}).get("F")
    Q = (prot or {}).get("selected_vus") if prot else None
    pexit = j(rec / "protocol_exit.json") or {}
    if refused:
        outcome = f"REFUSED by the verifier at step {2 if refused['event'].startswith('step2') else 4}: {refused.get('problems')}"
    elif pexit and pexit.get("kind") not in (None, "completed", "sampler_verify_problems"):
        outcome = f"NOT VERIFIED -- no verifier decision: {pexit['kind']} ({(pexit.get('reason') or '')[:200]}); costs of stages never reached are UNAVAILABLE"
    elif ver.get("status"):
        outcome = f"{ver['status']} (A3 reverify: {(ver.get('record_v3') or {}).get('status_line') or 'see verify/verify_v5.json'})"
    else:
        outcome = "UNAVAILABLE (verify_v5 not run)"
    if unsup and unsup.get("count"):
        outcome += f"; rejected at step 5: unsupported selected check ({unsup['count']} selected VU(s) of kinds without a checker in retained RUs)"

    tot = (cov or {}).get("totals") or {}
    per_ru = {"unsupported_vus_per_ru": tot.get("unsupported_vus_per_ru"), "vus_per_ru": tot.get("vus_per_ru"), "unsupported_share": tot.get("unsupported_share")}
    exposure = (cov or {}).get("exposure") or {}
    if cov:
        cov = {**cov, "supported_kinds": tot.get("kinds_with_checker"), "unsupported_kinds_n": len(cov.get("unsupported_kinds") or [])}
    doc: dict[str, Any] = {
        "schema": "veritor/a9/report-v5/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": cfg.run_id, "record_label": cfg.record_label, "protocol_exit": pexit or None,
        "source": {"commit": fz.get("commit"), "dirty_patch_sha256": fz.get("dirty_patch_sha256"), "dirty_patch_bytes": fz.get("dirty_patch_bytes"), "deps": deps.get("sources"),
                   "freeze_checks": sorted(p.name for p in rec.glob("freeze-check-*.json"))},
        "identities": {"instance_digest": (prot or {}).get("instance_digest") or ((rec / "instance" / "instance_digest.txt").read_text().strip() if (rec / "instance" / "instance_digest.txt").exists() else None),
                       "circuit": (inst.get("program") or {}), "partition_digest": pop.get("partition_digest"),
                       "population": {"schema": pop.get("schema"), "construction": pop.get("construction"), "content_sha256": pop.get("content_sha256"), "basis": pop.get("basis")},
                       "profile": (inst.get("production") or {}).get("profile"), "vocabulary": (inst.get("production") or {}).get("vocabulary")},
        "workload": cfg.as_json()["workload"], "hardware": (cost or {}).get("hardware") or (serving or {}).get("hardware"),
        "numerics": "BF16 batch-invariant eager vLLM (semantic profile llama-eager-batch-invariant-v3); greedy; prefix caching off (matched C)",
        "law": {"q": str(cfg.q), "s": str(cfg.s), "delta": str(cfg.delta), "s_source": cfg.s_source if hasattr(cfg, "s_source") else "run config"},
        "epoch": cfg.epoch.as_json(), "B_ex": {"value": cfg.epoch.B_ex, "unit": "RU", "label": cfg.epoch.label}, "f_policy": cfg.f_policy, "randomness_mode": cfg.randomness_mode,
        "counts": {"inputs_excluded_from_VUs": (pop.get("basis") or {}).get("excluded"), "gates": pop.get("gates"), "vus_per_ru": pop.get("vu_count"), "rus": pop.get("ru_count"),
                   "requests": cfg.requests, "widths": (pop.get("widths") or "see population kinds"), "J": J, "F": F, "Q": Q,
                   "coverage": {"supported_kinds": (cov or {}).get("supported_kinds"), "unsupported_kinds": (cov or {}).get("unsupported_kinds_n"), "per_ru_unsupported_n": per_ru,
                                "probability of selecting an unsupported check -- completion risk": exposure}, "verifier_outcome": outcome},
        "cost_classes": {c: (cost.get("classes") or {}).get(c) for c in CLASSES} if cost else "UNAVAILABLE (costlog_v5 not run)",
        "protocol_overhead_total_s": (cost or {}).get("protocol_overhead_total_s"), "other_verifier_s": (cost or {}).get("other_verifier_s"),
        "serving_overhead_R19": (serving or {}).get("overhead") or "UNAVAILABLE (serve_v5 not run or --skip-stock)", "serving_agreement": (serving or {}).get("agreement"),
        "equivalence_note": (serving or {}).get("equivalence_note"),
        "certificate": {"built": bool(cert), "path": str(rec / "verify" / "certificate_v2.json") if cert else None, "reason": (ver.get("certificate") or {}).get("reason"),
                        "U_note": "U = W is the expected value for a 4-request epoch at delta = 1e-12 (A7's structural finding)"},
        "nonfinite_W_A56_1": nonfin or "UNAVAILABLE (nonfinite_v5 not run)", "exemption_request": {k: (exreq or {}).get(k) for k in ("F", "budget_exceeded", "B_ex", "f_policy", "per_ru")} if exreq else None,
        "unsupported_selected": unsup, "obligations": {k: (obl or {}).get(k) for k in ("statements", "chunks", "vus", "provable", "unsupported")} if obl else None,
        "failed_or_aborted_phases": failed, "conformance": CONFORMANCE,
    }
    (rec / "report_v5.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")

    labels = ([cfg.record_label] if cfg.record_label else []) + (ver.get("labels") or [])
    L = [f"# {cfg.run_id} -- record_v5 report ({doc['utc']})", "",
         f"**Outcome:** {outcome}", "", *([f"**Labels:** " + " · ".join(dict.fromkeys(labels)), ""] if labels else []),
         f"- Source: `{fz.get('commit')}` dirty patch sha256 `{fz.get('dirty_patch_sha256')}` ({fz.get('dirty_patch_bytes')} B); freeze checks: {', '.join(doc['source']['freeze_checks']) or 'none'}",
         f"- Deps: {json.dumps(deps.get('sources'))}",
         f"- Workload: {cfg.workload} = {cfg.model}, prompt {cfg.prompt_len}, {cfg.max_tokens} slots (STEPS {cfg.max_tokens - 1}), {cfg.requests} requests, greedy, prefix caching off",
         f"- Hardware: {doc['hardware']}; numerics: {doc['numerics']}",
         f"- Instance `{(doc['identities']['instance_digest'] or 'n/a')[:16]}`; partition `{(pop.get('partition_digest') or 'n/a')[:16]}` ({pop.get('construction')}); population content `{(pop.get('content_sha256') or 'n/a')[:16]}`; circuit: {json.dumps(inst.get('program'))[:300] if inst.get('program') else 'n/a'}",
         f"- Law: q = {cfg.q}, s = {cfg.s}, delta = {cfg.delta}; epoch {cfg.epoch.as_json()}; **B_ex = {cfg.epoch.B_ex} RU** ({cfg.epoch.label}); F policy `{cfg.f_policy}`; randomness `{cfg.randomness_mode}`",
         f"- Counts: VUs per RU {pop.get('vu_count')}, RUs {pop.get('ru_count')} x {cfg.requests} requests, inputs excluded {json.dumps((pop.get('basis') or {}).get('excluded'))}; |J| = {len(J) if J is not None else 'n/a'} {J}, |F| = {len(F) if F is not None else 'n/a'} {F}, |Q| = {Q}",
         f"- Coverage: supported kinds {(cov or {}).get('supported_kinds')}, unsupported {(cov or {}).get('unsupported_kinds_n')}; per-RU unsupported n {json.dumps(per_ru)[:400]}",
         f"- Probability of selecting an unsupported check -- completion risk: {json.dumps(exposure)[:600]}",
         "", "## Cost classes", "", "| class | seconds | label | note |", "|---|---:|---|---|"]
    for c in CLASSES:
        row = (cost.get("classes") or {}).get(c) if cost else None
        if row is None:
            L.append(f"| {CLASS_TITLES[c]} | UNAVAILABLE | - | costlog_v5 not run |")
        else:
            note = row.get("reason") or f"{row.get('rows')} rows"
            L.append(f"| {CLASS_TITLES[c]} | {row['seconds'] if row['seconds'] is not None else 'UNAVAILABLE'} | {row['label']} | {note} |")
    L += [f"| other verifier work (sampling, step-4 checks; not one of the eight) | {(cost or {}).get('other_verifier_s')} | MEASURED | |",
          f"| **protocol overhead total (2-8, once)** | {(cost or {}).get('protocol_overhead_total_s')} | | |", "",
          "## Serving overhead (R19)", "", f"{json.dumps(doc['serving_overhead_R19'], indent=1)[:1500]}", "", f"Agreement: {json.dumps((serving or {}).get('agreement', {}) and {k: v for k, v in serving['agreement'].items() if k != 'per_request'})}",
          f"{doc['equivalence_note'] or ''}", "",
          "## Certificate", "", f"{'built: ' + doc['certificate']['path'] if cert else 'not built: ' + str(doc['certificate']['reason'])}. {doc['certificate']['U_note']}", "",
          "## W-A56-1 non-finite committed words", "", f"{json.dumps({k: v for k, v in (nonfin or {}).items() if k != 'per_ru'}) if nonfin else 'UNAVAILABLE (nonfinite_v5 not run)'}", "",
          "## Failed / aborted phases", "", *(["- " + x for x in failed] or ["- none recorded in logs/run.log"]), "",
          f"Conformance: {CONFORMANCE}", ""]
    (rec / "report_v5.md").write_text("\n".join(L))
    print(f"[report_v5] {outcome} -> {rec / 'report_v5.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
