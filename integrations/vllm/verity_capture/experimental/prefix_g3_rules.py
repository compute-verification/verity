"""[r8 lane E-cleanup 2026-09-18 — CANDIDATE rule, not a gate] prefix M1 as a G3 strengthening (WINDOW-20260917 post-window item 5;
lead.md 22:58Z: "M1 is re-scoped from mechanism to a G3 strengthening on declared leaves — alias root by storage identity + token
obligation; explicit `cached_kv` Input when the writer is outside the log").

G3 (root closure) today checks unknown roots, opaque / prescribed fallbacks and `root_policy.json`.  With prefix caching a warm
request's first `num_computed` prompt positions never receive a `Root("prompt_token", …)` in the log: the declared leaves at those
positions have NO gate of their own — the interface is over-declared unless the Program states, per reused KV slot, what stands in
for the missing root.  `verity_capture.patterns_prefix.obligations` (prefix lane, `2bc85c7ea`) derives exactly that from the
linkage record `verity_capture.prefix_cache.extract_linkage` builds out of the capture log:

    StateEdge        the reused slot's writer is a same-run producer row (storage identity: `reshape_and_cache_flash` last writer)
    AliasObligation  the consumer's declared prompt token at the position == the producer row's token
    ExternalState    the writer is outside the log (warm server / previous run): a committed `cached_kv` Input is required

Rule (candidate G3 += prefix closure):  a run whose linkage is `cold` (no reuse) is unchanged; a warm run must be `closed` — every
reused slot has a same-run StateEdge AND every AliasObligation holds (decided AND true).  `external-state`, `alias-violated` and
undecided alias obligations are G3 problems (an undecided obligation is an unproven alias claim, the same class G3 already rejects
through `root_policy.alias_claims`).  Nothing is relaxed: the rule only adds problems.

The linkage is RE-DERIVED here from `capture/log.jsonl.gz` (the committed `capture/prefix_linkage.json`, when present, is compared
with it and the agreement is recorded).  Output: `<out>/linkage.candidate-prefix.json`, `<out>/gates.candidate-prefix.json`
(frozen `gates.evaluate_gates` results, G3 replaced by the candidate), `<out>/CANDIDATE.md`.  Verdicts are `candidate_verdict`;
nothing is written to the run dir / gates.json / cards.

    python -m verity_capture.experimental.prefix_g3_rules --run-dir RUN_DIR --out OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verity_capture import gates as G
from verity_capture import patterns_prefix as P
from verity_capture import prefix_cache as PC

RULE_G3 = ("prefix-closure rule: a run with prefix reuse must be `closed` (every reused KV slot has a same-run StateEdge by storage "
           "identity and every AliasObligation on the declared leaves holds); external-state / alias-violated / undecided -> G3 problems; "
           "a cold run is unchanged")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(p: Path) -> Any:
    return json.loads(p.read_text())


def prefix_closure_problems(obligations: dict) -> list[str]:
    """The G3 problems the prefix-closure rule adds for one obligations document (`patterns_prefix.obligations`)."""
    run = obligations.get("run") or {}
    if run.get("status") in (None, "cold"):
        return []
    problems: list[str] = []
    if run.get("external_state"):
        problems.append(f"prefix: {run['external_state']} reused KV slot(s) whose writer is outside the log (no same-run state edge; a committed `cached_kv` Input is required)")
    if run.get("alias_failed"):
        problems.append(f"prefix: {run['alias_failed']} alias obligation(s) failed (declared prompt token != producer row's token)")
    if run.get("alias_undecided"):
        problems.append(f"prefix: {run['alias_undecided']} alias obligation(s) undecidable (declaration unavailable) = unproven alias claims")
    if run.get("status") not in ("closed", "external-state", "alias-violated"):
        problems.append(f"prefix: closure status {run.get('status')!r}")
    return problems


def candidate_G3(recorded_g3: dict, obligations: dict) -> dict:
    """The recorded G3 result plus the prefix-closure problems (never fewer problems than recorded)."""
    problems = [p.strip() for p in str(recorded_g3.get("reason") or "").split(";") if p.strip()]
    added = prefix_closure_problems(obligations)
    problems += added
    status = recorded_g3.get("status")
    if added and status in ("pass", None):
        status = "fail"
    run = obligations.get("run") or {}
    return {"id": "G3", "name": recorded_g3.get("name", "root closure"), "status": status, "reason": "; ".join(problems) if problems else None,
            "candidate": {"rule": RULE_G3, "applied": run.get("status") not in (None, "cold"), "closure": run,
                          "recorded_status": recorded_g3.get("status"), "recorded_reason": recorded_g3.get("reason")}}


def derive_linkage(run: Path) -> tuple[dict, dict]:
    """Re-derive the linkage record from the capture log; return (linkage, agreement with the committed prefix_linkage.json)."""
    from verity_capture.log import read_log
    log = run / "capture" / "log.jsonl.gz"
    if not log.exists():
        raise FileNotFoundError(log)
    wl = None
    for cand in (run / "workload.json", run / "capture" / "workload.json"):
        if cand.exists():
            wl = _read(cand)
            break
    inst = next((c for c in (run / "instances.jsonl", run / "instances.jsonl.gz") if c.exists()), None)
    linkage = PC.extract_linkage(read_log(log), workload=wl, instances=inst)
    linkage["source"] = {"log": str(log), "instances": str(inst) if inst else None, "workload": "run/workload.json" if wl else None,
                         "derived_by": "verity_capture.experimental.prefix_g3_rules (r8 lane E-cleanup)"}
    agreement: dict[str, Any] = {"committed": None}
    committed = run / "capture" / "prefix_linkage.json"
    if committed.exists():
        c = _read(committed)
        ob_c, ob_d = P.obligations(c), P.obligations(linkage)

        def strip(ob: dict) -> dict:                  # producer_instances come from the fold's instances.jsonl, which a compact record lacks
            return {rid: {**r, "state_edges": [{k: v for k, v in e.items() if k != "producer_instances"} for e in r.get("state_edges") or []]}
                    for rid, r in ob["requests"].items()}
        sc, sd = strip(ob_c), strip(ob_d)
        agreement = {"committed": str(committed.relative_to(run)), "closure_equal": c.get("closure") == linkage.get("closure"),
                     "obligations_run_equal": ob_c["run"] == ob_d["run"],
                     "requests_equal_modulo_producer_instances": {rid: sc.get(rid) == sd.get(rid) for rid in sorted(set(sc) | set(sd))},
                     "instances_available_here": inst is not None,
                     "committed_instances": ((c.get("source") or {}).get("instances"))}
    return linkage, agreement


def candidate_gates(run_dir: str | Path, out_dir: str | Path) -> dict:
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    linkage, agreement = derive_linkage(run)
    ob = P.obligations(linkage)
    (out / "linkage.candidate-prefix.json").write_text(json.dumps({"linkage": linkage, "obligations": ob, "agreement_with_committed": agreement,
                                                                    "utc": _utc()}, indent=1) + "\n")
    recorded = _read(run / "gates.json")
    results = G.evaluate_gates(run)                       # the frozen gates, untouched, over the record as committed
    g3 = candidate_G3(recorded["gates"]["G3"], ob)
    rows = []
    for r in results:
        rec = recorded["gates"].get(r.id) or {}
        if r.id == "G3":
            rows.append({"id": "G3", "name": r.name, "recorded": rec.get("status"), "candidate": g3["status"], "reason": g3["reason"],
                         "rule": RULE_G3 if g3["candidate"]["applied"] else None})
        else:
            rows.append({"id": r.id, "name": r.name, "recorded": rec.get("status"), "candidate": r.status, "reason": r.reason, "rule": None})
    core = {row["id"]: row["candidate"] for row in rows if row["id"] in G.GATES}
    cand_verdict = "REJECTED" if "fail" in core.values() else ("ACCEPTED" if all(v == "pass" for v in core.values()) else "INCOMPLETE")
    doc = {"schema": "verity-capture/gates/candidate-prefix-g3/v1", "utc": _utc(), "run": run.name, "recorded_verdict": recorded.get("verdict"),
           "candidate_verdict": cand_verdict, "gates": rows, "rules": {"G3": RULE_G3}, "closure": ob["run"], "agreement_with_committed": agreement,
           "method": "frozen gates.evaluate_gates over the record; G3 := recorded G3 problems + prefix-closure problems from the re-derived linkage",
           "note": "CANDIDATE — not a verdict.  Nothing written to gates.json / card.json / the cards dir."}
    (out / "gates.candidate-prefix.json").write_text(json.dumps(doc, indent=1) + "\n")
    md = [f"# CANDIDATE gate re-evaluation (prefix M1 as G3 strengthening) — `{run.name}`", "",
          f"recorded verdict **{recorded.get('verdict')}** → candidate **{cand_verdict}** (not a verdict; see `gates.candidate-prefix.json`)", "",
          f"linkage (re-derived from the capture log): closure **{ob['run']['status']}** — {ob['run']}", f"agreement with committed `prefix_linkage.json`: {agreement}", "",
          "| gate | recorded | candidate | rule applied | reason (candidate) |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        md.append(f"| {row['id']} | {row['recorded']} | **{row['candidate']}** | {'yes' if row['rule'] else '—'} | {(row['reason'] or '').replace('|', '/')[:160]} |")
    md += ["", f"- G3 rule: {RULE_G3}", "", P.blocker_text(linkage), ""]
    (out / "CANDIDATE.md").write_text("\n".join(md))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True, help="lane artefact dir (never the run dir, never the cards dir)")
    a = ap.parse_args(argv)
    doc = candidate_gates(a.run_dir, a.out)
    print(json.dumps({"run": doc["run"], "recorded_verdict": doc["recorded_verdict"], "candidate_verdict": doc["candidate_verdict"],
                      "closure": doc["closure"]["status"], "G3": [r for r in doc["gates"] if r["id"] == "G3"][0]["candidate"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
