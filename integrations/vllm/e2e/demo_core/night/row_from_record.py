"""[E8 NIGHT 2026-09-12] The MEASURED fields of a finished (or running) record_v5 run into the night table row
(out/night/table/<model>__<p>x<d>.json): serve, commit, sample, statements / proof objects, proving, live verification, clean re-verification,
verdicts, pod cost, record path.  Reads only what the run wrote (report_v5.json, costlog_v5.json, verify/verify_v5.json, protocol_v5.json,
obligations/, proofs/, the reverify_clean.json of record_v5/reverify_clean.sh) and labels every field MEASURED / NOT RUN by what exists.
Field-wise merge into the shared row (the integrator's night_table.py unions rows field-wise, latest utc wins): a run's MEASURED value
replaces a MODELED one; nothing is invented for a stage that has not happened (``NOT RUN (running since <utc>)`` per the brief).

    python -m e2e.demo_core.night.row_from_record --record <mirror of the record dir> [--reverify <reverify_clean.json>] \
        --prompt-len 32 --max-tokens 16 [--pod-usd-per-h 0.74 --pod-since 2026-09-12T05:28:32Z] [--table out/night/table] [--running-since <utc>]
"""

from __future__ import annotations

import argparse
import calendar
import json
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]


def j(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def cls_seconds(cost: dict | None, name: str) -> float | None:
    c = ((cost or {}).get("classes") or {}).get(name) or {}
    return c.get("seconds") if c.get("label", "UNAVAILABLE") != "UNAVAILABLE" else None


def measured(value: Any, where: str, note: str = "") -> dict[str, Any]:
    return {"value": value, "label": "MEASURED" + (f": {note}" if note else ""), "where": where}


def not_run(since: str | None, what: str) -> dict[str, Any]:
    return {"value": None, "label": f"NOT RUN (running since {since})" if since else f"NOT RUN ({what})"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--reverify", default=None, help="reverify_clean.json (record_v5/reverify_clean.sh on the pod)")
    ap.add_argument("--prompt-len", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--table", default=str(REPO / "out/night/table"))
    ap.add_argument("--pod-usd-per-h", type=float, default=0.74)
    ap.add_argument("--pod-since", default=None, help="pod start utc (cost = hours so far x rate; the whole pod, shared with the per-kind bench)")
    ap.add_argument("--running-since", default=None, help="utc the proofs started, when they are still running")
    ap.add_argument("--record-path", default=None, help="the record's durable path (pod) for the row")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--pod-name", default="night-e8-prove")
    ap.add_argument("--note", action="append", default=[], help="[D40] MEASURED notes appended to the row's `notes` list (history: n2 refusal, offline row-set test)")
    ap.add_argument("--extra-cost-usd", type=float, default=0.0, help="[D40] cost of other pods of this run (ct-e8-rows) added to pod_cost_usd")
    a = ap.parse_args()
    rec = Path(a.record)
    cost, rep, ver = j(rec / "costlog_v5.json"), j(rec / "report_v5.json"), j(rec / "verify" / "verify_v5.json")
    prot, serving = j(rec / "protocol_v5.json") or j(rec / "protocol.json"), j(rec / "serving_v5.json")
    obl = j(rec / "obligations" / "obligations.json")
    rv = j(Path(a.reverify)) if a.reverify else None
    hw = (cost or {}).get("hardware") or (serving or {}).get("hardware") or "night-e8-prove (RTX 4090)"
    where = f"{hw}; record {a.record_path or rec}"
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row: dict[str, Any] = {"model": a.model, "architecture": "LlamaForCausalLM", "prompt_len": a.prompt_len, "max_tokens": a.max_tokens, "utc": utc}
    fz = j(rec / "freeze.json") or {}
    row["revision"] = {"value": f"{(fz.get('commit') or '')[:12]} dirty {(fz.get('dirty_patch_sha256') or '')[:12]} ({fz.get('dirty_patch_bytes')} B)", "label": "MEASURED (freeze.json)"}
    row["record_path"] = {"value": a.record_path or str(rec), "label": "MEASURED"}
    # serve
    m = (serving or {}).get("matched") or {}
    if m.get("recorded_pass_wall_s") is not None:
        row["serve_s"] = measured(m["recorded_pass_wall_s"], where, f"serve_v5 matched recorded pass, one request (recording on/off {m.get('recording_on_wall_s')}/{m.get('recording_off_wall_s')}; engine build {m.get('engine_build_s')} s excluded)")
        row["serve_gpu"] = measured(hw, where)
    # commitments (class 3 + 5), interior commit RSS from the prover's /usr/bin/time when present
    cio, ic = cls_seconds(cost, "commit_io"), cls_seconds(cost, "interior_commit")
    if ic is not None or cio is not None:
        row["commit_s"] = measured({"commit_io_s": cio, "interior_commit_s": ic}, where, "costlog_v5 classes 3 (boundary commitments + I/O) and 5 (interior commitments) of THIS run")
    pt = (cost or {}).get("prover_time") or {}
    if pt.get("peak_rss_gb") is not None:
        row["commit_peak_rss_gb"] = measured(pt["peak_rss_gb"], where, "prover process peak RSS (/usr/bin/time -v; commit + prove driver)")
    # sample
    counts = (rep or {}).get("counts") or {}
    Q = counts.get("Q") if counts.get("Q") is not None else (prot or {}).get("selected_vus")
    if Q is not None:
        row["Q"] = measured(Q if isinstance(Q, int) else (len(Q) if isinstance(Q, list) else Q), where, "|Q| drawn (q = 1, s = 2^-14, fresh verifier randomness, ONE sample)")
    law = (rep or {}).get("law") or {}
    if law.get("s"):
        row["s"] = {"value": law["s"], "label": "MEASURED (run config / instance)"}
    # statements / proof objects
    if obl:
        st = obl.get("statements") or obl.get("chunks")
        row["statements"] = measured(st, where, "obligations_v5: statements packaged from the selected VUs (shipped batching)")
        proofs = sorted((rec / "proofs").glob("*.json")) if (rec / "proofs").exists() else []
        # [n3] chunk*.json only: proofs/ also carries the prover summary / preflight json (n3 counted 30 for 28 proofs)
        n_proofs = obl.get("proofs") or (len([p for p in proofs if p.name.startswith("chunk")]) or len([p for p in proofs if p.name != "index.json"]) or None)
        if n_proofs is not None:
            row["proof_objects"] = measured(n_proofs, where, "proof objects written by the prove phase")
    # proving
    pv = cls_seconds(cost, "prove")
    if pv is not None:
        row["prove_s"] = measured(pv, where, "costlog_v5 class 6 (SP1 GPU proving of every statement, one process per proof)")
        row["prove_gpu"] = measured(hw.split(" (")[0], where)
    else:
        row["prove_s"] = not_run(a.running_since, "prove phase not reached")
    gp = j(rec / "logs" / "prove_gpu_peak.json")
    if gp and gp.get("peak_mem_gb") is not None:
        row["prove_peak_mem_gb"] = measured(gp["peak_mem_gb"], where, "nvidia-smi sampled during the prove phase")
    # live verification (the shipped driver's step 5 through verify_v5) and the clean re-verification
    vp, ve = cls_seconds(cost, "verify_proofs"), cls_seconds(cost, "verifier_expected")
    if ver:
        row["verify_live_s"] = measured({"verify_proofs_s": vp, "verifier_expected_s": ve, "verify_v5_total_s": ver.get("seconds")}, where, "classes 8 / 7 of costlog_v5 + verify_v5 wall")
        r3 = ver.get("record_v3") or {}
        row["verdict_live"] = measured({"status": ver.get("status"), "status_line": r3.get("status_line"), "checks_failed": r3.get("checks_failed"),
                                        "verifier_outcome": counts.get("verifier_outcome") if counts else None}, where, "verify_v5 (live driver + reverify by import) of THIS run")
    else:
        row["verdict_live"] = not_run(a.running_since, "verify phase not reached")
    if rv:
        row["reverify_s"] = measured((rv.get("seconds") or {}).get("reverify"), rv.get("environment", {}).get("hostname", "pod"),
                                     "record_v5/reverify_clean.sh: fresh venv, shipped e2e.verifier_v3.reverify --instance-format record_v5")
        row["verdict_reverify"] = measured({"status": rv.get("status"), "rc": rv.get("rc"), "checks_failed": rv.get("checks_failed"), "host_sha256": (rv.get("host") or {}).get("sha256")},
                                           rv.get("environment", {}).get("hostname", "pod"), "reverify_clean.json")
    else:
        row["verdict_reverify"] = not_run(a.running_since, "reverify_clean not run yet")
    # [D44, user's rule (2)] coverage by VU identity, the three numbers separately (proof-checked / directly checked / input), from the
    # verifier's coverage.json of THIS run (live step 5 = verify/reverify_v3; the clean re-verification's copy when only that exists)
    cov = None
    for cand in (rec / "verify" / "reverify_v3" / "coverage.json", rec / "verify" / "coverage.json"):
        cov = j(cand)
        if cov:
            break
    if cov and cov.get("counts"):
        c = cov["counts"]
        row["coverage"] = measured({"coverage_line": c.get("coverage_line"), "sampled_vus_Q": c.get("sampled_vus_Q"), "retained": c.get("sampled_vus_retained"),
                                    "proof_checked_vus": c.get("proof_checked_vus"), "directly_checked_vus": c.get("directly_checked_vus"), "input_vus_V6": c.get("input_vus_V6"),
                                    "uncovered_vus": c.get("uncovered_vus"), "duplicate_vus": c.get("duplicate_vus"), "unlocated_vus": c.get("unlocated_vus"),
                                    "unaccounted": (c.get("sampled_vus_retained") or 0) - sum(int(c.get(k) or 0) for k in ("proof_checked_vus", "input_vus_V6", "uncovered_vus", "duplicate_vus", "unlocated_vus")) - sum((c.get("directly_checked_vus") or {}).values()),
                                    "proof_objects": c.get("proof_objects"), "proof_objects_verified": c.get("proof_objects_verified"), "completion_claim": c.get("completion_claim"),
                                    "refusal": (cov.get("refusal") if isinstance(cov.get("refusal"), str) else json.dumps(cov.get("refusal"))[:400]) if cov.get("refusal") else None},
                                   where, "verifier coverage.json#counts of THIS run (rule (2): proof-checked, directly checked and input VUs are separate numbers; proof objects are not VUs)")
    elif ver:
        row["coverage"] = not_run(a.running_since, "no coverage.json written by the verifier")
    # pod cost: this pod (shared with the per-kind proving bench and the rows)
    if a.pod_since:
        t0 = calendar.timegm(time.strptime(a.pod_since, "%Y-%m-%dT%H:%M:%SZ"))
        hours = (time.time() - t0) / 3600
        row["pod_cost_usd"] = measured(round(hours * a.pod_usd_per_h + a.extra_cost_usd, 2), a.pod_name, f"{a.pod_usd_per_h} $/h x {hours:.2f} h since {a.pod_since}" + (f" + {a.extra_cost_usd} $ other pods of this run" if a.extra_cost_usd else " (the whole pod: per-kind bench + rows + this run)"))
    if a.note:
        row["notes"] = [{"value": n, "label": "MEASURED"} for n in a.note]
    row["written_by"] = ["E8"]
    if a.dry:
        print(json.dumps(row, indent=1))
        return 0
    target = Path(a.table) / f"{a.model.split('/')[-1]}__{a.prompt_len}x{a.max_tokens}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    cur = j(target) or {}
    for k in ("model", "architecture", "prompt_len", "max_tokens"):
        cur.setdefault(k, row[k])
    for k, v in row.items():
        if k in ("model", "architecture", "prompt_len", "max_tokens", "written_by"):
            continue
        cur[k] = v
    wb = cur.get("written_by")
    cur["written_by"] = sorted(set((wb if isinstance(wb, list) else ([wb] if wb else [])) + ["E8"]))
    target.write_text(json.dumps(cur, indent=1) + "\n")
    e8dir = REPO / "out/night/e8/table"
    e8dir.mkdir(parents=True, exist_ok=True)
    (e8dir / f"{target.stem}.run.json").write_text(json.dumps(row, indent=1) + "\n")
    print(f"row -> {target} ({len(row)} fields); copy {e8dir / (target.stem + '.run.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
