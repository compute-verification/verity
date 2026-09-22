"""[E8 SCALE 2026-09-12] RAW per-stage cost numbers of one V5 run -> out/scale/<agent>/costlog.<p>x<d>.json (until E3's costlog schema v1 lands).

Reads what the record already carries (costlog_v5.json class seconds + run_v5 phase walls, serving_v5.json, protocol.json, report_v5.json law,
logs/prover.time, verify/reverify_v3/coverage.json, the pod's reverify <tag>.time) plus three pod-side captures that the pipeline does not write:
the rows producer's `/usr/bin/time -v` file, a `sizes.json` (du -sb of rows / trees / record / archive / slim, bytes) and the RSS sampler summary
(per-process RSS sampled every 2 s for the stages the pipeline does not wrap: prove = SP1 host, verify_live = verify_v5).  Every number is labelled
MEASURED (with its source) or ABSENT; nothing is modeled here.

  python -m e2e.demo_core.night.costlog_raw --record <dir> --tag sc1 --rows-time producer.time --sizes sizes.json --rss rss-sample.summary.json \
      --reverify-time sc1.time --out out/scale/e8/costlog.32x16.json --prompt 32 --decode 16 [--note ...]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


def _j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _time_v(p: Path | None) -> dict | None:
    """GNU time -v -> {wall_s, user_s, sys_s, peak_rss_kb, fs_inputs, fs_outputs, exit_status}."""
    if not p or not p.exists():
        return None
    t = p.read_text()
    out: dict = {"source": str(p), "label": "MEASURED (/usr/bin/time -v)"}
    m = re.search(r"Elapsed \(wall clock\) time.*?:\s*([\d:.]+)", t)
    if m:
        parts = [float(x) for x in m.group(1).split(":")]
        out["wall_s"] = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    for key, rx in (("user_s", r"User time \(seconds\):\s*([\d.]+)"), ("sys_s", r"System time \(seconds\):\s*([\d.]+)"),
                    ("peak_rss_kb", r"Maximum resident set size \(kbytes\):\s*(\d+)"), ("fs_inputs_blocks", r"File system inputs:\s*(\d+)"),
                    ("fs_outputs_blocks", r"File system outputs:\s*(\d+)"), ("exit_status", r"Exit status:\s*(\d+)")):
        m = re.search(rx, t)
        if m:
            out[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    if "peak_rss_kb" in out:
        out["peak_rss_gb"] = round(out["peak_rss_kb"] / 1e6, 3)
    m = re.search(r'Command being timed: "(.*?)"', t)
    if m:
        out["command"] = m.group(1)[:300]
    return out


def _phase_walls(cost: dict | None) -> dict:
    w = {}
    for r in (cost or {}).get("rows") or []:
        st = str(r.get("stage", ""))
        if st.startswith("run_v5 phase wall: "):
            w[st.split(": ", 1)[1]] = r.get("seconds")
        elif st.startswith("serve wall"):
            w["serve_matched_production"] = r.get("seconds")
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--decode", type=int, required=True)
    ap.add_argument("--rows-time")
    ap.add_argument("--rows-progress", help="the producer's progress.jsonl (rate, gates)")
    ap.add_argument("--sizes", help="pod-side sizes.json: {name: bytes}")
    ap.add_argument("--rss", help="rss_sampler summary json")
    ap.add_argument("--reverify-time")
    ap.add_argument("--reverify-json", help="reverify_clean.json (status, seconds)")
    ap.add_argument("--note", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = Path(a.record)
    cost = _j(rec / "costlog_v5.json")
    serving = _j(rec / "serving_v5.json") or {}
    proto = _j(rec / "protocol.json") or {}
    rep = _j(rec / "report_v5.json") or {}
    inst = _j(rec / "protocol_instance.json") or {}
    pop = _j(rec / "population.v5.json") or {}
    verify_v5 = _j(rec / "verify" / "verify_v5.json") or {}
    cov = _j(rec / "verify" / "reverify_v3" / "coverage.json") or {}
    sizes = _j(Path(a.sizes)) if a.sizes else {}
    rss = _j(Path(a.rss)) if a.rss else {}
    rows_time = _time_v(Path(a.rows_time)) if a.rows_time else None
    prover_time = _time_v(rec / "logs" / "prover.time")
    rev_time = _time_v(Path(a.reverify_time)) if a.reverify_time else None
    rev_json = _j(Path(a.reverify_json)) if a.reverify_json else None
    walls = _phase_walls(cost)
    classes = {k: v.get("seconds") for k, v in ((cost or {}).get("classes") or {}).items() if isinstance(v, dict)}
    per_pat = (rss or {}).get("per_pattern") or {}

    def rss_peak(*pats):
        best = None
        for p in pats:
            d = per_pat.get(p)
            if d and d.get("peak_rss_kb") and (best is None or d["peak_rss_kb"] > best["peak_rss_kb"]):
                best = {"peak_rss_kb": d["peak_rss_kb"], "peak_rss_gb": round(d["peak_rss_kb"] / 1e6, 3), "pattern": p, "at": d.get("peak_ts"),
                        "label": (rss or {}).get("label", "MEASURED (sampled)")}
        return best or {"label": "ABSENT (no sample matched)"}

    # rows: rate + gates from progress.jsonl `done` line if given
    rows_rate = None
    if a.rows_progress and Path(a.rows_progress).exists():
        for line in Path(a.rows_progress).read_text().splitlines()[::-1]:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("event") == "done" or d.get("done") or "gates_total" in d or "gates" in d:
                rows_rate = d
                break
    matched = serving.get("matched") or {}
    n_proofs = len(list((rec / "proofs").glob("**/chunk*.json"))) if (rec / "proofs").exists() else None
    law = rep.get("law") or {}
    security = inst.get("security") or {}
    n_v = pop.get("vu_count") or pop.get("N_V") or (pop.get("totals") or {}).get("vus") or security.get("N_V")
    counts = (cov.get("counts") or {})
    q_size = proto.get("selected_vus") if isinstance(proto.get("selected_vus"), int) else (len(proto["selected_vus"]) if isinstance(proto.get("selected_vus"), list) else None)
    out = {
        "schema": "veritor/e8/costlog-raw/v1 (RAW per-stage numbers; E3's costlog schema v1 supersedes when it lands)",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "record": rec.name, "tag": a.tag, "record_path_pod": (cost or {}).get("prover_time", {}).get("source", "").rsplit("/logs/", 1)[0] or None,
        "model": (rep.get("workload") or {}).get("model") if isinstance(rep.get("workload"), dict) else serving.get("workload"),
        "workload": serving.get("workload"), "prompt_tokens": a.prompt, "decode_tokens": a.decode, "requests": serving.get("requests"),
        "hardware": (cost or {}).get("hardware") or serving.get("hardware"),
        "engine_versions": serving.get("engine_versions"),
        "sampling": {"N_V": n_v, "s": law.get("s"), "q": law.get("q"), "delta": law.get("delta"), "B_ex": (proto.get("step4") or {}).get("B_ex", (proto.get("epoch") or {}).get("B_ex")),
                     "Q_selected_vus": q_size, "s_source": law.get("s_source"), "instance_id": inst.get("instance_id") or (proto.get("epoch") or {}).get("epoch_id"),
                     "coverage_line": counts.get("coverage_line"), "completion_claim": counts.get("completion_claim"),
                     "proof_checked_units": counts.get("proof_checked_vus"), "proof_objects": counts.get("proof_objects"),
                     "directly_checked_units": (sum(v for v in counts["directly_checked_vus"].values() if isinstance(v, int)) if isinstance(counts.get("directly_checked_vus"), dict) else counts.get("directly_checked_vus")),
                     "directly_checked_by_rule": counts.get("directly_checked_vus") if isinstance(counts.get("directly_checked_vus"), dict) else None,
                     "input_units_V6": counts.get("input_vus_V6"), "uncovered_units": counts.get("uncovered_vus"),
                     "coverage_counts_raw": counts, "label": "MEASURED (record: report_v5 law, protocol.json, coverage.json)"},
        "stages": {
            "serve": {"wall_s_matched_production_total": matched.get("wall_s_total"), "engine_build_s": matched.get("engine_build_s"),
                      "recorded_pass_wall_s": matched.get("recorded_pass_wall_s"), "run_v5_phase_wall_s": walls.get("serve"),
                      "gpu": serving.get("hardware"), "gpu_mem_peak_mib_sampled": (rss or {}).get("gpu_mem_peak_mib"), "label": "MEASURED (serving_v5.json; GPU mem sampled)"},
            "rows": {"time_v": rows_time, "progress_done": rows_rate, "bytes": (sizes or {}).get("rows_bytes"),
                     "label": "MEASURED (E1 producer's /usr/bin/time -v + progress.jsonl; reused rows, produced once)"},
            "commit": {"run_v5_phase_wall_s_protocol": walls.get("protocol"), "selective_prover_time_v": prover_time,
                       "class_seconds": {k: classes.get(k) for k in ("native", "execution", "commit_io", "replay", "interior_commit")},
                       "phase2_timing": (cost or {}).get("phase2_timing"), "protocol_timing": proto.get("timing"),
                       "bytes": {k: (sizes or {}).get(k) for k in ("rows_bytes", "trees_bytes", "record_bytes", "record_proofs_bytes", "record_adapter_bytes")},
                       "label": "MEASURED (costlog_v5 classes; prover.time; du -sb on the pod)"},
            "obligations": {"run_v5_phase_wall_s": walls.get("obligations"), "peak_rss_sampled": rss_peak("obligations_v5"), "label": "MEASURED"},
            "prove": {"run_v5_phase_wall_s": walls.get("prove"), "class_seconds_prove": classes.get("prove"), "n_proof_objects": n_proofs,
                      "peak_rss_sampled": rss_peak("veritor-zk-host", "rec_p1_prove"), "gpu_mem_peak_mib_sampled": (rss or {}).get("gpu_mem_peak_mib"), "label": "MEASURED"},
            "verify_live": {"run_v5_phase_wall_s": walls.get("verify"), "verify_v5_seconds": verify_v5.get("seconds"), "status": verify_v5.get("status"),
                            "class_seconds": {k: classes.get(k) for k in ("verifier_expected", "verify_proofs")}, "peak_rss_sampled": rss_peak("verify_v5", "verifier_v3.reverify"), "label": "MEASURED"},
            "reverify": {"time_v": rev_time, "status": (rev_json or {}).get("status"), "checks_failed": (rev_json or {}).get("checks_failed"),
                         "bytes_read_note": "GNU time 'File system inputs' counts block reads that hit the device (page-cache hits are not counted); E9 owns the per-check byte accounting",
                         "label": "MEASURED (/usr/bin/time -v around reverify_clean.sh)"},
            "archive": {"archive_bytes": (sizes or {}).get("archive_bytes"), "slim_bytes": (sizes or {}).get("slim_bytes"), "label": "MEASURED (stat on the pod)"},
        },
        "phase_walls_run_v5_s": walls, "cost_classes_s": classes,
        "protocol_overhead_total_s": (cost or {}).get("protocol_overhead_total_s"),
        "verdicts": {"live_driver_verify_v5": verify_v5.get("status"), "live_certificate": (verify_v5.get("certificate") or {}).get("status"),
                     "reverify_clean": (rev_json or {}).get("status")},
        "notes": a.note,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"costlog raw -> {a.out}: walls {walls}; proofs {n_proofs}; Q {q_size}; N_V {n_v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
