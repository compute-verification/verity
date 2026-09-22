#!/usr/bin/env python
"""[A9 contract 2026-09-10] ``costlog_v5.json`` -- the contract's EIGHT cost classes (docs/PROTOCOL.md, Reporting requirements), measured
separately, totals without double counting (adapted from D4's ``record_6k/costlog_6k.py``).

    1 native            native / trusted-replay baseline: stock vLLM serving of the same requests (serve_v5); the replay's bare forward
                        (``forward_s`` inside the prover's replay run) is shown as the replay-side native compute (info row, not summed)
    2 execution         original execution + collection: the matched-C recorded pass (inputs/outputs/seed captured)
    3 commit_io         initial commitments (request boundaries incl. v_out(R) of every RU, weights root) + the verifier's step-2 I/O checks
    4 replay            replay of J incl. attempted work for later-exempted RUs: replay_run_s + boundary re-hash (engine build excluded)
    5 interior_commit   interior commitments: build_commitments over the RU trees of J (phase2_commit_s)
    6 prove             proof generation: openings + witness + statement bytes (obligations_v5) + the SP1 prove wall per chunk
    7 verifier_expected verifier derivation of expected statements: pinned program + locator + expected positions (obligations_v5 class-7 row)
    8 verify_proofs     proof verification: host verify of every proof (+ A3 reverify's record checks, which it times separately)
    -- other_verifier   NOT one of the eight, reported beside them: sampling (r_q, r_s, chronology) + step-4 checks + output-copy agreement
    -- wall_only        whole-process walls kept for scheduling, never summed

Every timed function was inspected before its class was named (design.md section 7); the mapping is by the row's own ``class`` when it is
one of the names above, else by the stage name of the A3 / L1 rows (``RULES``).  The A3 driver's phase-2 wall row is REPLACED by its
decomposition from ``adapter/selective/selective_prover.json`` (classes 4 + 5; the residual scheduler/driver time is shown, not hidden).

    python -m record_v5.costlog_v5 --record REC [--hardware "RTX 4090 (pod)"]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

CLASSES = ["native", "execution", "commit_io", "replay", "interior_commit", "prove", "verifier_expected", "verify_proofs", "value_production"]
CLASS_TITLES = {"native": "1 native / trusted-replay baseline", "execution": "2 original execution + collection", "commit_io": "3 initial commitments + I/O checks",
                "replay": "4 replay incl. attempted work for later-exempted RUs", "interior_commit": "5 interior commitments", "prove": "6 proof generation",
                "verifier_expected": "7 verifier derivation of expected statements", "verify_proofs": "8 proof verification",
                # [E10 EOD 2026-09-11] the EOD brief's fifth stage: the interpreter producing the interior values (E1), measured on its own pod
                "value_production": "V value production (reference interpreter, E1; other hardware)"}
EXTRA = ["other_verifier", "wall_only"]
RULES: list[tuple[str, str]] = [
    (r"^commit_inputs_outputs_seed_weights \(phase 0\)", "commit_io"),
    (r"^replay_and_commit_interiors \(phase 2\)", "replay+interior_commit"),
    (r"^ru_challenge \+ vu_challenge", "other_verifier"),
    (r"^prove chunk", "prove"),
    (r"^prove", "prove"),
    (r"^verify chunk", "verify_proofs"),
    (r"^verifier checks", "verify_proofs"),
    (r"^obligations \(locate", "verifier_expected"),
    (r"^negatives", "wall_only"),
]
REASONS = {"native": "no stock pass (serve_v5 --skip-stock) or serve not run", "execution": "serve not run", "commit_io": "protocol phase 0 not reached",
           "replay": "phase 2 not reached", "interior_commit": "phase 2 not reached / prover timing absent", "prove": "no obligations/proving (refused before Q, or Q empty)",
           "verifier_expected": "obligations_v5 not run (refused before Q, or Q empty)", "verify_proofs": "verify_v5 not run or no proofs",
           "value_production": "no logs/value-production.json (no external interpreter run recorded; the interior trees, if any, are the capture's)"}


def classify(row: dict[str, Any]) -> str:
    c = str(row.get("class", ""))
    if re.match(RULES[1][0], str(row.get("stage", ""))):
        return "replay+interior_commit"  # A3's phase-2 wall row also says class "replay": it must be decomposed, never summed as class 4
    if c in CLASSES or c in EXTRA:
        return c
    if c == "prove_witness":
        return "prove"
    for pat, cls in RULES:
        if re.match(pat, str(row.get("stage", ""))):
            return cls
    return "unclassified"


def decompose_phase2(rec: Path, hw: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    sp_path = rec / "adapter" / "selective" / "selective_prover.json"
    if not sp_path.exists():
        return [], None
    sp = json.loads(sp_path.read_text())
    T = sp.get("timing") or {}
    cap = (sp.get("phase2") or {}).get("capture_timing") or {}  # Capture.timing: snapshot_s / emit_s / forward_s / sync_s (selective_prover.py:445)
    run = float(T.get("replay_run_s") or 0.0)
    fwd = float(cap.get("forward_s") or 0.0)
    bnd = float(T.get("phase2_boundaries_s") or 0.0)
    com = float(T.get("phase2_commit_s") or 0.0)
    # [E10 EOD 2026-09-11] E2's interior phase (selective_prover --interior-values: E2's layout over E1's interpreter rows, one tree per RU of J)
    # is timed by the prover as timing.phase2_interior_s (+ interior_commit.json); it is class 5 (interior commitments), NOT scheduler residual.
    # diag1 (22:54Z): 378.8 s for 4,622,106 words / 1,773 leaves / 105.5 M rows -- without this row it landed in class 4's residual.
    intr = float(T.get("phase2_interior_s") or 0.0)
    ic_path = rec / "adapter" / "selective" / "interior_commit.json"
    ic = json.loads(ic_path.read_text()) if ic_path.exists() else {}
    total = float(T.get("phase2_total_s") or (run + bnd + com + intr))
    resid = total - (run + bnd + com + intr)
    k = sp.get("k")
    rows = [
        {"stage": "replay: replay_run_s (forced replay of J incl. capture + leaf hashing) + boundary re-hash of J (the per-RU consistency determination)", "class": "replay",
         "seconds": round(run + bnd, 2), "hardware": hw, "batch": f"{k} selected request(s) in one batch", "label": "MEASURED",
         "note": f"selective_prover.json#timing: replay_run_s {run:.2f} + phase2_boundaries_s {bnd:.2f}; includes the work of RUs later exempted (F is decided after the replay)", "marker": "decomposition"},
        {"stage": "replay: bare forward inside replay_run_s (replay-side native compute)", "class": "native", "seconds": round(fwd, 2), "hardware": hw,
         "batch": f"{k} request(s)", "label": "MEASURED" if fwd else "UNAVAILABLE", "note": "Capture.timing.forward_s; a component of class 4, shown as the replay-side native baseline; NOT added to the class-1 total (serving side)",
         "marker": "decomposition-info"},
        {"stage": "interior commitments: build_commitments over the RU trees of J (leaf re-hash + trees)", "class": "interior_commit", "seconds": round(com, 2), "hardware": hw + " CPU",
         "batch": f"{(sp.get('phase2') or {}).get('ru_leaves')} RU leaves", "label": "MEASURED", "note": f"selective_prover.json#timing.phase2_commit_s {com:.2f}", "marker": "decomposition"},
        {"stage": "phase-2 residual (phase2_total_s minus replay_run + boundaries + commit + interior: scheduler/driver)", "class": "replay", "seconds": round(resid, 2), "hardware": hw,
         "batch": f"{k} request(s)", "label": "MEASURED (difference)", "note": f"phase2_total_s {total:.2f}; stated, not hidden", "marker": "decomposition"},
    ]
    if intr or ic:
        rows.insert(3, {"stage": "interior commitments: E2 layout over E1's interpreter rows (selective_prover --interior-values; one tree per RU of J)", "class": "interior_commit",
                        "seconds": round(intr, 2), "hardware": hw + " CPU (single-threaded Python, row parse + leaf hash + trees)",
                        "batch": f"{ic.get('words')} words / {ic.get('leaves')} leaves / {len(ic.get('committed') or [])} RU trees ({(ic.get('layout') or {}).get('rule')})",
                        "label": "MEASURED", "note": f"selective_prover.json#timing.phase2_interior_s {intr:.2f}; interior_commit.json; the value rows themselves are class `value_production` (E1, other pod)",
                        "marker": "decomposition"})
    return rows, {"replay_run_s": run, "forward_s": fwd, "boundaries_s": bnd, "commit_s": com, "interior_s": intr, "phase2_total_s": total, "residual_s": resid}


def value_production_rows(rec: Path) -> list[dict[str, Any]]:
    """[E10 EOD 2026-09-11] Class `value_production` (interpreter): E1's producer runs on ANOTHER machine (the CPU pod) before the protocol run;
    the runner records its /usr/bin/time measurement in ``logs/value-production.json`` (``seconds``, ``hardware``, ``batch``, ``note``, ``source``),
    one row per producer run.  MEASURED there, on that hardware -- reported beside the eight classes, summed into the protocol total ONCE, never
    mixed into class 5 (which is the tree over those rows).  Absent -> UNAVAILABLE with the reason."""
    p = rec / "logs" / "value-production.json"
    if not p.exists():
        return []
    doc = json.loads(p.read_text())
    rows = doc if isinstance(doc, list) else doc.get("rows") or [doc]
    out = []
    for r in rows:
        out.append({"stage": r.get("stage", "value production: reference interpreter over the RU (E1 producer)"), "class": "value_production",
                    "seconds": float(r["seconds"]), "hardware": r.get("hardware", "unknown"), "batch": r.get("batch", ""), "label": r.get("label", "MEASURED"),
                    "note": r.get("note", ""), "source": r.get("source", str(p)), "marker": "external"})
    return out


def read_gnu_time(p: Path) -> dict[str, Any] | None:
    """[E10 NIGHT 2026-09-12] GNU `/usr/bin/time -v` output of the prover process (protocol_v5 VERITOR_PROVER_TIME): peak RSS in GB, user/sys/wall
    seconds -- MEASURED on the record's hardware; None when the run was not wrapped (label UNAVAILABLE upstream)."""
    if not p.exists():
        return None
    import re
    t = p.read_text()
    def grab(rx: str) -> str | None:
        m = re.search(rx, t)
        return m.group(1).strip() if m else None
    el = grab(r"Elapsed \(wall clock\) time.*?: (.*)")
    wall = None
    if el:
        parts = [float(x) for x in el.split(":")]
        wall = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    rss_kb = grab(r"Maximum resident set size \(kbytes\): (\d+)")
    return {"label": "MEASURED", "source": str(p), "peak_rss_gb": round(int(rss_kb) / 1e6, 3) if rss_kb else None, "wall_s": wall,
            "user_s": float(grab(r"User time \(seconds\): ([0-9.]+)") or 0) or None, "sys_s": float(grab(r"System time \(seconds\): ([0-9.]+)") or 0) or None,
            "exit_status": grab(r"Exit status: (\d+)"),
            "meaning": "the whole selective_prover process of the protocol phase (phase 0 commitments + phase 2 replay + interior trees); peak RSS is the process maximum"}


def build(rec: Path, hardware: str) -> dict[str, Any]:
    log = rec / "costlog.jsonl"
    rows: list[dict[str, Any]] = []
    bad = 0
    for line in (log.read_text().splitlines() if log.exists() else []):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    decomp, timing = decompose_phase2(rec, hardware)
    classified: list[dict[str, Any]] = []
    for r in rows:
        cls = classify(r)
        if cls == "replay+interior_commit":
            if decomp:
                classified.append({**r, "v5_class": "wall_only", "note": (r.get("note") or "") + " | replaced by the decomposition rows (classes 4 + 5); kept as the phase-2 wall"})
                continue
            cls = "replay"
            r = {**r, "note": (r.get("note") or "") + " | UNSPLIT: no selective_prover.json; the phase-2 wall is charged to class 4 (it contains the interior commitment)"}
        classified.append({**r, "v5_class": cls})
    for d in decomp:
        classified.append({**d, "v5_class": d["class"] if d.get("marker") != "decomposition-info" else "info"})
    for d in value_production_rows(rec):
        classified.append({**d, "v5_class": "value_production"})
    totals: dict[str, dict[str, Any]] = {c: {"title": CLASS_TITLES[c], "seconds": 0.0, "rows": 0, "label": "UNAVAILABLE"} for c in CLASSES}
    extras: dict[str, float] = {}
    for r in classified:
        cls, sec = r["v5_class"], r.get("seconds")
        if sec is None:
            continue
        if cls in totals:
            totals[cls]["seconds"] = round(totals[cls]["seconds"] + float(sec), 2)
            totals[cls]["rows"] += 1
        else:
            extras[cls] = round(extras.get(cls, 0.0) + float(sec), 2)
    for c in CLASSES:
        t = totals[c]
        if t["rows"] == 0:
            t["seconds"] = None
            t["reason"] = REASONS[c]
        else:
            labels = {str(x.get("label", "MEASURED")).split(" ")[0] for x in classified if x["v5_class"] == c and x.get("seconds") is not None}
            t["label"] = "MEASURED" if labels == {"MEASURED"} else "MIXED: " + ",".join(sorted(labels))
        t["provenance"] = f"{log.name}: rows classified {c}"
    protocol_total = round(sum(t["seconds"] for c, t in totals.items() if c != "native" and t["seconds"] is not None), 2)
    prover_time = read_gnu_time(rec / "logs" / "prover.time")
    return {"schema": "veritor/a9/costlog-v5/v1", "prover_time": prover_time, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hardware": hardware, "classes": totals,
            "protocol_overhead_total_s": protocol_total,
            "protocol_overhead_note": "classes 2-8 + V (value production, other hardware) summed once (the decomposition replaces the phase-2 wall; wall_only / info rows excluded); class 1 is the baseline, not overhead",
            "other_verifier_s": extras.get("other_verifier", 0.0), "wall_only_s": extras.get("wall_only", 0.0), "unclassified_s": extras.get("unclassified", 0.0),
            "phase2_timing": timing, "rows": classified, "malformed_lines_skipped": bad,
            "labels": "MEASURED = wall clock of the named function on this record's hardware; UNAVAILABLE = the phase did not run (reason given); MODELED never appears here (predictions live in the report, labelled)"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--hardware", default="unknown (set --hardware)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rec = Path(a.record)
    doc = build(rec, a.hardware)
    out = Path(a.out) if a.out else rec / "costlog_v5.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print("| class | seconds | label | rows |")
    print("|---|---:|---|---:|")
    for c in CLASSES:
        t = doc["classes"][c]
        print(f"| {t['title']} | {t['seconds'] if t['seconds'] is not None else 'UNAVAILABLE'} | {t['label']} | {t['rows']} |")
    print(f"| other verifier work (sampling, step-4 checks) | {doc['other_verifier_s']} | MEASURED | - |")
    print(f"protocol overhead total (2-8): {doc['protocol_overhead_total_s']} s; unclassified {doc['unclassified_s']} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
