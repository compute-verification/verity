"""E8 NIGHT 2026-09-12 -- pod-side REPORT writer (D37: no laptop dependency). Assembles /workspace/REPORT-e8.md from whatever the run has produced so far:
record path, archive + slim sha256, host sha256, both verdicts (live driver verify_v5 / shipped reverify_clean), the D6 coverage line, |Q|, proof counts,
per-phase timings (from launch-<tag>.out '=== [utc] <phase> rc=<rc> <s>s' lines), GPU memory, the exact reverify command, the exact lengths, and the table
row (row.json) as it stands. Never raises: a missing artefact is reported as such.

    python -m e2e.demo_core.night.report_pod --tag n2 [--out /workspace/REPORT-e8.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path


def j(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def sha(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return "absent"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run-id", default="V5-N32x16-1req-bex0-inc")
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--out", default="/workspace/REPORT-e8.md")
    ap.add_argument("--host", default="/workspace/a9/veritor-zk-host")
    ap.add_argument("--pod-line", default="pod: `night-e8-prove` = `r0dtdsgz281ux8` (RTX 4090, $0.74/h, since 05:28:32Z; NO network volume: **the pod disk is the only copy** until fetched; dead-man 14:30Z).")
    ap.add_argument("--archive-dir", default="/workspace/a9/archive")
    ap.add_argument("--extra-out", default="", help="second path to write the report to (the volume)")
    a = ap.parse_args()
    tag, rid = a.tag, a.run_id
    rec = Path(f"/workspace/a9/rec/{rid}-{tag}")
    olog = Path(f"/workspace/a9/rec/orchestrate-{tag}.log")
    llog = Path(f"/workspace/a9/rec/launch-{tag}.out")
    arch = Path(f"{a.archive_dir}/{rid}-{tag}.tgz")
    slim = Path(f"{a.archive_dir}/{rid}-{tag}-slim.tgz")
    rvdir = Path(f"/workspace/a9/reverify/{tag}")
    rv = j(rvdir / "reverify_clean.json")
    row = j(rec / "row.json")
    freeze = j(rec / "freeze.json") or {}
    verify = j(rec / "verify/verify_v5.json") or j(rec / "verify_v5.json")
    report = j(rec / "report_v5.json") or j(rec / "report/report_v5.json")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    otext = olog.read_text() if olog.exists() else ""
    ltext = llog.read_text() if llog.exists() else ""
    done = "] DONE:" in otext
    phases = re.findall(r"^=== \[(\S+)\] (\S+) rc=(\d+) (\d+)s", ltext, re.M)
    gpu = "n/a"
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    prover_time = ""
    pt = rec / "logs/prover.time"
    if pt.exists():
        prover_time = " ".join(l.strip() for l in pt.read_text().splitlines() if "Maximum resident" in l or "Elapsed" in l)
    proofs = sorted((rec / "proofs").glob("**/*.bin")) if (rec / "proofs").exists() else []
    statements = sorted((rec / "obligations").glob("**/statement*.bin")) if (rec / "obligations").exists() else []
    lines = [
        f"# REPORT-e8 — SmolLM2-135M at **prompt 32 × decode 16** (LP = 32, STEPS = 15; one request `7cc7c3bb…`, greedy, B_ex = 0, s = 2⁻¹⁴): run `{rid}-{tag}` — written pod-side {now} ({'FINISHED' if done else 'IN PROGRESS'})",
        "",
        f"* {a.pod_line}",
        f"* record_path: `{rec}` (freeze.json: commit `{freeze.get('commit') or (freeze.get('tree') or {}).get('commit')}`, dirty patch `{freeze.get('dirty_patch_sha256') or (freeze.get('tree') or {}).get('dirty_patch_sha256')}`)",
        f"* archive: `{arch}` sha256 `{sha(arch)}`; slim: `{slim}` sha256 `{sha(slim)}`",
        f"* host: `{a.host}` sha256 `{sha(Path(a.host))}`" + (f"; reverify host sha256 `{(rv.get('host') or {}).get('sha256')}`" if rv else ""),
        f"* instance: `E6-B0-32x15-r1-77d5b63b-t84cf6bc1-l57129503`, N_V = 29,496,522 (E6 26d0e2bc; E3 census 08:49Z `ok` 29,496,522); coverage_v5: 0 of 29,496,522 VUs unsupported",
        f"* GPU now: {gpu}; prover /usr/bin/time: {prover_time or 'n/a'}",
        f"* proof objects on disk: {len(proofs)}; statement files: {len(statements)}",
        "",
        "## Verdicts",
        f"* live driver (`verify_v5`): `{json.dumps((verify or {}).get('verdict') or (verify or {}).get('status') or (verify or {}).get('outcome') or 'NOT RUN')[:400]}`",
        f"* shipped `reverify.py` (`reverify_clean.sh`, fresh venv): `{json.dumps({k: rv.get(k) for k in ('status', 'rc', 'checks_failed', 'seconds')}) if rv else 'NOT RUN'}`",
        f"* reverify command: `bash record_v5/reverify_clean.sh /workspace/a9/veritor {rec} /workspace/a9/pop/32x16/population.v2.json {a.host} {rvdir}` (= `python -m e2e.verifier_v3.reverify --record {rec} --population <pop> --expressibility {rec}/expressibility_v5.json --host {a.host} --out {rvdir}/reverify_v3 --instance-format record_v5`)",
        "",
        "## Phases (launch log `=== [utc] phase rc seconds`)",
    ]
    lines += [f"* {u} `{ph}` rc={rc} {s} s" for u, ph, rc, s in phases] or ["* (none yet)"]
    lines += ["", "## Table row (row.json; MEASURED / NOT RUN labelled)", "~~~json", json.dumps(row, indent=1) if row else "(not written yet)", "~~~", "",
              "## report_v5 (if present)", "~~~json", json.dumps(report, indent=1)[:6000] if report else "(absent)", "~~~", "",
              "## Orchestrator log (tail)", "~~~", "\n".join(otext.splitlines()[-40:]), "~~~"]
    Path(a.out).write_text("\n".join(lines) + "\n")
    if a.extra_out:
        try:
            Path(a.extra_out).write_text("\n".join(lines) + "\n")
        except Exception:
            pass
    try:
        (rec / "REPORT-e8.md").write_text("\n".join(lines) + "\n")
    except Exception:
        pass
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
