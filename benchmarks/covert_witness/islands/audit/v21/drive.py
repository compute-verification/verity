"""AUDIT driver: runs probe.py's variants on a local tiny start the way candidate.sh --leased runs an island (the harness's
`--script` run, then evaluate.py on its record), and writes OUT/summary.json. CPU, seconds per variant.

    cd benchmarks/covert_witness && CW_ROOT=/tmp/cwtest CW_LF_CORPUS=/tmp/cwtest/lf CW_DEVICE=cpu \\
        python islands/audit/v21/drive.py OUT [VARIANT...]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CW = HERE.parents[2]
VARIANTS = ["honest", "heldout", "fromfile", "forge", "thread", "getattr", "replay", "superset", "down", "down_ulp"]


def run(out: Path, v: str, cap: float, klass: str = "e1mb") -> dict:
    d = out / v
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    extra = ["--cached", str(out / "honest" / "sub" / "E.bin")] if v == "replay" else []
    meta = {"class": klass, "start": "s50", "launcher": "audit/v21/drive.py", "note": "audit"}
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "te_core.run", "--script", str(HERE / "probe.py"), "--out", str(d / "run"),
                        "--budget-fraction", "0.98", "--budget-minutes", str(cap / 60), "--meta", json.dumps(meta), "--",
                        "--start", "s50", "--out", str(d / "sub"), "--variant", v, "--class", klass, *extra],
                       cwd=CW, capture_output=True, text=True)
    (d / "train.log").write_text(p.stdout + p.stderr)
    (d / "timing.json").write_text(json.dumps({"process_seconds": round(time.time() - t0, 3), "rc": p.returncode,
                                               "gpu": ""}) + "\n")
    if not (d / "sub" / "meta.json").is_file() or not (d / "run" / "run.json").is_file():
        return {"variant": v, "error": "no submission or no run.json", "log_tail": (p.stdout + p.stderr)[-600:]}
    e = subprocess.run([sys.executable, str(CW / "evaluate.py"), "--submission", str(d / "sub"), "--class", klass,
                        "--timing", str(d / "timing.json"), "--run", str(d / "run" / "run.json"),
                        "--out", str(d / "witness.json")], cwd=CW, capture_output=True, text=True)
    (d / "evaluate.log").write_text(e.stdout + e.stderr)
    w = json.loads((d / "witness.json").read_text())
    r = json.loads((d / "run" / "run.json").read_text())
    h = w.get("harness", {})
    notes = w.get("notes") or {}
    return {"variant": v, "valid": w.get("valid"), "failed_checks": sorted(k for k, x in w["checks"].items() if x is not True),
            "bpb": w.get("score", {}).get("bpb"), "harness_macs": h.get("macs"), "harness_flops": h.get("flops"),
            "analytic_train_flops": w.get("train", {}).get("flops"), "steps": h.get("steps"),
            "gpu_seconds": w.get("gpu_seconds"), "harness_invalid": h.get("invalid"), "flags": h.get("flags"),
            "phi_declared": w.get("structure", {}).get("phi"), "phi_actual": w.get("structure_actual", {}).get("phi"),
            "changed_declared": len(w.get("changed", {}).get("declared", [])),
            "changed_actual": len(w.get("changed", {}).get("actual", [])), "E_bytes": w.get("E", {}).get("bytes"),
            "start_val_bpb_in_run": r.get("start", {}).get("val_bpb"), "curve": w.get("curve"),
            "notes": {k: notes[k] for k in notes if k not in ("args", "losses", "audit")},
            "last_losses": (notes.get("losses") or [])[-2:]}


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    want = sys.argv[2:] or VARIANTS
    os.environ.setdefault("AGENT_GUARD_LOCAL_OK", "1")
    sys.path.insert(0, str(CW))
    import prepare

    rows = []
    for v in want:
        rows.append(run(out, v, prepare.GPU_SECONDS_CAP))
        print(json.dumps({k: rows[-1].get(k) for k in ("variant", "valid", "failed_checks", "bpb", "harness_flops",
                                                       "phi_declared", "phi_actual", "error")}), flush=True)
    (out / "summary.json").write_text(json.dumps({"schema": "te-audit/v21-probes/v1", "cw_version": prepare.VERSION,
                                                  "root": str(prepare.ROOT), "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
