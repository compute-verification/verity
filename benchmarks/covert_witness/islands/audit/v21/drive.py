"""AUDIT driver: runs probe.py's and credit.py's variants on a local tiny start the way candidate.sh --leased runs an
island (the harness's `--script` run, then evaluate.py on its record), and writes OUT/summary.json with te_core's ρ at
r = 1 from each run's record. CPU, about a minute per variant.

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
#: credit.py's variants, in class reference
CREDIT = ["dense", "declare_full", "forge_decode"]


def rho_grid(run_json: Path) -> dict:
    """te_core's ρ at r = 1 for each k, from the run's record and its verified families' coverage (the ledger's rule)."""
    from te_core.annotate import annotate
    from te_core.score import score_grid

    rec = json.loads(run_json.read_text())
    ann = annotate(rec)
    covered: dict[int, float] = {}
    for f in rec.get("families", []):
        beta = {int(t): b for t, b in f.get("beta", {}).items()}
        for t in f.get("covers", []):
            if t in beta:
                covered[t] = min(beta[t], covered.get(t, beta[t]))
    grid = score_grid(ann, covered, ann.bits() or 16, rec["c"])
    return {"covered_steps": sorted(covered), "beta_t_max": max(covered.values(), default=None), "p": ann.bits() or 16,
            "rho_r1": {k: round(grid[f"{k}/r1"]["rho"], 3) for k in ("10kB", "100kB", "1MB")},
            "option_10kB_r1": grid["10kB/r1"]["option"],
            "families": [{k: f.get(k) for k in ("family", "declared", "sampled", "passed", "failed", "covers", "cost")}
                         for f in rec.get("families", [])]}


def run(out: Path, v: str, cap: float, klass: str = "e1mb") -> dict:
    d = out / v
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    extra = ["--cached", str(out / "honest" / "sub" / "E.bin")] if v == "replay" else []
    script = HERE / ("credit.py" if v in CREDIT else "probe.py")
    klass = "reference" if v in CREDIT else klass
    meta = {"class": klass, "start": "s50", "launcher": "audit/v21/drive.py", "note": "audit"}
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "te_core.run", "--script", str(script), "--out", str(d / "run"),
                        "--families", str(d / "families"), "--budget-fraction", "0.98", "--budget-minutes",
                        str(cap / 60), "--meta", json.dumps(meta), "--",
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
            "last_losses": (notes.get("losses") or [])[-2:], "te_core": rho_grid(d / "run" / "run.json")}


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
                                                       "phi_declared", "phi_actual", "error")}
                         | {"rho_r1": rows[-1].get("te_core", {}).get("rho_r1"),
                            "covered": rows[-1].get("te_core", {}).get("covered_steps")}), flush=True)
    old = json.loads((out / "summary.json").read_text())["rows"] if (out / "summary.json").exists() else []
    rows = [r for r in old if r["variant"] not in want] + rows
    (out / "summary.json").write_text(json.dumps({"schema": "te-audit/v21-probes/v1", "cw_version": prepare.VERSION,
                                                  "root": str(prepare.ROOT), "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
