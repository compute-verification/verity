"""Run probe_train.py under te_core's harness once per variant, each in its own process (the adapter's audit hook can't be
removed), and print what the counter recorded and the ρ the scorer gives, so each route's effect on |C|, U, p and ρ is
read from te_core itself.

    CW_FAKE_ROOT=DIR PYTHONPATH=<te_core's parent> python run_probes.py OUTDIR

It uses te_core's tiny fake start (te_core/tests/cw_fake), so each variant is a seconds-long CPU run. ρ is scored at a
fixed c, with p left to the scorer's default (`ann.bits()`), as te_core.ledger ingests a run.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

import te_core
from te_core.annotate import annotate
from te_core.score import score_grid

HERE = Path(__file__).resolve().parent
FAKE_PREPARE = Path(te_core.__file__).resolve().parent / "tests" / "cw_fake" / "prepare.py"
VARIANTS = ["baseline", "data_launder", "narrow_bits", "custom_op", "numpy_step", "dict_encoder", "index_lookup",
            "div_matmul", "thread", "replay"]
#: a c at which the baseline's deep updated share gives ρ > 1 at 10kB/r1 on the fake start's tiny |C|
C = 4.0e9


def stage(out: Path) -> Path:
    """probe_train.py beside te_core's fake prepare.py, so cw.import_prepare finds the fake before the real one."""
    d = out / "island"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(FAKE_PREPARE, d / "prepare.py")
    shutil.copy(HERE / "probe_train.py", d / "probe_train.py")
    spec = importlib.util.spec_from_file_location("fake_prepare", d / "prepare.py")
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    torch.save(fake.theta_of()["w"] + 0.05, d / "replay_w.pt")
    return d / "probe_train.py"


def one(variant: str, out: Path, script: Path) -> dict:
    run_out, sub = out / variant / "run", out / variant / "sub"
    proc = subprocess.run([sys.executable, "-m", "te_core.run", "--script", str(script), "--out", str(run_out),
                           "--evaluator", "none", "--budget-macs", "1e9", "--", "--start", "fake", "--out", str(sub),
                           "--variant", variant], capture_output=True, text=True)
    rec = json.loads((run_out / "run.json").read_text())
    ann = annotate(rec)
    p = ann.bits() or 16
    grid = score_grid(ann, {}, p, C)
    by: dict[str, int] = {}
    for s in rec["steps"]:
        for k, v in s["by"].items():
            by[k] = by.get(k, 0) + v
    return {"variant": variant, "rc": proc.returncode, "invalid": rec["invalid"], "flags": rec["flags"],
            "macs": rec["totals"]["macs"], "by": by, "updated_macs": ann.updated(), "depth": ann.depth(),
            "bits": ann.bits(), "p": p, "steps": len(rec["steps"]),
            "model_is_decode": rec.get("submitted", {}).get("model_is_decode"),
            "rho_10kB_r1": grid["10kB/r1"]["rho"], "phi_10kB_r1": grid["10kB/r1"]["phi"],
            "d_min_10kB_r1": grid["10kB/r1"]["d_min"]}


def main() -> int:
    out = Path(sys.argv[1])
    script = stage(out)
    rows = [one(v, out, script) for v in VARIANTS]
    (out / "probes.json").write_text(json.dumps(rows, indent=1) + "\n")
    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
