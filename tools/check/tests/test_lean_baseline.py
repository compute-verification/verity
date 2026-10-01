"""`tools/check/lean_baseline.py`: the per-module build times it reads from Lake's log."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("check_lean_baseline", ROOT / "tools" / "check" / "lean_baseline.py")
LB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LB)


def test_built_reads_each_of_lakes_time_formats():
    log = "\n".join([
        "✔ [2/30] Built Verity.TC.Pipeline (281ms)",
        "⚠ [12/30] Built Verity.TC.Proofs.Group (1.2s)",
        "warning: Verity/TC/Proofs/Group.lean:3:1: unused",
        "✔ [8/30] Built Verity.TC.Proofs.Prod (42s)",
        "✔ [9/30] (Optional) Built Dep.Extra (7ms)",
        "⚠ [10/30] Replayed Verity.TC.Proofs.Step",
        "✔ [29/30] Built Verity",
        "Build completed successfully (30 jobs).",
    ])
    assert LB.built(log) == {"Verity.TC.Proofs.Prod": 42.0, "Verity.TC.Proofs.Group": 1.2, "Verity.TC.Pipeline": 0.281,
                             "Dep.Extra": 0.007}
    assert list(LB.built(log))[0] == "Verity.TC.Proofs.Prod"
