"""job.sh's steps, run against a stand-in 85 that only says which step it is: how often each runs, and oprove's stamped log."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
POD = "verity/protocols/proofs/flock/prover/pod"
GNU_TIME = os.access("/usr/bin/time", os.X_OK) and subprocess.run(["/usr/bin/time", "-v", "true"],
                                                                     capture_output=True).returncode == 0


def _stand_in(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


@pytest.mark.skipif(not GNU_TIME, reason="job.sh times every step with GNU time at /usr/bin/time")
def test_only_the_steps_the_metrics_take_medians_of_repeat_and_oprove_s_log_is_stamped(tmp_path: Path) -> None:
    tree, bin_ = tmp_path / "tree", tmp_path / "bin"
    for d in ("benchmarks/recursion_e2e", "tools/config"):
        (tree / d).parent.mkdir(parents=True, exist_ok=True)
        (tree / d).symlink_to(ROOT / d)
    _stand_in(tree / POD / "85-rec-reprice.sh", 'echo "step $STEP"; echo "oprove L0 started"')
    (tree / POD / "lake_closure.py").write_text("print('0' * 16)\n")
    _stand_in(bin_ / "nproc", "echo 32")
    _stand_in(bin_ / "getconf", "case $1 in _PHYS_PAGES) echo 67108864 ;; *) echo 4096 ;; esac")
    for tool in ("cargo", "lake"):
        _stand_in(bin_ / tool, "exit 0")
    (bin_ / "python3").symlink_to(sys.executable)
    out = tmp_path / "out"
    subprocess.run(["/bin/bash", "benchmarks/recursion_e2e/job.sh", f"OUT={out}", f"FLOCK_WORK={tmp_path / 'work'}",
                    "STEPS=icheck,vstage,vcap,oprove,oreplay,lean", "REPEAT=3", "NEG=", "GPU=0"], cwd=tree,
                   env={"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path)}, capture_output=True, text=True)
    runs = {s.name: sorted(r.name for r in s.iterdir()) for s in (out / "steps").iterdir()}
    assert runs == {"icheck": ["r1"], "oreplay": ["r1"], "oprove": ["r1"], "vstage": ["r1", "r2", "r3"],
                    "vcap": ["r1", "r2", "r3"], "lean": ["r1", "r2", "r3"]}
    lines = (out / "steps" / "oprove" / "r1" / "step.log").read_text().splitlines()
    assert [re.sub(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ", "", ln) for ln in lines] == ["step oprove", "oprove L0 started"]
    assert (out / "steps" / "lean" / "r1" / "step.log").read_text().splitlines() == ["step lean", "oprove L0 started"]
