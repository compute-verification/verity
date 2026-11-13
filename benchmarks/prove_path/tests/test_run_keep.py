"""run.sh's ``keep`` copies a run's records out of its scratch directory, which its EXIT trap removes: a kept case's blobs
name the kept copies, never the scratch paths, so the served case, its hand-off and the Lean recomputation find them."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RUN_SH = Path(__file__).resolve().parents[1] / "run.sh"


def block(first: str, last: str) -> str:
    lines = RUN_SH.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(first))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith(last))
    return "\n".join(lines[i:j + 1]) + "\n"


def test_a_kept_cases_blobs_name_the_kept_files(tmp_path):
    w, o = tmp_path / "w", tmp_path / "out"
    window = w / "served" / "window"
    (window / "m0").mkdir(parents=True)
    (window / "m0" / "circuit.txt").write_text("c")
    (window / "m0" / "proof.bin").write_bytes(b"p")
    other = str(tmp_path / "elsewhere.bin")
    blobs = [str(window / "m0" / "circuit.txt"), str(window / "m0" / "proof.bin"), other]
    (window / "case-v2.json").write_text(json.dumps({"blobs": blobs, "commitment": "ab"}))
    (window / "audit.json").write_text(json.dumps({"accepted": True}))
    got = subprocess.run(["bash", "-c", f"set -uo pipefail\nW={w} O={o} PY={sys.executable}\n"
                          + block("keep() {", "}") + "keep served\n"], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    case = json.loads((o / "served" / "window" / "case-v2.json").read_text())
    kept = o.resolve() / "served" / "window" / "m0"
    assert case["blobs"] == [str(kept / "circuit.txt"), str(kept / "proof.bin"), other]
    assert all(Path(b).is_file() for b in case["blobs"][:2])
    assert case["commitment"] == "ab"
    assert json.loads((o / "served" / "window" / "audit.json").read_text()) == {"accepted": True}
