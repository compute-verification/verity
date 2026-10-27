"""The circuits area under the repository's one ruff configuration, the root ``ruff.toml``, with the ruff the root ``dev`` group
pins: no finding, today's count.  ``tests/lint/test_repo_ruff.py`` holds the tree per (file, rule) to
``tests/lint/allowlists/repo_ruff.json``, which has no entry here, and this holds the area to none in its own suite."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
AREA = "verity/ml/circuits"


def test_the_area_has_no_ruff_finding() -> None:
    r = subprocess.run([sys.executable, "-m", "ruff", "check", "--config", str(ROOT / "ruff.toml"), "--no-cache",
                        "--output-format", "concise", AREA], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (f"ruff (ruff.toml) on {AREA}: fix each finding (`ruff check --fix PATH` fixes some), or where the "
                               f"rule is wrong for that line, say why on it (`# noqa: RULE - why`)\n{r.stdout}{r.stderr}")
