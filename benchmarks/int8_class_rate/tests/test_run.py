"""run.sh's refusal: the clock it locks is the card's `clock.tensor`, which run.sh takes as a required argument and
never states, so a run without CLOCK stops before it builds or writes anything."""
from __future__ import annotations

import subprocess
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "run.sh"


def test_run_refuses_without_a_clock_and_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    done = subprocess.run(["bash", str(RUN)], env={"PATH": "/usr/bin:/bin", "OUT": str(out)}, capture_output=True,
                          text=True, check=False)
    assert done.returncode != 0
    assert "clock.tensor" in done.stderr
    assert not any(out.iterdir())
