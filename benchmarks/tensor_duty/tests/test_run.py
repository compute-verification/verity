"""run.sh's text: the SM clocks it locks start at the card's `clock.sm`, which run.sh takes as a required argument
(`CLOCKS`, refused by name when unset) and never states as a default; the one value it sets is --no-lock's `default`."""
from __future__ import annotations

import re
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "run.sh"


def test_run_requires_clocks_naming_clock_sm_and_gives_it_no_default() -> None:
    text = RUN.read_text()
    assert re.search(r"\$\{CLOCKS:\?[^}]*clock\.sm[^}]*\}", text)
    assert not re.search(r"\$\{CLOCKS:?[-=]", text)
    assert [v for v in re.findall(r"\bCLOCKS=(\S+)", text) if not v.startswith("${CLOCKS:?")] == ["default"]
