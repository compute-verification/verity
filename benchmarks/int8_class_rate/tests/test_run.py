"""run.sh's text: the clock it locks is the card's `clock.tensor`, which run.sh takes as a required argument (`CLOCK`,
refused by name when unset) and never states as a default."""
from __future__ import annotations

import re
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "run.sh"


def test_run_requires_clock_naming_clock_tensor_and_gives_it_no_default() -> None:
    text = RUN.read_text()
    assert re.search(r"\$\{CLOCK:\?[^}]*clock\.tensor[^}]*\}", text)
    assert not re.search(r"\$\{CLOCK:?[-=]", text)
