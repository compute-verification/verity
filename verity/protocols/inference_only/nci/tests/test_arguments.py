import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import arguments  # noqa: E402

LOCK = HERE.parents[2] / "Security" / "lean-audit.json"


def test_argument_map_has_no_problems():
    assert arguments.problems(LOCK) == []


def test_every_claim_is_rendered():
    text = "\n".join(arguments.render())
    for c in arguments.load()["claims"]:
        assert c["statement"] in text
