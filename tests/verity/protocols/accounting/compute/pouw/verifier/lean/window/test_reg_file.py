"""`pouw-leaf-own`'s sample policy (`PouwWindow.Lanes.checkFull`) on the shared harness (`verity_difftest`): it accepts a
session record that samples every unit and refuses one that doesn't.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import verity_difftest as D
from hypothesis import strategies as st

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = (Path(__file__).resolve().parents[9] / "verity" / "protocols" / "accounting" / "compute" / "pouw" / "verifier" / "lean" / "window")
EXE = "pouw-window-difftest"

#: `pouw-leaf-own`'s sample policy alone: a session record's verdict (no Python counterpart)
FULL = D.Port(EXE, "full_draw", reference=lambda x: None, inputs=st.nothing(), package=LEAN)

server = D.server_fixture(FULL)


def _record(draw: dict | None) -> str:
    return json.dumps({"format": "flock-session/v1"} | ({} if draw is None else {"unit_draw": draw}))


@pytest.mark.parametrize("n", [1, 3])
def test_the_full_draw_passes(n: int, server) -> None:
    """A session that samples every unit of its population, `subset` of all of them, passes the sample policy."""
    draw = {"law": "subset", "population": n, "k": n, "units": list(range(n))}
    assert FULL.lean(server, {"record": _record(draw)}) == ("output", "true")


@pytest.mark.parametrize(("draw", "why"), [
    (None, "has no draw"),
    ({"law": "subset", "population": 3, "k": 2, "units": [0, 2]}, "not every unit"),
    ({"law": "subset", "population": 0, "k": 0, "units": []}, "not every unit"),
    ({"law": "bernoulli", "population": 2, "p": "1/1", "units": [0, 1]}, "law is not subset"),
    ({"law": "subset", "population": 2, "k": 2, "units": [1, 0]}, "the session's draw: U2"),
])
def test_the_draw_policy_refuses_a_partial_draw(draw: dict | None, why: str, server) -> None:
    """A session record with no sample, a sample that leaves a unit out, an empty population, another sampling design, or a sample
    verity-verify-proof's format refuses, is refused by name."""
    kind, got = FULL.lean(server, {"record": _record(draw)})
    assert kind == "reject" and why in got
