"""M2's committed coin schedule is the statement's: the prover's `coin_spec_of` (flock-circuit's `zk_rounds`) and the Lean
verifier of record's setup (`Flock.zkRounds`, `Coins.sized`) size it from the statement's m, k_log and regions, and the
verifier's own `Hello` refuses any other (R7), so a statement that draws more rounds needs no code change. Here the
derivation is held to the rounds the live prover drew in recorded --zk sessions at two sizes (RoPE d64 at m = 25 and 26):
each rep's stream has exactly `zk_rounds` rounds, and the session's `Hello` committed `2^⌈log₂⌉` of them."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_flock.rec_algebra import fast100
from verity_flock.rec_live import PT_LOCAL

#: `art:b9cc903c`, test_lean_zk.py's sessions, which `suites.py` fetches before the suite runs
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "zk-sessions"
SESSIONS = {"selftest/honest": 25, "m26/l0000-618392": 26, "j2/l0000-619480": 25, "drawn/l0000-621181": 25}


def zk_rounds(m: int, k_log: int, regions: int) -> int:
    """`Flock.zkRounds`: the masked zerocheck and lincheck, the ring switch (one claim a region at each of two link points),
    Ligerito's levels and the inner proof."""
    s = fast100(m)
    return m + k_log - 6 + 3 + 2 * regions + s.k0 + 11 * (len(s.levels) - 1) + 7 + 4


def rounds_log(rounds: int) -> int:
    return (rounds - 1).bit_length()


def test_the_derivation_is_monotone_and_crosses_a_power_of_two():
    assert [rounds_log(r) for r in (1, 2, 3, 64, 65, 100, 128, 129)] == [0, 1, 2, 6, 7, 7, 7, 8]
    assert rounds_log(zk_rounds(26, 22, 3)) == 7 and rounds_log(zk_rounds(32, 27, 64)) == 8


@pytest.mark.parametrize("session", sorted(SESSIONS))
def test_recorded_sessions_draw_the_derived_schedule(session):
    stage = FIXTURE / "stage" / "circuit.txt"
    assert stage.is_file(), f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures {FIXTURE.relative_to(FIXTURE.parents[5])}`"
    meta = json.loads(stage.read_text().split("\n", 2)[1].removeprefix("META "))
    rec = json.loads((FIXTURE / "sessions" / session / "session.json").read_text())
    hello = json.loads(rec["hello"])
    # m = k_log + nbl, the link's m is PT_LOCAL + nbl; one region a port (`HmRow.regions`)
    m = meta["k_log"] + hello["link"]["m"] - PT_LOCAL
    assert m == SESSIONS[session]
    want = zk_rounds(m, meta["k_log"], len(meta["ports"]))
    assert {s["stream"]: len(s["rounds"]) for s in rec["streams"]} == {s: want for s in hello["coins"]["streams"][1:]}
    assert hello["coins"]["rounds_log"] == rec["coin_tree"]["spec"]["rounds_log"] == rounds_log(want), (
        f"{session}'s Hello commits 2^{hello['coins']['rounds_log']} rounds, not the statement's 2^{rounds_log(want)}: "
        "regenerate the fixture (verity/ml/flock/pod/92-zk-sessions.sh)")
