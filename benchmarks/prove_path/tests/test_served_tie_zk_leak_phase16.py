"""REDTEAM phase 16, target 3 against #3846 (cursor/served-tie-routed-061a @d62b712618): the routed served-rows tie
(ServedRowsTie_v2 via IO-TIE's IoTieRoute) and PROVE-PATH's changed ZK leak-scan rule (`served_tie_zk.classify`).

`classify` flags a received witness piece as ``leaked`` (which fails a statement) iff it has >= DISTINCT (8) distinct
byte values, or -- with more than one repetition -- it is found in every repetition. A LOW-entropy piece found in only
some repetitions is reported ``structured`` and does NOT fail a statement; the rule's stated reason is that the proof
streams hold small integers and zeros at fixed format offsets, valued afresh per repetition.

The miss: the routed tie's own witness is the routing SWITCHES (`sw_*`), which `blobs_of` serialises as u16 of small
switch indices -- low-entropy by construction, unlike the positions (`pos_*`, spanning the window) and the salts (64
random bytes). So the one scan dedicated to catching a verbatim switch leak exempts exactly the switch class whenever
it is received in a single repetition: a prover that leaks the step's routing (the served->token binding) into one
rep's received stream is reported ``structured``, not ``leaked``, and the statement holds. Positions and salts, being
high-entropy, are caught wherever they are found.

Every assertion is against the landed `served_tie_zk.classify`/`blobs_of`; this ties `blobs_of`'s real serialisation
of the switch witness to the `classify` blind spot (the PR's own `test_served_tie_zk.py` tests `classify` on an ad-hoc
blob only).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import served_tie_zk as Z  # noqa: E402

REPS = ("s-circuit.rep0.bin", "s-circuit.rep1.bin")


def _received(found: dict[str, list[str]]) -> dict:
    files = {f: {"bytes": 1, "found": {k: 1 for k in found.get(f, [])}} for f in (*REPS, "s-session.json")}
    return {"files": files, "verifier_found": sorted({k for ks in found.values() for k in ks})}


def _blobs() -> dict[str, bytes]:
    """One routed unit's witness as `blobs_of` serialises it: a switch row (small routing indices -> low entropy) and a
    positions row (spanning the window -> high entropy)."""
    u = SimpleNamespace(words={"sw_route": [0, 1, 2, 3, 2, 1, 0, 3], "pos_in": list(range(300, 320))})
    blobs, _held = Z.blobs_of([u], {})
    return blobs


def test_the_routed_ties_switch_witness_is_low_entropy_and_the_positions_are_not():
    blobs = _blobs()
    assert set(blobs) == {"sw_route 0", "pos_in 0"}
    assert len(set(blobs["sw_route 0"])) < Z.DISTINCT       # the routing switches: few distinct byte values
    assert len(set(blobs["pos_in 0"])) >= Z.DISTINCT        # the positions: >= DISTINCT distinct byte values


def test_a_switch_leak_in_a_single_repetition_is_reported_structured_not_leaked():
    """The scan's blind spot: the routed tie's switch witness, found verbatim in one rep's received stream, is
    exempted as structured -- it does not fail a statement -- though it is a real leak of the step's routing. Only when
    the same low-entropy switch piece sits in BOTH reps does the rule catch it."""
    blobs, sw = _blobs(), "sw_route 0"
    leaked, structured = Z.classify(_received({REPS[0]: [sw]}), blobs)
    assert leaked == [] and sw in structured                       # MISS: switch leak in rep 0 only
    assert Z.classify(_received({REPS[0]: [sw], REPS[1]: [sw]}), blobs)[0] == [sw]   # both reps -> caught


def test_positions_and_salts_are_caught_wherever_found():
    """The high-entropy witness (positions, and the non-opened salts) is leaked wherever found, even one rep -- so the
    exemption is precisely the single-rep low-entropy case the switches fall in."""
    blobs, pos = _blobs(), "pos_in 0"
    assert Z.classify(_received({REPS[0]: [pos]}), blobs)[0] == [pos]
    salted, _ = Z.blobs_of([SimpleNamespace(words={})], {"wit": [bytes(range(64))]})  # a witness-row salt (not OPENED)
    k = next(iter(salted))
    assert len(set(salted[k])) >= Z.DISTINCT
    assert Z.classify({"files": {"s-session.json": {"found": {k: 1}}}, "verifier_found": [k]}, salted)[0] == [k]
