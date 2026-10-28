"""`served_tie_zk`'s scan: the witness pieces it cuts (`blobs_of`) read by every zero-knowledge driver's rule
(`leak_scan`, whose own cases are ``benchmarks/network_traces/tests/test_leak_scan.py``). REDTEAM phase 16 finding 1's case (the
routed tie's switches, found in one repetition) is leaked."""
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


def test_phase16_a_switch_leak_in_one_repetition_is_leaked():
    """The routed tie's switch row as `blobs_of` writes it is low-entropy, and found in one repetition it fails the
    statement, as in both; a positions chunk does too, and a salt `prove` drew that a rescan can't recompute."""
    unit = SimpleNamespace(words={"sw_route": [0, 1, 2, 3, 2, 1, 0, 3], "pos_in": list(range(300, 320)), "o0": [1]})
    blobs, held = Z.blobs_of([unit], {"srv": [bytes(range(32))], "o0": [bytes(32)]})
    assert held == {"sw_route 0", "pos_in 0"} and set(blobs) == held | {"srv salt 0"}
    assert len(set(blobs["sw_route 0"])) <= 4 < len(set(blobs["pos_in 0"]))
    sw = "sw_route 0"
    assert Z.LS.classify(_received({REPS[0]: [sw]}), blobs) == ([sw], {})
    assert Z.LS.classify(_received({REPS[0]: [sw], REPS[1]: [sw]}), blobs) == ([sw], {})
    assert Z.LS.classify(_received({REPS[0]: ["pos_in 0"]}), blobs)[0] == ["pos_in 0"]
    got = _received({REPS[0]: ["srv salt 0", sw]})
    assert Z.LS.rescan(got, {k: blobs[k] for k in held}, None)["leaked"] == ["srv salt 0", sw]
