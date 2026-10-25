"""`served_tie_zk.classify`: a found piece of at least DISTINCT byte values is leaked wherever it is found, and a piece of
fewer is leaked only when every repetition's proof stream holds it, else structured."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import served_tie_zk as Z  # noqa: E402

BLOBS = {"salt @0": bytes(range(16)), "pos @0": (517).to_bytes(8, "little") + bytes(8)}
REPS = ("s-circuit.rep0.bin", "s-circuit.rep1.bin")


def _received(found: dict[str, list[str]]) -> dict:
    files = {f: {"bytes": 1, "found": {k: 1 for k in found.get(f, [])}} for f in (*REPS, "s-session.json")}
    return {"files": files, "verifier_found": sorted({k for ks in found.values() for k in ks})}


def test_classify():
    assert Z.classify(_received({}), BLOBS) == ([], {})
    assert Z.classify(_received({"s-session.json": ["salt @0"]}), BLOBS) == (["salt @0"], {})
    assert Z.classify(_received({REPS[1]: ["pos @0"], "s-session.json": ["pos @0"]}), BLOBS) == ([], {"pos @0": 3})
    assert Z.classify(_received({REPS[0]: ["pos @0"], REPS[1]: ["pos @0"]}), BLOBS) == (["pos @0"], {})
