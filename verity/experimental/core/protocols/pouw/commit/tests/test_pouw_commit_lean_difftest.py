"""The difftest of PoUW's commitments against their Lean port (`PouwCommit`): the Python half.

`tests/difftest_vectors.py` writes generated cases and this checkout's outcome on each (`tagged`, the bindings and
`weights_root`, domain ids, `commit_rows` and transcript roots, `verify_opening`, `coin_reason`). `check` runs the Lean half
in its Lean audit (`runs` in `lean-audit.json`): `PouwCommit/DifftestMain.lean` computes every case from the port and fails
on any difference. The audit checks the data against the record's SHA-256, and its pass is cached on the Lean package's
files alone, so the test here is what says the data is still what the Python computes: a change to the protocol's hashing
changes the data, fails this test, and `check.py --update` records the new data, which makes `check` run the Lean half on it.
The tests here also check that the cases reach every refusal, and that Lean differs from Python only by refusing.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from verity.experimental.protocols.pouw.protocol import tagged

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "tests" / "difftest_vectors.py"
_spec = importlib.util.spec_from_file_location("difftest_vectors", SCRIPT)
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((PACKAGE / "lean-audit.json").read_text())["runs"]["PouwCommit.DifftestMain"]


@functools.cache
def cases() -> tuple[dict, ...]:
    return tuple(vectors.generate())


def _of(kind: str) -> list[dict]:
    return [c for c in cases() if c["kind"] == kind]


def test_the_data_is_what_the_audit_pins(tmp_path):
    """Every case and the Python's outcome on it, byte for byte as the record pins them."""
    path = vectors.write(tmp_path)
    assert {"difftest/cases.jsonl": hashlib.sha256(path.read_bytes()).hexdigest()} == RUN["data"]
    assert RUN["args"] == ["{difftest/cases.jsonl}", str(len(cases()))]
    assert len(cases()) == sum(1 for _ in path.open())


def test_the_coin_cases_reach_every_refusal():
    outcomes = [c["python"] for c in _of("coins")]
    reasons = {o["reason"] for o in outcomes if "reason" in o}
    for start in ("coin is epoch ", "coin is the epoch's ", "coin opening is malformed",
                  "coin does not open against the epoch's coin commitment",
                  "coin was not opened after what the verifier recorded"):
        assert any(r.startswith(start) for r in reasons), start
    assert any(r.startswith("coin is the epoch's '\\x") for r in reasons)
    assert any(r.startswith("coin is the epoch's \"") for r in reasons)
    assert {"ok": None} in outcomes
    errors = collections.Counter(o["error"] for o in outcomes if "error" in o)
    assert {"DomainError", "KeyError", "ValueError", "TypeError", "ProtocolError"} <= set(errors)


def test_the_tree_cases_reach_every_branch():
    commits = _of("commit_rows")
    for h in ("sha256", "sha512", "hm96-sha512"):
        for d in (False, True):
            assert any("ok" in c["python"] and c["in"]["hash"] == h and c["in"]["digests"] == d and c["in"]["rows"]
                       for c in commits), (h, d)
    assert any("ok" in c["python"] and not c["in"]["rows"] for c in commits)
    assert any("ok" in c["python"] and c["in"]["hash"] == "hm96-sha512" and not c["in"]["digests"]
               and any(len(r) > 2 * 2048 for r in c["in"]["rows"]) for c in commits), "a row of three segments"
    assert any("error" in c["python"] for c in commits)
    verdicts = collections.Counter(json.dumps(c["python"]) for c in _of("verify"))
    assert {'{"ok": true}', '{"ok": false}', '{"error": "ProtocolError"}'} <= set(verdicts)
    assert any(c["in"]["hash"] == "hm96-sha512" and c["python"] == {"ok": True} for c in _of("verify"))
    assert {c["in"]["hash"] for c in _of("domain") if "ok" in c["python"]} == {"sha256", "sha512"}
    assert any("error" in c["python"] for c in _of("domain"))
    assert any("ok" in c["python"] for c in _of("transcript"))


def test_lean_differs_only_by_refusing():
    """Where a case says Lean's outcome is not Python's, Lean refuses it, and the case says why."""
    named = [c for c in cases() if "lean" in c]
    assert named
    assert all("error" in c["lean"] and c["why"] for c in named)
    assert any(c["python"] == {"ok": None} for c in named)


def test_python_refuses_what_the_cases_never_hold():
    """Lean's integer parts are ℕ, so the cases hold no negative integer, and Python refuses each."""
    with pytest.raises(TypeError):
        tagged("t", -1)
    with pytest.raises(TypeError):
        tagged("t", True)
