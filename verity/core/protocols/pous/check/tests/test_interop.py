"""`interop.py --all`: `wire.py` and `pous_check` agree byte for byte on every vector of `p2/PROTOCOL.md` §7, the outputs
cover acceptance and each failure code the vectors hold, and a difference in one byte of one side is reported."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

spec = importlib.util.spec_from_file_location("pous_interop", Path(__file__).resolve().parents[1] / "interop.py")
interop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interop)


@pytest.fixture(scope="module")
def compared():
    return interop.compare()


@pytest.mark.slow
def test_every_vector_agrees(compared, capsys):
    results, expected = compared
    assert interop.report(results, expected) == 0
    assert capsys.readouterr().out.endswith(f"{expected} transcripts, 0 mismatches\n")
    assert expected >= 31 + 1 + 2 + 4 + 2
    audits = [json.loads(r["python"])["audit"] for r in results if r["kind"] == "transcript"]
    assert {a["code"] for a in audits} >= {0, 1, 2, 3, 4, 6}
    assert sum(a["verdict"] for a in audits) == 4


@pytest.mark.slow
def test_a_difference_is_reported(compared, capsys):
    results, expected = compared
    final = json.loads(results[0]["lean"])
    last = final["audit"]["final"][-1]
    final["audit"]["final"] = final["audit"]["final"][:-1] + ("1" if last == "0" else "0")
    tampered = [results[0] | {"lean": interop.canonical(final)}] + results[1:]
    assert interop.report(tampered, expected) == 1
    out = capsys.readouterr().out
    assert f"MISMATCH transcript {results[0]['id']}" in out and out.endswith(f"{expected} transcripts, 1 mismatches\n")
    assert interop.report(results[1:], expected) == 1


def test_the_red_team_sessions_agree_word_for_word():
    """`difftest_vectors.redteam_audits` (a store whose payloads are not all below p, 2^79 blocks, a beacon round named
    twice, manifest floats), held to `wire.py` with every message, which the difftest's codes alone don't compare."""
    gen = interop._generator()
    py = gen._python(interop.ROOT)
    insts = gen.redteam_audits(py)
    with interop.Server([str(interop.build(interop.HERE, "pous-audit-difftest")), "serve"]) as lean:
        results = [{"id": i["name"], "kind": "transcript", "python": interop._python_side(interop.transcript, gen, py[1], i),
                    "lean": interop._lean_side(lean, "transcript", i)} for i in insts]
    assert interop.report(results, len(insts)) == 0
    setups = [json.loads(r["python"])["setup"] for r in results]
    assert sum(s.get("message") == "no P2 prime: every payload is below p" for s in setups) == 8
    assert sum(s.get("message") == "manifest holds a value canonical JSON cannot" for s in setups) == 5
    assert sum(s.get("message") == "setup block count disagrees with manifest" for s in setups) == 3
