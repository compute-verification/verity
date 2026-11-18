"""The program decoder (`Flock.Program.decode`, `../../verity/circuits/difftest/ProgramDifftest.lean`) against its Python mirror
`codec.decode_program`, on `tests/verity/circuits/decode_difftest.py`'s cases: the format's descriptor vectors, each of the
format test's refusals of the descriptor vector (`E_SCHEMA` and the Boolean map's `E_BODY_*` among them), a `boolean` key
named twice, and the GEMM chain with the catalog's Boolean bodies. Both answer an accept with the program's digest and its
Boolean map, and a refusal with its first `E_` code, which must be the same code."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D

REPO = Path(__file__).resolve().parents[3]
LEAN = REPO / "verity" / "circuits" / "difftest"
sys.path.insert(0, str(REPO / "tests" / "verity" / "circuits"))
import decode_difftest as C  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

CASES = C.cases()
REGISTRY = {text: registry for _, text, registry in CASES}


def reference(value: dict) -> dict:
    v = C.verdict(value["descriptor"], REGISTRY[value["descriptor"]])
    v.pop("message", None)
    return v


DECODE = D.Port("program-difftest", "decode", reference, st.sampled_from([{"descriptor": t} for _, t, _ in CASES]),
                package=LEAN, need_rejects=False)
server = D.server_fixture(DECODE)


@pytest.mark.parametrize("name", [n for n, _, _ in CASES])
def test_decode_agrees(server, name):
    text = next(t for n, t, _ in CASES if n == name)
    assert D.replay(DECODE, {"descriptor": text}, server)[0] == "output"


def test_cases_reach_accepts_and_every_body_refusal():
    """The cases hold accepts with a non-empty map, and each of the Boolean map's refusals and `E_SCHEMA`."""
    seen = [reference({"descriptor": t}) for _, t, _ in CASES]
    assert any(v["ok"] and v["boolean"] for v in seen)
    codes = {v.get("error") for v in seen if not v["ok"]}
    assert {"E_SCHEMA", "E_BODY_KEY", "E_BODY_MISSING", "E_BODY_NOT_BOOLEAN"} <= codes, codes
