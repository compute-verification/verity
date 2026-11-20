"""The program decoder (`Flock.Program.decode`, `../../verity/circuits/difftest/ProgramDifftest.lean`) against its Python mirror
`codec.decode_program`, on `tests/verity/circuits/decode_difftest.py`'s cases: the format's descriptor vectors, each of the
format test's refusals of the descriptor vector (`E_SCHEMA` and the Boolean map's `E_BODY_*` among them), a `boolean` key
named twice, the GEMM chain with the catalog's Boolean bodies, and `tests/verity/circuits/test_codec_refusals.py`'s descriptors
(each the smallest that trips one refusal of the Python decoder; three it takes against its registry, Lean's decoder having none, `LEAN_TAKES`). Both answer an accept with the program's digest and its
Boolean map, and a refusal with its first `E_` code, which must be the same code."""

from __future__ import annotations

import json
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
TEXTS = json.loads((Path(__file__).parent / "lean_decode_texts.json").read_text())
REWRITE = "Lean's refusal differs from the pinned text; a change meant to move it rewrites `lean_decode_texts.json` with `write_lean_texts.py`"
REGISTRY = {text: registry for _, text, registry in CASES}


def reference(value: dict) -> dict:
    v = C.verdict(value["descriptor"], REGISTRY[value["descriptor"]])
    v.pop("message", None)
    return v


DECODE = D.Port("program-difftest", "decode", reference, st.sampled_from([{"descriptor": t} for _, t, _ in CASES]),
                package=LEAN, need_rejects=False)
server = D.server_fixture(DECODE)


#: The refusals the Python decoder makes against its registry (`codec.decode_program`'s `registry`), which the Lean decoder, reading
#: the descriptor alone, takes: the primitive's declared ports, a binding record's keys and the registry composite's `refuse`.
LEAN_TAKES = ("refusal: a primitive with ports the registry lacks", "refusal: a binding the registry's composite has no static for",
              "refusal: a binding the registry's composite refuses")


@pytest.mark.parametrize("name", [n for n, _, _ in CASES if n not in LEAN_TAKES])
def test_decode_agrees(server, name):
    text = next(t for n, t, _ in CASES if n == name)
    assert D.replay(DECODE, {"descriptor": text}, server)[0] == "output"


@pytest.mark.parametrize("name", [n for n, _, _ in CASES])
def test_leans_refusal_text_is_the_pinned_one(server, name):
    text = next(t for n, t, _ in CASES if n == name)
    verdict = json.loads(server.call("decode_text", {"descriptor": text})[1])
    assert verdict.get("message") == TEXTS.get(name), REWRITE


@pytest.mark.parametrize("name", LEAN_TAKES)
def test_lean_takes_what_the_registry_refuses(server, name):
    text = next(t for n, t, _ in CASES if n == name)
    assert not reference({"descriptor": text})["ok"]
    assert json.loads(server.call("decode", {"descriptor": text})[1])["ok"] is True


def test_cases_reach_accepts_and_every_body_refusal():
    """The cases hold accepts with a non-empty map, and each of the Boolean map's refusals and `E_SCHEMA`."""
    seen = [reference({"descriptor": t}) for _, t, _ in CASES]
    assert any(v["ok"] and v["boolean"] for v in seen)
    codes = {v.get("error") for v in seen if not v["ok"]}
    assert {"E_SCHEMA", "E_BODY_KEY", "E_BODY_MISSING", "E_BODY_NOT_BOOLEAN"} <= codes, codes
