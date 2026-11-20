"""The program decoder (`Flock.Program.decode`, `../../verity/circuits/difftest/ProgramDifftest.lean`) against its Python mirror
`codec.decode_program`, on `tests/verity/circuits/decode_difftest.py`'s cases: the format's descriptor vectors, each of the
format test's refusals of the descriptor vector (`E_SCHEMA` and the Boolean map's `E_BODY_*` among them), a `boolean` key
named twice, the GEMM chain with the catalog's Boolean bodies, and `tests/verity/circuits/test_codec_refusals.py`'s descriptors
(each the smallest that trips one refusal of the Python decoder; three it takes against its registry, Lean's decoder having none, `LEAN_TAKES`). Both answer an accept with the program's digest and its
Boolean map, and a refusal with its first `E_` code, which must be the same code. Lean's encoder
(`Flock.Program.Prog.encode`) of what Lean decodes is held to `codec.descriptor_chunks` of what the codec decodes, byte for
byte."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.circuits import codec

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


#: What Lean refuses and the Python decoder accepts: a part of references past `Widths.WALK` = 2^20 whose widths it would compare leaf by
#: leaf, which Lean's decoder refuses by name (`UNCHECKED`) and the codec decides by arithmetic.
LEAN_REFUSES = ("lean refusal: a grid read in leaves of two widths past what is walked",)


@pytest.mark.parametrize("name", [n for n, _, _ in CASES if n not in LEAN_TAKES + LEAN_REFUSES])
def test_decode_agrees(server, name):
    text = next(t for n, t, _ in CASES if n == name)
    assert D.replay(DECODE, {"descriptor": text}, server)[0] == "output"


@pytest.mark.parametrize("name", [n for n, _, _ in CASES])
def test_leans_refusal_text_is_the_pinned_one(server, name):
    text = next(t for n, t, _ in CASES if n == name)
    verdict = json.loads(server.call("decode_text", {"descriptor": text})[1])
    assert verdict.get("message") == TEXTS.get(name), REWRITE


@pytest.mark.parametrize("name", [n for n, _, _ in CASES])
def test_decode_with_is_decode(server, name):
    """`Program.decodeWith` at SHA-512 (what the verifier's setups call) gives `decode`'s digest and Boolean map, or its text."""
    text = next(t for n, t, _ in CASES if n == name)
    with_ = json.loads(server.call("decode_with", {"descriptor": text})[1])
    plain = json.loads(server.call("decode", {"descriptor": text})[1])
    refused = json.loads(server.call("decode_text", {"descriptor": text})[1])
    assert with_["ok"] == plain["ok"] == refused["ok"]
    if with_["ok"]:
        assert with_ == plain
    else:
        assert with_["message"] == refused["message"]


@pytest.mark.parametrize("name", LEAN_REFUSES)
def test_lean_refuses_what_the_codec_decides(server, name):
    text = next(t for n, t, _ in CASES if n == name)
    assert reference({"descriptor": text})["ok"]
    assert json.loads(server.call("decode", {"descriptor": text})[1])["ok"] is False


@pytest.mark.parametrize("name", LEAN_TAKES)
def test_lean_takes_what_the_registry_refuses(server, name):
    text = next(t for n, t, _ in CASES if n == name)
    assert not reference({"descriptor": text})["ok"]
    assert json.loads(server.call("decode", {"descriptor": text})[1])["ok"] is True


def codec_encoding(text: str, registry) -> bytes | None:
    """`codec.descriptor_chunks` of the Program `codec.decode_program` reads from ``text``, or None where it refuses."""
    try:
        program = codec.decode_program(json.loads(text), registry)
    except Exception:  # noqa: BLE001
        return None
    return b"".join(codec.descriptor_chunks(program))


@pytest.mark.parametrize("name", [n for n, _, _ in CASES])
def test_encode_writes_the_codecs_bytes(server, name):
    """Lean's `Program.encode` of what Lean's decoder accepts: byte for byte the codec's encoding of what the codec decodes,
    so a descriptor in the writer's form is written back as itself; an encoding Lean decodes to its own digest and writes
    back unchanged."""
    text = next(t for n, t, _ in CASES if n == name)
    lean = json.loads(server.call("encode", {"descriptor": text})[1])
    decoded = json.loads(server.call("decode", {"descriptor": text})[1])
    assert lean["ok"] == decoded["ok"], lean.get("message")
    if not lean["ok"]:
        return
    encoded = lean["encoded"].encode()
    python = codec_encoding(text, REGISTRY[text])
    if python is not None:
        assert encoded == python
    again = json.loads(server.call("decode", {"descriptor": lean["encoded"]})[1])
    assert again == {"ok": True, "program_digest": hashlib.sha512(encoded).hexdigest(), "boolean": decoded["boolean"]}
    assert json.loads(server.call("encode", {"descriptor": lean["encoded"]})[1]) == lean


def test_encode_cases_hold_the_writers_form():
    """The cases the encoder is held to include descriptors in the writer's form, which it must write back as themselves."""
    canonical = {n for n, t, r in CASES if codec_encoding(t, r) == t.encode()}
    assert {"format", "wide", "registered_program", "chain"} <= canonical, canonical


_V, _A0 = ["Value", 1], ["Array", 0, ["Value", 1]]
#: A writer's descriptor with empty parts at a position (`Affine.slice` past a value's end): Lean's `Part` holds an empty part's
#: space alone, so its encoder writes `["n",0,0,0]` where the codec keeps `["n",0,3,0]`.
EMPTY_AT = codec.canonical_json({
    "schema": "verity-ir/descriptor/v2", "boolean": {}, "root": "Root[P_v1]_v1", "program_fn": "P_v1",
    "callees": ["Input1_v1", "E_v1"], "types": [["Array", 4, _V], _A0],
    "definitions": {
        "Input1_v1": {"form": "primitive", "id": "Input1_v1", "params": [], "ret": _V},
        "E_v1": {"form": "composite", "id": "E_v1", "bindings": {}, "body": {
            "params": [["x", _A0]], "ret_type": _A0, "nodes": [], "ret": {"affine": ["p", 0, 0, 0]}}},
        "Root[P_v1]_v1": {"form": "composite", "id": "Root[P_v1]_v1", "bindings": {}, "body": {
            "params": [], "ret_type": _A0,
            "nodes": [{"form": "batch", "fn": 0, "args": [], "out": 0, "axes": [], "n": 4},
                      {"form": "call", "fn": 1, "args": [{"type": 1, "refs": {"affine": ["n", 0, 3, 0]}}], "out": 1}],
            "ret": {"affine": ["n", 1, 0, 0]}}}}}).decode()


def test_codec_writes_an_empty_parts_position():
    assert codec_encoding(EMPTY_AT, REGISTRY[next(t for n, t, _ in CASES if n == "format")]) == EMPTY_AT.encode()


@pytest.mark.xfail(strict=True, reason="Lean's decoder drops an empty part's index and base, which the codec writes")
def test_encode_writes_an_empty_parts_position(server):
    assert json.loads(server.call("encode", {"descriptor": EMPTY_AT})[1])["encoded"] == EMPTY_AT


def test_cases_reach_accepts_and_every_body_refusal():
    """The cases hold accepts with a non-empty map, and each of the Boolean map's refusals and `E_SCHEMA`."""
    seen = [reference({"descriptor": t}) for _, t, _ in CASES]
    assert any(v["ok"] and v["boolean"] for v in seen)
    codes = {v.get("error") for v in seen if not v["ok"]}
    assert {"E_SCHEMA", "E_BODY_KEY", "E_BODY_MISSING", "E_BODY_NOT_BOOLEAN"} <= codes, codes
