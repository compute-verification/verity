"""The program decoder (`Flock.Program.decode`, `../../verity/circuits/difftest/ProgramDifftest.lean`) against its Python mirror
`codec.decode_program`, on `tests/verity/circuits/decode_difftest.py`'s cases: the format's descriptor vectors, each of the
format test's refusals of the descriptor vector (`E_SCHEMA` and the Boolean map's `E_BODY_*` among them), a `boolean` key
named twice, the GEMM chain with the catalog's Boolean bodies, and `tests/verity/circuits/test_codec_refusals.py`'s descriptors
(each the smallest that trips one refusal of the Python decoder; three it takes against its registry, Lean's decoder having none, `LEAN_TAKES`). Both answer an accept with the program's digest and its
Boolean map, and a refusal with its first `E_` code, which must be the same code. Lean's encoder
(`Flock.Program.Prog.encode`) of what Lean decodes is held to `codec.descriptor_chunks` of what the codec decodes, byte for
byte, and Lean's builder (`Flock.Program.Build`) to `Program` on the word equality `I32Eq` and on the product `Gemm_v2{K=32, N=2}`."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.circuits import codec, refs

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


def codec_encoding(text: str, registry) -> bytes | Exception:
    """`codec.descriptor_chunks` of the Program `codec.decode_program` reads from ``text``, or its refusal."""
    try:
        program = codec.decode_program(json.loads(text), registry)
    except Exception as e:  # noqa: BLE001
        return e
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
        assert lean["message"] == TEXTS.get(name), REWRITE
        return
    encoded = lean["encoded"].encode()
    python = codec_encoding(text, REGISTRY[text])
    if name not in LEAN_TAKES:
        assert isinstance(python, bytes), f"the codec no longer decodes an accepted case, so no bytes were compared: {python!r}"
    if isinstance(python, bytes):
        assert encoded == python
    again = json.loads(server.call("decode", {"descriptor": lean["encoded"]})[1])
    assert again == {"ok": True, "program_digest": hashlib.sha512(encoded).hexdigest(), "boolean": decoded["boolean"]}
    assert json.loads(server.call("encode", {"descriptor": lean["encoded"]})[1]) == lean


def test_encode_cases_hold_the_writers_form():
    """The cases the encoder is held to include descriptors in the writer's form, which it must write back as themselves."""
    canonical = {n for n, t, r in CASES if codec_encoding(t, r) == t.encode()}
    assert {"format", "wide", "registered_program", "chain"} <= canonical, canonical


_V, _A0 = ["Value", 1], ["Array", 0, ["Value", 1]]
#: Empty parts at a position, as `Affine.slice` past a value's end makes them, and an empty strided part: the decoders read
#: them, and the format's one form of an empty part is `{"explicit": []}`.
EMPTY_AT = codec.canonical_json({
    "schema": "verity-ir/descriptor/v2", "boolean": {}, "root": "Root[P_v1]_v1", "program_fn": "P_v1",
    "callees": ["Input1_v1", "E_v1"], "types": [["Array", 4, _V], _A0],
    "definitions": {
        "Input1_v1": {"form": "primitive", "id": "Input1_v1", "params": [], "ret": _V},
        "E_v1": {"form": "composite", "id": "E_v1", "bindings": {}, "body": {
            "params": [["x", _A0]], "ret_type": _A0, "nodes": [], "ret": {"strided": ["p", 0, 0, 1, 1, 1, 0]}}},
        "Root[P_v1]_v1": {"form": "composite", "id": "Root[P_v1]_v1", "bindings": {}, "body": {
            "params": [], "ret_type": _A0,
            "nodes": [{"form": "batch", "fn": 0, "args": [], "out": 0, "axes": [], "n": 4},
                      {"form": "call", "fn": 1, "args": [{"type": 1, "refs": {"affine": ["n", 0, 3, 0]}}], "out": 1}],
            "ret": {"affine": ["n", 1, 0, 0]}}}}}).decode()


def test_encode_writes_an_empty_part_as_the_empty_explicit(server):
    encoded = json.loads(server.call("encode", {"descriptor": EMPTY_AT})[1])["encoded"]
    written = EMPTY_AT
    for empty in ('{"strided":["p",0,0,1,1,1,0]}', '{"affine":["n",0,3,0]}', '{"affine":["n",1,0,0]}'):
        written = written.replace(empty, '{"explicit":[]}')
    assert encoded == written != EMPTY_AT
    assert json.loads(server.call("encode", {"descriptor": encoded})[1])["encoded"] == encoded


@pytest.mark.xfail(strict=True, reason="waits for Python's `Affine.slice` to return the empty explicit, which lands after the "
                   "amended decoders, which refuse an affine or strided part of count 0")
def test_codec_writes_an_empty_part_as_the_encoder_does():
    assert refs.Affine("n", 0, 3, 1).slice(1, 1).to_json() == {"explicit": []}


def with_ports(text: str) -> str:
    """``text`` with `ports` (§12) on its first primitive, in canonical form."""
    d = json.loads(text)
    key = next(k for k, e in sorted(d["definitions"].items()) if e["form"] == "primitive")
    d["definitions"][key]["ports"] = {"ret": [[0, 1, {"interval": [0, 1]}]]}
    return codec.canonical_json(d).decode()


def test_lean_decodes_a_description_with_ports(server):
    text = with_ports(next(t for n, t, _ in CASES if n == "format"))
    assert json.loads(server.call("decode", {"descriptor": text})[1])["ok"] is True


@pytest.mark.xfail(strict=True, reason="`Prog` keeps no ports, so `Prog.encode` writes none: they wait for `Defn.ports`, after "
                   "the landing")
def test_encode_writes_a_definitions_ports(server):
    text = with_ports(next(t for n, t, _ in CASES if n == "format"))
    assert json.loads(server.call("encode", {"descriptor": text})[1])["encoded"] == text


def test_cases_reach_accepts_and_every_body_refusal():
    """The cases hold accepts with a non-empty map, and each of the Boolean map's refusals and `E_SCHEMA`."""
    seen = [reference({"descriptor": t}) for _, t, _ in CASES]
    assert any(v["ok"] and v["boolean"] for v in seen)
    codes = {v.get("error") for v in seen if not v["ok"]}
    assert {"E_SCHEMA", "E_BODY_KEY", "E_BODY_MISSING", "E_BODY_NOT_BOOLEAN"} <= codes, codes


def lean_built(server, fn: str, value: dict) -> dict:
    answer = json.loads(server.call(fn, value)[1])
    assert answer["ok"], answer.get("message")
    return answer


def test_builder_writes_i32eqs_bytes(server):
    """`Build.eqBits 5`, the builder's word equality on 32 bits, is `Program(I32Eq)`: 159 gates, built from the Boolean primitives."""
    from verity.circuits.ops.boolean import scalar as BS
    from verity.circuits.program import Program
    text = b"".join(codec.descriptor_chunks(Program(BS.I32Eq))).decode()
    d = json.loads(text)
    lean = lean_built(server, "build_eq", {"k": 5, "root": d["root"], "program_fn": d["program_fn"]})
    assert lean["encoded"] == text


@pytest.mark.parametrize(("k", "n"), [(32, 2)])
def test_builder_writes_products_bytes(server, k, n):
    """The benchmark's product `Gemm_v2{K, N}`, built in Lean with its primitives and their Boolean Definitions adopted from the GEMM
    chain's Program (`Gemm_v2{K=16, N=16}`), is `Program(bind(Gemm, K, N))` byte for byte."""
    from verity.circuits.defs import bind
    from verity.circuits.ops import gemm as G
    from verity.circuits.program import Program
    text = b"".join(codec.descriptor_chunks(Program(bind(G.Gemm, K=k, N=n, **G.GEMM_AMPERE)))).decode()
    lean = lean_built(server, "build_product", {"registry": C.chain(), "K": k, "N": n})
    assert lean["encoded"] == text
    assert lean["program_digest"] == hashlib.sha512(text.encode()).hexdigest()
