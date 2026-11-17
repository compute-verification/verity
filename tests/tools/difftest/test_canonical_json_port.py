"""`Core.canonicalJson` under each JSON parser (`core-canonical-json-difftest`'s `canonical-parsers`, the interface's
`CoreProtocol/CanonicalJsonDifftest.lean`) against `canonical_answer`, its Python mirror: the parser's mirror
(`test_flock_json_port.parse`), then `canon`, `Flock.canon` on the value read. The port answers what the function
answers on the same bytes when their text is read with Lean core's parser and when it is read with the total one
(`Flock.JsonParse.parse`, which it reads with). Both answers are held to the one mirror, so on every text sampled the
two parsers give `canonicalJson` one answer: the same value, or the same refusal with the same text.

The samples are canonical texts (`HONEST`: values of the whole domain, and objects shaped as a protocol's parameters
are) and what one change makes of one (`BROKEN`): a character cut, inserted or replaced, the text cut short, the same
value written not canonically (spaces, keys out of order, characters outside ASCII left as they are), a member that no
canonical text holds (a key twice, a number with a fraction or an exponent, an escape `canon` doesn't write), and
bytes that are no text."""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import find, given, settings
from hypothesis import strategies as st
from test_flock_json_port import TEXTS, parse

import verity_difftest as D
from verity_difftest import protocol as P

INTERFACE = Path(__file__).resolve().parents[3] / "verity" / "protocols" / "interface"
lean = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

NOT_ASCII = "not ASCII without DEL (a byte at or above 0x7f)"
NOT_WHOLE = "canonical JSON: a non-integer number"
NOT_CANONICAL = "not canonical JSON (compact, keys sorted, integers only)"
LOWER_HEX = re.compile(r"(?:[0-9a-f]{2})*")


class NotWhole(Exception):
    """`Flock.canon`'s one refusal: a number with an exponent."""


def canon(v: list) -> str:
    """`Flock.canon` on a value as the port encodes it (`FlockJsonDifftest.enc`): an object's members are in key order
    there, as in the tree `canon` walks."""
    kind = v[0]
    if kind == "null":
        return "null"
    if kind == "bool":
        return "true" if v[1] else "false"
    if kind == "num":
        if v[2] != 0:
            raise NotWhole
        return str(v[1])
    if kind == "str":
        return json.dumps(v[1])
    if kind == "arr":
        return "[" + ",".join(canon(x) for x in v[1]) + "]"
    return "{" + ",".join(json.dumps(k) + ":" + canon(x) for k, x in v[1]) + "}"


def canonical_answer(b: bytes) -> list:
    """`Core.canonicalJson b` as the port answers it: `["ok", value]`, or `["error", why]` with the refusal's text."""
    if any(x >= 0x7f for x in b):
        return ["error", NOT_ASCII]
    s = b.decode("ascii")
    read = parse(s)
    if read[0] == "error":
        return ["error", f"not JSON ({read[1]})"]
    try:
        same = canon(read[1]) == s
    except NotWhole:
        return ["error", NOT_WHOLE]
    return read if same else ["error", NOT_CANONICAL]


def reference(case):
    if not isinstance(case, dict) or not isinstance(case.get("text"), str) or not LOWER_HEX.fullmatch(case["text"]):
        raise D.Reject("text: not bytes in lowercase hex")
    answer = canonical_answer(bytes.fromhex(case["text"]))
    return {"core": answer, "total": answer}


def value_of(v: list):
    """The Python value of a value as the port encodes it, whole numbers only."""
    kind = v[0]
    if kind == "null":
        return None
    if kind == "arr":
        return [value_of(x) for x in v[1]]
    if kind == "obj":
        return {k: value_of(x) for k, x in v[1]}
    assert kind != "num" or v[2] == 0, v
    return v[1]


#: Characters `canon` escapes or writes as they are, in keys and strings: a quote, a backslash, a slash, a control
#: character, DEL, one outside ASCII and one above U+FFFF.
CHARS = list('ab"\\/ \n\t\x01\x7f') + ["é", "€", "😀"]
KEYS = st.text(alphabet=st.sampled_from(CHARS), max_size=3)
DIGESTS = st.binary(min_size=1, max_size=64).map(bytes.hex)
INTS = st.one_of(st.integers(-3, 3), st.integers(-2 ** 70, 2 ** 70), st.sampled_from([2 ** 64 - 1, 2 ** 64, -2 ** 63]))
VALUES = st.recursive(st.none() | st.booleans() | INTS | KEYS | DIGESTS | st.text(max_size=4),
                      lambda inner: st.lists(inner, max_size=4) | st.dictionaries(KEYS, inner, max_size=4), max_leaves=10)
RECORDS = st.fixed_dictionaries({"digest": DIGESTS, "domain": st.sampled_from(["x", "weights", "é"]),
                                 "leaves": st.integers(0, 2 ** 40)})
#: An object shaped as a protocol's parameters, commitment or evidence are: named fields holding digests in hex,
#: naturals, a ratio as a string, null, and lists and objects of such records.
SHAPED = st.fixed_dictionaries({
    "delta": st.sampled_from(["1/1000000", "1/2"]), "k": st.integers(0, 64), "log": st.lists(RECORDS, max_size=3),
    "partition": st.none() | st.fixed_dictionaries({"program": DIGESTS, "query": st.dictionaries(KEYS, INTS, max_size=3)}),
    "ports": st.dictionaries(st.sampled_from(["w", "x", "y"]), RECORDS, max_size=3), "tied": st.booleans()})
DOMAIN = st.one_of(VALUES, SHAPED)
HONEST = DOMAIN.map(lambda v: D.canonical(v).encode())


def _reversed(v):
    """`v` with every object's keys in the reverse of their order."""
    if isinstance(v, dict):
        return {k: _reversed(v[k]) for k in sorted(v, reverse=True)}
    return [_reversed(x) for x in v] if isinstance(v, list) else v


#: Members no canonical text holds: numbers `canon` writes otherwise or refuses, escapes it doesn't write, half of a
#: surrogate pair, a key twice, space, and texts that are no JSON.
MEMBERS = ["1.0", "1e2", "1E+2", "10e-1", "-0", "0.0", "1.5", "01", '"\\u0041"', '"\\/"', '"\\u00E9"', '"\\ud83d"',
           '"\x01"', '{"a":1,"a":1}', '{"a":1,"a":2}', '{"b":1,"a":2}', "[1, 2]", " 1", "1 ", "1\n", "nul", "[", '{"a"}']
ALPHABET = list('{}[],:"\\/-+.eE019anu \n') + ["\x7f", "é"]


@st.composite
def _broken(draw) -> bytes:
    """One change to a canonical text, or to how its value is written."""
    v = draw(DOMAIN)
    t = D.canonical(v)
    how = draw(st.sampled_from(["cut", "insert", "replace", "truncate", "spaced", "reversed", "raw", "member", "beside"]))
    k = draw(st.integers(0, len(t)))
    c = draw(st.sampled_from(ALPHABET))
    if how == "cut":
        return (t[:k] + t[k + 1:]).encode()
    if how == "insert":
        return (t[:k] + c + t[k:]).encode()
    if how == "replace":
        return (t[:k] + c + t[k + 1:]).encode()
    if how == "truncate":
        return t[:k].encode()
    if how == "spaced":
        return json.dumps(v, sort_keys=True).encode()
    if how == "reversed":
        return json.dumps(_reversed(v), separators=(",", ":")).encode()
    if how == "raw":
        return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    member = draw(st.sampled_from(MEMBERS))
    return (f"[{member}]" if how == "member" else f'{{"a":{t},"b":{member}}}').encode()


BROKEN = st.one_of(_broken(), TEXTS.map(str.encode), st.binary(max_size=8))
#: A request the port refuses before it reads a text: no `text`, not a string, hex in uppercase or of odd length.
REFUSED = st.one_of(st.just({}), st.just({"text": 5}), st.just(5), HONEST.map(lambda b: {"text": b.hex().upper() + "AB"}),
                    HONEST.map(lambda b: {"text": b.hex() + "0"}))
CASES = st.one_of(HONEST.map(lambda b: {"text": b.hex()}), BROKEN.map(lambda b: {"text": b.hex()}), REFUSED)

PORT = D.Port("core-canonical-json-difftest", "canonical-parsers", reference, CASES, package=INTERFACE)
server = D.server_fixture(PORT)


@pytest.fixture(autouse=True)
def _big_ints():
    """The parser's port's texts (`TEXTS`) hold exponents of up to four digits, and `1e9999` reads as a mantissa of
    10,000 digits; Python refuses str() of an int past 4300 digits by default."""
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    yield
    sys.set_int_max_str_digits(old)


def test_reference_reads_examples():
    assert canonical_answer(b'{"a":[1,-2,"\\u00e9"],"b":null}') == [
        "ok", ["obj", [["a", ["arr", [["num", 1, 0], ["num", -2, 0], ["str", "é"]]]], ["b", ["null"]]]]]
    assert canonical_answer(b"18446744073709551616") == ["ok", ["num", 2 ** 64, 0]]
    assert canonical_answer(b'{"b":1,"a":2}') == ["error", NOT_CANONICAL]
    assert canonical_answer(b'{"a":1,"a":1}') == ["error", NOT_CANONICAL]
    assert canonical_answer(b"[1e2]") == ["error", NOT_CANONICAL]
    assert canonical_answer(b"[-0]") == ["error", NOT_CANONICAL]
    assert canonical_answer(b'"\\u0041"') == ["error", NOT_CANONICAL]
    assert canonical_answer(b"[1.0]") == ["error", NOT_WHOLE]
    assert canonical_answer(b"[1,]") == ["error", "not JSON (offset 3: unexpected input)"]
    assert canonical_answer(b"") == ["error", "not JSON (offset 0: unexpected end of input)"]
    assert canonical_answer('"é"'.encode()) == ["error", NOT_ASCII]
    assert canonical_answer(b'"\x7f"') == ["error", NOT_ASCII]
    assert canonical_answer(b"\xff") == ["error", NOT_ASCII]


def test_cases_reach_every_answer():
    """The samples reach a value and each of the four refusals, and a request refused before its text is read."""
    reached = {"ok": lambda a: a[0] == "ok", "ascii": lambda a: a == ["error", NOT_ASCII],
               "whole": lambda a: a == ["error", NOT_WHOLE], "canonical": lambda a: a == ["error", NOT_CANONICAL],
               "json": lambda a: a[0] == "error" and a[1].startswith("not JSON (offset ")}
    for name, is_it in reached.items():
        find(BROKEN if name != "ok" else HONEST, lambda b, is_it=is_it: is_it(canonical_answer(b)),
             settings=settings(database=None, derandomize=True, max_examples=2000))
    with pytest.raises(D.Reject):
        reference(find(REFUSED, lambda _: True, settings=settings(database=None, derandomize=True)))


@settings(database=None, derandomize=True, deadline=None, max_examples=300)
@given(DOMAIN)
def test_a_canonical_text_reads_back_to_its_value(v):
    """The mirror reads every canonical text, and to the value it was written from."""
    answer = canonical_answer(D.canonical(v).encode())
    assert answer[0] == "ok" and value_of(answer[1]) == v, (v, answer)


@settings(database=None, derandomize=True, deadline=None, max_examples=500)
@given(st.one_of(HONEST, BROKEN))
def test_the_mirror_accepts_what_the_protocols_reference_accepts(b):
    """`canonical_answer` and `verity_difftest.protocol.canonical_json`, which a protocol's Python reference reads its
    parameters with, accept the same texts and read each to the same value."""
    answer = canonical_answer(b)
    try:
        v = P.canonical_json(b)
    except D.Reject:
        assert answer[0] == "error", (b, answer)
    else:
        assert answer[0] == "ok" and value_of(answer[1]) == v, (b, answer, v)


@lean
def test_canonical_json_answers_alike_under_both_parsers(server):
    """`Core.canonicalJson` reads its text with the total parser, which a proof can read through. On every sampled
    text, canonical or broken, the function answers as it does reading with Lean core's parser, and as the mirror: the
    same value with nothing lost, or the same refusal. Then a sweep of each kind of text by name: every canonical text
    is read under both, and no broken one is read under one parser alone."""
    stats = D.run(PORT, examples=3000, server=server)
    kinds = Counter()

    def one(kind: str, b: bytes) -> None:
        out = json.loads(D.replay(PORT, {"text": b.hex()}, server)[1])
        assert out["core"] == out["total"], (kind, b, out)
        kinds[kind, out["total"][0]] += 1

    sweep = settings(database=None, derandomize=True, deadline=None, max_examples=200)

    @sweep
    @given(HONEST)
    def honest(b):
        one("honest", b)

    @sweep
    @given(BROKEN)
    def broken(b):
        one("broken", b)

    honest()
    broken()
    print(f"core-canonical-json-difftest canonical-parsers: {stats}; swept {dict(kinds)}")
    assert kinds["honest", "ok"] and not kinds["honest", "error"] and kinds["broken", "error"], kinds
