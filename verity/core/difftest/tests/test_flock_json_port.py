"""C-Flock's JSON parser (`Flock.JsonParse.parse`, `lean/FlockJsonDifftest.lean`) and Lean core's `Lean.Json.parse`, which
it reads as, against `parse`, their Python mirror: the same value, or the same refusal (`offset N: why`, at the same
UTF-8 byte offset). Both Lean functions are held to the one mirror, so the run also shows the two agree on every text
drawn. A refusal is an output (`["error", why]`), so its text is compared, not only the fact of it."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest
from hypothesis import find, settings
from hypothesis import strategies as st

import verity_difftest as D

LEAN = Path(__file__).resolve().parents[1] / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

EOF = "unexpected end of input"
USIZE = 2 ** 64
WS = "\t\n\r "


class Refuse(Exception):
    def __init__(self, o: int, why: str):
        super().__init__(o, why)
        self.o, self.why = o, why


def _size(c: str) -> int:
    return len(c.encode("utf-8"))


def _ws(s: str, i: int, o: int) -> tuple[int, int]:
    while i < len(s) and s[i] in WS:
        i, o = i + 1, o + 1
    return i, o


def _hex_char(s: str, i: int, o: int) -> tuple[int, int, int]:
    if i >= len(s):
        raise Refuse(o, EOF)
    c = s[i]
    if "0" <= c <= "9":
        return i + 1, o + 1, ord(c) - ord("0")
    if "a" <= c <= "f":
        return i + 1, o + 1, ord(c) - ord("a") + 10
    if "A" <= c <= "F":
        return i + 1, o + 1, ord(c) - ord("A") + 10
    raise Refuse(o + _size(c), "invalid hex character")


def _hex_n(n: int, s: str, i: int, o: int) -> tuple[int, int, int]:
    acc = 0
    for _ in range(n):
        i, o, d = _hex_char(s, i, o)
        acc = 16 * acc + d
    return i, o, acc


def _finish_pair(low: int, s: str, i: int, o: int) -> tuple[int, int, str] | None:
    """The low half of a surrogate pair, or None where Lean's `attempt` fails and falls back to U+FFFD."""
    if not (s[i:i + 2] == "\\u" and i + 2 < len(s) and s[i + 2] in "dD"):
        return None
    digits = s[i + 3:i + 6]
    if len(digits) < 3 or any(c not in "0123456789abcdefABCDEF" for c in digits):
        return None
    i, o, v = _hex_n(3, s, i + 3, o + 3)
    if v < 0xC00:
        return None
    cv = (low % 0x400) * 0x400 + v % 0x400 + 0x10000
    return (i, o, chr(cv)) if cv <= 0x10FFFF else None


ESCAPES = {"\\": "\\", '"': '"', "/": "/", "b": "\x08", "f": "\x0c", "n": "\n", "r": "\r", "t": "\t"}


def _escaped_char(s: str, i: int, o: int) -> tuple[int, int, str]:
    if i >= len(s):
        raise Refuse(o, EOF)
    c = s[i]
    if c in ESCAPES:
        return i + 1, o + 1, ESCAPES[c]
    if c != "u":
        raise Refuse(o + _size(c), "illegal \\u escape")
    i, o, v = _hex_n(4, s, i + 1, o + 1)
    if v < 0xD800 or v >= 0xE000:
        return i, o, chr(v)
    if v < 0xDC00:
        return _finish_pair(v, s, i, o) or (i, o, "\ufffd")
    return i, o, "\ufffd"


def _string(s: str, i: int, o: int) -> tuple[int, int, str]:
    """A string's characters after its opening quote, through its closing one."""
    acc = []
    while True:
        if i >= len(s):
            raise Refuse(o, EOF)
        c = s[i]
        if c == '"':
            return i + 1, o + 1, "".join(acc)
        if c == "\\":
            i, o, e = _escaped_char(s, i + 1, o + 1)
            acc.append(e)
        elif 0x20 <= ord(c) <= 0x10FFFF:
            acc.append(c)
            i, o = i + 1, o + _size(c)
        else:
            raise Refuse(o + _size(c), "unexpected character in string")


def _nat_after(first, why: str, s: str, i: int, o: int) -> tuple[int, int, int, int]:
    """A run of digits whose first passes `first`: their value and count."""
    if i >= len(s):
        raise Refuse(o, EOF)
    if not first(s[i]):
        raise Refuse(o, "expected " + why)
    acc = n = 0
    while i < len(s) and "0" <= s[i] <= "9":
        acc, n, i, o = 10 * acc + ord(s[i]) - ord("0"), n + 1, i + 1, o + 1
    return i, o, acc, n


def _digit(c: str) -> bool:
    return "0" <= c <= "9"


def _number(s: str, i: int, o: int) -> tuple[int, int, tuple[int, int]]:
    """A number as Lean's `JsonNumber`, (mantissa, exponent): the value is mantissa / 10^exponent."""
    sign = 1
    if i < len(s) and s[i] == "-":
        sign, i, o = -1, i + 1, o + 1
    if i >= len(s):
        raise Refuse(o, EOF)
    if s[i] == "0":
        whole, i, o = 0, i + 1, o + 1
    else:
        i, o, whole, _ = _nat_after(lambda c: "1" <= c <= "9", "1-9", s, i, o)
    if i < len(s) and s[i] == ".":
        i, o, n, d = _nat_after(_digit, "digit", s, i + 1, o + 1)
        if d > USIZE:
            raise Refuse(o, "too many decimals")
        m, e = sign * (whole * 10 ** d + n), d
    else:
        m, e = sign * whole, 0
    if i < len(s) and s[i] in "eE":
        if i + 1 >= len(s):
            raise Refuse(o + 1, EOF)
        if s[i + 1] == "-":
            i, o, k, _ = _nat_after(_digit, "0-9", s, i + 2, o + 2)
            return i, o, (m, e + k)
        i, o = (i + 2, o + 2) if s[i + 1] == "+" else (i + 1, o + 1)
        i, o, k, _ = _nat_after(_digit, "0-9", s, i, o)
        if k > USIZE:
            raise Refuse(o, "exp too large")
        return i, o, (m * 10 ** max(0, k - e), max(0, e - k))
    return i, o, (m, e)


def _word(w: str, j, s: str, i: int, o: int):
    if s[i:i + len(w)] != w:
        raise Refuse(o, "expected: " + w)
    i, o = _ws(s, i + len(w), o + len(w))
    return i, o, j


def _value(s: str, i: int, o: int):
    """One value and the whitespace after it, encoded as the port encodes it."""
    if i >= len(s):
        raise Refuse(o, EOF)
    c = s[i]
    if c in "[{":
        close = "]" if c == "[" else "}"
        i, o = _ws(s, i + 1, o + 1)
        if i >= len(s):
            raise Refuse(o, EOF)
        if s[i] == close:
            i, o = _ws(s, i + 1, o + 1)
            return i, o, ["arr", []] if c == "[" else ["obj", []]
        return _array(s, i, o) if c == "[" else _object(s, i, o)
    if c == '"':
        i, o, t = _string(s, i + 1, o + 1)
        i, o = _ws(s, i, o)
        return i, o, ["str", t]
    if c == "f":
        return _word("false", ["bool", False], s, i, o)
    if c == "t":
        return _word("true", ["bool", True], s, i, o)
    if c == "n":
        return _word("null", ["null"], s, i, o)
    if c == "-" or _digit(c):
        i, o, (m, e) = _number(s, i, o)
        i, o = _ws(s, i, o)
        return i, o, ["num", m, e]
    raise Refuse(o, "unexpected input")


def _array(s: str, i: int, o: int):
    acc = []
    while True:
        i, o, v = _value(s, i, o)
        acc.append(v)
        if i >= len(s):
            raise Refuse(o, EOF)
        c = s[i]
        i2, o2 = _ws(s, i + 1, o + 1)
        if c == "]":
            return i2, o2, ["arr", acc]
        if c != ",":
            raise Refuse(o + _size(c), "unexpected character in array")
        i, o = i2, o2


def _object(s: str, i: int, o: int):
    kvs = {}
    while True:
        if i >= len(s):
            raise Refuse(o, EOF)
        if s[i] != '"':
            raise Refuse(o, 'expected "')
        i, o, key = _string(s, i + 1, o + 1)
        i, o = _ws(s, i, o)
        if i >= len(s):
            raise Refuse(o, EOF)
        if s[i] != ":":
            raise Refuse(o, "expected :")
        i, o = _ws(s, i + 1, o + 1)
        i, o, v = _value(s, i, o)
        if i >= len(s):
            raise Refuse(o, EOF)
        kvs[key] = v
        c = s[i]
        i2, o2 = _ws(s, i + 1, o + 1)
        if c == "}":
            return i2, o2, ["obj", [[k, kvs[k]] for k in sorted(kvs)]]
        if c != ",":
            raise Refuse(o + _size(c), "unexpected character in object")
        i, o = i2, o2


def parse(text: str) -> list:
    """`Flock.JsonParse.parse text`, as the port answers it: `["ok", value]` or `["error", "offset N: why"]`."""
    i, o = _ws(text, 0, 0)
    try:
        i, o, v = _value(text, i, o)
    except Refuse as r:
        return ["error", f"offset {r.o}: {r.why}"]
    if i < len(text):
        return ["error", f"offset {o}: expected end of input"]
    return ["ok", v]


def reference(case):
    if not isinstance(case, dict) or not isinstance(case.get("text"), str):
        raise D.Reject("text: not a string")
    return parse(case["text"])


#: Characters a JSON text is made of, plus some it isn't: whitespace, structure, escapes, digits, letters of the literals,
#: a control character, a non-ASCII and an astral one. No surrogates: the text reaches Lean as a JSON string.
ALPHABET = list('\t\n\r {}[],:"\\/-+.eE0123456789abcdfnrtuDdl') + ["\x01", "\x1f", "é", "€", "😀"]
ATOMS = st.sampled_from(["null", "true", "false", "0", "-0", "12", "-7", "1.50", "0.0", "3e2", "1E-2", "2.5e+3", "-1.2e-3",
                         '""', '"a"', '"\\n\\t\\"\\\\\\/"', '"\\u00e9"', '"\\ud83d\\ude00"', '"\\ud83d"', '"\\udc00x"',
                         '"\\ud800\\u0041"', '"\\uD834\\uDD1E"', '"é😀"', "1e", "01", "-", "1.", ".5", "nul", "tru",
                         '"\\x"', '"\\u12G4"', '"a\x01"'])
SPACE = st.text(alphabet=WS, max_size=2)


def _json_text(inner: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    item = st.tuples(SPACE, inner, SPACE).map("".join)
    key = st.sampled_from(['"a"', '"b"', '"é"', '"a"', '""', '"\\u0062"'])
    member = st.tuples(SPACE, key, SPACE, st.just(":"), item).map("".join)
    return st.one_of(st.lists(item, max_size=4).map(lambda xs: "[" + ",".join(xs) + "]"),
                     st.lists(member, max_size=4).map(lambda xs: "{" + ",".join(xs) + "}"))


VALID = st.recursive(ATOMS, _json_text, max_leaves=12).flatmap(lambda t: st.tuples(SPACE, st.just(t), SPACE).map("".join))


@st.composite
def _mutated(draw) -> str:
    """A valid-looking text with a character cut, inserted or replaced, or cut short."""
    t = draw(VALID)
    k = draw(st.integers(0, len(t)))
    how = draw(st.sampled_from(["cut", "insert", "replace", "truncate"]))
    c = draw(st.sampled_from(ALPHABET))
    if how == "cut":
        return t[:k] + t[k + 1:]
    if how == "insert":
        return t[:k] + c + t[k:]
    if how == "replace":
        return t[:k] + c + t[k + 1:]
    return t[:k]


#: An exponent of five digits or more would have both sides compute 10^k for a huge k.
TEXTS = st.one_of(VALID, _mutated(), st.text(alphabet=st.sampled_from(ALPHABET), max_size=24)).filter(
    lambda t: not re.search(r"[eE][+-]?[0-9]{5}", t))
CASES = TEXTS.map(lambda t: {"text": t})


def port(fn: str) -> D.Port:
    return D.Port("flock-json-difftest", fn, reference, CASES, package=LEAN, need_rejects=False)


PARSE, PARSE_CORE = port("parse"), port("parse-core")
server = D.server_fixture(PARSE)


@pytest.fixture(autouse=True)
def _big_ints():
    """`1e400` reads as a mantissa of 401 digits; Python refuses str() of an int past 4300 digits by default."""
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    yield
    sys.set_int_max_str_digits(old)


def test_reference_reads_examples():
    assert parse(' {"b": [1, 2.50, -3e1], "a": "\\u00e9", "a": null} ') == [
        "ok", ["obj", [["a", ["null"]], ["b", ["arr", [["num", 1, 0], ["num", 250, 2], ["num", -30, 0]]]]]]]
    assert parse('"\\ud83d\\ude00\\ud83d"') == ["ok", ["str", "😀\ufffd"]]
    assert parse("[1,]") == ["error", "offset 3: unexpected input"]
    assert parse('{"é" 1}') == ["error", "offset 6: expected :"]
    assert parse("1 2") == ["error", "offset 2: expected end of input"]
    assert parse("") == ["error", "offset 0: unexpected end of input"]


def test_cases_reach_both():
    """The draws reach both of the parser's answers: some texts parse, some are refused."""
    for kind in ("ok", "error"):
        find(CASES, lambda c, kind=kind: parse(c["text"])[0] == kind, settings=settings(database=None, derandomize=True))


def test_parse(server):
    D.run(PARSE, examples=3000, server=server)


def test_parse_core(server):
    """Lean core's parser on the same draws: the same answers as the mirror, so as `Flock.JsonParse.parse`."""
    D.run(PARSE_CORE, examples=3000, server=server)
