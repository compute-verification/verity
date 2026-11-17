"""A string is asked whether it is a number with `str.isdecimal`, never with `str.isdigit` or `str.isnumeric`.

`int` reads the decimal digits and no other digit: `"²".isdigit()` and `"½".isnumeric()` are true, and `int` raises on
both. A line that asks `isdigit` and then calls `int` therefore raises `int`'s bare `ValueError` where it has an answer
of its own for "no number" (a refusal by name, None, the other branch), and a line that asks and calls no `int` takes
`²` for a number. Every string `isdecimal` accepts, `int` reads.

`bytes` has `isdigit`, true of the ten ASCII digits alone, and neither `isdecimal` nor `isnumeric`. A place that reads
bytes asks their latin-1 text, `b.decode("latin-1").isdecimal()`, which answers as `b.isdigit()` does for every value
and decodes every byte; the bare `isdecimal` there is an `AttributeError`, which mypy's lint names.

Checked in `verity_vllm/`, with no allowlist: an attribute named `isdigit` or `isnumeric`, called (`s.isdigit()`) or
taken as a value (`key=str.isnumeric`), whatever its receiver is. A comment or a string that spells the word is none,
and neither is a name put together at run time (`getattr(s, "isdigit")`).
"""
from __future__ import annotations

import ast
import sys
from functools import cache

from . import _ratchet as R

RULE = "decimal reads (a number is what `int` reads)"
OLD = ("isdigit", "isnumeric")
FIX = ("use `isdecimal`: `int` does not read every digit (`\"²\".isdigit()` is true, and `int(\"²\")` raises); of bytes, "
       "which have no `isdecimal`, ask the latin-1 text, `b.decode(\"latin-1\").isdecimal()`")


def file_violations(src: str, rel: str) -> list[R.Violation]:
    return [R.Violation(rel, "digit-test", symbol, node.attr, node.lineno)
            for node, symbol in R.walk(R.parse_src(src)) if isinstance(node, ast.Attribute) and node.attr in OLD]


@cache
def scan() -> tuple[R.Violation, ...]:
    return tuple(v for p in R.library_files(root=R.PKG) for v in file_violations(R.source(p), R.rel(p)))


def test_the_library_asks_isdecimal() -> None:
    found = scan()
    assert not found, (f"{RULE}: {len(found)} place(s) ask `isdigit` or `isnumeric`; {FIX}:\n"
                       + "\n".join(f"  {v.file}:{v.line} [{v.kind}] {v.symbol}: {v.detail}" for v in found))


def test_the_scan_finds_a_call_and_a_value_and_no_word_in_a_comment_or_a_string() -> None:
    src = (
        "def f(s, xs):\n"
        '    """isdigit in a docstring is no place."""\n'
        "    a = s.isdigit()          # isnumeric in a comment is none either\n"
        "    b = sorted(xs, key=str.isnumeric)\n"
        '    c = s.isdecimal() and "isdigit" in s\n'
        "    return a, b, c\n"
        "class C:\n"
        "    def g(self, t): return [x for x in t if x.strip().isdigit() or x.isnumeric()]\n"
    )
    got = [(v.symbol, v.detail, v.line) for v in file_violations(src, "verity_vllm/x.py")]
    assert got == [("f", "isdigit", 3), ("f", "isnumeric", 4), ("C.g", "isdigit", 8), ("C.g", "isnumeric", 8)], got
    assert file_violations("ok = '12'.isdecimal()\nisdigit = 1\n", "verity_vllm/y.py") == []


def test_int_reads_every_decimal_character_and_no_other_digit() -> None:
    """The rule's reason, held to this Python's tables: over every character, `isdecimal` is exactly "`int` reads it"."""
    def reads(c: str) -> bool:
        try:
            int(c)
        except ValueError:
            return False
        return True
    chars = [chr(i) for i in range(sys.maxunicode + 1)]
    assert [c for c in chars if c.isdecimal() != reads(c)] == []
    other_digits = [c for c in chars if (c.isdigit() or c.isnumeric()) and not c.isdecimal()]
    assert "²" in other_digits and "½" in other_digits and not any(reads(c) for c in other_digits)


def test_bytes_ask_their_latin_1_text() -> None:
    """What the rule tells a place that reads bytes: the text's `isdecimal` is the bytes' `isdigit`, for every value.
    Both are "not empty, and true of each element", so the 256 bytes and the empty value decide it."""
    assert not hasattr(b"", "isdecimal") and not hasattr(b"", "isnumeric")
    values = [b"", *(bytes([i]) for i in range(256))]
    assert [v for v in values if v.isdigit() != v.decode("latin-1").isdecimal()] == []
    assert [v for v in values if v.isdigit()] == [bytes([c]) for c in b"0123456789"]
