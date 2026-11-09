"""A `Nat` constant of 2^32 or more in Flock's Lean is a `@[noinline] def`.

Lean's C backend writes a `Nat` literal below 2^32 as `lean_unsigned_to_nat(n)` and one of 2^32 or more as
`lean_cstr_to_nat("…")`, which parses its decimal digits through GMP. The compiler inlines a constant whose value folds
to a literal into each function that reads it, so an unmarked `def FUEL : Nat := 2 ^ 62` is parsed again at every call
of every such function. Marked `@[noinline]`, each read loads the constant, set once when its module is initialized. A
scan of the Lean under `verity/core/service/flock/` refuses a `def` or `abbrev` of type `Nat` whose value, a literal or
`b ^ e` give or take a literal, is 2^32 or more, unless it is a `@[noinline] def`.
"""

from __future__ import annotations

import re

from test_repository import ROOT, _tracked

TREES = ("verity/core/service/flock/",)
LIMIT = 2**32

#: `/- … -/` and `-- …`, blanked so that line numbers stay
COMMENT = re.compile(r"/-.*?-/|--[^\n]*", re.S)
CONST = re.compile(r"^((?:@\[[^\]]*\]\s*)*)(?:(?:private|protected|public|noncomputable)\s+)*(def|abbrev)\s+"
                   r"([^\s:({\[]+)\s*:\s*(?:Nat|ℕ)\s*:=\s*([^\n]+?)\s*$", re.M)
_INT = r"(0x[0-9A-Fa-f][0-9A-Fa-f_]*|[0-9][0-9_]*)"
VALUE = re.compile(rf"\(?\s*{_INT}\s*(?:\^\s*{_INT}\s*)?\)?\s*(?:([+-])\s*{_INT})?")


def _int(literal: str) -> int:
    digits = literal.replace("_", "")
    return int(digits[2:], 16) if digits.startswith("0x") else int(digits)


def _value(text: str) -> int | None:
    """A literal's value, or that of `b ^ e` give or take a literal (`-` truncated at 0, as `Nat`'s); `None` otherwise."""
    m = VALUE.fullmatch(text)
    if not m or (m[2] and _int(m[2]) > 1 << 20):
        return None
    v = _int(m[1]) ** _int(m[2]) if m[2] else _int(m[1])
    if m[4]:
        v = v + _int(m[4]) if m[3] == "+" else max(v - _int(m[4]), 0)
    return v


def offenses(source: str) -> list[tuple[int, str]]:
    """(line, name) of each such constant in a Lean file that is not a `@[noinline] def`, comments aside."""
    text = COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), source)
    found = []
    for m in CONST.finditer(text):
        attrs, kind, name, value = m.groups()
        v = _value(value)
        if v is not None and v >= LIMIT and (kind == "abbrev" or not re.search(r"\bnoinline\b", attrs)):
            found.append((text.count("\n", 0, m.start(3)) + 1, name))
    return found


def test_big_nat_constants_are_noinline() -> None:
    found = [f"{path}:{line}: {name}" for path in _tracked() if path.startswith(TREES) and path.endswith(".lean")
             for line, name in offenses((ROOT / path).read_text())]
    assert not found, (
        "a `Nat` constant of 2^32 or more that the compiler inlines is parsed from its decimal digits at every call "
        "(`lean_cstr_to_nat`); write it `@[noinline] def NAME : Nat := …`, so each read loads it:\n" + "\n".join(found))


def test_the_scan_sees_each_form() -> None:
    assert offenses("def A : Nat := 2 ^ 62\n") == [(1, "A")]
    assert offenses("def B : ℕ := 4294967296\n") == [(1, "B")]
    assert offenses("/-- doc -/\nprivate def C : Nat := 0xFFFF_FFFF_FFFF\n") == [(2, "C")]
    assert offenses("abbrev D : Nat := 2 ^ 64 - 1\n") == [(1, "D")]
    assert offenses("@[inline] def E : Nat := 2 ^ 32\n") == [(1, "E")]
    assert offenses("@[noinline] def F : Nat := 2 ^ 62\n") == []
    assert offenses("@[noinline]\ndef G : Nat := 2 ^ 53 - 1\n") == []
    assert offenses("def H : Nat := 2 ^ 32 - 1\ndef I : Nat := 2 ^ k\ndef J : UInt64 := 2 ^ 63\n") == []
    assert offenses("def M : Nat := x ^^^ 0xFFFFFFFFFFFF\n") == []
    assert offenses("-- def K : Nat := 2 ^ 64\n/- def L : Nat := 2 ^ 64 -/\n") == []
