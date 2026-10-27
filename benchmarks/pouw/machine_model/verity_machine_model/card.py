"""A device's card file (`verity.hardware`, `verity/hardware/cards/<sku>.toml`): the one source of its ceilings. θ and the
data sheet name the card in `card = "SKU"` and each number the card states as `card = "EXPR"` in place of the number:
EXPR is card entries (dotted paths, `rate.tensor`) and numbers joined by `*` and `/`, read left to right, so a format's
tensor rate is `rate.tensor / weight.s8` and shared memory's bytes a SM-clock `rate.smem * 4`. Every entry it reads is
marked `ceiling`: a measurement or a placeholder is only a lower bound, which no sheet reading may take.
"""
from __future__ import annotations

import re
from fractions import Fraction

from verity.hardware import load

TOKEN = re.compile(r"\s*([*/])\s*")


def card(sku: str) -> dict:
    try:
        return load(sku)
    except KeyError as e:
        raise SystemExit(str(e.args[0])) from None


def value(c: dict, expr: str) -> Fraction:
    """EXPR's value on card ``c``, exact; refused unless every entry it reads is a ceiling."""
    parts = TOKEN.split(expr.strip())
    out = _term(c, parts[0], expr)
    for op, term in zip(parts[1::2], parts[2::2]):
        x = _term(c, term, expr)
        out = out * x if op == "*" else out / x
    return out


def number(c: dict, expr: str) -> int | float:
    """`value` as θ's numbers are written: an int when it is one."""
    v = value(c, expr)
    return int(v) if v.denominator == 1 else float(v)


def pointer(c: dict, expr: str) -> str:
    """Where EXPR's number is: the card file and the expression, which `render` and `theta` print as its source."""
    return f"verity/hardware/cards/{c['sku']}.toml: {expr}"


def _term(c: dict, term: str, expr: str) -> Fraction:
    if re.fullmatch(r"[0-9.eE+-]+", term):
        return Fraction(term)
    entry: object = c
    for key in term.split("."):
        if not isinstance(entry, dict) or key not in entry:
            raise SystemExit(f"{c['sku']}: {expr!r} reads {term}, which the card file doesn't have")
        entry = entry[key]
    if not isinstance(entry, dict) or "value" not in entry:
        raise SystemExit(f"{c['sku']}: {expr!r} reads {term}, which is no number of the card")
    if entry["mark"] != "ceiling":
        raise SystemExit(f"{c['sku']}: {term} is marked {entry['mark']}, not ceiling; write the number with its own "
                         f"document here, or mark it ceiling in the card file with its source")
    return Fraction(str(entry["value"]))
