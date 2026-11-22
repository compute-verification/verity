"""A session record's canonical text: Lean's ``Json.compress`` of the record, which the verifier re-renders and compares.

``session_text(record)`` is the text every writer of a ``flock-live-session/v1`` record (``session.json``) emits:

- objects with their keys in code-point order, and no whitespace;
- strings escaped as Lean's ``Json.escape`` does: ``"``, ``\\``, ``\\n`` and ``\\r`` with a backslash, every other character
  below U+0020 as ``\\u00xx`` (lowercase), and everything else as itself;
- numbers as ``JsonNumber.toString`` prints the number Lean's parser reads from the shortest round-trip text: an integer
  as itself, and a float without trailing zeros or a positive exponent (``-191.0`` → ``-191``, ``1e-05`` → ``0.00001``).

The Rust writers' ``canonical_json`` (``live/src/lib.rs``) is the same function, and both sides' tests hold it to
``live/src/record_text_vectors.json``, whose expected texts are Lean's.
"""
from __future__ import annotations

import math


def _lean_number(text: str) -> str:
    """``JsonNumber.toString`` of the number Lean's ``Json.parse`` reads from ``text`` (a JSON number)."""
    sign = -1 if text.startswith("-") else 1
    body, _, exp = text.lstrip("-").lower().partition("e")
    whole, _, frac = body.partition(".")
    m, e = sign * int(whole + frac), len(frac)
    if exp:
        k = int(exp)
        if k >= 0:
            m, e = m * 10 ** max(0, k - e), max(0, e - k)
        else:
            e -= k
    if e == 0:
        return str(m)
    sg, m = ("" if m >= 0 else "-"), abs(m)
    x = 9 + len(str(m)) - e
    x = x if x < 0 else 0
    ep = 10 ** (e + x)
    left = str(m // ep)
    if m % ep == 0 and x == 0:
        return sg + left
    right = str(ep + m % ep)[1:].rstrip("0")
    return f"{sg}{left}.{right}" + (f"e{x}" if x else "")


def _lean_string(s: str) -> str:
    out = ['"']
    for c in s:
        if c == '"':
            out.append('\\"')
        elif c == "\\":
            out.append("\\\\")
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif ord(c) < 0x20:
            out.append(f"\\u{ord(c):04x}")
        else:
            out.append(c)
    out.append('"')
    return "".join(out)


def session_text(v) -> str:
    """``v`` (what ``json.loads`` returns) as Lean's ``Json.compress`` prints it."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"session_text: {v} is not a JSON number")
        return _lean_number(repr(v))
    if isinstance(v, str):
        return _lean_string(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(session_text(x) for x in v) + "]"
    if isinstance(v, dict):
        if not all(isinstance(k, str) for k in v):
            raise ValueError("session_text: an object key is not a string")
        return "{" + ",".join(_lean_string(k) + ":" + session_text(v[k]) for k in sorted(v)) + "}"
    raise ValueError(f"session_text: {type(v).__name__} is not JSON")
