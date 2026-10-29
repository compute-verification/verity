"""A `verity.ml.circuits.types.Type`'s `repr` read back to the Type (a Program row's port types are written as reprs)."""
from __future__ import annotations

__all__ = ["TypeReprError", "parse_type_repr"]


class TypeReprError(ValueError):
    pass


def parse_type_repr(s: str):
    """The `repr` of a `verity.ml.circuits.types.Type` back to a Type: `Value<32>`, `Array<31,Value<32>>`, `(A, B)`, `{name: A, other: B}`."""
    from verity.ml.circuits.types import Array, RecordType, Tuple as TupleT, Value
    pos = 0
    n = len(s)

    def skip_ws() -> None:
        nonlocal pos
        while pos < n and s[pos] in " \n\t":
            pos += 1

    def expect(ch: str) -> None:
        nonlocal pos
        skip_ws()
        if pos >= n or s[pos] != ch:
            raise TypeReprError(f"type repr: expected {ch!r} at {pos} in {s[:80]!r}")
        pos += 1

    def parse():
        nonlocal pos
        skip_ws()
        if s[pos:pos + 6] == "Value<":
            pos += len("Value<")
            j = s.index(">", pos)
            w = int(s[pos:j])
            pos = j + 1
            return Value(w)
        if s[pos:pos + 6] == "Array<":
            pos += len("Array<")
            j = s.index(",", pos)
            k = int(s[pos:j])
            pos = j + 1
            elem = parse()
            expect(">")
            return Array(k, elem)
        if s[pos] == "(":
            pos += 1
            fields = []
            skip_ws()
            while s[pos] != ")":
                fields.append(parse())
                skip_ws()
                if s[pos] == ",":
                    pos += 1
                    skip_ws()
            pos += 1
            return TupleT(*fields)
        if s[pos] == "{":
            pos += 1
            pairs = []
            skip_ws()
            while s[pos] != "}":
                j = s.index(":", pos)
                key = s[pos:j].strip()
                pos = j + 1
                pairs.append((key, parse()))
                skip_ws()
                if s[pos] == ",":
                    pos += 1
                    skip_ws()
            pos += 1
            return RecordType(*pairs)
        raise TypeReprError(f"type repr: unexpected {s[pos:pos + 20]!r} at {pos}")

    t = parse()
    skip_ws()
    if pos != n:
        raise TypeReprError(f"type repr: trailing {s[pos:pos + 20]!r}")
    return t
