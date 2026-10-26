"""Representation types (IR §2).

`Value<w>` is the only leaf: one gate produces exactly one `Value<w>`. `Array<n,T>`, tuples and records are
static collections of value *references*; they are structure over references, not structured values.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any


class Type:
    # Memos of structural facts (`leaves`, `field_starts`, `width_set`, `field_types`), filled on first use. They are slots of
    # this base, not dataclass fields, so equality, hashing, repr, replace and pickling see only the type's own fields.
    __slots__ = ("_leaves", "_starts", "_widths", "_types")

    @property
    def leaves(self) -> int:
        raise NotImplementedError

    def leaf_widths(self) -> list[int]:
        raise NotImplementedError

    def width_set(self) -> frozenset:
        """The set of leaf widths, computed structurally (never materialises the leaves)."""
        raise NotImplementedError

    def homogeneous_width(self) -> int | None:
        ws = self.width_set()
        return next(iter(ws)) if len(ws) == 1 else None

    def to_json(self) -> Any:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Value(Type):
    w: int

    def __post_init__(self) -> None:
        if not (isinstance(self.w, int) and not isinstance(self.w, bool) and self.w > 0):
            raise TypeError(f"Value<w> needs a positive int width, got {self.w!r}")

    @property
    def leaves(self) -> int:
        return 1

    def leaf_widths(self) -> list[int]:
        return [self.w]

    def width_set(self) -> frozenset:
        return frozenset({self.w})

    def to_json(self) -> Any:
        return ["Value", self.w]

    def __repr__(self) -> str:
        return f"Value<{self.w}>"


@dataclass(frozen=True, slots=True)
class Array(Type):
    n: int
    elem: Type

    def __post_init__(self) -> None:
        if not (isinstance(self.n, int) and not isinstance(self.n, bool) and self.n >= 0):
            raise TypeError(f"Array<n,T> needs a non-negative int length, got {self.n!r}")
        if not isinstance(self.elem, Type):
            raise TypeError(f"Array element must be a Type, got {self.elem!r}")

    @property
    def leaves(self) -> int:
        try:
            return self._leaves
        except AttributeError:
            n = self.n * self.elem.leaves
            object.__setattr__(self, "_leaves", n)
            return n

    def leaf_widths(self) -> list[int]:
        return self.elem.leaf_widths() * self.n

    def width_set(self) -> frozenset:
        return self.elem.width_set() if self.n else frozenset()

    def to_json(self) -> Any:
        return ["Array", self.n, self.elem.to_json()]

    def __repr__(self) -> str:
        return f"Array<{self.n},{self.elem!r}>"


@dataclass(frozen=True, slots=True)
class Tuple(Type):
    fields: tuple[Type, ...]

    def __init__(self, *fields: Type) -> None:
        if len(fields) == 1 and isinstance(fields[0], (list, tuple)):
            fields = tuple(fields[0])
        for f in fields:
            if not isinstance(f, Type):
                raise TypeError(f"Tuple field must be a Type, got {f!r}")
        object.__setattr__(self, "fields", tuple(fields))

    @property
    def leaves(self) -> int:
        return field_starts(self)[-1]

    def leaf_widths(self) -> list[int]:
        out: list[int] = []
        for f in self.fields:
            out += f.leaf_widths()
        return out

    def width_set(self) -> frozenset:
        return _fields_width_set(self)

    def field_offset(self, i: int) -> int:
        return field_starts(self)[i]

    def to_json(self) -> Any:
        return ["Tuple", [f.to_json() for f in self.fields]]

    def __repr__(self) -> str:
        return "(" + ", ".join(repr(f) for f in self.fields) + ")"


@dataclass(frozen=True, slots=True)
class Record(Type):
    fields: tuple[tuple[str, Type], ...]

    def __init__(self, *pairs: tuple[str, Type], **kw: Type) -> None:
        items = list(pairs) + list(kw.items())
        names = [k for k, _ in items]
        if len(set(names)) != len(names):
            raise TypeError("duplicate record field names")
        for _, t in items:
            if not isinstance(t, Type):
                raise TypeError(f"Record field must be a Type, got {t!r}")
        object.__setattr__(self, "fields", tuple((k, t) for k, t in items))

    @property
    def leaves(self) -> int:
        return field_starts(self)[-1]

    def leaf_widths(self) -> list[int]:
        out: list[int] = []
        for _, t in self.fields:
            out += t.leaf_widths()
        return out

    def width_set(self) -> frozenset:
        return _fields_width_set(self)

    def index(self, name: str) -> int:
        for i, (k, _) in enumerate(self.fields):
            if k == name:
                return i
        raise KeyError(name)

    def field_offset(self, i: int) -> int:
        return field_starts(self)[i]

    def to_json(self) -> Any:
        return ["Record", [[k, t.to_json()] for k, t in self.fields]]

    def __repr__(self) -> str:
        return "{" + ", ".join(f"{k}: {t!r}" for k, t in self.fields) + "}"


def type_from_json(j: Any) -> Type:
    tag = j[0]
    if tag == "Value":
        return Value(j[1])
    if tag == "Array":
        return Array(j[1], type_from_json(j[2]))
    if tag == "Tuple":
        return Tuple(*[type_from_json(f) for f in j[1]])
    if tag == "Record":
        return Record(*[(k, type_from_json(t)) for k, t in j[1]])
    raise ValueError(f"unknown type tag {tag!r}")


def leaves_of(t: Type) -> int:
    return t.leaves


def field_types(t: Tuple | Record) -> tuple[Type, ...]:
    """The field types of a tuple or a record, in order: a record's computed once a type."""
    if not isinstance(t, Record):
        return t.fields
    try:
        return t._types
    except AttributeError:
        types = tuple(ft for _, ft in t.fields)
        object.__setattr__(t, "_types", types)
        return types


def field_starts(t: Tuple | Record) -> tuple[int, ...]:
    """Each field's first leaf in a tuple or a record, then its leaf count: computed once a type."""
    try:
        return t._starts
    except AttributeError:
        out = [0]
        for ft in field_types(t):
            out.append(out[-1] + ft.leaves)
        starts = tuple(out)
        object.__setattr__(t, "_starts", starts)
        return starts


def field_at(t: Tuple | Record, leaf: int) -> int:
    """The field of a tuple or a record whose leaves hold leaf ``leaf`` (``0 <= leaf < t.leaves``)."""
    return bisect.bisect_right(field_starts(t), leaf) - 1


def _fields_width_set(t: Tuple | Record) -> frozenset:
    try:
        return t._widths
    except AttributeError:
        ws = frozenset().union(*(ft.width_set() for ft in field_types(t)))
        object.__setattr__(t, "_widths", ws)
        return ws


def range_widths(t: Type, lo: int, hi: int) -> frozenset:
    """The widths of leaves [lo, hi) of `t`, computed structurally (never materialises the leaves)."""
    if not 0 <= lo <= hi <= t.leaves:
        raise IndexError(f"leaves [{lo}, {hi}) of a {t!r} ({t.leaves} leaves)")
    if hi == lo:
        return frozenset()
    if lo == 0 and hi == t.leaves:
        return t.width_set()
    if isinstance(t, Array):
        L = t.elem.leaves
        first, last = lo // L, (hi - 1) // L
        if first == last:
            return range_widths(t.elem, lo - first * L, hi - first * L)
        out = range_widths(t.elem, lo - first * L, L) | range_widths(t.elem, 0, hi - last * L)
        return out | t.elem.width_set() if last - first >= 2 else out
    fields, starts = field_types(t), field_starts(t)  # type: ignore[arg-type]
    out: frozenset = frozenset()
    i = field_at(t, lo)  # type: ignore[arg-type]
    while i < len(fields) and starts[i] < hi:
        a, b = max(lo, starts[i]), min(hi, starts[i + 1])
        if a < b:
            out |= range_widths(fields[i], a - starts[i], b - starts[i])
        i += 1
    return out


def remove_axis(t: Type, axis: int) -> tuple[int, Type]:
    """For `batch(..., axes=...)`: strip array axis `axis` from `t`; return (length, member type)."""
    if axis == 0:
        if not isinstance(t, Array):
            raise TypeError(f"axis 0 requires an Array, got {t!r}")
        return t.n, t.elem
    if not isinstance(t, Array):
        raise TypeError(f"axis {axis} requires nested Arrays, got {t!r}")
    n, inner = remove_axis(t.elem, axis - 1)
    return n, Array(t.n, inner)
