"""Layout JSON: the rows a sequence slot commits each serving step, in the form ``slots._layout_json`` writes
(``benchmarks/torch_gpu/commit_slots/slots.py``, whose README is the commit call's specification)::

  {"name": str,
   "groups": [{"sources": [bytes a slot has in tensor i, ...],
               "units": [[[[source, offset, nbytes], ...  a row's parts], ...  a unit's rows], ...  the group's units]}]}

A group is what one commit call takes (a layer, say), a unit a commit unit (one unit tree), and a row the concatenation of
its parts, each ``nbytes`` bytes at ``offset`` of the slot's bytes in the group's tensor ``source``. :class:`Part`,
:class:`Row` and :class:`Group` are ``slots``' own, with its checks.

- :func:`of_json` and :func:`load` read a layout, refusing anything else: another key, a non-integer, a unit over
  :data:`UNIT_MAX_BITS`, an empty unit or row, a part outside its tensor. Equal groups become one object, as a layout's
  layers are, so each is checked and indexed once.
- :func:`to_json` writes it back, ``_layout_json``'s output.
- :attr:`Layout.json_digest` is what a deployment file binds: ``identity_digest_sha512(JSON_TAG, to_json(layout))``, the
  parts included (``slots``' own ``layout_digest`` binds only the rows' sizes, which the unit domains need).
- :func:`check_shared` refuses a layout that isn't shared-row (CIR-113): a row of more than :data:`MAX_PARTS` parts, a
  tensor byte in no row or in two, or a producer's output that isn't a window of one row (:meth:`Layout.window`).
"""
from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from verity.errors import InvalidArtifact

from .identity import tagged_sha512_prefix

__all__ = ["JSON_TAG", "MAX_PARTS", "UNIT_MAX_BITS", "Group", "Layout", "Part", "Row", "check_shared", "load", "of_json",
           "to_json"]

JSON_TAG = "verity/commitments/layout-json/v1"
#: a unit's rows are at most 262,144 bits, the widest proof-block size class (``slots.UNIT_MAX_BITS``)
UNIT_MAX_BITS = 262_144
#: a shared row's parts: the commit engine gathers at most two
MAX_PARTS = 2


@dataclass(frozen=True, slots=True)
class Part:
    """``nbytes`` bytes at ``offset`` of a slot's bytes in the group's tensor ``source``."""

    source: int
    offset: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class Row:
    """One committed row: its parts' bytes, concatenated."""

    parts: tuple[Part, ...]

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes for p in self.parts)

    @property
    def bits(self) -> int:
        return 8 * self.nbytes


@dataclass(frozen=True, slots=True)
class Group:
    """What one commit call takes: ``sources[i]`` is the bytes a slot has in the group's tensor i, and ``units`` the group's
    units, each its rows."""

    sources: tuple[int, ...]
    units: tuple[tuple[Row, ...], ...]

    def __post_init__(self) -> None:
        for u, unit in enumerate(self.units):
            if not unit:
                raise InvalidArtifact(f"unit {u}: a unit has at least one row")
            if (bits := sum(r.bits for r in unit)) > UNIT_MAX_BITS:
                raise InvalidArtifact(f"unit {u}: a unit's rows are at most {UNIT_MAX_BITS} bits, not {bits}")
            for r, row in enumerate(unit):
                if not row.parts:
                    raise InvalidArtifact(f"unit {u} row {r}: a row has at least one byte")
                for p in row.parts:
                    if not 0 <= p.source < len(self.sources) or p.offset < 0 or p.nbytes < 1 \
                            or p.offset + p.nbytes > self.sources[p.source]:
                        raise InvalidArtifact(f"unit {u} row {r}: part {p} is outside the group's tensors {self.sources}")


@dataclass(frozen=True)
class Layout:
    """A step's rows a slot, the same for every slot: its groups in order (a repeated group one object); ``name`` names it."""

    name: str
    groups: tuple[Group, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise InvalidArtifact("a layout's name is a string")
        try:
            self.name.encode("utf-8")
        except UnicodeEncodeError:
            raise InvalidArtifact("a layout's name is UTF-8 text") from None

    @cached_property
    def unit_base(self) -> tuple[int, ...]:
        """Group i's first unit in the slot."""
        out, n = [], 0
        for g in self.groups:
            out.append(n)
            n += len(g.units)
        return tuple(out)

    @cached_property
    def json_digest(self) -> bytes:
        """``identity_digest_sha512(JSON_TAG, to_json(self))``, its canonical JSON written a distinct group at a time."""
        text: dict[int, bytes] = {}
        for g in self.groups:
            if id(g) not in text:
                text[id(g)] = _canonical(_group_json(g))
        pieces = [b'{"groups":[']
        for j, g in enumerate(self.groups):
            pieces += (b",", text[id(g)]) if j else (text[id(g)],)
        pieces += (b'],"name":', json.dumps(self.name, ensure_ascii=False).encode("utf-8"), b"}")
        h = hashlib.sha512(tagged_sha512_prefix(JSON_TAG, sum(map(len, pieces))))
        for p in pieces:
            h.update(p)
        return h.digest()

    @cached_property
    def _index(self) -> dict[int, dict[int, tuple[list[int], list[tuple[int, int, int, int, int]], int]]]:
        return {}

    def _parts(self, g: Group) -> dict[int, tuple[list[int], list[tuple[int, int, int, int, int]], int]]:
        """Group ``g``'s parts by tensor, sorted: (start, end, unit in the group, rank, offset in the row) with the starts
        alone, and the widest part's bytes."""
        if (ix := self._index.get(id(g))) is None:
            by: dict[int, list[tuple[int, int, int, int, int]]] = {}
            for u, unit in enumerate(g.units):
                for r, row in enumerate(unit):
                    at = 0
                    for p in row.parts:
                        by.setdefault(p.source, []).append((p.offset, p.offset + p.nbytes, u, r, at))
                        at += p.nbytes
            ix = {}
            for s, v in by.items():
                v.sort()
                ix[s] = ([x[0] for x in v], v, max(x[1] - x[0] for x in v))
            self._index[id(g)] = ix
        return ix

    def window(self, group: int, source: int, offset: int, nbytes: int) -> tuple[int, int, int]:
        """Where bytes ``offset .. offset + nbytes`` of tensor ``source`` of group ``group`` lie whole in one row: the row's
        unit in the slot, its rank in the unit and the bytes' offset in it, the first such row. Refuses bytes no one row
        holds."""
        if not 0 <= group < len(self.groups):
            raise InvalidArtifact(f"the layout has no group {group}")
        end, found = offset + nbytes, None
        if offset >= 0 and nbytes >= 1 and (a := self._parts(self.groups[group]).get(source)) is not None:
            starts, parts, widest = a
            k = bisect_right(starts, offset) - 1
            while k >= 0 and starts[k] >= end - widest:
                lo, hi, u, r, at = parts[k]
                if hi >= end and (found is None or (u, r) < found[:2]):
                    found = (u, r, at + offset - lo)
                k -= 1
        if found is None:
            raise InvalidArtifact(f"group {group} tensor {source}: bytes {offset}..{end} aren't a window of one row")
        return self.unit_base[group] + found[0], found[1], found[2]


def _group_json(g: Group) -> dict[str, Any]:
    return {"sources": list(g.sources), "units": [[[[p.source, p.offset, p.nbytes] for p in r.parts] for r in u] for u in g.units]}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def to_json(layout: Layout) -> dict[str, Any]:
    """``slots._layout_json(layout)``. A repeated group's object is one object: copy it before editing one."""
    memo: dict[int, dict[str, Any]] = {}
    for g in layout.groups:
        if id(g) not in memo:
            memo[id(g)] = _group_json(g)
    return {"name": layout.name, "groups": [memo[id(g)] for g in layout.groups]}


def _group(i: int, sources: list[Any], units: list[Any]) -> Group:
    if not all(type(n) is int and n >= 0 for n in sources):
        raise InvalidArtifact(f"group {i}: a tensor's bytes are a non-negative integer")
    out = []
    for u, unit in enumerate(units):
        if type(unit) is not list:
            raise InvalidArtifact(f"group {i} unit {u}: a unit is a list of rows")
        rows = []
        for r, row in enumerate(unit):
            if type(row) is not list:
                raise InvalidArtifact(f"group {i} unit {u} row {r}: a row is a list of parts")
            parts = []
            for p in row:
                if type(p) is not list or len(p) != 3 or not all(type(x) is int for x in p):
                    raise InvalidArtifact(f"group {i} unit {u} row {r}: a part is [source, offset, nbytes], integers")
                parts.append(Part(*p))
            rows.append(Row(tuple(parts)))
        out.append(tuple(rows))
    try:
        return Group(tuple(sources), tuple(out))
    except InvalidArtifact as e:
        raise InvalidArtifact(f"group {i} {e}") from None


def of_json(doc: Any) -> Layout:
    """The layout ``doc`` is (``slots._layout_json``'s form), refusing anything else; equal groups become one object."""
    if type(doc) is not dict or doc.keys() != {"name", "groups"}:
        raise InvalidArtifact('a layout is {"name", "groups"}')
    if type(doc["groups"]) is not list:
        raise InvalidArtifact("a layout's groups are a list")
    seen: dict[str, Group] = {}
    groups = []
    for i, g in enumerate(doc["groups"]):
        if type(g) is not dict or g.keys() != {"sources", "units"}:
            raise InvalidArtifact(f'group {i}: a group is {{"sources", "units"}}')
        if type(g["sources"]) is not list or type(g["units"]) is not list:
            raise InvalidArtifact(f"group {i}: a group's sources and units are lists")
        try:  # the text, not ==, so that a 1.0 or a true never passes as an earlier group's 1
            key = json.dumps(g, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            raise InvalidArtifact(f"group {i}: not JSON") from None
        if key not in seen:
            seen[key] = _group(i, g["sources"], g["units"])
        groups.append(seen[key])
    return Layout(doc["name"], tuple(groups))


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out = dict(pairs)
    if len(out) != len(pairs):
        raise InvalidArtifact("a layout's JSON repeats a key")
    return out


def load(path: str | os.PathLike[str]) -> Layout:
    """The layout in the Layout JSON file ``path``."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        doc = json.loads(raw, object_pairs_hook=_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise InvalidArtifact(f"{os.fspath(path)}: not JSON ({e})") from None
    return of_json(doc)


def check_shared(layout: Layout, outputs: Iterable[tuple[int, int, int, int]] = ()) -> None:
    """Refuse ``layout`` unless it is shared-row (CIR-113): every row at most :data:`MAX_PARTS` parts; in each group every byte
    of every tensor in exactly one row, except a tensor each row reading it reads whole (a quantizer's f32 scale, beside each
    of its chunks' codes); and each producer output in ``outputs``, (group, tensor, offset, bytes), a window of one row."""
    seen: set[int] = set()
    for i, g in enumerate(layout.groups):
        if id(g) in seen:
            continue
        seen.add(id(g))
        spans: list[list[tuple[int, int]]] = [[] for _ in g.sources]
        for u, unit in enumerate(g.units):
            for r, row in enumerate(unit):
                if len(row.parts) > MAX_PARTS:
                    raise InvalidArtifact(f"group {i} unit {u} row {r}: a shared row has at most {MAX_PARTS} parts, "
                                          f"not {len(row.parts)}")
                for p in row.parts:
                    spans[p.source].append((p.offset, p.offset + p.nbytes))
        for s, (n, sp) in enumerate(zip(g.sources, spans)):
            if sp and all(x == (0, n) for x in sp):
                continue
            end = 0
            for lo, hi in sorted(sp):
                if lo < end:
                    raise InvalidArtifact(f"group {i} tensor {s}: bytes {lo}..{min(hi, end)} are committed twice")
                if lo > end:
                    raise InvalidArtifact(f"group {i} tensor {s}: bytes {end}..{lo} are in no row")
                end = hi
            if end < n:
                raise InvalidArtifact(f"group {i} tensor {s}: bytes {end}..{n} are in no row")
    for o in outputs:
        layout.window(*o)
