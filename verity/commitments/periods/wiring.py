"""A period's wiring across statements: each (writer template, output field) is one tree (`tree`), its leaf i the output
row of the template's unit i, so every value is committed once. A reader's wired port commits nothing: unit by unit it
reads rows of that tree at the positions its wire gives in closed form (`Wired.positions`), and its statement opens each
row it reads by its path, so a sampled proof block's input is the committed output of the unit that wrote it."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from verity.commitments import hm96

BC_BYTES = hm96.SHA512.proven_bytes


def tree(template: str, field: str) -> str:
    """The name of the one tree holding template ``template``'s output field ``field``, in every root, domain and file."""
    return f"{template}/{field}"


def source(radix: Sequence[int], base: int, step: Sequence[int], i: Any) -> np.ndarray:
    """The closed form of a pattern's sources: index ``i``'s digits ``d`` in mixed radix ``radix`` (the first the slowest)
    give ``base + sum(d[k] * step[k])``."""
    i = np.asarray(i, dtype=np.int64).copy()
    out = np.full(i.shape, base, dtype=np.int64)
    for r, s in zip(reversed(radix), reversed(step)):
        out += (i % r) * s
        i //= r
    return out


@dataclass(frozen=True)
class Wired:
    """Reader port ``port``'s wire into the tree of writer template ``writer``'s output field ``out`` (`tree`): unit ``u``
    of the reader's template reads the tree's leaf ``source(radix, base, step, (u - at) * cut[1] + row - cut[0])``, for
    ``u`` in ``[at, at + prod(radix) / cut[1])`` (`interface.Wire`'s closed form at the port's row ``row`` of its cut).
    With reader steps ``rstep`` (an uncut wire), index ``i`` of ``prod(radix)`` is reader unit ``source(radix, at, rstep,
    i)`` reading leaf ``source(radix, base, step, i)``: strided on both sides, a repeat its slowest digit. Several such
    pieces may wire one port, each of its units read by one (`reads`)."""
    port: str
    writer: str
    out: str
    at: int
    radix: tuple[int, ...]
    base: int
    step: tuple[int, ...]
    cut: tuple[int, int] = (0, 1)
    row: int = 0
    rstep: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for k in ("radix", "step", "cut", "rstep"):
            object.__setattr__(self, k, tuple(int(x) for x in getattr(self, k)))
        if (not self.radix or len(self.step) != len(self.radix) or min(self.radix) < 1 or self.at < 0 or self.cut[1] < 1
                or not self.cut[0] <= self.row < self.cut[0] + self.cut[1]
                or (self.rstep and (len(self.rstep) != len(self.radix) or self.cut != (0, 1)))):
            raise ValueError(f"{self.doc()}: a wire is a first reader unit, one step a digit of radix 1 or more, and a row of "
                             "its cut; reader steps, one a digit, only uncut")

    @property
    def reader_units(self) -> int:
        return int(np.prod(self.radix)) // self.cut[1]

    def units(self, lo: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """The reader's units in ``[lo, lo + n)`` (its template's) the wire reads, as offsets from ``lo``, and the leaf of
        the writer's tree each reads."""
        if not self.rstep:
            u = np.arange(max(lo, self.at), min(lo + n, self.at + self.reader_units), dtype=np.int64)
            return u - lo, source(self.radix, self.base, self.step, (u - self.at) * self.cut[1] + self.row - self.cut[0])
        i = np.arange(self.reader_units, dtype=np.int64)
        u = source(self.radix, self.at, self.rstep, i)
        mine = (u >= lo) & (u < lo + n)
        return u[mine] - lo, source(self.radix, self.base, self.step, i[mine])

    def positions(self, lo: int, n: int) -> np.ndarray:
        """The leaves of the writer's tree that the reader's units ``lo`` to ``lo + n`` (its template's) read, every one
        of them the wire's."""
        return reads((self,), lo, n)

    def doc(self) -> dict[str, Any]:
        return {"port": self.port, "writer": self.writer, "out": self.out, "at": self.at, "radix": list(self.radix),
                "base": self.base, "step": list(self.step), **({"cut": list(self.cut), "row": self.row} if self.cut != (0, 1) else {}),
                **({"rstep": list(self.rstep)} if self.rstep else {})}

    @classmethod
    def of(cls, d: Mapping[str, Any]) -> Wired:
        return cls(d["port"], d["writer"], d["out"], int(d["at"]), tuple(d["radix"]), int(d["base"]), tuple(d["step"]),
                   tuple(d.get("cut", (0, 1))), int(d.get("row", 0)), tuple(d.get("rstep", ())))


def reads(pieces: Sequence[Wired], lo: int, n: int) -> np.ndarray:
    """The leaf each of a reader port's units ``lo`` to ``lo + n`` reads, from its pieces (one port's wires); refused
    unless each unit is read by exactly one."""
    out, seen = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    for w in pieces:
        at, leaves = w.units(lo, n)
        out[at] = leaves
        np.add.at(seen, at, 1)
    if n and (seen != 1).any():
        bad = np.flatnonzero(seen != 1)[:4] + lo
        raise ValueError(f"{[w.doc() for w in pieces][:2]}: reader units {bad.tolist()} are read by {seen[bad - lo].tolist()} "
                         "wires, not one each")
    return out


def table(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A reader port's table: the leaves its units read (``positions``, one a unit), each once in first-read order, and
    each unit's ref into them (core's ``registered.table`` order)."""
    u, first, inv = np.unique(np.asarray(positions, dtype=np.int64), return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(u), dtype=np.int64)
    rank[order] = np.arange(len(u), dtype=np.int64)
    return u[order], rank[inv.reshape(-1)]


def strings(publics: Mapping[str, bytes], m: Any, port: str) -> np.ndarray | None:
    """Each of member ``m``'s units' ``b || c`` of its row ``port``, from its public body (`rows.Committed.public_body`): a
    shared-row member's, the row of the port's table its ref names, the row its statement reads. None for a body that is
    not the member's or a port it lacks."""
    names = [q for q, _ in m.all_rows]
    body = publics.get(m.key)
    if body is None or port not in names:
        return None
    if not m.tables:
        if len(body) != m.n * len(names) * BC_BYTES:
            return None
        return np.frombuffer(body, dtype=np.uint8).reshape(m.n, len(names), BC_BYTES)[:, names.index(port)]
    counts = [dict(m.tables).get(q, m.n) for q in names]
    j, at = names.index(port), BC_BYTES * sum(counts)
    if len(body) != at + 4 * m.n * len(names):
        return None
    table = np.frombuffer(body, dtype=np.uint8, count=counts[j] * BC_BYTES, offset=BC_BYTES * sum(counts[:j]))
    refs = np.frombuffer(body, dtype="<u4", offset=at).reshape(m.n, len(names))[:, j]
    return None if (refs >= counts[j]).any() else table.reshape(counts[j], BC_BYTES)[refs]


def validate(members: Sequence[Any], templates: Mapping[str, int], owns: Mapping[str, int]) -> None:
    """Refuse a wiring that is not, for each wired port of a member, one row of its writer template's output field a unit,
    of the port's words and schema, inside that template's units; each port wired at most once. ``templates``: each
    template's unit count; ``owns``: each member's first unit among its template's own, by key. A port's pieces (`reads`)
    share one writer and output field, and read each of its units once."""
    outs = {}
    for m in members:
        for o, _w in m.outs:
            outs[(m.template, o)] = (dict(m.out_rows)[o], m.schema(o))
    for m in members:
        if len({w.port for w in m.wired}) != len({(w.port, w.writer, w.out) for w in m.wired}):
            raise ValueError(f"{m.key}: a port of {[w.port for w in m.wired]} is wired to two trees")
        lo = owns[m.key]
        for w in m.wired:
            if w.port not in dict(m.rows):
                raise ValueError(f"{m.key}: wire {w.doc()} names no input port of its own rows")
            got = outs.get((w.writer, w.out))
            if got != (dict(m.rows)[w.port], m.schema(w.port)):
                raise ValueError(f"{m.key}: its {w.port} rows ({dict(m.rows)[w.port]} words, {m.schema(w.port)}) are not "
                                 f"{tree(w.writer, w.out)}'s rows ({got})")
            _, at = w.units(lo, m.n)
            if len(at) and (at.min() < 0 or at.max() >= templates[w.writer]):
                raise ValueError(f"{m.key}: wire {w.doc()} reads leaves outside {tree(w.writer, w.out)}'s "
                                 f"{templates[w.writer]}")
        for p in {w.port for w in m.wired}:
            reads([w for w in m.wired if w.port == p], lo, m.n)
