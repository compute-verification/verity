"""A window's wiring across statements (`verifier/PROTOCOL.md` §16.13 ``Wiring``): a wired member (``serving_rows.Member.wired``,
one ``(port, out, writers)`` per wire) reads, unit by unit, other units' output rows: its unit i's ``port`` row is global unit
``writers[i]``'s ``out`` row, committed under that row's salt, so the two rows' ``b || c`` are one string.  Lean's links hold
inside one statement; these join statements (one coordinate's K-chunks, chunk c's ``acc`` chunk c - 1's output; split partials,
a combine unit's ``in<j>.s`` split s's), so the verifier holds them on the public bodies the leaf layer binds, before the draw
(`check`), and a drawn unit's input is the committed output of the unit that wrote it. A shared-row member (one reading
registered rows) may be wired on its other ports, each its own table of one row a unit: its unit's string is that table's
row at the unit's ref, both in its public body."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from verity.ml.commitments import hm96

SALT_BYTES, BC_BYTES = hm96.SHA512.salt_bytes, hm96.SHA512.proven_bytes


def _writers(members: Sequence[Any], ws: Sequence[int]) -> list[tuple[Any, np.ndarray, np.ndarray]]:
    """Each member that wrote some of a wire's rows (``ws``, one global unit per reader unit), with the reader's units it wrote
    (a mask) and its own units for them."""
    bases = np.array([x.base for x in members], dtype=np.int64)
    w = np.asarray(ws, dtype=np.int64)
    k = np.searchsorted(bases, w, side="right") - 1
    return [(members[j], k == j, w[k == j] - bases[j]) for j in np.unique(k)]


def validate(members: Sequence[Any]) -> None:
    """Refuse a wiring that is not one row per unit of a reader port, each another member's output row of its words and schema,
    and each port wired at most once."""
    total = sum(m.n for m in members)
    for m in members:
        if len({p for p, _o, _ws in m.wired}) != len(m.wired):
            raise ValueError(f"{m.key}: ports {[p for p, _o, _ws in m.wired]} are wired more than once")
        for p, o, ws in m.wired:
            if dict(m.tables).get(p, m.n) != m.n or p not in dict(m.rows) or len(ws) != m.n or min(ws) < 0 or max(ws) >= total:
                raise ValueError(f"{m.key}: wiring {p!r} is not one writer in [0, {total}) per unit of an input port of its own rows")
            for wm, _mine, _theirs in _writers(members, ws):
                if dict(wm.out_rows).get(o) != dict(m.rows)[p] or wm.schema(o) != m.schema(p):
                    raise ValueError(f"{m.key}: its {p} rows ({dict(m.rows)[p]} words, {m.schema(p)}) are not {wm.key}'s {o!r} output "
                                     f"rows ({dict(wm.out_rows).get(o)} words, {wm.schema(o)})")


def salts(members: Sequence[Any], rows: Mapping[str, np.ndarray], ys: dict[str, bytes]) -> None:
    """Give each wired row its writer's salt (``ys`` by ``<member>/<port>``, each row's 192 bytes); refused when a wired row's words
    are not its writer's output row's (``rows``, the committed rows)."""
    for m in members:
        for p, o, ws in m.wired:
            y = np.empty((m.n, SALT_BYTES), dtype=np.uint8)
            for wm, mine, theirs in _writers(members, ws):
                if not np.array_equal(rows[m.port(p)][mine], rows[wm.port(o)][theirs]):
                    raise ValueError(f"{m.key}: a {p} row is not the {o} row of the {wm.key} unit that wrote it")
                y[mine] = np.frombuffer(ys[wm.port(o)], dtype=np.uint8).reshape(wm.n, SALT_BYTES)[theirs]
            ys[m.port(p)] = y.tobytes()


def _layout(m: Any, body: bytes) -> tuple[list[str], list[int], np.ndarray] | None:
    """A shared-row member's public body (``rows.Committed.public_body``): each port's table size, in port order, and the
    refs (instance-major, a column a port); None when the body is not that."""
    names, tables = [q for q, _ in m.all_rows], dict(m.tables)
    sizes = [tables.get(q, m.n) for q in names]
    at = sum(sizes) * BC_BYTES
    if len(body) != at + 4 * m.n * len(names):
        return None
    refs = np.frombuffer(body, dtype="<u4", offset=at).reshape(m.n, len(names))
    return names, sizes, refs


def _strings(publics: Mapping[str, bytes], m: Any, port: str) -> np.ndarray | None:
    """Each of member ``m``'s units' ``port`` string in its public body: its own (instance-major), or in a shared-row member
    its port table's row at its ref."""
    names = [q for q, _ in m.all_rows]
    body = publics.get(m.key)
    if body is None:
        return None
    if not m.tables:
        if len(body) != m.n * len(names) * BC_BYTES:
            return None
        return np.frombuffer(body, dtype=np.uint8).reshape(m.n, len(names), BC_BYTES)[:, names.index(port)]
    lay = _layout(m, body)
    if lay is None:
        return None
    names, sizes, refs = lay
    j = names.index(port)
    col = refs[:, j]
    if (col >= sizes[j]).any():
        return None
    table = np.frombuffer(body, dtype=np.uint8, count=sizes[j] * BC_BYTES, offset=sum(sizes[:j]) * BC_BYTES)
    return table.reshape(sizes[j], BC_BYTES)[col]


def string_offset(m: Any, port: str, body: bytes) -> int:
    """The byte offset of unit 0's ``port`` string in member ``m``'s public ``body`` (what `check` reads for it)."""
    names = [q for q, _ in m.all_rows]
    if not m.tables:
        return names.index(port) * BC_BYTES
    lay = _layout(m, body)
    if lay is None:
        raise ValueError(f"{m.key}: its public body is not a shared-row member's")
    names, sizes, refs = lay
    j = names.index(port)
    return (sum(sizes[:j]) + int(refs[0, j])) * BC_BYTES


def check(members: Sequence[Any], publics: Mapping[str, bytes]) -> dict[str, str]:
    """The verifier's wiring check on every member's public body (``publics[member]``, which the leaf layer binds): each wired
    row's ``b || c`` is its writer's output row's.  ``{member: why}``; empty when every wire holds."""
    bad: dict[str, str] = {}
    for m in members:
        for p, o, ws in m.wired:
            mine = _strings(publics, m, p)
            if mine is None:
                bad[m.key] = f"its public body is not {m.n} units of {len(m.all_rows)} strings, or their tables and refs"
                break
            for wm, sel, theirs in _writers(members, ws):
                got = _strings(publics, wm, o)
                if got is None or not np.array_equal(mine[sel], got[theirs]):
                    bad[m.key] = f"its {p} strings are not the {o} strings of {wm.key}'s units that wrote them"
                    break
            if m.key in bad:
                break
    return bad
