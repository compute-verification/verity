"""A window's wiring across statements (`verifier/PROTOCOL.md` §16.13 ``Wiring``): a wired member (``serving_rows.Member.wired``,
one ``(port, out, writers)`` per wire) reads, unit by unit, other units' output rows: its unit i's ``port`` row is global unit
``writers[i]``'s ``out`` row, committed under that row's salt, so the two rows' ``b || c`` are one string.  Lean's links hold
inside one statement; these join statements (one coordinate's K-chunks, chunk c's ``acc`` chunk c - 1's output; split partials,
a combine unit's ``in<j>.s`` split s's), so the verifier holds them on the public bodies the leaf layer binds, and a sampled proof block's input is the committed output of
the unit that wrote it.  A unit whose wired row is not its writer's row is faulty in the wired flattening (its outputs
are not its Definition's on its writers' committed outputs), so the verifier holds the wires of the units it samples
(`check_drawn`, `SampledProofs.Wires.Summary.drawnOk` in Lean, whose `drawnOk_iff` makes it the expanded check on the
sampled proof blocks); `check`, every unit's, is the reference.  A member may be wired and read registered rows (a GEMM slice: its
``x`` its writer's output, its ``w`` a weight row): it is then a shared-row member, its wired port a table of its units' own
rows, and a unit's string is the row its ref names (`_strings`), the one its statement reads."""
from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np

from verity.ml.commitments import hm96

SALT_BYTES, BC_BYTES = hm96.SHA512.salt_bytes, hm96.SHA512.proven_bytes


def bases(members: Sequence[Any]) -> np.ndarray:
    """Each member's first global unit, in order: what `_writers` looks a writer up in, made once over a pass of the
    members and not once a wire, which would be quadratic in the members."""
    return np.array([x.base for x in members], dtype=np.int64)


def _writers(members: Sequence[Any], ws: Sequence[int], bases: np.ndarray) -> list[tuple[Any, np.ndarray, np.ndarray]]:
    """Each member that wrote some of a wire's rows (``ws``, one global unit per reader unit), with the reader's units it wrote
    (a mask) and its own units for them; ``bases``, the members' (`bases`)."""
    w = np.asarray(ws, dtype=np.int64)
    k = np.searchsorted(bases, w, side="right") - 1
    return [(members[j], k == j, w[k == j] - bases[j]) for j in np.unique(k)]


def validate(members: Sequence[Any]) -> None:
    """Refuse a wiring that is not one row per unit of a reader port, each another member's output row of its words and schema,
    and each port wired at most once. A shared-row member's wired port is a table of its units' own rows, one a unit."""
    total, at = sum(m.n for m in members), bases(members)
    for m in members:
        if len({p for p, _o, _ws in m.wired}) != len(m.wired):
            raise ValueError(f"{m.key}: ports {[p for p, _o, _ws in m.wired]} are wired more than once")
        for p, o, ws in m.wired:
            if (dict(m.tables).get(p, m.n) != m.n or p not in dict(m.rows) or len(ws) != m.n or min(ws) < 0
                    or max(ws) >= total):
                raise ValueError(f"{m.key}: wiring {p!r} is not one writer in [0, {total}) per unit of an input port of its own rows")
            for wm, _mine, _theirs in _writers(members, ws, at):
                if dict(wm.out_rows).get(o) != dict(m.rows)[p] or wm.schema(o) != m.schema(p):
                    raise ValueError(f"{m.key}: its {p} rows ({dict(m.rows)[p]} words, {m.schema(p)}) are not {wm.key}'s {o!r} output "
                                     f"rows ({dict(wm.out_rows).get(o)} words, {wm.schema(o)})")


def salts(members: Sequence[Any], rows: Mapping[str, np.ndarray], ys: dict[str, bytes], skip: Collection[str] = ()) -> None:
    """Give each wired row its writer's salt (``ys`` by ``<member>/<port>``, each row's 192 bytes); refused when a wired row's words
    are not its writer's output row's (``rows``, the committed rows). ``skip``: ports given by reference, their writers' rows
    (`rows.commit`)."""
    at = bases(members)
    for m in members:
        for p, o, ws in m.wired:
            if m.port(p) in skip:
                continue
            y = np.empty((m.n, SALT_BYTES), dtype=np.uint8)
            for wm, mine, theirs in _writers(members, ws, at):
                if not np.array_equal(rows[m.port(p)][mine], rows[wm.port(o)][theirs]):
                    raise ValueError(f"{m.key}: a {p} row is not the {o} row of the {wm.key} unit that wrote it")
                y[mine] = np.frombuffer(ys[wm.port(o)], dtype=np.uint8).reshape(wm.n, SALT_BYTES)[theirs]
            ys[m.port(p)] = y.tobytes()


def _strings(publics: Mapping[str, bytes], m: Any, port: str) -> np.ndarray | None:
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


def check(members: Sequence[Any], publics: Mapping[str, bytes]) -> dict[str, str]:
    """The verifier's wiring check on every member's public body (``publics[member]``, which the leaf layer binds): each wired
    row's ``b || c`` is its writer's output row's.  ``{member: why}``; empty when every wire holds."""
    bad: dict[str, str] = {}
    at = bases(members)
    for m in members:
        for p, o, ws in m.wired:
            mine = _strings(publics, m, p)
            if mine is None:
                bad[m.key] = f"its public body is not {m.n} units of {len(m.all_rows)} strings"
                break
            for wm, sel, theirs in _writers(members, ws, at):
                got = _strings(publics, wm, o)
                if got is None or not np.array_equal(mine[sel], got[theirs]):
                    bad[m.key] = f"its {p} strings are not the {o} strings of {wm.key}'s units that wrote them"
                    break
            if m.key in bad:
                break
    return bad


def check_drawn(members: Sequence[Any], publics: Mapping[str, bytes], units: Any) -> dict[str, str]:
    """`check` on the sampled proof blocks alone (``units``, global unit indices): each sampled proof block's wired rows' ``b || c`` are its
    writers' output rows', work proportional to the sampled proof blocks.  ``{member: why}``; empty when every drawn wire holds."""
    drawn = np.unique(np.asarray(list(units), dtype=np.int64))
    bad: dict[str, str] = {}
    at = bases(members)
    for m in members:
        local = drawn[(drawn >= m.base) & (drawn < m.base + m.n)] - m.base
        if not m.wired or not len(local):
            continue
        for p, o, ws in m.wired:
            mine = _strings(publics, m, p)
            if mine is None:
                bad[m.key] = f"its public body is not {m.n} units of {len(m.all_rows)} strings"
                break
            for wm, sel, theirs in _writers(members, np.asarray(ws, dtype=np.int64)[local], at):
                got = _strings(publics, wm, o)
                if got is None or not np.array_equal(mine[local][sel], got[theirs]):
                    bad[m.key] = f"a drawn unit's {p} strings are not the {o} strings of the {wm.key} units that wrote them"
                    break
            if m.key in bad:
                break
    return bad
