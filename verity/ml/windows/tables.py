"""A window's row tables on the verifier's side (`serving_rows.RowTable`): its linked members' public strings tied to the tables'
roots, the segments' roots and the registered ``row-table`` root (`check_row_tables`); each linked member's ``--registered`` file
for the Lean verifier of record (`registered_reads`); and the audit record's account of what each drawn member's verdict reported
reading (`reported_reads`).  Everything here reads the window, the members' public bodies and the roots, never what a table's
names or row keys mean.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity.ml.commitments import row_table as RT

from verity.ml.windows import rows as CS

__all__ = ["check_row_tables", "registered_reads", "reported_reads"]


def check_row_tables(window: CS.Binding, publics: Mapping[str, bytes], roots: Mapping[str, str | bytes],
                     segment_roots: Mapping[int, bytes] | None = None) -> dict[str, str]:
    """The verifier's tie between the row tables, their linked members and the recorded segment roots: each segment's root
    recomputed from the linked members' public ``x`` strings (``publics[member]``, their public bodies, which the leaf layer
    binds), two members' strings for one row the same, and the ``row-table`` port's registered root (``roots``) the tree over
    them; with ``segment_roots`` (what the integration recorded before each segment's coin, by segment), each segment's
    recomputed root is the recorded one and the registered root the tree over the recorded ones.  ``{key: why}`` keyed by a
    member, a table's port, a segment (``row-table/<segment>``) or the port; empty when everything holds."""
    strings, bad = _linked_strings(window, publics)
    table_roots: dict[int, bytes] = {}
    for k, (t, ss) in enumerate(zip(window.row_tables, strings, strict=True)):
        if t.port in bad:
            continue
        if any(s is None for s in ss):
            bad[t.port] = "a row no member's"
        else:
            table_roots[k] = RT.RowTable(window.table_domain(k), ss).root
    if not window.row_tables:
        return bad
    want = roots.get(CS.TABLE_PORT)
    want = bytes.fromhex(want) if isinstance(want, str) else want
    segs = {s: window.segment_root(s, [table_roots[k] for k in ks]) for s, ks in window.segments
            if all(k in table_roots for k in ks)}
    if segment_roots is not None:
        for s, root in segs.items():
            if root != segment_roots.get(s):
                bad[f"{CS.TABLE_PORT}/{s}"] = "the members' strings are not the segment root recorded"
        recorded = [segment_roots.get(s) for s, _ in window.segments]
        if any(r is None for r in recorded) or RT.RootTree(window.tables_domain, recorded).root != want:
            bad[CS.TABLE_PORT] = "its root is not the tree over the recorded segment roots"
    elif len(segs) < len(window.segments):
        bad[CS.TABLE_PORT] = "a segment whose tables do not hold"
    elif RT.RootTree(window.tables_domain, list(segs.values())).root != want:
        bad[CS.TABLE_PORT] = "the members' strings are not its root's rows"
    return bad


def registered_reads(window: CS.Binding, publics: Mapping[str, bytes], *, registered: Any) -> dict[str, dict[str, Any]]:
    """The Lean verifier's ``--registered`` file of each linked member (``registered-reads/v1``, in the format of ``registered``,
    core's ``one_stage.registered``, by member key): its ``x`` port reads its one row table, under the table's root recomputed
    from the members' public strings (the root ``check_row_tables`` ties to the registration and the segment roots, so call it
    after that passes) and its domain under the window's context, and unit ``m.base + i`` reads the table row its ``i``-th unit
    links; a member linking rows of two tables is refused, whatever built it.  The header's ``registered`` entry
    (``serving_rows.registered_entry``) must give each of the member's ``x`` rows a path to that root at that position."""
    strings, bad = _linked_strings(window, publics)
    out: dict[str, dict[str, Any]] = {}
    for m in window.members:
        if not m.links:
            continue
        if len({t for t, _ in m.links}) != 1:
            raise ValueError(f"{m.key}: its {CS.LINKED} rows are rows of several row tables, which no one registered root opens")
        k = m.links[0][0]
        t, ss = window.row_tables[k], strings[k]
        if m.key in bad or t.port in bad or any(s is None for s in ss):
            raise ValueError(f"{m.key}: {t.port}'s rows do not hold among the members' strings: {bad}")
        port = {"value": t.port, "root": RT.RowTable(window.table_domain(k), ss).root.hex(), "leaves": t.n, "words": t.words,
                "word_bits": registered.WORD_BITS, "domain": window.table_domain(k).domain_id.hex(),
                registered.SCHEMA_KEY: registered.SCHEMA_SEG, "reads": [m.links[j][1] for j in m.link_refs],
                registered.READS_FROM: m.base}
        out[m.key] = {"format": registered.READS_V1, "ports": {CS.LINKED: port}}
    return out


def reported_reads(reads: Mapping[str, Mapping[str, Any]], verdicts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The audit record's ``registered_reads``: for each drawn member with registered reads (``verdicts[key]``, its Lean
    verifier of record's verdict), the value each registered port reads (a linked member's ``x``, its row table) and the root
    the verifier's own ``--registered`` file named (``reads[key]``: `registered_reads`, or an outside value's), beside the
    ``registered`` finding the verdict reports opening the member's rows against.  ``ok`` when every member's verdict accepted
    it and reports exactly its file's ports, values and roots."""
    chunks = {}
    for key, v in verdicts.items():
        want = {p: {"value": e["value"], "root": e["root"]} for p, e in (reads.get(key) or {}).get("ports", {}).items()}
        got = v.get("registered")
        chunks[key] = {"reads": want, "reported": got, "ok": bool(want) and v.get("accepted") is True and got == want}
    return {"ok": all(c["ok"] for c in chunks.values()), "chunks": chunks}


def _linked_strings(window: CS.Binding, publics: Mapping[str, bytes]) -> tuple[list[list[bytes | None]], dict[str, str]]:
    """Each row table's strings as the linked members' public ``x`` strings give them (None where no member links a row), and
    ``{key: why}`` for a member whose body is short or a table two members' strings disagree on."""
    strings: list[list[bytes | None]] = [[None] * t.n for t in window.row_tables]
    bad: dict[str, str] = {}
    for m in window.members:
        if not m.links:
            continue
        body = publics.get(m.key, b"")
        if len(body) < CS.BC_BYTES * len(m.links):
            bad[m.key] = f"its public body is {len(body)} bytes, short of its {len(m.links)} {CS.LINKED} strings"
            continue
        for j, (t, r) in enumerate(m.links):
            s = body[j * CS.BC_BYTES:(j + 1) * CS.BC_BYTES]
            if strings[t][r] is not None and strings[t][r] != s:
                bad.setdefault(window.row_tables[t].port, f"row {r}: two members' strings differ ({m.key})")
            strings[t][r] = s
    return strings, bad
