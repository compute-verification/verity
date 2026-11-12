"""The verifier's side of a window's history tables (`interface.History`). The verifier of record computes each table's
root itself, over its rows' public strings in the positions its runs give (`history.order`), from its writers' public
bodies, which the leaf layer binds, and from the strings it holds of the table's earlier windows, the placeholder past
its length, and holds the root the window's registration records to it before the sample (one-stage
`registration.check`'s ``history``, `history.roots_of`, a root both its verifiers refuse being its ``R7-history``):
`params` is that ``history``, and the prover's root is never trusted. `pins` is the verifier's pin of each table at that
registered root, from which `interface.registered_reads` makes the table's supplied values and by which
`interface.input_count` counts its reads registered; `params` reads each table's root from its pin, so the root the reads
open against is the one the verifier of record holds. `strings` is a table's strings, which a later window's held runs
read."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from verity.ml.commitments import history as HT

from . import interface as I, rows as CS, wiring as WI


def strings(h: I.History, w: CS.Binding, publics: Mapping[str, bytes], held: Mapping[str, Sequence[bytes]]) -> list[bytes]:
    """History ``h``'s rows' public strings in position order: a unit run's its writer's units' ``out`` strings, from their
    members' public bodies (``publics``, by member key, in window ``w``); a held run's rows of a value the verifier holds
    (``held``: each value's strings, from the table's earlier windows). Refused (ValueError) as `history.order` refuses
    the runs, or where a member of the writer has no public body of its units' strings."""
    ranges = {t: (lo, n) for _k, t, lo, n in CS.template_partition(w.program, w.query).ranges}
    if h.writer not in ranges:
        raise ValueError(f"history {h.name!r}: its writer {h.writer} is no template of the window")
    first, units = ranges[h.writer]
    srcs = HT.order(h.runs, h.length, h.writer, h.out, units, {v: len(s) for v, s in held.items()})
    mine: np.ndarray | None = None
    if any(s[0] == "unit" for s in srcs):
        parts, at = [], first
        for m in sorted((m for m in w.members if m.template == h.writer), key=lambda m: m.base):
            got = WI.strings(publics, m, h.out) if m.base == at else None
            if got is None:
                raise ValueError(f"history {h.name!r}: {m.key} has no public body of its {h.out} strings")
            parts.append(got)
            at += m.n
        if at != first + units:
            raise ValueError(f"history {h.name!r}: {h.writer}'s members hold {at - first} of its {units} units")
        mine = np.concatenate(parts)
    return [bytes(mine[s[1]]) if s[0] == "unit" else bytes(held[s[1]][s[2]]) for s in srcs]


def _histories(doc: Mapping[str, Any]) -> tuple[I.History, ...]:
    reg = I.declarations(doc)[2]
    return reg.histories if reg is not None else ()


def _root(h: I.History, roots: Mapping[str, str]) -> str:
    got = roots.get(h.name)
    if got is None:
        raise ValueError(f"history {h.name!r}: the window's registration records no root for it")
    return got.lower()


def _pin(h: I.History, root: str) -> dict[str, Any]:
    return {"value": h.name, "root": root, "leaves": h.capacity, "words": h.words,
            "domain": HT.domain(h.program, h.name, h.words, h.capacity).domain_id.hex()}


def params(doc: Mapping[str, Any], w: CS.Binding, publics: Mapping[str, bytes], pinned: Mapping[str, Mapping[str, Any]],
           held: Mapping[str, Sequence[bytes]] | None = None) -> dict[str, Any] | None:
    """The verifier of record's ``history`` for the circuit record ``doc``'s history tables in its window ``w``
    (``{"digests", "tables", "held", "registered"}``, `history.roots_of`'s form): the SHA-512 of every member's public
    body (``publics``, by member key, the digests the leaf layer is over); each table's declaration, with the units its
    writer has in the window (``first``, ``units``) and its writer's members, each with its public body, its rows in the
    body's order (``ports``) and its shared-row tables; by table, the strings the verifier holds of each value its held
    runs read (``held``, by value, from the table's earlier windows); and each table's root, its pin's (``pinned``, the
    pins the reads are made from, `pins`). None for a record with no history tables. Refused (ValueError) where a table
    has no pin or a pin other than `pins` makes of it, its writer is no template of the window, or a member of its
    writer has no public body."""
    hs = _histories(doc)
    if not hs:
        return None
    ranges = {t: (lo, n) for _k, t, lo, n in CS.template_partition(w.program, w.query).ranges}
    held = held or {}
    tables, kept, registered = [], {}, {}
    for h in hs:
        pin = pinned.get(h.name)
        if pin is None or dict(pin) != _pin(h, str(pin.get("root"))):
            raise ValueError(f"history {h.name!r}: the reads' pin of it is not the table's (`history.pins`): {pin}")
        if h.writer not in ranges:
            raise ValueError(f"history {h.name!r}: its writer {h.writer} is no template of the window")
        registered[h.name] = pin["root"]
        members = []
        for m in sorted((m for m in w.members if m.template == h.writer), key=lambda m: m.base):
            if m.key not in publics:
                raise ValueError(f"history {h.name!r}: {m.key} has no public body")
            members.append({"key": m.key, "base": m.base, "n": m.n, "ports": [q for q, _ in m.all_rows],
                            "body": bytes(publics[m.key]).hex(), **({"tables": dict(m.tables)} if m.tables else {})})
        first, units = ranges[h.writer]
        tables.append({**h.doc(), "first": first, "units": units, "members": members})
        if values := [v for v in h.held if v in held]:
            kept[h.name] = {v: [bytes(s).hex() for s in held[v]] for v in values}
    return {"digests": {k: hashlib.sha512(b).hexdigest() for k, b in publics.items()}, "tables": tables, "held": kept,
            "registered": registered}


def pins(doc: Mapping[str, Any], roots: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """The verifier's pin of each of the circuit record ``doc``'s history tables (``{"value", "root", "leaves", "words",
    "domain"}``, a pinned registered parameter's, which `interface.registered_reads` and `interface.input_count` take):
    the root the window's registration records for it (``roots``, `interface.registered_roots`), which stands only beside
    `params` of the same pins, by which the verifier of record holds it to the root over the table's strings before the
    sample. Refused (ValueError) where the registration records none."""
    return {h.name: _pin(h, _root(h, roots)) for h in _histories(doc)}
