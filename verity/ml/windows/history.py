"""The verifier's side of a window's history tables (`interface.History`): each table's rows' public strings in the
positions its runs give (`history.order`), from its writers' public bodies (`wiring.strings`, which the leaf layer binds)
and from the strings the verifier holds of the table's earlier windows, and the table's root over them, the placeholder
past its length, to which it holds the root the window's registration records (`history.check`): the prover's root is
never trusted. Its results are the verifier's pin of each table, from which `interface.registered_reads` makes the
table's supplied values and by which `interface.input_count` counts its reads registered, and each table's strings, which
a later window's held runs read."""
from __future__ import annotations

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


def pins(doc: Mapping[str, Any], w: CS.Binding, publics: Mapping[str, bytes], roots: Mapping[str, str],
         held: Mapping[str, Sequence[bytes]] | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, list[bytes]]]:
    """The verifier's pin of each of the circuit record ``doc``'s history tables in its window ``w``
    (``{"value", "root", "leaves", "words", "domain"}``, a pinned registered parameter's, which
    `interface.registered_reads` and `interface.input_count` take), its root the one over the table's strings (`strings`,
    ``held`` the strings the verifier holds by value); refused (ValueError) unless the window's registration records that
    root for it (``roots``, `interface.registered_roots`). -> (the pins, each table's strings by name)."""
    reg = I.declarations(doc)[2]
    out, tables = {}, {}
    for h in (reg.histories if reg is not None else ()):
        bcs = strings(h, w, publics, held or {})
        got = roots.get(h.name)
        if got is None:
            raise ValueError(f"history {h.name!r}: the window's registration records no root for it")
        HT.check(h.program, h.name, h.words, bcs, bytes.fromhex(got), h.capacity)
        out[h.name] = {"value": h.name, "root": got.lower(), "leaves": h.capacity, "words": h.words,
                       "domain": HT.domain(h.program, h.name, h.words, h.capacity).domain_id.hex()}
        tables[h.name] = bcs
    return out, tables
