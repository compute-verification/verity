"""A window's operand rows: the rows its tiles read, committed per window before its salt.

A served window's tiles read their operands from committed rows, not from rows the prover chose after the fact: each
operand (``a``, the tiles' A rows; ``b``, their B rows where the tile does not read them from registered weights) is a
per-window registered value (`one_stage.registered`, a row in the reading port's own shape, ``hm96-sha512/row/v2``), its
domain binding the window's context, so a row committed for one window opens under no other's. The tiles' ports declare
reads of them at public positions (`interface.Read`), so the window commits each read port row under the value's salt and
Lean's registered reads open it against the value's root.

The registration's window names them (`block`, its ``activations``: ``{program, values: {value: entry}}``), so they are
fixed before the draw, and their root X (`root`) before the window's salt: the salt is released against X.
The verifier holds each entry to the domain its program, value, rows and the window's context make (`held`), and reads
the tiles' rows against those entries alone.

Not proved here: that the A rows are the rows the served model read (the served activations ``x`` are one token's row,
an A row R tokens' K/2 codes): that tie is a gather, still open (`GAPS`)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.sampled_proofs.one_stage import registered as G

#: the root X of a window's operands (`root`, `core_check.values_root`): SHA-512 over this tag, the program, and each
#: value's name, domain and root in name order
TAG = CC.VALUES_TAG
KEY = "activations"
#: what a window's operands don't establish yet, named wherever a record reports them (`stated`)
GAPS = ("the A rows are not proved to be the served activations x (an x row is one token's codes, an A row R tokens' "
        "K/2 codes): a gather",
        "the B rows are not proved to be the registered weights' rows")


def commit(program: str, context: str, rows: Mapping[str, Sequence[Sequence[int]]],
           salts: Mapping[str, Sequence[bytes]] | None = None) -> dict[str, G.Registered]:
    """Each operand's rows (u16 words, the reading port's row) committed as a per-window value of ``program`` under the
    window's ``context``; fresh salts unless ``salts`` (by value)."""
    return {v: G.Registered.commit(program, v, rs, (salts or {}).get(v), schema=G.SCHEMA_V2, window=context)
            for v, rs in sorted(rows.items())}


def block(program: str, values: Mapping[str, G.Registered]) -> dict[str, Any]:
    """What the registration's window names of the operands (its ``activations``)."""
    return {"program": program, "values": {v: r.entry() for v, r in sorted(values.items())}}


def root(b: Mapping[str, Any]) -> bytes:
    """X: the operands' root, which the window's salt is released against: SHA-512(`TAG` ‖ u64be(len program) ‖ program ‖
    for each value in name order: u64be(len name) ‖ name ‖ domain ‖ root), strings as UTF-8, domain and root as bytes."""
    x = CC.values_root(b)
    if x is None:
        raise ValueError("the operands are not {program, values} with entries naming their value, domain and root")
    return x


def held(window: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The registration window's operand entries by value, the verifier's pins of the tiles' reads: refused (ValueError)
    unless the window names them, each names its value and a root, and its domain is the one its program, value, rows and
    the window's context make."""
    try:
        b, context = window[KEY], window["context"]
        program, values = b["program"], dict(b["values"])
        if set(b) != {"program", "values"} or not values:
            raise ValueError(f"{KEY} is {{program, values}}, at least one value")
        for v, e in values.items():
            want = G.domain(program, v, e["words"], e["leaves"], e.get(G.WIDTH_KEY, G.WORD_BITS), window=context)
            if e.get("value") != v or not isinstance(e.get("root"), str) or e.get("domain") != want.domain_id.hex():
                raise ValueError(f"the entry of {v!r} names another value, no root, or a domain other than the one "
                                 f"{program!r}, its rows and the window's context make")
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"the window's {KEY} is {{program, values: {{value: entry}}}}: {type(e).__name__}: {e}") from e
    return {str(v): dict(e) for v, e in values.items()}


def stated(window: Mapping[str, Any]) -> dict[str, Any] | None:
    """What a record reports of a registration window's operands (`held`): X, each value's root, and `GAPS`; ``held``
    false and why, for operands `held` refuses; None for a window that names none."""
    if KEY not in window:
        return None
    try:
        values = held(window)
    except ValueError as e:
        return {"held": False, "why": str(e), "gaps": list(GAPS)}
    return {"held": True, "X": root(window[KEY]).hex(), "roots": {v: e["root"] for v, e in sorted(values.items())},
            "gaps": list(GAPS)}


__all__ = ["GAPS", "KEY", "TAG", "block", "commit", "held", "root", "stated"]
