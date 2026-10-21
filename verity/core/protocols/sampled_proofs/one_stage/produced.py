"""Produced values: a value committed after the draw that several statements of a session read, each read opened
against the one root the record names (PRF-134: a served H1 leaf proved as its 32 lanes and a combine, which share the
leaf's lane partials, `pouw.leaf_lanes`).

A *produced value* is rows the prover computes after the draw from what it committed before it (H1's partials: lane l's
share of the leaf's receipt, row l), committed as a registered value's rows are (`registered.Registered.commit`, in the
row format of the ports that read them) under its own domain. It is never a statement's output: outputs are public and at
most 16 bits, and a partial is 8,320 bits at K = 4096. It stays hidden: the statements read it as inputs under the tree's
hiding salts, and only their check bits leave.

* **The domain** (`domain`) is a registered value's (`registered.domain`) with ``"produced": D`` in its binding, D the
  digest of the registration record whose draw the values follow (`registration.digest`): a produced tree opens under no
  other registration's domain, so a leaf's partials are refused in any other leaf's, window's or prover's session, and a
  registered value's root (whose binding never has ``produced``) opens under none.
* **The record** (`record`, `FORMAT`), fixed after the draw and before any session that reads the values:
  ``{"format", "registration": D, "draw": the draw's digest, "produced": [{"value", "root", "leaves", "words", "domain"}
  ...]}`` (``"word_bits"`` beside an entry whose leaves are not 16 bits). `check` holds it to the verifier's registration
  digest, its draw's and its own entries (`entry`, each value's shape as the protocol states it): its codes ``P-keys``,
  ``P-format``, ``P-registration``, ``P-draw``, ``P-produced``, ``P-values`` and, per value, ``P-root``, ``P-leaves``,
  ``P-words``, ``P-word_bits``, ``P-domain``.
* **The reads** are the protocol's (H1's: `pouw.leaf_lanes.reads`), each a ``--registered`` port (`port`) at the row the
  verifier derives: a read at a position the verifier doesn't derive is no public read. Rows swapped in the tree open at
  no read's position under the root, and a row of another tree opens under no root the record names.
* **The opened roots** (`opened`): the roots the verifiers of record report opening the sessions' reads of the produced
  values against (`audit.session_registered`, unfiltered) must each be the record's: ``P-opened`` otherwise, a produced
  value not opened included.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from verity.ml.commitments.merkle import CommitmentDomain

from . import registered as G

FORMAT = "verity/registration/produced/v0"
KEYS = ("draw", "format", "produced", "registration")
PRODUCED_KEY = G.PRODUCED_KEY
_HEX = re.compile(r"[0-9a-f]+")


def _hex64(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 128 and bool(_HEX.fullmatch(s))


def domain(program: str, value: str, words: int, leaves: int, word_bits: int, registration: str) -> CommitmentDomain:
    """``registered.domain``'s, with ``"produced": registration`` (the registration's digest, lowercase hex of 64 bytes)
    in its binding."""
    return G.domain(program, value, words, leaves, word_bits, produced=registration)


def commit(program: str, value: str, rows: Sequence[Sequence[int]], registration: str, salts: Sequence[bytes] | None = None,
           word_bits: int = G.WORD_BITS, schema: str = G.SCHEMA) -> G.Registered:
    """The prover's produced tree: ``rows`` committed as a registered value's under `domain`, its `entry()` the record's
    and its `statement` the opened reads."""
    return G.Registered.commit(program, value, rows, salts, word_bits, schema, produced=registration)


def record(registration: str, draw: str, produced: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"format": FORMAT, "registration": registration, "draw": draw, "produced": [dict(e) for e in produced]}


def entry(program: str, value: str, leaves: int, words: int, registration: str,
          word_bits: int = G.WORD_BITS) -> dict[str, Any]:
    """The verifier's own ``{"leaves", "words", "word_bits", "domain"}`` of the produced value ``value``, ``leaves`` rows
    of ``words`` leaves of ``word_bits``, as the protocol states its shape: what `check` holds the record's entry to."""
    return {"leaves": int(leaves), "words": int(words), "word_bits": int(word_bits),
            "domain": domain(program, value, words, leaves, word_bits, registration).domain_id.hex()}


def port(value: str, e: Mapping[str, Any], root: str, rows: Sequence[int | None], schema: str = G.SCHEMA) -> dict[str, Any]:
    """The Lean verifier's ``--registered`` port (`registered.reads_file`'s form) reading ``value`` at ``rows`` (one per
    unit, None where the unit doesn't read it): ``e`` its entry (`entry`, or a registered value's from the accepted
    registration) and ``root`` the accepted record's."""
    return {"value": value, "root": root, "leaves": int(e["leaves"]), "words": int(e["words"]),
            "word_bits": int(e.get("word_bits", G.WORD_BITS)), "domain": e["domain"], **G.schema_entry(schema),
            "reads": list(rows)}


def check(rec: Any, registration: str, draw: str, mine: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """The produced record's failure codes against the verifier's registration digest, its draw's digest and its own
    entries (`entry`'s ``{value: {"leaves", "words", "word_bits", "domain"}}``): empty when it holds."""
    if not isinstance(rec, dict) or sorted(rec) != sorted(KEYS):
        return ("P-keys",)
    codes: list[str] = []

    def add(c: str) -> None:
        if c not in codes:
            codes.append(c)

    if rec["format"] != FORMAT:
        add("P-format")
    if rec["registration"] != registration:
        add("P-registration")
    if rec["draw"] != draw:
        add("P-draw")
    got = rec["produced"]
    width_keys = sorted((*G.ENTRY_KEYS, G.WIDTH_KEY))
    if not isinstance(got, list) or not all(isinstance(e, dict) and sorted(e) in (sorted(G.ENTRY_KEYS), width_keys) for e in got):
        add("P-produced")
        return tuple(codes)
    names = [e["value"] for e in got]
    if not all(isinstance(v, str) for v in names) or len(set(names)) != len(names) or set(names) != set(mine):
        add("P-values")
    by = {e["value"]: e for e in got if isinstance(e["value"], str)}
    for v in sorted(mine):
        want, e = mine[v], by.get(v)
        if e is None:
            continue
        if not _hex64(e["root"]):
            add("P-root")
        if not (type(e["leaves"]) is int and e["leaves"] == want["leaves"]):
            add("P-leaves")
        if not (type(e["words"]) is int and e["words"] == want["words"]):
            add("P-words")
        if e["domain"] != want["domain"]:
            add("P-domain")
        b = e.get(G.WIDTH_KEY)
        if (G.WIDTH_KEY not in e and want["word_bits"] != G.WORD_BITS) or (
                G.WIDTH_KEY in e and not (type(b) is int and b == want["word_bits"] and b != G.WORD_BITS)):
            add("P-word_bits")
    return tuple(codes)


def opened(rec: Mapping[str, Any], session_registered: Mapping[str, str | None] | None) -> tuple[str, ...]:
    """``("P-opened",)`` unless each produced value of the accepted record ``rec`` was opened against its root: the
    ``{value: root}`` the verifiers of record report (`audit.session_registered` without a registration, a value two of
    them report under different roots None), at every produced value."""
    want = {e["value"]: e["root"] for e in rec["produced"]}
    got = dict(session_registered or {})
    return () if all(got.get(v) == r for v, r in want.items()) else ("P-opened",)


__all__ = ["FORMAT", "KEYS", "PRODUCED_KEY", "check", "commit", "domain", "entry", "opened", "port", "record"]
