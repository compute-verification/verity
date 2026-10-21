"""Produced values: a value one statement of a session outputs and another reads, committed after the draw, each read
opened against the one root it was output under (PRF-134; a served H1 leaf proved as its lanes and a combine,
`pouw.circuit.h1lane`).

A *produced value* is a field of a template's result that another unit of the same Program reads: H1's lane partials,
each lane unit's ``partial``, which the combine unit reads as its parameter ``p``; or the whole result of a template that
returns no record, named by the template's id: H1's leaf seeds, ``PouwUnitSeeds_v2``, which every lane reads as its
``seeds``. Its row ``r`` is the field of the ``r``-th unit (in population order) whose template returns it; the rows are committed as a registered value's
(`registered.Registered.commit`, in the row format of the ports that hold them), after the draw, since they are outputs
of the drawn units' statements, whose inputs were committed before it. They stay hidden: the statements commit them under
the produced tree's salts, and only the units' check bits leave.

* **The domain** (`domain`) is a registered value's (`registered.domain`) with ``"produced": D`` in its binding, D the
  digest of the registration record whose draw the values follow (`registration.digest`): a produced tree opens under no
  other registration's domain, so a leaf's partials are refused in any other leaf's, window's or prover's session, and a
  registered value's root (whose binding never has ``produced``) opens under none.
* **The record** (`record`, `FORMAT`), fixed after the draw and before any session that reads the values:
  ``{"format", "registration": D, "draw": the draw's digest, "produced": [{"value", "root", "leaves", "words", "domain"}
  ...]}`` (``"word_bits"`` beside an entry whose leaves are not 16 bits). `check` holds it to the verifier's registration
  digest, its draw's and its own entries (`entries`, its shape of each value from its own Program, `values`): its codes
  ``P-keys``, ``P-format``, ``P-registration``, ``P-draw``, ``P-produced``, ``P-values`` and, per value, ``P-root``,
  ``P-leaves``, ``P-words``, ``P-word_bits``, ``P-domain``.
* **The reads** (`reads`) are the verifier's own, from its Program's cut between units: the producing unit's result field
  is its output row, which the statement opens at row ``r`` (an output row's registered port, §16.13); a consuming unit's
  parameter, or element ``name[i]`` of an array parameter, whose leaves are exactly row ``r``'s output leaves reads row
  ``r``. A read of part of a row, or of leaves of two rows, is refused (`ValueError`): a read at a position the verifier
  doesn't derive is no public read. Swapping two rows of the tree moves no read, so each producer's output no longer
  opens at its row; a consumer that reads every row (H1's combine, an XOR) still opens them all, so the producers'
  openings are what refuse a swap.
* **The opened roots** (`opened`): the roots the verifiers of record report opening the sessions' reads of the produced
  values against (`audit.session_registered`, unfiltered) must each be the record's: ``P-opened`` otherwise, a produced
  value not opened included.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from verity.ml.circuits.codec import descriptor_id
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.circuits.partition_object import TemplatePartition
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Record

from . import partition as P
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


def _row_leaves(t: Any, words: int, schema: str) -> tuple[int, int]:
    """(word_bits, leaves of ``t`` in one row of ``words``) for a field of type ``t``: `registered.width`'s and
    `registered.row_leaves`' rules on a result field."""
    w = t.homogeneous_width()
    if w is None or not 0 < w <= max(G.WIDTHS):
        raise ValueError(f"a produced value's leaves are one width of at most {max(G.WIDTHS)} bits, not {sorted(t.width_set())}")
    b = 16 * -(-w // 16)
    if G._schema(schema) == G.SCHEMA:
        return b, int(words)
    if words * b % w:
        raise ValueError(f"a {schema} row of {words} {b}-bit leaves is no whole number of {w}-bit leaves")
    return b, words * b // w


def _producers(program: Program, tp: TemplatePartition, values: Mapping[str, int],
               schemas: Mapping[str, str]) -> tuple[dict[str, list], dict[str, int]]:
    """Per value, ``(unit, its field's output gates)`` of each producing unit in population order; per value its
    `word_bits`."""
    nodes = program.root.body.nodes
    rows: dict[str, list] = {v: [] for v in values}
    bits: dict[str, int] = {}
    for u in range(tp.population):
        k, _m = tp.locate(u)
        ret = nodes[k].fn.signature()[1]
        fields = ret.fields if isinstance(ret, Record) else ((descriptor_id(nodes[k].fn), ret),)
        outs, at = tp.outputs(u), 0
        for name, t in fields:
            gates, at = outs[at:at + t.leaves], at + t.leaves
            if name not in values:
                continue
            b, per = _row_leaves(t, values[name], schemas.get(name, G.SCHEMA))
            if per != t.leaves:
                raise ValueError(f"unit {u}: its {name!r} is {t.leaves} leaves, not one row of {per}")
            if bits.setdefault(name, b) != b:
                raise ValueError(f"{name!r} is {bits[name]}-bit leaves in one template and {b}-bit in another")
            rows[name].append((u, gates))
    missing = [v for v in values if not rows[v]]
    if missing:
        raise ValueError(f"no unit returns {missing}: a produced value is a template's result field, or the whole result of "
                         f"a template that returns no record, named by the template's id")
    return rows, bits


def entries(program: Program, query: Mapping[str, Any], values: Mapping[str, int], registration: str,
            schemas: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """The verifier's own ``{value: {"leaves", "words", "word_bits", "domain"}}`` of the produced values ``values``
    (``{result field: words}``), from its own Program and query: ``leaves`` the units that return the field."""
    tp = P.evaluate(program, query)
    if not isinstance(tp, TemplatePartition):
        raise ValueError("a produced value is a template query's instances' output")
    rows, bits = _producers(program, tp, values, dict(schemas or {}))
    prog = P.program_digest(program)
    return {v: {"leaves": len(rows[v]), "words": int(w), "word_bits": bits[v],
                "domain": domain(prog, v, w, len(rows[v]), bits[v], registration).domain_id.hex()} for v, w in values.items()}


def reads(program: Program, query: Mapping[str, Any], values: Mapping[str, int],
          schemas: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Every unit's reads of the produced values, `registered.reads`' form ``{key: {"value", "rows"}}`` (``rows`` one entry
    per unit, None where the unit doesn't read it): each producing unit's field ``value`` under the key ``value`` (its
    output row), and each consuming unit's parameter ``name``, or element ``name[i]`` of an array parameter, whose leaves
    are exactly a producing unit's field (the row that unit produced)."""
    tp = P.evaluate(program, query)
    if not isinstance(tp, TemplatePartition):
        raise ValueError("a produced value is a template query's instances' output")
    rows, _bits = _producers(program, tp, values, dict(schemas or {}))
    owner: dict[int, tuple[str, int, int]] = {}
    out: dict[str, dict[str, Any]] = {}
    for v, rs in rows.items():
        entry_ = out.setdefault(v, {"value": v, "rows": [None] * tp.population})
        for r, (u, gates) in enumerate(rs):
            entry_["rows"][u] = r
            for j, g in enumerate(gates):
                owner[g] = (v, r, j)
    nodes = program.root.body.nodes

    def row_of(gates: list[int]) -> tuple[str, int] | None:
        hits = [owner.get(g) for g in gates]
        if not any(hits):
            return None
        v, r, _j = hits[0] if hits[0] is not None else (None, -1, -1)
        if v is None or hits != [(v, r, j) for j in range(len(gates))] or len(gates) != len(rows[v][r][1]):
            raise ValueError("a read of part of a produced row, or of leaves of two rows: no public read")
        return v, r

    for u in range(tp.population):
        k, _m = tp.locate(u)
        gates, at = tp.inputs(u), 0
        for name, t in nodes[k].fn.signature()[0]:
            s, at = gates[at:at + t.leaves], at + t.leaves
            if not any(g in owner for g in s):
                continue
            try:
                got = {name: row_of(s)}
            except ValueError:
                if not (isinstance(t, Array) and t.n > 1):
                    raise ValueError(f"unit {u}: parameter {name!r} is not one whole produced row, nor an array of them") from None
                w = t.elem.leaves
                got = {f"{name}[{i}]": row_of(s[i * w:(i + 1) * w]) for i in range(t.n)}
            for key, vr in got.items():
                if vr is None:
                    raise ValueError(f"unit {u}: {key!r} of an array of produced rows reads none")
                e = out.setdefault(key, {"value": vr[0], "rows": [None] * tp.population})
                if e["value"] != vr[0] or key in values:
                    raise ValueError(f"{key!r} reads {vr[0]!r} and {e['value']!r}: one key, one read")
                e["rows"][u] = vr[1]
    return out


def reads_ports(program: Program, query: Mapping[str, Any], values: Mapping[str, int], roots: Mapping[str, str],
                registration: str, ports: Mapping[str, str] | None = None,
                schemas: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """The Lean verifier's ``--registered`` ports of the produced values (`registered.reads_file`'s ``ports`` form): per
    read (`reads`) its port (``ports`` maps a read's key to its circuit port when they differ: a producer's field to its
    output row, ``o<j>``), the value, its root from ``roots`` (the produced record's, which `check` accepted), rows, width,
    domain and row format, and each unit's row."""
    mine = entries(program, query, values, registration, schemas)
    out: dict[str, dict[str, Any]] = {}
    for key, r in reads(program, query, values, schemas).items():
        port = (ports or {}).get(key, key)
        if port in out:
            raise ValueError(f"port {port!r} names two reads: each read is its own port")
        e = mine[r["value"]]
        out[port] = {"value": r["value"], "root": roots[r["value"]], "leaves": e["leaves"], "words": e["words"],
                     "word_bits": e["word_bits"], "domain": e["domain"],
                     **G.schema_entry((schemas or {}).get(r["value"], G.SCHEMA)), "reads": list(r["rows"])}
    return out


def check(rec: Any, registration: str, draw: str, mine: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """The produced record's failure codes against the verifier's registration digest, its draw's digest and its own
    entries (`entries`' ``{value: {"leaves", "words", "word_bits", "domain"}}``): empty when it holds."""
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
    for v, want in mine.items():
        e = by.get(v)
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


__all__ = ["FORMAT", "KEYS", "PRODUCED_KEY", "check", "commit", "domain", "entries", "opened", "reads", "reads_ports", "record"]
