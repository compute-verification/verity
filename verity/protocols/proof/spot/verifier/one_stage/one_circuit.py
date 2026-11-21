"""The one circuit at the served verifier's door: C_S, which the door derives from C (the registered Program), the
partition and the block's units (`restrict`), its Boolean block circuit, Lean's lowering of C_S (`lower`), and the holes
it does not yet derive, each refusing by name (`NotDerived`)."""

from __future__ import annotations

import bisect
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verity.circuits import edges as EG
from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import DecodedProgram, definition_digest, descriptor_id, lift_shape
from verity.circuits.program import input_id, recut_gates
from verity.circuits.types import RecordType, Tuple

from . import lean_child as LC

PREFIX = "one-circuit: not derived yet: "

RESTRICT = "restrict : C -> Partition -> Block -> Circuit (C_S, the lowering of the block's Definition in C)"

LOWER = "lower : C_S of a partition that is no template query (Q_template_instance, Q_template_instances) -> its Boolean block circuit"

#: Lean's lowering of C_S (`Flock.Lower.lower`, `Flock.Compose.compose`), a Lake target of Flock's verifier.
LOWER_EXE = "flock-lower-check"
#: The prefix of every refusal of `admitted`: `Spot.OneStageZk.NOT_ADMITTED`, word for word.
NOT_ADMITTED = "lower: not the one form the door lowers (Spot.OneStageZk.admitted)"

#: The templated statement's layout mode (`Flock.Tmpl`'s `iv/hybrid/T`) the door lowers C_S at.
LAYOUT = "iv/hybrid/4"

Refused = R.Refused


class NotDerived(ValueError):
    """A derivation from C and the partition that is not written yet (`Flock.OneCircuit.notDerived`)."""

    def __init__(self, what: str) -> None:
        super().__init__(PREFIX + what)


class Decoded:
    """C's bytes, decoded at most once (`restrict.program`) for the door's steps on that C."""

    def __init__(self, c: bytes) -> None:
        self.c, self._D = c, None

    def get(self) -> DecodedProgram:
        if self._D is None:
            self._D = R.program(self.c)
        return self._D


def block(c: bytes, partition: Mapping[str, Any], k: int, decoded: Decoded | None = None) -> list[tuple[int, int]]:
    """Proof block ``k``'s units, the ``k``-th pair of the partition object's one level of ``blocks``, which cover every unit
    of C's population exactly once (`partition_object.blocks`); `Refused` where the object is not valid, names no blocks or
    more than one level (`blocks-levels`), its blocks end short of or past the population, or has no block ``k``.
    ``decoded``: C's, when the caller holds it."""
    v = PO.validate(partition)
    if not v.ok:
        raise Refused("E_PARTITION", f"the partition object is not valid: {list(v.codes)}")
    try:
        bs = PO.blocks(partition)
    except ValueError as e:
        raise Refused("E_BLOCKS", str(e)) from e
    if bs is None:
        raise Refused("E_BLOCKS", "the partition object names no blocks")
    try:
        population = PO.TemplatePartition((decoded or Decoded(c)).get(), partition["query"]).population
    except PO.QueryInapplicable as e:
        raise Refused("E_QUERY", str(e)) from e
    if bs[-1][1] != population:
        raise Refused("E_BLOCKS", f"the partition's blocks end at {bs[-1][1]}, not at C's population {population}")
    if not (type(k) is int and 0 <= k < len(bs)):
        raise Refused("E_BLOCKS", f"no block {k!r} among the partition's {len(bs)}")
    return [bs[k]]


def restrict(c: bytes, partition: Mapping[str, Any], units: Sequence[tuple[int, int]],
             decoded: Decoded | None = None) -> bytes:
    """C_S (`OneStageZk.restrict`): the descriptor bytes of C compacted to the block of ``units`` (ascending ``[lo, hi)``
    ranges of the partition's units), Core's `Circuit.restrict` (`restrict.restrict`); `Refused` where C, the partition
    or the units give no block, and where C's root reads an input node recut, as the gate layout refuses it."""
    D = (decoded or Decoded(c)).get()
    if D.root.body.recut:
        raise Refused("E_FORM", recut_gates(D.root.body.recut))
    return R.restrict(c, partition, [tuple(u) for u in units], D=D).descriptor


def _listed(query: Mapping[str, Any]) -> list[Any] | None:
    """The templates a template query lists; None for another query."""
    params = query.get("params") or {}
    if query.get("name") == "Q_template_instance":
        return [params.get("template")]
    if query.get("name") == "Q_template_instances" and isinstance(params.get("templates"), list):
        return list(params["templates"])
    return None


def cuts(c: bytes, partition: Mapping[str, Any], decoded: Decoded | None = None) -> dict[str, dict[str, int]]:
    """in(S)'s row cuts (`Spot.OneStageZk.cutsOf`, `edges.io_cuts`): ``{template: {port: k}}`` on C and the partition's
    ranges; `NotDerived` (`LOWER`, as `lower`) before C is read where the partition is no template query, and at a scan
    node of C's root; `Refused` where the query doesn't apply to C or C's root reads a port in rows that no writer field
    divides, or where a template of C has no gate (`Partition.checkTemplates`, which Lean's runs only on a C with an
    edge: this refuses at least what Lean's does)."""
    if _listed(partition.get("query") or {}) is None:
        raise NotDerived(LOWER)
    D = (decoded or Decoded(c)).get()
    try:
        ranges = PO.TemplatePartition(D, partition["query"]).ranges
    except PO.QueryInapplicable as e:
        raise Refused("E_QUERY", str(e)) from e
    if t := next((t for k, t, _a, n in ranges if n and D.root.body.nodes[k].fn.gates == 0), None):
        raise Refused("E_QUERY", f"query-refused: template {t} has no gate")
    try:
        return EG.io_cuts(D.desc, ranges)
    except ValueError as e:
        if str(e).startswith(PREFIX):
            raise NotDerived(str(e)[len(PREFIX):]) from e
        raise Refused("E_LOWER", f"in(S)'s cuts: {e}") from e


def _not_admitted(clause: str, detail: str) -> Refused:
    return Refused("E_LOWER", f"{NOT_ADMITTED}, {clause}: {detail}")


def _plain(n: str) -> bool:
    return bool(n) and not any(c in ",:# \t\r\n" for c in n)


def admitted(D: DecodedProgram, partition: Mapping[str, Any],
             cut: Mapping[str, Mapping[str, int]] | None = None) -> tuple[str, list[tuple[str, int]], list[tuple[str, int]]]:
    """`Spot.OneStageZk.admitted`: C_S (decoded, ``D``) in the one form the door lowers, as (its template, its input rows,
    its return fields), each a ``(name, bits)``; `NotDerived` where the partition is no template query, else `Refused`
    under `NOT_ADMITTED` by the first clause that fails, with Lean's text."""
    listed = _listed(partition.get("query") or {})
    if listed is None:
        raise NotDerived(LOWER)
    called = sorted({descriptor_id(nd.fn) for nd in D.root.body.nodes if input_id(nd.fn.id) is None})
    if len(called) != 1:
        raise _not_admitted("one-template", f"C_S's root calls [{', '.join(called)}], and a block's units are instances "
                                            "of one template")
    key = called[0]
    if key not in listed:
        raise _not_admitted("listed", f"C_S's template {key} is not one the partition's query lists")
    rows, fields = template_form(D, key, (cut or {}).get(key, {}))
    return key, rows, fields


def _fields(t: Any) -> list[Any]:
    """`SecondStage.fields`: one per output port, a tuple's or a record's fields, else `t`."""
    if isinstance(t, RecordType):
        return [f for _n, f in t.fields]
    return list(t.fields) if isinstance(t, Tuple) else [t]


def template_form(D: DecodedProgram, key: str, ks: Mapping[str, int]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """`admitted`'s clauses from `listed` on, on template ``key`` of ``D`` at its cuts ``ks``: (its input rows, its return
    fields), or `Refused` by the first clause that fails. `leaves-are-bits` is held in Lean alone: it compares the
    template with its lifted Definition, which Python does not build, and `bit_type` of a type has its bits as leaves."""
    fn = D.defs.get(key)
    if fn is None:
        raise _not_admitted("listed", f"C_S's template {key} is no Definition of C_S")
    params, ret = fn.signature()
    rows = [r for name, bits in ((name, sum(t.leaf_widths())) for name, t in params)
            for r in ([(name, bits)] if ks.get(name, 1) <= 1 else
                      [(f"{name}#{j}", bits // ks[name]) for j in range(ks[name])])]
    fields = [(f"o{j}", sum(f.leaf_widths())) for j, f in enumerate(_fields(ret))]
    if bad := next((n for n, _ in params if not _plain(n)), None):
        raise _not_admitted("port-names", f"C_S's template {key}: parameter '{bad}' is empty or holds ',', ':', '#' or a space")
    names = [n for n, _ in rows + fields]
    if twice := next((n for n in names if names.count(n) > 1), None):
        raise _not_admitted("port-names", f"C_S's template {key}: port {twice} is named twice")
    try:
        wide = lift_shape(D.desc, key)[1]
    except ValueError as e:
        raise _not_admitted("on-bits", f"the lift to bits refuses C_S's template {key}") from e
    if wide:
        raise _not_admitted("on-bits", f"C_S's template {key} reaches {wide[0]}, whose leaves are not all one bit after "
                                       "the lift to bits")
    types = dict(params)
    for p, k in ks.items():
        if p not in types:
            raise _not_admitted("cut-rows", f"C_S's template {key}: a cut names {p}, which is no parameter")
        t = types[p]
        if k < 2 or t.leaves == 0 or t.homogeneous_width() is None or t.leaves % k:
            raise _not_admitted("cut-rows", f"C_S's template {key}: parameter {p} of {t.leaves} leaves cut {k} ways, not "
                                            "into equal runs of leaves of one width")
    if not fields:
        raise _not_admitted("whole-words", f"C_S's template {key} returns no field")
    for what, rs in (("input row", rows), ("return field", fields)):
        for name, bits in rs:
            if bits == 0 or bits % 16:
                raise _not_admitted("whole-words", f"C_S's template {key}: {what} {name} of {bits} bits is not a positive "
                                                   "whole number of u16 words")
    return rows, fields


def admitted_block(D: DecodedProgram, partition: Mapping[str, Any], units: Sequence[Sequence[int]], key: str) -> None:
    """`Spot.OneStageZk.admittedBlock`: the block of ``units`` (``[lo, hi)`` ranges) of C (decoded, ``D``), whose C_S
    `admitted` gave the template ``key``, `Refused` by the first clause that fails, in Lean's words: C's root reads no
    input node recut (`no-recut`), no unit is named twice and no two hold a gate in common (`disjoint-ranges`), every
    unit calls ``key`` (`one-callee`), and every unit's operand gates are outside the block (`operands-outside`)."""
    if D.root.body.recut:
        raise _not_admitted("no-recut", f"C's root reads input nodes {sorted(D.root.body.recut)} recut, in leaves "
                                              "of part of their width")
    named = sorted(u for lo, hi in units for u in range(lo, hi))
    for a, b in zip(named, named[1:]):
        if a == b:
            raise _not_admitted("disjoint-ranges", f"unit {a} is named twice in the block")
    tp = PO.TemplatePartition(D, partition["query"])
    us = [(u, *tp.gates(u)) for lo, hi in units for u in range(lo, hi)]
    by_lo = sorted(us, key=lambda x: x[1])
    for (u, _, hi), (v, lo, _) in zip(by_lo, by_lo[1:]):
        if lo < hi:
            raise _not_admitted("disjoint-ranges", f"units {u} and {v} of the block hold gates in common")
    nodes = D.root.body.nodes
    los = [lo for _, lo, _ in by_lo]
    for u, _, _ in us:
        k, _m = tp.locate(u)
        if (c := descriptor_id(nodes[k].fn)) != key:
            raise _not_admitted("one-callee", f"unit {u} calls {c}, not the block's template {key}")
        for g in tp.inputs(u):
            # the units' gate ranges are disjoint (checked above), so at most one holds g
            i = bisect.bisect_right(los, g) - 1
            held = by_lo[i][0] if i >= 0 and g < by_lo[i][2] else None
            if held is not None:
                raise _not_admitted("operands-outside", f"unit {u} reads gate {g}, which unit {held} of the block "
                                                        "computes: a unit reads only gates outside its block")


def formed_in(c: bytes | Decoded, partition: Mapping[str, Any], units: Sequence[Sequence[int]]) -> Callable[[str], None]:
    """`lower`'s ``formed`` for the block of ``units`` (``[lo, hi)`` ranges) of C, its bytes or `Decoded`:
    `admitted_block` at the template `admitted` gives."""
    decoded = c if isinstance(c, Decoded) else Decoded(c)
    return lambda key: admitted_block(decoded.get(), partition, units, key)


def NO_BLOCK_CHECK(key: str) -> None:
    """`lower`'s ``formed`` for a caller that lowers a block whatever its form in C (a census of what lowering
    refuses); never a verdict's."""


def lower(c_s: bytes, partition: Mapping[str, Any], exe: Callable[[], Path | str], layout: str = LAYOUT,
          cut: Mapping[str, Mapping[str, int]] | None = None, *, formed: Callable[[str], None]) -> bytes:
    """C_S's Boolean block circuit: Lean's lowering (`Flock.Lower.lower`, then `Flock.Compose.compose`),
    `flock-lower-check compose`, the executable ``exe()`` builds, on C_S refused unless `admitted`, at the rows and fields
    it gives: its input ports the template's parameters as bit rows, in order, each cut ``k`` ways into rows ``p#j``
    where ``cut`` (`cuts`, C's) gives the template's port ``p`` a ``k``, and one output port a return field, ``o{j}``,
    in u16 words, each a field's tree (`Flock.Registered.outReads`); META binds the partition, C's digest and the
    template's `definition_digest`. ``formed``: the door's check of the block in C at the template, after `admitted`
    (`formed_in`, or `NO_BLOCK_CHECK` by name). `NotDerived` where the partition is no template query, before C_S is read; `Refused` too where the
    lowering refuses."""
    if _listed(partition.get("query") or {}) is None:
        raise NotDerived(LOWER)
    D = R.program(c_s)
    key, rows, fields = admitted(D, partition, cut)
    formed(key)
    ins = ",".join(f"{name}:{bits}" for name, bits in rows)
    outs = ",".join(f"{name}:{bits // 16}" for name, bits in fields)
    digests = json.dumps({key: definition_digest(D.desc["definitions"][key], D.desc)})
    with tempfile.TemporaryDirectory(prefix="one-circuit-lower-") as d:
        src, dst = Path(d, "c_s.json"), Path(d, "circuit.txt")
        src.write_bytes(c_s)
        p = LC.run([str(exe()), "compose", str(src), key, f"unit/{key}", ins, outs, layout, partition["program"],
                    PO.digest(partition), digests, str(dst)], ok_statuses=LC.COMPOSE_EXITS)
        if p.returncode != 0:
            raise Refused("E_LOWER", f"{LOWER_EXE} compose ({p.returncode}): {(p.stderr or p.stdout)[-2000:]}")
        return dst.read_bytes()
