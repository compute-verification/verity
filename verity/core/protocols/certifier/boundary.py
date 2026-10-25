"""The run's boundary in a window's circuit: which units' rows hold each step's egress and ingress (`NetTiming.Layout`,
`Security/Definitions/NetworkCertifier/SampledAudit.lean`), as a window's circuit record declares them, and the
verifier's check, before it draws, that the layout fits the circuit (`NetTiming.Layout.Fits`, the half of the sampled
audit's scope `NetTiming.ScopeAcc` that isn't sampling's). The record is ``circuit.json``, `verity/window-circuit/v1`
(`verity.ml.windows.interface.circuit_record`), whose SHA-512 the registration binds, so the layout is public and fixed
before any coin.

**The record's ``boundary``**, one template a side, whose units are its link-windows' buckets:

    {"format": "verity/boundary/v1",
     "egress":  {"template": "<id>", "frame": "o0", "presence": "o1", "links": L, "buckets": T, "slots": R, "frame_bytes": B},
     "ingress": {"template": "<id>", "frame": "<parameter>", "links": LI, "buckets": T, "slots": RI, "frame_bytes": BI}}

Unit ℓ·T + t of a side's template, in its units' order (the partition's), is bucket t of link ℓ: the template has exactly
links·T units. Step t is bucket t of the window on both sides, so the sides share T, the horizon k. An integration writes
the field with `Boundary.record`, and the verifier reads it with `parse`.

* **Egress** (N2's encoder, `note:network-accounting/20261025T0720Z-draft-n2-egress-encoder`): ``frame`` is an output
  row of R·B bytes, slot σ's cell at bytes σ·B … (σ + 1)·B − 1, and row bit 8k + i is bit i of byte k, the order of the
  warden's registered row (`warden_rows`). ``presence`` is an output row whose bit 0 is every slot's presence gate: the
  warden's egress holds a cell in every slot. So `Layout.egress t ℓ σ` is (bit 0 of unit ℓ·T + t's presence row, bits
  8σB … 8(σ + 1)B − 1 of its frame row).
* **Ingress**: ``frame`` is a parameter row of RI·BI bytes, the warden's ingress row t of link ℓ, which the unit reads as
  its registered input port. `Layout.ingress t` is the ingress links' frame rows at bucket t, in link order, each in row
  order.

Every bucket's gates are the same whatever the load, so no arrival step, prompt length or occupancy is public: a
request's token ids are values the window's units parse from the cells, like every other value (top's gate-0 ruling, 25
Oct: arrival steps and prompt lengths stay private, the certifiers' observations are witness). The I/O tie reads row t of
link ℓ at unit `Boundary.unit` (``reads_from`` of `warden_rows.bucket_port` is link ℓ's bucket 0).

**The check** (`refusal`), by `Layout.Fits`' conjuncts over the horizon: every egress gate is one the circuit returns
(the frame and presence rows are output rows of the egress template, and every unit's output rows are returned and
committed); every ingress gate is an input of the circuit
(the frame is a parameter row of the ingress template that no wire and no row table fills); and no egress gate of a step
depends on an ingress gate of a later step. Dependence is taken at the record's grain: each output row of a unit depends
on each of its input rows, a wired input row is its writer's output row (`Member.wired`), and a row table's rows are
inputs, which no unit of the record writes. That dependence contains `Depends`, so a record the check accepts fits
(`prog_fits` is `Layout.Fits` itself on a flattened program, and the tests hold `refusal` to it); a record that fits only
at the gate grain is refused. `Fits` stays structural (top's ruling, 25 Oct: semantic Fits is dropped): a served window
fits in the step-indexed scan's form (`ingress_scan`, note:network-accounting/20261025T0830Z-report-net68-scan-router),
whose bucket t unit reads bucket t's ingress row and the state bucket t − 1 carries, and whose Programs start at fixed
boundaries and read their slots there, so an egress bucket reaches ingress only through earlier buckets' state, whatever
the arrival steps and prompt lengths. A window in any other form, such as an egress bucket that reads every request's
ids, is refused: the verifier gains no form ahead of its proof. The record must also name the boundary's rows, and no
others, as its ``external`` ports.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FORMAT = "verity/boundary/v1"
#: the circuit record the boundary sits in (`verity.ml.windows.interface.CIRCUIT`)
CIRCUIT = "verity/window-circuit/v1"
_COUNTS = ("links", "buckets", "slots", "frame_bytes")
FIELDS = {"egress": ("template", "frame", "presence", *_COUNTS), "ingress": ("template", "frame", *_COUNTS)}


@dataclass(frozen=True)
class Side:
    """One side of the boundary: its template, its frame row, its egress presence row, and its schedule's counts."""

    template: str
    frame: str
    links: int
    buckets: int
    slots: int
    frame_bytes: int
    presence: str | None = None

    @property
    def words(self) -> int:
        """The frame row's u16 words: the bucket's R·B bytes."""
        return self.slots * self.frame_bytes // 2

    @property
    def units(self) -> int:
        return self.links * self.buckets

    def ports(self) -> tuple[str, ...]:
        return tuple(f"{self.template}/{p}" for p in (self.frame, self.presence) if p is not None)


@dataclass(frozen=True)
class Boundary:
    """The record's boundary: its egress side and, when the window has ingress, its ingress side."""

    egress: Side
    ingress: Side | None = None

    @property
    def horizon(self) -> int:
        """k: the steps the layout covers, one a bucket."""
        return self.egress.buckets

    def sides(self) -> tuple[tuple[str, Side], ...]:
        return (("egress", self.egress),) + ((("ingress", self.ingress),) if self.ingress else ())

    def ports(self) -> set[str]:
        return {p for _, s in self.sides() for p in s.ports()}

    def record(self) -> dict[str, Any]:
        """The circuit record's ``boundary`` field, which `parse` reads back: what an integration writes."""
        return {"format": FORMAT, **{name: {k: getattr(s, k) for k in FIELDS[name]} for name, s in self.sides()}}

    def unit(self, doc: Mapping[str, Any], side: str, link: int, t: int) -> int:
        """The global unit (the population's index) that holds bucket t of link ``link`` of ``side``."""
        s = dict(self.sides())[side]
        if not (0 <= link < s.links and 0 <= t < s.buckets):
            raise ValueError(f"{side} has links [0, {s.links}) and buckets [0, {s.buckets}), not ({link}, {t})")
        return _template_units(doc)[s.template][0] + link * s.buckets + t


def _side(name: str, doc: Any) -> Side:
    if not isinstance(doc, Mapping) or set(doc) != set(FIELDS[name]):
        raise ValueError(f"boundary {name}: the fields {list(FIELDS[name])}, not {doc!r}")
    if any(not isinstance(doc[k], str) or not doc[k] for k in FIELDS[name] if k not in _COUNTS):
        raise ValueError(f"boundary {name}: its template and rows are named by nonempty strings, not {doc!r}")
    if any(type(doc[k]) is not int or doc[k] < 1 for k in _COUNTS):
        raise ValueError(f"boundary {name}: {list(_COUNTS)} are positive integers, not {[doc[k] for k in _COUNTS]}")
    if doc["frame_bytes"] % 2:
        raise ValueError(f"boundary {name}: a cell is whole u16 words, not {doc['frame_bytes']} bytes")
    if doc.get("presence") == doc["frame"]:
        raise ValueError(f"boundary {name}: the presence row is not the frame row {doc['frame']!r}")
    return Side(**{k: doc[k] for k in FIELDS[name]})


def parse(doc: Mapping[str, Any]) -> Boundary | None:
    """The circuit record's boundary, None when it declares none; ValueError for a malformed one."""
    b = doc.get("boundary")
    if b is None:
        return None
    if not isinstance(b, Mapping) or b.get("format") != FORMAT or not {"format", "egress"} <= set(b) <= {"format", *FIELDS}:
        raise ValueError(f"a boundary is {FORMAT} with an egress side and optionally an ingress side, not {b!r}")
    out = Boundary(_side("egress", b["egress"]), _side("ingress", b["ingress"]) if "ingress" in b else None)
    if out.ingress and out.ingress.buckets != out.egress.buckets:
        raise ValueError(f"boundary: one step a bucket on both sides, so one T, not egress {out.egress.buckets} and ingress "
                         f"{out.ingress.buckets}")
    return out


def _template_units(doc: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """Each template's global units ``[lo, lo + n)``: the templates' units consecutive from 0 in the record's order, each
    template's its members' consecutive chunks."""
    out, at = {}, 0
    for tid, n in doc["templates"]:
        lo = at
        for m in sorted((m for m in doc["members"] if m["template"] == tid), key=lambda m: m["base"]):
            if m["base"] != at:
                raise ValueError(f"{tid}: a member's units start at {m['base']}, not {at}: the templates' units are "
                                 "consecutive in their order, and a template's members' too")
            at += m["n"]
        if at - lo != n:
            raise ValueError(f"{tid}: its members hold {at - lo} units, not its {n}")
        out[tid] = (lo, n)
    return out


def _readers(doc: Mapping[str, Any], total: int) -> list[list[int]]:
    """Each global unit's readers: the units one of whose input rows is a wire from that unit's output row."""
    readers: list[list[int]] = [[] for _ in range(total)]
    for m in doc["members"]:
        for _port, _out, writers in m["wired"]:
            if len(writers) != m["n"] or any(type(w) is not int or not 0 <= w < total for w in writers):
                raise ValueError(f"{m['name'] or m['template']}: a wire is one writer in [0, {total}) a unit")
            for i, w in enumerate(writers):
                readers[w].append(m["base"] + i)
    return readers


def _latest(sources: Sequence[Sequence[int]], readers: Sequence[Sequence[int]]) -> tuple[list[int], list[int]]:
    """For each node, the latest step of a source it depends on (-1: none) and that source: ``sources[t]`` are step t's,
    and a node depends on each of its writers' sources (`readers`, writer to reader)."""
    latest, src = [-1] * len(readers), [-1] * len(readers)
    for t in range(len(sources) - 1, -1, -1):
        for s in sources[t]:
            if latest[s] >= 0:
                continue
            latest[s], src[s] = t, s
            todo = deque([s])
            while todo:
                for r in readers[todo.popleft()]:
                    if latest[r] < 0:
                        latest[r], src[r] = t, s
                        todo.append(r)
    return latest, src


def refusal(doc: Mapping[str, Any]) -> str | None:
    """None when the circuit record's boundary fits its circuit over its horizon (`NetTiming.Layout.Fits`, at the record's
    grain: module docstring), else why not, by the conjunct it fails."""
    if doc.get("format") != CIRCUIT:
        return f"boundary: a circuit record is {CIRCUIT}, not {doc.get('format')!r}"
    try:
        b = parse(doc)
        units = _template_units(doc)
    except (ValueError, KeyError, TypeError) as e:
        return f"boundary: {e}"
    if b is None:
        return "boundary: the circuit record declares none (NetTiming.Layout), so the network audit has no layout to read"
    if set(doc.get("external", ())) != b.ports():
        return (f"boundary: the record's external ports {sorted(doc.get('external', ()))} are not the boundary's rows "
                f"{sorted(b.ports())}: I/O the layout doesn't hold is I/O no audit reads")
    members = {tid: [m for m in doc["members"] if m["template"] == tid] for tid in units}
    for name, s in b.sides():
        if s.template not in units:
            return f"boundary {name}: {s.template} is no template of the circuit"
        if units[s.template][1] != s.units:
            return (f"boundary {name}: {s.template} has {units[s.template][1]} units, not links·buckets = "
                    f"{s.links}·{s.buckets}")
    for m in members[b.egress.template]:
        outs = dict(map(tuple, m["outs"]))
        if outs.get(b.egress.frame) != b.egress.words:
            return (f"Layout.Fits, egress gates are returned: {b.egress.template}'s {b.egress.frame!r} is not an output row "
                    f"of {b.egress.words} words (R·B = {b.egress.slots}·{b.egress.frame_bytes} bytes); its outputs are {outs}")
        if not outs.get(b.egress.presence):
            return (f"Layout.Fits, egress gates are returned: {b.egress.template}'s presence {b.egress.presence!r} is not an "
                    f"output row; its outputs are {outs}")
    if b.ingress:
        s = b.ingress
        for m in members[s.template]:
            rows = dict(map(tuple, m["rows"]))
            if rows.get(s.frame) != s.words:
                return (f"Layout.Fits, ingress gates are inputs: {s.template}'s {s.frame!r} is not a parameter row of "
                        f"{s.words} words (R·B = {s.slots}·{s.frame_bytes} bytes); its rows are {rows}")
            if m["tables"] or m["links"]:
                return (f"Layout.Fits, ingress gates are inputs: {m['name'] or s.template} reads row tables, and an ingress "
                        "frame is its own unit's row")
            if s.frame in {p for p, _o, _w in m["wired"]}:
                return (f"Layout.Fits, ingress gates are inputs: {m['name'] or s.template}'s {s.frame!r} is wired from "
                        "another unit's output, so its gates are that unit's, not inputs")
    total = sum(n for _lo, n in units.values())
    try:
        readers = _readers(doc, total)
    except (ValueError, KeyError, TypeError) as e:
        return f"boundary: {e}"
    T = b.horizon
    sources: list[list[int]] = [[] for _ in range(T)]
    if b.ingress:
        lo = units[b.ingress.template][0]
        for ell in range(b.ingress.links):
            for t in range(T):
                sources[t].append(lo + ell * T + t)
    latest, src = _latest(sources, readers)
    lo_e, lo_i = units[b.egress.template][0], units[b.ingress.template][0] if b.ingress else 0
    for ell in range(b.egress.links):
        for t in range(T):
            u = lo_e + ell * T + t
            if latest[u] > t:
                ell_i = (src[u] - lo_i) // T
                return (f"Layout.Fits, causal: egress bucket {t} of link {ell} (unit {u}) depends on ingress bucket "
                        f"{latest[u]} of link {ell_i} (unit {src[u]}), a later step, through the record's wires; "
                        "a window fits in the scan's form, its egress reading ingress through earlier buckets' "
                        "state (ingress_scan)")
    return None


def prog_fits(gates: Sequence[tuple[str, Sequence[int]]], outs: Sequence[int],
              egress: Callable[[int, int, int], tuple[int, Sequence[int]]], ingress: Callable[[int], Sequence[int]],
              L: int, S: int, k: int) -> bool:
    """`NetTiming.Layout.Fits lay C k` itself on a flattened program `C` (`Claims.Sampling.Prog`: ``gates`` each ``(kind,
    reads)``, kind one of ``input``, ``const``, ``not``, ``and``, ``xor``; ``outs`` its returned gates) and a layout
    (``egress(t, ℓ, σ)``, a slot's presence gate and frame gates, and ``ingress(t)``): before ``k``, every slot gate is in
    ``outs``, every ingress gate is an input, and no slot gate of a step reaches (`Depends`, reflexive) an ingress gate of
    a later step."""
    returned = set(outs)
    slot = {t: [g for ell in range(L) for s in range(S) for g in (egress(t, ell, s)[0], *egress(t, ell, s)[1])]
            for t in range(k)}
    if any(g not in returned for t in range(k) for g in slot[t]):
        return False
    if any(not 0 <= g < len(gates) or gates[g][0] != "input" for t in range(k) for g in ingress(t)):
        return False
    readers: list[list[int]] = [[] for _ in gates]
    for b, (_kind, reads) in enumerate(gates):
        for a in reads:
            if 0 <= a < len(gates):
                readers[a].append(b)
    latest, _ = _latest([list(ingress(t)) for t in range(k)], readers)
    return all(latest[g] <= t for t in range(k) for g in slot[t] if 0 <= g < len(gates))
