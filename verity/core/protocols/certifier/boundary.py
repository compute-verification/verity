"""The run's boundary in a window's circuit: which units' rows hold each step's egress and ingress
(`NetTiming.BoundaryGates`, `Security/Definitions/NetworkCertifier/SampledAudit.lean`), as a window's circuit record
declares them, and the verifier's check, before it draws, that the layout fits the circuit
(`NetTiming.BoundaryGates.Fits`, the half of the sampled audit's scope `NetTiming.ScopeAcc` that isn't sampling's). The
record is ``circuit.json``, `verity/window-circuit/v1` (`verity.ml.windows.interface.circuit_record`), whose SHA-512 the
registration binds, so the layout is public and fixed before any coin.

**The record's ``boundary``**, one template a side (and the ingress side's first bucket's), whose units are its
link-windows' buckets:

    {"format": "verity/boundary/v3",
     "egress":  {"template": "<id>", "frame": "<output>", "verdict": "<output>", "links": L, "buckets": T, "slots": R,
                 "frame_bytes": B},
     "ingress": {"template": "<id>", "frame": "<parameter>", "links": LI, "buckets": T, "slots": RI, "frame_bytes": BI,
                 "first": "<id>"}}

Unit ℓ·T + t of a side's template, in its units' order (the partition's), is bucket t of link ℓ: the template has exactly
links·T units. With ``first`` (optional, ingress only, T ≥ 2), unit ℓ of that template is bucket 0 of link ℓ and unit
ℓ·(T − 1) + t − 1 of ``template`` is bucket t ≥ 1, so the two have links and links·(T − 1) units, and both read the
frame row. Step t is bucket t of the window on both sides, so the sides share T, the horizon k. An integration writes
the field with `Boundary.record`, and the verifier reads it with `parse`.

* **Egress** (`egress.EgressBucket`, the scan's egress side, in generator form): ``frame`` is an output row of R·B
  bytes, the row the unit computes from its state, slot σ's cell at bytes σ·B … (σ + 1)·B − 1, row bit 8k + i bit i of
  byte k, which the unit's statement commits under the salt of the warden's egress row t of link ℓ and registers as a
  read of it (`warden_rows.bucket_port`), so the proof opens only when the row the unit computes is the warden's.
  ``verdict`` is an output row of one u16 word whose bit 0 is every slot's presence gate: 1 (`egress.ACCEPT`) exactly
  when the state renders a row. So `Layout.egress t ℓ σ` is (bit 0 of unit ℓ·T + t's verdict row, bits 8σB …
  8(σ + 1)B − 1 of its frame row), gates the unit computes and returns, none an input. Only the verdict is public: a
  unit's public output is at most 16 bits, and the frame is a committed output. v2's checked row, whose frame was a
  parameter row (an input) its verdict held to the cells the unit rendered, is retired with v2.
* **Ingress** (`ingress_scan.IngressScanBucket`, and `IngressScanFirst` its ``first``): ``frame`` is a parameter row of
  RI·BI bytes, the warden's ingress row t of link ℓ, which the unit reads as its registered input port.
  `Layout.ingress t` is the ingress links' frame rows at bucket t, in link order, each in row order.

Every bucket's gates are the same whatever the load, so no arrival step, prompt length or occupancy is public: a
request's token ids are values the window's units parse from the cells, like every other value (arrival steps and
prompt lengths stay private: the certifiers' observations are witness). The I/O tie reads row t of
link ℓ at unit `Boundary.unit` (``reads_from`` of a member's `warden_rows.bucket_port` is its unit of bucket t, less t).

**The check** (`refusal`, the verifier's), by `BoundaryGates.Fits`' conjuncts over the horizon: every egress gate is one
`C` returns (`C.outs`) and not an input (the verdict and the frame are output rows of the egress template, and every
unit's output rows are returned and committed; a frame that is a parameter row is an input, which the circuit's Program
reads from the advice, so it would accept any rows the trace carries and bound nothing); every ingress gate is an input
of the circuit (the frame is a parameter row of the ingress templates that no wire and no row table fills); and no
egress gate of a step depends on an ingress gate of a later step. Dependence is taken at the record's grain: each output
row of a unit depends on each of its input rows, a wired input row is its writer's output row (`Member.wired`), and a
row table's rows are inputs, which no unit of the record writes. That dependence contains `Depends`, so a record the
check accepts fits (`prog_fits` is `BoundaryGates.Fits` itself on a flattened program, and the tests hold `refusal` to
it); a record that fits only at the gate grain is refused. In the scan, ingress unit t's state output is ingress unit t
+ 1's state input and its view output egress unit t's state input: each chain is one registered leaf, written by one
unit and read by the other under one `b ‖ c`, so the record declares it a wire.

`Fits` is structural, never semantic: a served window fits in the step-indexed scan's form (`ingress_scan`), whose
bucket t unit reads bucket t's ingress row and the state bucket t − 1 carries (bucket 0's, `IngressScanFirst`'s, the
empty state, constants of its circuit), and whose Programs start at fixed boundaries and read their slots there, so
an egress bucket reaches ingress only through earlier buckets' state, whatever the arrival steps and prompt lengths. A
window in any other form, such as an egress bucket that reads every request's ids, is refused: the verifier gains no
form ahead of its proof. The record must also name the boundary's rows, and no
others, as its ``external`` ports.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FORMAT = "verity/boundary/v3"
#: the circuit record the boundary sits in (`verity.ml.windows.interface.CIRCUIT`)
CIRCUIT = "verity/window-circuit/v1"
_COUNTS = ("links", "buckets", "slots", "frame_bytes")
FIELDS = {"egress": ("template", "frame", "verdict", *_COUNTS), "ingress": ("template", "frame", *_COUNTS)}
#: each side's optional fields
OPTIONAL = {"egress": (), "ingress": ("first",)}


@dataclass(frozen=True)
class Side:
    """One side of the boundary: its template, its frame row, its egress verdict row, its schedule's counts, and the
    template of its first bucket when another holds it."""

    template: str
    frame: str
    links: int
    buckets: int
    slots: int
    frame_bytes: int
    verdict: str | None = None
    first: str | None = None

    @property
    def words(self) -> int:
        """The frame row's u16 words: the bucket's R·B bytes."""
        return self.slots * self.frame_bytes // 2

    def counts(self) -> dict[str, int]:
        """Each of its templates' units: links·T, or with ``first`` links and links·(T − 1)."""
        if self.first is None:
            return {self.template: self.links * self.buckets}
        return {self.first: self.links, self.template: self.links * (self.buckets - 1)}

    def at(self, link: int, t: int) -> tuple[str, int]:
        """The template that holds bucket t of link ``link`` and its unit there, in the template's units' order."""
        if self.first is None:
            return self.template, link * self.buckets + t
        return (self.first, link) if t == 0 else (self.template, link * (self.buckets - 1) + t - 1)

    def ports(self) -> tuple[str, ...]:
        return tuple(f"{tid}/{p}" for tid in self.counts() for p in (self.frame, self.verdict) if p is not None)


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
        return {"format": FORMAT, **{name: {k: getattr(s, k) for k in (*FIELDS[name], *OPTIONAL[name])
                                            if getattr(s, k) is not None} for name, s in self.sides()}}

    def unit(self, doc: Mapping[str, Any], side: str, link: int, t: int) -> int:
        """The global unit (the population's index) that holds bucket t of link ``link`` of ``side``."""
        s = dict(self.sides())[side]
        if not (0 <= link < s.links and 0 <= t < s.buckets):
            raise ValueError(f"{side} has links [0, {s.links}) and buckets [0, {s.buckets}), not ({link}, {t})")
        return _global(_template_units(doc), s, link, t)


def _side(name: str, doc: Any) -> Side:
    if not isinstance(doc, Mapping) or not set(FIELDS[name]) <= set(doc) <= {*FIELDS[name], *OPTIONAL[name]}:
        raise ValueError(f"boundary {name}: the fields {list(FIELDS[name])}, and optionally {list(OPTIONAL[name])}, not "
                         f"{doc!r}")
    if any(not isinstance(doc[k], str) or not doc[k] for k in doc if k not in _COUNTS):
        raise ValueError(f"boundary {name}: its templates and rows are named by nonempty strings, not {doc!r}")
    if any(type(doc[k]) is not int or doc[k] < 1 for k in _COUNTS):
        raise ValueError(f"boundary {name}: {list(_COUNTS)} are positive integers, not {[doc[k] for k in _COUNTS]}")
    if doc["frame_bytes"] % 2:
        raise ValueError(f"boundary {name}: a cell is whole u16 words, not {doc['frame_bytes']} bytes")
    if doc.get("verdict") == doc["frame"]:
        raise ValueError(f"boundary {name}: the verdict row is not the frame row {doc['frame']!r}")
    if "first" in doc and (doc["first"] == doc["template"] or doc["buckets"] < 2):
        raise ValueError(f"boundary {name}: a first bucket's template is another than {doc['template']!r}, over two "
                         f"buckets or more, not {doc['first']!r} over {doc['buckets']}")
    return Side(**{k: doc[k] for k in doc})


def parse(doc: Mapping[str, Any]) -> Boundary | None:
    """The circuit record's boundary, None when it declares none; ValueError for a malformed one."""
    b = doc.get("boundary")
    if b is None:
        return None
    if not isinstance(b, Mapping) or b.get("format") != FORMAT or not {"format", "egress"} <= set(b) <= {"format", *FIELDS}:
        raise ValueError(f"a boundary is {FORMAT} with an egress side and optionally an ingress side, not {b!r}")
    out = Boundary(_side("egress", b["egress"]), _side("ingress", b["ingress"]) if "ingress" in b else None)
    if out.ingress and out.ingress.buckets != out.egress.buckets:
        raise ValueError(f"boundary: one step a bucket on both sides, so one T, not egress {out.egress.buckets} and "
                         f"ingress {out.ingress.buckets}")
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


def _global(units: Mapping[str, tuple[int, int]], s: Side, link: int, t: int) -> int:
    tid, i = s.at(link, t)
    return units[tid][0] + i


def _own_row(s: Side, tid: str, m: Mapping[str, Any]) -> str | None:
    """None when member ``m`` of ingress template ``tid`` reads side ``s``'s frame as a parameter row of its R·B bytes
    that no wire and no row table fills, so its gates are the unit's own inputs; else why not."""
    who = m["name"] or tid
    rows = dict(map(tuple, m["rows"]))
    if rows.get(s.frame) != s.words:
        return (f"BoundaryGates.Fits, ingress gates are inputs: {tid}'s {s.frame!r} is not a parameter row of "
                f"{s.words} words (R·B = {s.slots}·{s.frame_bytes} bytes); its rows are {rows}")
    if m["tables"] or m["links"]:
        return (f"BoundaryGates.Fits, ingress gates are inputs: {who} reads row tables, and an ingress frame is its "
                "own unit's row")
    if s.frame in {p for p, _o, _w in m["wired"]}:
        return (f"BoundaryGates.Fits, ingress gates are inputs: {who}'s {s.frame!r} is wired from another unit's "
                "output, so its gates are that unit's, not inputs")
    return None


def _returned(e: Side, m: Mapping[str, Any]) -> str | None:
    """None when member ``m`` of the egress template returns its verdict, an output row of one u16 word, and its frame,
    an output row of its R·B bytes, so every egress gate is one the unit computes and returns; else why not."""
    rows, outs = dict(map(tuple, m["rows"])), dict(map(tuple, m["outs"]))
    if outs.get(e.verdict) != 1:
        return (f"BoundaryGates.Fits, egress gates are returned: {e.template}'s verdict {e.verdict!r} is not an output "
                f"row of one u16 word (a unit's public output is at most 16 bits); its outputs are {outs}")
    if e.frame in rows:
        return (f"BoundaryGates.Fits, egress gates are returned: {e.template}'s frame {e.frame!r} is a parameter row, "
                "an input, which the Program reads from the advice and so bounds nothing: NetTiming.BoundaryGates.Fits "
                "takes every slot gate to be one C returns (C.outs) and not an input, a frame the unit computes "
                "(prog_fits)")
    if outs.get(e.frame) != e.words:
        return (f"BoundaryGates.Fits, egress gates are returned: {e.template}'s frame {e.frame!r} is not an output row "
                f"of {e.words} words (R·B = {e.slots}·{e.frame_bytes} bytes); its outputs are {outs}")
    return None


def refusal(doc: Mapping[str, Any]) -> str | None:
    """The verifier's check: None when the circuit record's boundary fits its circuit over its horizon
    (`NetTiming.BoundaryGates.Fits`, at the record's grain: module docstring), else why not, by the conjunct it
    fails."""
    if doc.get("format") != CIRCUIT:
        return f"boundary: a circuit record is {CIRCUIT}, not {doc.get('format')!r}"
    try:
        b = parse(doc)
        units = _template_units(doc)
    except (ValueError, KeyError, TypeError) as e:
        return f"boundary: {e}"
    if b is None:
        return ("boundary: the circuit record declares none (NetTiming.BoundaryGates), so the network audit has no "
                "layout to read")
    if set(doc.get("external", ())) != b.ports():
        return (f"boundary: the record's external ports {sorted(doc.get('external', ()))} are not the boundary's rows "
                f"{sorted(b.ports())}: I/O the layout doesn't hold is I/O no audit reads")
    members = {tid: [m for m in doc["members"] if m["template"] == tid] for tid in units}
    for name, s in b.sides():
        how = ({s.template: f"links·buckets = {s.links}·{s.buckets}"} if s.first is None else
               {s.first: f"links = {s.links}", s.template: f"links·(buckets − 1) = {s.links}·{s.buckets - 1}"})
        for tid, n in s.counts().items():
            if tid not in units:
                return f"boundary {name}: {tid} is no template of the circuit"
            if units[tid][1] != n:
                return f"boundary {name}: {tid} has {units[tid][1]} units, not {how[tid]}"
    for m in members[b.egress.template]:
        if why := _returned(b.egress, m):
            return why
    if b.ingress:
        for tid in b.ingress.counts():
            for m in members[tid]:
                if why := _own_row(b.ingress, tid, m):
                    return why
    total = sum(n for _lo, n in units.values())
    try:
        readers = _readers(doc, total)
    except (ValueError, KeyError, TypeError) as e:
        return f"boundary: {e}"
    T = b.horizon
    sources: list[list[int]] = [[] for _ in range(T)]
    link_of: dict[int, int] = {}
    if b.ingress:
        for ell in range(b.ingress.links):
            for t in range(T):
                u = _global(units, b.ingress, ell, t)
                sources[t].append(u)
                link_of[u] = ell
    latest, src = _latest(sources, readers)
    for ell in range(b.egress.links):
        for t in range(T):
            u = _global(units, b.egress, ell, t)
            if latest[u] > t:
                return (f"BoundaryGates.Fits, causal: egress bucket {t} of link {ell} (unit {u}) depends on ingress "
                        f"bucket {latest[u]} of link {link_of[src[u]]} (unit {src[u]}), a later step, through the "
                        "record's wires; a window fits in the scan's form, its egress reading ingress through earlier "
                        "buckets' state (ingress_scan)")
    return None


def prog_fits(gates: Sequence[tuple[str, Sequence[int]]], outs: Sequence[int],
              egress: Callable[[int, int, int], tuple[int, Sequence[int]]], ingress: Callable[[int], Sequence[int]],
              L: int, S: int, k: int) -> bool:
    """`NetTiming.BoundaryGates.Fits lay C k` itself on a flattened program `C` (`Claims.Sampling.Prog`: ``gates`` each
    ``(kind, reads)``, kind one of ``input``, ``const``, ``not``, ``and``, ``xor``; ``outs`` its returned gates) and a
    layout (``egress(t, ℓ, σ)``, a slot's presence gate and frame gates, and ``ingress(t)``): before ``k``, every slot
    gate is in ``outs`` and not an input, every ingress gate is an input, and no slot gate of a step reaches (`Depends`,
    reflexive) an ingress gate of a later step."""
    returned = set(outs)
    pairs = [(t, egress(t, ell, s)) for t in range(k) for ell in range(L) for s in range(S)]
    slot = {t: [g for u, (p, frame) in pairs if u == t for g in (p, *frame)] for t in range(k)}
    is_input = lambda g: 0 <= g < len(gates) and gates[g][0] == "input"  # noqa: E731
    if any(g not in returned or is_input(g) for _t, (p, frame) in pairs for g in (p, *frame)):
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
