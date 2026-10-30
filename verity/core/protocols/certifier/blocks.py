"""A host's class circuit as network accounting reads it: the rows that cross the host's uplink, and the pins of its loads.

A circuit record is one host's: its class's window circuit, every slot's class Program whatever its traffic. Which host
runs which units, and which request occupies which slot and step, are private: the network certifier on the host's uplink
commits every row that crosses it, and the scan (`ingress_scan`, `egress`) places those rows into slots inside the
zero-knowledge proof, its state committed as the registered value `ingress_scan.SCAN_STATE`. What is public is the class
circuit itself, protocol structure, and its padded counts: proof blocks per host and bits per link, each execution
block's. So a record names no server and no transfer, and nothing here pins a request, a response or a value one host
sends another.

Two values cross the uplink into a host's units: its loads, the registration's values, each committed by the network
certifier as it crosses once a session (`transfer_rows`, `benchmarks/network_traces/load_relay.py`), and the scan's state,
which the verifier pins at the root of the scan's rows the window reads (`ingress_scan.window_state`), never here.

**The check** (`check`), over the record before any coin, the work per declaration and never per unit:

- the record's ``blocks`` state no server, server time or transfers (`PRIVATE`): placement is private, and a record is
  one host's;
- no ``ingress`` declaration: a host's ingress is the scan's state, a registered read, never an Ingress;
- every registered read and every table is of a load or of the scan's state: every registered parameter other than
  `ingress_scan.SCAN_STATE` is a load;
- each load read by some unit, each read a whole row of it (base and steps multiples of the reader row's w words, one w
  a value, its leaves whole rows), a table over it the whole value, and none of its reads, nor a table read's or an
  opening's port, an external port;
- every row of every external input port read from the scan's state, exactly once: the scan's state is the host's one
  ingress. An external output port's rows are the circuit's outputs, the slot entries the scan's Programs row takes.

**The pins** (`certified`, `pins`): each load's registration (`transfer_rows.registration`) read for the load the check
made of the record (`transfer_rows.check_registration`: the host's uplink, ingress, the session's first execution phase
`SESSION_WINDOW`, kind ``load``, the value, the record's program digest and the value's shape, its row format the
verifier's own: 16-bit leaves in ``hm96-sha512/row/v2`` rows unless ``formats`` names another, as a load's f32 scale is
one 32-bit leaf a row and its token embeddings ``hm96-sha512/row-seg/v1`` rows) under the host's network certifier's key
and session. A load without one, or a registration naming no load, is refused, so a load's root is pinned only from its
certificate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.core.protocols.certifier import transfer_rows as TR
from verity.core.protocols.certifier.certificate import Direction, PublicKey
from verity.core.protocols.certifier.ingress_scan import SCAN_STATE

#: the record's ``blocks`` keys that would make placement public; a record states none of them
PRIVATE = ("server", "server_time", "transfers")
#: the registration's ``window`` of a load: a load crosses once a session, in the session's first execution phase
SESSION_WINDOW = 0
_READ = ("reader", "port", "param", "at", "radix", "base", "step")


def _nat(x: Any, lo: int = 0) -> bool:
    return type(x) is int and x >= lo


@dataclass(frozen=True)
class Pattern:
    """A read's rows: template ``tid``'s port ``port``, rows ``cut[0] .. cut[0] + C − 1`` of its cut (C = ``cut[1]``),
    index i its unit ``at + i // C`` and row ``cut[0] + i % C``, its source ``base + Σ d[k]·step[k]`` (d: i's digits in
    mixed radix ``radix``), the value's leaves from there."""

    tid: str
    port: str
    at: int
    radix: tuple[int, ...]
    base: int
    step: tuple[int, ...]
    cut: tuple[int, int] = (0, 1)

    @property
    def units(self) -> int:
        return math.prod(self.radix)

    @property
    def unit_count(self) -> int:
        return self.units // self.cut[1]

    def span(self) -> tuple[int, int]:
        """The least and the greatest source."""
        reach = [s * (r - 1) for r, s in zip(self.radix, self.step)]
        return self.base + sum(min(0, x) for x in reach), self.base + sum(max(0, x) for x in reach)

    def aligned(self, w: int) -> bool:
        """Every source a multiple of ``w``: each index reads a whole row of ``w`` words."""
        return self.base % w == 0 and all(s % w == 0 for s in self.step)

    def doc(self) -> dict[str, Any]:
        return {"tid": self.tid, "port": self.port, "at": self.at, "radix": list(self.radix), "base": self.base,
                "step": list(self.step), **({"cut": list(self.cut)} if self.cut != (0, 1) else {})}


def _pattern(d: Any, keys: Sequence[str], what: str) -> Pattern:
    if not isinstance(d, Mapping) or not set(keys) <= set(d) <= {*keys, "cut"}:
        raise ValueError(f"a {what} is {{{', '.join(keys)}}} and optionally its cut, not {d!r}")
    radix, step, c = d["radix"], d["step"], d.get("cut", [0, 1])
    if not (isinstance(radix, list) and isinstance(step, list) and isinstance(c, list) and radix and len(step) == len(radix)
            and all(_nat(r, 1) for r in radix) and all(type(s) is int for s in step) and _nat(d["at"]) and type(d["base"]) is int
            and len(c) == 2 and _nat(c[0]) and _nat(c[1], 1) and isinstance(d[keys[0]], str) and isinstance(d[keys[1]], str)):
        raise ValueError(f"{what} {d!r}: a first unit, at least one digit of radix 1 or more, a base, one integer step a digit "
                         "and its rows [c0, C] (C at least 1)")
    p = Pattern(d[keys[0]], d[keys[1]], d["at"], tuple(radix), d["base"], tuple(step), (c[0], c[1]))
    if p.units % p.cut[1] or (p.cut[1] > 1 and p.radix[-1] != p.cut[1]):
        raise ValueError(f"{what} {d!r}: C divides its indices and is its last digit when C > 1")
    return p


@dataclass(frozen=True)
class Load:
    """One value the host loads: its name, its commitment's shape (``rows`` rows of ``words`` u16 words) and the reads of
    its rows."""

    value: str
    words: int
    rows: int
    gates: tuple[Pattern, ...]


def _templates(doc: Mapping[str, Any], layouts: Mapping[str, Any]) -> dict[str, tuple[dict[str, int], dict[str, int], int]]:
    out = {}
    for t, n in doc.get("templates", ()):
        if t not in layouts:
            raise ValueError(f"template {t}: no layout of it")
        out[t] = (dict(layouts[t].rows), dict(layouts[t].outs), int(n))
    if not out:
        raise ValueError("a circuit record with no templates")
    return out


def _disjoint(rects: Sequence[tuple[tuple[str, str], int, int, int, int]], what: str) -> None:
    """Refuse two rectangles (units × rows) of one port that meet."""
    by: dict[tuple[str, str], list[tuple[int, int, int, int]]] = {}
    for key, a, b, r0, r1 in rects:
        by.setdefault(key, []).append((a, b, r0, r1))
    for key, got in by.items():
        got.sort()
        live: list[tuple[int, int, int, int]] = []
        for x in got:
            live = [y for y in live if y[1] > x[0]]
            if any(y[2] < x[3] and x[2] < y[3] for y in live):
                raise ValueError(f"{key[0]}/{key[1]}: {what} fill one row of a unit from unit {x[0]} twice")
            live.append(x)


def _cut(doc: Mapping[str, Any], tid: str, port: str) -> int:
    k = (doc.get("cuts") or {}).get(tid, {}).get(port, 1)
    if not _nat(k, 1):
        raise ValueError(f"cuts: {tid} {port} cut {k!r} ways")
    return k


def check(doc: Mapping[str, Any], layouts: Mapping[str, Any]) -> tuple[Load, ...]:
    """The host's loads, by their order among the record's registered parameters; refused (ValueError, naming the rule)
    when the circuit record ``doc`` (``circuit.json``: its templates, external ports, cuts, registered reads, tables,
    roots and openings, and ingress, which the window interface's own checks have held well-formed, `check_inputs`)
    breaks the module docstring's rules against ``layouts``, each template's bit rows by descriptor id (the interface's
    `layout`: its ``rows`` and ``outs``, each ``(name, words)``), the verifier's own from its catalog."""
    tpl = _templates(doc, layouts)
    blocks = doc.get("blocks")
    if isinstance(blocks, Mapping) and (named := [k for k in PRIVATE if k in blocks]):
        raise ValueError(f"the record's blocks state {named}: placement is private, and a record is one host's; which host "
                         "runs which units, and which rows cross its uplink, are no key of it")
    if doc.get("ingress"):
        raise ValueError("an ingress declaration: a host's ingress is the scan's state, a registered read, never an Ingress")
    reg = doc.get("registered") or {}
    params = dict(reg.get("params", {}))
    reads = [_pattern(r, _READ, "registered read") for r in reg.get("reads", ())]
    read_param = [r["param"] for r in reg.get("reads", ())]
    tables = list(reg.get("tables", ()))
    over = {t["name"]: t["param"] for t in tables}
    roots, openings = list(reg.get("roots", ())), list(reg.get("openings", ()))
    external = set(doc.get("external", ()))
    for p in [*read_param, *over.values()]:
        if p not in params:
            raise ValueError(f"a registered read of {p!r}, which is no registered parameter of the circuit's ({sorted(params)})")
    loads: list[Load] = []
    for value, leaves in params.items():
        if value == SCAN_STATE:
            continue
        what = f"load {value!r}"
        gates: list[Pattern] = []
        widths: set[int] = set()
        for r, p in zip(reads, read_param):
            if p != value:
                continue
            k = _cut(doc, r.tid, r.port)
            w = tpl[r.tid][0][r.port] // k if r.tid in tpl and r.port in tpl[r.tid][0] else 0
            lo, hi = r.span()
            if not w or not r.aligned(w) or lo < 0 or hi + w > leaves:
                raise ValueError(f"{what}: read {r.doc()} isn't whole rows of {w} words among the value's {leaves} leaves")
            if f"{r.tid}/{r.port}" in external:
                raise ValueError(f"{what}: read {r.doc()}: a load's rows are never the circuit's external ports")
            gates.append(r)
            widths.add(w)
        for t in tables:
            if t["param"] == value:
                if t["offset"] != 0 or t["words"] * t["rows"] != leaves:
                    raise ValueError(f"{what}: table {t['name']} is a part of the value; a table over a load is the whole "
                                     "value, under the network certifier's root")
                widths.add(t["words"])
        for o in [*roots, *openings]:
            if over.get(o["table"]) == value:
                for p in ([o["port"]] if "port" in o else o["ports"]):
                    if f"{o['reader']}/{p}" in external:
                        raise ValueError(f"{what}: table {o['table']}'s read fills {o['reader']}/{p}, an external port")
        if not widths:
            raise ValueError(f"{what}: no unit reads it")
        if len(widths) != 1 or leaves % next(iter(widths)):
            raise ValueError(f"{what}: its reads are rows of {sorted(widths)} words, and its {leaves} leaves; a value is "
                             "rows of one width")
        words = next(iter(widths))
        loads.append(Load(value, words, leaves // words, tuple(gates)))
    state = [r for r, p in zip(reads, read_param) if p == SCAN_STATE]
    for r in state:
        if f"{r.tid}/{r.port}" not in external:
            raise ValueError(f"the scan's state: read {r.doc()} fills {r.tid}/{r.port}, no external port; the scan's state "
                             "is read only at the circuit's external input ports")
    for e in sorted(external):
        tid, _, port = e.rpartition("/")
        if tid not in tpl:
            raise ValueError(f"external port {e}: no template's")
        rows_, outs_, n = tpl[tid]
        if port in outs_:
            continue
        if port not in rows_:
            raise ValueError(f"external port {e}: no port of {tid}")
        rects = [((tid, port), r.at, r.at + r.unit_count, r.cut[0], r.cut[0] + r.cut[1]) for r in state
                 if (r.tid, r.port) == (tid, port)]
        _disjoint(rects, "reads")
        covered, want = sum((b - a) * (r1 - r0) for _k, a, b, r0, r1 in rects), n * _cut(doc, tid, port)
        if covered != want:
            raise ValueError(f"external port {e}: {want - covered} of its {want} unit rows are no read of the scan's state; "
                             "every row of an external input port crosses the host's uplink, as the scan's state")
    return tuple(loads)


def certified(doc: Mapping[str, Any], layouts: Mapping[str, Any], registrations: Iterable[Any], link: str,
              certifier: tuple[PublicKey, bytes],
              formats: Mapping[str, tuple[int, str]] | None = None) -> dict[str, TR.TransferHead]:
    """Each load's head, by value, from its registration (`transfer_rows.check_registration`, for the load `check` makes
    of the record, at the record's program digest) under ``certifier``, the key and session of the network certifier on
    the host's uplink ``link``. ``formats``: the verifier's own ``(word_bits, schema)`` of a load whose leaves aren't 16
    bits or whose rows aren't ``hm96-sha512/row/v2`` (its program's parameter width, `registered.width`, and its port's
    row format), a row of ``words`` u16 words then ``words · 16 / word_bits`` leaves. Refused (ValueError) when the record
    breaks `check`, a load has no registration or its registration fails, or a registration or a format names no load."""
    loads = check(doc, layouts)
    formats = dict(formats or {})
    if extra := sorted(set(formats) - {x.value for x in loads}):
        raise ValueError(f"formats of {extra}, which no load names")
    regs: dict[str, Any] = {}
    for r in registrations:
        v, d = (r.get("value"), r.get("direction")) if isinstance(r, Mapping) else (None, None)
        if not isinstance(v, str) or d != Direction.INGRESS.value or v in regs:
            raise ValueError(f"a load's registration names a value, ingress, once: {v!r} {d!r}")
        regs[v] = r
    if extra := sorted(set(regs) - {x.value for x in loads}):
        raise ValueError(f"registrations of {extra}, which no load names: a root is pinned only for a load the record reads")
    key, session = certifier
    heads = {}
    for x in loads:
        what = f"load {x.value!r} (on {link})"
        if x.value not in regs:
            raise ValueError(f"{what}: no certificate; a load's root is pinned from its network certifier's certificate alone")
        bits, schema = formats.get(x.value, (TR.WORD_BITS, TR.SCHEMA))
        try:
            if bits not in TR.WIDTHS or 16 * x.words % bits:
                raise ValueError(f"its rows of {x.words} u16 words are no whole number of {bits!r}-bit leaves")
            heads[x.value], _ = TR.check_registration(
                regs[x.value], key, session, link=link, direction=Direction.INGRESS, window=SESSION_WINDOW, kind="load",
                value=x.value, program=doc.get("program"), words=16 * x.words // bits, rows=x.rows, word_bits=bits,
                schema=schema)
        except ValueError as e:
            raise ValueError(f"{what}: {e}") from None
    return heads


def pins(doc: Mapping[str, Any], layouts: Mapping[str, Any], registrations: Iterable[Any], link: str,
         certifier: tuple[PublicKey, bytes], formats: Mapping[str, tuple[int, str]] | None = None) -> dict[str, str]:
    """The verifier's pins of the host's loads, ``{parameter: root}``: every load at its certified root (`certified`),
    the roots ``registration.check`` holds a pinned value to and the parameters the interface's ``input_count`` counts as
    registered. The scan's state is pinned beside them, at the root of the scan's rows the window reads."""
    return {v: h.root.hex() for v, h in certified(doc, layouts, registrations, link, certifier, formats).items()}
