"""The blocks as network accounting reads them: which proof units each server runs, the server-time block a
circuit is, and its transfer gates, every row that crosses a server block's boundary, each a row of a value the network
certifier on that server's uplink committed and signed (`transfer_rows`). The verifier pins a transfer value's root from
that certificate alone (`pins`), so with every other input row wired, the circuit's free inputs (the window interface's
`input_count` under those pins) are 0.

The schema, `verity/blocks/v1` (`FORMAT`), is kept here, in network accounting's module, until circuits' window interface
(`verity.ml.windows.interface`) carries it in ``circuit.json`` beside the circuit's other declarations, as it carries
circuits' proof and inference blocks (the record's ``blocks``); until then a check takes it beside the circuit record:

    {"format": "verity/blocks/v1",
     "server": [{"name": S, "uplink": LINK, "units": "all" | [[tid, at, n], ...]}, ...],
     "server_time": {"window": W, "steps": T},
     "transfers": [{"server": S, "dir": "in" | "out", "kind": "load" | "request" | "response" | "wire", "value": V,
                    "window": w, "t": null | {"base": b, "radix": [...], "step": [...]}}, ...]}

An ``out`` transfer adds ``"leaves": L`` and ``"writes": [{"writer": tid, "out": "o<j>", "at", "radix", "base", "step"}]``:
writer units ``[at, at + Π radix)``, each unit ``at + i`` writing its output row ``out`` as the value's leaves from
``base + Σ d[k]·step[k]`` (d: i's digits in mixed radix ``radix``, the first the slowest), the window interface's `Read`
pattern on an output row, which `Read` doesn't take yet (the Lean verifier's registered reads already open an output
port's row). An ``in`` transfer's gates are the circuit's registered reads of V (``registered.reads`` whose ``param`` is V,
and tables over V), its leaves ``registered.params[V]``. Leaves are 16 bits, as the interface's; a value of L leaves read
in rows of w words is L/w rows, the certifier's commitment. ``t`` is each row's protocol step in the server-time window
(row r's ``base + Σ d[k]·step[k]``, d: r's digits in ``radix``), null for the load, which crosses once a server session,
in the link-window ``window`` names; a request's, response's or wire's window is the server-time block's. A wire is one
value V and two transfers, V's ``out`` on the sending server and V's ``in`` on the receiving one, each certified by its
own uplink's network certifier; every other value is one transfer.

**The check** (`check`), declaration by declaration, before any coin; the one count over rows is an out value's (a byte
a row, the size of the commitment the certifier makes of it row by row), never over units:

- the servers each a new name on a new uplink (one network certifier an uplink), their units together a partition of
  every template's units ("all": the one server's);
- each inference block on one server (a server's ingress and egress carry only its inference
  blocks' declared inputs and outputs, and an inference block runs on a single server, so which server may read its
  input bits is unambiguous): the record's inference entries (``blocks.inference``, the window interface's
  `InferenceEntry`, each unit's (slot, step)) tile every template's units, and no (slot, step) has units on two servers;
  with two servers or more the record states them. Which server runs which units is protocol structure, deployment
  configuration as NCI's declaration is, public and not a private partition, so this is checked in the clear with no
  proof block (`server_blocks`). The work is per entry, server run and digit: an entry's units on one run are at most
  two boxes a digit of its mixed radix, each a rectangle of (slot, step);
- a transfer a new value, but a wire's other one, its kind in its direction (`transfer_rows.KINDS`), on a server, in its
  window, its ``t`` within the server-time block's T steps and over its rows;
- an ``in`` transfer's value a registered parameter read by its server's units alone, each read a whole row of it (base
  and steps multiples of the reader row's w words, one w), a table over it the whole value; an ``out`` transfer's writes
  its server's units' output rows, each a whole row of the value, every row of the value written by exactly one unit,
  and no output row written twice;
- a wire's two transfers paired: one ``out`` and one ``in``, on two servers (so two uplinks, two certifiers), of one
  shape (rows and words) and one step a row (``t``), since each row crosses both uplinks at once;
- a server's boundary rows are exactly its transfers' rows: no ``ingress`` declaration (a server's ingress is a
  certified read), no registered read of a parameter no transfer brings in, no wire between two servers' units (a value
  one server sends another is a wire transfer, the sender's ``out`` and the receiver's ``in``), a request's and a
  response's rows the circuit's external ports and a load's and a wire's never, and every row of every external port a
  transfer's: an input port's every unit and row read by ``in`` transfers, an output port's every unit's row written by
  ``out`` transfers.

**The pins** (`certified`, `pins`): each transfer's registration (`transfer_rows.registration`) read for the transfer
the check made of the record (`transfer_rows.check_registration`: its uplink, direction, window, kind, value, the
circuit's program digest and the value's shape, its row format the verifier's own: 16-bit leaves in
``hm96-sha512/row/v2`` rows unless ``formats`` names another, as a load's f32 scale is one 32-bit leaf a row and its
token embeddings ``hm96-sha512/row-seg/v1`` rows) under its uplink's certifier's key and session. A transfer without
one, or a registration naming no transfer, is refused, so a root is pinned only from its certificate. A wire's two heads
are under one domain (one program, value and shape), and the receiving certifier commits the rows it saw under the
sending one's salts (`transfer_rows`), so its two roots must be one: they are exactly when the rows that reached the
receiver's uplink are the rows that left the sender's, and a wire whose roots differ is refused.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.core.protocols.certifier import transfer_rows as TR
from verity.core.protocols.certifier.certificate import Direction, PublicKey

FORMAT = "verity/blocks/v1"
DIRS = {"in": Direction.INGRESS, "out": Direction.EGRESS}
#: the kinds whose rows are the circuit's external ports, the system's I/O; a load's and a wire's never are
EXTERNAL = ("request", "response")
_KEYS = ("format", "server", "server_time", "transfers")
_TRANSFER = ("server", "dir", "kind", "value", "window", "t")
_OUT = ("leaves", "writes")
_READ = ("reader", "port", "param", "at", "radix", "base", "step")
_WRITE = ("writer", "out", "at", "radix", "base", "step")
_WIRE = ("reader", "port", "writer", "out", "at", "radix", "base", "step")
#: an inference entry (the window interface's `InferenceEntry`): unit ``at + i``'s (slot, step) is its slot digit and
#: ``step`` plus its step digit, i's digits in mixed radix ``radix`` named by ``axes``
_ENTRY = ("template", "at", "radix", "axes", "step")
_AXES = ("slot", "step", "-")


def _nat(x: Any, lo: int = 0) -> bool:
    return type(x) is int and x >= lo


@dataclass(frozen=True)
class Pattern:
    """A read's or a write's rows: template ``tid``'s port ``port``, rows ``cut[0] .. cut[0] + C − 1`` of its cut (C =
    ``cut[1]``; an output row is uncut), index i its unit ``at + i // C`` and row ``cut[0] + i % C``, its source
    ``base + Σ d[k]·step[k]`` (d: i's digits in mixed radix ``radix``), the value's leaves from there."""

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


def _pattern(d: Any, keys: Sequence[str], what: str, cut: bool = False) -> Pattern:
    if not isinstance(d, Mapping) or not set(keys) <= set(d) <= {*keys, *(("cut",) if cut else ())}:
        raise ValueError(f"a {what} is {{{', '.join(keys)}}}{' and optionally its cut' if cut else ''}, not {d!r}")
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
class Server:
    """A server block: proof blocks ``units`` (each ``(tid, at, n)``) on uplink ``uplink``, which one network certifier
    watches."""

    name: str
    uplink: str
    units: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class Transfer:
    """One value crossing one server's boundary: its server and uplink, direction, kind, name and link-window, its
    commitment's shape (``rows`` rows of ``words`` u16 words), each row's step (``t``) and its gates, the reads (``in``)
    or writes (``out``) of its rows."""

    server: str
    uplink: str
    direction: Direction
    kind: str
    value: str
    window: int
    words: int
    rows: int
    t: Mapping[str, Any] | None
    gates: tuple[Pattern, ...]


@dataclass(frozen=True)
class Blocks:
    """A circuit's server blocks, its server-time block (``window``, ``steps``) and its transfers, as `check` read them."""

    servers: tuple[Server, ...]
    window: int
    steps: int
    transfers: tuple[Transfer, ...]


def _templates(doc: Mapping[str, Any], layouts: Mapping[str, Any]) -> dict[str, tuple[dict[str, int], dict[str, int], int]]:
    out = {}
    for t, n in doc.get("templates", ()):
        if t not in layouts:
            raise ValueError(f"template {t}: no layout of it")
        out[t] = (dict(layouts[t].rows), dict(layouts[t].outs), int(n))
    if not out:
        raise ValueError("a circuit record with no templates")
    return out


def _servers(docs: Any, tpl: Mapping[str, tuple[Any, Any, int]]) -> tuple[tuple[Server, ...], dict[str, list[list[Any]]]]:
    """The servers, and each template's units as runs ``[lo, hi, server]``, adjacent runs of one server merged; refused
    unless they partition every template's units."""
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"a circuit's servers are a nonempty list, not {docs!r}")
    servers, runs = [], {t: [] for t in tpl}
    for d in docs:
        if not isinstance(d, Mapping) or set(d) != {"name", "uplink", "units"} or not all(
                isinstance(d[k], str) and d[k] for k in ("name", "uplink")):
            raise ValueError(f"a server is {{name, uplink, units}}, a name and an uplink, not {d!r}")
        if d["name"] in {s.name for s in servers} or d["uplink"] in {s.uplink for s in servers}:
            raise ValueError(f"server {d['name']} on {d['uplink']}: a new name on a new uplink (one network certifier an uplink)")
        if d["units"] == "all":
            if len(docs) != 1:
                raise ValueError(f"server {d['name']}: \"all\" units are the one server's")
            units = tuple((t, 0, n) for t, (_r, _o, n) in tpl.items())
        elif isinstance(d["units"], list) and d["units"]:
            units = tuple(tuple(u) for u in d["units"])
            for u in units:
                if len(u) != 3 or u[0] not in tpl or not _nat(u[1]) or not _nat(u[2], 1) or u[1] + u[2] > tpl[u[0]][2]:
                    raise ValueError(f"server {d['name']}: units {list(u)} are [tid, at, n] of a template's units")
        else:
            raise ValueError(f"server {d['name']}: its units are \"all\" or a nonempty list of [tid, at, n]")
        servers.append(Server(d["name"], d["uplink"], units))
        for t, at, n in units:
            runs[t].append([at, at + n, d["name"]])
    for t, rs in runs.items():
        rs.sort()
        end = 0
        for lo, hi, s in rs:
            if lo != end:
                raise ValueError(f"template {t}: units [{min(lo, end)}, {max(lo, end)}) are "
                                 f"{'in no server' if lo > end else 'in two servers'}; the servers partition the units")
            end = hi
        if end != tpl[t][2]:
            raise ValueError(f"template {t}: units [{end}, {tpl[t][2]}) are in no server; the servers partition the units")
        merged: list[list[Any]] = []
        for r in rs:
            if merged and merged[-1][2] == r[2]:
                merged[-1][1] = r[1]
            else:
                merged.append(list(r))
        runs[t] = merged
    return tuple(servers), runs


def _server_of(runs: Mapping[str, list[list[Any]]], tid: str, lo: int, hi: int) -> str | None:
    """The server whose run holds units ``[lo, hi)`` of template ``tid``, or None when no one run holds them."""
    return next((s for a, b, s in runs.get(tid, ()) if a <= lo and hi <= b), None)


def _boxes(lo: int, hi: int, radix: Sequence[int], fixed: tuple[int, ...] = ()) -> Iterator[tuple[tuple[int, ...], int, int]]:
    """Indices ``[lo, hi)`` of the digits after ``fixed`` in mixed radix ``radix`` (the first the slowest), as at most two
    boxes a digit, each ``(fixed', a, b)``: the digits ``fixed'``, then one in ``[a, b)``, then every later one any."""
    if lo >= hi:
        return
    w = math.prod(radix[len(fixed) + 1:])
    a, ra = divmod(lo, w)
    b, rb = divmod(hi, w)
    if a == b:
        yield from _boxes(ra, rb, radix, (*fixed, a))
        return
    if ra:
        yield from _boxes(ra, w, radix, (*fixed, a))
        a += 1
    if a < b:
        yield fixed, a, b
    if rb:
        yield from _boxes(0, rb, radix, (*fixed, b))


def _digit(box: tuple[tuple[int, ...], int, int], radix: Sequence[int], d: int) -> tuple[int, int]:
    """Digit ``d``'s range ``[lo, hi)`` over a box of `_boxes`."""
    fixed, a, b = box
    return (fixed[d], fixed[d] + 1) if d < len(fixed) else (a, b) if d == len(fixed) else (0, radix[d])


def _one_server(doc: Mapping[str, Any], tpl: Mapping[str, tuple[Any, Any, int]], runs: Mapping[str, list[list[Any]]],
                servers: Sequence[Server]) -> None:
    """Refuse an inference block of the record's (``blocks.inference``) whose units are on two servers, entries that don't
    tile every template's units, and two servers or more over a record that states none (module docstring)."""
    blocks = doc.get("blocks")
    inference = blocks.get("inference") if isinstance(blocks, Mapping) else None
    if inference is None:
        if len(servers) > 1:
            raise ValueError(f"{len(servers)} servers over a circuit record that states no inference block "
                             "(blocks.inference): an inference block runs on one server, and with two servers "
                             "or more the record's inference entries are what say which")
        return
    entries = inference.get("entries") if isinstance(inference, Mapping) else None
    if not isinstance(entries, list):
        raise ValueError(f"a circuit's inference block is its entries, a list, not {inference!r}")
    for e in entries:
        if not (isinstance(e, Mapping) and set(e) == set(_ENTRY) and isinstance(e["template"], str) and e["template"] in tpl
                and _nat(e["at"]) and _nat(e["step"]) and isinstance(e["radix"], list) and e["radix"]
                and all(_nat(r, 1) for r in e["radix"]) and isinstance(e["axes"], list)
                and len(e["axes"]) == len(e["radix"]) and all(a in _AXES for a in e["axes"])
                and e["axes"].count("slot") == 1 and e["axes"].count("step") <= 1):
            raise ValueError(f"inference entry {e!r}: {{{', '.join(_ENTRY)}}}, a template of the record's, a first unit and "
                             f"a step, and one axis of {_AXES} a digit of radix 1 or more, one the slot and at most one "
                             "the step")
    for t, (_r, _o, n) in tpl.items():
        end = 0
        for at, k in sorted((e["at"], math.prod(e["radix"])) for e in entries if e["template"] == t):
            if at != end:
                raise ValueError(f"template {t}: units [{min(at, end)}, {max(at, end)}) are "
                                 f"{'in no inference entry' if at > end else 'in two inference entries'}; the entries "
                                 "tile each template's units exactly once")
            end = at + k
        if end != n:
            raise ValueError(f"template {t}: inference entries tile units [0, {end}) of its {n}; they tile each "
                             "template's units exactly once")
    if len(servers) == 1:
        return
    rects = []
    for e in entries:
        radix, axes, at = e["radix"], e["axes"], e["at"]
        for lo, hi, s in runs[e["template"]]:
            for box in _boxes(max(lo, at) - at, min(hi, at + math.prod(radix)) - at, radix):
                slot = _digit(box, radix, axes.index("slot"))
                step = _digit(box, radix, axes.index("step")) if "step" in axes else (0, 1)
                rects.append((*slot, e["step"] + step[0], e["step"] + step[1], s, e["template"]))
    rects.sort()
    live: list[tuple[Any, ...]] = []
    for r in rects:
        live = [q for q in live if q[1] > r[0]]
        for q in live:
            if q[4] != r[4] and q[2] < r[3] and r[2] < q[3]:
                raise ValueError(f"inference block (slot {r[0]}, step {max(q[2], r[2])}): units of {q[5]} on server "
                                 f"{q[4]} and of {r[5]} on server {r[4]}; an inference block runs on one server, the one "
                                 "that may read its inputs")
        live.append(r)


def server_blocks(doc: Mapping[str, Any], server: Any, layouts: Mapping[str, Any]) -> tuple[Server, ...]:
    """The servers ``server`` (a blocks object's, or the record's ``blocks.server``) of the circuit record ``doc`` against
    ``layouts`` (as `check` takes them), refused (ValueError, naming the rule) unless each is a new name on a new uplink,
    together they partition every template's units, and each inference block of the record's runs on one of them."""
    tpl = _templates(doc, layouts)
    servers, runs = _servers(server, tpl)
    _one_server(doc, tpl, runs, servers)
    return servers


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


def _once(writes: Sequence[Pattern], words: int, rows: int, what: str) -> None:
    """Refuse an out value of ``rows`` rows of ``words`` words a row of which no write fills or two fill: a byte a row,
    each write's sources marked one strided slice at a time along its widest digit (sources in the value's leaves,
    `check`'s bounds)."""
    seen = bytearray(rows)
    for g in writes:
        k = max(range(len(g.radix)), key=g.radix.__getitem__)
        r, s = g.radix[k], g.step[k] // words
        outer = [(g.radix[i], g.step[i] // words) for i in range(len(g.radix)) if i != k]
        for d in itertools.product(*(range(n) for n, _s in outer)):
            a = g.base // words + sum(x * st for x, (_n, st) in zip(d, outer))
            if s == 0 and r > 1:
                raise ValueError(f"{what}: write {g.doc()} writes its row {a} from {r} units; an out value's every row is "
                                 "written by exactly one unit")
            lo = a + min(0, s * (r - 1))
            cells = slice(lo, lo + abs(s) * (r - 1) + 1, abs(s) or 1)
            if 1 in seen[cells]:
                row = lo + seen[cells].index(1) * (abs(s) or 1)
                raise ValueError(f"{what}: its row {row} is written twice; an out value's every row is written by exactly "
                                 "one unit")
            seen[cells] = b"\x01" * r
    if 0 in seen:
        raise ValueError(f"{what}: its row {seen.index(0)} of {rows} is written by no unit; an out value's every row is "
                         "written by exactly one unit")


def _steps(t: Any, rows: int, steps: int, what: str) -> Mapping[str, Any]:
    if not (isinstance(t, Mapping) and set(t) == {"base", "radix", "step"} and type(t["base"]) is int
            and isinstance(t["radix"], list) and isinstance(t["step"], list) and t["radix"] and len(t["radix"]) == len(t["step"])
            and all(_nat(r, 1) for r in t["radix"]) and all(type(s) is int for s in t["step"])):
        raise ValueError(f"{what}: its t is {{base, radix, step}}, one integer step a digit of radix 1 or more, not {t!r}")
    reach = [s * (r - 1) for r, s in zip(t["radix"], t["step"])]
    lo, hi = t["base"] + sum(min(0, x) for x in reach), t["base"] + sum(max(0, x) for x in reach)
    if math.prod(t["radix"]) != rows or lo < 0 or hi >= steps:
        raise ValueError(f"{what}: its t is over its {rows} rows, each at a step among the server-time block's {steps}; "
                         f"its radix makes {math.prod(t['radix'])} rows at steps [{lo}, {hi}]")
    return {"base": t["base"], "radix": list(t["radix"]), "step": list(t["step"])}


def _cut(doc: Mapping[str, Any], tid: str, port: str) -> int:
    k = (doc.get("cuts") or {}).get(tid, {}).get(port, 1)
    if not _nat(k, 1):
        raise ValueError(f"cuts: {tid} {port} cut {k!r} ways")
    return k


def check(doc: Mapping[str, Any], blocks: Any, layouts: Mapping[str, Any]) -> Blocks:
    """Refuse (ValueError, naming the rule) a circuit's blocks that break the module docstring's rules, against its stated
    circuit record ``doc`` (``circuit.json``: its templates, wires, external ports, cuts, registered reads and ingress,
    which the window interface's own checks have held well-formed, `check_inputs`, and its inference block, each unit's
    (slot, step), which puts each inference block on one server, `server_blocks`) and ``layouts``, each template's bit rows
    by descriptor id (the interface's `layout`: its ``rows`` and ``outs``, each ``(name, words)``), the verifier's own from
    its catalog. The work is per declaration, never per unit."""
    tpl = _templates(doc, layouts)
    if not isinstance(blocks, Mapping) or blocks.get("format") != FORMAT or set(blocks) != set(_KEYS):
        raise ValueError(f"a circuit's blocks are a {FORMAT} object of {list(_KEYS)}")
    servers, runs = _servers(blocks["server"], tpl)
    _one_server(doc, tpl, runs, servers)
    by_name = {s.name: s for s in servers}
    st = blocks["server_time"]
    if not (isinstance(st, Mapping) and set(st) == {"window", "steps"} and _nat(st["window"]) and _nat(st["steps"], 1)):
        raise ValueError(f"a circuit's server time is {{window, steps}}, a link-window and at least one step, not {st!r}")
    if doc.get("ingress"):
        raise ValueError("an ingress declaration: a server's ingress is a request transfer, a certified registered read, "
                         "never an Ingress")
    reg = doc.get("registered") or {}
    params = dict(reg.get("params", {}))
    reads = [_pattern(r, _READ, "registered read", cut=True) for r in reg.get("reads", ())]
    read_param = [r["param"] for r in reg.get("reads", ())]
    tables = list(reg.get("tables", ()))
    over = {t["name"]: t["param"] for t in tables}
    roots, openings = list(reg.get("roots", ())), list(reg.get("openings", ()))
    wires = [_pattern(w, _WIRE, "wire", cut=True) for w in doc.get("wires", ())]
    writers = [(w["writer"], w["out"]) for w in doc.get("wires", ())]
    external = set(doc.get("external", ()))

    def ext(tid: str, port: str) -> bool:
        return f"{tid}/{port}" in external

    def within(tid: str, lo: int, hi: int, server: str, what: str) -> None:
        if _server_of(runs, tid, lo, hi) != server:
            raise ValueError(f"{what}: units [{lo}, {hi}) of {tid} are not all server {server}'s")

    if not isinstance(blocks["transfers"], list):
        raise ValueError("a circuit's transfers are a list")
    transfers: list[Transfer] = []
    for d in blocks["transfers"]:
        out_ = isinstance(d, Mapping) and d.get("dir") == "out"
        if not isinstance(d, Mapping) or set(d) != {*_TRANSFER, *(_OUT if out_ else ())} or d["dir"] not in DIRS:
            raise ValueError(f"a transfer is {{{', '.join(_TRANSFER)}}}, dir in or out, and an out transfer's leaves and "
                             f"writes, not {d!r}")
        what = f"transfer {d['value']!r}"
        direction, kind, value = DIRS[d["dir"]], d["kind"], d["value"]
        if direction not in TR.KINDS.get(kind, ()):
            raise ValueError(f"{what}: a {kind!r} transfer isn't {d['dir']}")
        if d["server"] not in by_name:
            raise ValueError(f"{what}: server {d['server']!r} is no server of the circuit's")
        prior = [(t.kind, t.direction) for t in transfers if t.value == value]
        if not (isinstance(value, str) and value) or (prior and not (kind == "wire" and len(prior) == 1 and prior[0] == (
                "wire", Direction.EGRESS if direction is Direction.INGRESS else Direction.INGRESS))):
            raise ValueError(f"{what}: a transfer is of a new value, named, but a wire's other one (its out and its in)")
        if not _nat(d["window"]) or (kind != "load" and d["window"] != st["window"]):
            raise ValueError(f"{what}: a {kind}'s window is the server-time block's, {st['window']}, not {d['window']!r}")
        server = by_name[d["server"]]
        gates: list[Pattern] = []
        widths: set[int] = set()
        if direction is Direction.INGRESS:
            if value not in params:
                raise ValueError(f"{what}: brings in no registered parameter of the circuit's ({sorted(params)})")
            leaves = params[value]
            for r, p in zip(reads, read_param):
                if p != value:
                    continue
                k = _cut(doc, r.tid, r.port)
                w = tpl[r.tid][0][r.port] // k if r.tid in tpl and r.port in tpl[r.tid][0] else 0
                lo, hi = r.span()
                if not w or not r.aligned(w) or lo < 0 or hi + w > leaves:
                    raise ValueError(f"{what}: read {r.doc()} isn't whole rows of {w} words among the value's {leaves} leaves")
                within(r.tid, r.at, r.at + r.unit_count, server.name, f"{what}: read {r.doc()}")
                if ext(r.tid, r.port) != (kind in EXTERNAL):
                    raise ValueError(f"{what}: read {r.doc()}: a {kind}'s rows are {'' if kind in EXTERNAL else 'never '}the "
                                     "circuit's external ports")
                gates.append(r)
                widths.add(w)
            for t in tables:
                if t["param"] == value:
                    if t["offset"] != 0 or t["words"] * t["rows"] != leaves:
                        raise ValueError(f"{what}: table {t['name']} is a part of the value; a table over a transfer is the "
                                         "whole value, under the certifier's root")
                    widths.add(t["words"])
            for o in [*roots, *openings]:
                if over.get(o["table"]) == value:
                    within(o["reader"], o["at"], o["at"] + o["units"], server.name, f"{what}: table {o['table']}'s read")
                    for p in ([o["port"]] if "port" in o else o["ports"]):
                        if ext(o["reader"], p):
                            raise ValueError(f"{what}: table {o['table']}'s read fills {o['reader']}/{p}, an external port")
            if not widths:
                raise ValueError(f"{what}: no unit reads it")
        else:
            leaves = d["leaves"]
            if not _nat(leaves, 1) or not isinstance(d["writes"], list) or not d["writes"]:
                raise ValueError(f"{what}: an out transfer is its value's leaves and at least one write")
            for x in d["writes"]:
                g = _pattern(x, _WRITE, f"{what}'s write")
                w = tpl[g.tid][1].get(g.port, 0) if g.tid in tpl else 0
                lo, hi = g.span()
                if not w or g.at + g.units > tpl[g.tid][2]:
                    raise ValueError(f"{what}: write {g.doc()} is no output row of its template's units")
                if not g.aligned(w) or lo < 0 or hi + w > leaves:
                    raise ValueError(f"{what}: write {g.doc()} isn't whole rows of {w} words among the value's {leaves} leaves")
                within(g.tid, g.at, g.at + g.units, server.name, f"{what}: write {g.doc()}")
                if ext(g.tid, g.port) != (kind in EXTERNAL):
                    raise ValueError(f"{what}: write {g.doc()}: a {kind}'s rows are {'' if kind in EXTERNAL else 'never '}the "
                                     "circuit's external ports")
                gates.append(g)
                widths.add(w)
        if len(widths) != 1 or leaves % next(iter(widths)):
            raise ValueError(f"{what}: its reads are rows of {sorted(widths)} words, and its {leaves} leaves; a value is "
                             "rows of one width")
        words = next(iter(widths))
        if direction is Direction.EGRESS:
            _once(gates, words, leaves // words, what)
        if kind == "load" and d["t"] is not None:
            raise ValueError(f"{what}: a load crosses once a server session, at no step of the window: its t is null")
        t = None if kind == "load" else _steps(d["t"], leaves // words, st["steps"], what)
        transfers.append(Transfer(server.name, server.uplink, direction, kind, value, d["window"], words, leaves // words,
                                  t, tuple(gates)))
    for v in dict.fromkeys(t.value for t in transfers if t.kind == "wire"):
        pair = {t.direction: t for t in transfers if t.value == v}
        if set(pair) != {Direction.EGRESS, Direction.INGRESS}:
            raise ValueError(f"wire {v!r}: its out and its in, one each; a wire leaves one server's uplink and reaches "
                             "another's")
        o, i = pair[Direction.EGRESS], pair[Direction.INGRESS]
        if o.server == i.server:
            raise ValueError(f"wire {v!r}: from server {o.server} to itself; a wire crosses from one server to another, and "
                             "within one server it is a wire of the circuit's")
        if (o.rows, o.words) != (i.rows, i.words):
            raise ValueError(f"wire {v!r}: its out is {o.rows} rows of {o.words} words and its in {i.rows} rows of "
                             f"{i.words}; a wire's in is the rows of its out")
        if o.t != i.t:
            raise ValueError(f"wire {v!r}: its out's rows cross at steps {o.t} and its in's at {i.t}; each row of a wire "
                             "crosses both uplinks at one step")
    brought = {t.value for t in transfers if t.direction is Direction.INGRESS}
    for p in [*read_param, *over.values()]:
        if p not in brought:
            raise ValueError(f"a registered read of {p!r}, which no transfer brings in: a registered parameter crosses a "
                             "server's boundary, as a load")
    for w, (wt, _o) in zip(wires, writers):
        lo, hi = w.span()
        a, b = _server_of(runs, w.tid, w.at, w.at + w.unit_count), _server_of(runs, wt, lo, hi + 1)
        if a is None or a != b:
            raise ValueError(f"wire {w.doc()} from {wt}: its reader and writer units are not one server's; a value one "
                             "server sends another is a wire transfer, the sender's out and the receiver's in")
    writes = [g for t in transfers if t.direction is Direction.EGRESS for g in t.gates]
    _disjoint([((g.tid, g.port), g.at, g.at + g.units, 0, 1) for g in writes], "writes")
    ins = [g for t in transfers if t.direction is Direction.INGRESS for g in t.gates]
    for e in sorted(external):
        tid, _, port = e.rpartition("/")
        if tid not in tpl:
            raise ValueError(f"external port {e}: no template's")
        rows_, outs_, n = tpl[tid]
        if port in rows_:
            k = _cut(doc, tid, port)
            rects = [((tid, port), g.at, g.at + g.unit_count, g.cut[0], g.cut[0] + g.cut[1]) for g in ins
                     if (g.tid, g.port) == (tid, port)]
            _disjoint(rects, "reads")
            covered, want = sum((b - a) * (r1 - r0) for _k, a, b, r0, r1 in rects), n * k
        elif port in outs_:
            covered, want = sum(g.units for g in writes if (g.tid, g.port) == (tid, port)), n
        else:
            raise ValueError(f"external port {e}: no port of {tid}")
        if covered != want:
            raise ValueError(f"external port {e}: {want - covered} of its {want} unit rows are no transfer's; every row of an "
                             "external port crosses its server's boundary as a transfer the network certifier signed")
    return Blocks(servers, st["window"], st["steps"], tuple(transfers))


def certified(doc: Mapping[str, Any], blocks: Any, layouts: Mapping[str, Any], registrations: Iterable[Any],
              certifiers: Mapping[str, tuple[PublicKey, bytes]],
              formats: Mapping[str, tuple[int, str]] | None = None) -> dict[tuple[str, Direction], TR.TransferHead]:
    """Each transfer's head, by value and direction, from its registration (`transfer_rows.check_registration`, for the
    transfer `check` makes of the record, at the record's program digest) under ``certifiers[uplink]``, its uplink's
    network certifier's key and session. ``formats``: the verifier's own ``(word_bits, schema)`` of a value whose leaves
    aren't 16 bits or whose rows aren't ``hm96-sha512/row/v2`` (its program's parameter width, `registered.width`, and
    its port's row format), a row of ``words`` u16 words then ``words · 16 / word_bits`` leaves. Refused (ValueError)
    when the blocks break `check`, a transfer has no registration or its registration fails, a registration names no
    transfer, or a wire's in root isn't its out root."""
    b = check(doc, blocks, layouts)
    formats = dict(formats or {})
    if extra := sorted(set(formats) - {t.value for t in b.transfers}):
        raise ValueError(f"formats of {extra}, which no transfer names")
    regs: dict[tuple[str, Direction], Any] = {}
    for r in registrations:
        v, d = (r.get("value"), r.get("direction")) if isinstance(r, Mapping) else (None, None)
        named = (v, next((x for x in Direction if x.value == d), None))
        if not isinstance(v, str) or named[1] is None or named in regs:
            raise ValueError(f"a transfer registration names a value and a direction, once: {v!r} {d!r}")
        regs[named] = r
    if extra := sorted(f"{v} {d.value}" for v, d in set(regs) - {(t.value, t.direction) for t in b.transfers}):
        raise ValueError(f"registrations of {extra}, which no transfer names: a root is pinned only for a transfer the blocks "
                         "declare")
    heads = {}
    for t in b.transfers:
        what = f"transfer {t.value!r} ({t.kind} {t.direction.value} on {t.uplink}, window {t.window})"
        if (t.value, t.direction) not in regs:
            raise ValueError(f"{what}: no certificate; a transfer's root is pinned from its network certifier's certificate "
                             "alone")
        if t.uplink not in certifiers:
            raise ValueError(f"{what}: no network certifier on {t.uplink}")
        key, session = certifiers[t.uplink]
        bits, schema = formats.get(t.value, (TR.WORD_BITS, TR.SCHEMA))
        try:
            if bits not in TR.WIDTHS or 16 * t.words % bits:
                raise ValueError(f"its rows of {t.words} u16 words are no whole number of {bits!r}-bit leaves")
            heads[t.value, t.direction], _ = TR.check_registration(
                regs[t.value, t.direction], key, session, link=t.uplink, direction=t.direction, window=t.window,
                kind=t.kind, value=t.value, program=doc.get("program"), words=16 * t.words // bits, rows=t.rows,
                word_bits=bits, schema=schema)
        except ValueError as e:
            raise ValueError(f"{what}: {e}") from None
    for t in b.transfers:
        if t.kind == "wire" and t.direction is Direction.INGRESS:
            o, i = heads[t.value, Direction.EGRESS], heads[t.value, Direction.INGRESS]
            if o.root != i.root:
                raise ValueError(f"wire {t.value!r}: the root {i.link}'s network certifier signed isn't the one {o.link}'s "
                                 "signed; the rows that reached the receiver aren't the rows that left the sender, or aren't "
                                 "under the sender's salts")
    return heads


def pins(doc: Mapping[str, Any], blocks: Any, layouts: Mapping[str, Any], registrations: Iterable[Any],
         certifiers: Mapping[str, tuple[PublicKey, bytes]],
         formats: Mapping[str, tuple[int, str]] | None = None) -> dict[str, str]:
    """The verifier's pins, ``{parameter: root}``: every ``in`` transfer's value at its certified root (`certified`), the
    roots ``registration.check`` holds a pinned value to and the parameters the interface's ``input_count`` counts as
    registered."""
    return {v: h.root.hex() for (v, d), h in certified(doc, blocks, layouts, registrations, certifiers, formats).items()
            if d is Direction.INGRESS}
