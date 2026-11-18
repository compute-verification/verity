"""`Q_fit` v1: proof blocks that fit a Flock proof block by rows and bits, cut from C with no Definition written to fit.

**Units.** The root's Calls (`partition_object.calls`), each a `call` or `batch` node: a call one unit, a batch member one, in
node order then member order, as the template queries number them but of any Definition.

**Blocks** (the object's ``blocks``, required). Each entry is one block, a set of gates inside one activation:

* ``[lo, hi]``: root units lo .. hi - 1, whole (the template queries' form);
* ``{"path": [[k, m], ...], "nodes": [a, b]}``: body nodes a .. b - 1, whole, of the activation the path reaches;
* ``{"path": [[k, m], ...], "node": j, "members": [a, b]}``: members (or iterations) a .. b - 1 of its node j.

A path starts at the root activation and is not empty; step ``[k, m]`` enters member m of body node k (m = 0 for a call),
whose Definition is a composite. The entries are in C's layout order and cover the root's Calls' gates exactly once: each
starts at or past where the one before ends, and together they hold every gate of the Calls (`cover`).

**Rows** (`Fit.rows`). A block's rows are those a Flock proof block opens: one per writer row its inputs touch and one per
run of its outputs. Its outputs out(S), in increasing index, are cut into runs at every change of width; each run is one row,
the block's writer row. An input gate of in(S) that a block computes lies in one writer row of that block; one that an input
node of the root gives lies in that node's row. A block's input rows are the distinct writer rows in(S) touches, each read
whole. in(S) and out(S) are `boundary`'s, computed from the descriptor, never by walking C's gates.

**Bits** (`Fit.k_log`). `proof_block_k_log` of the block's input rows' bytes, its output rows' bytes and its ANDs
(`count`), unpacked: the Flock block one proof block needs.

**The pin** (`Measure.pinned`). Some row of the block, in or out, is past `LAST_BLOCK_BYTES` bytes: otherwise none has a
compression before its last SHA-512 block and Flock's sha512x3 range, which holds the pin, has no slot
(`periods.interface.check_proof_block`'s refusal).

`verify` refuses a block past ``rows`` rows (`fit-rows`), whose Flock block is past ``bits`` bits (`fit-bits`), or with no
pin (`fit-pin`), by entry. `build` packs C greedily in layout order, descending into a unit, a node or a member only when it
alone does not fit, joins a block with no pin to the one before it where rows and bits allow, and refuses (`Unfit`) a gate
that alone does not fit, or a block with no pin it cannot join, before anything is committed.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NamedTuple, Sequence

from . import boundary as BD
from . import count as K
from .codec import descriptor_id
from .defs import PrimitiveDefinition
from .layout import Scope
from .parts import Part, Unsupported, n_members
from .program import is_input

#: Flock's composer's sizes, which `verity_flock.circuit` and the benchmarks read (through `periods.interface`): a Flock block
#: of at most 2^K_MAX bits and at least 2^K_MIN, the sha512x3 and hm96 slots of 2^SLOT_LOG bits, PER_SLOT compressions a
#: sha512x3 slot, the mask's 2^MASK_LOG, and 2^PACK_LOG bits, up to which it packs a template's proof blocks into one block
K_MAX, K_MIN, SLOT_LOG, PER_SLOT, MASK_LOG, PACK_LOG = 27, 20, 18, 3, 14, 22
#: the row bytes a row's last SHA-512 block holds beside the padding's 17: a longer row has a compression before it
LAST_BLOCK_BYTES = 111
#: the most rows a proof block may have: at one more, its hm96 range alone fills a Flock block of 2^K_MAX bits
MAX_ROWS = 1 << (K_MAX - SLOT_LOG - 1)

Q_FIT = ("Q_fit", 1)
BITS_MAX = 1 << 40


def proof_block_k_log(in_bytes: Sequence[int], out_bytes: Sequence[int] = (), ands: int = 0, *,
                      packed: bool = True) -> int:
    """The ``k_log`` Flock's composer gives a template's proof block (`verity_flock.circuit.compose`'s META ``k_log``,
    which the Lean verifier reads): input rows of ``in_bytes`` bytes, output rows of ``out_bytes``, and ``ands`` rows of
    logic between them (its ANDs, inline table reads' rows and assertions), its statement as `UnitStager` stages
    `unit_rows`. The proof block's circuit lays out its input bits, the logic, its output bits and a constant, each of the
    two port groups an aligned power of two of 128-bit words, in a slot of at least 2^7 cells. A Flock block holds the
    hm96 slots (one a row, a power of two in all), that slot and the mask, then the sha512x3 slots; and the composer
    packs proof blocks into it while it stays within 2^`PACK_LOG` bits or one proof block's size, which doubling, at most
    doubling the block, ends at exactly the larger. ``packed=False``: one proof block's own block before that packing
    (compose's ``k1``, at least 2^`K_MIN`), which a sampling plan's statement cap reads, counting the packing itself."""
    def cells(row_bytes: Sequence[int]) -> int:
        return 128 << (-(-8 * sum(row_bytes) // 128) - 1).bit_length() if sum(row_bytes) else 0

    rows = [*in_bytes, *out_bytes]
    i, o = cells(in_bytes), cells(out_bytes)
    useful = (-(-(i + ands) // o) * o if o else i + ands) + o + 1
    plain = ((1 << (len(rows) - 1).bit_length()) << SLOT_LOG) + (1 << max(7, (useful - 1).bit_length())) + (1 << MASK_LOG)
    comps = sum(-(-(b - LAST_BLOCK_BYTES) // 128) for b in rows if b > LAST_BLOCK_BYTES)
    bits = (-(-plain >> SLOT_LOG) + -(-comps // PER_SLOT)) << SLOT_LOG
    return max(PACK_LOG if packed else K_MIN, (bits - 1).bit_length())


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def params_ok(p: Mapping[str, Any]) -> bool:
    return _is_int(p.get("rows")) and 1 <= p["rows"] <= 1 << 16 and _is_int(p.get("bits")) and 1 <= p["bits"] <= BITS_MAX


def query(rows: int = MAX_ROWS, bits: int = 1 << K_MAX) -> dict[str, Any]:
    """`Q_fit{rows, bits}` v1, Flock's composer's bounds by default."""
    if not params_ok({"rows": rows, "bits": bits}):
        raise ValueError(f"Q_fit's rows are 1..65536 and its bits 1..2^40, not {rows!r}, {bits!r}")
    return {"name": Q_FIT[0], "version": Q_FIT[1], "params": {"rows": rows, "bits": bits}}


class Unfit(ValueError):
    """What `Q_fit` refuses at registration, by code: ``fit-gate`` (a gate that alone does not fit), ``fit-pin`` (a block
    with no pin that joins no neighbour), ``fit-form`` (a root node or a node to descend into that the query does not cut),
    ``fit-unsupported`` (a boundary `boundary` cannot give exactly), ``fit-no-body`` (a word primitive with no Boolean
    body)."""

    def __init__(self, code: str, what: str) -> None:
        super().__init__(f"{code}: {what}")
        self.code = code


class FitPartition:
    """`Q_fit` v1's units on a program: ``ranges`` per root Call (root-body index, first unit, units), ``population``,
    ``locate(i)`` -> (root-body index, member), ``gates(i)`` the unit's gates [lo, hi)."""

    def __init__(self, program: Any, q: Mapping[str, Any]) -> None:
        from .partition_object import QueryInapplicable, calls
        if (q.get("name"), q.get("version")) != Q_FIT or not isinstance(q.get("params"), Mapping) or \
                set(q["params"]) != {"rows", "bits"} or not params_ok(q["params"]):
            raise QueryInapplicable(f"not Q_fit v1 with its parameters: {dict(q)!r}", {"query": dict(q)})
        self.query, self.program = dict(q), program
        self.rows, self.bits = q["params"]["rows"], q["params"]["bits"]
        nodes = program.root.body.nodes
        self.ranges: list[tuple[int, int, int]] = []
        at = 0
        for k, _fn in calls(program):
            if nodes[k].form not in ("call", "batch"):
                raise QueryInapplicable(f"root node {k} is a {nodes[k].form}: a unit of Q_fit is a call or a batch member",
                                        {"call": k, "form": nodes[k].form})
            n = nodes[k].n if nodes[k].form == "batch" else 1
            self.ranges.append((k, at, n))
            at += n
        self.population = at
        self._start = [a for _k, a, _n in self.ranges]

    def locate(self, i: int) -> tuple[int, int]:
        if not 0 <= i < self.population:
            raise IndexError(f"unit {i} outside 0..{self.population - 1}")
        k, a, _n = self.ranges[bisect.bisect_right(self._start, i) - 1]
        return k, i - a

    def gates(self, i: int) -> tuple[int, int]:
        k, m = self.locate(i)
        body = self.program.root.body
        lo = body.offsets[k] + m * body.nodes[k].fn.gates
        return lo, lo + body.nodes[k].fn.gates


def _pair(e: Any) -> bool:
    return isinstance(e, list) and len(e) == 2 and all(_is_int(x) for x in e) and 0 <= e[0] < e[1]


def _range(r: Any) -> bool:
    return _pair(r)


def entry_form(e: Any) -> bool:
    """An entry of the module docstring's three forms, by its shape alone."""
    if _pair(e):
        return True
    if not isinstance(e, Mapping) or "path" not in e:
        return False
    p = e["path"]
    if not (isinstance(p, list) and p and all(isinstance(s, list) and len(s) == 2 and all(_is_int(x) and x >= 0 for x in s) for s in p)):
        return False
    if set(e) == {"path", "nodes"}:
        return _range(e["nodes"])
    return set(e) == {"path", "node", "members"} and _is_int(e["node"]) and e["node"] >= 0 and _range(e["members"])


def blocks_form(b: Any, fail: Callable[[str, Any], None]) -> None:
    """`validate`'s check of a `Q_fit` object's ``blocks``: a non-empty list of entries of the three forms (`blocks-form`)."""
    if not isinstance(b, list) or not b or not all(map(entry_form, b)):
        fail("blocks-form", next((e for e in b if not entry_form(e)), b) if isinstance(b, list) and b else b)


def entry_doc(scope_path: Sequence[tuple[int, int]], kind: str, j: int, lo: int, hi: int) -> Any:
    if kind == "units":
        return [lo, hi]
    path = [[k, m] for k, m in scope_path]
    return {"path": path, "nodes": [lo, hi]} if kind == "nodes" else {"path": path, "node": j, "members": [lo, hi]}


def part_of(program: Any, fp: FitPartition, e: Any) -> Part:
    """Entry ``e``'s gates as a `Part`; ValueError where it names no gates of C (`fit-entry`)."""
    root = program.circuit.root_scope
    if _pair(e):
        lo, hi = e
        if hi > fp.population:
            raise ValueError(f"units [{lo}, {hi}) past the population's {fp.population}")
        return Part(root, frozenset(fp.locate(u) for u in range(lo, hi)))
    scope = root
    for k, m in e["path"]:
        body = scope.spec.body
        if k >= len(body.nodes):
            raise ValueError(f"path step {[k, m]}: {scope.spec.id} has {len(body.nodes)} nodes")
        node = body.nodes[k]
        if isinstance(node.fn, PrimitiveDefinition) or node.form == "primitive" or m >= n_members(node) or \
                (node.form == "call" and m):
            raise ValueError(f"path step {[k, m]}: node {k} of {scope.spec.id} has no composite member {m}")
        if scope is root and is_input(node.fn):
            raise ValueError(f"path step {[k, m]}: an input node")
        scope = scope.child(k, m)
    body = scope.spec.body
    if "nodes" in e:
        a, b = e["nodes"]
        if b > len(body.nodes):
            raise ValueError(f"nodes [{a}, {b}) past {scope.spec.id}'s {len(body.nodes)}")
        items = frozenset((j, m) for j in range(a, b) for m in range(n_members(body.nodes[j])))
    else:
        j, (a, b) = e["node"], e["members"]
        if j >= len(body.nodes) or b > n_members(body.nodes[j]):
            raise ValueError(f"members [{a}, {b}) of node {j}: past {scope.spec.id}'s nodes or that node's members")
        items = frozenset((j, m) for m in range(a, b))
    if not items:
        raise ValueError(f"entry {e!r} holds no gates")
    return Part(scope, items)


def _calls_gates(program: Any, fp: FitPartition) -> int:
    body = program.root.body
    return sum(body.offsets[k + 1] - body.offsets[k] for k, _a, _n in fp.ranges)


def cover(program: Any, fp: FitPartition, parts: Sequence[Part]) -> list[tuple[str, Any]]:
    """Refusals of the parts as a cover of the root Calls' gates, in layout order and each gate once: ``blocks-unsorted`` (an
    entry starting before the one before ends), ``blocks-gap`` (gates of the Calls in none)."""
    bad: list[tuple[str, Any]] = []
    end, total = 0, 0
    for i, p in enumerate(parts):
        ivs = p.intervals()
        if ivs[0][0] < end:
            bad.append(("blocks-unsorted", [i, ivs[0][0], end]))
        end = max(end, ivs[-1][1])
        total += sum(hi - lo for lo, hi in ivs)
    want = _calls_gates(program, fp)
    if total != want and not bad:
        bad.append(("blocks-gap", [total, want]))
    return bad


@dataclass
class _Writers:
    """The finished blocks' writer rows: each block's gate intervals, its out(S) in increasing index and each one's row."""
    starts: list[int] = field(default_factory=list)
    ivs: list[tuple[int, int, int]] = field(default_factory=list)
    outs: list[list[int]] = field(default_factory=list)
    run_of: list[list[int]] = field(default_factory=list)
    run_bytes: list[list[int]] = field(default_factory=list)

    def pop(self) -> None:
        b = len(self.outs) - 1
        keep = [i for i, (_lo, _hi, x) in enumerate(self.ivs) if x != b]
        self.starts = [self.starts[i] for i in keep]
        self.ivs = [self.ivs[i] for i in keep]
        for xs in (self.outs, self.run_of, self.run_bytes):
            xs.pop()

    def add(self, part: Part, outs: list[int], widths: list[int]) -> None:
        b = len(self.outs)
        for lo, hi in part.intervals():
            at = bisect.bisect_left(self.starts, lo)
            self.starts.insert(at, lo)
            self.ivs.insert(at, (lo, hi, b))
        runs, rb = [], []
        for i, w in enumerate(widths):
            if i == 0 or w != widths[i - 1]:
                rb.append(0)
            runs.append(len(rb) - 1)
            rb[-1] += w
        self.outs.append(outs)
        self.run_of.append(runs)
        self.run_bytes.append([-(-x // 8) for x in rb])

    def row(self, g: int) -> tuple[tuple, int] | None:
        at = bisect.bisect_right(self.starts, g) - 1
        if at < 0:
            return None
        lo, hi, b = self.ivs[at]
        if not lo <= g < hi:
            return None
        i = bisect.bisect_left(self.outs[b], g)
        if i == len(self.outs[b]) or self.outs[b][i] != g:
            return None
        r = self.run_of[b][i]
        return ("w", b, r), self.run_bytes[b][r]


class Measure(NamedTuple):
    """A block's rows, its Flock block's ``k_log``, its out(S) with their widths, and whether a row holds the pin."""
    rows: int
    k: int
    outs: list[int]
    widths: list[int]
    pinned: bool


class Fit:
    """A program's rows and bits of a proof block (module docstring), over the blocks finished so far (`finish`)."""

    def __init__(self, program: Any) -> None:
        self.program = program
        self.writers = _Writers()
        self._counts: dict[str, K.Count] | None = None
        self._input_rows: dict[int, int] = {}
        body = program.root.body
        self._input_nodes = [k for k, n in enumerate(body.nodes) if is_input(n.fn)]
        self._input_starts = [body.offsets[k] for k in self._input_nodes]

    def _ports(self, part: Part) -> tuple[list[int], list[int]]:
        ins, outs = BD.in_gates(part), BD.out_gates(part)
        for x in (ins, outs):
            if isinstance(x, Unsupported):
                raise Unfit("fit-unsupported", f"{part.scope.spec.id} items {sorted(part.items)[:4]}...: {x.reason} {x.detail}")
        return sorted(ins), sorted(outs)

    def _width(self, g: int) -> int:
        return BD.gate_width(self.program, g)

    def _input_row(self, g: int) -> tuple[tuple, int] | None:
        at = bisect.bisect_right(self._input_starts, g) - 1
        if at < 0:
            return None
        k = self._input_nodes[at]
        body = self.program.root.body
        if not body.offsets[k] <= g < body.offsets[k + 1]:
            return None
        if k not in self._input_rows:
            node = body.nodes[k]
            self._input_rows[k] = -(-n_members(node) * node.fn.ret.w // 8)
        return ("in", k), self._input_rows[k]

    def _ands(self, part: Part) -> int:
        if self._counts is None:
            try:
                self._counts = K.count(self.program)
            except K.NoBody as e:
                raise Unfit("fit-no-body", str(e)) from None
        body = part.scope.spec.body
        return sum(self._counts[descriptor_id(body.nodes[j].fn)].ands for j, _m in part.items)

    def measure(self, part: Part) -> Measure:
        """The part as a block over the finished blocks."""
        ins, outs = self._ports(part)
        widths = [self._width(g) for g in outs]
        rows: dict[tuple, int] = {}
        for g in ins:
            got = self._input_row(g) or self.writers.row(g)
            if got is None:
                raise Unfit("fit-order", f"gate {g} that {part.scope.spec.id}'s part reads is in no earlier block nor input")
            rows[got[0]] = got[1]
        out_bytes = []
        for i, w in enumerate(widths):
            if i == 0 or w != widths[i - 1]:
                out_bytes.append(0)
            out_bytes[-1] += w
        out_bytes = [-(-x // 8) for x in out_bytes]
        k = proof_block_k_log(list(rows.values()), out_bytes, self._ands(part), packed=False)
        pinned = any(b > LAST_BLOCK_BYTES for b in [*rows.values(), *out_bytes])
        return Measure(len(rows) + len(out_bytes), k, outs, widths, pinned)

    def finish(self, part: Part, outs: list[int], widths: list[int]) -> None:
        self.writers.add(part, outs, widths)


def verify(program: Any, fp: FitPartition, entries: Sequence[Any], bad: dict[str, list],
           inference: Callable[[int], Any] | None = None) -> None:
    """`partition_object.verify`'s checks of a `Q_fit` object's ``blocks`` (module docstring), into ``bad`` by code: an entry
    naming no gates of C (``fit-entry``), the cover (`cover`), every block within ``rows`` (``fit-rows``) and ``bits``
    (``fit-bits``) and with a pin (``fit-pin``), and, given ``inference`` (a unit's inference block), no block of units of two (``fit-straddles``)."""
    parts = []
    for i, e in enumerate(entries):
        try:
            parts.append(part_of(program, fp, e))
        except ValueError as x:
            bad.setdefault("fit-entry", []).append([i, str(x)])
    if len(parts) != len(entries):
        return
    for code, what in cover(program, fp, parts):
        bad.setdefault(code, []).append(what)
    if bad:
        return
    if inference is not None:
        for i, e in enumerate(entries):
            us = range(e[0], e[1]) if _pair(e) else [_unit_of(fp, e)]
            if len({inference(u) for u in us}) > 1:
                bad.setdefault("fit-straddles", []).append([i, e])
    fit = Fit(program)
    try:
        for i, p in enumerate(parts):
            rows, k, outs, widths, pinned = fit.measure(p)
            if rows > fp.rows:
                bad.setdefault("fit-rows", []).append([i, rows, fp.rows])
            if (1 << k) > fp.bits:
                bad.setdefault("fit-bits", []).append([i, k, fp.bits])
            if not pinned:
                bad.setdefault("fit-pin", []).append([i, LAST_BLOCK_BYTES])
            fit.finish(p, outs, widths)
    except Unfit as x:
        bad.setdefault(x.code, []).append(str(x))


def _unit_of(fp: FitPartition, e: Mapping[str, Any]) -> int:
    k, m = e["path"][0]
    for kk, a, n in fp.ranges:
        if kk == k:
            return a + m
    raise ValueError(f"path step {e['path'][0]}: no root Call")


def _computes(node: Any) -> bool:
    return n_members(node) > 0 and node.fn.gates > 0


def build(program: Any, rows: int = MAX_ROWS, bits: int = 1 << K_MAX, cap: int | None = None) -> list[Any]:
    """`Q_fit{rows, bits}`'s blocks of ``program`` (module docstring's entries), packed greedily in layout order: as many whole
    root units as fit (at most ``cap`` a block), else into the unit, packing its nodes, members and their bodies alike.
    Refuses (`Unfit`) a gate that alone does not fit, and a scan or primitive the cut would have to enter."""
    fp = FitPartition(program, query(rows, bits))
    fit = Fit(program)
    out: list[Any] = []
    root = program.circuit.root_scope

    def fits(part: Part):
        m = fit.measure(part)
        return (m.rows <= rows and (1 << m.k) <= bits), m

    def emit(part: Part, doc: Any, m: Measure) -> None:
        fit.finish(part, m.outs, m.widths)
        out.append(doc)

    def greedy(n: int, make: Callable[[int, int], Part], doc: Callable[[int, int], Any], into: Callable[[int], None],
               limit: int | None = None) -> None:
        a, last = 0, None
        while a < n:
            top = n if limit is None else min(n, a + limit)
            ok, m = fits(make(a, a + 1))
            if not ok:
                into(a)
                a, last = a + 1, None
                continue
            good, gb, bad, step = m, a + 1, None, 1
            while gb < top:
                b = min(top, gb + step)
                ok, m = fits(make(a, b))
                if not ok:
                    bad = b
                    break
                good, gb, step = m, b, step * 2
            while bad is not None and bad - gb > 1:
                mid = (gb + bad) // 2
                ok, m = fits(make(a, mid))
                if ok:
                    good, gb = m, mid
                else:
                    bad = mid
            if not good.pinned:
                start = a
                if last is not None and (limit is None or gb - last <= limit):
                    fit.writers.pop()
                    out.pop()
                    ok, m = fits(make(last, gb))
                    if ok and m.pinned:
                        start, good = last, m
                if not good.pinned:
                    raise Unfit("fit-pin", f"{doc(a, gb)!r}: no row past {LAST_BLOCK_BYTES} B, so Flock's sha512x3 range, "
                                           "which holds the pin, has no slot, and joining it to the block before it does "
                                           "not fit or does not give one")
                a = start
            emit(make(a, gb), doc(a, gb), good)
            a, last = gb, a

    def body_of(scope: Scope, path: tuple) -> None:
        body = scope.spec.body
        idx = [j for j, nd in enumerate(body.nodes) if _computes(nd)]
        if not idx:
            return
        # nodes without gates between computing ones join the range around them; leading and trailing ones belong to none
        def make(a: int, b: int) -> Part:
            return Part(scope, frozenset((j, m) for j in range(idx[a], idx[b - 1] + 1) for m in range(n_members(body.nodes[j]))))

        greedy(len(idx), make, lambda a, b: entry_doc(path, "nodes", 0, idx[a], idx[b - 1] + 1),
               lambda a: node_of(scope, path, idx[a]))

    def node_of(scope: Scope, path: tuple, j: int) -> None:
        node = scope.spec.body.nodes[j]
        if isinstance(node.fn, PrimitiveDefinition):
            if node.form == "batch":
                greedy(node.n, lambda a, b: Part(scope, frozenset((j, m) for m in range(a, b))),
                       lambda a, b: entry_doc(path, "members", j, a, b),
                       lambda a: _refuse(scope, j, a))
                return
            _refuse(scope, j, 0)
        if node.form == "scan":
            raise Unfit("fit-form", f"node {j} of {scope.spec.id} is a scan that alone does not fit; Q_fit v1 cuts no scan")
        if node.form == "call":
            body_of(scope.child(j, 0), path + ((j, 0),))
            return
        greedy(node.n, lambda a, b: Part(scope, frozenset((j, m) for m in range(a, b))),
               lambda a, b: entry_doc(path, "members", j, a, b),
               lambda a: body_of(scope.child(j, a), path + ((j, a),)))

    def _refuse(scope: Scope, j: int, m: int) -> None:
        raise Unfit("fit-gate", f"member {m} of node {j} of {scope.spec.id}, one gate, has more than {rows} rows or {bits} "
                                "bits alone: no cut fits it")

    def unit_part(a: int, b: int) -> Part:
        return Part(root, frozenset(fp.locate(u) for u in range(a, b)))

    def into_unit(u: int) -> None:
        k, m = fp.locate(u)
        body_of(root.child(k, m), ((k, m),))

    greedy(fp.population, unit_part, lambda a, b: [a, b], into_unit, cap)
    return out


__all__ = ["BITS_MAX", "FitPartition", "Fit", "Measure", "K_MAX", "K_MIN", "LAST_BLOCK_BYTES", "MASK_LOG", "MAX_ROWS", "PACK_LOG", "PER_SLOT",
           "Q_FIT", "SLOT_LOG", "Unfit", "blocks_form", "build", "cover", "entry_form", "params_ok", "part_of", "proof_block_k_log",
           "query", "verify"]
