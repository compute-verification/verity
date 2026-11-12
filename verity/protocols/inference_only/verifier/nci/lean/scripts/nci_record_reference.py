"""The inference-only policy's check of an execution phase's circuit record, unit by unit: the reference
`NciCheck.Record` is held to.

    check_detail(instance, trusted) -> {"blocks", "gates", "inputs", "fresh", "cross_wires", "free", "ingress",
                                        "unregistered", "openings", "wired_rows", "wired_bits", "complies", "failure"}
                                       | {"error": code} | {"error": code, "message": what to fix}
    check(instance, trusted) -> {"complies"} | {"error": code} | {"error": code, "message": what to fix}
    kinds(instance, trusted, rows) -> [kind, …] | {"error": code}
    exported(instance, trusted, rows) -> [[slot, why the row is θ or None], …] | {"error": code}
    count(instance, trusted, limits) -> {"bool_gates", "complies", "failure"} | {"error": code}

`Record`'s docstring gives the instance, the trusted input, the stages and their refusals, and what each field
of the summary counts; `RecordCount`'s gives the count's limits. Where the Lean reads each template's gates
from the Program's own count and its rows from its Definition's parameters, takes one closed form per wire and per
declaration, and multiplies through the slots, this file expands: each template's gates gate by gate
(`nci_reference.activation_gates`), every unit's slot from the inference entries' digits, every input row of every
unit with what fills it, and every reader row of every wire with its writer. A block is one slot's units; a wire is
inside a block when every reader row it covers is read from a writer of the reader's slot, and every row of any other
wire counts as read across blocks. Every input row of every unit is classed once (`KINDS`, by `expand`'s `kind`): one
nothing shows inside its block or θ is fresh; and it counts whole, 16 bits a word, whether or not a gate reads it.

Five rules are no expansion. The wires' own check is SPOT's (`Spot.Wires.check`, which the Lean runs on the wires as
`Spot.Wires.Wiring.ofJson` reads them); here it is Python's statement of the same rule,
`verity.commitments.windows.interface.check_wires`. θ's rule, a pinned reader's layout and the rule for a table opening are the
Lean's, restated (`theta`, `layout` and `roles`, `held`): nothing unrolls a registration log, and a layout is read from
a template's signature. A Definition's closure digest is the period check's (`closure_digest`).

Two refusals are the Lean's rule restated and then expanded, so that what the rule stands for is held on every case
it passes. `own_rows`: the root's arguments in the form the Lean reads, then every leaf of every row of every unit
resolved to its gate, an input gate and no other leaf's. `leaf_widths`: a wire into a closed row (one-bit leaves its
Definition passes none of on, `passed_on`), then of every unit of that template no leaf of an output row resolved to
a gate of the row (`hands_on`); else one width a port and the writer's dividing the reader's, then every leaf of the
writer's output inside one leaf of each row its wire fills (`lands_whole`). With `form_only` (`check_detail`,
`count`) each of the two is its rule restated and no expansion: the expansions visit every leaf of every row of every
unit, more than a served class's circuit lets the reference walk, and the verdict is then the Lean's rules' alone,
with nothing holding what they stand for on that case.
"""
from __future__ import annotations

import bisect
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

from verity.circuits import codec
from verity.circuits.defs import PrimitiveDefinition
from verity.circuits.program import input_id, is_input
from verity.circuits.refs import Affine, Concat, Explicit
from verity.circuits.types import Array


def _load(name: str, where: Path | None = None):
    spec = importlib.util.spec_from_file_location(name, (where or Path(__file__).resolve().parent) / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    if where is not None:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference = _load("nci_reference")
Malformed = reference.Malformed
_nat = reference._nat

FIELDS = ("program", "record", "gamma", "alpha", "pins")
#: the count's limits: the gate table, and `(c, k, n)`
LIMITS = ("table", "c", "k", "n")
WIRE = {"reader", "port", "writer", "out", "at", "radix", "base", "step"}
READ = {"reader", "port", "param", "at", "radix", "base", "step"}
REGISTERED = {"params", "tables", "reads", "roots", "openings"}
TABLE = {"name", "param", "offset", "words", "rows"}
TABLE_READ = {"reader", "port", "table", "of", "at", "units"}
OPENING = {"reader", "table", "key", "ports", "verdict", "at", "units"}
INGRESS = {"reader", "port", "at", "units", "step"}
#: `Record.Kind`: what an input row of a unit is classed as, each row exactly one. `inside`: a wire fills it
#: from a writer of its own block. `theta`: a registered read of θ, a table read of a table over θ, or a port of an
#: opening held against such a table (`held`). The rest are fresh (`FRESH`): `cross_wires`, a wire that is not inside;
#: `free`, nothing fills it; `ingress`; `unregistered`, a registered read or a table read of a parameter that is not θ;
#: `openings`, a port of an opening that is not held.
KINDS = ("inside", "theta", "cross_wires", "free", "ingress", "unregistered", "openings")
FRESH = KINDS[2:]
#: the summary's fields, in the order `Record.Summary` holds them
SUMMARY = ("blocks", "gates", "inputs", "fresh", *FRESH, "wired_rows", "wired_bits")


def is_digest(x) -> bool:
    """Whether `x` is a SHA-512 in lower-case hex."""
    return isinstance(x, str) and len(x) == 128 and all(c in "0123456789abcdef" for c in x)


def parse(inst) -> dict:
    """The instance's fields: `instance` where one is missing or of the wrong JSON kind, `params` where `Γ`, `α` or a
    pin is not as stated."""
    if not (isinstance(inst, dict) and all(k in inst for k in FIELDS) and isinstance(inst["program"], str)
            and isinstance(inst["record"], str) and isinstance(inst["pins"], list)):
        raise Malformed("instance")
    gamma, alpha = inst["gamma"], inst["alpha"]
    if not (_nat(gamma) and isinstance(alpha, list) and len(alpha) == 2 and all(map(_nat, alpha)) and alpha[1] != 0
            and all(map(is_digest, inst["pins"]))):
        raise Malformed("params")
    return {"program": inst["program"], "record": inst["record"], "gamma": gamma, "num": alpha[0], "den": alpha[1],
            "pins": inst["pins"], **({"order": inst["order"]} if "order" in inst else {})}


def closure_digest(desc: dict, key: str) -> str:
    """The closure digest of Definition `key` of the Program whose descriptor is `desc`, which a pin names
    (`DefinitionDigest.closureDigest`), by its one Python mirror, the network suite's
    `period_case.slot_entry_digest`, read from the checkout `verity.circuits` comes from."""
    if "period_case" not in sys.modules:
        _load("period_case", Path(codec.__file__).resolve().parents[1] / "protocols" / "certifier" / "verifier" / "network" / "tests")
    return sys.modules["period_case"].slot_entry_digest(desc, key)


def fields_of(ret) -> tuple:
    """A return type's fields, each one output row: a Tuple's or a Record's fields, else the type itself. A Tuple or a
    Record of no field is itself its one field, as `verity.commitments.windows.interface` takes it: a row with no bit."""
    return tuple(f[1] if isinstance(f, tuple) else f for f in (getattr(ret, "fields", None) or (ret,)))


def template_gates(fn) -> list[str]:
    """One activation of a template on its own, gate by gate: each gate's primitive."""
    from verity.circuits.layout import Scope
    if isinstance(fn, PrimitiveDefinition):
        return [codec.descriptor_id(fn)]
    return [codec.descriptor_id(prim) for prim, _, _ in reference.activation_gates(Scope(fn, 0, None, 0, 0))]


def _parts(refs) -> list:
    """A reference sequence's parts, a concatenation spliced in."""
    return [q for p in refs.parts for q in _parts(p)] if isinstance(refs, Concat) else [refs]


def passed_on(fn, memo: dict | None = None) -> frozenset[int]:
    """The Lean's `passedOn`, of the one Definition `fn`: the parameters a leaf of its return may be, with no gate
    between, by index. Read from its body's reference parts, callees first (`memo`, by Definition), a parameter at a
    time: a part of a parameter is that parameter; a part of a node's output is every argument of the node, at a
    parameter its callee passes on, by the same rule (a call, a batch and a scan alike). A primitive passes nothing
    on."""
    memo = {} if memo is None else memo
    if isinstance(fn, PrimitiveDefinition):
        return frozenset()
    if id(fn) not in memo:
        reach: list[frozenset[int]] = []

        def of(refs) -> frozenset[int]:
            """The parameters the parts of `refs` name, or reach through the nodes they name."""
            named = {(s, i) for p in _parts(refs) if p.count
                     for s, i in ({r[:2] for r in p.items} if isinstance(p, Explicit) else [(p.space, p.idx)])}
            return frozenset().union(*({i} if s == "p" else reach[i] for s, i in named))

        for n in fn.body.nodes:
            reach.append(frozenset().union(*(of(n.args[i].refs) for i in passed_on(n.fn, memo))))
        memo[id(fn)] = of(fn.body.ret)
    return memo[id(fn)]


class Template:
    """A template of the record: its Definition and units; its rows, each parameter's `(name, words, outer length)`; its
    outputs, each return field's `(o<j>, words)`; each port's leaves' widths, leaf by leaf (`widths`); its `closed`
    rows, the parameters whose leaves are of one bit each and none of which its Definition passes on (`passed_on`);
    and one activation's gates (`template_gates`). `templates` where a parameter is not whole u16 words, an output
    has no bit, or two ports share a name."""

    def __init__(self, tid: str, fn, units: int) -> None:
        self.tid, self.fn, self.units = tid, fn, units
        params, ret = fn.signature()
        on = passed_on(fn)
        self.closed = [name for i, (name, ty) in enumerate(params) if set(ty.leaf_widths()) == {1} and i not in on]
        self.rows, self.outs = [], []
        self.widths: dict[str, tuple[int, ...]] = {}
        for name, ty in params:
            bits = sum(ty.leaf_widths())
            if bits == 0 or bits % 16:
                raise Malformed("templates")
            self.rows.append((name, bits // 16, ty.n if isinstance(ty, Array) else 1))
            self.widths[name] = tuple(ty.leaf_widths())
        for j, ty in enumerate(fields_of(ret)):
            bits = sum(ty.leaf_widths())
            if bits == 0:
                raise Malformed("templates")
            self.outs.append((f"o{j}", -(-bits // 16)))
            self.widths.setdefault(f"o{j}", tuple(ty.leaf_widths()))
        if len({p for p, *_ in self.rows + self.outs}) != len(self.rows) + len(self.outs):
            raise Malformed("templates")
        self.prims = template_gates(fn)

    def port(self, p: str, outs: bool = False) -> int | None:
        """Port `p`'s words: an input row's, or with `outs` an output's too."""
        return {q: w for q, w, *_ in self.rows + (self.outs if outs else [])}.get(p)


def templates_of(program, rec: dict, digest: str) -> list[Template]:
    """The record's templates, held to the Program: the record names the Program, and its templates are the root's
    nodes that are no input, each a call or a batch of a template of its own, in order. A root node of a composite
    keyed as an input id is refused, listed or not (`RecordText.keyedAsInput`)."""
    ts = rec.get("templates")
    if rec.get("program") != digest or not (isinstance(ts, list) and all(
            isinstance(t, list) and len(t) == 2 and isinstance(t[0], str) and _nat(t[1]) and t[1] >= 1 for t in ts)):
        raise Malformed("templates")
    nodes = [n for n in program.root.body.nodes if not is_input(n.fn)]
    if any(input_id(codec.descriptor_id(n.fn)) is not None for n in nodes):
        raise Malformed("templates")
    if any(n.form not in ("call", "batch") for n in nodes) or len({t for t, _ in ts}) != len(ts) \
            or [[codec.descriptor_id(n.fn), n.n if n.form == "batch" else 1] for n in nodes] != ts:
        raise Malformed("templates")
    return [Template(t, n.fn, k) for (t, k), n in zip(ts, nodes, strict=True)]


def refused(code: str, message: str) -> Malformed:
    """A refusal that says its fix: `code`, with the `message` the Lean's `messageOf` gives it."""
    e = Malformed(code)
    e.message = message
    return e


#: `Record.OWN_ROWS`
OWN_ROWS = ("the root hands a unit a row that is not input gates of its own. Every argument of a root node that is no "
            "input node is affine runs of leaves of the root's input nodes, a batch maps every argument over its "
            "members, and no leaf is handed to two rows. Fix the Program, not the record: build its root as templates "
            "over input nodes of their own, each input leaf handed once.")


def own_rows(program, form_only: bool = False) -> None:
    """The Lean's `ownRows` (`own-rows`). The rule, restated: every argument of a root node that is no input node is
    affine runs of leaves of the root's input nodes (each a primitive or a batch), a batch maps every argument over its
    members, and no two runs, of one argument or of two, share a leaf. Then, unless `form_only`, what the rule stands
    for, leaf by leaf: every leaf of every parameter of every unit, resolved through the root
    (`layout.resolve_gate`), is an input gate, and no gate is two of them."""
    from verity.circuits.layout import Scope, resolve_gate
    from verity.circuits.parts import n_members
    nodes = program.root.body.nodes

    def run(p) -> bool:
        return (isinstance(p, Affine) and p.space == "n" and 0 <= p.idx < len(nodes) and is_input(nodes[p.idx].fn)
                and nodes[p.idx].form in ("primitive", "batch"))
    units = [(k, node) for k, node in enumerate(nodes) if not is_input(node.fn)]
    parts = [p for _, node in units for a in node.args for p in _parts(a.refs)]
    runs = sorted((p.idx, p.base, p.count) for p in parts if run(p) and p.count > 0)
    if any(node.form == "batch" and None in node.axes for _, node in units) or not all(map(run, parts)) \
            or any(a[0] == b[0] and a[1] + a[2] > b[1] for a, b in zip(runs, runs[1:])):
        raise refused("own-rows", OWN_ROWS)
    if form_only:
        return
    root = Scope(program.root, 0, None, 0, 0)
    seen: set[int] = set()
    for k, node in units:
        for m in range(n_members(node)):
            unit = root.child(k, m)
            for i, (_, ty) in enumerate(node.fn.signature()[0]):
                for leaf in range(ty.leaves):
                    gate, prim = resolve_gate(unit, ("p", i, leaf))
                    if not is_input(prim) or gate in seen:
                        raise refused("own-rows", OWN_ROWS)
                    seen.add(gate)


def lands_whole(row, out) -> bool:
    """Whether every leaf of an output row, its leaves' widths `out`, lies inside one leaf of a reader row, its
    leaves' widths `row`: a row is its leaves' bits, in leaf order."""
    edges = list(itertools.accumulate(row, initial=0))
    lo = 0
    for w in out:
        j = bisect.bisect_right(edges, lo) - 1
        if j >= len(row) or lo + w > edges[j + 1]:
            return False
        lo += w
    return True


def hands_on(program, tid: str, port: str) -> bool:
    """Whether some unit of template `tid` hands a leaf of its row `port` on, leaf by leaf: a leaf of its return,
    resolved through the root (`layout.resolve_gate`), is the gate a leaf of that row is."""
    from verity.circuits.layout import Scope, resolve_gate
    from verity.circuits.parts import n_members
    root = Scope(program.root, 0, None, 0, 0)
    for k, node in enumerate(program.root.body.nodes):
        if is_input(node.fn) or codec.descriptor_id(node.fn) != tid or isinstance(node.fn, PrimitiveDefinition):
            continue
        params, ret = node.fn.signature()
        i = [name for name, _ in params].index(port)
        for m in range(n_members(node)):
            unit = root.child(k, m)
            row = {resolve_gate(unit, ("p", i, leaf))[0] for leaf in range(params[i][1].leaves)}
            if any(resolve_gate(unit, node.fn.body.ret[j])[0] in row for j in range(ret.leaves)):
                return True
    return False


def leaf_widths(program, templates: list[Template], wires: list[dict], cuts: dict, form_only: bool = False) -> None:
    """The Lean's `unwhole` (`leaf-widths`). The rule, restated. Of every wire, the writer's output's bits are its
    own words', which are the row's (a wire's two rows have the same words). And a wire is taken where the reader's
    row is closed (`Template.closed`), or where the reader's row and the writer's output each have leaves of one
    width and the writer's divides the reader's. The first wire that fails either is refused, by name, with its two
    rows' bits or its two widths, and the fix. Then, unless `form_only`, what the rule stands for, leaf by leaf, of
    each row of the reader's port the wire fills, a cut port's row `j` of `k` its leaves `[j·n/k, (j+1)·n/k)`. The
    row's leaves hold exactly the bits the writer's output's leaves do: no bit of the row is past its writer's. Of a
    closed row: no unit of the reader's template hands a leaf of it on (`hands_on`, once a row). Of the other: every
    leaf of the writer's output lies inside one leaf of the row (`lands_whole`)."""
    by = {t.tid: t for t in templates}
    kept: set[tuple[str, str]] = set()

    def name(ws: tuple[int, ...]) -> str:
        return f"{ws[0]} bits" if len(set(ws)) == 1 else "more than one width"

    for w in wires:
        row, out = by[w["reader"]].widths[w["port"]], by[w["writer"]].widths[w["out"]]
        wire = f"wire {w['writer']}.{w['out']} -> {w['reader']}.{w['port']}: "
        k, n = int(cuts.get(w["reader"], {}).get(w["port"], 1)), len(row)
        filled = [] if form_only else [row[j * n // k:(j + 1) * n // k] for j in range(w["c0"], w["c0"] + w["C"])]
        bits, words = sum(out), by[w["writer"]].port(w["out"], outs=True)
        if bits != 16 * words:
            assert all(sum(leaves) == 16 * words > bits for leaves in filled)
            raise refused("leaf-widths", (
                f"{wire}the writer's output is {bits} bits and the row it fills {16 * words}. Every bit of a row a "
                "wire fills is a bit of its writer's output: a row is whole 16-bit words, and a bit of it past its "
                "writer's output is no gate's. Fix the Definitions, not the record: give the writer's output whole "
                "16-bit words, each bit a gate of its own."))
        assert all(sum(leaves) == bits for leaves in filled)
        if w["port"] in by[w["reader"]].closed:
            if not form_only and (w["reader"], w["port"]) not in kept:
                assert not hands_on(program, w["reader"], w["port"])
                kept.add((w["reader"], w["port"]))
            continue
        if len(set(row)) != 1 or len(set(out)) != 1 or row[0] % out[0]:
            raise refused("leaf-widths", (
                f"{wire}the writer's leaves are of {name(out)} "
                f"and the reader's of {name(row)}. Every leaf of a writer's output lies inside one leaf of the row "
                "that reads it, each side's leaves of one width and the writer's dividing the reader's; or the "
                "reader's row is one-bit leaves and its Definition passes none of them on: its return refers to no "
                "part of that row, itself or through a call, a batch or a scan. Fix the Definitions, not the "
                "record: give the reader's row leaves of the writer's width or a multiple of it, or the writer's "
                "output leaves of a width that divides the reader's, or make the reader's row one-bit leaves and "
                "have its Definition return gates of its own in their place."))
        assert all(lands_whole(leaves, out) for leaves in filled)


def spot_instance(templates: list[Template], rec: dict, order=None) -> dict:
    """The record's wires as SPOT's `wires` piece reads them (`Spot.Wires.Wiring.ofJson`): the templates' units, rows
    and outputs, the cuts, the wires, and the order the record states, else `order` where one is given."""
    out = {"templates": {t.tid: {"units": t.units, "rows": {p: [w, n] for p, w, n in t.rows}, "outs": dict(t.outs)}
                         for t in templates},
           "cuts": rec.get("cuts", {}), "wires": rec.get("wires")}
    if "order" in rec:
        out["order"] = rec["order"]
    elif order is not None:
        out["order"] = order[0]
    return out


def wires_of(templates: list[Template], rec: dict, order) -> tuple[list[dict], dict]:
    """The record's wires and cuts, refused (`wires`) as `verity.commitments.windows.interface.check_wires` refuses them, under
    the record's order, else the instance's (`order`: a one-entry tuple where the instance gives one)."""
    import verity.commitments.windows.interface as IF
    spot = spot_instance(templates, rec, order)
    lays = {t.tid: (IF.RowLayout(tuple((p, w) for p, w, _ in t.rows), tuple(t.outs), (), tuple(n for *_, n in t.rows)),
                    t.units) for t in templates}
    try:
        if not (isinstance(spot["wires"], list) and isinstance(spot["cuts"], dict)):
            raise ValueError("the record states its wires as a list and its cuts as an object")
        wires = tuple(IF.Wire.of(x) for x in spot["wires"])
        cuts = IF._cuts(spot["cuts"])
        IF.check_wires(lays, wires, cuts, IF.order_of(spot))
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise Malformed("wires") from e
    return [{**w.doc(), "c0": w.cut[0], "C": w.cut[1]} for w in wires], cuts


def pattern(d, keys: set[str]) -> dict:
    """A registered read's record, in a wire's form (`Spot.Wires.Wire.ofJson`'s rules)."""
    if not (isinstance(d, dict) and set(d) in (keys, keys | {"cut"})):
        raise Malformed("inputs")
    radix, step, cut = d["radix"], d["step"], d.get("cut", [0, 1])
    if not (all(isinstance(d[k], str) for k in keys - {"at", "radix", "base", "step"}) and _nat(d["at"])
            and type(d["base"]) is int and isinstance(radix, list) and isinstance(step, list) and radix
            and len(radix) == len(step) and all(_nat(r) and r >= 1 for r in radix)
            and all(type(s) is int for s in step) and isinstance(cut, list) and len(cut) == 2 and all(map(_nat, cut))
            and cut[1] >= 1):
        raise Malformed("inputs")
    return {**d, "c0": cut[0], "C": cut[1]}


def source(w: dict, i: int) -> int:
    """Reader index `i`'s source: `base + Σ d_k · step_k` over its mixed-radix digits, the first the slowest."""
    out = w["base"]
    for r, s in zip(reversed(w["radix"]), reversed(w["step"])):
        out += (i % r) * s
        i //= r
    return out


def entries(w: dict):
    """Each reader row a pattern covers: its reader unit, row and source."""
    for i in range(math.prod(w["radix"])):
        yield w["at"] + i // w["C"], w["c0"] + i % w["C"], source(w, i)


def _objects(xs, keys: set[str]) -> list[dict]:
    if not (isinstance(xs, list) and all(isinstance(x, dict) and set(x) == keys for x in xs)):
        raise Malformed("inputs")
    return xs


class Inputs:
    """What the record declares to fill its units' input rows beside its wires, each declaration refused (`inputs`)
    unless it is well formed for the templates and cuts, and unless no row is filled twice (`fill`, row by row):
    `params`, the registered parameters' leaves; `tables`, by name; `reads`, the registered reads; `roots`, the table
    reads; `openings`; and `fill`, by `(template, port)`, each filled `(unit, row)` with what fills it:
    `("wire", index)`, `("read", parameter)`, `("root", table, what it reads of it)`, `("opening", index)` or
    `("ingress",)`."""

    def __init__(self, templates: list[Template], cuts: dict, wires: list[dict], rec: dict) -> None:
        by = {t.tid: t for t in templates}
        self.fill: dict[tuple[str, str], dict[tuple[int, int], tuple]] = {}

        def cut(tid: str, p: str) -> int:
            return int(cuts.get(tid, {}).get(p, 1))

        def port(tid, p, outs: bool = False) -> tuple[int, int]:
            words = by[tid].port(p, outs) if tid in by else None
            if words is None:
                raise Malformed("inputs")
            return words, cut(tid, p)

        def units(tid, at, n) -> None:
            if not (_nat(at) and _nat(n) and n >= 1 and tid in by and at + n <= by[tid].units):
                raise Malformed("inputs")

        def put(tid: str, p: str, cells, what: tuple) -> None:
            held = self.fill.setdefault((tid, p), {})
            for cell in cells:
                if cell in held:
                    raise Malformed("inputs")
                held[cell] = what

        for i, w in enumerate(wires):
            put(w["reader"], w["port"], ((u, r) for u, r, _ in entries(w)), ("wire", i))
        reg = rec.get("registered", {"params": {}, "tables": [], "reads": [], "roots": [], "openings": []})
        if not (isinstance(reg, dict) and REGISTERED <= set(reg) <= REGISTERED | {"values"}
                and isinstance(reg["params"], dict) and all(_nat(v) and v >= 1 for v in reg["params"].values())
                and isinstance(reg["reads"], list)):
            raise Malformed("inputs")
        self.params = reg["params"]
        self.tables: dict[str, dict] = {}
        for t in _objects(reg["tables"], TABLE):
            if not (isinstance(t["name"], str) and isinstance(t["param"], str) and t["name"] not in self.tables
                    and t["param"] in self.params and all(_nat(t[k]) for k in ("offset", "words", "rows"))
                    and t["words"] >= 1 and t["rows"] >= 1
                    and t["offset"] + t["words"] * t["rows"] <= self.params[t["param"]]):
                raise Malformed("inputs")
            self.tables[t["name"]] = t
        self.reads = [pattern(r, READ) for r in reg["reads"]]
        for r in self.reads:
            words, k = port(r["reader"], r["port"], outs=True)
            n = math.prod(r["radix"])
            if not (r["c0"] + r["C"] <= k and (r["C"] == 1 or r["radix"][-1] == r["C"]) and n % r["C"] == 0):
                raise Malformed("inputs")
            units(r["reader"], r["at"], n // r["C"])
            got = [s for _, _, s in entries(r)]
            if r["param"] not in self.params or min(got) < 0 or max(got) + words // k > self.params[r["param"]]:
                raise Malformed("inputs")
            put(r["reader"], r["port"], ((u, j) for u, j, _ in entries(r)), ("read", r["param"]))
        self.roots = _objects(reg["roots"], TABLE_READ)
        for r in self.roots:
            if not all(isinstance(r[k], str) for k in ("reader", "port", "table", "of")):
                raise Malformed("inputs")
            words, k = port(r["reader"], r["port"])
            units(r["reader"], r["at"], r["units"])
            if r["table"] not in self.tables or r["of"] not in ("root", "domain") or words != 32:
                raise Malformed("inputs")
            put(r["reader"], r["port"], ((u, j) for u in range(r["at"], r["at"] + r["units"]) for j in range(k)),
                ("root", r["table"], r["of"]))
        self.openings = _objects(reg["openings"], OPENING)
        for i, o in enumerate(self.openings):
            if not (all(isinstance(o[k], str) for k in ("reader", "table", "key", "verdict"))
                    and isinstance(o["ports"], list) and o["ports"] and all(isinstance(p, str) for p in o["ports"])):
                raise Malformed("inputs")
            units(o["reader"], o["at"], o["units"])
            port(o["reader"], o["key"])
            if o["table"] not in self.tables or o["verdict"] not in dict(by[o["reader"]].outs):
                raise Malformed("inputs")
            for p in o["ports"]:
                k = port(o["reader"], p)[1]
                put(o["reader"], p, ((u, j) for u in range(o["at"], o["at"] + o["units"]) for j in range(k)),
                    ("opening", i))
        for g in _objects(rec.get("ingress", []), INGRESS):
            if not (isinstance(g["reader"], str) and isinstance(g["port"], str) and _nat(g["step"])):
                raise Malformed("inputs")
            k = port(g["reader"], g["port"])[1]
            units(g["reader"], g["at"], g["units"])
            put(g["reader"], g["port"], ((u, j) for u in range(g["at"], g["at"] + g["units"]) for j in range(k)),
                ("ingress",))


def slots_of(templates: list[Template], rec: dict) -> tuple[int, dict[str, list[int]]]:
    """The class's slots and every unit's slot, from the record's inference entries: unit `at + i` of an entry has
    `i`'s digits in mixed radix `radix`, the first the slowest, and its slot is the digit `axes` names `slot`. `blocks`
    unless every entry names a template and has one slot digit, of the class's radix, and the entries tile each
    template's units once."""
    inf = rec.get("blocks", {}).get("inference") if isinstance(rec.get("blocks"), dict) else None
    slots, es = (inf.get("slots"), inf.get("entries")) if isinstance(inf, dict) else (None, None)
    if not (_nat(slots) and slots >= 1 and isinstance(es, list)):
        raise Malformed("blocks")
    out: dict[str, list] = {t.tid: [None] * t.units for t in templates}
    for e in es:
        radix, axes = (e.get("radix"), e.get("axes")) if isinstance(e, dict) else (None, None)
        if not (isinstance(radix, list) and isinstance(axes, list) and isinstance(e.get("template"), str)
                and e["template"] in out and _nat(e.get("at")) and radix and len(radix) == len(axes)
                and all(_nat(r) and r >= 1 for r in radix)):
            raise Malformed("blocks")
        at = [k for k, a in enumerate(axes) if a == "slot"]
        if len(at) != 1 or radix[at[0]] != slots:
            raise Malformed("blocks")
        q, held = math.prod(radix[at[0] + 1:]), out[e["template"]]
        for i in range(math.prod(radix)):
            if e["at"] + i >= len(held) or held[e["at"] + i] is not None:
                raise Malformed("blocks")
            held[e["at"] + i] = i // q % slots
    if any(s is None for held in out.values() for s in held):
        raise Malformed("blocks")
    return slots, out


def theta(T: dict, params: dict, p: str) -> bool:
    """Whether registered parameter `p` is θ: the registration log's first record registers a value `p` (its last
    entry of that name) under the root the run's own record commits `p` under (its window's `registered_roots`), with
    the record's leaves for `p` in 16-bit words."""
    e = reference.last_with(T["log"][0], "registered", "value", p)
    window = T["log"][T["run"]].get("window")
    roots = window.get("registered_roots") if isinstance(window, dict) else None
    root = roots.get(p) if isinstance(roots, dict) else None
    if e is None or not isinstance(root, str) or e.get("root") != root:
        return False
    m = reference.dims(e)
    return m is not None and m[0] * m[1] == params[p] and m[2] == 16


def layout(t: Template) -> dict | None:
    """The Lean's `layoutOf`: template `t`'s roles by position, or None where it has no layout. It has one with
    at least six input rows, the second and the third of 32 words each, the first named unlike the fourth, fifth and
    sixth, and at least two outputs: input row 0 is the `key`, 1 the table's `root`, 2 its `domain` id, 3 to 5 the
    opened rows (`ports`), and output 1 the `verdict`."""
    names = [p for p, *_ in t.rows]
    if len(t.rows) < 6 or len(t.outs) < 2 or t.rows[1][1] != 32 or t.rows[2][1] != 32 or names[0] in names[3:6]:
        return None
    return {"key": names[0], "root": names[1], "domain": names[2], "ports": names[3:6], "verdict": t.outs[1][0]}


def roles(templates: list[Template], is_pinned, x: Inputs) -> None:
    """The Lean's `rolesOf`, over the pinned templates that an opening or a table read names as its reader:
    `reader-layout` where one has no layout; else `reader-roles` where an opening of one has another key, other ports
    (the layout's, in order) or another verdict than its layout's, or a table read by one is in another row than its
    layout's root row (of the root) or domain id row (of the domain id)."""
    by = {t.tid: t for t in templates}
    named = {d["reader"] for d in (*x.openings, *x.roots) if is_pinned(d["reader"])}
    layouts = {tid: layout(by[tid]) for tid in named}
    if None in layouts.values():
        raise Malformed("reader-layout")
    if any(o["reader"] in layouts and (o["key"], o["ports"], o["verdict"]) != tuple(
            layouts[o["reader"]][k] for k in ("key", "ports", "verdict")) for o in x.openings):
        raise Malformed("reader-roles")
    if any(r["reader"] in layouts and r["port"] != layouts[r["reader"]][r["of"]] for r in x.roots):
        raise Malformed("reader-roles")


def held(x: Inputs, is_theta, is_pinned, o: dict) -> bool:
    """Whether opening `o` is checked against its registered table inside its units: its reader's Definition is one the
    policy pins, the table is over a θ parameter, and one table read of the table's root and one of its domain id, by
    the opening's reader, each cover its units. On declarations `roles` took, its ports are the opened rows of its
    reader's layout, and the two table reads fill the layout's root and domain id rows."""
    def covered(of: str) -> bool:
        return any(r["reader"] == o["reader"] and r["table"] == o["table"] and r["of"] == of and r["at"] <= o["at"]
                   and o["at"] + o["units"] <= r["at"] + r["units"] for r in x.roots)
    return is_pinned(o["reader"]) and is_theta(x.tables[o["table"]]["param"]) and covered("root") and covered("domain")


def front(inst, tr, form_only: bool = False) -> tuple:
    """`Record.front`, the stages as far as `own-rows`: the instance's fields, the Program, its descriptor,
    the record, the trusted input and the templates. Refused (`Malformed`) with the first stage's code."""
    i = parse(inst)
    program = reference.desc_program(i["program"])
    try:
        rec = json.loads(i["record"])
    except ValueError as e:
        raise Malformed("record") from e
    if not isinstance(rec, dict):
        raise Malformed("record")
    T = reference.trusted_of(tr)
    run = T["log"][T["run"]]
    desc = json.loads(i["program"])
    digest = codec.program_digest(desc)
    if run.get("program") != digest:
        raise Malformed("run-program")
    window = run.get("window")
    if not (isinstance(window, dict) and window.get("circuit") == hashlib.sha512(i["record"].encode()).hexdigest()):
        raise Malformed("run-circuit")
    templates = templates_of(program, rec, digest)
    own_rows(program, form_only)
    return i, program, desc, rec, T, templates


def closed(inst, tr) -> list | dict:
    """Each template's closed rows as the form reads them (`Record.Leaves.closed`), `[template, rows]` each
    in the record's order, a template's rows in its parameters'; or the refusal of a stage as far as `own-rows`."""
    try:
        return [[t.tid, t.closed] for t in front(inst, tr)[-1]]
    except Malformed as m:
        return {"error": m.code}


def expand(inst, tr, form_only: bool = False) -> dict:
    """Everything the summary is read from, unit by unit: the instance's fields (`i`), the templates, the wires and
    cuts, the declarations (`x`), the slots and each unit's (`slots`, `slot`), each wire's `within`, `kind` (the class
    of what fills a row), each template's pin where the policy pins its Definition (`pins`), and per slot the sums
    `gates`, `inputs` (every row, whole), each of `KINDS`' bits, and `fresh`, the sum of `FRESH`'s. Refused
    (`Malformed`) with the first stage's code, in `Record`'s order."""
    i, program, desc, rec, T, templates = front(inst, tr, form_only)
    wires, cuts = wires_of(templates, rec, (i["order"],) if "order" in i else None)
    leaf_widths(program, templates, wires, cuts, form_only)
    x = Inputs(templates, cuts, wires, rec)
    slots, slot = slots_of(templates, rec)
    by = {t.tid: t for t in templates}
    inside = [all(slot[w["reader"]][u] == slot[w["writer"]][v] for u, _, v in entries(w)) for w in wires]
    is_theta = {p: theta(T, x.params, p) for p in x.params}.__getitem__
    digests = {t.tid: closure_digest(desc, t.tid) for t in templates}
    pins = {tid: d if d in i["pins"] else None for tid, d in digests.items()}

    def is_pinned(tid: str) -> bool:
        return pins[tid] is not None
    roles(templates, is_pinned, x)
    holds = [held(x, is_theta, is_pinned, o) for o in x.openings]

    def kind(what: tuple | None) -> str:
        """The class of a row filled by `what`, None where nothing fills it: one of `KINDS`."""
        if what is None:
            return "free"
        if what[0] == "wire":
            return "inside" if inside[what[1]] else "cross_wires"
        if what[0] == "read":
            return "theta" if is_theta(what[1]) else "unregistered"
        if what[0] == "root":
            return "theta" if is_theta(x.tables[what[1]]["param"]) else "unregistered"
        if what[0] == "opening":
            return "theta" if holds[what[1]] else "openings"
        return "ingress"

    sums = {k: [0] * slots for k in ("gates", "inputs", *KINDS)}
    for t in templates:
        for u in range(t.units):
            s = slot[t.tid][u]
            sums["gates"][s] += len(t.prims)
            for p, words, _ in t.rows:
                k = int(cuts.get(t.tid, {}).get(p, 1))
                sums["inputs"][s] += 16 * words
                for row in range(k):
                    sums[kind(x.fill.get((t.tid, p), {}).get((u, row)))][s] += 16 * words // k
    assert all(sums["inputs"][s] == sum(sums[k][s] for k in KINDS) for s in range(slots))
    wired = [(math.prod(w["radix"]), 16 * by[w["reader"]].port(w["port"]) // int(cuts.get(w["reader"], {}).get(w["port"], 1)))
             for w in wires]
    return {"i": i, "templates": templates, "wires": wires, "cuts": cuts, "x": x, "slots": slots, "slot": slot,
            "within": inside, "kind": kind, "pins": pins, **sums,
            "fresh": [sum(sums[k][s] for k in FRESH) for s in range(slots)],
            "wired_rows": sum(n for n, _ in wired), "wired_bits": sum(n * b for n, b in wired)}


def evaluate(inst, tr, form_only: bool = False) -> dict:
    """`Record.evaluate`: the summary, each per-block count the greatest over the slots, and the rate rule on
    it: `size` where a block has more than `Γ` gates, else `nci` where its fresh bits exceed `α` times its gates."""
    e = expand(inst, tr, form_only)
    i = e["i"]
    out = {"blocks": e["slots"], **{k: max(e[k]) for k in ("gates", "inputs", "fresh", *FRESH)},
           "wired_rows": e["wired_rows"], "wired_bits": e["wired_bits"]}
    failure = ("size" if out["gates"] > i["gamma"] else
               "nci" if out["fresh"] * i["den"] > i["num"] * out["gates"] else None)
    return {**out, "complies": failure is None, "failure": failure}


def check_detail(inst, tr, form_only: bool = False) -> dict:
    """`Record.checkDetail`: the whole verdict, or the refusal (with its message, where it has one), for the
    private record only. With `form_only`, `own-rows` and `leaf-widths` are their rules alone, as the Lean has them,
    without the leaf-by-leaf expansions (the module docstring): nothing then holds a taken record's rows to being input
    gates of their own unit, or a taken wire's leaves to lying whole in the row it fills."""
    try:
        return evaluate(inst, tr, form_only)
    except Malformed as e:
        return {"error": e.code, **({"message": e.message} if hasattr(e, "message") else {})}


def check(inst, tr) -> dict:
    """`Record.check`, what the form publishes: the judgment, or the refusal."""
    res = check_detail(inst, tr)
    return res if "error" in res else {"complies": res["complies"]}


def kinds(inst, tr, rows) -> list | dict:
    """`Record.kinds`: the class of each of `rows`, `[template, port, unit, row]` each (`free` where the
    record's declarations fill no such row), or the refusal."""
    try:
        e = expand(inst, tr)
    except Malformed as m:
        return {"error": m.code}
    return [e["kind"](e["x"].fill.get((t, p), {}).get((u, r))) for t, p, u, r in rows]


def exported(inst, tr, rows) -> list | dict:
    """`Record.exported`: of each of `rows`, `[template, port, unit, row]` each, `[slot, reason]`: the unit's
    slot (`Case.slot`), and why the row is θ (`Case.heldAt`), None where it is not: `["read", parameter]`,
    `["table", name, parameter, whether it reads the root]` or `["opened", name, parameter, the reader's pin]`. Or
    the refusal."""
    try:
        e = expand(inst, tr)
    except Malformed as m:
        return {"error": m.code}
    x = e["x"]

    def reason(what: tuple | None) -> list | None:
        if e["kind"](what) != "theta":
            return None
        if what[0] == "read":
            return ["read", what[1]]
        if what[0] == "root":
            return ["table", what[1], x.tables[what[1]]["param"], what[2] == "root"]
        o = x.openings[what[1]]
        return ["opened", o["table"], x.tables[o["table"]]["param"], e["pins"][o["reader"]]]
    return [[e["slot"][t][u], reason(x.fill.get((t, p), {}).get((u, r)))] for t, p, u, r in rows]


def count(inst, tr, limits, form_only: bool = False) -> dict:
    """`RecordCount.count`: `(c, k, n)` over the blocks' Boolean gates (each gate its primitive's entry of the
    gate table), fresh bits and number. `instance` or `params` where the limits are not as stated, then the form's own
    refusal, then `gate-table` where the table lacks a primitive a block computes. `form_only` as `check_detail`'s."""
    if not (isinstance(limits, dict) and all(k in limits for k in LIMITS) and isinstance(limits["table"], dict)):
        return {"error": "instance"}
    table = limits["table"]
    if not (all(_nat(limits[k]) for k in ("c", "k", "n")) and all(map(_nat, table.values()))):
        return {"error": "params"}
    try:
        e = expand(inst, tr, form_only)
    except Malformed as m:
        return {"error": m.code}
    if any(p not in table for t in e["templates"] for p in t.prims):
        return {"error": "gate-table"}
    bools = [0] * e["slots"]
    for t in e["templates"]:
        for u in range(t.units):
            bools[e["slot"][t.tid][u]] += sum(table[p] for p in t.prims)
    failure = ("gates" if max(bools) > limits["c"] else "fresh" if max(e["fresh"]) > limits["k"] else
               "count" if e["slots"] > limits["n"] else None)
    return {"bool_gates": max(bools), "complies": failure is None, "failure": failure}
