"""The inference-only policy's check of an execution phase's circuit record, unit by unit: the reference
`NciCheck.Record` is held to.

    check_detail(instance, trusted) -> {"blocks", "gates", "inputs", "fresh", "cross_wires", "free", "ingress",
                                        "unregistered", "openings", "wired_rows", "wired_bits", "complies", "failure"}
                                       | {"error": code}
    check(instance, trusted) -> {"complies"} | {"error": code}
    kinds(instance, trusted, rows) -> [kind, …] | {"error": code}
    count(instance, trusted, limits) -> {"bool_gates", "complies", "failure"} | {"error": code}

`NciCheck.Record`'s docstring gives the instance, the trusted input, the stages and their refusals, and what each field
of the summary counts; `NciCheck.RecordCount`'s gives the count's limits. Where the Lean reads each template's gates
from the Program's own count and its rows from its Definition's parameters, takes one closed form per wire and per
declaration, and multiplies through the slots, this file expands: each template's gates gate by gate
(`nci_reference.activation_gates`), every unit's slot from the inference entries' digits, every input row of every
unit with what fills it, and every reader row of every wire with its writer. A block is one slot's units; a wire is
inside a block when every reader row it covers is read from a writer of the reader's slot, and every row of any other
wire counts as read across blocks. Every input row of every unit is classed once (`KINDS`, by `expand`'s `kind`): one
nothing shows inside its block or θ is fresh; and it counts whole, 16 bits a word, whether or not a gate reads it.

Four rules are no expansion. The wires' own check is SPOT's (`Spot.Wires.check`, which the Lean runs on the wires as
`Spot.Wires.Wiring.ofJson` reads them); here it is Python's statement of the same rule,
`verity.ml.windows.interface.check_wires`. θ's rule and the rule for a table opening are the Lean's, restated
(`theta`, `held`): nothing unrolls a registration log. A Definition's closure digest is the period check's
(`closure_digest`).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

from verity.ml.circuits import codec
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.program import is_input
from verity.ml.circuits.types import Array


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
#: `NciCheck.Record.Kind`: what an input row of a unit is classed as, each row exactly one. `inside`: a wire fills it
#: from a writer of its own block. `theta`: a registered read of θ, a table read of a table over θ, or a port of an
#: opening held against such a table (`held`). The rest are fresh (`FRESH`): `cross_wires`, a wire that is not inside;
#: `free`, nothing fills it; `ingress`; `unregistered`, a registered read or a table read of a parameter that is not θ;
#: `openings`, a port of an opening that is not held.
KINDS = ("inside", "theta", "cross_wires", "free", "ingress", "unregistered", "openings")
FRESH = KINDS[2:]
#: the summary's fields, in the order `NciCheck.Record.Summary` holds them
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
    """The closure digest of Definition `key` of the Program whose descriptor is `desc`, which a pin names: the period
    check's (`slotEntryDigest`), by its Python mirror, `period_case.slot_entry_digest` of the network suite."""
    if "period_case" not in sys.modules:
        _load("period_case", Path(__file__).resolve().parents[3] / "tests")
    return sys.modules["period_case"].slot_entry_digest(desc, key)


def fields_of(ret) -> tuple:
    """A return type's fields, each one output row: a Tuple's or a Record's fields, else the type itself."""
    return tuple(f[1] if isinstance(f, tuple) else f for f in (getattr(ret, "fields", None) or (ret,)))


def template_gates(fn) -> list[str]:
    """One activation of a template on its own, gate by gate: each gate's primitive."""
    from verity.ml.circuits.layout import Scope
    if isinstance(fn, PrimitiveDefinition):
        return [codec.descriptor_id(fn)]
    return [codec.descriptor_id(prim) for prim, _, _ in reference.activation_gates(Scope(fn, 0, None, 0, 0))]


class Template:
    """A template of the record: its Definition and units; its rows, each parameter's `(name, words, outer length)`; its
    outputs, each return field's `(o<j>, words)`; and one activation's gates (`template_gates`). `templates` where a
    parameter is not whole u16 words, an output has no bit, or two ports share a name."""

    def __init__(self, tid: str, fn, units: int) -> None:
        self.tid, self.fn, self.units = tid, fn, units
        params, ret = fn.signature()
        self.rows, self.outs = [], []
        for name, ty in params:
            bits = sum(ty.leaf_widths())
            if bits == 0 or bits % 16:
                raise Malformed("templates")
            self.rows.append((name, bits // 16, ty.n if isinstance(ty, Array) else 1))
        for j, ty in enumerate(fields_of(ret)):
            bits = sum(ty.leaf_widths())
            if bits == 0:
                raise Malformed("templates")
            self.outs.append((f"o{j}", -(-bits // 16)))
        if len({p for p, *_ in self.rows + self.outs}) != len(self.rows) + len(self.outs):
            raise Malformed("templates")
        self.prims = template_gates(fn)

    def port(self, p: str, outs: bool = False) -> int | None:
        """Port `p`'s words: an input row's, or with `outs` an output's too."""
        return {q: w for q, w, *_ in self.rows + (self.outs if outs else [])}.get(p)


def templates_of(program, rec: dict, digest: str) -> list[Template]:
    """The record's templates, held to the Program: the record names the Program, and its templates are the root's
    nodes that are no input, each a call or a batch of a template of its own, in order."""
    ts = rec.get("templates")
    if rec.get("program") != digest or not (isinstance(ts, list) and all(
            isinstance(t, list) and len(t) == 2 and isinstance(t[0], str) and _nat(t[1]) and t[1] >= 1 for t in ts)):
        raise Malformed("templates")
    nodes = [n for n in program.root.body.nodes if not is_input(n.fn)]
    if any(n.form not in ("call", "batch") for n in nodes) or len({t for t, _ in ts}) != len(ts) \
            or [[codec.descriptor_id(n.fn), n.n if n.form == "batch" else 1] for n in nodes] != ts:
        raise Malformed("templates")
    return [Template(t, n.fn, k) for (t, k), n in zip(ts, nodes, strict=True)]


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
    """The record's wires and cuts, refused (`wires`) as `verity.ml.windows.interface.check_wires` refuses them, under
    the record's order, else the instance's (`order`: a one-entry tuple where the instance gives one)."""
    import verity.ml.windows.interface as IF
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
    `("wire", index)`, `("read", parameter)`, `("root", table)`, `("opening", index)` or `("ingress",)`."""

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
                ("root", r["table"]))
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


def held(x: Inputs, is_theta, is_pinned, o: dict) -> bool:
    """Whether opening `o` is checked against its registered table inside its units: its reader's Definition is one the
    policy pins, the table is over a θ parameter, and one table read of the table's root and one of its domain id, by
    the opening's reader, each cover its units."""
    def covered(of: str) -> bool:
        return any(r["reader"] == o["reader"] and r["table"] == o["table"] and r["of"] == of and r["at"] <= o["at"]
                   and o["at"] + o["units"] <= r["at"] + r["units"] for r in x.roots)
    return is_pinned(o["reader"]) and is_theta(x.tables[o["table"]]["param"]) and covered("root") and covered("domain")


def expand(inst, tr) -> dict:
    """Everything the summary is read from, unit by unit: the instance's fields (`i`), the templates, the wires and
    cuts, the declarations (`x`), the slots and each unit's (`slots`, `slot`), each wire's `within`, `kind` (the class
    of what fills a row), and per slot the sums `gates`, `inputs` (every row, whole), each of `KINDS`' bits, and
    `fresh`, the sum of `FRESH`'s. Refused (`Malformed`) with the first stage's code, in `NciCheck.Record`'s order."""
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
    wires, cuts = wires_of(templates, rec, (i["order"],) if "order" in i else None)
    x = Inputs(templates, cuts, wires, rec)
    slots, slot = slots_of(templates, rec)
    by = {t.tid: t for t in templates}
    inside = [all(slot[w["reader"]][u] == slot[w["writer"]][v] for u, _, v in entries(w)) for w in wires]
    is_theta = {p: theta(T, x.params, p) for p in x.params}.__getitem__
    is_pinned = {t.tid: closure_digest(desc, t.tid) in i["pins"] for t in templates}.__getitem__
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
            "within": inside, "kind": kind, **sums, "fresh": [sum(sums[k][s] for k in FRESH) for s in range(slots)],
            "wired_rows": sum(n for n, _ in wired), "wired_bits": sum(n * b for n, b in wired)}


def evaluate(inst, tr) -> dict:
    """`NciCheck.Record.evaluate`: the summary, each per-block count the greatest over the slots, and the rate rule on
    it: `size` where a block has more than `Γ` gates, else `nci` where its fresh bits exceed `α` times its gates."""
    e = expand(inst, tr)
    i = e["i"]
    out = {"blocks": e["slots"], **{k: max(e[k]) for k in ("gates", "inputs", "fresh", *FRESH)},
           "wired_rows": e["wired_rows"], "wired_bits": e["wired_bits"]}
    failure = ("size" if out["gates"] > i["gamma"] else
               "nci" if out["fresh"] * i["den"] > i["num"] * out["gates"] else None)
    return {**out, "complies": failure is None, "failure": failure}


def check_detail(inst, tr) -> dict:
    """`NciCheck.Record.checkDetail`: the whole verdict, or the refusal, for the private record only."""
    try:
        return evaluate(inst, tr)
    except Malformed as e:
        return {"error": e.code}


def check(inst, tr) -> dict:
    """`NciCheck.Record.check`, what the form publishes: the judgment, or the refusal."""
    res = check_detail(inst, tr)
    return res if "error" in res else {"complies": res["complies"]}


def kinds(inst, tr, rows) -> list | dict:
    """`NciCheck.Record.kinds`: the class of each of `rows`, `[template, port, unit, row]` each (`free` where the
    record's declarations fill no such row), or the refusal."""
    try:
        e = expand(inst, tr)
    except Malformed as m:
        return {"error": m.code}
    return [e["kind"](e["x"].fill.get((t, p), {}).get((u, r))) for t, p, u, r in rows]


def count(inst, tr, limits) -> dict:
    """`NciCheck.RecordCount.count`: `(c, k, n)` over the blocks' Boolean gates (each gate its primitive's entry of the
    gate table), fresh bits and number. `instance` or `params` where the limits are not as stated, then the form's own
    refusal, then `gate-table` where the table lacks a primitive a block computes."""
    if not (isinstance(limits, dict) and all(k in limits for k in LIMITS) and isinstance(limits["table"], dict)):
        return {"error": "instance"}
    table = limits["table"]
    if not (all(_nat(limits[k]) for k in ("c", "k", "n")) and all(map(_nat, table.values()))):
        return {"error": "params"}
    try:
        e = expand(inst, tr)
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
