"""The record form's difftest data: execution phases' circuit records with their Programs, registration logs and limits,
for `NciCheck/RecordDifftestMain.lean`.

    python scripts/record_difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes two files under `DIR/record_difftest/`, one line per case: `cases.jsonl`, `{"instance", "trusted", "spot",
"count"}`: the instance and the trusted input as `NciCheck.Record.check` reads them (`NciCheck.Record`'s docstring), the
record's wires as SPOT's `wires` piece reads them (`nci_record_reference.spot_instance`, from Python's own layout of
each template; `null` where the reference refuses the case before its wires), and `(c, k, n)` for the count; and
`python.jsonl`, `{"detail", "public", "count"}`: `nci_record_reference`'s `check_detail`, `check` and `count`. The Lean
audit runs `NciCheck/RecordDifftestMain.lean` on the two (`runs` in `lean-audit.json`).

Case `seed`, for the first `COUNT`, is a random circuit (`random_circuit`): one to four templates of a few gates, each
reading some of its parameters' leaves (directly, through a call and through a batch), their units laid out by one or
two inference entries each with the slot digit anywhere among the digits; wires from earlier templates that keep to the
reader's slot, cross slots or do either unit by unit, written unit by unit and merged into runs
(`verity.ml.windows.interface.coalesce`, as a lowering states them); registered reads, a table with its table reads and
opening, and ingress; and a registration log in R6's chain whose first record registers some of the parameters, under
the root the run's own record commits or another. About two in five are broken in one way, each refused at one stage
(`BROKEN`). Then `shaped()`: circuits in the shape of a served class (`lowered`: a slot scan, an embedding read through
a table opening, a norm, matmul rows over registered weights, an attention template per position reading every earlier
position's rows by a cut, a mix, a selection and a token feed reading its own earlier units), at two slots and three
positions and at the served class's four slots and 35, with the weights registered at the start of the deployment and
with none, under another root, with the opening's root read dropped, with a wire into the next slot, and with the order
given beside the record.

Every wire of every case is one `NciCheck.Record.sameSlot`'s search decides within its fuel (`searched`, the search
mirrored and held to the expansion as the cases are made), so the Lean's count of a wire is the reference's.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path

import verity.ml.circuits
import verity.ml.windows.interface as IF
from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.population import mixed_population_program
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load("nci_record_reference")
reference = ref.reference

COUNT = 600
BIG = 10 ** 15
#: `NciCheck.Record.FUEL`: the boxes `sameSlot`'s search visits for one wire before it counts the wire as crossing
FUEL = 4096
V2 = "verity/registration/v2"
#: how a random case is broken, by the stage that refuses it
BROKEN = {
    "instance": ["no-field", "program-type", "record-type", "table-type"],
    "params": ["gamma", "alpha", "table-value"],
    "program": ["program-text"],
    "not-topological": ["forward"],
    "record": ["record-text", "record-list"],
    "trusted": ["trusted-keys", "chain", "run-unknown"],
    "run-program": ["run-program"],
    "run-circuit": ["run-circuit", "no-window"],
    "templates": ["record-program", "units", "swap", "half-word"],
    "wires": ["overlap", "bounds", "width", "wire-port", "wire-template", "wire-cut", "cycle", "order", "no-wires"],
    "inputs": ["read-param", "read-leaves", "read-on-wire", "table-param", "root-port", "verdict", "ingress-units",
               "registered-keys"],
    "blocks": ["no-blocks", "slot-radix", "two-slots", "gap", "twice", "entry-template"],
}

_PRIMS: dict[tuple, PrimitiveDefinition] = {}


def _zero(*_a):
    return 0


def prim(ins: tuple[int, ...], r: int) -> PrimitiveDefinition:
    """A primitive reading values of widths `ins` and returning `Value<r>`."""
    if (ins, r) not in _PRIMS:
        name = f"RecOp{'x'.join(map(str, ins))}r{r}" if ins else f"RecConst{r}"
        _PRIMS[(ins, r)] = PrimitiveDefinition(name, 1, tuple((f"a{i}", Value(w)) for i, w in enumerate(ins)), Value(r),
                                               _zero, register=False)
    return _PRIMS[(ins, r)]


def leaves_of(c) -> list:
    """A parameter's leaves, in leaf order."""
    return [c] if isinstance(c.type, Value) else [x for i in range(c.type.n) for x in leaves_of(c[i])]


def template(name: str, params: list, outs: list, plan: list):
    """A template: parameters `params` (`(name, type)` each), one output row per type of `outs`, and `plan`, per
    output its gates: `("batch", i, r)`, a primitive mapped over parameter `i`'s elements; else per output leaf
    `(form, r, operands)`, a primitive returning `Value<r>` over the parameter leaves `operands` (`(parameter, leaf)`
    each), called directly (`"op"`) or inside a composite of its own (`"sub"`)."""
    ret = outs[0] if len(outs) == 1 else Tuple(*outs)

    def body(B, S, *args):
        ls, hs = [leaves_of(a) for a in args], []
        for j, (t, made) in enumerate(zip(outs, plan, strict=True)):
            if made[0] == "batch":
                hs.append(B.batch(prim((args[made[1]].type.elem.w,), made[2]), args[made[1]], axes=(0,)))
                continue
            got = []
            for x, (form, r, ops) in enumerate(made):
                f, xs = prim(tuple(ls[i][k].type.w for i, k in ops), r), [ls[i][k] for i, k in ops]
                if form == "sub":
                    sig = (tuple((f"a{n}", h.type) for n, h in enumerate(xs)), Value(r))
                    f = bind(CompositeDefinition(f"{name}Sub{j}x{x}", 1, (), lambda S, sig=sig: sig,
                                                 lambda B, S, *a, f=f: B.call(f, *a), register=False))
                got.append(B.call(f, *xs))
            hs.append(got[0] if isinstance(t, Value) else array_of(got))
        return hs[0] if len(hs) == 1 else tuple_of(*hs)
    return bind(CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False))


def reading(rng, params: list, outs: list, share: float) -> list:
    """A `template` plan: each output leaf a gate over up to three of the parameter leaves a `share` of them names (a
    leaf outside it is read by no gate), now and then inside a composite; with a `share` of 1 the gates read every
    leaf between them. An output with the form of a parameter mapped element by element is, half the time, that
    batch."""
    pool = [(i, k) for i, (_, t) in enumerate(params) for k in range(t.leaves) if rng.random() < share]
    plan: list = []
    for t in outs:
        maps = [i for i, (_, p) in enumerate(params) if isinstance(t, Array) and isinstance(p, Array)
                and isinstance(p.elem, Value) and p.n == t.n]
        if maps and rng.random() < 0.5:
            plan.append(("batch", rng.choice(maps), t.elem.w))
            continue
        plan.append([["sub" if rng.random() < 0.25 else "op", w, rng.sample(pool, min(len(pool), rng.randint(0, 3)))]
                     for w in t.leaf_widths()])
    gates = [g for made in plan if made[0] != "batch" for g in made]
    for x, g in enumerate(gates if share == 1.0 else []):
        g[2] = pool[x::len(gates)]
    for g in gates:
        g[0] = g[0] if g[2] else "op"
    return plan


class Circuit:
    """A circuit as its record states it, built declaration by declaration: `templates`, each `(Definition, units)`;
    `cuts`; `wires` and `reads`, `verity.ml.windows.interface` patterns; the registered `params`, `tables`, `roots`
    (table reads) and `openings`; `ingress`; the inference `entries` with the class's `slots` and `steps`; the `order`
    it states, or None; and `values`, the registered parameters its Program takes as inputs of its own."""

    def __init__(self, name: str, slots: int, steps: int = 1) -> None:
        self.name, self.slots, self.steps = name, slots, steps
        self.templates: list = []
        self.cuts: dict[str, dict[str, int]] = {}
        self.wires: list = []
        self.reads: list = []
        self.params: dict[str, int] = {}
        self.tables: list[dict] = []
        self.roots: list[dict] = []
        self.openings: list[dict] = []
        self.ingress: list[dict] = []
        self.entries: list[dict] = []
        self.order: list | None = None
        self.values: list[str] = []
        self.registered = True

    def add(self, fn, entries: list[tuple[list[int], list[str], int]]) -> str:
        """Template `fn` with its units laid out by `entries` (`(radix, axes, step)` each, in unit order)."""
        tid, at = codec.descriptor_id(fn), 0
        for radix, axes, step in entries:
            self.entries.append({"template": tid, "at": at, "radix": list(radix), "axes": list(axes), "step": step})
            at += math.prod(radix)
        self.templates.append((fn, at))
        return tid

    @property
    def units(self) -> dict[str, int]:
        return {codec.descriptor_id(fn): n for fn, n in self.templates}

    def program(self) -> tuple[str, str]:
        """The circuit's Program, a population of its templates over its values: its descriptor's text and digest."""
        desc = codec.encode_program(mixed_population_program(
            self.templates, values=[(v, self.params[v], 16) for v in self.values]))
        return codec.canonical_json(desc).decode(), codec.program_digest(desc)

    def doc(self, digest: str) -> dict:
        """The circuit's record (`verity/window-circuit/v1`), as far as a stated record's description goes."""
        reg = {"params": dict(self.params), "tables": self.tables, "reads": [r.doc() for r in self.reads],
               "roots": self.roots, "openings": self.openings, **({"values": self.values} if self.values else {})}
        return json.loads(json.dumps({
            "format": IF.CIRCUIT, "name": self.name, "templates": [[t, n] for t, n in self.units.items()],
            "program": digest, "cap": 1, "wires": [w.doc() for w in self.wires],
            **({"cuts": self.cuts} if self.cuts else {}), **({"registered": reg} if self.registered else {}),
            **({"ingress": self.ingress} if self.ingress else {}),
            **({"order": [list(b) for b in self.order]} if self.order is not None else {}),
            "blocks": {"format": IF.BLOCKS, "proof": {"units": sum(self.units.values()), "partition": ""},
                       "inference": {"slots": self.slots, "steps": self.steps, "entries": self.entries}}}))

    def slot(self) -> dict[str, list[int]]:
        """Each unit's slot, entry by entry."""
        out = {t: [0] * n for t, n in self.units.items()}
        for e in self.entries:
            q = math.prod(e["radix"][e["axes"].index("slot") + 1:])
            for i in range(math.prod(e["radix"])):
                out[e["template"]][e["at"] + i] = i // q % self.slots
        return out

    def ordered(self) -> list:
        """An order of the units' ranges its wires follow, each template cut where an entry's slots start over."""
        starts: dict[str, list[int]] = {}
        for e in self.entries:
            period = self.slots * math.prod(e["radix"][e["axes"].index("slot") + 1:])
            starts.setdefault(e["template"], []).extend(range(e["at"], e["at"] + math.prod(e["radix"]), period))
        return [list(b) for b in IF.ordered(self.units, self.wires, starts)]


def grids(c: Circuit) -> dict[str, list[tuple[int, int, int]]]:
    """`NciCheck.Record.gridsOf`: per template its entries in unit order as `(first unit, units, units a slot holds in
    a row)`, neighbours of one row length merged."""
    out: dict[str, list[tuple[int, int, int]]] = {}
    for e in sorted(c.entries, key=lambda e: e["at"]):
        q, got = math.prod(e["radix"][e["axes"].index("slot") + 1:]), out.setdefault(e["template"], [])
        if got and got[-1][2] == q:
            got[-1] = (got[-1][0], got[-1][1] + math.prod(e["radix"]), q)
        else:
            got.append((e["at"], math.prod(e["radix"]), q))
    return out


def searched(S: int, g: dict[str, list[tuple[int, int, int]]], w) -> bool | None:
    """`NciCheck.Record.sameSlot`, mirrored: whether the search shows wire `w` inside one slot, None where it runs out
    of `FUEL`. A box is the wire's reader rows with some digits fixed; it is inside where its reader units lie in one
    grid, its writers in one, no digit carries into a grid's slot digit and every digit moves both slots alike; else
    it is split on the digit that reaches furthest (the writers', or the readers' when only they leave their grid)."""
    if S == 1:
        return True
    m = len(w.radix) - (1 if w.cut[1] > 1 else 0)
    rho = [math.prod(w.radix[k + 1:m]) for k in range(m)] + [0] * (len(w.radix) - m)
    todo, seen = [(w.at, w.base, [d for d in zip(w.radix, rho, w.step) if d[0] > 1])], 0

    def grid(gs, lo: int, hi: int):
        return next(((a, q) for a, n, q in gs if a <= lo and hi <= a + n), None)

    def leaf(gr, gw, at: int, base: int, digs) -> bool:
        (ar, qr), (aw, qw) = gr, gw
        x, y = at - ar, base - aw
        hi_r = sum(p * (r - 1) for r, p, _ in digs if p % qr)
        lo_w = sum(min(0, s * (r - 1)) for r, _, s in digs if s % qw)
        hi_w = sum(max(0, s * (r - 1)) for r, _, s in digs if s % qw)
        return (x % qr + hi_r < qr and 0 <= y % qw + lo_w and y % qw + hi_w < qw and (x // qr - y // qw) % S == 0
                and all(((p // qr if p % qr == 0 else 0) - (s // qw if s % qw == 0 else 0)) % S == 0
                        for _, p, s in digs))

    while todo:
        at, base, digs = todo.pop()
        seen += 1
        if seen > FUEL:
            return None
        units = 1 + sum(p * (r - 1) for r, p, _ in digs)
        lo = base + sum(min(0, s * (r - 1)) for r, _, s in digs)
        hi = base + sum(max(0, s * (r - 1)) for r, _, s in digs)
        gr, gw = grid(g[w.reader], at, at + units), grid(g[w.writer], lo, hi + 1)
        if gr is not None and gw is not None and leaf(gr, gw, at, base, digs):
            continue
        if not digs:
            return False
        reach = [(abs(s) * (r - 1) if gw is None or gr is not None else p * (r - 1)) for r, p, s in digs]
        k = reach.index(max(reach))
        r, p, s = digs[k]
        todo += [(at + d * p, base + d * s, digs[:k] + digs[k + 1:]) for d in reversed(range(r))]
    return True


# The registration log


def _hex(rng) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(128))


def digest_of(text: str) -> str:
    return hashlib.sha512(text.encode()).hexdigest()


def trusted(rng, digest: str, text: str, held: dict[str, tuple[str, int, int]], roots: dict[str, str],
            before: int = 2, after: int = 0) -> dict:
    """A deployment's registration log in R6's chain and the run's digest: the first record registers each value of
    `held` (`(root, leaves, words)`), `before - 1` records follow it, then the run's own, of the Program `digest` and
    the record `text`, committing the registered parameters under `roots`; `after` records follow. With `before` 0 the
    run's record opens the log, and registers `held` itself."""
    entries = [{"value": v, "root": r, "leaves": n, "words": w, "domain": _hex(rng)} for v, (r, n, w) in held.items()]
    run = {"format": V2, "protocol": "verity/sampled-proofs/one-stage/v1", "program": digest,
           "window": {"circuit": digest_of(text), "registered_roots": dict(roots)}, "registered": []}
    log: list[dict] = []
    for k in range(before + 1 + after):
        rec = copy.deepcopy(run) if k == before else {"format": V2, "registered": [
            {"value": f"decoy/{k}/{j}", "root": _hex(rng), "leaves": rng.randint(1, 9), "words": rng.randint(1, 9),
             "domain": _hex(rng), "word_bits": 32} for j in range(rng.randint(0, 2))]}
        if k == 0:
            rec["registered"] = entries + rec["registered"]
        rec["previous"] = reference.record_digest(log[-1]) if log else None
        log.append(rec)
    return {"log": log, "run": reference.record_digest(log[before]), "params": []}


# Random circuits

PORTS = [Value(16), Value(32), Array(2, Value(16)), Array(4, Value(16)), Array(4, Value(8)),
         Array(2, Array(2, Value(16))), Array(32, Value(1)), Array(6, Value(16))]
OUTS = [Value(16), Value(1), Value(32), Array(2, Value(16)), Array(4, Value(16)), Array(8, Value(8)),
        Array(3, Value(16)), Value(8)]
DIGEST = Array(8, Value(64))


def _words(t) -> int:
    return -(-sum(t.leaf_widths()) // 16)


def _layout(rng, S: int) -> list[tuple[list[int], list[str], int]]:
    """One or two inference entries: the slot digit among up to three others, at most one of them the step."""
    out = []
    for step in range(rng.choice([1, 1, 2])):
        radix, axes = [S], ["slot"]
        for _ in range(rng.choice([0, 1, 1, 2, 3])):
            at = rng.randint(0, len(radix))
            radix.insert(at, rng.choice([1, 2, 2, 3]))
            axes.insert(at, "step" if "step" not in axes and rng.random() < 0.4 else "-")
        out.append((radix, axes, step))
    return out


def random_circuit(rng, seed: int) -> Circuit:
    """A random circuit (the module docstring)."""
    S = rng.choice([1, 2, 2, 3, 4])
    c = Circuit(f"record{seed}", S, 2)
    specs = []
    for j in range(rng.randint(1, 4)):
        if rng.random() < 0.2:
            params = [("tok", Value(16)), ("root", DIGEST), ("domain", DIGEST), ("row", Array(2, Value(16))),
                      ("salt", Value(32)), ("path", Array(4, Value(16)))]
            outs = [Array(2, Value(16)), Value(1)]
        else:
            params = [(f"p{i}", rng.choice(PORTS)) for i in range(rng.randint(1, 3))]
            outs = [rng.choice(OUTS) for _ in range(rng.randint(1, 3))]
        fn = template(f"Rec{seed}T{j}", params, outs, reading(rng, params, outs, rng.choice([0.3, 0.7, 1.0])))
        if rng.random() < 0.08:
            fn = prim(tuple(rng.choice([16, 32]) for _ in range(rng.randint(1, 2))), 16)
            if codec.descriptor_id(fn) in c.units:
                continue
            params, outs = list(fn.params), [fn.ret]
        tid = c.add(fn, _layout(rng, S))
        specs.append((tid, params, outs))
        for p, t in params:
            ks = [k for k in range(2, 7) if isinstance(t, Array) and t.n % k == 0 and _words(t) % k == 0]
            if ks and rng.random() < 0.5:
                c.cuts.setdefault(tid, {})[p] = rng.choice(ks)
    slot, units = c.slot(), c.units
    by = {t: [[u for u in range(n) if slot[t][u] == s] for s in range(S)] for t, n in units.items()}
    filled: set[tuple[str, str]] = set()
    for j, (tid, params, _) in enumerate(specs):
        opening = [p for p, _ in params] == ["tok", "root", "domain", "row", "salt", "path"]
        for p, t in params:
            k = c.cuts.get(tid, {}).get(p, 1)
            row = _words(t) // k
            writers = [(wt, f"o{x}") for wt, _, outs in specs[:j] for x, o in enumerate(outs) if _words(o) == row]
            if opening and p != "tok" or not writers or rng.random() < 0.3:
                continue
            wt, out = rng.choice(writers)
            C = rng.choice([1, k]) if k > 1 else 1
            c0 = rng.randrange(k - C + 1)
            lo = rng.randrange(units[tid]) if rng.random() < 0.3 else 0
            hi = rng.randint(lo + 1, units[tid]) if rng.random() < 0.3 else units[tid]
            mode, a, b = rng.choice(["inside", "inside", "cross", "mixed"]), rng.choice([0, 1, 1, 2]), rng.randint(0, 3)
            made = []
            for u in range(lo, hi):
                mine = by[wt][slot[tid][u]]
                rank = by[tid][slot[tid][u]].index(u)
                inside = mode == "inside" or (mode == "mixed" and rng.random() < 0.8)
                v = mine[(rank * a + b) % len(mine)] if inside else (u * a + b) % units[wt]
                d = 0 if inside or C == 1 else rng.choice([0, 1])
                if v + d * (C - 1) >= units[wt]:
                    d = 0
                made.append(IF.Wire(tid, p, wt, out, u, (C,), v, (d,), (c0, C)))
            c.wires += IF.coalesce(made)
            filled.add((tid, p))
    for j, (tid, params, outs) in enumerate(specs):
        if [p for p, _ in params] == ["tok", "root", "domain", "row", "salt", "path"] and rng.random() < 0.85:
            c.params[f"weights/{j}/table"] = 2 * rng.randint(2, 5)
            table = f"table/weights/{j}"
            c.tables.append({"name": table, "param": f"weights/{j}/table", "offset": 0, "words": 2,
                             "rows": c.params[f"weights/{j}/table"] // 2})
            cover = [(0, units[tid])] if rng.random() < 0.8 else [(0, units[tid] - 1)]
            c.roots += [{"reader": tid, "port": of, "table": table, "of": of, "at": a, "units": n}
                        for of in ("root", "domain") if rng.random() < 0.9 for a, n in cover if n]
            c.openings.append({"reader": tid, "table": table, "key": "tok", "ports": ["row", "salt", "path"],
                               "verdict": "o1", "at": 0, "units": units[tid]})
            filled |= {(tid, p) for p in ("root", "domain", "row", "salt", "path")}
        for p, t in params:
            if (tid, p) in filled or rng.random() < 0.3:
                continue
            k = c.cuts.get(tid, {}).get(p, 1)
            row = _words(t) // k
            if rng.random() < 0.15:
                c.ingress.append({"reader": tid, "port": p, "at": 0, "units": rng.randint(1, units[tid]), "step": 0})
                continue
            name = f"weights/{j}/{p}"
            rows = rng.randint(1, 4)
            c.params[name] = row * rows + rng.choice([0, 0, 1])
            a, C = rng.choice([0, 1, 1]), rng.choice([1, k])
            c.reads += IF.coalesce([IF.Read(tid, p, name, u, (C,), row * ((u * a) % rows), (0,), (0, C))
                                    for u in range(units[tid])])
        if rng.random() < 0.1:
            name = f"outputs/{j}"
            c.params[name] = _words(outs[0]) * units[tid]
            c.reads += IF.coalesce([IF.Read(tid, "o0", name, u, (1,), _words(outs[0]) * u, (0,)) for u in range(units[tid])])
    if rng.random() < 0.3:
        c.values = rng.sample(sorted(c.params), rng.randint(0, len(c.params)))
    c.steps = max(e["step"] + (e["radix"][e["axes"].index("step")] if "step" in e["axes"] else 1) for e in c.entries)
    return c


# Circuits in the shape of a served class


def lowered(S: int, T: int, *, cross: bool = False, root_read: bool = True) -> Circuit:
    """A circuit in the shape of a served class of `S` slots and `T` positions (the module docstring), E = 2 words a
    hidden row: `Scan` (one unit a slot, its state a registered read of `scan-state`, the phase's own input), `Embed`
    (a table opening at the token), `Norm`, `Gemm` (per position and slot, two rows of the query weights then four of
    the mixing weights), `Attn<v>` (one template a position, its keys every earlier position's query rows by a cut of
    `v`), `Mix`, `Select` and `Feed` (the next position's token; its state its own earlier unit's). With `cross`, the
    mix of one position reads the attention of the next slot; without `root_read`, no table read of the root."""
    c = Circuit(f"lowered{S}x{T}", S, T)
    E, v16 = 2, Value(16)
    row = Array(E, v16)

    def make(name: str, params: list, outs: list, used: dict[int, list[int]]):
        """A template whose gates, one an output leaf, read between them the leaves `used` names (parameter: leaves)."""
        n = sum(t.leaves for t in outs)
        plan, x = [], 0
        for t in outs:
            plan.append([("op", w, [(i, k) for i, ks in used.items() for k in ks[x + j::n]])
                         for j, w in enumerate(t.leaf_widths())])
            x += t.leaves
        return template(f"Low{name}", params, outs, plan)

    scan = c.add(make("Scan", [("state", Array(4, v16))], [v16, row], {0: [0, 1, 2]}), [([S], ["slot"], 0)])
    embed = c.add(make("Embed", [("tok", v16), ("root", DIGEST), ("domain", DIGEST), ("row", row), ("salt", Value(32)),
                                 ("path", Array(4, v16))], [row, Value(1)],
                       {0: [0], 1: list(range(8)), 2: list(range(8)), 3: [0, 1], 4: [0], 5: [0, 1, 2, 3]}),
                  [([T, S], ["step", "slot"], 0)])
    norm = c.add(make("Norm", [("x", row), ("w", row)], [row], {0: [0, 1], 1: [0, 1]}), [([T, S], ["step", "slot"], 0)])
    gemm = c.add(make("Gemm", [("x", row), ("w", Array(4, v16))], [row], {0: [0, 1], 1: [0, 1, 2]}),
                 [e for t in range(T) for e in (([S, 2], ["slot", "-"], t), ([S, 4], ["slot", "-"], t))])
    attn = [c.add(make(f"Attn{v}", [("q", row), ("k", Array(v, row))], [row], {0: [0, 1], 1: list(range(0, 2 * v, 2))}),
                  [([2, S], ["-", "slot"], v - 1)]) for v in range(1, T + 1)]
    mix = c.add(make("Mix", [("a", Array(2, row)), ("g", Array(4, row)), ("res", row)], [v16, row],
                     {0: [0, 1, 2], 1: [0, 3, 5, 6], 2: [0, 1]}), [([T, S], ["step", "slot"], 0)])
    select = c.add(make("Select", [("x", v16)], [v16], {0: [0]}), [([T, S], ["step", "slot"], 0)])
    feed = c.add(make("Feed", [("sel", v16), ("st", row)], [v16, row], {0: [0], 1: [0]}),
                 [([T - 1, S], ["step", "slot"], 1)]) if T > 1 else None
    c.cuts = {mix: {"a": 2, "g": 4}, **{attn[v - 1]: {"k": v} for v in range(2, T + 1)}}
    G = 6 * S

    def wire(reader, port, writer, out, cells):
        """One wire per reader unit of `cells` (`(unit, first writer, writers' step, rows)`), merged into runs."""
        c.wires += IF.coalesce([IF.Wire(reader, port, writer, out, u, (n,), v, (d if n > 1 else 0,), (0, n))
                                for u, v, d, n in cells])

    wire(embed, "tok", scan, "o0", [(s, s, 0, 1) for s in range(S)])
    wire(norm, "x", embed, "o0", [(u, u, 0, 1) for u in range(T * S)])
    wire(gemm, "x", norm, "o0", [(t * G + i, t * S + (i // 2 if i < 2 * S else (i - 2 * S) // 4), 0, 1)
                                 for t in range(T) for i in range(G)])
    for v in range(1, T + 1):
        t = v - 1
        wire(attn[t], "q", gemm, "o0", [(h * S + s, t * G + 2 * s + h, 0, 1) for h in range(2) for s in range(S)])
        wire(attn[t], "k", gemm, "o0", [(h * S + s, 2 * s + h, G, v) for h in range(2) for s in range(S)])
        wire(mix, "a", attn[t], "o0", [(t * S + s, (s + 1) % S if cross and t == 1 else s, S, 2) for s in range(S)])
    wire(mix, "g", gemm, "o0", [(t * S + s, t * G + 2 * S + 4 * s, 1, 4) for t in range(T) for s in range(S)])
    wire(mix, "res", norm, "o0", [(u, u, 0, 1) for u in range(T * S)])
    wire(select, "x", mix, "o0", [(u, u, 0, 1) for u in range(T * S)])
    if feed is not None:
        wire(feed, "sel", select, "o0", [(u, u, 0, 1) for u in range((T - 1) * S)])
        wire(feed, "st", scan, "o1", [(s, s, 0, 1) for s in range(S)])
        wire(feed, "st", feed, "o1", [(u, u - S, 0, 1) for u in range(S, (T - 1) * S)])
        wire(embed, "tok", feed, "o0", [(u + S, u, 0, 1) for u in range((T - 1) * S)])
    c.params = {"scan-state": 4 * S, "weights/embed/table": 2 * 16, "weights/norm": 2, "weights/query": 2 * 4,
                "weights/mixing": 4 * 4}
    c.tables = [{"name": "table/weights/embed", "param": "weights/embed/table", "offset": 0, "words": 2, "rows": 16}]
    c.roots = [{"reader": embed, "port": of, "table": "table/weights/embed", "of": of, "at": 0, "units": T * S}
               for of in ("domain", "root") if root_read or of == "domain"]
    c.openings = [{"reader": embed, "table": "table/weights/embed", "key": "tok", "ports": ["row", "salt", "path"],
                   "verdict": "o1", "at": 0, "units": T * S}]
    c.reads = IF.coalesce(
        [IF.Read(scan, "state", "scan-state", s, (1,), 4 * s, (0,)) for s in range(S)]
        + [IF.Read(norm, "w", "weights/norm", u, (1,), 0, (0,)) for u in range(T * S)]
        + [IF.Read(gemm, "w", "weights/query" if i < 2 * S else "weights/mixing", t * G + i,
                   (1,), 4 * (i % 2 if i < 2 * S else (i - 2 * S) % 4), (0,)) for t in range(T) for i in range(G)])
    c.order = c.ordered()
    return c


WEIGHTS = ("weights/embed/table", "weights/norm", "weights/query", "weights/mixing")


def shaped() -> list[dict]:
    """The cases on `lowered` circuits, at two slots and three positions and at four slots and 35, each size under one
    policy: `Γ` a block's gates, `α` and `(c, k, n)` exactly what a block holds with the weights registered at the
    start of the deployment. That case is accepted: a block's fresh bits are its scan state's. With none registered,
    and with the run committing none, the weights' reads and the opening are fresh and the rate refuses (`nci`), the
    count `fresh`. Then at two slots and three positions: the query weights under another root than the run commits,
    the opening with no table read of its root, and a wire into the next slot, each refused the same way; the order
    given beside the record, and a deployment whose first record is the run's own, each accepted."""
    out = []
    for n, (S, T) in enumerate([(2, 3), (4, 35)]):
        policy: dict = {}

        def one(what: str, c: Circuit, seed: int, *, held=WEIGHTS, committed=WEIGHTS, moved=(), beside=False,
                before=2, policy=policy, n=n) -> None:
            rng = random.Random(50_000 + 100 * n + seed)
            roots = {p: _hex(rng) for p in c.params}
            order = None
            if beside:
                order, c.order = c.order, None
            made = finish(rng, c, {p: (_hex(rng) if p in moved else roots[p], c.params[p] // 2, 2) for p in held},
                          {p: roots[p] for p in (*committed, "scan-state")}, before=before, order=order, policy=policy)
            out.append(made | {"shape": what, "broken": None})

        one(f"{S}x{T} registered", lowered(S, T), 0)
        one(f"{S}x{T} none registered", lowered(S, T), 1, held=())
        one(f"{S}x{T} none committed", lowered(S, T), 2, committed=())
        if n == 0:
            one("2x3 another root", lowered(S, T), 3, moved=("weights/query",))
            one("2x3 no root read", lowered(S, T, root_read=False), 4)
            one("2x3 a wire into the next slot", lowered(S, T, cross=True), 5)
            one("2x3 the order beside the record", lowered(S, T), 6, beside=True)
            one("2x3 the run's record first", lowered(S, T), 7, before=0)
    return out


# Cases


def finish(rng, c: Circuit, held: dict[str, tuple[str, int, int]], roots: dict[str, str], *, before: int = 2,
           after: int = 0, order=None, policy: dict | None = None) -> dict:
    """The case on circuit `c`: its instance (the Program, the record, a gate table, `Γ` and `α`), the trusted input
    (`trusted`), the wires as SPOT reads them and `(c, k, n)`; with what it was made from. The limits are at, just
    under or just over what the circuit's blocks hold; or `policy`'s, which an empty `policy` is first filled with: a
    table of every primitive and exactly what the blocks hold."""
    text, digest = c.program()
    record = IF.circuit_bytes(c.doc(digest)).decode()
    inside = [all(s == t for s, t in _slots(c, w)) for w in c.wires]
    g = grids(c)
    for w, want in zip(c.wires, inside, strict=True):
        if searched(c.slots, g, w) is not want:
            raise AssertionError(f"{c.name}: the search doesn't decide wire {w.doc()} within its fuel")
    reference.desc_program(text)
    prims = sorted({p for fn, _ in c.templates for p in ref.template_use(fn)[2]})
    if policy is None:
        table = {p: rng.choice([0, 1, 2, rng.randint(0, 40)]) for p in prims}
        if rng.random() < 0.1:
            del table[rng.choice(prims)]
    else:
        table = policy.get("table", {p: 3 + 2 * k for k, p in enumerate(prims)})
    inst = {"program": text, "record": record, "table": table, "gamma": BIG, "alpha": [BIG, 1],
            **({"order": order} if order is not None else {})}
    tr = trusted(rng, digest, record, held, roots, before, after)
    probe = ref.check_detail(inst, tr)
    assert "error" not in probe, (c.name, probe)

    def near(x: int) -> int:
        return x if policy is not None else max(0, x + rng.choices([0, -1, 1, rng.randint(-3, 3)], [5, 2, 2, 1])[0])
    den = rng.randint(1, 5) if policy is None else probe["gates"]
    limits = {"table": table, "gamma": near(probe["gates"]),
              "alpha": [near(-(-probe["fresh"] * den // max(probe["gates"], 1))), den],
              "count": [near(probe["bool_gates"] or 0), near(probe["fresh"]), near(probe["blocks"])]}
    if policy is not None:
        limits = policy if policy else policy.update(limits) or policy
    inst.update(gamma=limits["gamma"], alpha=limits["alpha"])
    return {"case": {"instance": inst, "trusted": tr, "spot": spot_of(c, json.loads(record), order),
                     "count": limits["count"]}, "circuit": c,
            "inside": inside, "theta": sorted(p for p, (root, n, w) in held.items()
                                              if roots.get(p) == root and n * w == c.params[p])}


def spot_of(c: Circuit, doc: dict, order=None) -> dict:
    """The record `doc`'s wires as SPOT's `wires` piece reads them (`Spot.Wires.Wiring.ofJson`), each template's rows
    and outputs Python's own layout of it (`verity.ml.windows.interface.layout`, what `check_wires` reads): the order
    the record's, else `order` where one is given beside it."""
    lays = {codec.descriptor_id(fn): (IF.layout(fn), n) for fn, n in c.templates}
    return {"templates": {t: {"units": n, "rows": {p: [w, o] for (p, w), o in zip(lay.rows, lay.outer, strict=True)},
                              "outs": dict(lay.outs)} for t, (lay, n) in lays.items()},
            "cuts": doc.get("cuts", {}), "wires": doc.get("wires"),
            **({"order": doc["order"]} if "order" in doc else {"order": order} if order is not None else {})}


def _slots(c: Circuit, w):
    """Each reader row of wire `w`: its reader's slot and its writer's."""
    slot = c.slot()
    for u, _, v in ref.entries({**w.doc(), "c0": w.cut[0], "C": w.cut[1]}):
        yield slot[w.reader][u], slot[w.writer][v]


def case(seed: int) -> dict:
    """Case `seed`: below `COUNT` a random circuit with a random registration log, broken with probability 0.4 in one
    of `BROKEN`'s ways; then `shaped()`'s."""
    if seed >= COUNT:
        return shaped()[seed - COUNT]
    rng = random.Random(seed)
    c = random_circuit(rng, seed)
    roots = {p: _hex(rng) for p in c.params if rng.random() < 0.9}
    held = {}
    for p, n in c.params.items():
        how = rng.choice(["held", "held", "held", "none", "root", "shape"])
        if how == "none":
            continue
        w = rng.choice([d for d in range(1, n + 1) if n % d == 0])
        held[p] = (roots.get(p, _hex(rng)) if how != "root" else _hex(rng), n // w + (how == "shape"), w)
    made = finish(rng, c, held, roots, before=rng.choice([0, 1, 2, 2, 3]), after=rng.choice([0, 0, 1]))
    stage = rng.choice(sorted(BROKEN)) if rng.random() < 0.4 else None
    how = rng.choice(BROKEN[stage]) if stage else None
    if how and not _break(rng, made, how):
        stage = how = None
    if stage in ("instance", "params", "program", "not-topological", "record", "trusted", "run-program", "run-circuit",
                 "templates"):
        made["case"]["spot"] = None
    return made | {"shape": "random", "broken": stage, "how": how}


def _rewrite(made: dict, edit) -> None:
    """The record edited by `edit` (on its object), with the run's own record still committing its bytes."""
    case = made["case"]
    doc = json.loads(case["instance"]["record"])
    edit(doc)
    _commit(case, IF.circuit_bytes(doc).decode())


def _commit(case: dict, record: str, program: str | None = None) -> None:
    """The instance's record replaced by `record`, the run's own record committing it (and the Program `program`,
    where one is given), and the log's chain remade."""
    tr = case["trusted"]
    at = [reference.record_digest(r) for r in tr["log"]].index(tr["run"])
    case["instance"]["record"] = record
    tr["log"][at]["window"]["circuit"] = digest_of(record)
    if program is not None:
        tr["log"][at]["program"] = program
    for k in range(1, len(tr["log"])):
        tr["log"][k]["previous"] = reference.record_digest(tr["log"][k - 1])
    tr["run"] = reference.record_digest(tr["log"][at])
    doc = json.loads(record)
    if case.get("spot") is not None and isinstance(doc, dict):
        case["spot"] = {"templates": case["spot"]["templates"], "cuts": doc.get("cuts", {}), "wires": doc.get("wires"),
                        **({"order": doc["order"]} if "order" in doc else
                           {"order": case["instance"]["order"]} if "order" in case["instance"] else {})}


def _break(rng, made: dict, how: str) -> bool:
    """Break the case in the way `how` names; False where the circuit has nothing to break that way."""
    case, c = made["case"], made["circuit"]
    inst, tr = case["instance"], case["trusted"]
    doc = json.loads(inst["record"])
    units = c.units
    ids = list(units)
    wires, reg = doc["wires"], doc.get("registered", {})
    if how == "no-field":
        del inst[rng.choice(ref.FIELDS)]
    elif how == "program-type":
        inst["program"] = json.loads(inst["program"])
    elif how == "record-type":
        inst["record"] = doc
    elif how == "table-type":
        inst["table"] = rng.choice([[], 3, "{}"])
    elif how == "gamma":
        inst["gamma"] = rng.choice([-1, 1.5, "2", None, True])
    elif how == "alpha":
        inst["alpha"] = rng.choice([[1, 0], [1], 3, [1, -2], [1.5, 2], "1/2", [1, 2, 3]])
    elif how == "table-value":
        inst["table"][rng.choice(sorted(inst["table"]) or ["RecOp16r16_v1"])] = rng.choice([-1, 1.5, "2", None])
    elif how == "program-text":
        inst["program"] = rng.choice(["", "{", "[]", inst["program"].replace('"root":', '"rooot":', 1)])
    elif how == "forward":
        return _forward(inst)
    elif how == "record-text":
        inst["record"] = rng.choice(["", "{", inst["record"][:-2]])
    elif how == "record-list":
        _commit(case, rng.choice(["[]\n", "3\n", '"record"\n', "null\n"]))
    elif how == "trusted-keys":
        if rng.random() < 0.5:
            del tr[rng.choice(sorted(tr))]
        else:
            tr["window"] = {}
    elif how == "chain":
        tr["log"][-1]["previous"] = rng.choice([_hex(rng), 3]) if len(tr["log"]) > 1 else _hex(rng)
        tr["run"] = reference.record_digest(tr["log"][-1]) if rng.random() < 0.5 else tr["run"]
    elif how == "run-unknown":
        tr["run"] = rng.choice([_hex(rng), 3, None])
    elif how in ("run-program", "run-circuit", "no-window"):
        at = [reference.record_digest(r) for r in tr["log"]].index(tr["run"])
        run = tr["log"][at]
        if how == "run-program":
            run["program"] = rng.choice([_hex(rng), None, 3])
        elif how == "run-circuit":
            run["window"]["circuit"] = rng.choice([_hex(rng), None, digest_of(inst["record"]).upper()])
        else:
            run["window"] = rng.choice([None, [], "window"])
            if run["window"] is None and rng.random() < 0.5:
                del run["window"]
        for k in range(1, len(tr["log"])):
            tr["log"][k]["previous"] = reference.record_digest(tr["log"][k - 1])
        tr["run"] = reference.record_digest(run)
    elif how == "record-program":
        _rewrite(made, lambda d: d.update(program=rng.choice([_hex(rng), None])))
    elif how == "units":
        _rewrite(made, lambda d: d["templates"][rng.randrange(len(ids))].__setitem__(1, rng.choice([0, 1.5, units[ids[0]] + 1])))
    elif how == "swap":
        if len(ids) < 2:
            return False
        _rewrite(made, lambda d: d.update(templates=d["templates"][::-1]))
    elif how == "half-word":
        fn = template(f"Half{c.name}", [("p", Value(rng.choice([8, 24])))], [Value(16)], [[("op", 16, [(0, 0)])]])
        text, digest = Circuit.program(_only(c, fn))
        inst["program"] = text
        _commit(case, IF.circuit_bytes({**doc, "program": digest, "templates": [[codec.descriptor_id(fn), c.slots]],
                                         "wires": [], "cuts": {}}).decode(), digest)
    elif how == "overlap":
        if not wires:
            return False
        _rewrite(made, lambda d: d["wires"].append(copy.deepcopy(rng.choice(d["wires"]))))
    elif how in ("bounds", "width", "wire-port", "wire-template", "wire-cut"):
        if not wires:
            return False
        k = rng.randrange(len(wires))

        def edit(d: dict) -> None:
            w = d["wires"][k]
            if how == "bounds":
                w.update(rng.choice([{"at": w["at"] + units[w["reader"]]}, {"base": w["base"] + units[w["writer"]]},
                                     {"base": -1 - w["base"]}]))
            elif how == "width":
                d["wires"][k] = _other_width(c, w) or {**w, "out": "o9"}
            elif how == "wire-port":
                w.update(rng.choice([{"port": "nope"}, {"out": "o9"}, {"port": "o0"}]))
            elif how == "wire-template":
                w.update(rng.choice([{"reader": "Nope_v1"}, {"writer": "Nope_v1"}]))
            else:
                w.update(rng.choice([{"cut": [0, 0]}, {"cut": [7, 1]}, {"cut": [0, 2], "radix": [*w["radix"], 3],
                                                                         "step": [*w["step"], 0]}, {"cut": [1]}]))
        _rewrite(made, edit)
    elif how == "cycle":
        if not wires:
            return False
        w = rng.choice(wires)
        back = {"reader": w["writer"], "port": w["port"], "writer": w["reader"], "out": w["out"], "at": 0, "radix": [1],
                "base": 0, "step": [0]}
        _rewrite(made, lambda d: d["wires"].append({**w, "writer": w["reader"]} if rng.random() < 0.5 else back))
    elif how == "order":
        bad = rng.choice([[[ids[0], 0, units[ids[0]] + 1]], [["Nope_v1", 0, 1]], [[ids[0], 0]], 3, None,
                          [[t, 0, n] for t, n in units.items()][1:] + [[ids[0], 1, units[ids[0]]]]])
        if rng.random() < 0.5:
            _rewrite(made, lambda d: d.update(order=bad))
        else:
            inst["order"] = case["spot"]["order"] = bad
    elif how == "no-wires":
        _rewrite(made, lambda d: d.pop("wires") if rng.random() < 0.5 else d.update(wires={}))
    elif how in ("read-param", "read-leaves", "read-on-wire", "table-param", "root-port", "verdict", "ingress-units",
                 "registered-keys"):
        if "registered" not in doc:
            return False
        reads = reg["reads"]

        def edit(d: dict) -> None:
            r = d["registered"]
            if how == "read-param":
                rng.choice(r["reads"]).update(param="weights/nope")
            elif how == "read-leaves":
                x = rng.choice(r["reads"])
                x.update(rng.choice([{"base": x["base"] + r["params"][x["param"]]}, {"base": -1 - x["base"]},
                                     {"at": x["at"] + units[x["reader"]]}, {"radix": [0] * len(x["radix"])}]))
            elif how == "read-on-wire":
                w = rng.choice(d["wires"])
                p = sorted(r["params"])[0]
                r["reads"].append({"reader": w["reader"], "port": w["port"], "param": p, "at": w["at"], "radix": [1],
                                   "base": 0, "step": [0], **({"cut": [w["cut"][0], 1]} if "cut" in w else {})})
                r["params"][p] += 64
            elif how == "table-param":
                r["tables"].append(rng.choice([{"name": "table/nope", "param": "weights/nope", "offset": 0, "words": 1,
                                                "rows": 1},
                                               {"name": "table/big", "param": sorted(r["params"])[0], "offset": 1,
                                                "words": r["params"][sorted(r["params"])[0]], "rows": 1},
                                               {"name": "table/zero", "param": sorted(r["params"])[0], "offset": 0,
                                                "words": 0, "rows": 1}]))
            elif how == "root-port":
                x = rng.choice(r["roots"])
                x.update(rng.choice([{"port": "tok"}, {"of": "opened"}, {"table": "table/nope"}, {"units": 0},
                                     {"at": units[x["reader"]]}, {"port": "o0"}]))
            elif how == "verdict":
                x = rng.choice(r["openings"])
                x.update(rng.choice([{"verdict": "row"}, {"key": "o0"}, {"ports": []}, {"ports": ["row", "o1"]},
                                     {"reader": "Nope_v1"}, {"table": "table/nope"}, {"ports": ["row", "row"]}]))
            elif how == "ingress-units":
                t = rng.choice(ids)
                d.setdefault("ingress", []).append(rng.choice([
                    {"reader": t, "port": "nope", "at": 0, "units": 1, "step": 0},
                    {"reader": t, "port": c.templates[ids.index(t)][0].signature()[0][0][0], "at": units[t], "units": 1,
                     "step": 0},
                    {"reader": t, "port": c.templates[ids.index(t)][0].signature()[0][0][0], "at": 0, "units": 1,
                     "step": -1}]))
            else:
                rng.choice([lambda: r.pop("roots"), lambda: r.update(extra=[]), lambda: r.update(params=[]),
                            lambda: r["params"].update(zero=0)])()
        if (how in ("read-param", "read-leaves") and not reads or how == "read-on-wire" and not (wires and reg["params"])
                or how == "table-param" and not reg["params"] or how == "root-port" and not reg["roots"]
                or how == "verdict" and not reg["openings"]):
            return False
        _rewrite(made, edit)
    elif how == "no-blocks":
        _rewrite(made, lambda d: rng.choice([lambda: d.pop("blocks"), lambda: d["blocks"].pop("inference"),
                                             lambda: d["blocks"]["inference"].update(slots=0),
                                             lambda: d["blocks"]["inference"].update(entries={})])())
    else:
        def edit(d: dict) -> None:
            es = d["blocks"]["inference"]["entries"]
            e = rng.choice(es)
            at = e["axes"].index("slot")
            if how == "slot-radix":
                e["radix"][at] += 1
            elif how == "two-slots":
                e.update(rng.choice([{"axes": ["slot", *e["axes"]], "radix": [c.slots, *e["radix"]]},
                                     {"axes": ["-" if a == "slot" else a for a in e["axes"]]},
                                     {"axes": e["axes"][1:]}, {"radix": []}]))
            elif how == "gap":
                es.remove(e)
            elif how == "twice":
                es.append(copy.deepcopy(e))
            else:
                e["template"] = "Nope_v1"
        _rewrite(made, edit)
    return True


def _only(c: Circuit, fn) -> Circuit:
    """A circuit of the one template `fn`, a unit a slot."""
    out = Circuit(c.name, c.slots)
    out.add(fn, [([c.slots], ["slot"], 0)])
    return out


def _other_width(c: Circuit, w: dict) -> dict | None:
    """Wire `w` reading an output of another width, where its writer has one."""
    fn = next(fn for fn, _ in c.templates if codec.descriptor_id(fn) == w["writer"])
    outs = [(f"o{j}", _words(t)) for j, t in enumerate(ref.fields_of(fn.signature()[1]))]
    mine = dict(outs)[w["out"]]
    other = [o for o, n in outs if n != mine]
    return {**w, "out": other[0]} if other else None


def _forward(inst: dict) -> bool:
    """A gate of some template redirected to read itself (`not-topological`)."""
    desc = json.loads(inst["program"])
    for key in sorted(desc["definitions"]):
        d = desc["definitions"][key]
        if d.get("form") != "composite" or key == desc["root"]:
            continue
        nodes = d["body"]["nodes"]
        for k in range(len(nodes) - 1, -1, -1):
            for arg in nodes[k].get("args", []) if nodes[k]["form"] == "primitive" else []:
                if "affine" in arg["refs"] and arg["refs"]["affine"][3] == 1:
                    arg["refs"] = {"affine": ["n", k, 0, 1]}
                    inst["program"] = codec.canonical_json(desc).decode()
                    return True
    return False


def line(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


NAMES = ("cases", "python")


def total() -> int:
    """How many cases the audit's data has: the random ones and `shaped()`'s."""
    return COUNT + 11


def results(case: dict) -> dict:
    """The reference's results on a case: the whole verdict, what is published, and the count."""
    detail = ref.check_detail(case["instance"], case["trusted"])
    return {"detail": detail, "public": ref.check(case["instance"], case["trusted"]),
            "count": ref.count(detail, *case["count"])}


def write(out: Path, count: int | None = None) -> dict[str, Path]:
    """The two files for cases 0 … count − 1 (all of them, `total()`, by default) under ``out/record_difftest``."""
    count = total() if count is None else count
    d = out / "record_difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {f"record_difftest/{n}.jsonl": d / f"{n}.jsonl" for n in NAMES}
    files = [open(paths[f"record_difftest/{n}.jsonl"], "w") for n in NAMES]
    try:
        late = shaped() if count > COUNT else []
        for seed in range(count):
            made = (late[seed - COUNT] if seed >= COUNT else case(seed))["case"]
            for f, obj in zip(files, (made, results(made)), strict=True):
                f.write(line(obj))
    finally:
        for f in files:
            f.close()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.ml.circuits reads the programs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=None, help="the first N cases (default: all, `total()`)")
    a = ap.parse_args()
    here = Path(verity.ml.circuits.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.ml.circuits comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
