"""The record form's difftest data: execution phases' circuit records with their Programs, registration logs and limits,
for `NciCheck/RecordDifftestMain.lean`.

    python scripts/record_difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes two files under `DIR/record_difftest/`, one line per case: `cases.jsonl`, `{"instance", "trusted", "spot",
"rows", "count"}`: the instance and the trusted input as `NciCheck.Record.check` reads them (`Record`'s
docstring), the record's wires as SPOT's `wires` piece reads them (`spot_of`, each template's rows and outputs
Python's own layout of it; `null` where the reference refuses the case before its wires, or that layout has no row
for a port: leaves of two widths, or an output past a word that is not whole words), some input rows of its units (`[template, port, unit, row]` each) and the count's limits
(`{"table", "c", "k", "n"}`); and `python.jsonl`, `{"detail", "public", "kinds", "exported", "count", "closed"}`:
`nci_record_reference`'s `check_detail`, `check`, `kinds` and `exported` of those rows, `count`, and each
template's closed rows (`closed`). The Lean audit runs `NciCheck/RecordDifftestMain.lean` on the two (`runs` in
`lean-audit.json`).

Case `seed`, for the first `COUNT`, is a random circuit (`random_circuit`): one to four templates of a few gates, each
reading some of its parameters' leaves (directly, through a call and through a batch) and now and then passing some
of a parameter's one-bit leaves on (`PASSES`: itself, through a call, a callee inside a callee, a batch member or a
scan, some only through its carry or only as its outputs, or handed to a callee that returns none of them), their units laid out by one or two inference entries each with the slot digit anywhere
among the digits; wires from earlier templates that keep to the reader's slot, cross slots or do either unit by unit,
each from an output of the row's bits whose leaves' width divides the reader's (the same width, or bits under a
word) or, into a closed row, from any output of the row's bits, written unit by unit and merged into runs
(`verity.commitments.windows.interface.coalesce`, as a lowering states them); registered reads, a table with its table reads and
opening, and ingress; a registration log in R6's chain whose first record registers some of the parameters, under the
root the run's own record commits or another; and pins of most templates' Definitions, among other digests. Two in
five are broken in one way, each refused at one stage (`BROKEN`, every way in turn, `WAYS`). Then `shaped()`:
circuits in the shape of a served class (`lowered`: a slot scan, an embedding read through a table opening, a norm,
matmul rows over registered weights, an attention template per position reading every earlier position's rows by a
cut, a mix, a selection and a token feed reading its own earlier units), at two slots and three positions and at the
served class's four slots and 35, with the weights registered at the start of the deployment and with none, under
another root, with the opening's root read dropped, with the opening's reader another Definition under the pinned
one's name whose verdict is a constant 1, with one wire from the next slot, and with the order given beside the
record; and the negative vectors of a pinned reader's layout (the form's `layoutOf`), one a role: the key listed
among the opening's ports, another row named the key, the root or the domain id read in another row, an opened row
left out or out of order, another output named the verdict, and a pinned reader whose rows are in no layout's order
or that has one output, with the same records under no pin. Then a wire's leaves: the served shape with its token a
word into a key of 32 one-bit leaves that only gates of its reader read, a closed row, and 32 one-bit leaves into a
key of one word, both taken; a template that passes on one bit of each of sixteen 16-bit gates, read as one 16-bit
leaf by a unit of another slot (`passed_on`), refused however it passes them on (itself, with no gate of its own at
all, through a call, a callee inside a callee, a batch member or a scan, or to an output no wire reads), refused
where it passes on one of them alone through a scan's carry or the other fifteen as a scan's outputs, refused too
where it hands them to a callee that returns none of them, and taken where its output is
gates of its own over them; an output whose leaves have two widths (`two_widths`), refused into a 32-bit leaf and
taken into a closed row; an output of 8 bits (`short_output`), which lacks bits of the 16-bit row it fills, refused
into one leaf and into a closed row alike; and an output of 31 bits (`odd_output`), two words, its bits rounded
up: refused into a row of two words, which holds a bit it lacks, and a record that wires it into a row of one
word refused by the wires' own check. Then a root that
hands a unit a row that is not input gates of its own: the served shape with one input run handed to two rows, with
a row every unit shares, and with a row of computed gates. Then a text the form reads nothing of: the served shape's
record as one that lists its members, which states no wires, refused by the wires' reader; and the text of no wire
and nothing else, which names no Program, refused with its templates. Then a row declared twice (`TWICE`): the served
shape's record with one declaration more, into a row it already declares, a case each pair of a wire, a registered
read, a table read, an opening's port and ingress, each refused. Then a template with an output that has no bit
(`BITLESS`): one that returns no field, and one with a field of no bit beside a field of 16, each refused with its
templates. Then a template keyed as an input id (`KEYED`), bound per run and bound at registration: a composite, so
its gates are computed and a root node of it is no input. The record of that one template, and the served shape's
record with a unit of it more at the Program's root that the record does not list, each refused with its templates.

Every wire of every case is one `Record.sameSlot`'s search decides within its fuel (`searched`, the search
mirrored and held to the expansion as the cases are made), so the Lean's count of a wire is the reference's.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path

import verity.circuits
import verity.commitments.windows.interface as IF
from verity.circuits import codec
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.population import mixed_population_program
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple, Value

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
#: `Record.FUEL`: the boxes `sameSlot`'s search visits for one wire before it counts the wire as crossing
FUEL = 4096
V2 = "verity/registration/v2"
#: how a random case is broken, by the stage that refuses it
BROKEN = {
    "instance": ["no-field", "program-type", "record-type", "pins-type"],
    "params": ["gamma", "alpha", "pin-value"],
    "program": ["program-text"],
    "not-topological": ["forward"],
    "record": ["record-text", "record-list"],
    "trusted": ["trusted-keys", "chain", "run-unknown"],
    "run-program": ["run-program"],
    "run-circuit": ["run-circuit", "no-window"],
    "templates": ["record-program", "units", "swap", "half-word"],
    "own-rows": ["row-twice", "row-aliased", "row-broadcast", "row-computed", "row-explicit", "row-strided"],
    "wires": ["overlap", "bounds", "width", "wire-port", "wire-template", "wire-cut", "cycle", "order", "no-wires"],
    "leaf-widths": ["leaf-wider", "leaf-short"],
    "inputs": ["read-param", "read-leaves", "read-on-wire", "table-param", "root-port", "verdict", "ingress-units",
               "registered-keys"],
    "blocks": ["no-blocks", "slot-radix", "two-slots", "gap", "twice", "entry-template"],
    "reader-layout": ["reader-other"],
    "reader-roles": ["key-port", "key-row", "root-row", "domain-row", "rows-swapped", "port-out", "ports-order",
                     "verdict-out"],
}
#: each way with its stage: the broken cases take the ways in turn, in seed order (`_turns`)
WAYS = [(stage, how) for stage, hows in BROKEN.items() for how in hows]
#: the stages that refuse a case before SPOT reads its wires: such a case carries no wires for SPOT's reader
EARLY = ("instance", "params", "program", "not-topological", "record", "trusted", "run-program", "run-circuit",
         "templates", "own-rows")
#: the circuits a broken case tries for one that has what its way breaks
TRIES = 40

_PRIMS: dict[tuple, PrimitiveDefinition] = {}


def _zero(*_a):
    return 0


def prim(ins: tuple[int, ...], r: int) -> PrimitiveDefinition:
    """A primitive reading values of widths `ins` and returning `Value<r>`: 0, or with no operand the constant of `r`
    ones."""
    if (ins, r) not in _PRIMS:
        name = f"RecOp{'x'.join(map(str, ins))}r{r}" if ins else f"RecConst{r}"
        _PRIMS[(ins, r)] = PrimitiveDefinition(name, 1, tuple((f"a{i}", Value(w)) for i, w in enumerate(ins)), Value(r),
                                               _zero if ins else lambda: (1 << r) - 1, register=False)
    return _PRIMS[(ins, r)]


def leaves_of(c) -> list:
    """A parameter's leaves, in leaf order."""
    return [c] if isinstance(c.type, Value) else [x for i in range(c.type.n) for x in leaves_of(c[i])]


#: how a template's output passes leaves of a parameter on (`template`'s `"pass"`)
PASSES = ("direct", "call", "nested", "batch", "scan", "carry", "ys", "beside")


def _hand(name: str, params: list, ret, body):
    """A composite of no gate: `body` returns leaves of its parameters."""
    return bind(CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False))


def passed(B, name: str, bits: list, how: str):
    """The one-bit leaves `bits` passed on, `how` (`PASSES`), as an array of as many one-bit leaves, with no gate
    where every one is passed on. `"direct"`: the leaves themselves. `"call"`: the return of a composite that
    returns its parameter. `"nested"`: the return of a composite that calls that one, a callee inside a callee.
    `"batch"`: a composite that returns its one bit, mapped over them. `"scan"`: a step that returns its element as
    the carry and its carry as the element, the first leaf its initial carry and the rest its elements: the outputs
    then the carry are the leaves, in order, each element reaching its output through the carry of the iteration
    before. `"carry"` and `"ys"`: a step that returns its carry unchanged and its element as its output, the scan's
    carry then its outputs returned. With `"carry"` the first leaf is the initial carry and the elements are gates,
    one over each other leaf: one leaf is passed on, through every iteration's carry. With `"ys"` the initial carry
    is a gate over the first leaf and the elements are the other leaves: they are passed on, as the scan's outputs.
    `"beside"`: no leaf is passed on; a composite that returns the second half of its parameter is called on the
    leaves then as many gates, one over each leaf, and returns the gates."""
    bit, n = Value(1), len(bits)
    if how == "direct":
        return array_of(bits)
    if how in ("call", "nested"):
        same = _hand(name, [("a", Array(n, bit))], Array(n, bit), lambda B, S, a: a)
        if how == "nested":
            same = _hand(f"{name}Outer", [("a", Array(n, bit))], Array(n, bit), lambda B, S, a, same=same: B.call(same, a))
        return B.call(same, array_of(bits))
    if how == "batch":
        return B.batch(_hand(name, [("a", bit)], bit, lambda B, S, a: a), array_of(bits), axes=(0,))
    if how == "scan":
        step = _hand(name, [("carry", bit), ("x", bit)], Tuple(bit, bit), lambda B, S, carry, x: tuple_of(x, carry))
        out = B.scan(step, bits[0], [array_of(bits[1:])])
        return array_of([*leaves_of(out[1]), out[0]])
    if how in ("carry", "ys"):
        step = _hand(name, [("carry", bit), ("x", bit)], Tuple(bit, bit), lambda B, S, carry, x: tuple_of(carry, x))
        gate = [B.call(prim((1,), 1), b) for b in (bits[1:] if how == "carry" else bits[:1])]
        out = B.scan(step, *((bits[0], [array_of(gate)]) if how == "carry" else (gate[0], [array_of(bits[1:])])))
        return array_of([out[0], *leaves_of(out[1])])
    assert how == "beside", how
    half = _hand(name, [("a", Array(2 * n, bit))], Array(n, bit), lambda B, S, a: array_of(leaves_of(a)[n:]))
    return B.call(half, array_of([*bits, *(B.call(prim((1,), 1), b) for b in bits)]))


def template(name: str, params: list, outs: list, plan: list):
    """A template: parameters `params` (`(name, type)` each), one output row per type of `outs`, and `plan`, per
    output its gates: `("batch", i, r)`, a primitive mapped over parameter `i`'s elements; `("pass", i, leaves, how)`,
    the output's leaves parameter `i`'s one-bit `leaves`, at least two, passed on (`passed`; `how` one of `PASSES`,
    `"direct"` where none is given); else per output leaf `(form, r, operands)`, a primitive returning `Value<r>`
    over the parameter leaves `operands` (`(parameter, leaf)` each), called directly (`"op"`) or inside a composite
    of its own (`"sub"`). An output of no leaf, a Tuple of no field, has no gate."""
    ret = outs[0] if len(outs) == 1 else Tuple(*outs)

    def body(B, S, *args):
        ls, hs = [leaves_of(a) for a in args], []
        for j, (t, made) in enumerate(zip(outs, plan, strict=True)):
            if made and made[0] == "batch":
                hs.append(B.batch(prim((args[made[1]].type.elem.w,), made[2]), args[made[1]], axes=(0,)))
                continue
            if made and made[0] == "pass":
                hs.append(passed(B, f"{name}Hand{j}", [ls[made[1]][k] for k in made[2]], (*made, "direct")[3]))
                continue
            got = []
            for x, (form, r, ops) in enumerate(made):
                f, xs = prim(tuple(ls[i][k].type.w for i, k in ops), r), [ls[i][k] for i, k in ops]
                if form == "sub":
                    sig = (tuple((f"a{n}", h.type) for n, h in enumerate(xs)), Value(r))
                    f = bind(CompositeDefinition(f"{name}Sub{j}x{x}", 1, (), lambda S, sig=sig: sig,
                                                 lambda B, S, *a, f=f: B.call(f, *a), register=False))
                got.append(B.call(f, *xs))
            hs.append(got[0] if isinstance(t, Value) else tuple_of(*got) if isinstance(t, Tuple) else array_of(got))
        return hs[0] if len(hs) == 1 else tuple_of(*hs)
    return bind(CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False))


def reading(rng, params: list, outs: list, share: float) -> list:
    """A `template` plan: each output leaf a gate over up to three of the parameter leaves a `share` of them names (a
    leaf outside it is read by no gate), now and then inside a composite; with a `share` of 1 the gates read every
    leaf between them. An output with the form of a parameter mapped element by element is, half the time, that
    batch. An output of one-bit leaves is, half the time where a parameter has as many, some of that parameter's
    leaves passed on (`PASSES`, any of them)."""
    pool = [(i, k) for i, (_, t) in enumerate(params) for k in range(t.leaves) if rng.random() < share]
    plan: list = []
    for t in outs:
        maps = [i for i, (_, p) in enumerate(params) if isinstance(t, Array) and isinstance(p, Array)
                and isinstance(p.elem, Value) and p.n == t.n]
        if maps and rng.random() < 0.5:
            plan.append(("batch", rng.choice(maps), t.elem.w))
            continue
        hands = [i for i, (_, p) in enumerate(params) if isinstance(t, Array) and t.elem == Value(1)
                 and set(p.leaf_widths()) == {1} and p.leaves >= t.n]
        if hands and rng.random() < 0.5:
            i = rng.choice(hands)
            plan.append(("pass", i, rng.sample(range(params[i][1].leaves), t.n), rng.choice(PASSES)))
            continue
        plan.append([["sub" if rng.random() < 0.25 else "op", w, rng.sample(pool, min(len(pool), rng.randint(0, 3)))]
                     for w in t.leaf_widths()])
    gates = [g for made in plan if isinstance(made, list) for g in made]
    for x, g in enumerate(gates if share == 1.0 else []):
        g[2] = pool[x::len(gates)]
    for g in gates:
        g[0] = g[0] if g[2] else "op"
    return plan


class Circuit:
    """A circuit as its record states it, built declaration by declaration: `templates`, each `(Definition, units)`;
    `cuts`; `wires` and `reads`, `verity.commitments.windows.interface` patterns; the registered `params`, `tables`, `roots`
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
            "program": digest, "wires": [w.doc() for w in self.wires],
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
    """`RecordText.gridsOf`: per template its entries in unit order as `(first unit, units, units a slot holds in
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
    """`Record.sameSlot`, mirrored: whether the search shows wire `w` inside one slot, None where it runs out
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
         Array(2, Array(2, Value(16))), Array(32, Value(1)), Array(6, Value(16)), Array(16, Value(1))]
OUTS = [Value(16), Value(1), Value(32), Array(2, Value(16)), Array(4, Value(16)), Array(8, Value(8)),
        Array(3, Value(16)), Value(8), Array(16, Value(1)), Array(32, Value(1))]
DIGEST = Array(8, Value(64))


def _words(t) -> int:
    return -(-sum(t.leaf_widths()) // 16)


def _bits(t) -> int:
    return sum(t.leaf_widths())


def _leaf(t) -> int:
    """A port's leaves' width: every port here has leaves of one width."""
    (w,) = set(t.leaf_widths())
    return w


def closed(fn) -> set[str]:
    """Template `fn`'s closed rows (the form's `Leaves.closed`): the parameters whose leaves are of one bit each and
    none of which it passes on."""
    on = ref.passed_on(fn)
    return {p for i, (p, t) in enumerate(fn.signature()[0]) if set(t.leaf_widths()) == {1} and i not in on}


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
    specs, shut = [], {}
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
        shut[tid] = closed(fn)
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
            writers = [(wt, f"o{x}") for wt, _, outs in specs[:j] for x, o in enumerate(outs)
                       if _bits(o) == 16 * row and (p in shut[tid] or _leaf(t) % _leaf(o) == 0)]
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


#: `Embed`'s parameters in a layout's order (the form's `layoutOf`): the key, the root, the domain id, the opened rows
READER = ("tok", "root", "domain", "row", "salt", "path")


def lowered(S: int, T: int, *, cross: bool = False, root_read: bool = True, forged: bool = False,
            key_wired: bool = True, rows: tuple[str, ...] = READER, verdict: bool = True, bits: str | None = None) -> Circuit:
    """A circuit in the shape of a served class of `S` slots and `T` positions (the module docstring), E = 2 words a
    hidden row: `Scan` (one unit a slot, its state a registered read of `scan-state`, the phase's own input), `Embed`
    (a table opening at the token), `Norm`, `Gemm` (per position and slot, two rows of the query weights then four of
    the mixing weights), `Attn<v>` (one template a position, its keys every earlier position's query rows by a cut of
    `v`), `Mix`, `Select` and `Feed` (the next position's token; its state its own earlier unit's). No wire crosses
    slots; with `cross`, one does: the first slot's mix at the second position reads the next slot's attention. Without
    `root_read`, no table read of the root; with `forged`, `Embed` is another Definition under the same name, its
    verdict a constant 1 that reads no port. Without `key_wired`, no wire fills `Embed`'s token. `rows` is the order of
    `Embed`'s parameters, a layout's by default; without `verdict`, `Embed` has the one output, the row, which its
    opening names as its verdict. `bits` makes the token 32 bits, a word on one side of its wires and 32 one-bit
    leaves on the other: `"reader"`, `Embed`'s key the bits and `Scan`'s and `Feed`'s token the word; `"writer"`,
    their token the bits and `Embed`'s key the word."""
    c = Circuit(f"lowered{S}x{T}", S, T)
    E, v16 = 2, Value(16)
    row, word = Array(E, v16), {None: (v16, v16), "reader": (Array(32, Value(1)), Value(32)),
                                "writer": (Value(32), Array(32, Value(1)))}[bits]
    key, tok = word

    def make(name: str, params: list, outs: list, used: dict[int, list[int]], constant: tuple[int, ...] = ()):
        """A template whose gates, one an output leaf, read between them the leaves `used` names (parameter: leaves);
        each output of `constant` is a constant instead, reading none."""
        n = sum(t.leaves for t in outs)
        plan, x = [], 0
        for o, t in enumerate(outs):
            plan.append([("op", w, [] if o in constant else [(i, k) for i, ks in used.items() for k in ks[x + j::n]])
                         for j, w in enumerate(t.leaf_widths())])
            x += t.leaves
        return template(f"Low{name}", params, outs, plan)

    scan = c.add(make("Scan", [("state", Array(4, v16))], [tok, row], {0: [0, 1, 2]}), [([S], ["slot"], 0)])
    ports = {"tok": (key, [0]), "root": (DIGEST, list(range(8))),
             "domain": (DIGEST, list(range(8))), "row": (row, [0, 1]),
             "salt": (Value(32), [0]), "path": (Array(4, v16), [0, 1, 2, 3])}
    embed = c.add(make("Embed", [(p, ports[p][0]) for p in rows], [row, Value(1)] if verdict else [row],
                       {i: ports[p][1] for i, p in enumerate(rows)}, constant=(1,) if forged else ()),
                  [([T, S], ["step", "slot"], 0)])
    norm = c.add(make("Norm", [("x", row), ("w", row)], [row], {0: [0, 1], 1: [0, 1]}), [([T, S], ["step", "slot"], 0)])
    gemm = c.add(make("Gemm", [("x", row), ("w", Array(4, v16))], [row], {0: [0, 1], 1: [0, 1, 2]}),
                 [e for t in range(T) for e in (([S, 2], ["slot", "-"], t), ([S, 4], ["slot", "-"], t))])
    attn = [c.add(make(f"Attn{v}", [("q", row), ("k", Array(v, row))], [row], {0: [0, 1], 1: list(range(0, 2 * v, 2))}),
                  [([2, S], ["-", "slot"], v - 1)]) for v in range(1, T + 1)]
    mix = c.add(make("Mix", [("a", Array(2, row)), ("g", Array(4, row)), ("res", row)], [v16, row],
                     {0: [0, 1, 2], 1: [0, 3, 5, 6], 2: [0, 1]}), [([T, S], ["step", "slot"], 0)])
    select = c.add(make("Select", [("x", v16)], [v16], {0: [0]}), [([T, S], ["step", "slot"], 0)])
    feed = c.add(make("Feed", [("sel", v16), ("st", row)], [tok, row], {0: [0], 1: [0]}),
                 [([T - 1, S], ["step", "slot"], 1)]) if T > 1 else None
    c.cuts = {mix: {"a": 2, "g": 4}, **{attn[v - 1]: {"k": v} for v in range(2, T + 1)}}
    G = 6 * S

    def wire(reader, port, writer, out, cells):
        """One wire per reader unit of `cells` (`(unit, first writer, writers' step, rows)`), merged into runs."""
        c.wires += IF.coalesce([IF.Wire(reader, port, writer, out, u, (n,), v, (d if n > 1 else 0,), (0, n))
                                for u, v, d, n in cells])

    if key_wired:
        wire(embed, "tok", scan, "o0", [(s, s, 0, 1) for s in range(S)])
    wire(norm, "x", embed, "o0", [(u, u, 0, 1) for u in range(T * S)])
    wire(gemm, "x", norm, "o0", [(t * G + i, t * S + (i // 2 if i < 2 * S else (i - 2 * S) // 4), 0, 1)
                                 for t in range(T) for i in range(G)])
    for v in range(1, T + 1):
        t = v - 1
        wire(attn[t], "q", gemm, "o0", [(h * S + s, t * G + 2 * s + h, 0, 1) for h in range(2) for s in range(S)])
        wire(attn[t], "k", gemm, "o0", [(h * S + s, 2 * s + h, G, v) for h in range(2) for s in range(S)])
        if cross and t == 1:
            wire(mix, "a", attn[t], "o0", [(t * S, 1, S, 2)])
        wire(mix, "a", attn[t], "o0", [(t * S + s, s, S, 2) for s in range(cross and t == 1, S)])
    wire(mix, "g", gemm, "o0", [(t * S + s, t * G + 2 * S + 4 * s, 1, 4) for t in range(T) for s in range(S)])
    wire(mix, "res", norm, "o0", [(u, u, 0, 1) for u in range(T * S)])
    wire(select, "x", mix, "o0", [(u, u, 0, 1) for u in range(T * S)])
    if feed is not None:
        wire(feed, "sel", select, "o0", [(u, u, 0, 1) for u in range((T - 1) * S)])
        wire(feed, "st", scan, "o1", [(s, s, 0, 1) for s in range(S)])
        wire(feed, "st", feed, "o1", [(u, u - S, 0, 1) for u in range(S, (T - 1) * S)])
        if key_wired:
            wire(embed, "tok", feed, "o0", [(u + S, u, 0, 1) for u in range((T - 1) * S)])
    c.params = {"scan-state": 4 * S, "weights/embed/table": 2 * 16, "weights/norm": 2, "weights/query": 2 * 4,
                "weights/mixing": 4 * 4}
    c.tables = [{"name": "table/weights/embed", "param": "weights/embed/table", "offset": 0, "words": 2, "rows": 16}]
    c.roots = [{"reader": embed, "port": of, "table": "table/weights/embed", "of": of, "at": 0, "units": T * S}
               for of in ("domain", "root") if root_read or of == "domain"]
    c.openings = [{"reader": embed, "table": "table/weights/embed", "key": "tok", "ports": ["row", "salt", "path"],
                   "verdict": "o1" if verdict else "o0", "at": 0, "units": T * S}]
    c.reads = IF.coalesce(
        [IF.Read(scan, "state", "scan-state", s, (1,), 4 * s, (0,)) for s in range(S)]
        + [IF.Read(norm, "w", "weights/norm", u, (1,), 0, (0,)) for u in range(T * S)]
        + [IF.Read(gemm, "w", "weights/query" if i < 2 * S else "weights/mixing", t * G + i,
                   (1,), 4 * (i % 2 if i < 2 * S else (i - 2 * S) % 4), (0,)) for t in range(T) for i in range(G)])
    c.order = c.ordered()
    return c


def passed_on(S: int = 2, how: str = "direct", gate: bool = True) -> Circuit:
    """A circuit whose wires `Spot.Wires.check` takes, word for word, and that lays out no circuit: a 16-bit leaf
    that reads sixteen 16-bit gates of another slot through one 16-bit row. `Wide`, sixteen units a slot, each one
    16-bit gate over a registered weight. `On`, a unit a slot: its parameter sixteen rows of sixteen one-bit leaves,
    row `j` wired from the slot's `j`-th `Wide`; its first output sixteen one-bit leaves, leaf `j` its row `j`'s leaf
    `j` itself, passed on with no gate. `Take`, a unit a slot: one gate over one 16-bit leaf, wired from the next
    slot's `On`. Each of `On`'s one-bit leaves lies under a 16-bit leaf of its writer, so through the one row that
    crosses slots, 16 bits, `Take`'s leaf reads sixteen gates of 16 bits.

    `how` is how `On` passes those leaves on (`PASSES`): itself, or through a call, a callee inside a callee, a
    batch member or a scan; with `"carry"` it passes one of them on, through a scan's carry, and with `"ys"` the
    other fifteen, as a scan's outputs, the rest of its output gates of its own; with `"beside"` it passes none, and its output
    is sixteen gates of its own, one over each of those leaves, returned through a callee whose argument holds the
    leaves beside them. With `"gates"` the output is those sixteen gates, called directly: `On`'s row is closed,
    and `Take`'s leaf reads sixteen one-bit gates of `On`. With `"unread"` the first output is those sixteen gates
    and a second output, which no wire reads, is the leaves themselves: `On` passes them on all the same. Without
    `gate`, `On` has that one output and no other, so no gate where it passes its leaves on."""
    c = Circuit(f"passed{S}", S)
    v16, bits = Value(16), Array(16, Value(1))
    one, diagonal = [[("op", 16, [(0, 0)])]], [17 * j for j in range(16)]
    wide = c.add(template("PassWide", [("w", v16)], [v16], one), [([S, 16], ["slot", "-"], 0)])
    own = [("op", 1, [(0, k)]) for k in diagonal]
    first = own if how in ("gates", "unread") else ("pass", 0, diagonal, how)
    second = (bits, ("pass", 0, diagonal, "direct")) if how == "unread" else (v16, [("op", 16, [])])
    on = c.add(template("PassOn", [("x", Array(16, bits))], [bits, second[0]] if gate else [bits],
                        [first, second[1]] if gate else [first]), [([S], ["slot"], 0)])
    take = c.add(template("PassTake", [("v", v16)], [v16], one), [([S], ["slot"], 0)])
    c.cuts = {on: {"x": 16}}
    c.wires = IF.coalesce([IF.Wire(on, "x", wide, "o0", s, (16,), 16 * s, (1,), (0, 16)) for s in range(S)]
                          + [IF.Wire(take, "v", on, "o0", s, (1,), (s + 1) % S, (0,), (0, 1)) for s in range(S)])
    c.params = {"weights/wide": 16 * S}
    c.reads = IF.coalesce([IF.Read(wide, "w", "weights/wide", u, (1,), u, (0,)) for u in range(16 * S)])
    c.order = c.ordered()
    return c


def two_widths(S: int = 2, bits: bool = False) -> Circuit:
    """A circuit with a wire from an output whose leaves have two widths: `Two`, a unit a slot, its first output a
    leaf of 8 bits and one of 24; `Whole`, a unit a slot, one gate over one 32-bit leaf wired from its slot's `Two`.
    With `bits`, `Whole`'s row is 32 one-bit leaves, the first read by its gate and none passed on: a closed row."""
    c = Circuit(f"widths{S}", S)
    v16 = Value(16)
    two = c.add(template("WidthsTwo", [("a", v16)], [Tuple(Value(8), Value(24)), v16],
                         [[("op", 8, [(0, 0)]), ("op", 24, [(0, 0)])], [("op", 16, [(0, 0)])]]), [([S], ["slot"], 0)])
    whole = c.add(template("WidthsWhole", [("v", Array(32, Value(1)) if bits else Value(32))], [v16],
                           [[("op", 16, [(0, 0)])]]), [([S], ["slot"], 0)])
    c.wires = IF.coalesce([IF.Wire(whole, "v", two, "o0", s, (1,), s, (0,), (0, 1)) for s in range(S)])
    c.order = c.ordered()
    return c


def short_output(S: int = 2, bits: bool = False) -> Circuit:
    """A circuit with a wire from an output that lacks bits of the row it fills: `Eight`, a unit a slot, its first
    output one 8-bit leaf; `Whole`, a unit a slot, one gate over one 16-bit leaf wired from its slot's `Eight`. The
    two rows have the same words, one, and 8 of the row's bits are no gate's. With `bits`, `Whole`'s row is 16
    one-bit leaves, the first read by its gate and none passed on: a closed row, which the output lacks bits of all
    the same."""
    c = Circuit(f"short{S}", S)
    v16 = Value(16)
    eight = c.add(template("ShortEight", [("a", v16)], [Value(8), v16],
                           [[("op", 8, [(0, 0)])], [("op", 16, [(0, 0)])]]), [([S], ["slot"], 0)])
    whole = c.add(template("ShortWhole", [("v", Array(16, Value(1)) if bits else v16)], [v16],
                           [[("op", 16, [(0, 0)])]]), [([S], ["slot"], 0)])
    c.wires = IF.coalesce([IF.Wire(whole, "v", eight, "o0", s, (1,), s, (0,), (0, 1)) for s in range(S)])
    c.order = c.ordered()
    return c


def odd_output(S: int = 2, words: int = 2) -> Circuit:
    """A circuit with an output of 31 bits, which is two words, its bits rounded up: `Odd`, a unit a slot, its first
    output one 31-bit leaf; `Whole`, a unit a slot, one gate over the first of its row's `16 * words` one-bit
    leaves and none passed on, a closed row. With two words the row is wired from its slot's `Odd`: the two rows
    have the same words, and the row holds a bit the output lacks. With one word nothing is wired, and `ODD_WIRE`
    is the wire a record then adds: the output's two words against the row's one."""
    c = Circuit(f"odd{S}x{words}", S)
    v16 = Value(16)
    odd = c.add(template("OddOut", [("a", v16)], [Value(31), v16],
                         [[("op", 31, [(0, 0)])], [("op", 16, [(0, 0)])]]), [([S], ["slot"], 0)])
    whole = c.add(template("OddWhole", [("v", Array(16 * words, Value(1)))], [v16],
                           [[("op", 16, [(0, 0)])]]), [([S], ["slot"], 0)])
    if words == 2:
        c.wires = IF.coalesce([IF.Wire(whole, "v", odd, "o0", s, (1,), s, (0,), (0, 1)) for s in range(S)])
    c.order = c.ordered()
    return c


#: the wire a record of `odd_output(words=1)` adds: its first `Whole`'s row from its first `Odd`'s 31-bit output
ODD_WIRE = {"reader": "OddWhole_v1", "port": "v", "writer": "OddOut_v1", "out": "o0", "at": 0, "radix": [1],
            "base": 0, "step": [0]}
WEIGHTS = ("weights/embed/table", "weights/norm", "weights/query", "weights/mixing")
EMBED, NORM, MIX, TABLE = "LowEmbed_v1", "LowNorm_v1", "LowMix_v1", "table/weights/embed"


def _put(key: str, x: dict):
    """The edit that declares `x` among a record's `key`: its `wires` or `ingress`, or one of `registered`'s lists."""
    def edit(d: dict) -> None:
        (d if key in ("wires", "ingress") else d["registered"]).setdefault(key, []).append(dict(x))
    return edit


def _port(port: str):
    """The edit that lists `port` once more among the ports of a record's opening."""
    return lambda d: d["registered"]["openings"][0]["ports"].append(port)


def _take(key: str, reader: str, port: str):
    """The edit that drops what a record's `key` declares to fill `reader`'s row `port`: of its `openings`, the port
    from each one's ports, once; else every declaration into that row."""
    def edit(d: dict) -> None:
        at = d if key == "wires" else d["registered"]
        if key == "openings":
            for o in at[key]:
                o["ports"].remove(port)
        else:
            at[key] = [x for x in at[key] if (x["reader"], x["port"]) != (reader, port)]
    return edit


def _read(reader: str, port: str, param: str, **more) -> dict:
    """A registered read of `param`'s first leaves into row `port` of `reader`'s first unit."""
    return {"reader": reader, "port": port, "param": param, "at": 0, "radix": [1], "base": 0, "step": [0], **more}


def _ingress(reader: str, port: str) -> dict:
    return {"reader": reader, "port": port, "at": 0, "units": 1, "step": 0}


#: a row of `lowered(2, 3)` declared twice, a case each pair of what can fill a row that its record has a row for:
#: the stage that refuses the record, the edit that declares the row a second time, and the edit that drops what
#: the record as built declares to fill it. Each second declaration is well formed: with the first dropped, the
#: stage takes it.
TWICE: dict[str, tuple] = {
    "a wire into a wired row":
        ("wires", _put("wires", {"reader": EMBED, "port": "tok", "writer": "LowScan_v1", "out": "o0", "at": 0,
                                 "radix": [1], "base": 0, "step": [0]}), _take("wires", EMBED, "tok")),
    "a registered read into a wired row":
        ("inputs", _put("reads", _read(NORM, "x", "weights/norm")), _take("wires", NORM, "x")),
    "a registered read into one row of a wired cut":
        ("inputs", _put("reads", _read(MIX, "a", "weights/norm", cut=[1, 1])), _take("wires", MIX, "a")),
    "a registered read into a row a registered read fills":
        ("inputs", _put("reads", _read(NORM, "w", "weights/norm")), _take("reads", NORM, "w")),
    "a registered read into a row a table read fills":
        ("inputs", _put("reads", _read(EMBED, "root", "weights/embed/table")), _take("roots", EMBED, "root")),
    "a registered read into an opened row":
        ("inputs", _put("reads", _read(EMBED, "row", "weights/embed/table")), _take("openings", EMBED, "row")),
    "a table read into a row a table read fills":
        ("inputs", _put("roots", {"reader": EMBED, "port": "root", "table": TABLE, "of": "domain", "at": 0, "units": 1}),
         _take("roots", EMBED, "root")),
    "a wired row among an opening's ports": ("inputs", _port("tok"), _take("wires", EMBED, "tok")),
    "a row a table read fills among an opening's ports": ("inputs", _port("root"), _take("roots", EMBED, "root")),
    "a row twice among an opening's ports": ("inputs", _port("path"), _take("openings", EMBED, "path")),
    "ingress into a wired row": ("inputs", _put("ingress", _ingress(NORM, "x")), _take("wires", NORM, "x")),
    "ingress into a row a registered read fills":
        ("inputs", _put("ingress", _ingress(NORM, "w")), _take("reads", NORM, "w")),
    "ingress into a row a table read fills":
        ("inputs", _put("ingress", _ingress(EMBED, "root")), _take("roots", EMBED, "root")),
    "ingress into an opened row": ("inputs", _put("ingress", _ingress(EMBED, "row")), _take("openings", EMBED, "row")),
}

#: a template with an output that has no bit: its name, its return's fields and its plan (`template`), its one
#: parameter a word. A return of no field is itself its one output, as `verity.commitments.windows.interface` lays it out.
BITLESS: dict[str, tuple] = {
    "a template that returns no field": ("NoField", [], []),
    "a return field of no bit": ("NoBit", [Value(16), Tuple()], [[("op", 16, [(0, 0)])], []]),
}

#: a template keyed as an input id (`verity.circuits.program.input_id`), of a width no input of the Program has: a
#: composite of one gate over its one parameter, a word. Its gates are computed, and a root node of it is no input.
KEYED: dict[str, str] = {
    "a template keyed as an input bound per run": "Input7",
    "a template keyed as an input bound at registration": "RegisteredInput7",
}


def shaped() -> list[dict]:
    """The cases on `lowered` circuits, at two slots and three positions and at four slots and 35, each size under one
    policy: `Γ` a block's gates, `α` and `(c, k, n)` exactly what a block holds with the weights registered at the
    start of the deployment, and one pin, `Embed`'s closure digest. That case is accepted: a block's fresh bits are
    its scan state's. With none registered, and with the run committing none, the weights' reads and the opening are
    fresh and the rate refuses (`nci`), the count `fresh`. Then at two slots and three positions: the query weights
    under another root than the run commits, the opening with no table read of its root, the opening's reader another
    Definition under `Embed`'s name whose verdict is a constant 1, and one wire from the next slot, each refused the
    same way; the order given beside the record, and a deployment whose first record is the run's own, each accepted.

    Then a pinned reader's layout, at two slots and three positions under the same policy and pin. `Embed`'s token left
    unwired is free and the rate refuses. With the token then listed among the opening's ports the record is refused
    (`reader-roles`), as it is with another row named the key, the root read in the domain id's row or the domain id in
    the root's, the two swapped, an opened row left out, the opened rows in another order, and another output named the
    verdict. `Embed` with its token its second row, and `Embed` with one output, are in no layout: pinned, the record
    is refused (`reader-layout`). Under no pin none of those records is refused by its roles: the opening's ports are
    fresh and the rate refuses.

    Then a wire's leaves, each case under the policy of exactly what its blocks hold, counted without the refusal
    where the case is refused: the token a word into a key of bits that only gates read, and bits into a key of one
    word (both accepted); `passed_on`, its leaves passed on each way of `PASSES`, by a Definition of no gate and to
    an output no wire reads (`leaf-widths`, the one that hands them to a callee that returns none among them and
    the two that pass on some of them alone, through a scan's carry and as a scan's outputs), and
    its output gates of its own (accepted); `two_widths` (`leaf-widths`), and into a closed row (accepted);
    `short_output`, into one leaf and into a closed row (`leaf-widths`, both); `odd_output`, its 31 bits two words:
    into a row of two words (`leaf-widths`), and a record that wires it into a row of one (`wires`). Then the
    accepted case with its Program's root edited (`own-rows`): `Norm`'s weights' row handed the input gates its `x`
    has, `Feed`'s selection not mapped, and `Select`'s row handed `Norm`'s output.

    Then the accepted case with its record's text another: one that lists its members and so states no wires
    (`listed`, refused `wires`), and the text of no wire and nothing else, which names no Program (`templates`).

    Then the accepted case with one row declared twice (`TWICE`): its record with one declaration more, into a row
    the record already declares, a case each pair of a wire, a registered read, a table read, an opening's port and
    ingress that the circuit has a row for. Two wires into one row are refused by the wires' own check (`wires`), and
    every other pair with the declarations (`inputs`).

    Then the accepted case with its Program and record those of one template with an output that has no bit
    (`BITLESS`), a unit a slot: a template that returns no field, and one whose return has a field of no bit beside a
    field of 16. Each is refused with its templates (`templates`).

    Then a template keyed as an input id (`KEYED`), a composite of one gate over a word: the accepted case with its
    Program and record those of that one template, a unit a slot (`_alone`), and the accepted case with a unit of
    it more at its Program's root, which its record does not list (`_unlisted`). Each is refused with its templates
    (`templates`): a root node of a composite is no input, whatever its key reads as."""
    out = []
    first = None
    for n, (S, T) in enumerate([(2, 3), (4, 35)]):
        policy: dict = {}
        pins = [ref.closure_digest(json.loads(lowered(S, T).program()[0]), EMBED)]

        def one(what: str, c: Circuit, seed: int, *, held=WEIGHTS, committed=WEIGHTS, moved=(), beside=False,
                before=2, policy=policy, pins=pins, n=n, edit=None, pinned=None, broken=None, refused=None,
                reroot=None, alone=None, unlisted=None) -> None:
            """The case on `c`; with `edit`, its record then edited; with `reroot`, its Program's root then edited;
            with `alone`, its Program and record then those of a circuit of that one template (`_alone`); with
            `unlisted`, its Program's root then with a unit of that template more, which its record does not list
            (`_unlisted`); with `pinned`, its pins then those; `broken`, the stage that refuses it; `refused`, that
            stage where it refuses the circuit as built."""
            rng = random.Random(50_000 + 100 * n + seed)
            roots = {p: _hex(rng) for p in c.params}
            order = None
            if beside:
                order, c.order = c.order, None
            made = finish(rng, c, {p: (_hex(rng) if p in moved else roots[p], c.params[p] // 2, 2)
                                   for p in held if p in c.params},
                          {p: roots[p] for p in (*committed, "scan-state") if p in c.params}, pins, before=before,
                          order=order, policy=policy, refused=refused)
            if edit is not None:
                _rewrite(made, edit)
            if reroot is not None:
                _reroot(made, reroot)
            if alone is not None:
                _alone(made["case"], c, alone)
            if unlisted is not None:
                _unlisted(made["case"], c, unlisted)
            if broken in EARLY:
                made["case"]["spot"] = None
            if pinned is not None:
                made["case"]["instance"]["pins"] = pinned
            out.append(made | {"shape": what, "broken": broken or refused})

        first = first or one
        one(f"{S}x{T} registered", lowered(S, T), 0)
        one(f"{S}x{T} none registered", lowered(S, T), 1, held=())
        one(f"{S}x{T} none committed", lowered(S, T), 2, committed=())
        if n == 0:
            one("2x3 another root", lowered(S, T), 3, moved=("weights/query",))
            one("2x3 no root read", lowered(S, T, root_read=False), 4)
            one("2x3 a constant verdict", lowered(S, T, forged=True), 8)
            one("2x3 one wire from the next slot", lowered(S, T, cross=True), 5)
            one("2x3 the order beside the record", lowered(S, T), 6, beside=True)
            one("2x3 the run's record first", lowered(S, T), 7, before=0)

    def opening(**fields):
        return lambda d: d["registered"]["openings"][0].update(fields)

    def table_reads(**row):
        """Each table read moved to the row `row` gives what it reads (`root`, `domain`), and dropped where it gives none."""
        return lambda d: d["registered"].update(roots=[{**r, "port": row[r["of"]]} for r in d["registered"]["roots"]
                                                       if r["of"] in row])

    roles, among = "reader-roles", ["tok", "row", "salt", "path"]
    first("2x3 the key not wired", lowered(2, 3, key_wired=False), 9)
    first("2x3 the key among the ports", lowered(2, 3, key_wired=False), 10, edit=opening(ports=among), broken=roles)
    first("2x3 the key among the ports, no pin", lowered(2, 3, key_wired=False), 11, pins=[], edit=opening(ports=among))
    first("2x3 another row the key", lowered(2, 3), 12, edit=opening(key="row"), broken=roles)
    first("2x3 the root read in the domain id's row", lowered(2, 3), 13, edit=table_reads(root="domain"), broken=roles)
    first("2x3 the domain id read in the root's row", lowered(2, 3), 14, edit=table_reads(domain="root"), broken=roles)
    first("2x3 the root's and the domain id's rows swapped", lowered(2, 3), 15,
          edit=table_reads(root="domain", domain="root"), broken=roles)
    first("2x3 an opened row left out", lowered(2, 3), 16, edit=opening(ports=["row", "salt"]), broken=roles)
    first("2x3 the opened rows in another order", lowered(2, 3), 17, edit=opening(ports=["salt", "row", "path"]),
          broken=roles)
    first("2x3 another output the verdict", lowered(2, 3), 18, edit=opening(verdict="o0"), broken=roles)
    for seed, what, shape in ((19, "the key the second row", {"rows": ("root", "tok", "domain", "row", "salt", "path")}),
                              (21, "a reader of one output", {"verdict": False})):
        pin = [ref.closure_digest(json.loads(lowered(2, 3, **shape).program()[0]), EMBED)]
        first(f"2x3 {what}, pinned", lowered(2, 3, **shape), seed, pins=[], pinned=pin, broken="reader-layout")
        first(f"2x3 {what}, no pin", lowered(2, 3, **shape), seed + 1, pins=[])

    whole, units = "leaf-widths", lambda nodes: [n for n in nodes if n["args"]]
    for seed, side in ((23, "reader"), (24, "writer")):
        pin = [ref.closure_digest(json.loads(lowered(2, 3, bits=side).program()[0]), EMBED)]
        first(f"2x3 the key's {side} bits, the other side a word", lowered(2, 3, bits=side), seed, pins=pin, policy={})
    for seed, what, shape, refused in (
            (25, "a leaf that reads sixteen gates through one row", {}, whole),
            (30, "sixteen gates through one row, passed on by a Definition of no gate", {"gate": False}, whole),
            (31, "sixteen gates through one row, passed on through a call", {"how": "call"}, whole),
            (32, "sixteen gates through one row, passed on through a batch member", {"how": "batch"}, whole),
            (33, "sixteen gates through one row, passed on through a scan", {"how": "scan"}, whole),
            (39, "sixteen gates through one row, passed on through a callee inside a callee", {"how": "nested"}, whole),
            (40, "a gate through one row, passed on through a scan's carry", {"how": "carry"}, whole),
            (44, "fifteen gates through one row, passed on as a scan's outputs", {"how": "ys"}, whole),
            (41, "sixteen gates through one row, passed on to an output no wire reads", {"how": "unread"}, whole),
            (34, "a row a callee is handed and returns no leaf of", {"how": "beside"}, whole),
            (35, "a closed row: sixteen gates read, none passed on", {"how": "gates"}, None)):
        first(what, passed_on(**shape), seed, held=("weights/wide",), committed=("weights/wide",), pins=[], policy={},
              refused=refused)
    first("a wire from an output of two widths", two_widths(), 26, held=(), committed=(), pins=[], policy={},
          refused=whole)
    first("an output of two widths into a closed row", two_widths(bits=True), 36, held=(), committed=(), pins=[],
          policy={})
    first("an output that lacks bits of its row", short_output(), 37, held=(), committed=(), pins=[], policy={},
          refused=whole)
    first("an output that lacks bits of a closed row", short_output(bits=True), 38, held=(), committed=(), pins=[],
          policy={}, refused=whole)
    first("an output of 31 bits into a row of two words", odd_output(), 42, held=(), committed=(), pins=[], policy={},
          refused=whole)
    first("an output of 31 bits wired into a row of one word", odd_output(words=1), 43, held=(), committed=(), pins=[],
          policy={}, edit=lambda d: d["wires"].append(dict(ODD_WIRE)), broken="wires")

    def twice(nodes: list, _desc: dict) -> None:
        """`Norm`'s weights' row handed the input gates its `x` has."""
        norm = units(nodes)[2]
        norm["args"][1]["refs"] = copy.deepcopy(norm["args"][0]["refs"])

    def shared(nodes: list, desc: dict) -> None:
        """`Feed`'s selection not mapped: every unit handed the first unit's input gates."""
        _share(units(nodes)[-1], 0, desc["types"].index(["Value", 16]))

    def computed(nodes: list, _desc: dict) -> None:
        """`Select`'s row handed `Norm`'s output, computed gates, in place of input gates."""
        select = units(nodes)[-2]
        count = select["args"][0]["refs"]["affine"][3]
        select["args"][0]["refs"] = {"affine": ["n", nodes.index(units(nodes)[2]), 0, count]}

    first("2x3 one input run handed to two rows", lowered(2, 3), 27, reroot=twice, broken="own-rows")
    first("2x3 a row every unit shares", lowered(2, 3), 28, reroot=shared, broken="own-rows")
    first("2x3 a row of computed gates", lowered(2, 3), 29, reroot=computed, broken="own-rows")
    c = lowered(2, 3)
    first("2x3 a record that lists its members", c, 45, edit=listed(c), broken="wires")
    first("2x3 a text of no wire that names no Program", lowered(2, 3), 46,
          edit=lambda d: (d.clear(), d.update(wires=[])), broken="templates")
    for k, (what, (stage, declare, _)) in enumerate(TWICE.items()):
        first(f"2x3 {what}", lowered(2, 3), 47 + k, edit=declare, broken=stage)
    for k, (what, (name, outs, plan)) in enumerate(BITLESS.items()):
        first(f"2x3 {what}", lowered(2, 3), 47 + len(TWICE) + k, broken="templates",
              alone=template(name, [("p", Value(16))], outs, plan))
    for k, (what, name) in enumerate(KEYED.items()):
        keyed = template(name, [("p", Value(16))], [Value(16)], [[("op", 16, [(0, 0)])]])
        seed = 47 + len(TWICE) + len(BITLESS) + 2 * k
        first(f"2x3 {what}, alone", lowered(2, 3), seed, broken="templates", alone=keyed)
        first(f"2x3 {what}, unlisted", lowered(2, 3), seed + 1, broken="templates", unlisted=keyed)
    return out


def listed(c: Circuit):
    """The edit that makes `c`'s record one that lists its members, as `verity.commitments.windows.interface` writes one
    (`_listed`): a member a template, in the templates' order, beside the registered reads, with no wires, cuts,
    order, ingress or blocks."""
    def edit(d: dict) -> None:
        for k in ("wires", "cuts", "order", "ingress", "blocks"):
            d.pop(k, None)
        d["members"], base = [], 0
        for fn, n in c.templates:
            lay = IF.layout(fn)
            d["members"].append(IF._member_doc(IF.CS.Member(codec.descriptor_id(fn), lay.rows, lay.outs, n, base)))
            base += n
    return edit


# Cases


def finish(rng, c: Circuit, held: dict[str, tuple[str, int, int]], roots: dict[str, str], pins: list[str], *,
           before: int = 2, after: int = 0, order=None, policy: dict | None = None, refused: str | None = None) -> dict:
    """The case on circuit `c`: its instance (the Program, the record, `Γ`, `α` and the pins), the trusted input
    (`trusted`), the wires as SPOT reads them, up to six of its units' input rows and the count's limits; with what it
    was made from. The limits are at, just under or just over what the circuit's blocks hold; or `policy`'s, which an
    empty `policy` is first filled with: a table of every primitive and exactly what the blocks hold; a primitive a
    later circuit computes that the policy's table lacks costs nothing. The reference takes the circuit as built;
    or `refused` names the stage that refuses it, and what the blocks hold is what the reference counts without that
    stage."""
    text, digest = c.program()
    record = IF.circuit_bytes(c.doc(digest)).decode()
    inside = [all(s == t for s, t in _slots(c, w)) for w in c.wires]
    g = grids(c)
    for w, want in zip(c.wires, inside, strict=True):
        if searched(c.slots, g, w) is not want:
            raise AssertionError(f"{c.name}: the search doesn't decide wire {w.doc()} within its fuel")
    reference.desc_program(text)
    prims = sorted({p for fn, _ in c.templates for p in ref.template_gates(fn)})
    if policy is None:
        table = {p: rng.choice([0, 1, 2, rng.randint(0, 40)]) for p in prims}
        if rng.random() < 0.1:
            del table[rng.choice(prims)]
    else:
        table = dict.fromkeys(prims, 0) | policy.get("table", {p: 3 + 2 * k for k, p in enumerate(prims)})
    inst = {"program": text, "record": record, "gamma": BIG, "alpha": [BIG, 1], "pins": pins,
            **({"order": order} if order is not None else {})}
    tr = trusted(rng, digest, record, held, roots, before, after)
    probe = ref.check_detail(inst, tr)
    assert probe.get("error") == refused, (c.name, probe)
    with _without(refused):
        probe = ref.check_detail(inst, tr)
        bools = ref.count(inst, tr, {"table": table, "c": 0, "k": 0, "n": 0}).get("bool_gates", 0)
    assert "error" not in probe, (c.name, probe)

    def near(x: int) -> int:
        return x if policy is not None else max(0, x + rng.choices([0, -1, 1, rng.randint(-3, 3)], [5, 2, 2, 1])[0])
    den = rng.randint(1, 5) if policy is None else probe["gates"]
    limits = {"gamma": near(probe["gates"]), "alpha": [near(-(-probe["fresh"] * den // max(probe["gates"], 1))), den],
              "count": {"c": near(bools), "k": near(probe["fresh"]), "n": near(probe["blocks"])}}
    if policy is not None:
        limits = policy if policy else policy.update(limits, table=table) or policy
    inst.update(gamma=limits["gamma"], alpha=limits["alpha"])
    rows = [(codec.descriptor_id(fn), p, u, r) for fn, n in c.templates for p, _ in fn.signature()[0]
            for u in range(n) for r in range(c.cuts.get(codec.descriptor_id(fn), {}).get(p, 1))]
    return {"case": {"instance": inst, "trusted": tr, "spot": spot_of(c, json.loads(record), order),
                     "rows": [list(r) for r in rng.sample(rows, min(6, len(rows)))],
                     "count": {"table": table, **{k: limits["count"][k] for k in ("c", "k", "n")}}},
            "circuit": c, "inside": inside,
            "theta": sorted(p for p, (root, n, w) in held.items() if roots.get(p) == root and n * w == c.params[p])}


@contextlib.contextmanager
def _without(stage: str | None):
    """The reference without the stage `stage` (`leaf-widths`, or None: the reference whole)."""
    if stage is None:
        yield
        return
    assert stage == "leaf-widths", stage
    whole, ref.leaf_widths = ref.leaf_widths, lambda *_a: None
    try:
        yield
    finally:
        ref.leaf_widths = whole


def spot_of(c: Circuit, doc: dict, order=None) -> dict | None:
    """The record `doc`'s wires as SPOT's `wires` piece reads them (`Spot.Wires.Wiring.ofJson`), each template's rows
    and outputs Python's own layout of it (`verity.commitments.windows.interface.layout`, what `check_wires` reads): the order
    the record's, else `order` where one is given beside it. None where that layout has no row for a port: its leaves
    of two widths, or an output of 16 bits or more that is not whole 16-bit words."""
    if any(len(ty.width_set()) != 1 for fn, _ in c.templates
           for ty in (*(ty for _, ty in fn.signature()[0]), *ref.fields_of(fn.signature()[1]))):
        return None
    if any(_bits(ty) > 16 and _bits(ty) % 16 for fn, _ in c.templates for ty in ref.fields_of(fn.signature()[1])):
        return None
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


@functools.cache
def _turns() -> dict[int, int]:
    """The seeds below `COUNT` whose case is broken, two in five, each with its turn among them."""
    return {seed: k for k, seed in enumerate(s for s in range(COUNT) if random.Random(s).random() < 0.4)}


def case(seed: int) -> dict:
    """Case `seed`: below `COUNT` a random circuit with a random registration log, two in five of them broken, each
    in the way `WAYS` gives its turn among the broken, so every way comes up; on the first of the seed's circuits
    that has what that way breaks. Then `shaped()`'s."""
    if seed >= COUNT:
        return shaped()[seed - COUNT]
    turn = _turns().get(seed)
    stage, how = (None, None) if turn is None else WAYS[turn % len(WAYS)]
    for attempt in range(TRIES):
        rng = random.Random(TRIES * seed + attempt)
        c = random_circuit(rng, seed)
        roots = {p: _hex(rng) for p in c.params if rng.random() < 0.9}
        held = {}
        for p, n in c.params.items():
            kind = rng.choice(["held", "held", "held", "none", "root", "shape"])
            if kind == "none":
                continue
            w = rng.choice([d for d in range(1, n + 1) if n % d == 0])
            held[p] = (roots.get(p, _hex(rng)) if kind != "root" else _hex(rng), n // w + (kind == "shape"), w)
        desc = json.loads(c.program()[0])
        pins = ([ref.closure_digest(desc, t) for t in c.units if rng.random() < 0.7]
                + [_hex(rng) for _ in range(rng.randint(0, 2))])
        rng.shuffle(pins)
        made = finish(rng, c, held, roots, pins, before=rng.choice([0, 1, 2, 2, 3]), after=rng.choice([0, 0, 1]))
        if how is None or _break(rng, made, how):
            break
    else:
        raise AssertionError(f"case {seed}: none of its {TRIES} circuits has what {how!r} breaks")
    if rng.random() < 0.04:
        limits = made["case"]["count"]
        rng.choice([lambda: limits.pop(rng.choice(ref.LIMITS)), lambda: limits.update(table=[]),
                    lambda: limits.update(c=rng.choice([-1, "3", 1.5, None])), lambda: limits.update(n=[1]),
                    lambda: limits["table"].update(RecNope_v1=rng.choice([-1, 1.5, "2"]))])()
    if stage in EARLY:
        made["case"]["spot"] = None
    return made | {"shape": "random", "broken": stage, "how": how}


def _rewrite(made: dict, edit) -> None:
    """The record edited by `edit` (on its object), with the run's own record still committing its bytes."""
    case = made["case"]
    doc = json.loads(case["instance"]["record"])
    edit(doc)
    _commit(case, IF.circuit_bytes(doc).decode())


def _reroot(made: dict, edit) -> None:
    """The Program's root edited by `edit` (on its body's nodes and the descriptor, as it states them), the record
    naming the Program it then is and the run's own record committing both."""
    case = made["case"]
    desc = json.loads(case["instance"]["program"])
    edit(desc["definitions"][desc["root"]]["body"]["nodes"], desc)
    case["instance"]["program"] = codec.canonical_json(desc).decode()
    digest = codec.program_digest(desc)
    _commit(case, IF.circuit_bytes({**json.loads(case["instance"]["record"]), "program": digest}).decode(), digest)


def _clear(d: dict, t: str, p: str) -> None:
    """Nothing fills template `t`'s row `p`, in the record `d`."""
    d["wires"] = [w for w in d["wires"] if (w["reader"], w["port"]) != (t, p)]
    if "registered" in d:
        d["registered"]["reads"] = [r for r in d["registered"]["reads"] if (r["reader"], r["port"]) != (t, p)]
    if "ingress" in d:
        d["ingress"] = [g for g in d["ingress"] if (g["reader"], g["port"]) != (t, p)]


def _leaves(t: list) -> list[int]:
    """A descriptor's type's leaf widths."""
    if t[0] == "Value":
        return [t[1]]
    return _leaves(t[2]) * t[1] if t[0] == "Array" else [w for f in t[1] for w in _leaves(f)]


def _share(node: dict, j: int, member: int) -> None:
    """A batch's argument `j` no longer mapped: every member handed the first member's leaves, of the type at
    `member` of the descriptor's types."""
    at, count = node["args"][j]["refs"]["affine"][1::2]
    node["axes"][j] = None
    node["args"][j].update(type=member, refs={"affine": ["n", at, 0, count // node["n"]]})


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
    elif how == "pins-type":
        inst["pins"] = rng.choice([{}, 3, "pins", None])
    elif how == "gamma":
        inst["gamma"] = rng.choice([-1, 1.5, "2", None, True])
    elif how == "alpha":
        inst["alpha"] = rng.choice([[1, 0], [1], 3, [1, -2], [1.5, 2], "1/2", [1, 2, 3]])
    elif how == "pin-value":
        inst["pins"].append(rng.choice([3, None, "abc", _hex(rng).upper().replace("0", "A"), _hex(rng)[:-1], [_hex(rng)]]))
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
        _alone(case, c, template(f"Half{c.name}", [("p", Value(rng.choice([8, 24])))], [Value(16)],
                                 [[("op", 16, [(0, 0)])]]))
    elif how in BROKEN["own-rows"]:
        desc = json.loads(inst["program"])
        nodes, types = desc["definitions"][desc["root"]]["body"]["nodes"], desc["types"]
        #: every argument of every unit's node, an input node's leaves: (node, argument, input node, leaves)
        runs = [(k, j, *a["refs"]["affine"][1::2]) for k, n in enumerate(nodes) for j, a in enumerate(n["args"])]
        if how == "row-twice":
            found = [(a, b) for a in runs for b in runs if a[:2] != b[:2] and nodes[a[2]]["fn"] == nodes[b[2]]["fn"]
                     and b[3] <= a[3]]
        elif how == "row-broadcast":
            found = [(r, types.index(types[nodes[r[0]]["args"][r[1]]["type"]][2])) for r in runs
                     if types[nodes[r[0]]["args"][r[1]]["type"]][2] in types]
        elif how == "row-computed":
            found = [(r, k) for r in runs for k in range(r[0])
                     if nodes[k]["args"] and _leaves(types[nodes[k]["out"]])[:r[3]] == _leaves(types[nodes[r[2]]["out"]])]
        else:
            found = [r for r in runs if how != "row-aliased" or r[3] >= 2]
        if not found:
            return False
        pick = rng.choice(found)

        def edit(nodes: list, _desc: dict) -> None:
            if how == "row-twice":
                a, b = pick
                nodes[b[0]]["args"][b[1]]["refs"] = {"affine": ["n", a[2], 0, b[3]]}
                return
            (k, j, at, count), other = pick if how in ("row-broadcast", "row-computed") else (pick, None)
            arg = nodes[k]["args"][j]
            if how == "row-aliased":
                half = rng.randint(1, count - 1)
                arg["refs"] = {"concat": [{"affine": ["n", at, 0, half]}, {"affine": ["n", at, 0, count - half]}]}
            elif how == "row-broadcast":
                _share(nodes[k], j, other)
            elif how == "row-computed":
                arg["refs"] = {"affine": ["n", other, 0, count]}
            elif how == "row-explicit":
                arg["refs"] = {"explicit": [["n", at, leaf] for leaf in range(count)]}
            else:
                arg["refs"] = {"strided": ["n", at, 0, count, 0, 1, count]}
        _reroot(made, edit)
    elif how in ("leaf-wider", "leaf-short"):
        sigs = {codec.descriptor_id(fn): (fn.signature()[0], ref.fields_of(fn.signature()[1])) for fn, _ in c.templates}
        #: the rows of a table's reader: the new wire is the one thing that fills a row of another
        tabled = ({(x["reader"], x["port"]) for x in reg.get("roots", [])}
                  | {(o["reader"], p) for o in reg.get("openings", []) for p in (o["key"], *o["ports"])})
        shut = {codec.descriptor_id(fn): closed(fn) for fn, _ in c.templates}

        def breaks(t: str, p: str, ty, o) -> bool:
            """Whether a wire from an output of type `o` into row `p` of `t`, of type `ty` and the same words, is
            refused this way: the output lacks bits of the row; or its leaves' width does not divide the row's,
            and the row is not closed."""
            if how == "leaf-short":
                return _bits(o) != 16 * _words(o)
            return p not in shut[t] and _leaf(ty) % _leaf(o) != 0

        found = [{"reader": t, "port": p, "writer": wt, "out": f"o{x}", "at": 0, "radix": [1], "base": 0, "step": [0]}
                 for j, t in enumerate(ids) for p, ty in sigs[t][0]
                 if (t, p) not in tabled and p not in c.cuts.get(t, {})
                 for wt in ids[:j] for x, o in enumerate(sigs[wt][1]) if _words(o) == _words(ty) and breaks(t, p, ty, o)]
        if not found:
            return False
        #: a row the closed-row rule turns on, where the circuit has one: for a wider leaf, a row of bits its
        #: template passes on; for an output short of its row, a closed row, which is refused all the same
        turns = [w for w in found if (w["port"] in shut[w["reader"]] if how == "leaf-short" else
                                      _leaf(dict(sigs[w["reader"]][0])[w["port"]]) == 1)]
        w = rng.choice(turns or found)
        _rewrite(made, lambda d: (_clear(d, w["reader"], w["port"]), d["wires"].append(w)))
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
    elif how in (*BROKEN["reader-layout"], *BROKEN["reader-roles"]):
        desc = json.loads(inst["program"])
        lays = {t: ref.layout(ref.Template(t, fn, n)) for (fn, n), t in zip(c.templates, ids, strict=True)}
        pinned = [t for t in ids if ref.closure_digest(desc, t) in inst["pins"]]
        opened = [k for k, o in enumerate(reg.get("openings", [])) if o["reader"] in pinned]
        read = {of: [t for t in pinned if any(r["reader"] == t and r["of"] == of for r in reg.get("roots", []))]
                for of in ("root", "domain")}
        bare = [t for t in pinned if lays[t] is None]

        def move(d: dict, t: str, row: dict) -> None:
            """Template `t`'s table reads, each in the row `row` gives what it reads, and dropped where it gives none."""
            d["registered"]["roots"] = [{**r, "port": row[r["of"]]} if r["reader"] == t else r
                                        for r in d["registered"]["roots"] if r["reader"] != t or r["of"] in row]

        def edit(d: dict) -> None:
            r = d["registered"]
            if how == "reader-other":
                t = rng.choice(bare)
                p = c.templates[ids.index(t)][0].signature()[0][0][0]
                _clear(d, t, p)
                r["openings"].append({"reader": t, "table": rng.choice(r["tables"])["name"], "key": p, "ports": [p],
                                      "verdict": "o0", "at": 0, "units": units[t]})
            elif how in ("root-row", "domain-row", "rows-swapped"):
                of = {"root-row": "root", "domain-row": "domain"}.get(how)
                t = rng.choice(read[of] if of else [t for t in read["root"] if t in read["domain"]])
                lay = lays[t]
                move(d, t, {"root": lay["domain"]} if of == "root" else {"domain": lay["root"]} if of else
                     {"root": lay["domain"], "domain": lay["root"]})
            else:
                o = r["openings"][rng.choice(opened)]
                lay = lays[o["reader"]]
                if how == "key-port":
                    _clear(d, o["reader"], o["key"])
                    o["ports"] = [o["key"], *o["ports"]]
                elif how == "key-row":
                    o["key"] = rng.choice([lay["root"], lay["domain"], *lay["ports"]])
                elif how == "port-out":
                    o["ports"].remove(rng.choice(o["ports"]))
                elif how == "ports-order":
                    o["ports"] = rng.choice([o["ports"][::-1], o["ports"][1:] + o["ports"][:1]])
                else:
                    o["verdict"] = "o0"
        if (how == "reader-other" and not (bare and reg.get("tables"))
                or how in ("root-row", "domain-row") and not read[how[:-4]]
                or how == "rows-swapped" and not set(read["root"]) & set(read["domain"])
                or how in ("key-port", "key-row", "port-out", "ports-order", "verdict-out") and not opened):
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


def _alone(case: dict, c: Circuit, fn) -> None:
    """The case's Program and record replaced by those of a circuit of the one template `fn`, a unit a slot of `c`'s,
    with no wire and no cut; the record's other fields kept."""
    doc = json.loads(case["instance"]["record"])
    text, digest = Circuit.program(_only(c, fn))
    case["instance"]["program"] = text
    _commit(case, IF.circuit_bytes({**doc, "program": digest, "templates": [[codec.descriptor_id(fn), c.slots]],
                                     "wires": [], "cuts": {}}).decode(), digest)


def _unlisted(case: dict, c: Circuit, fn) -> None:
    """The case's Program replaced by `c`'s with one root node more, a unit of the template `fn` over an input of
    its own, which the record does not list; the record naming the Program it then is, its other fields kept."""
    desc = codec.encode_program(mixed_population_program(
        [*c.templates, (fn, 1)], values=[(v, c.params[v], 16) for v in c.values]))
    case["instance"]["program"] = codec.canonical_json(desc).decode()
    digest = codec.program_digest(desc)
    _commit(case, IF.circuit_bytes({**json.loads(case["instance"]["record"]), "program": digest}).decode(), digest)


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
    return COUNT + 50 + len(TWICE) + len(BITLESS) + 2 * len(KEYED)


def results(case: dict) -> dict:
    """The reference's results on a case: the whole verdict, what is published, its rows' classes, what the form
    exports of them (each row's slot and why it is held), the count, and each template's closed rows."""
    inst, tr = case["instance"], case["trusted"]
    return {"detail": ref.check_detail(inst, tr), "public": ref.check(inst, tr), "kinds": ref.kinds(inst, tr, case["rows"]),
            "exported": ref.exported(inst, tr, case["rows"]), "count": ref.count(inst, tr, case["count"]),
            "closed": ref.closed(inst, tr)}


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
    ap.add_argument("verity", type=Path, help="the checkout whose verity.circuits reads the programs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=None, help="the first N cases (default: all, `total()`)")
    a = ap.parse_args()
    here = Path(verity.circuits.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.circuits comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
