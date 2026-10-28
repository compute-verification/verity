"""Declarations whose references name leaves of other widths than their types give them (`E_REINTERPRET`,
`verity/ml/circuits/PROTOCOL.md` §4.4), and their well-typed neighbours, for the NCI difftests.

The Python codec refuses such a Program as it decodes it (`defs.Builder.finish`), which C-Flock's decoder doesn't look
for, so `NciCheck` checks it after decoding (`NciCheck.Widths`). The descriptors here are written as JSON, since the
Builder refuses to build them. `cases()` holds, as `NciCheck.Desc` reads an instance (θ and the owners per root node):

* `shapes()`: the five shapes on which a draft of the liveness test gave `NciCheck` mistyped Programs
  (`swap`, `exit_xs`, `carry_read` with and without `xs`, `xs_part`, each with inputs of 8, 16, 32 and 64 bits a carry or a
  batch reads as 8), at a small `n` and at one past anything walked (2^40 to 2^52), and `exit-xs` at `n = 2`;
* `pairs()`: an `Array<N, (V8, V16)>` read by a strided part (leaf 0 or leaf 1 of every pair), by an affine one shifted by
  one leaf (`(V16, V8)` pairs, and `(V8, V16)` ones), as `(V8, V16, V8, V16)` quads (whole, and shifted by one leaf), a
  parameter read at another width, an explicit item, a body's return, and a later node read at another width (refused
  `not-topological`, which is not this check's), each with its well-typed neighbour, at `N = 4` and at `N = 2^20` (the
  well-typed ones at `2^14`, `PAIR_SIZES`);
* `tampered()`: `TAMPERED` of `desc_difftest_vectors`'s random declarations with one reference moved to another leaf
  source, of the same width or another, kept within its leaves.
"""
from __future__ import annotations

import json
import random

SCHEMA = "verity-ir/descriptor/v1"


def V(w):
    return ["Value", w]


def A(n, e):
    return ["Array", n, e]


def Tup(*fs):
    return ["Tuple", list(fs)]


class Program:
    """A descriptor being written: its definitions and its callees and types tables."""

    def __init__(self):
        self.defs: dict[str, dict] = {}
        self.callees: list[str] = []
        self.types: list = []
        self.gates: dict[str, int] = {}

    def ty(self, t) -> int:
        if t not in self.types:
            self.types.append(t)
        return self.types.index(t)

    def fn(self, key: str) -> int:
        if key not in self.callees:
            self.callees.append(key)
        return self.callees.index(key)

    def prim(self, key: str, params: list, ret) -> str:
        self.defs[key] = {"form": "primitive", "id": key, "params": [[f"a{i}", p] for i, p in enumerate(params)],
                          "ret": ret}
        self.gates[key] = 1
        return key

    def composite(self, key: str, body: "Body", ret, refs) -> str:
        self.defs[key] = {"form": "composite", "id": key, "bindings": {}, "body": {
            "params": [[f"x{i}", t] for i, t in enumerate(body.params)], "ret_type": ret, "nodes": body.nodes,
            "ret": refs}}
        self.gates[key] = body.gates
        return key

    def text(self) -> str:
        d = {"annotations": {}, "callees": self.callees, "definitions": self.defs, "program_fn": "R_v1", "root": "R_v1",
             "schema": SCHEMA, "types": self.types}
        return json.dumps(d, separators=(",", ":"), sort_keys=True)


class Body:
    def __init__(self, P: Program, params: list):
        self.P, self.params, self.nodes, self.gates, self.inputs = P, params, [], 0, []

    def arg(self, refs, t) -> dict:
        return {"refs": refs, "type": self.P.ty(t)}

    def node(self, form: str, key: str, args: list, out, n: int = 1, **extra) -> int:
        nd = {"args": args, "fn": self.P.fn(key), "form": form, "out": self.P.ty(out), **extra}
        if form in ("batch", "scan"):
            nd["n"] = n
        self.nodes.append(nd)
        self.gates += (n if form in ("batch", "scan") else 1) * self.P.gates[key]
        return len(self.nodes) - 1

    def input(self, w: int) -> int:
        k = self.node("primitive", self.P.prim(f"RegisteredInput{w}_v1", [], V(w)), [], V(w))
        self.inputs.append(k)
        return k


def aff(s, i, b, c):
    return {"affine": [s, i, b, c]}


def finish(P: Program, root: Body, ret_node: int) -> dict:
    """The instance: every other root node in IU 0, θ empty."""
    P.composite("R_v1", root, V(8), aff("n", ret_node, 0, 1))
    owner = ["input" if k in root.inputs else 0 for k in range(len(root.nodes))]
    return {"program": P.text(), "theta": [], "owner": owner, "gamma": 10, "alpha": [8, 1]}


def swap(n: int) -> dict:
    """A carry `(V8, V16)` from two inputs, the step returning it swapped, so each leaf reads the other's width."""
    P = Program()
    root = Body(P, [])
    x8, x16 = root.input(8), root.input(16)
    ct = Tup(V(8), V(16))
    step = P.composite("Swap_v1", Body(P, [ct]), Tup(ct, Tup()), {"explicit": [["p", 0, 1], ["p", 0, 0]]})
    init = {"concat": [aff("n", x8, 0, 1), aff("n", x16, 0, 1)]}
    s = root.node("scan", step, [root.arg(init, ct)], Tup(ct, A(n, Tup())), n, n_xs=0)
    g = root.node("primitive", P.prim("Op8_v1", [V(8)], V(8)), [root.arg(aff("n", s, 0, 1), V(8))], V(8))
    return finish(P, root, g)


def exit_xs(n: int) -> dict:
    """A carry of three leaves from inputs of 8, 16 and 32 bits read as `V8`s, shifted down one place an iteration, the
    last filled from the iteration's `xs` row, a batch of `n` members of `W` sharing a `V64` input read as a `V8`."""
    P = Program()
    root = Body(P, [])
    xs = [root.input(w) for w in (8, 16, 32, 64)]
    w = P.composite("W_v1", Body(P, [V(8)]), V(8), aff("p", 0, 0, 1))
    b = root.node("batch", w, [root.arg(aff("n", xs[3], 0, 1), V(8))], A(n, V(8)), n, axes=[None])
    ct = Tup(V(8), V(8), V(8))
    step = P.composite("Shift_v1", Body(P, [ct, V(8)]), Tup(ct, Tup()),
                       {"concat": [aff("p", 0, 1, 2), aff("p", 1, 0, 1)]})
    init = {"concat": [aff("n", xs[0], 0, 1), aff("n", xs[1], 0, 1), aff("n", xs[2], 0, 1)]}
    s = root.node("scan", step, [root.arg(init, ct), root.arg(aff("n", b, 0, n), A(n, V(8)))],
                  Tup(ct, A(n, Tup())), n, n_xs=1)
    g = root.node("primitive", P.prim("Op8_v1", [V(8)], V(8)), [root.arg(aff("n", s, 0, 1), V(8))], V(8))
    return finish(P, root, g)


def carry_read(n: int, xs: bool) -> dict:
    """A step with a gate reading its carry's leaf 0. Without `xs`: a carry `(V8, V16)` that the step returns as
    `(c[1], c[1])`, a `V16` read as a `V8`; with `xs`: a carry `V8` the step returns from its `xs` row, a batch of `n`
    members of `W` sharing a `V32` input read as a `V8`."""
    P = Program()
    root = Body(P, [])
    op = P.prim("Op8_v1", [V(8)], V(8))
    if not xs:
        x8, x16 = root.input(8), root.input(16)
        ct = Tup(V(8), V(16))
        sb = Body(P, [ct])
        sb.node("primitive", op, [sb.arg(aff("p", 0, 0, 1), V(8))], V(8))
        step = P.composite("Step_v1", sb, Tup(ct, Tup()), {"explicit": [["p", 0, 1], ["p", 0, 1]]})
        init = {"concat": [aff("n", x8, 0, 1), aff("n", x16, 0, 1)]}
        s = root.node("scan", step, [root.arg(init, ct)], Tup(ct, A(n, Tup())), n, n_xs=0)
    else:
        x8, x32 = root.input(8), root.input(32)
        w = P.composite("W_v1", Body(P, [V(8)]), V(8), aff("p", 0, 0, 1))
        b = root.node("batch", w, [root.arg(aff("n", x32, 0, 1), V(8))], A(n, V(8)), n, axes=[None])
        sb = Body(P, [V(8), V(8)])
        sb.node("primitive", op, [sb.arg(aff("p", 0, 0, 1), V(8))], V(8))
        step = P.composite("Step_v1", sb, Tup(V(8), Tup()), aff("p", 1, 0, 1))
        s = root.node("scan", step, [root.arg(aff("n", x8, 0, 1), V(8)), root.arg(aff("n", b, 0, n), A(n, V(8)))],
                      Tup(V(8), A(n, Tup())), n, n_xs=1)
    g = root.node("primitive", op, [root.arg(aff("n", s, 0, 1), V(8))], V(8))
    return finish(P, root, g)


def xs_part(n: int) -> dict:
    """A step with a gate reading leaf 1 of its `xs` row `(V8, V8)`, the rows a batch of `n` members of `W2`, which
    returns its input and its input passed through `W`, sharing a `V16` input read as a `V8`."""
    P = Program()
    root = Body(P, [])
    x8, x16 = root.input(8), root.input(16)
    op = P.prim("Op8_v1", [V(8)], V(8))
    w = P.composite("W_v1", Body(P, [V(8)]), V(8), aff("p", 0, 0, 1))
    w2b = Body(P, [V(8)])
    w2b.node("call", w, [w2b.arg(aff("p", 0, 0, 1), V(8))], V(8))
    row = Tup(V(8), V(8))
    w2 = P.composite("W2_v1", w2b, row, {"concat": [aff("p", 0, 0, 1), aff("n", 0, 0, 1)]})
    b = root.node("batch", w2, [root.arg(aff("n", x16, 0, 1), V(8))], A(n, row), n, axes=[None])
    sb = Body(P, [V(8), row])
    sb.node("primitive", op, [sb.arg(aff("p", 1, 1, 1), V(8))], V(8))
    step = P.composite("Step_v1", sb, Tup(V(8), Tup()), aff("p", 0, 0, 1))
    s = root.node("scan", step, [root.arg(aff("n", x8, 0, 1), V(8)), root.arg(aff("n", b, 0, 2 * n), A(n, row))],
                  Tup(V(8), A(n, Tup())), n, n_xs=1)
    g = root.node("primitive", op, [root.arg(aff("n", s, 0, 1), V(8))], V(8))
    return finish(P, root, g)


#: (id, the instance at n, small n, big n)
SHAPES = [
    ("swap", swap, 5, 2 ** 52 + 1),
    ("exit-xs", exit_xs, 5, 2 ** 40),
    ("exit-xs-short", exit_xs, 2, 2),
    ("carry-read", lambda n: carry_read(n, xs=False), 6, 2 ** 40),
    ("carry-xs", lambda n: carry_read(n, xs=True), 6, 2 ** 40),
    ("xs-part", xs_part, 6, 2 ** 40),
]


def shapes() -> list[dict]:
    return [make(n) for _, make, small, big in SHAPES for n in dict.fromkeys((small, big))]


PAIR = Tup(V(8), V(16))
SWAPPED = Tup(V(16), V(8))
QUAD = Tup(V(8), V(16), V(8), V(16))


def pairs(N: int, how: str) -> dict:
    """`N` pairs `(V8, V16)`, a batch of `Pair` sharing two inputs, passed to `C`, whose body reads them as `how` says;
    `C` returns a `V8` its last node gives (`ret` the pairs' leaf 1 instead, a `V16`)."""
    P = Program()
    root = Body(P, [])
    x8, x16 = root.input(8), root.input(16)
    pair = P.composite("Pair_v1", Body(P, [V(8), V(16)]), PAIR, {"concat": [aff("p", 0, 0, 1), aff("p", 1, 0, 1)]})
    bt = root.node("batch", pair, [root.arg(aff("n", x8, 0, 1), V(8)), root.arg(aff("n", x16, 0, 1), V(16))],
                   A(N, PAIR), N, axes=[None, None])
    cb = Body(P, [A(N, PAIR), V(16)])
    op8 = P.prim("Op8_v1", [V(8)], V(8))
    if how in ("leaf0", "leaf1"):
        s = P.prim(f"Sum{N}w8_v1", [A(N, V(8))], V(8))
        refs = {"strided": ["p", 0, 0 if how == "leaf0" else 1, N, 0, 2, N]}
        last = cb.node("primitive", s, [cb.arg(refs, A(N, V(8)))], V(8))
    elif how in ("swapped", "shifted"):
        t = A(N - 1, SWAPPED if how == "swapped" else PAIR)
        last = cb.node("primitive", P.prim(f"Pairs{how}{N}_v1", [t], V(8)), [cb.arg(aff("p", 0, 1, 2 * N - 2), t)], V(8))
    elif how in ("quads", "quads-off"):
        t = A(N // 2 - (how == "quads-off"), QUAD)
        refs = aff("p", 0, 0, 2 * N) if how == "quads" else aff("p", 0, 1, 2 * N - 4)
        last = cb.node("primitive", P.prim(f"Quads{how}{N}_v1", [t], V(8)), [cb.arg(refs, t)], V(8))
    elif how in ("param", "param-ok"):
        last = cb.node("primitive", op8 if how == "param" else P.prim("Op16_v1", [V(16)], V(8)),
                       [cb.arg(aff("p", 1, 0, 1), V(8) if how == "param" else V(16))], V(8))
    elif how in ("explicit", "explicit-ok"):
        last = cb.node("primitive", P.prim("Op8x8_v1", [Tup(V(8), V(8))], V(8)),
                       [cb.arg({"explicit": [["p", 0, 2], ["p", 0, 3 if how == "explicit" else 4]]}, Tup(V(8), V(8)))],
                       V(8))
    elif how == "ret":
        cb.node("primitive", op8, [cb.arg(aff("p", 0, 0, 1), V(8))], V(8))
        P.composite("C_v1", cb, V(8), aff("p", 0, 1, 1))
        c = root.node("call", "C_v1", [root.arg(aff("n", bt, 0, 2 * N), A(N, PAIR)), root.arg(aff("n", x16, 0, 1), V(16))],
                      V(8))
        return finish(P, root, c)
    elif how == "later":
        # node 0 reads node 1, a V16, as a V8: a later node, which `not-topological` refuses
        cb.node("primitive", op8, [cb.arg(aff("n", 1, 0, 1), V(8))], V(8))
        cb.node("primitive", P.prim("Op16r16_v1", [V(16)], V(16)), [cb.arg(aff("p", 1, 0, 1), V(16))], V(16))
        last = cb.node("primitive", op8, [cb.arg(aff("n", 0, 0, 1), V(8))], V(8))
    else:
        raise ValueError(how)
    P.composite("C_v1", cb, V(8), aff("n", last, 0, 1))
    c = root.node("call", "C_v1", [root.arg(aff("n", bt, 0, 2 * N), A(N, PAIR)), root.arg(aff("n", x16, 0, 1), V(16))],
                  V(8))
    return finish(P, root, c)


PAIR_HOWS = ("leaf0", "leaf1", "swapped", "shifted", "quads", "quads-off", "param", "param-ok", "explicit",
             "explicit-ok", "ret", "later")
WELL_TYPED = ("leaf0", "swapped", "quads", "param-ok", "explicit-ok")
#: the refused ones at `2^20`; the well-typed ones at `2^14`, since `NciCheck.Desc` reads a primitive's operands one by
#: one, and does so in seconds past it
PAIR_SIZES = {False: (4, 2 ** 20), True: (4, 2 ** 14)}


def pairs_all() -> list[dict]:
    return [pairs(N, how) for how in PAIR_HOWS for N in PAIR_SIZES[how in WELL_TYPED]]


TAMPERED = 300
FIRST_SEED = 10 ** 6


def _sources(body: dict, k: int, types: list) -> list[tuple[str, int, int]]:
    """(space, index, leaves) of the collections a reference of node `k` (the return: `k` past the last node) may name."""
    from verity.ml.circuits.types import type_from_json
    out = [("p", i, type_from_json(t).leaves) for i, (_, t) in enumerate(body["params"])]
    out += [("n", i, type_from_json(types[nd["out"]]).leaves) for i, nd in enumerate(body["nodes"][:k])]
    return [s for s in out if s[2] > 0]


def _parts(refs: dict):
    """The parts of a reference object, nested concatenations included."""
    if "concat" in refs:
        for r in refs["concat"]:
            yield from _parts(r)
    elif any(k in refs for k in ("affine", "strided", "explicit")):
        yield refs


def tamper(rng: random.Random, inst: dict) -> dict | None:
    """`inst` with one reference of a random composite's body moved to another leaf source, or None where it has none."""
    desc = json.loads(inst["program"])
    sites = []
    for d in desc["definitions"].values():
        if d.get("form") != "composite":
            continue
        body = d["body"]
        for k, nd in enumerate(body["nodes"]):
            for a in nd["args"]:
                sites += [(body, k, p) for p in _parts(a["refs"])]
        sites += [(body, len(body["nodes"]), p) for p in _parts(body["ret"])]
    rng.shuffle(sites)
    for body, k, p in sites:
        srcs = _sources(body, k, desc["types"])
        if "affine" in p:
            s, i, b, c = p["affine"]
            fit = [x for x in srcs if x[2] >= c and (x[0], x[1]) != (s, i)]
            if c == 0 or not fit:
                continue
            s2, i2, n2 = rng.choice(fit)
            p["affine"] = [s2, i2, rng.randint(0, n2 - c), c]
        elif "explicit" in p:
            if not p["explicit"] or not srcs:
                continue
            j = rng.randrange(len(p["explicit"]))
            s2, i2, n2 = rng.choice(srcs)
            p["explicit"][j] = [s2, i2, rng.randrange(n2)]
        else:
            s, i, b, inner, os_, is_, c = p["strided"]
            q, last = (c - 1) // inner, (c - 1) % inner
            corners = [(0, 0), (0, min(inner, c) - 1), (q, 0), (q, last)] + ([(q - 1, inner - 1)] if q > 0 else [])
            vals = [b + x * os_ + y * is_ for x, y in corners]
            lo, hi = min(vals), max(vals)
            fit = [x for x in srcs if x[2] > hi - lo and (x[0], x[1]) != (s, i)]
            if c == 0 or not fit:
                continue
            s2, i2, n2 = rng.choice(fit)
            p["strided"] = [s2, i2, b - lo + rng.randint(0, n2 - 1 - (hi - lo)), inner, os_, is_, c]
        return inst | {"program": json.dumps(desc, separators=(",", ":"), sort_keys=True)}
    return None


def tampered(instance) -> list[dict]:
    """`TAMPERED` instances: the well-formed ones of `instance(0)`, `instance(1)`, … (`desc_difftest_vectors.instance`,
    its random declarations), each tampered with seed `FIRST_SEED + seed`."""
    out, seed = [], 0
    while len(out) < TAMPERED:
        made = instance(seed)
        if made["malformed"] is None:
            t = tamper(random.Random(FIRST_SEED + seed), made["instance"])
            if t is not None:
                out.append(t)
        seed += 1
    return out


def cases(instance) -> list[dict]:
    """Every instance of this module: `shapes()`, `pairs_all()`, then `tampered(instance)`."""
    return [*shapes(), *pairs_all(), *tampered(instance)]
