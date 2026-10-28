"""The desc fuzzer's data: generated descriptor v1 Programs for `NciCheck/DescFuzzMain.lean`.

    python scripts/desc_fuzz_programs.py <verity checkout> --out DIR

Writes `DIR/desc_fuzz/programs.jsonl`, one `{"id", "family", "program", "agree"}` object per line, from fixed seeds
(`FAMILIES`), with `"instances"` (IU assignments, each `{"theta", "owner", "gamma", "alpha"}`) on the shared families'
(`agree`: how many instances of its own `NciCheck.DescFuzz.agrees` checks beside them).
The Lean audit runs `NciCheck/DescFuzzMain.lean` on it (`runs` in `lean-audit.json`, which pins its SHA-256), which
decides the statements behind `DescAgrees` on each Program (`NciCheck.DescFuzz`'s docstring) and fails if one fails;
`verity/core/protocols/network/tests/test_nci_desc_fuzz.py` checks that this script still writes it. The search itself,
at millions of Programs, is the lane's (`verity/core/protocols/network/nci/fuzz/`); this is a seeded sample of it at a size
the audit runs on every change to the package.

`random` writes descriptor v1 JSON directly, not through `verity.ml.circuits`' Builder, so that every reference form the
decoder reads appears (affine, strided with negative and zero strides, explicit, concat, alias) and so that shapes the
Builder never emits do too: scans of no iteration and of one, batches mapped over an inner axis or over none, composites
with no nodes, parameters read by no gate and threaded through returns, returns repeating or reversing gates, nested
composites calling each other in batches and scans, composites keyed with an input id, and widths up to 2^53 - 1. The
decoder checks a reference's count against its type and not its leaves' widths, so any leaves of the right count fill an
argument. `big` is `random` at three times the Definitions, nodes per body and root nodes, `chain` Definitions nested 8 to
64 deep, and `copies` batches read as segment copies and their slices. Each Program of these is also written once more,
with probability `mutate`, with one to three random edits (a count, an axis, a reference, a type index); the decoder
refuses most of them, and the rest are what it lets through.

`hand` writes the worked shapes: the Programs on which `Walk.OperandsResolve` fails as stated (a composite node's
argument that its callee never reads names a scan of no iterations, at the root, in a callee, and as a scan's shared
argument), each with the variant its callee reads, and gates reading the carry of a scan of 2^4 to 2^14 pass-through
iterations. `seeds` and `shared` are the lane's shared-members family (`fuzz/gen/shared.py`, ported as it is): whole
members of a batch returning leaves of arguments they share, each another root node's output, read by a gate.
`desc_difftest_vectors.shared()`'s six declarations first (`shared-seed`, each with its two instances), which
`NciCheck.Desc` once attributed to their first piece's node alone, then `decl` on parameters drawn at random
(`draw`): two or three shared arguments, each made by a root node of its own (a primitive on an input, a call of a
composite of two gates, one member of a batch of primitives, or the input itself, registered or bound per run); the
members' return, each argument's leaves in any order and repeated, now and then with a gate of the member's own; two to
four members, with a mapped argument or none; the reader (a primitive, a composite's primitive, or either reading a
composite that makes the batch) and what it reads (every member, a run of whole members, or a run cut inside them), now
and then a second primitive reading another run. Each comes with up to `ASSIGNMENTS` IU assignments of its root nodes
that hold gates (all of them when there are no more, else drawn without repeats), θ the registered inputs, `Γ` all the
gates and `α` 1.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import verity.ml.circuits
from verity.ml.circuits import codec
from verity.ml.circuits import program as ir
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Affine, Coll, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCHEMA = "verity-ir/descriptor/v1"
WIDTHS = [1, 2, 3, 8, 16, 32, 64]
WIDE = 2 ** 53 - 1


def V(w):
    return ("V", w)


def A(n, e):
    return ("A", n, e)


def Tup(*fs):
    return ("T", tuple(fs))


def Rec(*fs):
    return ("R", tuple(fs))


def leaves(t) -> int:
    if t[0] == "V":
        return 1
    if t[0] == "A":
        return t[1] * leaves(t[2])
    return sum(leaves(f) for f in t[1])


def tj(t):
    if t[0] == "V":
        return ["Value", t[1]]
    if t[0] == "A":
        return ["Array", t[1], tj(t[2])]
    if t[0] == "R":
        return ["Record", [[f"f{i}", tj(f)] for i, f in enumerate(t[1])]]
    return ["Tuple", [tj(f) for f in t[1]]]


def insert_axis(t, ax: int, n: int):
    """The type whose axis `ax` stripped (`Ty.removeAxis`) is `t`, of length `n`."""
    if ax == 0:
        return A(n, t)
    return A(t[1], insert_axis(t[2], ax - 1, n))


def array_depth(t) -> int:
    return 1 + array_depth(t[2]) if t[0] == "A" else 0


class Program:
    """A descriptor being written: its definitions, callees and types tables."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.defs: dict[str, dict] = {}
        self.sig: dict[str, tuple] = {}
        self.callees: list[str] = []
        self.types: list = []
        self._tidx: dict[str, int] = {}
        self.composites: list[str] = []
        self.n = 0

    def ty(self, t) -> int:
        k = json.dumps(tj(t))
        if k not in self._tidx:
            self._tidx[k] = len(self.types)
            self.types.append(tj(t))
        return self._tidx[k]

    def fn(self, key: str) -> int:
        if key not in self.callees:
            self.callees.append(key)
        return self.callees.index(key)

    def width(self) -> int:
        r = self.rng.random()
        return WIDE if r < 0.03 else self.rng.choice(WIDTHS)

    def prim(self, params, ret, key: str | None = None) -> str:
        if key is None:
            key = f"P{self.n}_v1"
            self.n += 1
        if key not in self.sig:
            self.defs[key] = {"form": "primitive", "id": key, "params": [[f"a{i}", tj(p)] for i, p in enumerate(params)],
                              "ret": tj(ret)}
            self.sig[key] = (True, list(params), ret, 1)
        return key

    def input(self, w: int, registered: bool) -> str:
        key = f"{'Registered' if registered else ''}Input{w}_v1"
        while key in self.composites:
            w += 1
            key = f"{'Registered' if registered else ''}Input{w}_v1"
        return self.prim([], V(w), key)

    def small_type(self, depth: int = 0):
        rng = self.rng
        r = rng.random()
        if depth > 1 or r < 0.5:
            return V(self.width())
        if r < 0.8:
            return A(rng.choice([0, 1, 2, 3, 4]), self.small_type(depth + 1))
        if r < 0.9:
            return Rec(*[self.small_type(depth + 1) for _ in range(rng.choice([0, 1, 2, 3]))])
        return Tup(*[self.small_type(depth + 1) for _ in range(rng.choice([0, 1, 2, 3]))])

    def composite(self, body: "Body", ret, ret_refs, key: str | None = None) -> str:
        if key is None:
            key = f"C{self.n}_v1"
            self.n += 1
        self.defs[key] = {"form": "composite", "id": key, "bindings": {}, "body": {
            "params": [[f"x{i}", tj(t)] for i, t in enumerate(body.params)], "ret_type": tj(ret), "nodes": body.nodes,
            "ret": ret_refs}}
        self.sig[key] = (False, list(body.params), ret, body.gates)
        self.composites.append(key)
        return key

    def text(self, root: str) -> str:
        d = {"annotations": {}, "callees": self.callees, "definitions": self.defs, "program_fn": root, "root": root,
             "schema": SCHEMA, "types": self.types}
        return json.dumps(d, separators=(",", ":"), sort_keys=True)


class Body:
    """A body being written: its parameters, its nodes so far and their outputs, gates and argument counts."""

    def __init__(self, prog: Program, params: list, budget: int):
        self.P = prog
        self.rng = prog.rng
        self.params = params
        self.nodes: list[dict] = []
        self.outs: list = []
        self.kinds: list[str] = []
        self.argcounts: list[list[int]] = []
        self.gates = 0
        self.budget = budget

    def colls(self, bias: str | None):
        cs = [("p", i, leaves(t), "param") for i, t in enumerate(self.params)]
        cs += [("n", k, leaves(t), self.kinds[k]) for k, t in enumerate(self.outs)]
        cs = [c for c in cs if c[2] > 0]
        if bias and any(c[3] == bias for c in cs) and self.rng.random() < 0.6:
            cs = [c for c in cs if c[3] == bias]
        return cs

    def simple(self, L: int, cs) -> dict:
        rng = self.rng
        big = [c for c in cs if c[2] >= L]
        if big and rng.random() < 0.7:
            s, i, N, _ = rng.choice(big)
            return {"affine": [s, i, rng.randint(0, N - L), L]}
        items = []
        for _ in range(L):
            s, i, N, _ = rng.choice(cs)
            items.append([s, i, rng.randrange(N)])
        return {"explicit": items}

    def refs(self, L: int, bias: str | None = None, depth: int = 0) -> dict | None:
        """References to `L` leaves of the parameters and the nodes so far."""
        rng = self.rng
        if L == 0:
            return rng.choice([{"affine": ["p", 0, 0, 0]}, {"explicit": []}, {"concat": []},
                               {"strided": ["n", 0, 0, 1, 0, 0, 0]}])
        cs = self.colls(bias)
        if not cs:
            return None
        r = rng.random()
        if r < 0.08:
            same = [(k, a) for k, cnts in enumerate(self.argcounts) for a, c in enumerate(cnts) if c == L]
            if same:
                k, a = rng.choice(same)
                return {"alias": [k, a]}
        if r < 0.42:
            big = [c for c in cs if c[2] >= L]
            if big:
                s, i, N, _ = rng.choice(big)
                return {"affine": [s, i, rng.randint(0, N - L), L]}
        if r < 0.68:
            s, i, N, _ = rng.choice(cs)
            divs = [d for d in range(1, L + 1) if L % d == 0]
            for _ in range(12):
                inner = rng.choice([1, 2, 3, L, *divs])
                os_, is_ = rng.randint(-3, 4), rng.randint(-2, 3)
                b = rng.randint(0, N - 1)
                els = [b + (k // inner) * os_ + (k % inner) * is_ for k in range(L)]
                if all(0 <= e < N for e in els):
                    return {"strided": [s, i, b, inner, os_, is_, L]}
        if r < 0.86 and L <= 16:
            return self.simple(L, cs) if rng.random() < 0.3 else {"explicit": [
                [c[0], c[1], rng.randrange(c[2])] for c in (rng.choice(cs) for _ in range(L))]}
        if depth > 2:
            return self.simple(L, cs)
        parts, left = [], L
        while left > 0:
            m = rng.randint(1, left)
            parts.append(self.refs(m, bias, depth + 1) or self.simple(m, cs))
            left -= m
        if rng.random() < 0.2:
            parts.insert(rng.randint(0, len(parts)), {"affine": ["n", 0, 0, 0]})
        return {"concat": parts}

    def push(self, node: dict, out, counts: list[int], gates: int, kind: str) -> None:
        self.nodes.append(node)
        self.outs.append(out)
        self.argcounts.append(counts)
        self.kinds.append(kind)
        self.gates += gates

    def args_for(self, types: list, bias: str | None) -> list[dict] | None:
        out = []
        for t in types:
            r = self.refs(leaves(t), bias)
            if r is None:
                return None
            out.append({"refs": r, "type": self.P.ty(t)})
        return out

    def bias(self) -> str | None:
        return self.rng.choice([None, None, "param", "scan", "batch", "call", "prim"])

    def add_prim(self) -> bool:
        P, rng = self.P, self.rng
        wide = [c for c in self.colls(None) if c[2] >= 2]
        if wide and rng.random() < 0.25:
            # one argument over most or all of one parameter's or node's leaves: whole batch members at once
            batches = [c for c in wide if c[3] == "batch"]
            s, i, N, _ = rng.choice(batches if batches and rng.random() < 0.7 else wide)
            L = rng.choice([N, N, rng.randint(2, N)])
            b = rng.randint(0, N - L)
            params = [A(L, V(P.width()))]
            args = [{"refs": {"affine": [s, i, b, L]}, "type": P.ty(params[0])}]
        else:
            params = [P.small_type(1) for _ in range(rng.choice([0, 1, 1, 2, 2, 3, 4]))]
            args = self.args_for(params, self.bias())
        if args is None:
            params, args = [], []
        key = P.prim(params, V(P.width()))
        out = P.sig[key][2]
        self.push({"args": args, "fn": P.fn(key), "form": "primitive", "out": P.ty(out)}, out,
                  [leaves(t) for t in params], 1, "prim")
        return True

    def add_call(self) -> bool:
        P = self.P
        if not P.composites:
            return False
        key = self.rng.choice(P.composites)
        _, params, ret, g = P.sig[key]
        if self.gates + g > self.budget:
            return False
        args = self.args_for(params, self.bias())
        if args is None:
            return False
        self.push({"args": args, "fn": P.fn(key), "form": "call", "out": P.ty(ret)}, ret, [leaves(t) for t in params], g,
                  "call")
        return True

    def add_batch(self) -> bool:
        P, rng = self.P, self.rng
        if P.composites and rng.random() < 0.6:
            key = rng.choice(P.composites)
        else:
            params = [P.small_type(1) for _ in range(rng.choice([0, 1, 1, 2, 3]))]
            key = P.prim(params, V(P.width()))
        prim, params, ret, g = P.sig[key]
        n = rng.choice([0, 1, 1, 2, 2, 3, 4, 5])
        if self.gates + n * g > self.budget:
            return False
        axes, types = [], []
        for pt in params:
            if rng.random() < 0.65:
                ax = rng.randint(0, array_depth(pt))
                axes.append(ax)
                types.append(insert_axis(pt, ax, n))
            else:
                axes.append(None)
                types.append(pt)
        args = self.args_for(types, self.bias())
        if args is None:
            return False
        out = A(n, ret)
        self.push({"args": args, "axes": axes, "fn": P.fn(key), "form": "batch", "n": n, "out": P.ty(out)}, out,
                  [leaves(t) for t in types], n * g, "batch")
        return True

    def add_scan(self) -> bool:
        P, rng = self.P, self.rng
        steps = [k for k in P.composites if P.sig[k][1] and P.sig[k][2][0] == "T" and len(P.sig[k][2][1]) == 2
                 and P.sig[k][2][1][0] == P.sig[k][1][0]]
        if not steps:
            return False
        key = rng.choice(steps)
        _, params, ret, g = P.sig[key]
        n = rng.choice([0, 0, 1, 1, 2, 3, 5])
        if self.gates + n * g > self.budget:
            return False
        nxs = rng.randint(0, len(params) - 1)
        types = [params[0]] + [A(n, pt) if i <= nxs else pt for i, pt in enumerate(params) if i > 0]
        args = self.args_for(types, self.bias())
        if args is None:
            return False
        out = Tup(params[0], A(n, ret[1][1]))
        self.push({"args": args, "fn": P.fn(key), "form": "scan", "n": n, "n_xs": nxs, "out": P.ty(out)}, out,
                  [leaves(t) for t in types], n * g, "scan")
        return True

    def add_random(self) -> None:
        r = self.rng.random()
        for f in ([self.add_scan] if r < 0.22 else [self.add_batch] if r < 0.45 else [self.add_call] if r < 0.68 else []):
            if f():
                return
        self.add_prim()


def ret_of(body: Body, L_max: int = 6):
    """A return type and its references: of the body's nodes, or its parameters passed through."""
    P, rng = body.P, body.rng
    gates = [c for c in body.colls(None) if c[3] == "prim"]
    if gates and rng.random() < 0.3:
        # gates only, in no single run: what a batch of the Definition returns as copies of one member's pieces
        L = rng.randint(2, L_max)
        items = [["n", *rng.choice(gates)[1:2], 0] for _ in range(L)]
        return A(L, V(P.width())), {"explicit": items}
    for _ in range(8):
        t = P.small_type(0) if rng.random() < 0.6 else Tup(*[P.small_type(1) for _ in range(rng.randint(1, 3))])
        if leaves(t) > L_max:
            continue
        r = body.refs(leaves(t), rng.choice([None, "param", "param", "scan", "batch", "prim"]))
        if r is not None:
            return t, r
    return Tup(), {"affine": ["p", 0, 0, 0]}


def composite(P: Program, budget: int, scale: int = 1) -> None:
    rng = P.rng
    params = [P.small_type(0) for _ in range(rng.choice([0, 1, 1, 2, 2, 3]))]
    body = Body(P, params, budget)
    for _ in range(rng.choice([0, 0, 1, 1, 2, 3, 4, 6]) * scale):
        body.add_random()
    if params and rng.random() < 0.45:
        carry = params[0]
        y = P.small_type(1)
        cr = body.refs(leaves(carry), rng.choice([None, "param", "param"]))
        yr = body.refs(leaves(y))
        if cr is not None and yr is not None:
            P.composite(body, Tup(carry, y), {"concat": [cr, yr]}, rng.choice([None] * 9 + ["Input16_v1"])
                        if "Input16_v1" not in P.defs else None)
            return
    t, r = ret_of(body)
    P.composite(body, t, r, rng.choice([None] * 15 + ["Input8_v1"]) if "Input8_v1" not in P.defs else None)


def program(seed: int, scale: int = 1) -> str:
    """`random`'s Program; with `scale`, that many times the Definitions, nodes per body and root nodes (`big`)."""
    rng = random.Random(seed)
    P = Program(rng)
    budget = rng.choice([8, 32, 128, 512])
    for _ in range(rng.choice([0, 1, 2, 3, 4, 6, 8]) * scale):
        composite(P, budget, scale)
    root = Body(P, [], budget * 4)
    for _ in range(rng.choice([0, 1, 1, 2, 3])):
        w, reg = P.width(), rng.random() < 0.4
        key = P.input(w if w < 2 ** 20 else 8, reg)
        out = P.sig[key][2]
        if rng.random() < 0.5:
            root.push({"args": [], "fn": P.fn(key), "form": "primitive", "out": P.ty(out)}, out, [], 1, "input")
        else:
            n = rng.choice([1, 1, 2, 3])
            root.push({"args": [], "axes": [], "fn": P.fn(key), "form": "batch", "n": n, "out": P.ty(A(n, out))},
                      A(n, out), [], n, "input")
    for _ in range(rng.randint(1, 8) * scale):
        root.add_random()
    t, r = ret_of(root, 12)
    P.composite(root, t, r, "R_v1")
    return P.text("R_v1")


def chain(seed: int) -> str:
    """Definitions nested `D` deep (8 to 64), each `(c, s) -> (carry, ())` with one node on the one below (a call, a scan
    of zero to two iterations, or a batch of one or two with no axis mapped), now and then a primitive beside it, and a
    carry that is a parameter passed through, the node's carry or the primitive's gate; the root calls the top with its
    inputs and reads the result."""
    rng = random.Random(seed)
    P = Program(rng)
    w = rng.choice(WIDTHS)
    t = V(w)
    unit = Tup()
    ret = Tup(t, unit)
    op = P.prim([t, t], t, "Op_v1")
    base_b = Body(P, [t, t], 1)
    if rng.random() < 0.5:
        base_b.push({"args": [{"refs": {"affine": ["p", rng.randint(0, 1), 0, 1]}, "type": P.ty(t)},
                              {"refs": {"affine": ["p", rng.randint(0, 1), 0, 1]}, "type": P.ty(t)}],
                     "fn": P.fn(op), "form": "primitive", "out": P.ty(t)}, t, [1, 1], 1, "prim")
        carry = rng.choice([["n", 0, 0, 1], ["p", 0, 0, 1], ["p", 1, 0, 1]])
    else:
        carry = ["p", rng.randint(0, 1), 0, 1]
    below = P.composite(base_b, ret, {"affine": carry}, "L0_v1")
    D = rng.choice([8, 16, 32, 64])
    for lvl in range(1, D + 1):
        b = Body(P, [t, t], 10 ** 9)
        g = P.sig[below][3]
        x, y = rng.randint(0, 1), rng.randint(0, 1)
        a0 = {"refs": {"affine": ["p", x, 0, 1]}, "type": P.ty(t)}
        a1 = {"refs": {"affine": ["p", y, 0, 1]}, "type": P.ty(t)}
        r = rng.random()
        if r < 0.4:
            b.push({"args": [a0, a1], "fn": P.fn(below), "form": "call", "out": P.ty(ret)}, ret, [1, 1], g, "call")
            leaf = 0
        elif r < 0.75:
            n = rng.choices([0, 1, 2], [1, 30, 6])[0]
            out = Tup(t, A(n, unit))
            b.push({"args": [a0, a1], "fn": P.fn(below), "form": "scan", "n": n, "n_xs": 0, "out": P.ty(out)}, out,
                   [1, 1], n * g, "scan")
            leaf = 0
        else:
            n = rng.choices([1, 2], [8, 2])[0]
            out = A(n, ret)
            b.push({"args": [a0, a1], "axes": [None, None], "fn": P.fn(below), "form": "batch", "n": n,
                    "out": P.ty(out)}, out, [1, 1], n * g, "batch")
            leaf = 0
        choices = [["p", 0, 0, 1], ["p", 1, 0, 1], ["n", 0, leaf, 1]]
        if rng.random() < 0.3:
            b.push({"args": [{"refs": {"affine": rng.choice(choices)}, "type": P.ty(t)},
                             {"refs": {"affine": ["p", rng.randint(0, 1), 0, 1]}, "type": P.ty(t)}],
                    "fn": P.fn(op), "form": "primitive", "out": P.ty(t)}, t, [1, 1], 1, "prim")
            choices.append(["n", 1, 0, 1])
        below = P.composite(b, ret, {"affine": rng.choice(choices)}, f"L{lvl}_v1")
    root = Body(P, [], 10 ** 9)
    for reg in (True, False):
        key = P.input(w, reg)
        root.push({"args": [], "fn": P.fn(key), "form": "primitive", "out": P.ty(t)}, t, [], 1, "input")
    root.push({"args": [{"refs": {"affine": ["n", 0, 0, 1]}, "type": P.ty(t)},
                        {"refs": {"affine": ["n", 1, 0, 1]}, "type": P.ty(t)}],
               "fn": P.fn(below), "form": "call", "out": P.ty(ret)}, ret, [1, 1], P.sig[below][3], "call")
    root.push({"args": [{"refs": {"affine": ["n", 2, 0, 1]}, "type": P.ty(t)},
                        {"refs": {"affine": ["n", rng.randint(0, 1), 0, 1]}, "type": P.ty(t)}],
               "fn": P.fn(op), "form": "primitive", "out": P.ty(t)}, t, [1, 1], 1, "prim")
    P.composite(root, t, {"affine": ["n", 3, 0, 1]}, "R_v1")
    return P.text("R_v1")


def copies(seed: int) -> str:
    """Segments of copies (`Desc.Seg.reps > 1`) and their slices: `C0`, one to three primitives on its parameter,
    returning its gates in no single run (repeated, reversed, interleaved); `C1`, a batch of three to six of `C0`
    returning a slice of the batch (`Desc.Seg.slice` through the copies), now and then a second batch beside it; at the
    root, inputs, a call or a batch of `C1`, and primitives reading slices of what they return, whole members and cut
    ones, some through `W`, which passes a slice through."""
    rng = random.Random(seed)
    P = Program(rng)
    w = rng.choice(WIDTHS)
    t = V(w)
    b0 = Body(P, [t], 10)
    for _ in range(rng.randint(1, 3)):
        key = P.prim([t], V(rng.choice(WIDTHS)))
        out = P.sig[key][2]
        src = ["p", 0, 0, 1] if not b0.nodes or rng.random() < 0.5 else ["n", rng.randrange(len(b0.nodes)), 0, 1]
        b0.push({"args": [{"refs": {"affine": src}, "type": P.ty(t)}], "fn": P.fn(key), "form": "primitive",
                 "out": P.ty(out)}, out, [1], 1, "prim")
    G = len(b0.nodes)
    L = rng.randint(2, 4)
    items = [["n", rng.randrange(G), 0] for _ in range(L)]
    if G > 1 and rng.random() < 0.5:
        items = [["n", G - 1 - (i % G), 0] for i in range(L)]
    # the return's leaves are gates of several widths, typed as one array: the decoder checks a reference's count only
    c0 = P.composite(b0, A(L, t), {"explicit": items})
    c0_ret = P.sig[c0][2]
    b1 = Body(P, [t], 10 ** 6)
    n1 = rng.randint(3, 6)
    for _ in range(rng.choice([1, 1, 2])):
        out = A(n1, c0_ret)
        b1.push({"args": [{"refs": {"affine": ["p", 0, 0, 1]}, "type": P.ty(t)}], "axes": [None], "fn": P.fn(c0),
                 "form": "batch", "n": n1, "out": P.ty(out)}, out, [1], n1 * G, "batch")
    N = n1 * L
    a = rng.choice([0, 0, rng.randrange(N)])
    ln = rng.randint(1, N - a)
    c1 = P.composite(b1, A(ln, V(w)), {"affine": ["n", rng.randrange(len(b1.nodes)), a, ln]})
    wb = Body(P, [A(ln, V(w))], 0)
    wa = rng.randrange(ln)
    wl = rng.randint(1, ln - wa)
    wfn = P.composite(wb, A(wl, V(w)), {"affine": ["p", 0, wa, wl]})
    root = Body(P, [], 10 ** 6)
    key = P.input(w, rng.random() < 0.5)
    root.push({"args": [], "fn": P.fn(key), "form": "primitive", "out": P.ty(t)}, t, [], 1, "input")
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.5:
            root.push({"args": [{"refs": {"affine": ["n", 0, 0, 1]}, "type": P.ty(t)}], "fn": P.fn(c1), "form": "call",
                       "out": P.ty(A(ln, V(w)))}, A(ln, V(w)), [ln], P.sig[c1][3], "call")
        else:
            nb = rng.randint(2, 4)
            out = A(nb, A(ln, V(w)))
            root.push({"args": [{"refs": {"affine": ["n", 0, 0, 1]}, "type": P.ty(t)}], "axes": [None], "fn": P.fn(c1),
                       "form": "batch", "n": nb, "out": P.ty(out)}, out, [1], nb * P.sig[c1][3], "batch")
    made = [k for k in range(1, len(root.nodes))]
    for _ in range(rng.randint(1, 4)):
        k = rng.choice(made)
        M = leaves(root.outs[k])
        if rng.random() < 0.3:
            s0 = rng.randrange(M - ln + 1)
            root.push({"args": [{"refs": {"affine": ["n", k, s0, ln]}, "type": P.ty(A(ln, V(w)))}], "fn": P.fn(wfn),
                       "form": "call", "out": P.ty(A(wl, V(w)))}, A(wl, V(w)), [ln], 0, "call")
            k, M = len(root.nodes) - 1, wl
        s0 = rng.randrange(M)
        sl = rng.randint(1, M - s0)
        key = P.prim([A(sl, V(w))], V(w))
        root.push({"args": [{"refs": {"affine": ["n", k, s0, sl]}, "type": P.ty(A(sl, V(w)))}], "fn": P.fn(key),
                   "form": "primitive", "out": P.ty(V(w))}, V(w), [sl], 1, "prim")
    P.composite(root, t, {"affine": ["n", len(root.nodes) - 1, 0, 1]}, "R_v1")
    return P.text("R_v1")


def passscan(n: int) -> str:
    """A gate reading the carry of a scan of `n` iterations of a step that passes its carry through, from an input."""
    P = Program(random.Random(0))
    x = P.input(8, True)
    op = P.prim([V(8)], V(8), "Op8_v1")
    step = P.composite(Body(P, [V(8)], 0), Tup(V(8), Tup()), {"affine": ["p", 0, 0, 1]}, "Step_v1")
    root = Body(P, [], 10)
    root.push({"args": [], "fn": P.fn(x), "form": "primitive", "out": P.ty(V(8))}, V(8), [], 1, "input")
    out = Tup(V(8), A(n, Tup()))
    root.push({"args": [{"refs": {"affine": ["n", 0, 0, 1]}, "type": P.ty(V(8))}], "fn": P.fn(step), "form": "scan",
               "n": n, "n_xs": 0, "out": P.ty(out)}, out, [1], 0, "scan")
    root.push({"args": [{"refs": {"affine": ["n", 1, 0, 1]}, "type": P.ty(V(8))}], "fn": P.fn(op),
               "form": "primitive", "out": P.ty(V(8))}, V(8), [1], 1, "prim")
    P.composite(root, V(8), {"affine": ["n", 2, 0, 1]}, "R_v1")
    return P.text("R_v1")


def mutate(rng: random.Random, text: str) -> str:
    """One to three edits of a descriptor: a node's count, axis or type, or a reference's base, space or leaf."""
    d = json.loads(text)
    comps = [v for v in d["definitions"].values() if v.get("form") == "composite"]
    for _ in range(rng.randint(1, 3)):
        c = rng.choice(comps)
        nodes = c["body"]["nodes"]
        what = rng.random()
        if nodes and what < 0.3:
            nd = rng.choice(nodes)
            if "n" in nd:
                nd["n"] = max(0, nd["n"] + rng.choice([-1, 1, -nd["n"]]))
            if "n_xs" in nd and rng.random() < 0.5:
                nd["n_xs"] = max(0, nd["n_xs"] + rng.choice([-1, 1]))
            if "axes" in nd and nd["axes"] and rng.random() < 0.5:
                i = rng.randrange(len(nd["axes"]))
                nd["axes"][i] = rng.choice([None, 0, 1])
        elif nodes and what < 0.45:
            nd = rng.choice(nodes)
            nd["out"] = rng.randrange(len(d["types"]))
        else:
            targets = [c["body"]] + [a for nd in nodes for a in nd["args"]]
            tgt = rng.choice(targets)
            key = "ret" if "ret" in tgt else "refs"
            tgt[key] = _mutate_refs(rng, tgt[key])
    return json.dumps(d, separators=(",", ":"), sort_keys=True)


def _mutate_refs(rng, r):
    r = copy.deepcopy(r)
    if "affine" in r:
        a = r["affine"]
        i = rng.choice([0, 1, 2])
        if i == 0:
            a[0] = "p" if a[0] == "n" else "n"
        else:
            a[i] = max(0, a[i] + rng.choice([-1, 1]))
    elif "strided" in r:
        a = r["strided"]
        i = rng.choice([1, 2, 4, 5])
        a[i] = a[i] + rng.choice([-1, 1])
        if i == 1:
            a[1] = max(0, a[1])
    elif "explicit" in r and r["explicit"]:
        it = rng.choice(r["explicit"])
        j = rng.choice([1, 2])
        it[j] = max(0, it[j] + rng.choice([-1, 1]))
    elif "concat" in r and r["concat"]:
        i = rng.randrange(len(r["concat"]))
        r["concat"][i] = _mutate_refs(rng, r["concat"][i])
    elif "alias" in r:
        r["alias"][rng.choice([0, 1])] += rng.choice([-1, 1])
    return r


# --- the worked shapes --------------------------------------------------------------------------------------------


def _scan0(read: bool, where: str) -> str:
    """A root whose one gate is a constant, a scan of no iterations of a pass-through step, and a composite node `F`
    whose argument names that scan's carry: `F` reads it when `read`, else ignores it. `where`: `root` (F called at the
    root), `callee` (in a Definition the root calls: the scan and F's node there), `shared` (the scan's carry as F's
    shared argument in a scan of one iteration) or `batch` (F batched over two members, the scan's carry shared)."""
    P = Program(random.Random(0))
    k = P.prim([], V(8), "K8_v1")
    op = P.prim([V(8)], V(8), "Op8_v1")
    step_b = Body(P, [V(8)], 0)
    step = P.composite(step_b, Tup(V(8), Tup()), {"affine": ["p", 0, 0, 1]}, "Step_v1")
    f_b = Body(P, [V(8)], 10)
    if read:
        f_b.push({"args": [{"refs": {"affine": ["p", 0, 0, 1]}, "type": P.ty(V(8))}], "fn": P.fn(op),
                  "form": "primitive", "out": P.ty(V(8))}, V(8), [1], 1, "prim")
    else:
        f_b.push({"args": [], "fn": P.fn(k), "form": "primitive", "out": P.ty(V(8))}, V(8), [], 1, "prim")
    f = P.composite(f_b, V(8), {"affine": ["n", 0, 0, 1]}, "F_v1")

    def scan0_then(b: Body) -> None:
        b.push({"args": [], "fn": P.fn(k), "form": "primitive", "out": P.ty(V(8))}, V(8), [], 1, "prim")
        b.push({"args": [{"refs": {"affine": ["n", 0, 0, 1]}, "type": P.ty(V(8))}], "fn": P.fn(step), "form": "scan",
                "n": 0, "n_xs": 0, "out": P.ty(Tup(V(8), A(0, Tup())))}, Tup(V(8), A(0, Tup())), [1], 0, "scan")
        carry = {"refs": {"affine": ["n", 1, 0, 1]}, "type": P.ty(V(8))}
        if where in ("root", "callee"):
            b.push({"args": [carry], "fn": P.fn(f), "form": "call", "out": P.ty(V(8))}, V(8), [1], 1, "call")
        elif where == "batch":
            b.push({"args": [carry], "axes": [None], "fn": P.fn(f), "form": "batch", "n": 2,
                    "out": P.ty(A(2, V(8)))}, A(2, V(8)), [1], 2, "batch")
        else:
            sb = Body(P, [V(8), V(8)], 10)
            sb.push({"args": [{"refs": {"affine": ["p", 1, 0, 1]}, "type": P.ty(V(8))}], "fn": P.fn(f), "form": "call",
                     "out": P.ty(V(8))}, V(8), [1], 1, "call")
            s2 = P.composite(sb, Tup(V(8), Tup()), {"affine": ["p", 0, 0, 1]}, "Step2_v1")
            b.push({"args": [{"refs": {"affine": ["n", 0, 0, 1]}, "type": P.ty(V(8))}, carry], "fn": P.fn(s2),
                    "form": "scan", "n": 1, "n_xs": 0, "out": P.ty(Tup(V(8), A(1, Tup())))}, Tup(V(8), A(1, Tup())),
                   [1, 1], 1, "scan")

    root = Body(P, [], 10)
    if where == "callee":
        g_b = Body(P, [], 10)
        scan0_then(g_b)
        g = P.composite(g_b, V(8), {"affine": ["n", 0, 0, 1]}, "G_v1")
        root.push({"args": [], "fn": P.fn(g), "form": "call", "out": P.ty(V(8))}, V(8), [], g_b.gates, "call")
    else:
        scan0_then(root)
    P.composite(root, V(8), {"affine": ["n", 0, 0, 1]}, "R_v1")
    return P.text("R_v1")


def minimal() -> str:
    """`_scan0` shrunk: an input `x`; `s`, a scan of no iterations of `Step_v1` (which passes its carry through and has no
    nodes) from `x`; and a call of `Step_v1` on `s`'s carry. The root returns `x`. `operands` of the call refuses (the
    scan has no output), and no primitive node reads the scan."""
    P = Program(random.Random(0))
    x = P.input(8, False)
    step = P.composite(Body(P, [V(8)], 0), Tup(V(8), Tup()), {"affine": ["p", 0, 0, 1]}, "Step_v1")
    root = Body(P, [], 1)
    root.push({"args": [], "fn": P.fn(x), "form": "primitive", "out": P.ty(V(8))}, V(8), [], 1, "input")
    out = Tup(V(8), A(0, Tup()))
    root.push({"args": [{"refs": {"affine": ["n", 0, 0, 1]}, "type": P.ty(V(8))}], "fn": P.fn(step), "form": "scan",
               "n": 0, "n_xs": 0, "out": P.ty(out)}, out, [1], 0, "scan")
    root.push({"args": [{"refs": {"affine": ["n", 1, 0, 1]}, "type": P.ty(V(8))}], "fn": P.fn(step), "form": "call",
               "out": P.ty(Tup(V(8), Tup()))}, Tup(V(8), Tup()), [1], 0, "call")
    P.composite(root, V(8), {"affine": ["n", 0, 0, 1]}, "R_v1")
    return P.text("R_v1")


def hand() -> list[dict]:
    out = [{"id": "hand-scan0-minimal", "family": "hand", "program": minimal()}]
    for where in ("root", "callee", "shared", "batch"):
        for read in (False, True):
            out.append({"id": f"hand-scan0-{where}-{'read' if read else 'unread'}", "family": "hand",
                        "program": _scan0(read, where)})
    for e in (4, 8, 10, 12, 14):
        out.append({"id": f"hand-passscan-2^{e}", "family": "hand", "program": passscan(2 ** e)})
    return out


# --- batch members returning leaves of shared arguments (the lane's `fuzz/gen/shared.py`) -----------------------------

ASSIGNMENTS = 12
NAMES = ("a", "b", "c")
INPUTS = ("y", "z", "u")

desc = _load("desc_difftest_vectors")
vectors = desc.vectors
reference = desc.reference


@dataclass
class Spec:
    """One declaration: shared argument `NAMES[i]` made by `makers[i]` (`prim`, `call`, `batch` or `input`) from root
    input `INPUTS[i]`, registered where `registered[i]` (a `batch` maker's input an array of `picks[i][0]` values, the
    argument its member `picks[i][1]`); each of `n` members returns `order`'s leaves (`g`: a gate of its own, on the
    mapped `x` where `mapped`, else on `a`); `reader` reads leaves `span` of the members' returns (all of them where
    `None`), and a primitive those of `second`."""
    reader: str
    order: tuple[str, ...]
    makers: tuple[str, ...] = ("prim", "prim")
    registered: tuple[bool, ...] = (True, True)
    picks: tuple[tuple[int, int] | None, ...] = (None, None)
    w: int = 8
    n: int = 2
    mapped: bool = True
    span: tuple[int, int] | None = None
    second: tuple[int, int] | None = None
    tag: str = field(default="")


def decl(s: Spec) -> str:
    w, n, L, k = s.w, s.n, len(s.order), len(s.makers)
    op = vectors.prim((w, w), w)
    tag = s.tag or f"{s.reader}{''.join(s.order)}"
    ret = Tuple(*[Value(w)] * L)
    params = tuple((NAMES[i], Value(w)) for i in range(k)) + ((("x", Value(w)),) if s.mapped else ())
    axes = (None,) * k + ((0,) if s.mapped else ())

    def member(B, S, *xs):
        own = {NAMES[i]: xs[i] for i in range(k)}
        if "g" in s.order:
            g = xs[k] if s.mapped else xs[0]
            own["g"] = B.call(op, g, g)
        return tuple_of(*[own[o] for o in s.order])
    F = bind(CompositeDefinition(f"NciShared{tag}", 1, (), lambda S: (params, ret), member, register=False))

    def batch(B, *xs):
        return B.batch(F, *xs, axes=axes, n=None if s.mapped else n)
    gparams = params[:k] + ((("x", Array(n, Value(w))),) if s.mapped else ())
    G = bind(CompositeDefinition(f"NciSharedBatch{tag}", 1, (), lambda S: (gparams, Array(n, ret)),
                                 lambda B, S, *xs: batch(B, *xs), register=False))
    lo, hi = s.span or (0, L * n)
    total = vectors.summing(hi - lo, w, w)
    R = bind(CompositeDefinition(f"NciSharedRead{tag}", 1, (), lambda S: ((("v", Array(hi - lo, Value(w))),), Value(w)),
                                 lambda B, S, v: B.call(total, v), register=False))
    Pair = bind(CompositeDefinition(f"NciSharedPair{tag}", 1, (), lambda S: ((("y", Value(w)),), Value(w)),
                                    lambda B, S, y: B.call(op, B.call(op, y, y), y), register=False))

    def made(B, i, y):
        how = s.makers[i]
        if how == "prim":
            return B.call(op, y, y)
        if how == "call":
            return B.call(Pair, y)
        if how == "batch":
            return B.batch(op, y, y, axes=(0, 0))[s.picks[i][1]]
        return y

    def body(B, S, *ins):
        shared = [made(B, i, ins[i]) for i in range(k)]
        xs = shared + ([ins[k]] if s.mapped else [])
        out = B.call(G, *xs) if s.reader.startswith("wrapped") else batch(B, *xs)
        r = out.refs

        def run(a, b):
            return Coll(Array(b - a, Value(w)), Affine(r.space, r.idx, r.base + a, b - a))
        first = B.call(R, run(lo, hi)) if s.reader.endswith("composite") else B.call(total, run(lo, hi))
        if s.second:
            B.call(vectors.summing(s.second[1] - s.second[0], w, w), run(*s.second))
        return first
    root = tuple((INPUTS[i], Array(s.picks[i][0], Value(w)) if s.makers[i] == "batch" else Value(w)) for i in range(k))
    root += (("x", Array(n, Value(w))),) if s.mapped else ()
    fn = bind(CompositeDefinition(f"NciSharedDecl{tag}", 1, (), lambda S: (root, Value(w)), body, register=False))
    reg = tuple(INPUTS[i] for i in range(k) if s.registered[i])
    return codec.canonical_json(codec.encode_program(ir.Program(fn, registered=reg))).decode()


def _run(rng: random.Random, N: int, L: int) -> tuple[int, int]:
    """Leaves `[lo, hi)` of `N`, `L` per member: every member, a run of whole members, or a run cut inside them."""
    r = rng.random()
    if r < 0.4:
        return 0, N
    if r < 0.75:
        m0 = rng.randrange(N // L)
        return m0 * L, rng.randint(m0 + 1, N // L) * L
    lo = rng.randrange(N)
    return lo, rng.randint(lo + 1, N)


def draw(rng: random.Random, seed: int) -> Spec:
    k = rng.choice([2, 2, 3])
    makers = tuple(rng.choice(["prim", "prim", "call", "batch", "input"]) for _ in range(k))
    picks = tuple((m := rng.choice([2, 3]), rng.randrange(m)) if how == "batch" else None for how in makers)
    pool = list(NAMES[:k]) + (["g"] if rng.random() < 0.2 else [])
    while True:
        order = tuple(rng.choice(pool) for _ in range(rng.randint(2, 5)))
        if len({o for o in order if o != "g"}) >= 2:
            break
    n = rng.choice([2, 2, 3, 4])
    N = n * len(order)
    return Spec(reader=rng.choice(["primitive", "composite", "wrapped", "wrappedcomposite"]), order=order,
                makers=makers, registered=tuple(rng.random() < 0.75 for _ in range(k)), picks=picks,
                w=rng.choice([8, 8, 1, 3, 16]), n=n, mapped=rng.random() < 0.7, span=_run(rng, N, len(order)),
                second=_run(rng, N, len(order)) if rng.random() < 0.25 else None, tag=f"V{seed}")


def _partitions(c: int):
    """Every assignment of `c` nodes to IUs numbered from 0 in order of first use."""
    if c == 0:
        yield ()
        return
    for p in _partitions(c - 1):
        for u in range(max(p, default=-1) + 2):
            yield p + (u,)


def instances(rng: random.Random, text: str) -> list[dict]:
    """Up to `ASSIGNMENTS` IU assignments of the root nodes that hold gates (all of them when there are no more, else
    drawn without repeats), θ the registered inputs, `Γ` all the gates and `α` 1."""
    inp, reg, spans = reference.root_nodes(text)
    held = [k for k in range(len(inp)) if not inp[k] and spans[k][1] > spans[k][0]]
    every = list(_partitions(len(held)))
    chosen = every if len(every) <= ASSIGNMENTS else rng.sample(every, ASSIGNMENTS)
    theta = [k for k in range(len(inp)) if inp[k] and reg[k]]
    out = []
    for p in chosen:
        owner: list = ["input"] * len(inp)
        for k, u in zip(held, p, strict=True):
            owner[k] = u
        out.append({"theta": theta, "owner": owner, "gamma": spans[-1][1] if spans else 0, "alpha": [1, 1]})
    return out


def seeds() -> list[dict]:
    """`shared()`'s six declarations, each with its two instances, as `decl` writes them."""
    given: dict[str, list[dict]] = {}
    for inst in desc.shared():
        given.setdefault(inst["program"], []).append({k: v for k, v in inst.items() if k != "program"})
    out = []
    for reader in ("primitive", "composite", "wrapped"):
        for order in (("a", "b"), ("b", "b", "a")):
            text = decl(Spec(reader=reader, order=order))
            if text != desc.shared_decl(reader, order):
                raise SystemExit(f"desc_fuzz_programs: decl({reader}, {order}) is not shared_decl's declaration")
            out.append({"id": f"shared-seed-{reader}{''.join(order)}", "family": "shared-seed", "program": text,
                        "instances": given.pop(text)})
    if given:
        raise SystemExit(f"desc_fuzz_programs: {len(given)} of shared()'s declarations are not shared_decl's")
    return out


def shared(seed: int) -> tuple[str, list[dict]]:
    """Seed `seed`'s declaration, `decl` on parameters drawn at random (`draw`), and its IU assignments (`instances`)."""
    rng = random.Random(seed)
    text = decl(draw(rng, seed))
    return text, instances(rng, text)


# --- the data ----------------------------------------------------------------------------------------------------------

def _bare(make):
    return lambda s: (make(s), [])


#: (family, (Program, instances) of a seed, seeds, probability of a mutant, instances of its own `agree` checks): each
#: instance costs two decodes of the Program, most of the run's time, and a shared seed comes with six on average.
FAMILIES = (
    ("shared", shared, range(0, 200), 0.3, 1),
    ("random", _bare(program), range(0, 600), 0.3, 1),
    ("copies", _bare(copies), range(0, 150), 0.2, 1),
    ("chain", _bare(chain), range(0, 20), 0.2, 1),
    ("big", _bare(lambda s: program(s, scale=3)), range(0, 40), 0.3, 1),
)


def _line(id_: str, family: str, text: str, given: list[dict], k: int) -> dict:
    return {"id": id_, "family": family, "program": text, "agree": k, **({"instances": given} if given else {})}


def programs():
    """Every line of the data, in order: `hand()`, `seeds()`, then `FAMILIES`, each seed's Program and its mutant where
    there is one, with the same instances."""
    yield from ({**x, "agree": 4} for x in hand())
    for x in seeds():
        yield _line(x["id"], x["family"], x["program"], x["instances"], 4)
    for fam, make, seeds_, mut, k in FAMILIES:
        for s in seeds_:
            text, given = make(s)
            yield _line(f"{fam}-{s}", fam, text, given, k)
            rng = random.Random(~s)
            if rng.random() < mut:
                yield _line(f"{fam}-mutant-{s}", f"{fam}-mutant", mutate(rng, text), given, k)


def write(out: Path) -> dict[str, Path]:
    """`programs()` under ``out/desc_fuzz/programs.jsonl``."""
    path = out / "desc_fuzz" / "programs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for x in programs():
            f.write(json.dumps(x, separators=(",", ":"), sort_keys=True) + "\n")
    return {"desc_fuzz/programs.jsonl": path}


def count() -> int:
    return sum(1 for _ in programs())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.ml.circuits builds the shared vectors")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    here = Path(verity.ml.circuits.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.ml.circuits comes from {here}, not from the checkout {a.verity}")
    write(a.out)


if __name__ == "__main__":
    main()
