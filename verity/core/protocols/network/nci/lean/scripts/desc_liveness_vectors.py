"""The liveness cases of the description-sized check, for `NciCheck/DescLivenessMain.lean`.

    python scripts/desc_liveness_vectors.py <verity checkout> --out DIR

Writes `DIR/desc_liveness/cases.jsonl`, one case per line: `{"id", "mode", "small", "big"}`, two instances of
`NciCheck.Desc` (θ and the owners per root node) on one Program shape, with `n` (a scan's iterations, a batch's members)
small and past anything walked one by one (2^22 to 2^52). `NciCheck/DescLivenessMain.lean` fails unless
`NciCheck.Desc.check` gives on `small` what `NciCheck.check` gives on it expanded gate by gate, and on `big` a verdict
equal to `small`'s (`mode` `same`) or with the same novel reads per IU (`novel`, where the gates grow with `n`).

The shapes are the fuzzer's, on which `NciCheck.Desc` once
overflowed its stack, ran for a time linear in `n` or ran out of memory, and the closed forms that replaced those walks:

* `passscan`: a gate reading the carry of a scan of `n` iterations of a step that passes its carry through
  (`hand-passscan`, 983 bytes at 2^22: a stack overflow);
* `unread`: the same scan, which nothing reads (2^52 in 814 bytes: about a year);
* `batch-ret`, `batch-gate`, `batch-mapped`: a batch of `n` members of `W`, which passes its argument through, sharing
  the input, read whole by a Definition's return (`batch-ret`, a gate then reading two leaves of its call), by a gate
  (`batch-gate`), or through a second batch of `W` mapped over it (`batch-mapped`) (2^40: out of memory);
* `swap-odd`, `swap-even`: a carry of two leaves the step swaps, read after `n` iterations (the cycle's parity);
* `exit-xs`: a carry of three leaves shifted one place an iteration, the last filled from the iteration's `xs` row;
* `carry-read`, `carry-xs`, `xs-part`: steps with a gate reading their carry (whose leaves pass through to a cycle, or come
  from the iteration before's `xs` row) or part of their `xs` row, every iteration's read at once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "verity-ir/descriptor/v1"


def V(w):
    return ["Value", w]


def A(n, e):
    return ["Array", n, e]


def Tup(*fs):
    return ["Tuple", list(fs)]


def leaves(t) -> int:
    if t[0] == "Value":
        return 1
    if t[0] == "Array":
        return t[1] * leaves(t[2])
    return sum(leaves(f) for f in t[1])


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


def finish(P: Program, root: Body, ret_node: int, theta: tuple[int, ...] = ()) -> dict:
    """The instance: every other root node in IU 0, and θ the inputs `theta`, which a read then doesn't count."""
    P.composite("R_v1", root, V(8), aff("n", ret_node, 0, 1))
    owner = ["input" if k in root.inputs else 0 for k in range(len(root.nodes))]
    return {"program": P.text(), "theta": list(theta), "owner": owner, "gamma": 10, "alpha": [8, 1]}


def passscan(n: int, read: bool = True) -> dict:
    """The `passscan` shape: an input, a scan of `n` iterations of `Step_v1` (which passes its carry through) from it,
    and a gate reading the scan's carry (`read`) or the input."""
    P = Program()
    root = Body(P, [])
    x = root.input(8)
    op = P.prim("Op8_v1", [V(8)], V(8))
    step = P.composite("Step_v1", Body(P, [V(8)]), Tup(V(8), Tup()), aff("p", 0, 0, 1))
    s = root.node("scan", step, [root.arg(aff("n", x, 0, 1), V(8))], Tup(V(8), A(n, Tup())), n, n_xs=0)
    g = root.node("primitive", op, [root.arg(aff("n", s if read else x, 0, 1), V(8))], V(8))
    return finish(P, root, g)


def passbatch(n: int, how: str) -> dict:
    P = Program()
    root = Body(P, [])
    x = root.input(8)
    w = P.composite("W_v1", Body(P, [V(8)]), V(8), aff("p", 0, 0, 1))
    if how == "ret":
        db = Body(P, [V(8)])
        db.node("batch", w, [db.arg(aff("p", 0, 0, 1), V(8))], A(n, V(8)), n, axes=[None])
        d = P.composite("D_v1", db, A(n, V(8)), aff("n", 0, 0, n))
        c = root.node("call", d, [root.arg(aff("n", x, 0, 1), V(8))], A(n, V(8)))
        op2 = P.prim("Op88_v1", [V(8), V(8)], V(8))
        g = root.node("primitive", op2, [root.arg(aff("n", c, 1, 1), V(8)), root.arg(aff("n", c, n - 1, 1), V(8))],
                      V(8))
        return finish(P, root, g)
    b = root.node("batch", w, [root.arg(aff("n", x, 0, 1), V(8))], A(n, V(8)), n, axes=[None])
    if how == "mapped":
        b = root.node("batch", w, [root.arg(aff("n", b, 0, n), A(n, V(8)))], A(n, V(8)), n, axes=[0])
    opn = P.prim("OpN_v1", [A(n, V(8))], V(8))
    g = root.node("primitive", opn, [root.arg(aff("n", b, 0, n), A(n, V(8)))], V(8))
    return finish(P, root, g)


def swap(n: int) -> dict:
    """A carry of two leaves from inputs `a` and `b` (in θ), the step returning it swapped; a gate reads the final
    carry's leaf 0, `a` after an even number of iterations (8 novel bits) and `b` after an odd number (none)."""
    P = Program()
    root = Body(P, [])
    xa, xb = root.input(8), root.input(8)
    ct = Tup(V(8), V(8))
    step = P.composite("Swap_v1", Body(P, [ct]), Tup(ct, Tup()), {"explicit": [["p", 0, 1], ["p", 0, 0]]})
    init = {"concat": [aff("n", xa, 0, 1), aff("n", xb, 0, 1)]}
    s = root.node("scan", step, [root.arg(init, ct)], Tup(ct, A(n, Tup())), n, n_xs=0)
    g = root.node("primitive", P.prim("Op8_v1", [V(8)], V(8)), [root.arg(aff("n", s, 0, 1), V(8))], V(8))
    return finish(P, root, g, (xb,))


def exit_xs(n: int) -> dict:
    """A carry of three leaves from three inputs, shifted down one place an iteration, the last filled from the
    iteration's `xs` row, a batch of `n` members of `W` sharing a fourth input, in θ; a gate reads the final carry's
    leaf 0: the `xs` row of iteration `n - 3` (no novel bits), or for `n < 3` the initial carry's leaf `n` (8)."""
    P = Program()
    root = Body(P, [])
    xs = [root.input(8) for _ in range(4)]
    w = P.composite("W_v1", Body(P, [V(8)]), V(8), aff("p", 0, 0, 1))
    b = root.node("batch", w, [root.arg(aff("n", xs[3], 0, 1), V(8))], A(n, V(8)), n, axes=[None])
    ct = Tup(V(8), V(8), V(8))
    step = P.composite("Shift_v1", Body(P, [ct, V(8)]), Tup(ct, Tup()),
                       {"explicit": [["p", 0, 1], ["p", 0, 2], ["p", 1, 0]]})
    init = {"concat": [aff("n", xs[0], 0, 1), aff("n", xs[1], 0, 1), aff("n", xs[2], 0, 1)]}
    s = root.node("scan", step, [root.arg(init, ct), root.arg(aff("n", b, 0, n), A(n, V(8)))],
                  Tup(ct, A(n, Tup())), n, n_xs=1)
    g = root.node("primitive", P.prim("Op8_v1", [V(8)], V(8)), [root.arg(aff("n", s, 0, 1), V(8))], V(8))
    return finish(P, root, g, (xs[3],))


def carry_read(n: int, xs: bool) -> dict:
    """A step with a gate reading its carry's leaf 0, the first iteration's an input in θ. Without `xs`: a carry of two
    leaves from inputs `a` (in θ) and `b` that the step returns as `(c[1], c[1])`, so every iteration after the first
    reads `b`; with `xs`: a carry of one leaf from `a` the step returns from its `xs` row, a batch of `n` members of `W`
    sharing `b`. Either way, 8 novel bits, which only the iterations after the first read."""
    P = Program()
    root = Body(P, [])
    op = P.prim("Op8_v1", [V(8)], V(8))
    xa, xb = root.input(8), root.input(8)
    if not xs:
        ct = Tup(V(8), V(8))
        sb = Body(P, [ct])
        sb.node("primitive", op, [sb.arg(aff("p", 0, 0, 1), V(8))], V(8))
        step = P.composite("Step_v1", sb, Tup(ct, Tup()), {"explicit": [["p", 0, 1], ["p", 0, 1]]})
        init = {"concat": [aff("n", xa, 0, 1), aff("n", xb, 0, 1)]}
        s = root.node("scan", step, [root.arg(init, ct)], Tup(ct, A(n, Tup())), n, n_xs=0)
    else:
        w = P.composite("W_v1", Body(P, [V(8)]), V(8), aff("p", 0, 0, 1))
        b = root.node("batch", w, [root.arg(aff("n", xb, 0, 1), V(8))], A(n, V(8)), n, axes=[None])
        sb = Body(P, [V(8), V(8)])
        sb.node("primitive", op, [sb.arg(aff("p", 0, 0, 1), V(8))], V(8))
        step = P.composite("Step_v1", sb, Tup(V(8), Tup()), aff("p", 1, 0, 1))
        s = root.node("scan", step, [root.arg(aff("n", xa, 0, 1), V(8)), root.arg(aff("n", b, 0, n), A(n, V(8)))],
                      Tup(V(8), A(n, Tup())), n, n_xs=1)
    g = root.node("primitive", op, [root.arg(aff("n", s, 0, 1), V(8))], V(8))
    return finish(P, root, g, (xa,))


def xs_part(n: int) -> dict:
    """A step with a gate reading leaf 1 of its `xs` row `(V8, V8)`, the rows a batch of `n` members of `W2`, which
    returns its first input, in θ, and its second passed through `W`, both shared: 8 novel bits from leaf 1, none from
    leaf 0, and 8 from the initial carry, which the last gate reads."""
    P = Program()
    root = Body(P, [])
    x8, xa, xb = root.input(8), root.input(8), root.input(8)
    op = P.prim("Op8_v1", [V(8)], V(8))
    w = P.composite("W_v1", Body(P, [V(8)]), V(8), aff("p", 0, 0, 1))
    w2b = Body(P, [V(8), V(8)])
    w2b.node("call", w, [w2b.arg(aff("p", 1, 0, 1), V(8))], V(8))
    row = Tup(V(8), V(8))
    w2 = P.composite("W2_v1", w2b, row, {"concat": [aff("p", 0, 0, 1), aff("n", 0, 0, 1)]})
    b = root.node("batch", w2, [root.arg(aff("n", xa, 0, 1), V(8)), root.arg(aff("n", xb, 0, 1), V(8))], A(n, row), n,
                  axes=[None, None])
    sb = Body(P, [V(8), row])
    sb.node("primitive", op, [sb.arg(aff("p", 1, 1, 1), V(8))], V(8))
    step = P.composite("Step_v1", sb, Tup(V(8), Tup()), aff("p", 0, 0, 1))
    s = root.node("scan", step, [root.arg(aff("n", x8, 0, 1), V(8)), root.arg(aff("n", b, 0, 2 * n), A(n, row))],
                  Tup(V(8), A(n, Tup())), n, n_xs=1)
    g = root.node("primitive", op, [root.arg(aff("n", s, 0, 1), V(8))], V(8))
    return finish(P, root, g, (xa,))


#: (id, mode, the instance at n, small n, big n)
CASES = [
    ("passscan-2^22", "same", passscan, 16, 2 ** 22),
    ("passscan-2^52", "same", passscan, 16, 2 ** 52),
    ("unread-2^52", "same", lambda n: passscan(n, read=False), 16, 2 ** 52),
    ("batch-ret-2^40", "same", lambda n: passbatch(n, "ret"), 16, 2 ** 40),
    ("batch-gate-2^40", "same", lambda n: passbatch(n, "gate"), 16, 2 ** 40),
    ("batch-mapped-2^40", "same", lambda n: passbatch(n, "mapped"), 16, 2 ** 40),
    ("swap-odd-2^52", "same", swap, 5, 2 ** 52 + 1),
    ("swap-even-2^52", "same", swap, 6, 2 ** 52),
    ("exit-xs-2^40", "same", exit_xs, 5, 2 ** 40),
    ("exit-xs-short", "same", exit_xs, 2, 2),
    ("carry-read-2^40", "novel", lambda n: carry_read(n, xs=False), 6, 2 ** 40),
    ("carry-xs-2^40", "novel", lambda n: carry_read(n, xs=True), 6, 2 ** 40),
    ("xs-part-2^40", "novel", xs_part, 6, 2 ** 40),
]


def cases() -> list[dict]:
    return [{"id": i, "mode": mode, "small": make(s), "big": make(b)} for i, mode, make, s, b in CASES]


def write(out: Path) -> dict[str, Path]:
    d = out / "desc_liveness"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "cases.jsonl"
    path.write_text("".join(json.dumps(c, separators=(",", ":"), sort_keys=True) + "\n" for c in cases()))
    return {"desc_liveness/cases.jsonl": path}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    for name, p in write(a.out).items():
        print(name, p)


if __name__ == "__main__":
    main()
