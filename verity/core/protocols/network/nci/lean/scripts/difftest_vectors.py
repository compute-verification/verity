"""The difftest's data: random declarations and divisions, and the Python reference's results on each, for
`NciCheck/DifftestMain.lean`.

    python scripts/difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/difftest/instances.jsonl` (N random instances, then `pinned()`'s), one instance per line as `NciCheck.check`
reads it (`NciCheck.Check`'s docstring: `program`, `theta`, `owner`, `gamma`, `alpha`), and `DIR/difftest/python.jsonl`,
the same line's results from
`scripts/nci_reference.py`. The Lean audit runs `NciCheck/DifftestMain.lean` on both files (`runs` in `lean-audit.json`,
which pins their SHA-256), and `verity/core/protocols/network/tests/test_nci_lean_difftest.py` checks that this script
still writes them.

A declaration is a descriptor v1 Program built with `verity.ml.circuits` in one of five shapes: a random DAG, a chain, a
reduction tree, a chain of matrix-vector products (the NCI draft's updated multiplications: products and their sums),
and a nested one whose gates sit in calls, batches (of a primitive and of a composite) and a scan. Its parameters are
values and arrays of widths from 1 to 32 bits, some bound at registration, and its primitives take one to four values
or an array. θ is empty, every input (refused, `theta-not-registered`, where one is bound per run), the registered
inputs, or a random set of them. A division is one IU, one IU per gate, runs of consecutive gates or a random
assignment, numbered in a random order; `Γ` is at, below or above the division's largest IU, and `α` at, just below or
just above its largest NCI, written reduced or not. About a third of the instances are malformed, each in one way the
check refuses (every code of `NciCheck.Check`). Then come `mistyped_vectors.cases()`'s (`mistyped()`): references
that name leaves of other widths than their types give them, which the codec and the check refuse as `program`, and
their well-typed neighbours. The data ends with `pinned()`'s two instances on the declaration worked by hand (`hand()`):
θ every input, which the check must refuse, and θ the registered input, which it must accept.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import random
from fractions import Fraction
from pathlib import Path

import verity.ml.circuits
from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load("nci_reference")

COUNT = 4000
WIDTHS = [1, 2, 3, 8, 16, 32]
SHAPES = ["dag", "chain", "tree", "matmul", "nested"]
MALFORMED = ["instance", "params", "program", "not-topological", "theta-unknown", "theta-not-input",
             "theta-not-registered", "theta-duplicate", "owner-length", "owner-entry", "input-in-iu", "gate-in-no-iu",
             "empty-iu"]

_PRIMS: dict[tuple, PrimitiveDefinition] = {}


def _zero(*_a):
    return 0


def prim(ins: tuple[int, ...], r: int) -> PrimitiveDefinition:
    """A primitive reading values of widths `ins` and returning `Value<r>`."""
    key = ("op", ins, r)
    if key not in _PRIMS:
        name = f"NciOp{'x'.join(map(str, ins))}r{r}" if ins else f"NciConst{r}"
        _PRIMS[key] = PrimitiveDefinition(name, 1, tuple((f"a{i}", Value(w)) for i, w in enumerate(ins)), Value(r), _zero,
                                          register=False)
    return _PRIMS[key]


def summing(k: int, w: int, r: int) -> PrimitiveDefinition:
    """A primitive reading one `Array<k, Value<w>>`."""
    key = ("sum", k, w, r)
    if key not in _PRIMS:
        _PRIMS[key] = PrimitiveDefinition(f"NciSum{k}w{w}r{r}", 1, (("x", Array(k, Value(w))),), Value(r), _zero,
                                          register=False)
    return _PRIMS[key]


class Body:
    """A body as data, replayed by `build` each time the Builder asks for it: its parameters' types, then steps over
    handles (each parameter leaf, then each value a step produces), and the handles it returns."""

    def __init__(self, params: list) -> None:
        self.params = params
        self.widths = [w for t in params for w in t.leaf_widths()]
        self.steps: list[tuple] = []

    def add(self, step: tuple, widths: list[int]) -> list[int]:
        self.steps.append(step)
        first = len(self.widths)
        self.widths += widths
        return list(range(first, len(self.widths)))

    def build(self, B, colls, ret: list[int]):
        hs = []
        for c in colls:
            hs += [c] if isinstance(c.type, Value) else [c[i] for i in range(c.type.n)]
        for step in self.steps:
            kind = step[0]
            if kind == "call":
                hs.append(B.call(step[1], *[hs[i] for i in step[2]]))
            elif kind == "sum":
                hs.append(B.call(step[1], array_of([hs[i] for i in step[2]])))
            elif kind == "batch":
                out = B.batch(step[1], colls[step[2]], axes=(0,))
                hs += [out[i] for i in range(out.type.n)]
            else:
                out = B.scan(step[1], hs[step[2]], xs=[colls[step[3]]], shared=[hs[step[4]]])
                hs += [out[0]] + [out[1][i] for i in range(out.type.fields[1].n)]
        return hs[ret[0]] if len(ret) == 1 else tuple_of(*[hs[i] for i in ret])

    def ret_type(self, ret: list[int]):
        return Value(self.widths[ret[0]]) if len(ret) == 1 else Tuple(*[Value(self.widths[i]) for i in ret])


def composite(name: str, body: Body, ret: list[int]):
    params = tuple((f"p{i}", t) for i, t in enumerate(body.params))
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, body.ret_type(ret)),
                                    lambda B, S, *colls: body.build(B, colls, ret), register=False))


def _op(rng, body: Body, srcs: list[int], k: int | None = None, exact: list[int] | None = None) -> list[int]:
    """One primitive over the handles `exact`, or over `k` (else one to four) drawn from `srcs`, or now and then a
    constant."""
    if exact is None and (not srcs or rng.random() < 0.04):
        return body.add(("call", prim((), r := rng.choice(WIDTHS)), []), [r])
    ops = exact if exact is not None else [rng.choice(srcs) for _ in range(k or rng.choices([1, 2, 3, 4], [3, 6, 3, 1])[0])]
    r = rng.choice(WIDTHS)
    return body.add(("call", prim(tuple(body.widths[i] for i in ops), r), ops), [r])


def _params(rng, n: int) -> list:
    return [Value(rng.choice(WIDTHS)) if rng.random() < 0.6 else Array(rng.randint(2, 4), Value(rng.choice(WIDTHS)))
            for _ in range(n)]


def _ret(rng, body: Body, made: list[int]) -> list[int]:
    pool = made or list(range(len(body.widths)))
    return sorted(set(rng.sample(pool, min(len(pool), rng.randint(1, 3)))) | {pool[-1]})


def shape(rng, kind: str, seed: int) -> tuple[Body, list[int]]:
    """A root body of shape `kind` and the handles it returns."""
    if kind == "matmul":
        L, d = rng.randint(1, 2), rng.randint(1, 3)
        bw, ba = rng.choice(WIDTHS), rng.choice(WIDTHS)
        dims = [d] + [rng.randint(1, 3) for _ in range(L)]
        body = Body([Array(dims[1] * dims[0], Value(bw)), Array(d, Value(ba))]
                    + ([Array(dims[2] * dims[1], Value(bw))] if L > 1 else []))
        weights = [list(range(dims[0] * dims[1])), list(range(dims[0] * dims[1] + d, len(body.widths)))]
        x = list(range(dims[0] * dims[1], dims[0] * dims[1] + d))
        made = []
        for ell in range(L):
            e = dims[ell + 1]
            ys = []
            for i in range(e):
                prods = [body.add(("call", prim((bw, body.widths[x[j]]), ba), [weights[ell][i * len(x) + j], x[j]]), [ba])[0]
                         for j in range(len(x))]
                acc = prods[0]
                for p in prods[1:]:
                    acc = body.add(("call", prim((ba, ba), ba), [acc, p]), [ba])[0]
                ys.append(acc)
            made += ys
            x = ys if ell + 1 == L else [body.add(("call", prim((ba,), ba), [y]), [ba])[0] for y in ys]
            made += x if ell + 1 < L else []
        return body, _ret(rng, body, made)
    body = Body(_params(rng, rng.randint(1, 4)))
    inputs = list(range(len(body.widths)))
    if rng.random() < 0.02:
        return body, _ret(rng, body, [])
    made: list[int] = []
    if kind == "tree":
        layer = inputs + [h for _ in range(rng.randint(0, 3)) for h in _op(rng, body, inputs, 1)]
        while len(layer) > 1:
            nxt = []
            for i in range(0, len(layer) - 1, 2):
                nxt += _op(rng, body, [], exact=layer[i:i + 2])
            layer = nxt + ([layer[-1]] if len(layer) % 2 else [])
            made += nxt
        return body, _ret(rng, body, made or _op(rng, body, inputs))
    n = rng.randint(1, 18)
    for _ in range(n):
        if kind == "chain":
            prev = made[-1:] or inputs[-1:]
            srcs = prev + ([rng.choice(inputs)] if rng.random() < 0.5 else [])
            made += _op(rng, body, [], exact=srcs)
        elif kind == "nested" and rng.random() < 0.35:
            made += _nested(rng, body, inputs + made, seed)
        elif rng.random() < 0.1:
            k = rng.randint(2, 4)
            srcs = [rng.choice(inputs + made) for _ in range(k)]
            w = body.widths[srcs[0]]
            srcs = [s for s in srcs if body.widths[s] == w]
            r = rng.choice(WIDTHS)
            made += body.add(("sum", summing(len(srcs), w, r), srcs), [r])
        else:
            made += _op(rng, body, inputs + made)
    return body, _ret(rng, body, made)


def _nested(rng, body: Body, srcs: list[int], seed: int) -> list[int]:
    """Gates inside a call, a batch over an array parameter (of a primitive or a composite) or a scan over one."""
    arrays = [i for i, t in enumerate(body.params) if isinstance(t, Array)]
    tag = f"{seed}n{len(body.steps)}"
    pick = rng.choice(["call", "batch", "scan"] if arrays else ["call"])
    if pick == "call":
        args = [rng.choice(srcs) for _ in range(rng.randint(1, 3))]
        sub = Body([Value(body.widths[a]) for a in args])
        for _ in range(rng.randint(1, 4)):
            _op(rng, sub, list(range(len(sub.widths))))
        fn = composite(f"NciSub{tag}", sub, [len(sub.widths) - 1])
        return body.add(("call", fn, args), [sub.widths[-1]])
    a = rng.choice(arrays)
    t = body.params[a]
    w = t.elem.w
    if pick == "batch":
        if rng.random() < 0.5:
            fn = prim((w,), r := rng.choice(WIDTHS))
        else:
            sub = Body([Value(w)])
            for _ in range(rng.randint(1, 3)):
                _op(rng, sub, list(range(len(sub.widths))))
            fn, r = composite(f"NciMember{tag}", sub, [len(sub.widths) - 1]), sub.widths[-1]
        return body.add(("batch", fn, a), [r] * t.n)
    init, shared = rng.choice(srcs), rng.choice(srcs)
    wc, ws, wy = body.widths[init], body.widths[shared], rng.choice(WIDTHS)
    step = Body([Value(wc), Value(w), Value(ws)])
    c2 = step.add(("call", prim((wc, w), wc), [0, 1]), [wc])[0]
    y = step.add(("call", prim((wc, ws), wy), [c2, 2]), [wy])[0]
    return body.add(("scan", composite(f"NciStep{tag}", step, [c2, y]), init, a, shared), [wc] + [wy] * t.n)


def program(rng, seed: int) -> tuple[str, list[str], list[str]]:
    """A declaration's Program as canonical descriptor text, its shape, and the names of its registered parameters."""
    kind = rng.choice(SHAPES)
    body, ret = shape(rng, kind, seed)
    names = [f"p{i}" for i in range(len(body.params))]
    fn = composite(f"NciDecl{seed}", body, ret)
    registered = [n for n in names if rng.random() < 0.3]
    desc = codec.encode_program(Program(fn, registered=tuple(registered)))
    return codec.canonical_json(desc).decode(), kind, registered


def _dense(rng, labels: list[int]) -> list[int]:
    """The labels renumbered 0 … m − 1, in a random order or in order of first use."""
    used = sorted(set(labels))
    order = used[:] if rng.random() < 0.4 else rng.sample(used, len(used))
    to = {u: i for i, u in enumerate(order)}
    return [to[x] for x in labels]


def division(rng, computed: list[int]) -> tuple[str, list[int]]:
    """A valid division of the computed gates: (strategy, IU of each computed gate)."""
    n = len(computed)
    how = rng.choice(["single", "singletons", "runs", "random", "random"])
    if how == "single" or n == 0:
        return how, [0] * n
    if how == "singletons":
        return how, _dense(rng, list(range(n)))
    if how == "runs":
        labels, u = [], 0
        while len(labels) < n:
            labels += [u] * rng.randint(1, max(1, n // 2))
            u += 1
        return how, _dense(rng, labels[:n])
    return how, _dense(rng, [rng.randrange(rng.randint(1, n)) for _ in range(n)])


def _alpha(rng, crit: Fraction) -> list[int]:
    """`α` at, just below or just above `crit`, or 0, as `[num, den]`, reduced or scaled."""
    k = rng.choice([1, 1, 2, 3])
    p, q = crit.numerator * k, crit.denominator * k
    how = rng.choices(["at", "below", "above", "zero", "far"], [4, 3, 3, 1, 1])[0]
    if how == "below" and p > 0:
        return [p - 1, q]
    if how == "above":
        return [p + 1, q]
    if how == "zero":
        return [0, rng.randint(1, 3)]
    if how == "far":
        return [rng.randint(0, 200), rng.randint(1, 5)]
    return [p, q]


def _forward(desc: dict) -> bool:
    """A root node's single-leaf argument redirected to the node itself: the gate reads itself (`not-topological`)."""
    nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
    for k in range(len(nodes) - 1, -1, -1):
        if nodes[k]["form"] != "primitive":
            continue
        for arg in nodes[k]["args"]:
            refs = arg["refs"]
            if "affine" in refs and refs["affine"][3] == 1:
                arg["refs"] = {"affine": ["n", k, 0, 1]}
                return True
    return False


def instance(seed: int) -> dict:
    """Instance `seed`, with what it was made from (`shape`, `theta_kind`, `division`, `malformed`)."""
    rng = random.Random(seed)
    text, kind, _ = program(rng, seed)
    w, inp, reg, _ = reference.circuit(text)
    N = len(w)
    inputs = [g for g in range(N) if inp[g]]
    computed = [g for g in range(N) if not inp[g]]
    reg_gates = [g for g in range(N) if reg[g]]
    theta_kind = rng.choice(["none", "all", "registered", "random"])
    theta = {"none": [], "all": inputs, "registered": reg_gates,
             "random": [g for g in reg_gates if rng.random() < 0.5]}[theta_kind]
    theta = rng.sample(theta, len(theta))
    how, labels = division(rng, computed)
    owner: list = ["input"] * N
    for g, u in zip(computed, labels, strict=True):
        owner[g] = u
    inst = {"program": text, "theta": theta, "owner": owner, "gamma": 0, "alpha": [0, 1]}
    probe = reference.check(inst | {"theta": [g for g in theta if reg[g]], "gamma": N, "alpha": [sum(w) + 1, 1]})
    ius = probe["ius"]
    biggest = max((s for s, _ in ius), default=0)
    crit = max((Fraction(r, s) for s, r in ius), default=Fraction(0))
    inst["gamma"] = max(0, biggest + rng.choices([0, -1, 1, rng.randint(-3, 3)], [5, 3, 2, 1])[0])
    inst["alpha"] = _alpha(rng, crit)
    bad = rng.choice(MALFORMED) if rng.random() < 0.35 else None
    if bad:
        _break(rng, inst, bad, inp, reg)
    return {"instance": inst, "shape": kind, "theta_kind": theta_kind, "division": how, "malformed": bad}


def _break(rng, inst: dict, bad: str, inp: list[bool], reg: list[bool]) -> None:
    """Make `inst` malformed in the way `bad` names."""
    N = len(inp)
    owner, theta = inst["owner"], inst["theta"]
    computed = [g for g in range(N) if not inp[g]]
    inputs = [g for g in range(N) if inp[g]]
    registered = [g for g in inputs if reg[g]]
    per_run = [g for g in inputs if not reg[g]]
    if bad == "instance":
        how = rng.randrange(4)
        if how == 0:
            del inst[rng.choice(reference.FIELDS)]
        elif how == 1:
            inst["program"] = json.loads(inst["program"])
        else:
            inst[rng.choice(["theta", "owner"])] = rng.choice([{}, 3, "0"])
    elif bad == "params":
        if rng.random() < 0.4:
            inst["gamma"] = rng.choice([-1, 1.5, "2", None, True])
        else:
            inst["alpha"] = rng.choice([[1, 0], [-1, 2], [1], [1, 2, 3], "1/2", 0.5, [0.5, 1], [True, 1]])
    elif bad == "program":
        inst["program"] = rng.choice(["", "{", "[]", '{"schema": "verity-ir/descriptor/v0"}',
                                      inst["program"].replace('"root":', '"rooot":', 1)])
    elif bad == "not-topological":
        desc = json.loads(inst["program"])
        if _forward(desc):
            inst["program"] = codec.canonical_json(desc).decode()
    elif bad == "theta-unknown":
        theta.insert(rng.randint(0, len(theta)), rng.choice([N, N + rng.randint(1, 5), -1, "0", None, 1.5]))
    elif bad == "theta-not-input" and computed:
        theta.insert(rng.randint(0, len(theta)), rng.choice(computed))
    elif bad == "theta-not-registered" and per_run:
        theta.insert(rng.randint(0, len(theta)), rng.choice(per_run))
    elif bad == "theta-duplicate" and registered:
        g = rng.choice(theta or registered)
        if not theta:
            theta.append(g)
        theta.insert(rng.randint(0, len(theta)), g)
    elif bad == "owner-length":
        if rng.random() < 0.5 and owner:
            owner.pop(rng.randrange(len(owner)))
        else:
            owner.append(rng.choice(["input", 0]))
    elif bad == "owner-entry":
        owner[rng.randrange(N)] = rng.choice([-1, 1.5, None, "x", True, [0], "Input"])
    elif bad == "input-in-iu" and inputs:
        owner[rng.choice(inputs)] = rng.choice([0, max([o for o in owner if o != "input"], default=0)])
    elif bad == "gate-in-no-iu" and computed:
        owner[rng.choice(computed)] = "input"
    elif bad == "empty-iu" and computed:
        top = max(o for o in owner if o != "input")
        owner[rng.choice(computed)] = rng.choice([top + 2, top + 1 + N, 10 ** 15])


def hand() -> str:
    """A declaration worked by hand: a: Value<8> (registered), b: Value<16> (bound per run); c = f(a, b): Value<4>;
    d = f(c, b): Value<2>; e = f(d, a): Value<1>. Gates, and root nodes, 0 a, 1 b, 2 c, 3 d, 4 e."""
    f1, f2, f3 = prim((8, 16), 4), prim((4, 16), 2), prim((2, 8), 1)

    def body(B, S, a, b):
        c = B.call(f1, a, b)
        d = B.call(f2, c, b)
        return B.call(f3, d, a)
    fn = bind(CompositeDefinition("HandDecl", 1, (), lambda S: ((("a", Value(8)), ("b", Value(16))), Value(1)), body,
                                  register=False))
    return codec.canonical_json(codec.encode_program(Program(fn, registered=("a",)))).decode()


def pinned() -> list[dict]:
    """The two instances the data ends with, on `hand()` divided as {c}, {d, e}, with the result each must get (`expect`):
    θ every input (`theta_kind` "all"), which holds b, bound per run, so it is refused; and θ the registered input
    alone, which saves IU 1 a's 8 bits and gets a verdict. Gate ids are root node ids here, so each is an instance of
    `NciCheck.Desc` too."""
    base = {"program": hand(), "owner": ["input", "input", 0, 1, 1], "gamma": 2, "alpha": [16, 1]}
    return [
        {"instance": base | {"theta": [0, 1]}, "theta_kind": "all", "expect": {"error": "theta-not-registered"}},
        {"instance": base | {"theta": [0]}, "theta_kind": "registered",
         "expect": {"gates": 5, "ius": [[1, 16], [2, 20]], "complies": True, "failure": None}},
    ]


@functools.cache
def mistyped() -> list[dict]:
    """`mistyped_vectors.cases()` as `check` reads them: gate by gate (`nci_reference.expand`) where the Program decodes,
    else as they are, since a Program the decoder refuses is refused before θ and the owners are read."""
    desc = _load("desc_difftest_vectors")
    out = []
    for inst in _load("mistyped_vectors").cases(desc.instance):
        try:
            reference.root_nodes(inst["program"])
        except (ValueError, KeyError, TypeError, AttributeError):
            out.append(inst)
            continue
        out.append(reference.expand(inst))
    return out


def total(count: int = COUNT) -> int:
    """How many instances the data has: `count` random ones, then `mistyped()`'s, then `pinned()`'s."""
    return count + len(mistyped()) + len(pinned())


def python_results(inst: dict) -> dict:
    return reference.check(inst)


def line(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1, then `mistyped()`'s and `pinned()`'s, under ``out/difftest``."""
    d = out / "difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"difftest/instances.jsonl": d / "instances.jsonl", "difftest/python.jsonl": d / "python.jsonl"}
    insts = [*(instance(seed)["instance"] for seed in range(count)), *mistyped(), *(p["instance"] for p in pinned())]
    with open(paths["difftest/instances.jsonl"], "w") as fi, open(paths["difftest/python.jsonl"], "w") as fp:
        for inst in insts:
            fi.write(line(inst))
            fp.write(line(python_results(inst)))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.ml.circuits reads the programs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.ml.circuits.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.ml.circuits comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
