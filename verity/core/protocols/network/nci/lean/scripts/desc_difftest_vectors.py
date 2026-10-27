"""The description-sized check's difftest data: `difftest_vectors.py`'s declarations, divided into IUs of root nodes, for
`NciCheck/DescDifftestMain.lean`.

    python scripts/desc_difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes three files under `DIR/desc_difftest/`, one line per instance: `instances.jsonl`, as `NciCheck.Desc.check` reads
it (`NciCheck.Desc`'s docstring: θ and the owners per root node); `expanded.jsonl`, the same instance expanded gate by
gate as `NciCheck.check` reads it (`nci_reference.expand`), or `null` where `NciCheck.Desc` refuses it; and `python.jsonl`,
`nci_reference.check_desc`'s results. The Lean audit runs `NciCheck/DescDifftestMain.lean` on the three (`runs` in
`lean-audit.json`), which fails unless `NciCheck.Desc.check` gives Python's result on every line and `NciCheck.check`'s on
every expansion: the description-sized check against the unrolled one, on the same declarations.

Instance `seed`'s program, for the first `COUNT`, is one of `difftest_vectors.program`'s or (a quarter of them) one read
across its arrays' rows (`transposed`), then each distinct descriptor
the circuits fixtures hold (`verity/ml/circuits/tests/*_vectors.json`: pass-throughs, scans with carries, batches of
composites, root batches and scans) for `PER_FIXTURE` instances each, then six on a composite keyed with an input id
(`decoys`), which is no input node, then two on a gate reading a nested scan's carry (`walks`), one past
`NciCheck.Desc.walkBound`'s limit, then four on a scan of no iterations (`empties`), three of them refused, its carry a
value, read or not, then twelve on whole members of a batch returning leaves of two arguments they share, each another
node's (`shared`), then `mistyped_vectors.cases()`'s (`mistyped`: references that name leaves of other widths than
their types give them, which both sides refuse as `program`, and their well-typed neighbours), then
`difftest_vectors.pinned()`'s two (θ every
input, refused, and θ the registered one, accepted). θ is no input node, every one (refused, `theta-not-registered`,
where one is bound per run), the registered ones or a random set of them. A division puts every Call in one IU, each in
its own, runs of consecutive Calls or a random assignment, and a Call without gates in any IU or none. `Γ` and `α` are
set as `difftest_vectors` sets them, at, below or above the division's largest IU and NCI. About a third of the
instances are malformed, each in one way `NciCheck.Desc` refuses.
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
from verity.ml.circuits.defs import Builder, CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Strided, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("difftest_vectors")
reference = vectors.reference

COUNT = 3000
PER_FIXTURE = 24
TRANSPOSED = 0.25
FIXTURES = Path(verity.ml.circuits.__file__).resolve().parent / "tests"
MALFORMED = vectors.MALFORMED


def division(rng, calls: list[int]) -> tuple[str, list[int]]:
    """A division of the Calls with gates: (strategy, IU of each)."""
    return vectors.division(rng, calls)


def _descriptors(x):
    """The descriptor texts in a fixture's JSON, in document order."""
    if isinstance(x, str) and x.startswith("{") and '"definitions"' in x:
        d = json.loads(x)
        if isinstance(d, dict) and {"definitions", "root", "callees"} <= d.keys():
            yield x
    elif isinstance(x, dict):
        if {"definitions", "root", "callees"} <= x.keys():
            yield json.dumps(x, separators=(",", ":"))
            return
        for v in x.values():
            yield from _descriptors(v)
    elif isinstance(x, list):
        for v in x:
            yield from _descriptors(v)


@functools.cache
def fixtures() -> list[tuple[str, str]]:
    """Every distinct descriptor the circuits fixtures hold that the decoder reads: (file stem, text)."""
    out: dict[str, str] = {}
    for f in sorted(FIXTURES.glob("*_vectors.json")):
        for text in _descriptors(json.loads(f.read_text())):
            try:
                reference.root_nodes(text)
            except Exception:
                continue
            out.setdefault(text, f.stem)
    return [(name, text) for text, name in out.items()]


def _strided(c: Coll, t, inner: int, outer: int, stride: int) -> Coll:
    """Leaves of `c` (one run of a parameter's or a node's leaves) as `t`, leaf j at `(j // inner)·outer +
    (j % inner)·stride` of `c`."""
    r = c.refs
    return Coll(t, Strided(r.space, r.idx, r.base, inner, outer, stride, t.leaves))


def transposed(rng, seed: int) -> str:
    """A declaration read across its arrays' rows, as served models read weights (`Embedding_v1`'s table): batches mapped
    over a transposed parameter and a transposed batch output, a primitive reading a transposed block whole or in part,
    and a column Definition that reads part of its column and passes leaves of it through, called in a batch and stepped
    in a scan whose carry is such a pass-through."""
    R, C = rng.randint(2, 5), rng.randint(2, 5)
    w, r = rng.choice(vectors.WIDTHS), rng.choice(vectors.WIDTHS)
    lo = rng.randint(0, R - 1)
    hi = rng.randint(lo + 1, R)
    j = rng.randrange(R)
    op, mix = vectors.prim((w, w), w), vectors.prim((w, w), r)
    col_t = Array(R, Value(w))
    k = hi - lo
    col_ret = Tuple(Array(k, Value(r)), Value(w))

    def col(B, S, a, s):
        b = B.batch(mix, a[lo:hi], s, axes=(0, None))
        return tuple_of(b, a[j])
    Col = bind(CompositeDefinition(f"NciT{seed}Col", 1, (), lambda S: ((("a", col_t), ("s", Value(w))), col_ret), col,
                                   register=False))

    def step(B, S, carry, a):
        nxt = a[j] if rng_step else B.call(op, carry, a[lo])
        return tuple_of(nxt, B.call(mix, carry, a[j]))
    rng_step = rng.random() < 0.5
    Step = bind(CompositeDefinition(f"NciT{seed}Step", 1, (), lambda S: ((("c", Value(w)), ("a", col_t)),
                                                                        Tuple(Value(w), Value(r))), step, register=False))
    uses = [u for u in ("batch", "scan", "block", "back") if rng.random() < 0.7] or ["batch"]
    sub = rng.randint(1, C)
    params = (("x", Array(R, Array(C, Value(w)))), ("y", Value(w)))

    def body(B, S, x, y):
        xt = _strided(x, Array(C, col_t), R, 1, C)
        outs = []
        if "batch" in uses or "back" in uses:
            n0 = B.batch(Col, xt, y, axes=(0, None))
            outs.append(n0)
            if "back" in uses:
                view = _strided(n0, Array(k, Array(C, Value(r))), C, 1, k + 1)
                outs.append(B.batch(vectors.summing(C, r, r), view, axes=(0,)))
        if "scan" in uses:
            outs.append(B.scan(Step, y, xs=[xt]))
        if "block" in uses:
            block = _strided(x, Array(R * sub, Value(w)), R, 1, C)
            outs.append(B.call(vectors.summing(R * sub, w, r), block))
        return outs[0] if len(outs) == 1 else tuple_of(*outs)
    out_t = _probe_type(params, body)
    fn = bind(CompositeDefinition(f"NciDecl{seed}", 1, (), lambda S: (params, out_t), body, register=False))
    registered = tuple(n for n, _ in params if rng.random() < 0.3)
    return codec.canonical_json(codec.encode_program(Program(fn, registered=registered))).decode()


def _probe_type(params, body) -> "Type":
    """The type `body` returns, by building it once."""
    B = Builder(params, Tuple())
    colls = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
    return body(B, None, *colls).type


def decoy(batch: bool) -> str:
    """A declaration whose root Calls (or batches, `batch`) a composite keyed with an input id, `Input16_v1`: its gates
    are computed, so its root node is no input node, though its callee's key reads as one."""
    w = 8
    op = vectors.prim((w, w), w)
    Decoy = bind(CompositeDefinition("Input16", 1, (), lambda S: ((("x", Value(w)),), Value(w)),
                                     lambda B, S, x: B.call(op, x, x), register=False))
    xt = Array(2, Value(w)) if batch else Value(w)

    def body(B, S, x):
        d = B.batch(Decoy, x, axes=(0,)) if batch else B.call(Decoy, x)
        return B.call(op, d[0], d[1]) if batch else B.call(op, d, x)
    fn = bind(CompositeDefinition(f"NciDecoy{int(batch)}", 1, (), lambda S: ((("x", xt),), Value(w)), body,
                                  register=False))
    return codec.canonical_json(codec.encode_program(Program(fn, registered=()))).decode()


def decoys() -> list[dict]:
    """Instances on `decoy`'s declarations: the decoy node in θ and in no IU, in no IU, and in an IU. Their inputs are
    bound per run, so θ holds no input node."""
    out = []
    for batch in (False, True):
        text = decoy(batch)
        inp, _, spans = reference.root_nodes(text)
        dk = next(k for k in range(len(inp)) if not inp[k])
        for theta, at in (([dk], "input"), ([], "input"), ([], 0)):
            owner = ["input" if inp[k] else 0 for k in range(len(inp))]
            owner[dk] = at
            out.append({"program": text, "theta": theta, "owner": owner, "gamma": spans[-1][1], "alpha": [16, 1]})
    return out


def walk(n: int) -> str:
    """A declaration whose one gate reads the carry of a scan of `n` iterations, each a scan of `n` identity steps: no
    other gate, and about `n²` activations, so as many steps to locate the gate's operand. At `n = 2^32` that is past
    what `Prog.gate` walks (`Steps.FUEL`), and `NciCheck.Desc` refuses it (`walkBound`)."""
    w = 8
    op = vectors.prim((w,), w)
    ret = Tuple(Value(w), Tuple())
    Step = bind(CompositeDefinition(f"NciWalkStep{n}", 1, (), lambda S: ((("c", Value(w)),), ret),
                                    lambda B, S, c: tuple_of(c, tuple_of()), register=False))
    Mid = bind(CompositeDefinition(f"NciWalkMid{n}", 1, (), lambda S: ((("c", Value(w)),), ret),
                                   lambda B, S, c: tuple_of(B.scan(Step, c, n=n)[0], tuple_of()), register=False))
    fn = bind(CompositeDefinition(f"NciWalk{n}", 1, (), lambda S: ((("x", Value(w)),), Value(w)),
                                  lambda B, S, x: B.call(op, B.scan(Mid, x, n=n)[0]), register=False))
    return codec.canonical_json(codec.encode_program(Program(fn, registered=()))).decode()


WALKS = (3, 2 ** 32)


def walks() -> list[dict]:
    """Instances on `walk`'s declarations, θ their registered inputs (none) and the Call in IU 0: one `NciCheck.Desc`
    gives a verdict on, and one past `walkBound`'s limit, which it refuses."""
    out = []
    for n in WALKS:
        text = walk(n)
        inp, reg, spans = reference.root_nodes(text)
        out.append({"program": text, "theta": [k for k in range(len(inp)) if inp[k] and reg[k]],
                    "owner": ["input" if inp[k] else 0 for k in range(len(inp))], "gamma": spans[-1][1],
                    "alpha": [16, 1]})
    return out


def empty(kind: str) -> str:
    """A declaration whose root holds a scan of no iterations: its carry a value no gate reads (`unread`), one passed
    to a Call whose callee reads none of it (`passed`), or one a primitive reads (`read`); or its carry empty (`bare`).
    `Flock.Program.resolve` can't resolve the carry, and `NciCheck.Desc` refuses the first three (`resolvable`) and
    gives a verdict on the fourth."""
    w = 8
    op = vectors.prim((w, w), w)
    carry = Tuple() if kind == "bare" else Value(w)
    Step = bind(CompositeDefinition(f"NciEmptyStep{kind.capitalize()}", 1, (),
                                    lambda S: ((("c", carry),), Tuple(carry, Tuple())),
                                    lambda B, S, c: tuple_of(c, tuple_of()), register=False))
    Skip = bind(CompositeDefinition("NciEmptySkip", 1, (), lambda S: ((("c", Value(w)), ("x", Value(w))), Value(w)),
                                    lambda B, S, c, x: B.call(op, x, x), register=False))

    def body(B, S, x):
        s = B.scan(Step, tuple_of() if kind == "bare" else x, n=0)
        if kind == "passed":
            return B.call(Skip, s[0], x)
        return B.call(op, s[0], x) if kind == "read" else B.call(op, x, x)
    fn = bind(CompositeDefinition(f"NciEmpty{kind.capitalize()}", 1, (), lambda S: ((("x", Value(w)),), Value(w)), body,
                                  register=False))
    return codec.canonical_json(codec.encode_program(Program(fn, registered=()))).decode()


EMPTIES = ("unread", "passed", "read", "bare")


def empties() -> list[dict]:
    """Instances on `empty`'s declarations, θ their registered inputs (none) and the Call in IU 0: three refused
    (`program`) and one given a verdict."""
    out = []
    for kind in EMPTIES:
        text = empty(kind)
        inp, reg, spans = reference.root_nodes(text)
        out.append({"program": text, "theta": [k for k in range(len(inp)) if inp[k] and reg[k]],
                    "owner": ["input" if inp[k] else 0 for k in range(len(inp))], "gamma": spans[-1][1],
                    "alpha": [16, 1]})
    return out


def shared_decl(reader: str, order: tuple[str, ...]) -> str:
    """A declaration whose root batches a composite over two members, each returning its shared parameters `a` and `b`
    (in `order`), the outputs of two Calls `A` and `B` on registered inputs, and a gate (a primitive, `reader =
    "primitive"`; a composite's, `"composite"`; or a primitive reading a composite that makes the batch, `"wrapped"`)
    reading every member's return: one copy of pieces of both Calls."""
    w, n, L = 8, 2, len(order)
    op = vectors.prim((w, w), w)
    total = vectors.summing(L * n, w, w)
    tag = f"{reader}{''.join(order)}"
    ret = Tuple(*[Value(w)] * L)
    F = bind(CompositeDefinition(f"NciShared{tag}", 1, (), lambda S: ((("a", Value(w)), ("b", Value(w)),
                                                                       ("x", Value(w))), ret),
                                 lambda B, S, a, b, x: tuple_of(*[{"a": a, "b": b}[o] for o in order]), register=False))
    G = bind(CompositeDefinition(f"NciSharedBatch{tag}", 1, (), lambda S: ((("a", Value(w)), ("b", Value(w)),
                                                                            ("x", Array(n, Value(w)))), Array(n, ret)),
                                 lambda B, S, a, b, x: B.batch(F, a, b, x, axes=(None, None, 0)), register=False))
    R = bind(CompositeDefinition(f"NciSharedRead{tag}", 1, (), lambda S: ((("v", Array(L * n, Value(w))),), Value(w)),
                                 lambda B, S, v: B.call(total, v), register=False))

    def body(B, S, y, z, x):
        a, b = B.call(op, y, y), B.call(op, z, z)
        k = B.call(G, a, b, x) if reader == "wrapped" else B.batch(F, a, b, x, axes=(None, None, 0))
        r = k.refs
        v = Coll(Array(L * n, Value(w)), Affine(r.space, r.idx, r.base, L * n))
        return B.call(R, v) if reader == "composite" else B.call(total, v)
    fn = bind(CompositeDefinition(f"NciSharedDecl{tag}", 1, (), lambda S: ((("y", Value(w)), ("z", Value(w)),
                                                                            ("x", Array(n, Value(w)))), Value(w)),
                                  body, register=False))
    return codec.canonical_json(codec.encode_program(Program(fn, registered=("y", "z")))).decode()


def shared() -> list[dict]:
    """Instances on `shared_decl`'s declarations, θ the registered inputs and α 1: the reading gate in `A`'s IU and `B`
    in another, and the other way round. A read is one node's gates, so `B`'s gates are novel to the first IU however
    the copy's pieces are ordered, and `A`'s not to the second."""
    out = []
    for reader in ("primitive", "composite", "wrapped"):
        for order in (("a", "b"), ("b", "b", "a")):
            text = shared_decl(reader, order)
            inp, reg, spans = reference.root_nodes(text)
            calls = [k for k in range(len(inp)) if not inp[k] and spans[k][1] > spans[k][0]]
            for other in calls[:2]:
                owner = ["input" if inp[k] or spans[k][1] == spans[k][0] else 0 for k in range(len(inp))]
                owner[other] = 1
                out.append({"program": text, "theta": [k for k in range(len(inp)) if inp[k] and reg[k]],
                            "owner": owner, "gamma": spans[-1][1], "alpha": [1, 1]})
    return out


@functools.cache
def mistyped() -> list[dict]:
    """`mistyped_vectors.cases()`: references naming leaves of other widths than their types give them, which the codec
    refuses (`E_REINTERPRET`), and their well-typed neighbours."""
    return _load("mistyped_vectors").cases(instance)


def instance(seed: int) -> dict:
    """Instance `seed`, with what it was made from (`shape`, `theta_kind`, `division`, `malformed`): below `COUNT` a random
    declaration, then `PER_FIXTURE` instances on each of `fixtures()`, then `decoys()`, `walks()`, `empties()`,
    `shared()` and `mistyped()`."""
    rng = random.Random(seed)
    first_decoy = COUNT + PER_FIXTURE * len(fixtures())
    first_walk = first_decoy + len(decoys())
    first_empty = first_walk + len(walks())
    first_shared = first_empty + len(empties())
    first_mistyped = first_shared + len(shared())
    if seed >= first_mistyped:
        return {"instance": mistyped()[seed - first_mistyped], "shape": "mistyped", "theta_kind": None,
                "division": None, "malformed": None}
    if seed >= first_shared:
        return {"instance": shared()[seed - first_shared], "shape": "shared", "theta_kind": None, "division": None,
                "malformed": None}
    if seed >= first_empty:
        return {"instance": empties()[seed - first_empty], "shape": "empty", "theta_kind": None, "division": None,
                "malformed": None}
    if seed >= first_walk:
        return {"instance": walks()[seed - first_walk], "shape": "walk", "theta_kind": None, "division": None,
                "malformed": None}
    if seed >= first_decoy:
        return {"instance": decoys()[seed - first_decoy], "shape": "decoy", "theta_kind": None, "division": None,
                "malformed": None}
    if seed < COUNT and rng.random() < TRANSPOSED:
        text, kind = transposed(rng, seed), "transposed"
    elif seed < COUNT:
        text, kind, _ = vectors.program(rng, seed)
    else:
        name, text = fixtures()[(seed - COUNT) // PER_FIXTURE]
        kind = f"fixture:{name}"
    inp, reg, spans = reference.root_nodes(text)
    K = len(inp)
    inputs = [k for k in range(K) if inp[k]]
    registered = [k for k in inputs if reg[k]]
    calls = [k for k in range(K) if not inp[k] and spans[k][1] > spans[k][0]]
    empty = [k for k in range(K) if not inp[k] and spans[k][1] == spans[k][0]]
    theta_kind = rng.choice(["none", "all", "registered", "random"])
    theta = {"none": [], "all": inputs, "registered": registered,
             "random": [k for k in registered if rng.random() < 0.5]}[theta_kind]
    theta = rng.sample(theta, len(theta))
    how, labels = division(rng, calls)
    owner: list = ["input"] * K
    for k, u in zip(calls, labels, strict=True):
        owner[k] = u
    for k in empty:
        owner[k] = rng.choice(["input", rng.randint(0, max(labels, default=0))])
    inst = {"program": text, "theta": theta, "owner": owner, "gamma": 0, "alpha": [0, 1]}
    N = spans[-1][1] if spans else 0
    w, _, _, _ = reference.circuit(text)
    probe = reference.check_desc(inst | {"theta": [k for k in theta if reg[k]], "gamma": N, "alpha": [sum(w) + 1, 1]})
    ius = probe["ius"]
    biggest = max((s for s, _ in ius), default=0)
    crit = max((Fraction(r, s) for s, r in ius), default=Fraction(0))
    inst["gamma"] = max(0, biggest + rng.choices([0, -1, 1, rng.randint(-3, 3)], [5, 3, 2, 1])[0])
    inst["alpha"] = vectors._alpha(rng, crit)
    bad = rng.choice(MALFORMED) if rng.random() < 0.35 else None
    if bad:
        _break(rng, inst, bad, inp, reg, spans)
    return {"instance": inst, "shape": kind, "theta_kind": theta_kind, "division": how, "malformed": bad}


def _break(rng, inst: dict, bad: str, inp: list[bool], reg: list[bool], spans: list[tuple[int, int]]) -> None:
    """Make `inst` malformed in the way `bad` names."""
    K = len(inp)
    owner, theta = inst["owner"], inst["theta"]
    calls = [k for k in range(K) if not inp[k] and spans[k][1] > spans[k][0]]
    inputs = [k for k in range(K) if inp[k] and spans[k][1] > spans[k][0]]
    registered = [k for k in inputs if reg[k]]
    per_run = [k for k in range(K) if inp[k] and not reg[k]]
    if bad in ("instance", "params", "program", "not-topological"):
        vectors._break(rng, inst, bad, [], [])
    elif bad == "theta-unknown":
        theta.insert(rng.randint(0, len(theta)), rng.choice([K, K + rng.randint(1, 5), -1, "0", None, 1.5]))
    elif bad == "theta-not-input" and calls:
        theta.insert(rng.randint(0, len(theta)), rng.choice(calls))
    elif bad == "theta-not-registered" and per_run:
        theta.insert(rng.randint(0, len(theta)), rng.choice(per_run))
    elif bad == "theta-duplicate" and registered:
        k = rng.choice(theta or registered)
        if not theta:
            theta.append(k)
        theta.insert(rng.randint(0, len(theta)), k)
    elif bad == "owner-length":
        if rng.random() < 0.5 and owner:
            owner.pop(rng.randrange(len(owner)))
        else:
            owner.append(rng.choice(["input", 0]))
    elif bad == "owner-entry" and owner:
        owner[rng.randrange(K)] = rng.choice([-1, 1.5, None, "x", True, [0], "Input"])
    elif bad == "input-in-iu" and inputs:
        owner[rng.choice(inputs)] = rng.choice([0, max([o for o in owner if o != "input"], default=0)])
    elif bad == "gate-in-no-iu" and calls:
        owner[rng.choice(calls)] = "input"
    elif bad == "empty-iu" and calls:
        top = max(o for k, o in enumerate(owner) if k in calls)
        owner[rng.choice(calls)] = rng.choice([top + 2, top + 1 + K, 10 ** 15])


def expanded(inst: dict, result: dict):
    """The instance gate by gate, where `NciCheck.Desc` gives a verdict."""
    return None if "error" in result else reference.expand(inst)


def line(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


NAMES = ("instances", "expanded", "python")


def total() -> int:
    """How many instances the audit's data has: the random ones, every fixture's, the decoys,
    `difftest_vectors.pinned()`'s, the walks, the empties, the shared members' and the mistyped ones."""
    return (COUNT + PER_FIXTURE * len(fixtures()) + len(decoys()) + len(vectors.pinned()) + len(walks())
            + len(empties()) + len(shared()) + len(mistyped()))


def write(out: Path, count: int | None = None) -> dict[str, Path]:
    """The three files for instances 0 … count − 1, then `difftest_vectors.pinned()`'s (all of them, `total()`, by
    default) under ``out/desc_difftest``."""
    pins = [p["instance"] for p in vectors.pinned()]
    count = total() - len(pins) if count is None else count
    d = out / "desc_difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {f"desc_difftest/{n}.jsonl": d / f"{n}.jsonl" for n in NAMES}
    files = [open(paths[f"desc_difftest/{n}.jsonl"], "w") for n in NAMES]
    try:
        for inst in [*(instance(seed)["instance"] for seed in range(count)), *pins]:
            res = reference.check_desc(inst)
            for f, obj in zip(files, (inst, expanded(inst, res), res), strict=True):
                f.write(line(obj))
    finally:
        for f in files:
            f.close()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.ml.circuits reads the programs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=None, help="the first N instances (default: all, `total()`)")
    a = ap.parse_args()
    here = Path(verity.ml.circuits.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.ml.circuits comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
