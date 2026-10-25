"""The shared-members family for `nci-desc-fuzz`: whole members of a batch returning leaves of arguments they share,
each another root node's output, read by a gate, with the IU assignments to check them on. One `{"id", "family",
"program", "instances"}` object per line (`instances`: `θ`, owners, `Γ` and `α`, which `agree` checks beside its own
four).

    uv run python gen/shared.py [--seeds] --seed S --count N --out FILE [--mutate P]

It starts from `scripts/desc_difftest_vectors.py`'s `shared()`, the twelve vectors of #4066's multi-node reads: with
`--seeds`, their six declarations first, each with its two instances (`shared-seed`), written by `decl`, which on a
seed's parameters gives `shared_decl`'s text (checked on every run). Then `N` declarations from seed `S` on (`shared`),
`decl` on parameters drawn at random (`draw`): two or three shared arguments, each made by a root node of its own (a
primitive on an input, a call of a composite of two gates, one member of a batch of primitives, or the input itself,
registered or bound per run); the members' return, each argument's leaves in any order and repeated, now and then with a
gate of the member's own; two to four members, with a mapped argument or none; the reader (a primitive, a composite's
primitive, or either reading a composite that makes the batch) and what it reads (every member, a run of whole members,
or a run cut inside them), now and then a second primitive reading another run. Each comes with up to `ASSIGNMENTS` IU
assignments of its root nodes that hold gates (all of them when there are no more, else drawn without repeats), θ the
registered inputs, `Γ` all the gates and `α` 1. With `--mutate P`, each is also written once more with one to three
random edits (`programs.mutate`), with the same instances, with probability `P` (`shared-mutant`).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "lean" / "scripts"
ASSIGNMENTS = 12
NAMES = ("a", "b", "c")
INPUTS = ("y", "z", "u")


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


desc = _load(SCRIPTS / "desc_difftest_vectors.py")
vectors = desc.vectors
reference = desc.reference
programs = _load(HERE / "programs.py")


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
    return codec.canonical_json(codec.encode_program(Program(fn, registered=reg))).decode()


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
                raise SystemExit(f"shared.py: decl({reader}, {order}) is not shared_decl's declaration")
            out.append({"id": f"shared-seed-{reader}{''.join(order)}", "family": "shared-seed", "program": text,
                        "instances": given.pop(text)})
    if given:
        raise SystemExit(f"shared.py: {len(given)} of shared()'s declarations are not shared_decl's")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--mutate", type=float, default=0.0)
    ap.add_argument("--seeds", action="store_true", help="write the six seed declarations first")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        for x in seeds() if a.seeds else []:
            f.write(json.dumps(x) + "\n")
        for s in range(a.seed, a.seed + a.count):
            rng = random.Random(s)
            text = decl(draw(rng, s))
            given = instances(rng, text)
            f.write(json.dumps({"id": f"shared-{s}", "family": "shared", "program": text, "instances": given}) + "\n")
            mrng = random.Random(~s)
            if mrng.random() < a.mutate:
                f.write(json.dumps({"id": f"shared-mutant-{s}", "family": "shared-mutant",
                                    "program": programs.mutate(mrng, text), "instances": given}) + "\n")


if __name__ == "__main__":
    main()
