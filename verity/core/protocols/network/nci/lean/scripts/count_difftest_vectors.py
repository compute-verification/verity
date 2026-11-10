"""The count mode's difftest data: declarations, the block Definitions they name, gate tables and `(c, k, n)`, for
`NciCheck/CountDifftestMain.lean`.

    python scripts/count_difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes three files under `DIR/count_difftest/`, one line per instance: `instances.jsonl`, as `NciCheck.Count.check`
reads it (`NciCheck.Count`'s docstring: the Program, θ by root node, the block Definitions by key, the gate table and
`(c, k, n)`); `detail.jsonl`, `nci_reference.check_count_detail`'s results, the whole verdict; and `public.jsonl`,
`nci_reference.check_count`'s, what the count publishes. The Lean audit runs `NciCheck/CountDifftestMain.lean` on the
three (`runs` in `lean-audit.json`), which fails unless `NciCheck.Count.checkDetail` and `NciCheck.Count.check` give
Python's results on every line: one summary per Definition against the count activation by activation.

Instance `seed`'s program, for the first `COUNT`, is one of `difftest_vectors.program`'s or (a quarter of them) one read
across its arrays' rows (`desc_difftest_vectors.transposed`), then each distinct descriptor the circuits fixtures hold
(`desc_difftest_vectors.fixtures`) for `PER_FIXTURE` instances each, then `hand()`'s, worked by hand. θ is none, the
registered root inputs or a random set of them. The blocks are every primitive the root reaches (`primitives`), with a
random set of its composites (`composites`) or all of them (`all`), in a random order; the gate table gives each of
those primitives 0 to 40 gates, and now and then names one the Program doesn't. `c`, `k` and `n` are at, below or above
the largest block's gates and fresh bits and the total. About a third of the instances are malformed, each in one way
`NciCheck.Count` refuses (`MALFORMED`: every code of its docstring but `unwalked`, which `desc_liveness_vectors.py`'s
shapes reach and these don't).
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import random
from pathlib import Path

import verity.circuits
from verity.circuits import codec
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program, is_input
from verity.circuits.types import Array, Value

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


desc = _load("desc_difftest_vectors")
vectors = desc.vectors
reference = vectors.reference

COUNT = 1500
PER_FIXTURE = 6
TRANSPOSED = 0.25
BIG = 10 ** 12
MALFORMED = ["instance", "params", "program", "not-topological", "theta-unknown", "theta-not-input",
             "theta-not-registered", "theta-duplicate", "block-unknown", "block-root", "block-input", "block-duplicate",
             "gate-in-no-block", "gate-table"]


def parts(text: str) -> tuple[str, dict, list[str], list[str]]:
    """The Program's root key, the Definitions the root reaches by key, and of those the primitives that are no input
    and the composites but the root."""
    program = reference.desc_program(text)
    defs = reference.reached(program)
    root = codec.descriptor_id(program.root)
    prims = [k for k, f in defs.items() if isinstance(f, PrimitiveDefinition) and not is_input(f)]
    comps = [k for k, f in defs.items() if not isinstance(f, PrimitiveDefinition) and k != root]
    return root, defs, sorted(prims), sorted(comps)


def hand() -> str:
    """A declaration worked by hand: a: Array<2, Value<8>> (registered), b: Value<4> (bound per run); the root batches
    `NciHandBlk` over a's rows with b shared, each member u = f(x, y), v = f(u, y) for f: (8, 4) → 8, and computes
    h(m0, m1) for h: (8, 8) → 8 on the two members' v. Root nodes 0 a, 1 b, 2 the batch, 3 h."""
    f, h = vectors.prim((8, 4), 8), vectors.prim((8, 8), 8)

    def blk(B, S, x, y):
        return B.call(f, B.call(f, x, y), y)
    Blk = bind(CompositeDefinition("NciHandBlk", 1, (), lambda S: ((("x", Value(8)), ("y", Value(4))), Value(8)), blk,
                                   register=False))

    def body(B, S, a, b):
        m = B.batch(Blk, a, b, axes=(0, None))
        return B.call(h, m[0], m[1])
    fn = bind(CompositeDefinition("NciHandCount", 1, (), lambda S: ((("a", Array(2, Value(8))), ("b", Value(4))),
                                                                     Value(8)), body, register=False))
    return codec.canonical_json(codec.encode_program(Program(fn, registered=("a",)))).decode()


#: `hand()`'s keys: the block composite and the two primitives
BLK, F, H = "NciHandBlk_v1", "NciOp8x4r8_v1", "NciOp8x8r8_v1"


@functools.cache
def hands() -> list[dict]:
    """Instances on `hand()`'s declaration, table f 3 gates and h 5. With θ a and blocks `NciHandBlk` and h, the rows are
    `NciHandBlk` (6 gates, 4 fresh bits: b, as a's rows are θ; 2 blocks) and h (5 gates, 16 fresh bits: the two members'
    v; 1 block), 3 in all: at `(6, 16, 3)` it complies, at `c = 5` fails on `NciHandBlk`'s gates, at `k = 15` on h's
    fresh bits and at `n = 2` on the count. With θ empty `NciHandBlk` reads 12 fresh bits. With the blocks f and h, f's
    `x` is θ at one call site and not the other, so f reads 12 fresh bits in each of its 4 blocks. Without h a gate is in
    no block."""
    text = hand()
    base = {"program": text, "theta": [0], "blocks": [BLK, H], "table": {F: 3, H: 5}, "c": 6, "k": 16, "n": 3}
    return [base, base | {"c": 5}, base | {"k": 15}, base | {"n": 2}, base | {"theta": []},
            base | {"blocks": [F, H], "c": 5, "k": 16, "n": 5}, base | {"blocks": [BLK]}]


def instance(seed: int) -> dict:
    """Instance `seed`, with what it was made from (`shape`, `theta_kind`, `blocks_kind`, `malformed`): below `COUNT` a
    random declaration, then `PER_FIXTURE` instances on each of `desc_difftest_vectors.fixtures()`, then `hands()`."""
    rng = random.Random(seed)
    first_hand = COUNT + PER_FIXTURE * len(desc.fixtures())
    if seed >= first_hand:
        return {"instance": hands()[seed - first_hand], "shape": "hand", "theta_kind": None, "blocks_kind": None,
                "malformed": None}
    if seed < COUNT and rng.random() < TRANSPOSED:
        text, kind = desc.transposed(rng, seed), "transposed"
    elif seed < COUNT:
        text, kind, _ = vectors.program(rng, seed)
    else:
        name, text = desc.fixtures()[(seed - COUNT) // PER_FIXTURE]
        kind = f"fixture:{name}"
    inp, reg, _ = reference.root_nodes(text)
    registered = [k for k in range(len(inp)) if inp[k] and reg[k]]
    theta_kind = rng.choice(["none", "registered", "random"])
    theta = {"none": [], "registered": registered, "random": [k for k in registered if rng.random() < 0.5]}[theta_kind]
    theta = rng.sample(theta, len(theta))
    try:
        root, defs, prims, comps = parts(text)
    except reference.Malformed:
        root, defs, prims, comps = None, {}, [], []
    blocks_kind = rng.choice(["primitives", "composites", "composites", "all"])
    chosen = {"primitives": [], "composites": [c for c in comps if rng.random() < 0.5], "all": comps}[blocks_kind]
    blocks = rng.sample(prims + chosen, len(prims) + len(chosen))
    table = {p: rng.choice([0, 1, 2, rng.randint(0, 40)]) for p in prims}
    if rng.random() < 0.1:
        table["NciUnused_v1"] = rng.randint(0, 9)
    inst = {"program": text, "theta": theta, "blocks": blocks, "table": table, "c": 0, "k": 0, "n": 0}
    probe = reference.check_count_detail(inst | {"c": BIG, "k": BIG, "n": BIG})
    rows = probe.get("blocks", [])
    active = [r for r in rows if r[3] > 0]

    def near(x: int) -> int:
        return max(0, x + rng.choices([0, -1, 1, rng.randint(-3, 3)], [6, 2, 2, 1])[0])
    inst["c"] = near(max((r[1] for r in active), default=0))
    inst["k"] = near(max((r[2] for r in active), default=0))
    inst["n"] = near(probe.get("total", 0))
    bad = rng.choice(MALFORMED) if rng.random() < 0.35 else None
    if bad:
        _break(rng, inst, bad, inp, reg, root, defs, rows)
    return {"instance": inst, "shape": kind, "theta_kind": theta_kind, "blocks_kind": blocks_kind, "malformed": bad}


def _break(rng, inst: dict, bad: str, inp: list[bool], reg: list[bool], root: str | None, defs: dict,
           rows: list) -> None:
    """Make `inst` malformed in the way `bad` names."""
    K = len(inp)
    theta, blocks, table = inst["theta"], inst["blocks"], inst["table"]
    inputs = [k for k in range(K) if inp[k]]
    registered = [k for k in inputs if reg[k]]
    per_run = [k for k in inputs if not reg[k]]
    computed = [k for k in range(K) if not inp[k]]
    if bad == "instance":
        how = rng.randrange(4)
        if how == 0:
            del inst[rng.choice(reference.COUNT_FIELDS)]
        elif how == 1:
            inst["program"] = json.loads(inst["program"])
        elif how == 2:
            inst[rng.choice(["theta", "blocks"])] = rng.choice([{}, 3, "0"])
        else:
            inst["table"] = rng.choice([[], 3, "{}", [["NciOp8r8_v1", 1]]])
    elif bad == "params":
        how = rng.randrange(3)
        if how == 0:
            inst[rng.choice(["c", "k", "n"])] = rng.choice([-1, 1.5, "2", None, True, [3]])
        elif how == 1:
            blocks.insert(rng.randint(0, len(blocks)), rng.choice([3, None, ["NciOp8r8_v1"], {"key": "x"}]))
        else:
            table[rng.choice(sorted(table) or ["NciOp8r8_v1"])] = rng.choice([-1, 1.5, "2", None, True])
    elif bad == "program":
        inst["program"] = rng.choice(["", "{", "[]", '{"schema": "verity-ir/descriptor/v0"}',
                                      inst["program"].replace('"root":', '"rooot":', 1)])
    elif bad == "not-topological":
        d = json.loads(inst["program"])
        if vectors._forward(d):
            inst["program"] = codec.canonical_json(d).decode()
    elif bad == "theta-unknown":
        theta.insert(rng.randint(0, len(theta)), rng.choice([K, K + rng.randint(1, 5), -1, "0", None, 1.5]))
    elif bad == "theta-not-input" and computed:
        theta.insert(rng.randint(0, len(theta)), rng.choice(computed))
    elif bad == "theta-not-registered" and per_run:
        theta.insert(rng.randint(0, len(theta)), rng.choice(per_run))
    elif bad == "theta-duplicate" and registered:
        k = rng.choice(theta or registered)
        if not theta:
            theta.append(k)
        theta.insert(rng.randint(0, len(theta)), k)
    elif bad == "block-unknown":
        blocks.insert(rng.randint(0, len(blocks)), rng.choice(["NoSuch_v1", "", "Input8_v2", root.lower() if root else "x"]))
    elif bad == "block-root" and root:
        blocks.insert(rng.randint(0, len(blocks)), root)
    elif bad == "block-input":
        ins = sorted(k for k, f in defs.items() if is_input(f))
        if ins:
            blocks.insert(rng.randint(0, len(blocks)), rng.choice(ins))
    elif bad == "block-duplicate" and blocks:
        blocks.insert(rng.randint(0, len(blocks)), rng.choice(blocks))
    elif bad == "gate-in-no-block":
        walked = [r[0] for r in rows if r[3] > 0 and isinstance(defs[r[0]], PrimitiveDefinition)]
        if walked:
            blocks.remove(rng.choice(walked))
    elif bad == "gate-table":
        held = [p for p in table if p in defs]
        if held:
            del table[rng.choice(held)]


def line(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


NAMES = ("instances", "detail", "public")


def total() -> int:
    """How many instances the audit's data has: the random ones, every fixture's and the hand-worked ones."""
    return COUNT + PER_FIXTURE * len(desc.fixtures()) + len(hands())


def write(out: Path, count: int | None = None) -> dict[str, Path]:
    """The three files for instances 0 … count − 1 (all of them, `total()`, by default) under ``out/count_difftest``."""
    count = total() if count is None else count
    d = out / "count_difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {f"count_difftest/{n}.jsonl": d / f"{n}.jsonl" for n in NAMES}
    files = [open(paths[f"count_difftest/{n}.jsonl"], "w") for n in NAMES]
    try:
        for seed in range(count):
            inst = instance(seed)["instance"]
            for f, obj in zip(files, (inst, reference.check_count_detail(inst), reference.check_count(inst)),
                              strict=True):
                f.write(line(obj))
    finally:
        for f in files:
            f.close()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.circuits reads the programs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=None, help="the first N instances (default: all, `total()`)")
    a = ap.parse_args()
    here = Path(verity.circuits.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.circuits comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
