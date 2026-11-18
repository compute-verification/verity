"""The record form's wired reading's difftest data, for `InferenceOnlyVerifier/RecordWiredDifftestMain.lean`.

    python scripts/record_wired_difftest_vectors.py <verity checkout> --out DIR

Writes two files under `DIR/record_wired_difftest/`, one line per case: `cases.jsonl`, `{"instance", "trusted",
"rows", "edge"}`, and `python.jsonl`, `{"detail", "kinds"}`: `inference_only_record_reference`'s `check_detail` and
`kinds` with `wired`, `RecordWired`'s reading. `edge` is whether the case's root has an edge (`has_edge`): on a case
with none the difftest also holds `RecordWired` to `Record`, field for field.

The cases: `TOYS`, roots whose units read earlier units' outputs, each with the verdict it is made for (`TOYS`); then
the record difftest's `shaped()` cases and its first `RANDOM` random ones (`record_difftest_vectors`), on which the
wired reading is the form's own, but for a root that hands a unit a unit's output.

Each toy's templates are one range each, as `RecordText.templatesOf` takes them: a template over many ranges is
refused with its templates, the last toy.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
from pathlib import Path

import verity.circuits
from verity.circuits import codec
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import array_of
from verity.circuits.types import Array, Tuple, Value

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = _load("record_difftest_vectors")
ref, IF = V.ref, V.IF
#: the record difftest's random cases this one reads under the wired reading
RANDOM = 120

U4, U8 = Array(4, Value(16)), Array(8, Value(16))
A = V.template("WiredA", [("x", U4), ("h", U4)], [U4], [[("op", 16, [(0, k), (1, k)]) for k in range(4)]])
A2 = V.template("WiredPair", [("x", U4), ("h", U4)], [U4, U4],
                [[("op", 16, [(0, k)]) for k in range(4)], [("op", 16, [(1, k)]) for k in range(4)]])
B = V.template("WiredB", [("y", U4)], [U4], [[("op", 16, [(0, k)]) for k in range(4)]])
WIDE = V.template("WiredWide", [("y", U8)], [U4], [[("op", 16, [(0, k), (0, k + 4)]) for k in range(4)]])
NARROW = V.template("WiredNarrow", [("y", Array(4, Value(8)))], [Value(16)], [[("op", 16, [(0, 0)])]])


class Wired(V.Circuit):
    """A circuit whose root is `body` over input arrays `inputs` (`(name, type)` each), the record's templates and
    units `units` (`(Definition, units, entries)` each, `V.Circuit.add`'s entries)."""

    def __init__(self, name: str, slots: int, inputs: list, ret, body, units: list) -> None:
        super().__init__(name, slots)
        self.inputs, self.ret, self.body = inputs, ret, body
        for fn, _n, entries in units:
            self.add(fn, entries)

    def program(self) -> tuple[str, str]:
        sig = (tuple(self.inputs), self.ret)
        root = CompositeDefinition(self.name, 1, (), lambda S: sig, self.body, register=False)
        desc = codec.encode_program(Program(bind(root)))
        return codec.canonical_json(desc).decode(), codec.program_digest(desc)


def one(slots: int, n: int) -> list:
    """`n` units, unit `i` in slot `i % slots`."""
    return [([n], ["slot"], 0)] if n == slots else [([n // slots, slots], ["x", "slot"], 0)]


def _ab(body, n: int = 2, m: int = 2, slots: int = 1, reader=B, writer=A, ret=None):
    """`writer` batched over `n` members' `x` and `h`, then `reader` as `body` has it, `m` units."""
    return Wired("Wired", slots, [("x", Array(n, U4)), ("h", Array(n, U4))], ret or Array(m, U4), body,
                 [(writer, n, one(slots, n)), (reader, m, one(slots, m))])


def _mapped(B_, S, x, h):
    return B_.batch(B, B_.batch(A, x, h, axes=(0, 0)), axes=(0,))


def _shifted(B_, S, x, h):
    """`B`'s member `m` reads `A`'s member `m + 1`."""
    a = B_.batch(A, x, h, axes=(0, 0))
    return B_.batch(B, array_of([a[1], a[2]]), axes=(0,))


def _fanout(B_, S, x, h):
    """Every member of `B` reads `A`'s member 0."""
    a = B_.batch(A, x, h, axes=(0, 0))
    return B_.batch(B, a[0], axes=(None,), n=2)


def _call(reader=B):
    """`reader` called once, on `A`'s member 1."""
    return lambda B_, S, x, h: B_.call(reader, B_.batch(A, x, h, axes=(0, 0))[1])


def _first_two(B_, S, x, h):
    """`B`'s member `m` reads `A`'s member `m`, of three."""
    a = B_.batch(A, x, h, axes=(0, 0))
    return B_.batch(B, array_of([a[0], a[1]]), axes=(0,))


def _second(B_, S, x, h):
    """`B`'s member `m` reads the second output of `WiredPair`'s member `m`."""
    a = B_.batch(A2, x, h, axes=(0, 0))
    return B_.batch(B, array_of([a[0][1], a[1][1]]), axes=(0,))


def _half(B_, S, x, h):
    """`WiredWide` called on `A`'s member 0 then `x`'s member 1: half an edge, half input gates."""
    a = B_.batch(A, x, h, axes=(0, 0))
    return B_.call(WIDE, array_of([*(a[0][k] for k in range(4)), *(x[1][k] for k in range(4))]))


def _twice(B_, S, x, h):
    """`A` batched twice, the second reading `B`'s outputs: a template over two ranges."""
    a = B_.batch(A, x, h, axes=(0, 0))
    b = B_.batch(B, a, axes=(0,))
    return B_.batch(A, x, b, axes=(0, 0))


def _repoint(ref_: list, k: int, j: int = 0, form: str = "affine") -> callable:
    """The edit that sets root node `k`'s argument `j` to the references `ref_` of the form `form`."""
    def edit(nodes: list, _desc: dict) -> None:
        nodes[k]["args"][j]["refs"] = {form: ref_}
    return edit


def _narrow() -> Wired:
    """`A` batched, and `WiredNarrow` called on 8-bit inputs of its own, which the toy repoints to `A`'s member 1."""
    def body(B_, S, x, h, z):
        B_.batch(A, x, h, axes=(0, 0))
        return B_.call(NARROW, z)
    return Wired("Wired", 1, [("x", Array(2, U4)), ("h", Array(2, U4)), ("z", Array(4, Value(8)))], Value(16), body,
                 [(A, 2, one(1, 2)), (NARROW, 1, one(1, 1))])


#: each toy: its circuit, the edit of its Program's root (or None), the edit of its record (or None), and the code the
#: wired reading refuses it with, None where it takes it
TOYS = {
    "an edge, member to member, one slot": (lambda: _ab(_mapped), None, None, None),
    "an edge, member to member, two slots": (lambda: _ab(_mapped, slots=2), None, None, None),
    "an edge across slots": (lambda: _ab(_shifted, n=4, slots=2), None, None, None),
    "an edge every member reads": (lambda: _ab(_fanout), None, None, None),
    "an edge into a call": (lambda: _ab(_call(), m=1, ret=U4), None, None, None),
    "an edge from a writer's second output": (lambda: _ab(_second, writer=A2),
                                              _repoint(["n", 2, 4, 4, 8, 1, 8], 3, form="strided"), None, None),
    "an edge from a writer's second output, in two parts": (lambda: _ab(_second, writer=A2), None, None,
                                                            "edge-rows"),
    "a partial row": (lambda: _ab(_first_two, n=3, m=2), _repoint(["n", 2, 2, 8], 3), None, "edge-rows"),
    "a row of another width": (_narrow, _repoint(["n", 3, 4, 4], 4), None, "program"),
    "a row half its own input and half an edge": (lambda: _ab(_half, m=1, reader=WIDE, ret=U4), None, None,
                                                  "edge-rows"),
    "an edge into a port the record cuts": (lambda: _ab(_mapped), None,
                                            lambda d: d.update(cuts={"WiredB_v1": {"y": 2}}), "edge-rows"),
    "ingress onto an edge's row": (lambda: _ab(_mapped), None, lambda d: d.update(
        ingress=[{"reader": "WiredB_v1", "port": "y", "at": 0, "units": 1, "step": 0}]), "inputs"),
    "a registered read onto an edge's row": (lambda: _ab(_mapped), None, lambda d: d.update(registered={
        "params": {"weights/w": 8}, "tables": [], "roots": [], "openings": [],
        "reads": [{"reader": "WiredB_v1", "port": "y", "param": "weights/w", "at": 1, "radix": [1], "base": 0,
                   "step": [0]}]}), "inputs"),
    "a reference to a later node": (lambda: _ab(_mapped), _repoint(["n", 3, 0, 8], 2, 1), None, "not-topological"),
    "a template over two ranges": (lambda: Wired("Wired", 1, [("x", Array(2, U4)), ("h", Array(2, U4))], Array(2, U4),
                                                 _twice, [(A, 4, one(1, 4)), (B, 2, one(1, 2))]),
                                   None, None, "templates"),
}


def toy(what: str) -> dict:
    """The case of toy `what` (`TOYS`): its instance under no pin, `Γ` and `α` the largest, a registration log of no
    registered value, and every input row of every unit."""
    make, root, record, _ = TOYS[what]
    c = make()
    text, digest = c.program()
    if root is not None:
        desc = json.loads(text)
        root(desc["definitions"][desc["root"]]["body"]["nodes"], desc)
        text, digest = codec.canonical_json(desc).decode(), codec.program_digest(desc)
    doc = c.doc(digest)
    if record is not None:
        record(doc)
    text_record = IF.circuit_bytes(doc).decode()
    rng = random.Random(what)
    inst = {"program": text, "record": text_record, "gamma": V.BIG, "alpha": [V.BIG, 1], "pins": []}
    rows = [[codec.descriptor_id(fn), p, u, 0] for fn, n in c.templates for p, _ in fn.signature()[0] for u in range(n)]
    return {"instance": inst, "trusted": V.trusted(rng, digest, text_record, {}, {}), "rows": rows}


def _edge(case: dict) -> bool:
    """Whether the case's root has an edge (`has_edge`); False where its Program does not decode."""
    try:
        return ref.has_edge(ref.reference.desc_program(case["instance"]["program"]))
    except Exception:  # noqa: BLE001 - a Program the reference refuses has no root to read
        return False


def cases() -> list[dict]:
    """Every case, with whether its root has an edge."""
    made = [toy(what) for what in TOYS]
    made += [copy.deepcopy(m["case"]) for m in V.shaped()]
    made += [V.case(seed)["case"] for seed in range(RANDOM)]
    return [{"instance": m["instance"], "trusted": m["trusted"], "rows": m["rows"], "edge": _edge(m)} for m in made]


def results(case: dict) -> dict:
    """The reference's wired reading of a case: its whole verdict, and its rows' classes."""
    inst, tr = case["instance"], case["trusted"]
    return {"detail": ref.check_detail(inst, tr, wired=True), "kinds": ref.kinds(inst, tr, case["rows"], wired=True)}


def write(out: Path) -> dict[str, Path]:
    """The two files under ``out/record_wired_difftest``."""
    d = out / "record_wired_difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {n: d / f"{n}.jsonl" for n in ("cases", "python")}
    made = cases()
    paths["cases"].write_text("".join(V.line(c) for c in made))
    paths["python"].write_text("".join(V.line(results(c)) for c in made))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.circuits reads the programs")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    here = Path(verity.circuits.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.circuits comes from {here}, not from the checkout {a.verity}")
    for n, p in write(a.out).items():
        print(f"{n}: {p} ({sum(1 for _ in p.open())} lines)")


if __name__ == "__main__":
    main()
