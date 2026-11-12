"""The count of `(c, k)`-inference blocks (`NciCheck.Count`) against the Python reference: the Python half.

`verity/protocols/inference_only/verifier/nci/lean/scripts/count_difftest_vectors.py` writes 1,687 instances of the count (1,500
random, 6 on each distinct descriptor of the circuits fixtures, 7 worked by hand) and, on each,
`scripts/nci_reference.py`'s `check_count_detail` (the whole verdict, for the private record) and `check_count` (what
the count publishes, the judgment against n). `check` runs the Lean half in its Lean audit (`runs` in that package's
`lean-audit.json`): `NciCheck/CountDifftestMain.lean` fails unless `NciCheck.Count.checkDetail` and
`NciCheck.Count.check` give Python's results on every instance. The audit checks the data against the record's SHA-256
and caches its pass on the Lean package's files alone, so the first test is what says the data is still what the
generator and the reference write. The second checks that the generator reaches what its docstring says and that the
published result is the judgment alone, the third the hand-worked rows, and the fourth that what the count gives bounds
what the unrolled circuit holds: each block's gates are the row's, its reads from outside it that are not θ's gates at
most the row's fresh bits, and the blocks cover the computed gates once each.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[4] / "inference_only" / "verifier" / "nci" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("count_difftest_vectors")
reference = vectors.reference
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NciCheck.CountDifftestMain"]


@pytest.mark.slow
def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 1,687 instances and the reference's results, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{count_difftest/instances.jsonl}", "{count_difftest/detail.jsonl}",
                           "{count_difftest/public.jsonl}", str(vectors.total())]


def _seeds() -> list[int]:
    fixtures = vectors.desc.fixtures()
    return [*range(400), *(vectors.COUNT + i * vectors.PER_FIXTURE + j for i in range(len(fixtures)) for j in (0, 1))]


def test_the_generator_covers_what_it_says():
    made = [vectors.instance(s) for s in _seeds()]
    results = [reference.check_count_detail(m["instance"]) for m in made]
    published = [reference.check_count(m["instance"]) for m in made]
    assert published == [r if "error" in r else {"complies": r["complies"]} for r in results]
    shapes = {m["shape"] for m in made}
    assert set(vectors.vectors.SHAPES) | {"transposed"} <= shapes
    assert {f"fixture:{name}" for name, _ in vectors.desc.fixtures()} <= shapes
    assert {m["theta_kind"] for m in made} == {"none", "registered", "random"}
    assert {m["blocks_kind"] for m in made} == {"primitives", "composites", "all"}
    assert {r.get("error") for r in results} == set(vectors.MALFORMED) | {None}
    good = [(m, r) for m, r in zip(made, results, strict=True) if "error" not in r]
    assert {r["failure"][1] for _, r in good if r["failure"]} == {"gates", "fresh", "count"}
    assert any(r["complies"] for _, r in good)
    assert any(m["shape"].startswith("fixture:") and r["total"] > 0 for m, r in good)
    assert any(r[3] > 1 for _, res in good for r in res["blocks"]), "a block Definition with more than one block"
    assert any(r[3] == 0 for _, res in good for r in res["blocks"]), "a block Definition only inside other blocks"


def test_the_hand_worked_rows():
    """`hands()`: the rows, total and verdicts worked out in its docstring, and of them only the judgment published."""
    assert [reference.check_count(i) for i in vectors.hands()] == [
        {"complies": True}, {"complies": False}, {"complies": False}, {"complies": False}, {"complies": True},
        {"complies": True}, {"error": "gate-in-no-block"}]
    blk, h = [vectors.BLK, 6, 4, 2], [vectors.H, 5, 16, 1]
    got = [reference.check_count_detail(i) for i in vectors.hands()]
    assert got == [
        {"blocks": [blk, h], "total": 3, "complies": True, "failure": None},
        {"blocks": [blk, h], "total": 3, "complies": False, "failure": [0, "gates"]},
        {"blocks": [blk, h], "total": 3, "complies": False, "failure": [1, "fresh"]},
        {"blocks": [blk, h], "total": 3, "complies": False, "failure": [None, "count"]},
        {"blocks": [[vectors.BLK, 6, 12, 2], h], "total": 3, "complies": True, "failure": None},
        {"blocks": [[vectors.F, 3, 12, 4], h], "total": 5, "complies": True, "failure": None},
        {"error": "gate-in-no-block"},
    ]
    first = vectors.total() - len(vectors.hands())
    assert [vectors.instance(s)["instance"] for s in range(first, vectors.total())] == vectors.hands()


def test_the_rows_bound_the_unrolled_blocks():
    """On each instance of `_seeds()` given a verdict, and each hand-worked one, the blocks of the unrolled circuit, one
    per activation of a block Definition outside every block: their gates are the computed gates, each once; each holds
    its row's Boolean gates, by the table; and the gates outside it that it reads, θ's gates aside, are at most its row's
    fresh bits wide."""
    from verity.circuits.program import is_input
    insts = [vectors.instance(s)["instance"] for s in _seeds()] + vectors.hands()
    checked = 0
    for inst in insts:
        res = reference.check_count_detail(inst)
        if "error" in res:
            continue
        program = reference.desc_program(inst["program"])
        C = program.circuit
        _, _, spans = reference.root_nodes(inst["program"])
        theta = {g for t in inst["theta"] for g in range(*spans[t])}
        rows = {r[0]: r for r in res["blocks"]}
        _, _, sites = reference.walk_blocks(program, set(inst["blocks"]), set(inst["theta"]))
        held: list[int] = []
        for key, scope, k, m in sites:
            fn = scope.spec.body.nodes[k].fn
            lo = scope.offset + scope.spec.body.offsets[k] + m * fn.gates
            gates = range(lo, lo + fn.gates)
            held += gates
            assert sum(inst["table"][reference.codec.descriptor_id(C.gate(g).prim)] for g in gates) == rows[key][1]
            into = {a for g in gates for a in C.gate(g).operands if a not in gates} - theta
            assert sum(C.gate(a).prim.ret.w for a in into) <= rows[key][2]
        assert sorted(held) == [g for g in range(C.size) if not is_input(C.gate(g).prim)]
        assert sum(r[3] for r in res["blocks"]) == len(sites) == res["total"]
        checked += 1
    assert checked > 200
