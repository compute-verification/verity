"""The difftest of the NCI policy's Lean check against its Python reference: the Python half.

`verity/core/protocols/network/nci/lean/scripts/difftest_vectors.py` writes 4,000 random declarations with divisions,
335 whose references name leaves of other widths than their types give them (`E_REINTERPRET`) or are their well-typed
neighbours, and two pinned ones (θ with an input bound per run, refused, and θ the registered input, accepted), and
`scripts/nci_reference.py`'s results on each (per IU its size and novel reads, the verdict and the first failing IU, or
the first malformation's code). `check` runs the Lean half in its Lean audit (`runs` in that package's
`lean-audit.json`): `NciCheck/DifftestMain.lean` checks every instance with `NciCheck.check` and fails on any difference.
The audit checks the data against the record's SHA-256, and its pass is cached on the Lean package's files alone, so the
test here is what says the data is still what the generator and the reference write: a change to either changes the
data, fails this test, and `audit.py --update` records the new data, which makes `check` run the Lean half on it. The
tests here also check, on the first 400 instances, that the generator reaches what its docstring says, and the
reference itself on a declaration worked by hand.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest


LEAN = Path(__file__).resolve().parents[1] / "nci" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("difftest_vectors")
reference = vectors.reference
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NciCheck.DifftestMain"]


@pytest.mark.slow
def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 4,337 instances and the reference's results on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{difftest/instances.jsonl}", "{difftest/python.jsonl}", str(vectors.total())]


def test_the_generator_covers_what_it_says():
    made = [vectors.instance(s) for s in range(400)]
    results = [vectors.python_results(m["instance"]) for m in made]
    assert {m["shape"] for m in made} == set(vectors.SHAPES)
    assert {m["theta_kind"] for m in made} == {"none", "all", "registered", "random"}
    assert {m["division"] for m in made} == {"single", "singletons", "runs", "random"}
    assert {r.get("error") for r in results} == set(vectors.MALFORMED) | {None}
    good = [(m["instance"], r) for m, r in zip(made, results, strict=True) if "error" not in r]
    assert {r["failure"][1] for _, r in good if r["failure"]} == {"size", "nci"}
    assert any(r["complies"] for _, r in good) and any(r["failure"] and r["failure"][0] > 0 for _, r in good)
    assert any(r["ius"] == [] for _, r in good), "a declaration with no computed gate"
    assert any(i["theta"] and reference.check(i | {"theta": []})["ius"] != r["ius"] for i, r in good), "θ that saves reads"
    biggest = [max((s for s, _ in r["ius"]), default=0) for _, r in good]
    assert any(r["complies"] and i["gamma"] == b > 0 for (i, r), b in zip(good, biggest, strict=True))
    assert any(r["failure"] and r["failure"][1] == "size" and i["gamma"] == b - 1 for (i, r), b in zip(good, biggest, strict=True))
    crit = [max((Fraction(n, s) for s, n in r["ius"]), default=Fraction(0)) for _, r in good]
    assert any(r["complies"] and c > 0 and Fraction(*i["alpha"]) == c for (i, r), c in zip(good, crit, strict=True))
    assert any(r["failure"] and r["failure"][1] == "nci" and Fraction(*i["alpha"]) < c for (i, r), c in zip(good, crit, strict=True))
    assert any(math.gcd(*i["alpha"]) > 1 for i, _ in good), "an α not in lowest terms"
    forms = {n["form"] for i, _ in good for d in json.loads(i["program"])["definitions"].values() if "body" in d
             for n in d["body"]["nodes"]}
    assert forms == {"primitive", "call", "batch", "scan"}
    widths = {reference.circuit(i["program"])[0][g] for i, _ in good[:80] for g in range(len(i["owner"]))}
    assert set(vectors.WIDTHS) <= widths


def test_the_reference_on_a_declaration_worked_by_hand():
    """`vectors.hand()`: IU 0 = {c} reads a (baseline) and b: 16 novel bits. IU 1 = {d, e} reads c, b and a: 4 + 16
    novel bits, a free. b is bound per run, so it can't be baseline."""
    inst = {"program": vectors.hand(), "theta": [0], "owner": ["input", "input", 0, 1, 1], "gamma": 2, "alpha": [16, 1]}
    assert reference.check(inst) == {"gates": 5, "ius": [[1, 16], [2, 20]], "complies": True, "failure": None}
    assert reference.check(inst | {"alpha": [31, 2]})["failure"] == [0, "nci"]
    assert reference.check(inst | {"alpha": [32, 2], "gamma": 1})["failure"] == [1, "size"]
    assert reference.check(inst | {"theta": []})["ius"] == [[1, 24], [2, 28]]
    assert reference.check(inst | {"owner": ["input", "input", 1, 0, 0]})["ius"] == [[2, 20], [1, 16]]
    assert reference.check(inst | {"owner": ["input", "input", 0, 0, 0]})["ius"] == [[3, 16]]
    for owner, code in ((["input", "input", 0, 2, 2], "empty-iu"), (["input", "input", 0, 1], "owner-length"),
                        ([0, "input", 0, 1, 1], "input-in-iu"), (["input", "input", "input", 1, 1], "gate-in-no-iu"),
                        (["input", "input", 0, 1, -1], "owner-entry")):
        assert reference.check(inst | {"owner": owner}) == {"error": code}
    for theta, code in (([2], "theta-not-input"), ([0, 0], "theta-duplicate"), ([5], "theta-unknown"),
                        ([1], "theta-not-registered"), ([0, 1], "theta-not-registered"), ([1, 2], "theta-not-registered")):
        assert reference.check(inst | {"theta": theta}) == {"error": code}


def test_the_pinned_instances_get_their_results():
    """The data's last two lines: θ holding b, bound per run, is refused (`theta-not-registered`), and θ of the
    registered a alone gets its verdict. The difftest holds `NciCheck.check` to these results."""
    pins = vectors.pinned()
    assert [p["theta_kind"] for p in pins] == ["all", "registered"]
    for p in pins:
        assert reference.check(p["instance"]) == p["expect"]
