"""The difftest of the inference-only policy's Lean check against its Python reference: the Python half.

`verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts/difftest_vectors.py` writes 4,000 random declarations with divisions,
335 whose references name leaves of other widths than their types give them (`E_REINTERPRET`) or are their well-typed
neighbours, and three pinned ones (θ with an input bound per run, refused, θ the registered input, accepted, and a
declaration with a constant gate, accepted), then `weighing()`'s inference blocks padded with calls that compute nothing
(refused) or with their ids' neighbours (accepted), and
`scripts/inference_only_reference.py`'s results on each (per inference block its size and fresh input bits, the verdict and the first failing inference block, or
the first malformation's code). `check` runs the Lean half in its Lean audit (`runs` in that package's
`lean-audit.json`): `InferenceOnlyCheck/DifftestMain.lean` checks every instance with `InferenceOnlyCheck.check` and fails on any difference.
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


LEAN = Path(__file__).resolve().parents[7] / "verity" / "protocols" / "workload_classification" / "training_detection" / "inference_only" / "verifier" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("difftest_vectors")
reference = vectors.reference
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["InferenceOnlyVerifier.DifftestMain"]


@pytest.mark.slow
def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 4,338 instances and the reference's results on them, byte for byte as the record pins them."""
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
    """`vectors.hand()`: IU 0 = {c} reads a (baseline) and b: 16 fresh input bits. IU 1 = {d, e} reads c, b and a: 4 + 16
    fresh input bits, a free. b is bound per run, so it can't be baseline."""
    inst = {"program": vectors.hand(), "theta": [0], "owner": ["input", "input", 0, 1, 1], "gamma": 2, "alpha": [16, 1]}
    assert reference.check(inst) == {"gates": 5, "ius": [[1, 16], [2, 20]], "weighed": [1, 2], "complies": True,
                                     "failure": None}
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
    """The data's last three lines: θ holding b, bound per run, is refused (`theta-not-registered`), θ of the
    registered a alone gets its verdict, and so does the declaration with a constant gate, which reads nothing (a
    `Const<w>` primitive, the deployed spelling). The difftest holds `InferenceOnlyCheck.check` to these results."""
    pins = vectors.pinned()
    assert [p["theta_kind"] for p in pins] == ["all", "registered", "registered"]
    defs = json.loads(pins[-1]["instance"]["program"])["definitions"]
    assert [k for k in defs if k.startswith("Const")] == ["Const8[0x2a]_v1"]
    assert defs["Const8[0x2a]_v1"] == {"form": "primitive", "id": "Const8[0x2a]_v1", "params": [], "ret": ["Value", 8]}
    for p in pins:
        assert reference.check(p["instance"]) == p["expect"]


def test_a_call_that_computes_nothing_buys_no_fresh_bit():
    """`vectors.weighing()`: an inference block padded with the wiring between leaf widths (`WordBits{16}`, `BitsWord{16}`), a
    `GATELESS` primitive or a constant, which the rule that weighed every call accepted, is refused `nci`, by `check` on it
    gate by gate and by `check_desc` on it node by node; the same block padded with a neighbour of a wiring id is accepted.
    The difftest holds `InferenceOnlyCheck.check` and `InferenceOnlyCheck.Desc.check` to these results."""
    cases = vectors.weighing()
    assert {c["expect"]["complies"] for c in cases} == {True, False}
    for c in cases:
        inst, expect = c["instance"], c["expect"]
        assert reference.check_desc(inst) == expect == reference.check(reference.expand(inst))
        (size, fresh), = expect["ius"]
        num, den = inst["alpha"]
        assert c["accepted_unweighed"] == (fresh * den <= num * size)
        assert expect["weighed"][0] <= size
    named = cases[2:-1]
    assert [c["expect"]["complies"] for c in named] == [not free for _, _, free in vectors.NAMED]
    assert not cases[0]["expect"]["complies"] and cases[1]["expect"]["complies"] and not cases[-1]["expect"]["complies"]


@pytest.mark.parametrize("pid, free", [
    ("BitOf16x0_v1", True), ("BitOf1x0_v2", True), ("BitOf32x31_v1", True), ("WordOf1_v1", True), ("WordOf64_v2", True),
    ("BitOf16x01_v1", False), ("BitOf00x0_v1", False), ("BitOf16x0_v3", False), ("BitOf16x0_v1 ", False),
    ("BitOf16x0_v1\n", False), ("WordOf16_v12", False), ("WordOf_v1", False), ("BitOfx0_v1", False), ("BitOf16x_v1", False),
    ("WordOf١٦_v1", False), ("bitof16x0_v1", False)])
def test_the_wiring_ids_are_exactly_the_populations(pid, free):
    """`WIRING_ID` frees exactly the ids `verity.circuits.population` spells, at version 1 or 2: a leading zero, another
    version, a trailing newline (which `re`'s `$` would let through) or a non-ASCII digit is a neighbour, which weighs."""
    from verity.circuits import population
    assert (reference.WIRING_ID.fullmatch(pid) is not None) == free
    if free:
        v1 = pid.removesuffix("_v2") + "_v1" if pid.endswith("_v2") else pid
        assert population.BIT_OF_ID.match(v1) or population.WORD_OF_ID.match(v1)
