"""The difftest of the description-sized NCI check (`NciCheck.Desc`) against the unrolled one: the Python half.

`verity/core/protocols/network/nci/lean/scripts/desc_difftest_vectors.py` writes 3,662 declarations divided into IUs of
root nodes (six on a composite keyed with an input id, which is no input node, two on a gate reading a nested scan's
carry, one past `NciCheck.Desc.walkBound`'s limit, four on a scan of no iterations, then two pinned: θ with an input
bound per run, refused, and θ the registered one, accepted), each expanded gate by gate where it is well-formed, and
`scripts/nci_reference.py`'s `check_desc` on each.
`check` runs the Lean half in its Lean audit (`runs` in that package's `lean-audit.json`): `NciCheck/DescDifftestMain.lean`
fails unless `NciCheck.Desc.check` gives Python's result on every instance and `NciCheck.check`'s on every expansion. As
for `test_nci_lean_difftest.py`, the audit checks the data against the record's SHA-256 and caches its pass on the Lean
package's files alone, so this test is what says the data is still what the generator and the reference write. The
other test checks that the generator reaches what its docstring says, on the first 400 instances and the first on each
fixture's descriptor, the third that the data holds the decoys: a root Call or batch of a composite keyed with an input
id, whose gates are computed, which `NciCheck.Desc` once took for an input node, the fourth that it holds the walks: a
gate reading a nested scan's carry, past `NciCheck.Desc.walkBound`'s limit, where `Prog.gate` runs out of steps, and the
fifth that it holds the empty scans: a scan of no iterations carrying a value, which `Flock.Program.resolve` can't
resolve, refused read or not.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[1] / "nci" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("desc_difftest_vectors")
reference = vectors.reference
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NciCheck.DescDifftestMain"]


@pytest.mark.slow
def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 3,662 instances, their expansions and the reference's results, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{desc_difftest/instances.jsonl}", "{desc_difftest/expanded.jsonl}",
                           "{desc_difftest/python.jsonl}", str(vectors.total())]


def test_the_generator_covers_what_it_says():
    fixtures = vectors.fixtures()
    seeds = [*range(400), *(vectors.COUNT + i * vectors.PER_FIXTURE for i in range(len(fixtures)))]
    made = [vectors.instance(s) for s in seeds]
    results = [reference.check_desc(m["instance"]) for m in made]
    shapes = {m["shape"] for m in made}
    assert set(vectors.vectors.SHAPES) | {"transposed"} <= shapes
    assert {f"fixture:{name}" for name, _ in fixtures} <= shapes
    assert {"fixture:partition_v2_vectors", "fixture:template_instance_vectors", "fixture:qcall_vectors"} <= shapes
    assert {m["theta_kind"] for m in made} == {"none", "all", "registered", "random"}
    assert {m["division"] for m in made} == {"single", "singletons", "runs", "random"}
    assert {r.get("error") for r in results} == set(vectors.MALFORMED) | {None}
    good = [(m, r) for m, r in zip(made, results, strict=True) if "error" not in r]
    assert {r["failure"][1] for _, r in good if r["failure"]} == {"size", "nci"}
    assert any(r["complies"] for _, r in good)
    assert any(m["shape"] == "transposed" for m, _ in good) and any(m["shape"].startswith("fixture:") for m, _ in good)
    wide = [m for m, _ in good if len(vectors.expanded(m["instance"], {})["owner"]) > len(m["instance"]["owner"])]
    assert wide, "an instance whose expansion has more gates than root nodes"


def test_the_decoys_are_no_input_nodes():
    """The decoys come just before the walks, and the reference refuses each one the unrolled check refuses: the decoy
    node in θ (`theta-not-input`) or in no IU (`gate-in-no-iu`), and gives a verdict with the decoy node in an IU."""
    first = (vectors.total() - len(vectors.vectors.pinned()) - len(vectors.empties()) - len(vectors.walks())
             - len(vectors.decoys()))
    made = [vectors.instance(s)["instance"] for s in range(first, first + len(vectors.decoys()))]
    assert made == vectors.decoys()
    got = [reference.check_desc(m).get("error") for m in made]
    assert got == ["theta-not-input", "gate-in-no-iu", None] * 2
    for m in made:
        assert reference.check_desc(m) == reference.check(reference.expand(m))


def test_the_walk_past_the_bound_is_refused():
    """The walks come just before the empty scans: the reference gives the unrolled check's verdict on the small
    one, and refuses (`program`) the one of about 2^64 activations, whose gate `Prog.gate` can't locate in `Steps.FUEL`
    steps."""
    first = vectors.total() - len(vectors.vectors.pinned()) - len(vectors.empties()) - len(vectors.walks())
    made = [vectors.instance(s)["instance"] for s in range(first, first + len(vectors.walks()))]
    assert made == vectors.walks()
    small, big = made
    assert reference.check_desc(small) == reference.check(reference.expand(small))
    assert "error" not in reference.check_desc(small)
    assert reference.check_desc(big) == {"error": "program"}
    bound = []
    for m in made:
        desc = json.loads(m["program"])
        bound.append(reference.walk_bound(reference.codec.decode_program(desc, reference.declared(desc))))
    assert bound[0] == (1 + 3 * (1 + 3) + 1) * (3 + 2)
    assert bound[1] > 2 ** 64 > reference.FUEL


def test_the_empty_scans_are_refused_where_they_carry_a_value():
    """The empty scans come just before the pinned instances: the reference refuses (`program`) the three whose scan
    carries a value, read or not, and gives the unrolled check's verdict on the one whose carry is empty."""
    first = vectors.total() - len(vectors.vectors.pinned()) - len(vectors.empties())
    made = [vectors.instance(s)["instance"] for s in range(first, first + len(vectors.empties()))]
    assert made == vectors.empties()
    *refused, bare = made
    assert [reference.check_desc(m) for m in refused] == [{"error": "program"}] * 3
    assert "error" not in reference.check_desc(bare)
    assert reference.check_desc(bare) == reference.check(reference.expand(bare))


def test_the_pinned_instances_get_their_results():
    """The data's last two lines, `difftest_vectors.pinned()`'s: θ holding an input node bound per run is refused
    (`theta-not-registered`), and θ of the registered one alone gets its verdict, which the unrolled check gives too."""
    for p in vectors.vectors.pinned():
        assert reference.check_desc(p["instance"]) == p["expect"]
        assert reference.check(reference.expand(p["instance"])) == p["expect"]
