"""The difftest of the description-sized NCI check (`NciCheck.Desc`) against the unrolled one: the Python half.

`verity/core/protocols/network/nci/lean/scripts/desc_difftest_vectors.py` writes 3,648 declarations divided into IUs of
root nodes, each expanded gate by gate where it is well-formed, and `scripts/nci_reference.py`'s `check_desc` on each.
`check` runs the Lean half in its Lean audit (`runs` in that package's `lean-audit.json`): `NciCheck/DescDifftestMain.lean`
fails unless `NciCheck.Desc.check` gives Python's result on every instance and `NciCheck.check`'s on every expansion. As
for `test_nci_lean_difftest.py`, the audit checks the data against the record's SHA-256 and caches its pass on the Lean
package's files alone, so this test is what says the data is still what the generator and the reference write. The
other test checks that the generator reaches what its docstring says, on the first 400 instances and the first on each
fixture's descriptor.
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
    """All 3,648 instances, their expansions and the reference's results, byte for byte as the record pins them."""
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
