"""The liveness cases of the description-sized NCI check (`NciCheck.Desc`): the Python half.

`verity/core/protocols/network/nci/lean/scripts/desc_liveness_vectors.py` writes 13 Program shapes, each at a small `n`
and at one of `2^22` to `2^52 + 1` scan iterations or batch members, past anything `NciCheck.Desc` walks one by one.
`check` runs the Lean half in its Lean audit (`runs` in that package's `lean-audit.json`): `NciCheck/DescLivenessMain.lean`
fails unless `NciCheck.Desc.check` gives on every small instance what `NciCheck.check` gives on it expanded, and on the
big one that verdict (or its novel reads). The audit checks the data against the record's SHA-256 and caches its pass on
the Lean package's files alone, so the first test is what says the data is still what the generator writes; the second
that `scripts/nci_reference.py` gives each small instance a verdict, the unrolled check's.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

LEAN = Path(__file__).resolve().parents[1] / "nci" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("desc_liveness_vectors")
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NciCheck.DescLivenessMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{desc_liveness/cases.jsonl}", str(len(vectors.CASES))]


def test_the_small_instances_get_the_unrolled_verdict():
    reference = _load("desc_difftest_vectors").reference
    for c in vectors.cases():
        got = reference.check_desc(c["small"])
        assert "error" not in got, c["id"]
        assert got == reference.check(reference.expand(c["small"])), c["id"]
