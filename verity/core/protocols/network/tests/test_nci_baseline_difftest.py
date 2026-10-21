"""The difftest of θ's provenance check (`NciCheck.Baseline`): the Python half.

`verity/core/protocols/network/nci/lean/scripts/baseline_difftest_vectors.py` writes 1,210 declarations with θ, θ's
provenance and the service's trusted input (the registration log, the accepted record digests, the run and its
parameters), each expanded gate by gate where the policy check gives a verdict, and `scripts/nci_reference.py`'s
`check_baseline` on each. `check` runs the Lean half in its Lean audit (`runs` in that package's `lean-audit.json`):
`NciCheck/BaselineDifftestMain.lean` fails unless `NciCheck.Baseline.check` gives Python's result on every case and
`NciCheck.Baseline.checkUnrolled` the same on every expansion. As for the other two NCI difftests, the audit checks the data
against the record's SHA-256 and caches its pass on the Lean package's files alone, so the first test is what says the data
is still what the generator and the reference write. The second checks that the random cases reach every refusal code and
an acceptance, and the third that each pinned case gets the code it was built for, gate by gate as well.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[1] / "nci" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("baseline_difftest_vectors")
reference = vectors.reference
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NciCheck.BaselineDifftestMain"]


@pytest.mark.slow
def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 1,210 cases, their expansions and the reference's results, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{baseline_difftest/cases.jsonl}", "{baseline_difftest/expanded.jsonl}",
                           "{baseline_difftest/python.jsonl}", str(vectors.total())]


def test_the_random_cases_reach_every_code():
    got = Counter()
    for seed in range(300):
        c = vectors.case(seed)["case"]
        out = reference.check_baseline(c["instance"], c["trusted"])
        got[out.get("error")] += 1
        if "error" not in out:
            assert out == reference.check_desc(c["instance"])
    assert set(got) == set(vectors.CODES) | {None}


def test_the_pinned_cases_get_their_codes():
    """One accepted, one per refusal code, then θ holding an input bound per run (`theta-not-registered`); the unrolled
    check on each expansion gives the same."""
    pinned = vectors.pinned()
    assert [p["malformed"] for p in pinned] == [None, *vectors.CODES, "theta-not-registered"]
    for p in pinned:
        inst, tr = p["case"]["instance"], p["case"]["trusted"]
        out = reference.check_baseline(inst, tr)
        assert out.get("error") == p["malformed"]
        if (e := vectors.expanded(p["case"])) is not None:
            assert reference.check_baseline_unrolled(e["instance"], e["trusted"]) == out
