"""The difftest of θ's provenance check (`NciCheck.Baseline`): the Python half.

`verity/core/protocols/network/nci/lean/scripts/baseline_difftest_vectors.py` writes 1,213 declarations with θ, θ's
provenance and the service's trusted input (the deployment's registration log, the run and its parameters), each expanded
gate by gate where the policy check gives a verdict, and `scripts/nci_reference.py`'s `check_baseline` on each. `check`
runs the Lean half in its Lean audit (`runs` in that package's `lean-audit.json`): `NciCheck/BaselineDifftestMain.lean`
fails unless `NciCheck.Baseline.check` gives Python's result on every case and `NciCheck.Baseline.checkUnrolled` the same
on every expansion. As for the other two NCI difftests, the audit checks the data against the record's SHA-256 and caches
its pass on the Lean package's files alone, so the first test is what says the data is still what the generator and the
reference write. The second checks that the random cases reach every refusal code and an acceptance, the third that each
pinned case gets the code it was built for, gate by gate as well, the fourth that an entry of the `output` kind is
refused as any unknown kind is, and the fifth that θ cites the log's first record, from the first run and from a later
one, and never a record registered mid-deployment.
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
    """All 1,213 cases, their expansions and the reference's results, byte for byte as the record pins them."""
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
    """Two accepted, one per refusal code, an entry of the `output` kind, θ holding an input bound per run
    (`theta-not-registered`), then a field of another JSON type than the string or null the check compares it with
    (`TYPED`); the unrolled check on each expansion gives the same."""
    pinned = vectors.pinned()
    assert [p["malformed"] for p in pinned] == [None, None, *vectors.CODES, "theta-provenance", "theta-not-registered",
                                                *(code for _, code in vectors.TYPED)]
    for p in pinned:
        inst, tr = p["case"]["instance"], p["case"]["trusted"]
        out = reference.check_baseline(inst, tr)
        assert out.get("error") == p["malformed"]
        if (e := vectors.expanded(p["case"])) is not None:
            assert reference.check_baseline_unrolled(e["instance"], e["trusted"]) == out


def test_an_output_entry_is_refused_as_an_unknown_kind():
    """θ holds registered public data only, never an earlier run's output. The pinned `output` entry
    cites a port that an earlier record of the log holds under the root the run registers the parameter with, in the form
    the check took until then: it is refused `theta-provenance`, the code of an entry of an unknown kind, while the same
    record's registered value of that root, cited as `registered`, is accepted."""
    [p] = [p for p in vectors.pinned() if any(e.get("kind") == "output" for e in p["case"]["instance"].get("provenance", []))]
    inst, tr = p["case"]["instance"], p["case"]["trusted"]
    t, out = next((t, e) for t, e in enumerate(inst["provenance"]) if e["kind"] == "output")
    assert set(out) == {"kind", "record", "port", "root"}
    digests = [reference.record_digest(r) for r in tr["log"]]
    src = tr["log"][digests.index(out["record"])]
    assert reference.last_with(src, "roots", "port", out["port"])["root"] == out["root"]
    assert reference.check_baseline(inst, tr)["error"] == "theta-provenance"
    unknown = inst | {"provenance": [*inst["provenance"][:t], out | {"kind": "weights"}, *inst["provenance"][t + 1:]]}
    assert reference.check_baseline(unknown, tr)["error"] == "theta-provenance"
    value = next(e["value"] for e in src["registered"] if e["root"] == out["root"])
    registered = {"kind": "registered", "registration": out["record"], "value": value, "root": out["root"]}
    fixed = inst | {"provenance": [*inst["provenance"][:t], registered, *inst["provenance"][t + 1:]]}
    assert reference.check_baseline(fixed, tr) == reference.check_desc(inst)


def test_theta_cites_the_first_record_of_the_deployments_log():
    """One registration log is one deployment, and θ cites its first record. The two pinned
    acceptances cite it from the deployment's first run, whose own record it is, and from its third. The pinned
    `theta-not-at-deployment-start` case cites, from the third run, the second record, which registers the same value
    under the same root and precedes the run's own, so the rule it replaces (a record before the run's own) took it; cited
    at the log's first record instead, it is accepted."""
    pinned = vectors.pinned()
    first, third = pinned[0], pinned[1]
    [mid] = [p for p in pinned if p["malformed"] == "theta-not-at-deployment-start"]
    for p, run in ((first, 0), (third, 2)):
        inst, tr = p["case"]["instance"], p["case"]["trusted"]
        digests = [reference.record_digest(r) for r in tr["log"]]
        assert tr["log"][0]["previous"] is None and digests.index(tr["run"]) == run
        assert inst["provenance"] and all(e["registration"] == digests[0] for e in inst["provenance"])
        assert reference.check_baseline(inst, tr) == reference.check_desc(inst)
    inst, tr = mid["case"]["instance"], mid["case"]["trusted"]
    digests = [reference.record_digest(r) for r in tr["log"]]
    t, e = next((t, e) for t, e in enumerate(inst["provenance"]) if e["registration"] != digests[0])
    assert digests.index(e["registration"]) == 1 and digests.index(tr["run"]) == 2
    assert reference.last_with(tr["log"][1], "registered", "value", e["value"])["root"] == e["root"]
    out = reference.check_baseline(inst, tr)
    assert out["error"] == "theta-not-at-deployment-start"
    first_record = e | {"registration": digests[0]}
    fixed = inst | {"provenance": [*inst["provenance"][:t], first_record, *inst["provenance"][t + 1:]]}
    assert reference.check_baseline(fixed, tr) == reference.check_desc(inst)
