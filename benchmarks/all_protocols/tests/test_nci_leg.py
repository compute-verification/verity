"""The intensity leg (`legs/nci.py`) on a window the loop served:

* its case is NCI's on the window's own record and the verifier's program: the commitment is the registration record
  and the answer's commitment its digest, the program the one the registration names, every input node ``input`` and
  the units one IU, θ empty; the reference accepts it at the leg's policy, with the IU's size and novel reads in the
  profile;
* the same declaration doesn't comply at the GLM-5.3 demo's α = 2^-25, so the leg's policy is what it accepts at;
* a window whose circuit record is not the one the verifier registered is refused before any case is built.
"""
from __future__ import annotations

import json
from argparse import Namespace
from fractions import Fraction
from pathlib import Path

import pytest
from verity_all_protocols import runner as RN
from verity_all_protocols import table as T

from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.circuits import codec

NCI = T.load("nci")


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("nci")
    a = Namespace(law="subset:2", cap=8, requests="3", steps=1, tamper=set(), node="test-node", tools={}, pouw=None,
                  delta="1/100", start=0, windows=1, weights=None, session_jobs=1)
    d, _ = RN.loop_window(0, "test", a, tmp / "scratch", tmp / "out")
    return d


@pytest.mark.slow
def test_the_case_is_the_windows_declaration_and_the_reference_accepts_it(served, tmp_path):
    a = NCI.leg(served, "test-node", tmp_path)
    reg = json.loads((served / "registration.json").read_text())
    case = json.loads((tmp_path / "nci.case.json").read_text())
    assert T.check_answer(a) == a and a["verdict"] == "accept" and a["protocol"] == "verity/nci/v1"
    assert bytes.fromhex(case["commitment"]) == codec.canonical_json(reg) and a["commitment"] == R.digest(reg)
    p, e = case["params"], json.loads(bytes.fromhex(case["evidence"]))
    assert codec.program_digest(json.loads(p["program"])) == reg["program"]
    assert p["log"] == [] and reg["previous"] is None
    assert e["theta"] == [] == e["provenance"] and e["owner"][-1] == 0 and set(e["owner"][:-1]) == {"input"}
    [[size, novel]] = a["profile"]["ius"]
    assert Fraction(novel, size) == Fraction(2112, 9) <= Fraction(*NCI.ALPHA) and size <= NCI.GAMMA
    assert json.loads((tmp_path / "nci.json").read_text())["lean"] is None


@pytest.mark.slow
def test_the_declaration_does_not_comply_at_the_glm_demos_policy(served):
    case = NCI.case_of(*NCI.declaration(served))
    strict = {**case, "params": {**case["params"], "alpha": [1, 2 ** 25]}}
    a = NCI.answer(strict)
    assert a["verdict"] == "reject" and a["reason"].startswith("the division does not comply: isolation unit 0")


@pytest.mark.slow
def test_a_circuit_the_verifier_did_not_register_is_refused(served, tmp_path):
    d = tmp_path / "w0"
    d.mkdir()
    for f in served.iterdir():
        if f.is_file():
            (d / f.name).write_bytes(f.read_bytes())
    p = json.loads((d / "period.json").read_text())
    (d / "period.json").write_text(json.dumps({**p, "circuit": "00" * 64}))
    with pytest.raises(ValueError, match="not the circuit record the verifier registered"):
        NCI.leg(d, "test-node", tmp_path / "out")
