"""The registration check's leg (`legs/nci.py`, `table.REGISTRATION`) on a window the loop served:

* its case is NCI's on the window's own record and the verifier's program: the commitment is the registration record
  and the answer's commitment its digest, the program the one the registration names, every input node ``input`` and
  the units one IU, θ empty; the reference accepts it at the composed verdict's policy, with the IU's size and novel
  reads in the profile;
* its policy report states what it measured (2,112 bits over 9 gates a unit, about 235 a gate, and its share of Γ), the
  placeholder label, and that the same declaration doesn't comply at the GLM-5.3 demo's α = 2^-25;
* a window whose circuit record is not the one the verifier registered is refused before any case is built;
* a window the runner's ``--registration-tamper`` names is checked on a record other than its own, one root changed:
  NCI accepts it, under a digest that isn't the window's registration's (onsite's mismatch).
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

NCI = T.load(T.REGISTRATION.component)


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
def test_the_policy_report_states_the_measured_reads_and_the_verdict_at_the_glm_demos_alpha(served, tmp_path):
    NCI.leg(served, "test-node", tmp_path)
    at = json.loads((tmp_path / "policy.json").read_text())
    assert at["label"] == "placeholder (toy)" and at["alpha"] == [256, 1] and at["gamma"] == 2 ** 44
    assert at["verdict"] == "complies" and at["at_glm"] == {"alpha": [1, 2 ** 25], "gamma": 2 ** 44,
                                                           "verdict": "doesn't comply"}
    [iu] = at["measured"]["ius"]
    assert Fraction(iu["novel_bits"], iu["gates"]) == Fraction(2112, 9) and iu["bits_a_gate"] == 234.67
    assert iu["gamma_use"] == iu["gates"] / 2 ** 44
    row = RN.row_of(T.REGISTRATION, NCI.answer(json.loads((tmp_path / "nci.case.json").read_text())), tmp_path)
    assert row["policy"] == at


@pytest.mark.slow
def test_the_declaration_does_not_comply_at_the_glm_demos_policy(served):
    case = NCI.case_of(*NCI.declaration(served))
    a = NCI.answer(NCI.at_alpha(case, NCI.GLM_ALPHA))
    assert a["verdict"] == "reject" and a["reason"].startswith("the division does not comply: isolation unit 0")


def window_with(served: Path, d: Path, **period) -> Path:
    """A copy of the served window in ``d``, its ``period.json`` updated with ``period``."""
    d.mkdir()
    for f in served.iterdir():
        if f.is_file():
            (d / f.name).write_bytes(f.read_bytes())
    p = json.loads((d / "period.json").read_text())
    (d / "period.json").write_text(json.dumps({**p, **period}))
    return d


@pytest.mark.slow
def test_a_circuit_the_verifier_did_not_register_is_refused(served, tmp_path):
    d = window_with(served, tmp_path / "w0", circuit="00" * 64)
    with pytest.raises(ValueError, match="not the circuit record the verifier registered"):
        NCI.leg(d, "test-node", tmp_path / "out")


@pytest.mark.slow
def test_a_tampered_registration_is_accepted_under_a_digest_that_is_not_the_windows(served, tmp_path):
    d = window_with(served, tmp_path / "w0", registration_tamper=True)
    a = NCI.leg(d, "test-node", tmp_path / "out")
    reg = json.loads((served / "registration.json").read_text())
    other = NCI.tampered(reg)
    assert [k for k in reg if reg[k] != other[k]] == ["roots"] and other["roots"][1:] == reg["roots"][1:]
    assert a["verdict"] == "accept" and a["commitment"] == R.digest(other) != R.digest(reg)
    assert bytes.fromhex(json.loads((tmp_path / "out" / "nci.case.json").read_text())["commitment"]) \
        == codec.canonical_json(other)
    assert json.loads((tmp_path / "out" / "nci.json").read_text())["tampered"] is True


def test_a_registration_tamper_is_in_the_runs_mode(tmp_path, capsys):
    assert RN.main(["--out", str(tmp_path), "--tools", "/dev/null", "--check", "--windows", "2",
                    "--registration-tamper", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["flags"]["registration_tamper"] == [1]
