"""The campaign result contract: what it accepts, every way it says no, and that the workload emits conforming results."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from conftest import REPO, VECTOR_RUN, conforming_result

from verity.verification.target import FIRST, K16_VECTOR
from verity_numerical.bench import contract

FIXTURES = REPO / "fixtures" / "typed-obligation-v0"


def scale(m: dict, value: float) -> dict:
    return {**m, "value": value}


def with_measurements(result: dict, *replace: dict, drop: tuple[str, ...] = ()) -> dict:
    names = {m["name"] for m in replace}
    kept = [m for m in result["measurements"] if m["name"] not in names and m["name"] not in drop]
    return {**result, "measurements": kept + list(replace)}


# --- validate --------------------------------------------------------------------------------------------------------


def test_conforming_result_validates(conforming: dict) -> None:
    assert contract.validate(conforming) == []


def test_every_missing_fingerprint_field_is_named(conforming: dict) -> None:
    problems = contract.validate({**conforming, "workload_fingerprint": {}})
    for field in contract.REQUIRED_FIELDS:
        assert f"workload_fingerprint.{field} missing" in problems, field
    assert len([p for p in problems if p.endswith(" missing")]) == len(contract.REQUIRED_FIELDS)
    # one field at a time, too
    for field in contract.REQUIRED_FIELDS:
        fp = {k: v for k, v in conforming["workload_fingerprint"].items() if k != field}
        assert contract.validate({**conforming, "workload_fingerprint": fp}) == [f"workload_fingerprint.{field} missing"], field


def test_envelope_problems_are_reported(conforming: dict) -> None:
    assert "schema must be 'research/result/v0.1', got 'x'" in contract.validate({**conforming, "schema": "x"})
    assert "run_id must be a non-empty string" in contract.validate({**conforming, "run_id": ""})
    assert any(p.startswith("validation must be") for p in contract.validate({**conforming, "validation": {"status": "maybe"}}))
    assert contract.validate("nope") == ["result must be an object"]


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("profile", "another-target", "profile must be the campaign target"),
        ("profile", None, "profile must be the campaign target"),
        ("domain", "", "domain must be a non-empty string"),
        ("K", 24, "K must be a positive multiple of 16"),
        ("K", None, "K must be a positive multiple of 16"),
        ("K", True, "K must be a positive multiple of 16"),
        ("B", 0, "B must be a positive int"),
        ("proof_class", "ZK_PROOF", "proof_class must be one of"),
        ("authentication", "partial", "authentication must be one of"),
        ("zk_mode", "", "zk_mode must be a non-empty string"),
        ("security", {"note": "no target key"}, "security.target missing"),
        ("security", {"target": 100}, "security.target is log2 of the soundness error"),
        ("security", {"target": "2^-100"}, "security.target is log2 of the soundness error"),
        ("hardware", {"gpu": None}, "hardware.cpu missing"),
        ("hardware", {"gpu": {"driver": "x", "cuda": None}, "cpu": None}, "hardware.gpu.name must be a non-empty string"),
        ("hardware", {"gpu": {"name": "g", "cuda": None}, "cpu": None}, "hardware.gpu.driver missing"),
        ("software", {"toolchain": {}}, "software.backend must be an object"),
        ("software", {"backend": {"name": "x"}, "toolchain": {}}, "software.backend needs version and/or commit"),
        ("software", {"backend": {"name": "x", "version": "1"}}, "software.toolchain must be an object"),
        ("preprocessing", "", "preprocessing must be a non-empty string or an object"),
        ("preprocessing", None, "preprocessing must be a non-empty string or an object"),
        ("retention", "kept", "retention must be one of"),
    ],
)
def test_each_fingerprint_rule(conforming: dict, field: str, value, fragment: str) -> None:
    fp = {**conforming["workload_fingerprint"], field: value}
    problems = contract.validate({**conforming, "workload_fingerprint": fp})
    assert len(problems) == 1 and fragment in problems[0], problems


def test_b_real_is_optional_but_bounded(conforming: dict) -> None:
    fp = conforming["workload_fingerprint"]
    assert contract.validate_fingerprint({**fp, "B": 8, "B_real": 5}) == []
    assert contract.validate_fingerprint({**fp, "B": 8, "B_real": 9}) == ["B_real (9) exceeds B (8)"]
    assert contract.validate_fingerprint({**fp, "B_real": 0}) == ["B_real must be a positive int when present, got 0"]


def test_zk_mode_must_agree_with_proof_class(conforming: dict) -> None:
    fp = conforming["workload_fingerprint"]
    assert contract.validate_fingerprint({**fp, "proof_class": "COMPLETE_ZK_BACKEND", "zk_mode": "none"}) == [
        "proof_class COMPLETE_ZK_BACKEND with zk_mode 'none' is a contradiction"
    ]
    assert contract.validate_fingerprint({**fp, "proof_class": "COMPLETE_ZK_BACKEND", "zk_mode": "masked-sumcheck"}) == []
    problems = contract.validate_fingerprint({**fp, "proof_class": "NON_ZK_PROOF", "zk_mode": "masked-sumcheck"})
    assert problems == ["proof_class NON_ZK_PROOF is not zero-knowledge; zk_mode must be 'none', got 'masked-sumcheck'"]


# --- measurements: buckets, joints, double counting ----------------------------------------------------------------


def test_bucket_inside_a_joint_is_double_counting(conforming: dict) -> None:
    doubled = with_measurements(conforming, contract.measurement("t.arithmetic", 0.1))
    problems = contract.validate(doubled)
    assert "bucket 'arithmetic' is counted twice: in 't.arithmetic' and in 't.joint.encoding_commitment+arithmetic'" in problems


def test_two_joints_sharing_a_bucket_is_double_counting(conforming: dict) -> None:
    doubled = with_measurements(conforming, contract.measurement(contract.joint("arithmetic", "lookup"), 0.1), drop=("t.lookup",))
    problems = contract.validate(doubled)
    assert any(p.startswith("bucket 'arithmetic' is counted twice") for p in problems)


def test_buckets_exceeding_total_by_over_one_percent_is_double_counting(conforming: dict) -> None:
    # buckets + joints = 3.75 s; t.total = 4.0 s conforms; 3.75 / 1.01 rounds down to a total just inside the tolerance
    ok = with_measurements(conforming, contract.measurement("t.total", 3.75 / 1.01 + 1e-9), drop=("vu.seconds_per_vu", "vu.per_second"))
    assert contract.validate(ok) == []
    bad = with_measurements(conforming, contract.measurement("t.total", 3.75 / 1.02), drop=("vu.seconds_per_vu", "vu.per_second"))
    problems = contract.validate(bad)
    assert len(problems) == 1 and problems[0].startswith("buckets and joints sum to 3.75 s, more than t.total 3.67647 s by over 1%"), problems


def test_unbucketed_overhead_below_total_is_fine(conforming: dict) -> None:
    loose = with_measurements(conforming, contract.measurement("t.total", 40.0), contract.measurement("vu.seconds_per_vu", 40.0), contract.measurement("vu.per_second", 0.025))
    assert contract.validate(loose) == []


def test_vu_throughput_must_match_total(conforming: dict) -> None:
    off = with_measurements(conforming, contract.measurement("vu.seconds_per_vu", 3.0))
    assert contract.validate(off) == ["vu.seconds_per_vu 3 is not t.total / vu.count = 4"]
    off = with_measurements(conforming, contract.measurement("vu.per_second", 1.0))
    assert contract.validate(off) == ["vu.per_second 1 is not vu.count / t.total = 0.25"]


def test_a_proof_needs_total_and_a_bucket(conforming: dict) -> None:
    problems = contract.validate({**conforming, "measurements": [contract.measurement("own_name", 1, "cycles")]})
    assert problems == ["a NON_ZK_PROOF result must report t.total", "a NON_ZK_PROOF result must report at least one timing bucket or joint stage"]
    no_proof = {**conforming, "workload_fingerprint": {**conforming["workload_fingerprint"], "proof_class": "NO_PROOF"}, "measurements": []}
    assert contract.validate(no_proof) == []


def test_non_zk_proof_cannot_spend_zk_time(conforming: dict) -> None:
    problems = contract.validate(with_measurements(conforming, contract.measurement("t.zk_additional", 0.1), contract.measurement("t.total", 4.1),
                                                   contract.measurement("vu.seconds_per_vu", 4.1), contract.measurement("vu.per_second", 1 / 4.1)))
    assert problems == ["t.zk_additional is 0.1 but proof_class NON_ZK_PROOF has no zero-knowledge work"]


def test_units_names_and_shapes(conforming: dict) -> None:
    bad = [
        {"name": "t.total", "value": 1.0, "unit": "ms"},
        {"name": "t.witnes", "value": 1.0, "unit": "s"},
        {"name": "vu.count", "value": "1", "unit": "count"},
        {"name": "t.joint.witness", "value": 1.0, "unit": "s"},
        {"name": "t.joint.witness+witness", "value": 1.0, "unit": "s"},
        {"name": "t.joint.arithmetic+witness", "value": 1.0, "unit": "s"},
        {"name": "t.lookup", "value": -1.0, "unit": "s"},
        {"name": "t.lookup", "value": 1.0, "unit": "s"},
        {"name": "shape", "value": 1},
        "not a dict",
    ]
    problems = contract.validate_measurements(bad)
    assert "measurement 't.total' must be in 's', got 'ms'" in problems
    assert "measurement 't.witnes' uses a contract prefix but is not a contract name" in problems
    assert "measurement 'vu.count' must be a number, got '1'" in problems
    assert any("a joint stage fuses at least two buckets" in p for p in problems)
    assert any("lists each bucket once" in p for p in problems)
    assert "joint stage 't.joint.arithmetic+witness' must be written in canonical bucket order: 't.joint.witness+arithmetic'" in problems
    assert "measurement 't.lookup' must be non-negative, got -1.0" in problems
    assert "measurement 't.lookup' appears more than once" in problems
    assert "measurements[8] needs name (str), value, unit (str)" in problems
    assert "measurements[9] needs name (str), value, unit (str)" in problems
    assert contract.validate_measurements("x") == ["measurements must be a list"]


# --- builders --------------------------------------------------------------------------------------------------------


def test_joint_is_canonical_and_checked() -> None:
    assert contract.joint("lookup", "witness", "arithmetic") == "t.joint.witness+arithmetic+lookup"
    assert contract.parse_joint("t.joint.witness+lookup") == ("witness", "lookup")
    assert contract.parse_joint("t.total") is None
    with pytest.raises(ValueError, match="at least two"):
        contract.joint("witness")
    with pytest.raises(ValueError, match="not \\['fri'\\]"):
        contract.joint("witness", "fri")
    assert contract.unit_of("t.joint.witness+lookup") == "s" and contract.unit_of("t.joint.witness") is None and contract.unit_of("cycles.x") is None


def test_measurement_supplies_and_checks_units() -> None:
    assert contract.measurement("t.total", 1.5) == {"name": "t.total", "value": 1.5, "unit": "s"}
    assert contract.measurement("transcript.bytes", 10) == {"name": "transcript.bytes", "value": 10, "unit": "B"}
    assert contract.measurement("prove_seconds", 1.0, "s") == {"name": "prove_seconds", "value": 1.0, "unit": "s"}
    with pytest.raises(ValueError, match="give its unit"):
        contract.measurement("prove_seconds", 1.0)
    with pytest.raises(ValueError, match="measured in 's', not 'ms'"):
        contract.measurement("t.total", 1.0, "ms")
    with pytest.raises(ValueError, match="must be a number"):
        contract.measurement("t.total", True)
    with pytest.raises(ValueError, match="at least two"):
        contract.measurement("t.joint.witness", 1.0)


def test_fingerprint_builder_emits_conforming_and_refuses_bad(conforming: dict) -> None:
    fields = dict(conforming["workload_fingerprint"])
    fp = contract.fingerprint(**fields, vector="x", B_real=1)
    assert contract.validate_fingerprint(fp) == [] and fp["vector"] == "x" and fp["B_real"] == 1
    with pytest.raises(ValueError, match="proof_class must be one of"):
        contract.fingerprint(**{**fields, "proof_class": "SOUND"})
    with pytest.raises(ValueError, match="K must be a positive multiple of 16.*retention must be one of"):
        contract.fingerprint(**{**fields, "K": 10, "retention": "kept"})
    with pytest.raises(TypeError):
        contract.fingerprint(**{k: v for k, v in fields.items() if k != "hardware"})  # a required field cannot be left out


def test_probe_hardware_shape() -> None:
    hw = contract.probe_hardware()
    assert set(hw) == {"gpu", "cpu"}
    assert contract._validate_hardware(hw) == []


# --- comparability -----------------------------------------------------------------------------------------------------


def test_incomparable_lists_differences_and_missing(conforming: dict) -> None:
    fp = conforming["workload_fingerprint"]
    assert contract.incomparable(fp, fp) == []
    assert contract.incomparable(fp, {**fp, "K": 1536}) == [("K", 16, 1536)]
    assert contract.incomparable(fp, {**fp, "K": 1536}, allow=["K"]) == []
    other = {**fp, "proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded"}
    assert [f for f, _, _ in contract.incomparable(fp, other)] == ["proof_class", "authentication"]
    lacking = {k: v for k, v in fp.items() if k != "domain"}
    assert contract.incomparable(fp, lacking) == [("domain", "finite", contract.MISSING)]
    with pytest.raises(ValueError, match="allow takes identity fields"):
        contract.incomparable(fp, fp, allow=["B"])


# --- the workload emits conforming results ----------------------------------------------------------------------------


def _vector_run():
    spec = importlib.util.spec_from_file_location("vector_run_under_test", VECTOR_RUN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_campaign_fields_read_off_the_object() -> None:
    vr = _vector_run()
    rec = json.loads((FIXTURES / f"{K16_VECTOR}.json").read_text())
    assert vr.campaign_fields(rec, "sp1-prove", False) == {
        "profile": FIRST.name, "domain": "finite", "K": 16, "B": 1, "proof_class": "NON_ZK_PROOF", "authentication": "included",
        "statement_machinery": "included",
    }
    diagnostic = vr.campaign_fields(rec, "sp1-prove", True)
    assert diagnostic["proof_class"] == "ARITHMETIC_DIAGNOSTIC" and diagnostic["authentication"] == "excluded"
    assert diagnostic["statement_machinery"] == "included", "skip-authentication still parses the statement and builds the table"
    relation_only = vr.campaign_fields(rec, "sp1-prove", True, True)
    assert relation_only["proof_class"] == "ARITHMETIC_DIAGNOSTIC" and relation_only["authentication"] == "excluded"
    assert relation_only["statement_machinery"] == "excluded"
    assert vr.campaign_fields(rec, "sp1-execute", False)["proof_class"] == "NO_PROOF"
    hw = json.loads((FIXTURES / "accept--tc-store-cast-nan-hw.json").read_text())
    assert vr.campaign_fields(hw, "sp1-prove", False)["profile"] is None, "family @2 is not the campaign target: null, not a guess"
    many = json.loads((FIXTURES / "accept--shared-leaf-many-obligations.json").read_text())
    assert vr.campaign_fields(many, "reference", False)["B"] == 4


def test_sp1_prove_mapping_is_one_joint_and_validates(conforming: dict) -> None:
    vr = _vector_run()
    report = SimpleNamespace(setup_seconds=17.35, prove_seconds=3.58, verify_seconds=0.245, proof_bytes=5898276)
    ms = vr.sp1_prove_measurements(report)
    by_name = {m["name"]: m for m in ms}
    assert vr.SP1_PROVE_STAGE == "t.joint.witness+encoding_commitment+arithmetic+lookup"
    assert by_name[vr.SP1_PROVE_STAGE]["value"] == 3.58 and by_name["t.total"]["value"] == 3.58
    assert by_name["fixed.setup_seconds"]["value"] == 17.35 and by_name["verifier.seconds"]["value"] == 0.245
    assert by_name["transcript.bytes"] == {"name": "transcript.bytes", "value": 5898276, "unit": "B"}
    assert by_name["t.zk_additional"]["value"] == 0.0
    assert "t.serialization" not in by_name, "the proof's write to disk is untimed by the host: absent, not guessed"
    result = {**conforming, "measurements": ms + [contract.measurement("vu.count", 1), contract.measurement("prove_seconds", 3.58, "s")]}
    assert contract.validate(result) == []


def test_reference_run_writes_a_conforming_result(tmp_path) -> None:
    out = tmp_path / "run"
    proc = subprocess.run([sys.executable, str(VECTOR_RUN), "--vector", K16_VECTOR, "--backend", "reference", "--out", str(out)],
                          capture_output=True, text=True, timeout=120, env={**os.environ, "RESEARCH_RUN_ID": "r-local-test"})
    assert proc.returncode == 0, proc.stderr
    result = json.loads((out / "result.json").read_text())
    assert result["run_id"] == "r-local-test"
    assert contract.validate(result) == [], contract.validate(result)
    fp = result["workload_fingerprint"]
    assert (fp["profile"], fp["domain"], fp["K"], fp["B"], fp["proof_class"], fp["authentication"], fp["zk_mode"], fp["retention"]) == (
        FIRST.name, "finite", 16, 1, "NO_PROOF", "included", "none", "retained")
    assert fp["security"]["target"] is None and set(fp["hardware"]) == {"gpu", "cpu"}
    assert fp["software"]["backend"]["name"] == "verity.verification.transparent"
    assert {m["name"] for m in result["measurements"]} >= {"reference_check_wall", "vu.count", "workload_wall"}
    assert json.loads(proc.stdout)["contract"] == "conforms"


def test_hvzk_is_a_zk_class_but_not_the_complete_label(conforming: dict) -> None:
    """Honest-verifier ZK carries a zk_mode like the complete class, and is a distinct label the judge keeps apart."""
    fp = conforming["workload_fingerprint"]
    assert "COMPLETE_HVZK_BACKEND" in contract.PROOF_CLASSES and "COMPLETE_HVZK_BACKEND" in contract.ZK_CLASSES
    assert contract.validate_fingerprint({**fp, "proof_class": "COMPLETE_HVZK_BACKEND", "zk_mode": "ligero-masked"}) == []
    assert contract.validate_fingerprint({**fp, "proof_class": "COMPLETE_HVZK_BACKEND", "zk_mode": "none"}) == [
        "proof_class COMPLETE_HVZK_BACKEND with zk_mode 'none' is a contradiction"
    ]
    a = {**fp, "proof_class": "COMPLETE_HVZK_BACKEND", "zk_mode": "m"}
    b = {**fp, "proof_class": "COMPLETE_ZK_BACKEND", "zk_mode": "m"}
    assert contract.incomparable(a, b, ()) , "HVZK and malicious-verifier ZK results must never be compared as equals"


def test_n_subbatches_is_written_and_validated(conforming: dict) -> None:
    fp = conforming["workload_fingerprint"]
    built = contract.fingerprint(**{k: fp[k] for k in contract.REQUIRED_FIELDS})
    assert built["N_subbatches"] == 1
    assert contract.fingerprint(**{k: fp[k] for k in contract.REQUIRED_FIELDS}, N_subbatches=16)["N_subbatches"] == 16
    assert contract.validate_fingerprint({**fp, "N_subbatches": 0}) == ["N_subbatches must be a positive int when present, got 0"]


def test_latency_and_census_names_are_defined_with_units() -> None:
    for name, unit in (("rounds.protocol", "count"), ("rounds.sequential_depth", "count"), ("t.e2e.rtt_0ms", "s"),
                       ("t.e2e.rtt_0.1ms", "s"), ("t.e2e.rtt_50ms", "s"), ("census.committed_elements", "count"),
                       ("census.bytes_encoded", "B"), ("census.hash_bytes", "B")):
        assert contract.unit_of(name) == unit, name
    # the bare name is not a contract name (protocol rounds and sequential depth are different quantities)
    assert contract.unit_of("rounds") is None
    assert contract.validate_measurements([contract.measurement("rounds.made_up", 480, "count")], None), "rounds. is reserved"
    assert not contract.validate_measurements([contract.measurement("rounds.sequential_depth", 480)], None)
