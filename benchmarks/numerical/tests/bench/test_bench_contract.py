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

from verity.ml.circuits.targets.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, FP8_SM120_MMA, K16_VECTOR, NVFP4_SM120
from verity_numerical.bench import census, contract

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
        ("profile", "another-target", "profile must be a target of verity.ml.circuits.targets.target"),
        ("profile", None, "profile must be a target of verity.ml.circuits.targets.target"),
        ("domain", "", "domain must be one of"),
        ("domain", "exact", "domain must be one of"),
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


def test_authentication_included_hash_is_a_recognised_column(conforming: dict) -> None:
    """``authentication = included-hash`` (the operands hashed inside the proof, lane hash-relation) validates and is a
    different identity from ``excluded`` (the judge never compares the two columns as one)."""
    fp = conforming["workload_fingerprint"]
    hashed = {**conforming, "workload_fingerprint": {**fp, "authentication": "included-hash"}}
    assert contract.validate(hashed) == []
    assert [f for f, _, _ in contract.incomparable(fp, hashed["workload_fingerprint"])] == ["authentication"]
    assert "included-hash" in contract.AUTHENTICATION and "authentication" in contract.IDENTITY_FIELDS


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


def test_live_verifier_fields_are_additive_and_total_live_is_at_least_total(conforming: dict) -> None:
    # an old row (no live fields) still validates; the live fields validate with their units; t.total_live < t.total is refused
    assert contract.validate(conforming) == []
    live = with_measurements(conforming, contract.measurement("t.total_live", 4.7), contract.measurement("net.rtt_ms", 12.5, "ms"),
                             contract.measurement("net.messages", 42, "count"), contract.measurement("net.bytes_out", 123456, "B"),
                             contract.measurement("net.bytes_in", 4321, "B"), contract.measurement("net.wait_seconds", 0.7, "s"),
                             contract.measurement("verify.wall_s", 3.2, "s"), contract.measurement("verify.cpu_s", 3.1, "s"))
    assert contract.validate(live) == []
    below = with_measurements(conforming, contract.measurement("t.total_live", 3.0))
    assert contract.validate(below) == ["t.total_live 3 s is below t.total 4 s: it is t.total plus the network wait"]
    assert contract.unit_of("t.total_live") == "s" and contract.unit_of("net.rtt_ms") == "ms" and contract.unit_of("verify.cpu_s") == "s"
    with pytest.raises(ValueError):
        contract.measurement("t.total_live", 4.7, "ms")


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


def test_fingerprint_records_the_malloc_tunables(conforming: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    fields = dict(conforming["workload_fingerprint"])
    fields["software"] = {k: v for k, v in fields["software"].items() if k != "allocator"}
    monkeypatch.delenv("MALLOC_MMAP_MAX_", raising=False)
    monkeypatch.delenv("MALLOC_TRIM_THRESHOLD_", raising=False)
    assert contract.fingerprint(**fields)["software"]["allocator"] == {"MALLOC_MMAP_MAX_": "unset", "MALLOC_TRIM_THRESHOLD_": "unset"}
    monkeypatch.setenv("MALLOC_MMAP_MAX_", "0")
    monkeypatch.setenv("MALLOC_TRIM_THRESHOLD_", "1000000000000")
    fp = contract.fingerprint(**fields)
    assert fp["software"]["allocator"] == {"MALLOC_MMAP_MAX_": "0", "MALLOC_TRIM_THRESHOLD_": "1000000000000"}
    assert "allocator" not in fields["software"] and contract.validate_fingerprint(fp) == []
    given = {"MALLOC_MMAP_MAX_": "unset", "MALLOC_TRIM_THRESHOLD_": "unset"}
    assert contract.fingerprint(**{**fields, "software": {**fields["software"], "allocator": given}})["software"]["allocator"] == given


def test_pod_env_sets_the_malloc_tunables() -> None:
    text = (REPO / "archive" / "direct" / "ligero" / "pod_bootstrap.sh").read_text()
    env_sh = text.split("cat > $W/env.sh <<EOF", 1)[1].split("\nEOF", 1)[0]
    assert r"MALLOC_MMAP_MAX_=\${MALLOC_MMAP_MAX_:-0}" in env_sh
    assert r"MALLOC_TRIM_THRESHOLD_=\${MALLOC_TRIM_THRESHOLD_:-1000000000000}" in env_sh


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
    assert contract.incomparable(fp, lacking) == [("domain", "finite-only", contract.MISSING)]
    with pytest.raises(ValueError, match="allow takes identity fields"):
        contract.incomparable(fp, fp, allow=["zk_mode"])


def test_overhead_is_prover_seconds_over_native_peak_seconds(conforming: dict) -> None:
    """One K=1536 VU is 3072 FLOP; at the A100's 312 TFLOPS dense BF16 peak that is 9.85 ps (~1e11 VU/s), so a prover
    at 34 ms/VU is 3.5e9x overhead.  The reported ratio must agree with t.total / vu.count at the fingerprint's K."""

    assert contract.native_seconds_per_vu(1536) == pytest.approx(9.846e-12, rel=1e-3)
    assert contract.overhead(0.0344, 1536) == pytest.approx(3.49e9, rel=1e-2)
    with pytest.raises(ValueError):
        contract.overhead(0.0, 1536)
    with pytest.raises(ValueError):
        contract.native_seconds_per_vu(0)
    assert contract.unit_of("overhead.vs_native_peak") == "ratio"

    res = json.loads(json.dumps(conforming))
    fp = res["workload_fingerprint"]
    values = {m["name"]: m["value"] for m in res["measurements"]}
    want = contract.overhead(values["t.total"] / values["vu.count"], fp["K"])
    res["measurements"].append(contract.measurement("overhead.vs_native_peak", want))
    assert contract.validate(res) == []
    res["measurements"][-1]["value"] = want * 2
    assert any("overhead.vs_native_peak" in p for p in contract.validate(res))


# --- the native reference is the proved target's ---------------------------------------------------------------------


def test_native_reference_is_the_targets_anchor_peak_and_the_bf16_default_is_unchanged() -> None:
    """The default profile is FIRST, whose reference is the A100's 312 TFLOPS dense BF16 peak: every pre-existing number
    (9.846 ps per K=1536 VU; 340 us/VU = 3.45e7x) is unchanged."""

    ref = contract.native_reference()
    assert ref is contract.native_peak_of(FIRST) and ref is contract.native_reference(FIRST.name)
    assert ref.flop_per_second == 312e12 and ref.accumulate == "fp32" and "A100" in ref.device
    assert contract.native_seconds_per_vu(1536) == pytest.approx(9.846e-12, rel=1e-3)
    assert contract.native_seconds_per_vu(1536) == contract.native_seconds_per_vu(1536, FIRST.name) == contract.native_seconds_per_vu(1536, contract.PROFILE)
    assert contract.overhead(340e-6, 1536) == pytest.approx(3.45e7, rel=1e-2)
    assert contract.overhead(340e-6, 1536) == contract.overhead(340e-6, 1536, FIRST.name)
    # the pre-native_reference module attribute is still the campaign target's peak, as a plain JSON-able mapping
    assert contract.NATIVE_REFERENCE["flop_per_second"] == ref.flop_per_second == 312e12
    assert contract.NATIVE_REFERENCE["device"] == ref.device and contract.NATIVE_REFERENCE["profile"] == contract.PROFILE
    json.dumps(contract.NATIVE_REFERENCE)
    assert contract.PROFILES == (FIRST.name, BF16_HOPPER.name, FP8_HOPPER_WGMMA.name, FP8_ADA_MMA.name, NVFP4_SM120.name,
                                 FP8_SM120_MMA.name) and contract.PROFILE == FIRST.name
    with pytest.raises(ValueError, match="profile must be one of"):
        contract.native_reference("another-target")
    with pytest.raises(ValueError, match="profile must be one of"):
        contract.native_seconds_per_vu(1536, "another-target")


def test_fp8_targets_divide_by_their_own_anchor_peak() -> None:
    """FP8 Ada: the RTX 4090's dense FP8 peak *with FP32 accumulate* (330.3 TFLOPS -- Ada halves FP8 under FP32
    accumulate; 660.6 is the FP16-accumulate figure): 9.30 ps per K=1536 VU, so 401 us/VU is 4.31e7x, not the 4.07e7x
    the BF16 constant would give.  FP8 Hopper: the H100's 1978.9 TFLOPS dense FP8 peak, 1.55 ps per VU."""

    ada = contract.native_reference(FP8_ADA_MMA.name)
    assert ada is contract.native_peak_of(FP8_ADA_MMA) and ada.flop_per_second == 330.3e12 and ada.accumulate == "fp32" and "4090" in ada.device
    assert contract.native_seconds_per_vu(1536, FP8_ADA_MMA.name) == pytest.approx(9.30e-12, rel=1e-3)
    assert contract.native_seconds_per_vu(1536, FP8_ADA_MMA.name) == pytest.approx(2 * 1536 / 330.3e12)
    assert contract.overhead(401e-6, 1536, FP8_ADA_MMA.name) == pytest.approx(4.31e7, rel=1e-2)
    assert contract.overhead(401e-6, 1536, FP8_ADA_MMA.name) != pytest.approx(contract.overhead(401e-6, 1536), rel=1e-2)
    hopper = contract.native_reference(FP8_HOPPER_WGMMA.name)
    assert hopper is contract.native_peak_of(FP8_HOPPER_WGMMA) and hopper.flop_per_second == 1978.9e12 and hopper.accumulate == "fp32" and "H100" in hopper.device
    assert contract.native_seconds_per_vu(1536, FP8_HOPPER_WGMMA.name) == pytest.approx(2 * 1536 / 1978.9e12)
    assert contract.native_seconds_per_vu(1536, FP8_HOPPER_WGMMA.name) == pytest.approx(1.552e-12, rel=1e-3)
    for t in (FIRST, FP8_HOPPER_WGMMA, FP8_ADA_MMA):
        assert census.native_seconds_per_instance(contract.native_peak_of(t), 1536) == contract.native_seconds_per_vu(1536, t.name)
        assert "sparsity" not in contract.native_peak_of(t).rate and "dense" in contract.native_peak_of(t).rate and census.source(contract.native_peak_of(t))


def test_every_target_is_a_valid_profile_and_an_unknown_one_is_not(conforming: dict) -> None:
    fp = conforming["workload_fingerprint"]
    for profile in contract.PROFILES:
        assert contract.validate({**conforming, "workload_fingerprint": {**fp, "profile": profile}}) == [], profile
    fp8 = contract.fingerprint(**{**{k: fp[k] for k in contract.REQUIRED_FIELDS}, "profile": FP8_ADA_MMA.name})
    assert fp8["profile"] == FP8_ADA_MMA.name and contract.validate_fingerprint(fp8) == []
    problems = contract.validate({**conforming, "workload_fingerprint": {**fp, "profile": "fp8-ada-mma-draft/2099-01-01"}})
    assert len(problems) == 1 and problems[0].startswith("profile must be a target of verity.ml.circuits.targets.target, one of (") and "fp8-ada-mma-draft/2099-01-01" in problems[0]
    with pytest.raises(ValueError, match="profile must be a target"):
        contract.fingerprint(**{**{k: fp[k] for k in contract.REQUIRED_FIELDS}, "profile": "another-target"})


def test_overhead_must_be_against_the_fingerprints_profile(conforming: dict) -> None:
    """An FP8 Ada result whose overhead was divided by the BF16 A100 constant is off by 330.3/312 = 5.9% and is rejected;
    the same number against the Ada peak conforms."""

    res = json.loads(json.dumps(conforming))
    res["workload_fingerprint"]["profile"] = FP8_ADA_MMA.name
    values = {m["name"]: m["value"] for m in res["measurements"]}
    per_vu, K = values["t.total"] / values["vu.count"], res["workload_fingerprint"]["K"]
    bf16_constant = contract.overhead(per_vu, K, FIRST.name)
    res["measurements"].append(contract.measurement("overhead.vs_native_peak", bf16_constant))
    problems = contract.validate(res)
    assert len(problems) == 1 and problems[0].startswith("overhead.vs_native_peak") and FP8_ADA_MMA.name in problems[0], problems
    res["measurements"][-1]["value"] = contract.overhead(per_vu, K, FP8_ADA_MMA.name)
    assert contract.validate(res) == []
    # and the other way round: a FIRST result cannot carry an overhead divided by the Ada peak
    back = {**res, "workload_fingerprint": {**res["workload_fingerprint"], "profile": FIRST.name}}
    assert len(contract.validate(back)) == 1 and contract.validate(back)[0].startswith("overhead.vs_native_peak")
    # a profile that is not a target is the fingerprint's problem; the overhead is not checked against a guess
    unknown = {**res, "workload_fingerprint": {**res["workload_fingerprint"], "profile": "another-target"}}
    assert [p.split(" ")[0] for p in contract.validate(unknown)] == ["profile"]


def test_rate_measurements_are_three_consistent_numbers_the_validator_recomputes(conforming: dict) -> None:
    """``rate_measurements`` emits the proved rate 2KB / t.total, the profile's native peak and their exact ratio; a
    tampered rate (or one under another profile's peak) is rejected to 1e-6."""

    t_total, B, K = 1.643, 4096, 1536
    ms = contract.rate_measurements(t_total, B, K, FP8_ADA_MMA.name)
    assert [m["name"] for m in ms] == ["rate.proved_flop_per_second", "rate.native_peak_flop_per_second", "overhead.vs_native_peak"]
    assert [m["unit"] for m in ms] == ["FLOP/s", "FLOP/s", "ratio"]
    proved, native, ovh = (m["value"] for m in ms)
    assert proved == 2 * K * B / t_total == contract.proved_flop_per_second(t_total, B, K)
    assert native == 330.3e12 == contract.native_reference(FP8_ADA_MMA.name).flop_per_second
    assert ovh == native / proved
    assert ovh == pytest.approx(contract.overhead(t_total / B, K, FP8_ADA_MMA.name), rel=1e-12)
    assert contract.rate_measurements(t_total, B, K)[1]["value"] == 312e12, "the default profile is FIRST"
    assert contract.unit_of("rate.proved_flop_per_second") == contract.unit_of("rate.native_peak_flop_per_second") == "FLOP/s"
    assert contract.validate_measurements([contract.measurement("rate.made_up", 1.0, "FLOP/s")]) == ["measurement 'rate.made_up' uses a contract prefix but is not a contract name"]
    with pytest.raises(ValueError, match="t_total must be positive"):
        contract.rate_measurements(0.0, B, K)
    with pytest.raises(ValueError, match="profile must be one of"):
        contract.rate_measurements(t_total, B, K, "another-target")

    res = json.loads(json.dumps(conforming))
    res["workload_fingerprint"].update(profile=FP8_ADA_MMA.name, K=1536, B=4096)
    res["measurements"] = [m for m in res["measurements"] if not m["name"].startswith(("t.", "vu."))]
    res["measurements"] += [contract.measurement("t.arithmetic", t_total), contract.measurement("t.total", t_total), contract.measurement("vu.count", B), *ms]
    assert contract.validate(res) == []
    for i, name in enumerate(("rate.proved_flop_per_second", "rate.native_peak_flop_per_second", "overhead.vs_native_peak")):
        tampered = json.loads(json.dumps(res))
        tampered["measurements"][-3 + i]["value"] *= 1.05
        problems = contract.validate(tampered)
        assert len(problems) == 1 and problems[0].startswith(name), (name, problems)
    # the two rates are computed numbers, held to 1e-6; the overhead keeps the contract's 1% (rounded ledger values)
    tight = json.loads(json.dumps(res))
    tight["measurements"][-3]["value"] *= 1 + 1e-5
    tight["measurements"][-1]["value"] *= 1 + 1e-3
    problems = contract.validate(tight)
    assert len(problems) == 1 and problems[0].startswith("rate.proved_flop_per_second"), problems
    # the rates under another profile's peak are rejected; the proved rate is profile-free and stays right
    wrong_peak = json.loads(json.dumps(res))
    wrong_peak["measurements"][-3:] = contract.rate_measurements(t_total, B, K, FIRST.name)
    problems = contract.validate(wrong_peak)
    assert [p.split(" ")[0] for p in problems] == ["rate.native_peak_flop_per_second", "overhead.vs_native_peak"], problems
    # a rate without the numbers it is checked against is a problem, not a pass
    bare = {**res, "measurements": [m for m in res["measurements"] if m["name"] != "vu.count"]}
    assert any("needs K in the fingerprint and positive t.total and vu.count" in p for p in contract.validate(bare))


def test_results_of_different_profiles_are_never_compared(conforming: dict) -> None:
    fp = conforming["workload_fingerprint"]
    first, ada = {**fp, "profile": FIRST.name}, {**fp, "profile": FP8_ADA_MMA.name}
    assert contract.validate_fingerprint(first) == [] and contract.validate_fingerprint(ada) == []
    assert contract.incomparable(first, ada) == [("profile", FIRST.name, FP8_ADA_MMA.name)]
    assert contract.identity(first)["profile"] != contract.identity(ada)["profile"]
    assert contract.incomparable(first, ada, allow=["profile"]) == [], "only an experiment that names the field may cross it"


def test_batch_hardware_and_soundness_target_are_identity(conforming: dict) -> None:
    """B=64 is never compared with B=4096, a 4090 run never with a CPU run, 2^-80 never with 2^-128 -- unless named."""

    fp = conforming["workload_fingerprint"]
    assert contract.incomparable(fp, {**fp, "B": fp["B"] * 2}) == [("B", fp["B"], fp["B"] * 2)]
    other_box = {**fp, "hardware": {**fp["hardware"], "cpu": {"model": "somewhere else"}}}
    assert [f for f, _, _ in contract.incomparable(fp, other_box)] == ["hardware"]
    assert contract.incomparable(fp, other_box, allow=["hardware"]) == []
    weaker = {**fp, "security": {**fp["security"], "target": -80}}
    assert [f for f, _, _ in contract.incomparable(fp, weaker)] == ["security.target"]
    no_target = {**fp, "security": {k: v for k, v in fp["security"].items() if k != "target"}}
    assert contract.incomparable(fp, no_target)[0][0] == "security.target"


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
        "profile": FIRST.name, "domain": "finite-only", "K": 16, "B": 1, "proof_class": "NON_ZK_PROOF", "authentication": "included",
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
                          capture_output=True, text=True, env={**os.environ, "RESEARCH_RUN_ID": "r-local-test"})
    assert proc.returncode == 0, proc.stderr
    result = json.loads((out / "result.json").read_text())
    assert result["run_id"] == "r-local-test"
    assert contract.validate(result) == [], contract.validate(result)
    fp = result["workload_fingerprint"]
    assert (fp["profile"], fp["domain"], fp["K"], fp["B"], fp["proof_class"], fp["authentication"], fp["zk_mode"], fp["retention"]) == (
        FIRST.name, "finite-only", 16, 1, "NO_PROOF", "included", "none", "retained")
    assert fp["security"]["target"] is None and set(fp["hardware"]) == {"gpu", "cpu"}
    assert fp["software"]["backend"]["name"] == "verity_sp1.proofs.transparent"
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


# --- instances: which bench-instances/v1 ids were proved ------------------------------------------------------------------------------

SHA = "f2b7cd4ca31e4adf0ce1e656bb0a2ef882f02301d5058670350b9c2b8d1f81a6"


def test_instances_is_required_but_nullable(conforming: dict) -> None:
    fp = conforming["workload_fingerprint"]
    assert "instances" in contract.REQUIRED_FIELDS and fp["instances"] is None
    assert contract.validate_fingerprint(fp) == []
    assert contract.validate_fingerprint({k: v for k, v in fp.items() if k != "instances"}) == ["workload_fingerprint.instances missing"]
    built = contract.fingerprint(**{k: fp[k] for k in contract.REQUIRED_FIELDS if k != "instances"})
    assert built["instances"] is None, "always written: null = not measured on the shared input set"


def test_instances_ref_and_its_validation(conforming: dict) -> None:
    fp = {**conforming["workload_fingerprint"], "B": 64}
    ref = contract.instances_ref("vu-k1536", 0, 64, SHA)
    assert ref == {"dataset": "bench-instances/v1", "tier": "vu-k1536", "range": [0, 64], "manifest_sha256": SHA}
    assert contract.validate_fingerprint({**fp, "instances": ref}) == []
    assert contract.validate_fingerprint({**fp, "B": 4096, "B_real": 64, "instances": ref}) == []
    assert contract.validate_fingerprint({**fp, "B": 65, "instances": ref}) == ["instances.range [0, 64) holds 64 ids but B is 65"]
    assert contract.validate_fingerprint({**fp, "instances": {**ref, "tier": "vu-k32"}}) == [
        f"instances.tier must be one of {contract.INSTANCES_TIERS} for bench-instances/v1, got 'vu-k32'"
    ]
    assert contract.validate_fingerprint({**fp, "instances": {**ref, "range": [64, 64]}}) == ["instances.range must be [lo, hi) with 0 <= lo < hi, got [64, 64]"]
    assert contract.validate_fingerprint({**fp, "instances": {**ref, "manifest_sha256": "abc"}}) == [
        "instances.manifest_sha256 must be a 64-hex-digit sha256 of the dataset's manifest.json, got 'abc'"
    ]
    assert contract.validate_fingerprint({**fp, "instances": {"dataset": "x"}}) == ["instances.tier missing", "instances.range missing", "instances.manifest_sha256 missing"]
    assert contract.validate_fingerprint({**fp, "instances": "vu-k1536"}) == ["instances must be an object {dataset, tier, range, manifest_sha256} or null, got 'vu-k1536'"]
    with pytest.raises(ValueError, match="instances does not conform"):
        contract.instances_ref("vu-k1536", 5, 5, SHA)


def test_compare_instances_conflicts_and_warnings(conforming: dict) -> None:
    fp = conforming["workload_fingerprint"]
    same = contract.instances_ref("vu-k1536", 0, 64, SHA)
    a, b = {**fp, "B": 64, "instances": same}, {**fp, "B": 64, "instances": dict(same)}
    assert contract.compare_instances(a, b) == (None, [])
    # a different manifest, tier or dataset: conflict
    other = {**same, "manifest_sha256": "0" * 64}
    conflict, warnings = contract.compare_instances(a, {**b, "instances": other})
    assert conflict == (same, other) and warnings == []
    assert contract.compare_instances(a, {**b, "instances": {**same, "tier": "tu-k16"}})[0] is not None
    # different ids for the same batch size: conflict; different ids because B differs: a scaling point, no conflict
    shifted = contract.instances_ref("vu-k1536", 64, 128, SHA)
    assert contract.compare_instances(a, {**b, "instances": shifted})[0] == (same, shifted)
    bigger = contract.instances_ref("vu-k1536", 0, 4096, SHA)
    assert contract.compare_instances(a, {**b, "B": 4096, "instances": bigger}) == (None, [])
    # null or absent on either side: no conflict, a warning naming the side
    conflict, warnings = contract.compare_instances(a, {**b, "instances": None})
    assert conflict is None and warnings == ["candidate instances is null: not measured on the shared input set"]
    conflict, warnings = contract.compare_instances({k: v for k, v in a.items() if k != "instances"}, {**b, "instances": None})
    assert conflict is None and len(warnings) == 2 and warnings[0].startswith("baseline does not record instances")



def test_no_writer_hardcodes_the_domain_and_new_fingerprints_use_the_vocab() -> None:
    """red-team-flock-3's M0 (coordinator 2026-09-27 13:45Z): every cell recorded ``domain: finite`` whatever it proved.  A
    fingerprint's domain is now its statement's, one of the vocab's; the old hardcode is refused on new writes, still read on
    old results (as ``finite-only``, so identity comparisons with them hold)."""
    import re
    from pathlib import Path

    from research.store import vocab

    assert set(contract.DOMAINS) == set(vocab.DOMAINS)
    root = Path(contract.__file__).resolve().parents[5]
    writers = [*root.glob("verity/ml/flock/python/verity_flock/*.py"), *root.glob("benchmarks/numerical/python/verity_numerical/bench/*.py")]
    assert writers and not [str(p) for p in writers if re.search(r'domain\s*=\s*"finite"', p.read_text())]
    kw = dict(profile=None, K=None, B=4, proof_class="NON_ZK_PROOF", authentication="included-hash", zk_mode="none",
              security={"target": -128}, hardware={"gpu": None, "cpu": {"model": "x"}},
              software={"backend": {"name": "b", "commit": None}, "toolchain": {}}, preprocessing="none", retention="retained",
              instances=None, subcircuit="silu-mul/i8192")
    assert contract.fingerprint(domain="total", **kw)["domain"] == "total"
    with pytest.raises(ValueError, match="pre-2026-09-27 hardcode"):
        contract.fingerprint(domain="finite", **kw)
    old = {**contract.fingerprint(domain="finite-only", **kw), "domain": "finite"}
    assert contract.validate_fingerprint(old) == [] and contract.identity(old)["domain"] == "finite-only"
    assert contract.incomparable(old, {**old, "domain": "finite-only"}) == []
    assert [f for f, _, _ in contract.incomparable(old, {**old, "domain": "total"})] == ["domain"]
