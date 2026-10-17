import pytest
from verity_numerical.bench import contract

from verity_proof_throughput import result as RS
from verity_proof_throughput.workloads import workload

W = workload("bf16-k6144")
SW = {"backend": {"name": "flock-circuit", "commit": "0" * 40}, "toolchain": {"python": "3.12"}, "verifier": {}}
LEAN = [{"session": f"l000{i}-1", "accepted": True, "why": None, "wall_s": 2.0 + i, "cpu_s": 1.5 + i, "max_rss_kb": 500_000}
        for i in range(3)]


def point(record, hardware, lean=None, B=16):
    return RS.point(W, B, record, run_id="r20261017-000000-abcd", hardware=hardware, software=SW, gpu_peak_mib=20_000.0,
                    lean=lean, evidence=["out/results.jsonl"])


def test_a_point_conforms_to_the_contract_and_its_rates_are_the_contracts(record, hardware):
    doc = point(record, hardware, LEAN)
    assert contract.validate(doc) == [] and doc["validation"]["status"] == "passed"
    total = 0.6 + 6.0
    assert RS.value(doc, "t.total") == pytest.approx(total)
    assert RS.value(doc, "vu.per_second") == pytest.approx(16 / total)
    want = {m["name"]: m["value"] for m in contract.rate_measurements(total, 16, W.K, W.profile)}
    assert RS.value(doc, "overhead.vs_native_peak") == pytest.approx(want["overhead.vs_native_peak"])
    assert RS.value(doc, "t.arithmetic") == pytest.approx(3.0) and RS.value(doc, "prove.zerocheck_s") == pytest.approx(1.4)
    assert RS.value(doc, "t.serialization") == pytest.approx(0.02) and RS.value(doc, "prove.host_stage_s") == pytest.approx(0.1)
    assert RS.value(doc, "proof_bytes") == 1_800_000 and RS.value(doc, "transcript.bytes") == 1_854_000
    assert RS.value(doc, "mem.peak_host_bytes") == int(12.5 * 2**30) and RS.value(doc, "mem.peak_device_bytes") == 20_000 << 20
    assert RS.value(doc, "verifier.seconds") == 3.0 and RS.value(doc, "verify.peak_rss_bytes") == 500_000 * 1024
    assert RS.value(doc, "fixed.setup_seconds") == 41.0 and RS.value(doc, "t.total_live") == 8.5


def test_the_fingerprint_carries_the_identity_the_comparisons_need(record, hardware):
    fp = point(record, hardware, LEAN)["workload_fingerprint"]
    assert (fp["K"], fp["B"], fp["profile"], fp["proof_class"], fp["zk_mode"]) == (6144, 16, W.profile, "NON_ZK_PROOF", "none")
    assert fp["security"]["target"] == -100 and fp["hardware"] == hardware
    assert fp["instances"]["range"] == [0, 16] and fp["instances"]["tier"] == W.name
    assert fp["workload"] == W.name and fp["vu"] == W.vu and fp["config"] == "plain"


def test_the_prover_run_passes_on_its_live_verdicts_alone(record, hardware):
    doc = point(record, hardware)
    assert doc["validation"]["status"] == "passed" and "verifier.seconds" not in {m["name"] for m in doc["measurements"]}
    assert "Lean verifier not run" in doc["validation"]["checker"]


def test_a_rejection_anywhere_fails_the_point(record, hardware):
    lean = [*LEAN[:2], {**LEAN[2], "accepted": False, "why": "zerocheck: final claim mismatch"}]
    doc = point(record, hardware, lean)
    assert doc["validation"]["status"] == "failed" and "final claim mismatch" in doc["validation"]["detail"]
    record["live"] |= {"accepted": False, "why": "sessions [2] not accepted"}
    doc = point(record, hardware, LEAN)
    assert doc["validation"]["status"] == "failed" and "sessions [2]" in doc["validation"]["detail"]
    assert contract.validate(doc) == [] and RS.value(doc, "t.total") == pytest.approx(6.6)


def test_the_lean_leg_passes_on_the_drawless_refusal_alone_and_says_so(record, hardware):
    lean = [{**v, "accepted": False, "why": RS.DRAWLESS} for v in LEAN]
    doc = point(record, hardware, lean)
    assert doc["validation"]["status"] == "passed" and contract.validate(doc) == []
    assert "0 of 3 sessions accepted, 3 passed every check and were refused only for having no draw" in doc["validation"]["detail"]
    assert (RS.value(doc, "verify.lean_accepted"), RS.value(doc, "verify.lean_drawless")) == (0, 3)
    lean[1] = {**lean[1], "why": "refused: a subset draw without --zk: no soundness claim draws by this law"}
    doc = point(record, hardware, lean)
    assert doc["validation"]["status"] == "failed" and "subset draw" in doc["validation"]["detail"]


def test_a_run_without_a_session_record_fails_with_its_reason(hardware):
    doc = point({"broke": "staging: out of memory"}, hardware)
    assert doc["validation"]["status"] == "failed"
    assert doc["validation"]["detail"].startswith("live verifier: staging: out of memory; contract:")


def test_statement_seconds_takes_the_pipelined_witness_when_it_is_slower():
    assert RS.statement_seconds({"witness_s": 1.0, "prove_total_s": 5.0}) == 6.0
    assert RS.statement_seconds({"witness_s": 0.0, "prove_total_s": 5.0, "witness_prebuilt_s": 40.0, "pipeline_depth": 4}) == 10.0


def test_the_dominant_bucket_splits_arithmetic_into_the_provers_phases(record, hardware):
    name, share = RS.dominant_bucket(point(record, hardware, LEAN))
    assert name == "t.encoding_commitment" and share == pytest.approx(1.8 / 6.6)
    record["live"]["buckets"] = [{**RS_REP, "t.ligerito": 2.0} for RS_REP in record["live"]["buckets"]]
    assert RS.dominant_bucket(point(record, hardware, LEAN))[0] == "prove.ligerito_s"
