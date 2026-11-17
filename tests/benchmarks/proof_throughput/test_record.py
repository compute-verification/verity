import json
from pathlib import Path

from verity_bench import kit

from verity_proof_throughput import record as REC


def verifier_result(status="passed", rate=10.0):
    return {"validation": {"status": status}, "measurements": [{"name": "vu.per_second", "value": rate, "unit": "VU/s"}]}


def point(d, status="passed", rate=10.0, span=(0.0, 5.0)):
    return {"dir": d, "result": verifier_result(status, rate), "span": span}


def test_a_member_names_the_model_the_card_and_the_prover_count():
    assert REC.member("bf16-k6144", "NVIDIA H200", 1) == "glm-5.3-h200-1-prover"
    assert REC.member("bf16-k6144", "NVIDIA H200", 2) == "glm-5.3-h200-2-provers"
    assert REC.member("bf16-k6144", "none", 1) == "glm-5.3-cpu-1-prover"


def test_provers_at_once_sum_their_verified_rates():
    assert REC.throughput([point("a", rate=10.0), point("b", rate=7.5, span=(1.0, 6.0))])[0] == 17.5
    assert REC.throughput([point("a", rate=10.0)])[0] == 10.0


def test_no_value_without_every_verifier_passing_or_without_overlapping_sessions():
    value, why = REC.throughput([point("a"), point("b", status="failed")])
    assert value is None and "b" in why
    value, why = REC.throughput([point("a", span=(0.0, 1.0)), point("b", span=(2.0, 3.0))])
    assert value is None and "at once" in why
    value, why = REC.throughput([point("a", span=None)])
    assert value is None and "lease" in why


class Done:
    def __init__(self, rc=0):
        self.rc = rc

    def wait(self):
        return self.rc


def stand_in(seen, rates):
    def start(root, script, env, log):
        seen.append((script, env))
        d = Path(env["OUT"])
        d.mkdir(parents=True, exist_ok=True)
        if script == "job.sh":
            (d / "bundle.json").write_text(json.dumps({"id": f"x{len(seen)}"}))
            (d / "phases.tsv").write_text("lease\tstart\t1.0\nlease\tend\t9.0\n")
        else:
            Path(env["RESULT"]).write_text(json.dumps(verifier_result(rate=rates.pop(0))))
        return Done()
    return start


def test_record_runs_one_prover_then_two_at_once_each_verified_and_writes_a_reading_a_prover_count(tmp_path):
    seen = []
    out = tmp_path / "out"
    points = REC.run(tmp_path, out, "bf16-k6144", 64, [2, 1], ["32-47", "80-95"], "node-a", "r1", start=stand_in(seen, [4.0, 3.0, 5.0]))
    assert [s for s, _ in seen] == ["job.sh", "verify.sh", "job.sh", "job.sh", "verify.sh", "verify.sh"]
    jobs = [e for s, e in seen if s == "job.sh"]
    assert [e["LEASE_CORES"] for e in jobs] == ["32-47", "32-47", "80-95"]
    assert len({e["SESSIONS_LOCK"] for e in jobs[1:]}) == 2 and all(e["B"] == "64" for e in jobs)
    assert len({e["PORT"] for e in jobs[1:]}) == 2
    assert all(e["BUNDLE"].startswith("art:") and e["PROVE_RUN"] == "r1" for s, e in seen if s == "verify.sh")
    assert REC.write(kit.writer(tmp_path), out, "bf16-k6144", 64, points) == 0
    doc = json.loads((tmp_path / "result.json").read_text())
    got = {m["member"].rsplit("-", 2)[-2]: m["value"] for m in doc["measurements"] if m["name"] == REC.METRIC}
    assert got == {"1": 4.0, "2": 8.0} and doc["validation"]["status"] == "passed"
    assert all(f"out/{p['dir']}/verifier/result.json" in doc["validation"]["evidence"] for g in points.values() for p in g)


def test_a_prover_that_fails_leaves_its_count_without_a_value_and_the_run_failed(tmp_path):
    def start(root, script, env, log):
        Path(env["OUT"]).mkdir(parents=True, exist_ok=True)
        return Done(1)
    out = tmp_path / "out"
    points = REC.run(tmp_path, out, "bf16-k6144", 64, [1], ["32-47"], "node-a", "r1", start=start)
    assert REC.write(kit.writer(tmp_path), out, "bf16-k6144", 64, points) == 1
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["measurements"][0]["value"] is None and doc["validation"]["status"] == "failed"
