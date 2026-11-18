import json
from pathlib import Path

import pytest
from verity_bench import kit

from verity_proof_throughput import cli
from verity_proof_throughput import record as REC


def verifier_result(status="passed", rate=10.0):
    return {"validation": {"status": status}, "measurements": [{"name": n, "value": rate + d, "unit": "VU/s"}
                                                               for n, d in zip(REC.SESSIONS.values(), (0, -1, 1))]}


def point(d, status="passed", rate=10.0, span=(0.0, 5.0)):
    return {"dir": d, "result": verifier_result(status, rate), "span": span}


def test_a_member_names_the_model_the_card_and_the_prover_count():
    assert REC.member("bf16-k6144", "NVIDIA H200", 1) == "glm-5.3-h200-1-prover"
    assert REC.member("bf16-k6144", "NVIDIA H200", 2) == "glm-5.3-h200-2-provers"
    assert REC.member("bf16-k6144", "none", 1) == "glm-5.3-cpu-1-prover"
    assert REC.member("bf16-k2048", "NVIDIA H200", 1) == "llama-3.2-1b-h200-1-prover"
    assert REC.member("bf16-k2048", "NVIDIA H100 80GB HBM3", 2) == "llama-3.2-1b-h100-2-provers"


def test_a_card_no_record_holds_gets_no_reading_and_fails_the_run(tmp_path, capsys):
    w = kit.writer(tmp_path)
    w.stamp["card"] = "NVIDIA Nonesuch"
    assert REC.write(w, tmp_path / "out", "bf16-k2048", 64, {1: [point("a")]}) == 1
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["measurements"] == [] and "NVIDIA Nonesuch" in doc["validation"]["detail"]
    assert "NVIDIA Nonesuch" in capsys.readouterr().out


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
    points = REC.run(tmp_path, out, "bf16-k6144", 64, [2, 1], ["0", "1"], ["0-9", "10-19"], "node-a", "r1", start=stand_in(seen, [4.0, 3.0, 5.0]))
    assert [s for s, _ in seen] == ["job.sh", "verify.sh", "job.sh", "job.sh", "verify.sh", "verify.sh"]
    jobs = [e for s, e in seen if s == "job.sh"]
    assert [(e["LEASE_CARD"], e["LEASE_CORES"]) for e in jobs] == [("0", "0-9"), ("0", "0-9"), ("1", "10-19")]
    assert len({e["SESSIONS_LOCK"] for e in jobs[1:]}) == 2 and all(e["B"] == "64" for e in jobs)
    assert len({e["PORT"] for e in jobs[1:]}) == 2 and all(e["RUNS"] == "3" for e in jobs)
    assert all(e["BUNDLE"].startswith("art:") and e["PROVE_RUN"] == "r1" for s, e in seen if s == "verify.sh")
    assert REC.write(kit.writer(tmp_path), out, "bf16-k6144", 64, points) == 0
    doc = json.loads((tmp_path / "result.json").read_text())
    got = {m["member"].rsplit("-", 2)[-2]: m["value"] for m in doc["measurements"] if m["name"] == REC.METRIC}
    assert got == {"1": 4.0, "2": 8.0} and doc["validation"]["status"] == "passed"
    assert all(m["model"] == "glm-5.3" and m["values"].startswith("stand-in") for m in doc["measurements"])
    leases = {m["provers"]: m["leases"] for m in doc["measurements"]}
    first, second = {"card": "0", "cores": "0-9"}, {"card": "1", "cores": "10-19"}
    assert leases == {1: [first], 2: [first, second]}
    assert {m["provers"]: (m["lowest"], m["highest"]) for m in doc["measurements"]} == {1: (3.0, 5.0), 2: (6.0, 10.0)}
    assert [m["sessions"] for m in doc["measurements"] if m["provers"] == 1] == [[{"median": 4.0, "lowest": 3.0, "highest": 5.0}]]
    assert all(f"out/{p['dir']}/verifier/result.json" in doc["validation"]["evidence"] for g in points.values() for p in g)


@pytest.mark.parametrize("cards, cores, flag", [("0", "0-9,10-19", "--cards"), ("0,0", "0-9,10-19", "--cards"),
                                                ("0,1", "0-9,0-9", "--cores")])
def test_record_refuses_provers_without_a_card_and_cores_of_their_own(cards, cores, flag):
    with pytest.raises(SystemExit, match=flag):
        cli.main(["record", "--provers", "1,2", "--cards", cards, "--cores", cores])


def test_a_prover_that_fails_leaves_its_count_without_a_value_and_the_run_failed(tmp_path):
    def start(root, script, env, log):
        Path(env["OUT"]).mkdir(parents=True, exist_ok=True)
        return Done(1)
    out = tmp_path / "out"
    points = REC.run(tmp_path, out, "bf16-k6144", 64, [1], ["0"], ["0-9"], "node-a", "r1", start=start)
    assert REC.write(kit.writer(tmp_path), out, "bf16-k6144", 64, points) == 1
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["measurements"][0]["value"] is None and doc["validation"]["status"] == "failed"
