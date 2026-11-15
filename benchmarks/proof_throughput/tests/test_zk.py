import json

import pytest

from verity_proof_throughput import cli, zk
from verity_proof_throughput.workloads import workload

W = workload("bf16-k6144")


def test_configuration_two_reports_its_fields_and_their_sources():
    text = zk.describe()
    names = [f[0] for f in zk.FIELDS]
    assert names == ["worker.vu_per_second", "zk_server.cpu_s_per_session", "zk_server.cpu_s_per_vu", "net.bytes_worker_to_server",
                     "proof_bytes.released", "zk_server.cores_per_gpu"]
    assert all(name in text for name in names)
    assert {f[2] for f in zk.FIELDS} <= {"config1", "stand-in", "stub"}


def zk_records(d, zk_on=True, zk_server=True):
    fw = {"t.zk_server_shadow": 1.5, "t.zk_server_wait": 0.25, "t.zk_server_recheck": 4.0, "cpu.zk_server_recheck": 40.0} if zk_server else {}
    reps = [{"t.total": 1.0} | fw, {"t.total": 1.0} | ({"t.zk_server_shadow": 1.5} if zk_server else {})]
    live = [{"warm": i == 0, "accepted": True, "buckets": reps, "identity": {"hooks": {"zk": zk_on}}} for i in range(4)]
    (d / "prove.out").write_text("".join("LIVE\t" + json.dumps(v) + "\n" for v in live))
    (d / "results.jsonl").write_text(json.dumps({"live": {"accepted": True, "zk": zk_on, "prove_total_s": 2.0, "prover_cpu_s": 400.0,
                                                          "wire_sent": 3_000_000, "wire_recv": 40_000, "proof_bytes": [1_000, 1_200],
                                                          "prover_peak_gb": 2.0}}) + "\n")


def test_the_zk_point_reports_the_zk_servers_times_and_the_prover_cpu(tmp_path, monkeypatch, capsys):
    zk_records(tmp_path)
    monkeypatch.setenv("FC_ZK_SERVER", "1")
    out = tmp_path / "result.json"
    assert cli.main(["zk-point", "--workload", W.name, "--B", "16", "--dir", str(tmp_path), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["schema"] == "research/result/v0.1" and doc["validation"]["status"] == "passed"
    assert doc["workload_fingerprint"]["firewall"] is True and doc["workload_fingerprint"]["kind"] == "proof-throughput/zk-standin/v1"
    v = {m["name"]: m["value"] for m in doc["measurements"]}
    assert v["prover.cpu_s_per_session"] == 100.0 and v["vu.per_second"] == 8.0 and v["proof_bytes"] == 2_200
    assert v["zk_server.t_shadow_s"] == 3.0 and v["zk_server.cpu_recheck_s"] == 40.0 and v["net.bytes_out"] == 3_000_000


def test_the_zk_point_fails_unless_its_sessions_were_masked_and_zk_served(tmp_path, monkeypatch):
    monkeypatch.setenv("FC_ZK_SERVER", "1")
    zk_records(tmp_path, zk_on=False)
    assert "not --zk" in zk.point(tmp_path, W, 16, True, None)["validation"]["detail"]
    zk_records(tmp_path, zk_server=False)
    assert "no ZK server times" in zk.point(tmp_path, W, 16, True, None)["validation"]["detail"]
    assert zk.point(tmp_path, W, 16, False, None)["validation"]["status"] == "passed"


def test_a_job_that_sets_the_old_switch_and_not_the_new_one_is_refused_by_the_new_name():
    with pytest.raises(SystemExit, match="FC_FIREWALL is set and FC_ZK_SERVER is not.*set FC_ZK_SERVER instead"):
        zk.zk_server_on({"FC_FIREWALL": "1"})
    assert zk.zk_server_on({"FC_FIREWALL": "1", "FC_ZK_SERVER": "1"}) and not zk.zk_server_on({"FC_ZK_SERVER": "0"})


def test_a_session_recorded_before_the_rename_reports_the_same_zk_server_times():
    new = {"buckets": [{"t.zk_server_recheck": 4.0, "cpu.zk_server_recheck": 40.0, "t.total": 1.0}]}
    old = {"buckets": [{"t.firewall_recheck": 4.0, "cpu.firewall_recheck": 40.0, "t.total": 1.0}]}
    assert zk._zk_server(old) == zk._zk_server(new) == {"t.zk_server_recheck": 4.0, "cpu.zk_server_recheck": 40.0}
