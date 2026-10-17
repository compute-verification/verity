from verity_proof_throughput import zk


def test_configuration_two_reports_its_fields_and_their_sources():
    text = zk.describe()
    names = [f[0] for f in zk.FIELDS]
    assert names == ["worker.vu_per_second", "zk_server.cpu_s_per_session", "zk_server.cpu_s_per_vu", "net.bytes_worker_to_server",
                     "proof_bytes.released", "zk_servers_per_gpu"]
    assert all(name in text for name in names)
    assert {f[2] for f in zk.FIELDS} <= {"config1", "stand-in", "stub"}
