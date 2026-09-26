"""cell.finish: the plateau point of a live-verifier sweep plus the same run's probes, as the B-interactive cell's result."""
from __future__ import annotations

import json
from pathlib import Path

from . import cell as CL


def _point(tmp: Path, name: str, n: int, wait: float, sessions: int = 5, round_trips: int = 30) -> Path:
    d = tmp / name
    (d / "proofs" / "rep1").mkdir(parents=True)
    doc = {"workload_fingerprint": {"N_subbatches": 7, "software": {"backend": {"pipeline": 4}}, "B": n},
           "measurements": [{"name": "t.total", "value": 2.0}, {"name": "net.rtt_ms", "value": 3.5}, {"name": "net.wait_seconds", "value": wait},
                            {"name": "net.bytes_out", "value": 1e8}, {"name": "vu.count", "value": n}],
           "validation": {"evidence": {"live_verifier": {"sessions": sessions, "round_trips": round_trips,
                                                         "per_rep": [{"rep": r + 1, "drain_s": 0.25, "bytes_out": 2e7, "bytes_in": 900}
                                                                     for r in range(sessions)]},
                                       "phase_rep": {"index": 2}}},
           "artifacts": [f"sweep/{name}/proofs"]}
    (d / "result.json").write_text(json.dumps(doc))
    return d


def test_finish_assembles_the_interaction_record(tmp_path) -> None:
    from verity_numerical.bench import views
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    _point(sweep, "p0-1024", 1024, 0.05)
    _point(sweep, "p1-2048", 2048, 0.04)
    (sweep / "sweep.json").write_text(json.dumps({"id": "s", "stopped": "x", "plateau_point": 1, "relation": "r",
                                                  "points": [{"point": 0, "total_vus": 1024, "vu_per_second": 1.0, "dir": "p0-1024"},
                                                             {"point": 1, "total_vus": 2048, "vu_per_second": 2.0, "dir": "p1-2048"}]}))
    probe = {"ping_ms_median": 0.9, "ping_ms_min": 0.8, "mbps_median": 9000.0, "ping_method": "ping over the open connection: 50 x",
             "flows": [{"bytes": 64 << 20}]}
    loop = {"measurements": [{"name": "net.wait_seconds", "value": 0.012}, {"name": "vu.count", "value": 2048}],
            "validation": {"evidence": {"live_verifier": {"sessions": 2, "round_trips": 28}}}}
    doc = CL.finish(sweep, probe, loop, "10.0.0.3:7000", tmp_path, operator="lane-x")
    m = CL._meas(doc)
    assert m["net.rtt_ms"] == 0.9 and m["live.opening_rtt_ms_median"] == 3.5 and m["net.bandwidth_bps"] == 9e9
    assert m["rounds.sequential_depth"] == 3 * 2 and m["live.per_proof_depth"] == 3            # ceil(7 proofs / 4 in flight) = 2 waves
    assert m["net.bytes_out"] == 2e7 and m["net.bytes_in"] == 900 and m["live.transfer_tail_seconds"] == 0.25
    assert m["net.pass_wait_seconds"] == 0.04 and abs(m["net.wait_seconds"] - 0.29) < 1e-12 and abs(m["t.total_live"] - 2.29) < 1e-12
    assert abs(m["live.loopback_round_seconds"] - 0.012 / 14) < 1e-12
    assert doc["workload_fingerprint"]["verifier"]["placement"] == "10.0.0.3:7000" and doc["artifacts"] == ["sweep/p1-2048/proofs"]
    assert views.rtt_method_enum(doc["interaction"]["rtt_method"]) == "open-connection-ping" and doc["interaction"]["loopback_method"]
    ix = views.interaction(doc)
    assert ix["live"] and ix["rounds"] == 6 and ix["rtt_ms"] == 0.9 and ix["bandwidth_bps"] == 9e9 and ix["loopback_method"]
