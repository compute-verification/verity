"""cell.finish: the plateau point of a live-verifier sweep plus the same run's probes, as the B-interactive cell's result."""
from __future__ import annotations

import json
import math
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


def _registrable(tmp: Path, name: str, relation: str, per_proof: float, n_proofs: int, terms: dict | None = None) -> Path:
    run = tmp / name
    run.mkdir()
    snd = {"per_proof_total_log2": per_proof, "n_proofs": n_proofs, "total_log2": per_proof + math.log2(n_proofs),
           "field": "BabyBear^6", "fiat_shamir_queries_log2": None, "terms_log2": terms or {"irs_query": per_proof}}
    doc = {"workload_fingerprint": {"statement_relation": relation, "relation": relation.partition("+")[0],
                                    "leaf": {"scheme": relation.partition("+")[2]},
                                    "security": {"achieved_log2": snd["total_log2"], "achieved_bits": -snd["total_log2"]}},
           "measurements": [], "validation": {"evidence": {"soundness": snd}}}
    (run / "result.json").write_text(json.dumps(doc))
    return run


def test_register_books_the_chain_term_and_refuses_above_2_128(tmp_path) -> None:
    """A result made before the chain test's field term was booked gets it at register (red-team-bligero-real-k A1); a
    bound above 2^-128 with it is refused: art:b1d710da's figures (2^-128.104 over 60 sub-batches -> 2^-127.97), not
    art:11208bf7's (2^-128.350 over 32 -> 2^-128.26)."""
    import pytest
    tight = _registrable(tmp_path, "b1d710da", "bf16-ampere-x4-k8192+blake3-xob", -134.01044941441342, 60)
    with pytest.raises(CL.Refused, match=r"2\^-127\.97\d over 60 sub-batches .* above 2\^-128"):
        CL.register(tight, None, "lane-x")
    assert CL.main(["register", "--prover-run", str(tight), "--lane", "lane-x", "--out", str(tmp_path / "m.json")]) == 3
    assert not (tmp_path / "m.json").exists()
    ok = CL.register(_registrable(tmp_path, "11208bf7", "bf16-hopper-x4-k8192+blake3-xob", -133.3502994173009, 32), None, "lane-x")
    snd = ok["validation"]["evidence"]["soundness"]
    assert snd["chain_extras"] == 1092 and snd["chain_degree"] == 183 and abs(snd["terms_log2"]["chain_field"] + 137.396) < 0.001
    assert -128.27 < snd["total_log2"] < -128.26 and snd["chain_field_booked"].startswith("at register")
    assert ok["workload_fingerprint"]["security"]["achieved_log2"] == snd["total_log2"]
    # a result made with the term booked is taken as it is; a bare statement has no extras
    booked = _registrable(tmp_path, "new", "bf16-ampere-x4-k8192+blake3-xob", -134.2, 60, {"irs_query": -134.3, "chain_field": -137.4})
    assert "chain_field_booked" not in CL.register(booked, None, "lane-x")["validation"]["evidence"]["soundness"]
    bare = _registrable(tmp_path, "bare", "bf16-ampere-x4-k8192", -134.01, 60)
    json_doc = json.loads((bare / "result.json").read_text())
    del json_doc["workload_fingerprint"]["leaf"]
    (bare / "result.json").write_text(json.dumps(json_doc))
    assert "chain_field" not in CL.register(bare, None, "lane-x")["validation"]["evidence"]["soundness"]["terms_log2"]
