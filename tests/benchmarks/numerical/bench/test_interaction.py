"""``bench.interaction``: a result's interaction record (measured for a live verifier, derived for a file-verified proof), the
tolerance check against the run's own network, and the measurement protocol ``cell check`` and Flock's ``register`` apply."""
from __future__ import annotations

import pytest

from verity_numerical.bench import interaction as I

LIVE = {"rounds.sequential_depth": 3, "net.bytes_out": 1.25e9, "net.bytes_in": 384.0, "net.rtt_ms": 1.0,
        "rtt_method": "ping median of 20, same pod pair", "net.wait_seconds": 0.103}
BLOCK_KEYS = ("rtt_method", "t_total_includes_wait", "loopback_method")


def result(t_total: float = 0.4, fp: dict | None = None, **ix) -> dict:
    """A result document: ``t.total`` plus ``ix`` as measurements, its interaction-block keys in ``interaction``."""
    doc = {"workload_fingerprint": fp or {}, "measurements": [{"name": "t.total", "value": t_total, "unit": "s"}]}
    for name, value in ix.items():
        if name in BLOCK_KEYS:
            doc.setdefault("interaction", {})[name] = value
        else:
            doc["measurements"].append({"name": name, "value": value, "unit": "-"})
    return doc


def test_a_live_record_is_measured_and_its_time_is_the_formula() -> None:
    ix = I.interaction(result(**LIVE))
    assert ix["live"] and not ix["derived"] and ix["rounds"] == 3 and ix["rounds_source"] == "measured"
    assert (ix["bytes_up"], ix["bytes_down"], ix["rtt_ms"], ix["wait_s"]) == (1.25e9, 384.0, 1.0, 0.103)
    assert I.interaction_time(ix) == pytest.approx(0.4 + 3 * 0.001 + (1.25e9 + 384) * 8 / 100e9)
    assert I.REF_BANDWIDTH_BPS == 100e9 and I.interaction_problem(ix) is None


def test_a_record_whose_t_total_includes_the_wait_counts_its_compute_once() -> None:
    ix = I.interaction(result(t_total=0.4 + 0.103, **LIVE, t_total_includes_wait=True))
    assert ix["prover_s"] == pytest.approx(0.4) and ix["wait_s"] == pytest.approx(0.103)


def test_live_coin_handling_measured_on_loopback_is_prover_compute_not_network() -> None:
    ix = I.interaction(result(**LIVE, **{"live.loopback_round_seconds": 0.01}, loopback_method="BORROWED from r1"))
    assert ix["loopback_s"] == pytest.approx(0.03) and ix["wait_s"] == pytest.approx(0.103 - 0.03)
    assert ix["prover_s"] == pytest.approx(0.43) and ix["loopback_method"] == "BORROWED from r1"


def test_file_verified_rounds_multiply_by_the_sequential_sub_batch_waves() -> None:
    fp = {"B": 32768, "N_subbatches": 193, "software": {"backend": {"name": "ligero", "pipeline": 4}}}
    meta = result(t_total=7.0, fp=fp, **{"rounds.sequential_depth": 3, "proof_bytes": 1.25e9, "relation.statement_bytes_total": 5e6})
    ix = I.interaction(meta, interactive=True)
    assert ix["derived"] and not ix["live"] and ix["rounds"] == 3 * 49 and ix["rounds_source"] == "derived: file-verified"
    assert ix["rtt_ms"] is None and ix["bytes_up"] == 1.255e9 and I.interaction_problem(ix) is None
    fp["software"]["backend"].pop("pipeline")
    assert I.interaction(meta, interactive=True)["rounds"] == 3 * 193
    fs = I.interaction(meta)
    assert fs["rounds"] == 0 and fs["rounds_source"] == "fiat-shamir" and fs["derived"]


@pytest.mark.parametrize(("over", "why"), [
    ({"rounds.sequential_depth": None}, "a live-verifier result without its rounds"),
    ({"net.rtt_ms": None}, "no measured RTT"),
    ({"rtt_method": None}, "no how the RTT was measured"),
    ({"net.wait_seconds": None}, "no network wait"),
    ({"net.wait_seconds": 2.5}, "% from the model's"),
])
def test_a_live_result_needs_its_record_and_the_tolerance(over: dict, why: str) -> None:
    ix = I.interaction(result(**{k: v for k, v in {**LIVE, **over}.items() if v is not None}))
    assert why in (I.interaction_problem(ix) or "")


def test_the_one_sided_check_lets_a_run_faster_than_the_model_pass() -> None:
    ix = {"live": True, "rounds": 10, "rtt_ms": 1.0, "rtt_method": "ping", "wait_s": 0.0, "prover_s": 0.5, "verifier_s": 0.0,
          "bytes_up": 0.0, "bytes_down": 0.0}
    assert I.interaction_problem(ix) is None and I.interaction_problem(ix, one_sided=True) is None   # -2%: inside both
    over = {**ix, "wait_s": 0.2}                                    # 0.7 s measured vs a 0.51 s model: over it, fails both ways
    under = {**ix, "prover_s": 0.3, "rounds": 300}                  # 0.3 s measured vs a 0.6 s model: under it, fails only two-sided
    assert I.interaction_problem(over) and I.interaction_problem(over, one_sided=True)
    assert I.interaction_problem(under) and I.interaction_problem(under, one_sided=True) is None


def test_the_published_rtt_method_is_an_enum_never_the_recorded_text() -> None:
    assert I.rtt_method_enum("median of 20 TCP connects to the verifier pod's ssh port (10.0.0.1:2222)") == "tcp-connect"
    assert I.rtt_method_enum("net.rtt_ms = round trip of a 64-byte Ping request on the open session connection to the verifier") \
        == "open-connection-ping"
    assert I.rtt_method_enum("iperf") == "unclassified" and I.rtt_method_enum(None) is None


@pytest.mark.parametrize(("protocol", "sweep", "why"), [
    (None, None, "protocol not recorded"),
    ({"warm": True, "runs": 4, "contended": False}, {"plateau": True}, "need warm, >= 5, uncontended"),
    ({"warm": True, "runs": 5, "contended": True}, {"plateau": True}, "need warm, >= 5, uncontended"),
    ({"warm": True, "runs": 5, "contended": False}, {"plateau": False}, "not a plateau point"),
    ({"warm": True, "runs": 5, "contended": False}, {"plateau": True}, None),
])
def test_the_protocol_is_warm_timed_uncontended_and_a_plateau_point(protocol, sweep, why) -> None:
    fp = {k: v for k, v in (("protocol", protocol), ("sweep", sweep)) if v is not None}
    got = I.protocol_problem({"workload_fingerprint": fp})
    assert (got is None) if why is None else (why in got)
