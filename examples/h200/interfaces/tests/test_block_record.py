"""The block record: the fixture records (built by `from-runs` from two all-protocols runs and a wipe run) fit their schema
and check clean, every derived field is refused when it is not its legs', and a period whose six legs all accept on one
server composes to accept."""
from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import jsonschema
import pytest

from verity_e2e_demo import block_record as R

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA = json.loads((Path(R.__file__).parent / "schemas" / "block-record.v1.schema.json").read_text())
NAMES = ("block-record-r2-period0", "block-record-r1-window10")


def _load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.mark.parametrize("name", NAMES)
def test_the_fixture_records_fit_their_schema_and_check_clean(name: str) -> None:
    rec = _load(name)
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert R.check(rec) == {}


def test_period0_is_five_real_legs_and_a_wipe_elsewhere() -> None:
    rec = _load("block-record-r2-period0")
    assert {k: v["quantity"]["value"] for k, v in rec["legs"].items()} == {
        "integrity": 299, "pouw": 262144, "pous": 0, "bandwidth": 6963200, "certifier": 15764, "wipe": 268435456}
    assert all(v["source"] == "real" for v in rec["legs"].values())
    c = rec["composed"]
    assert (c["verdict"], c["missing"], c["elsewhere"]) == ("pending", ["wipe"], ["wipe"])
    assert Fraction(c["joint_horizon"]["error_sum"]) == (Fraction(1, 2) + Fraction(1, 100) + Fraction(1, 10**6)
                                                         + Fraction(4194729, 2**62))
    assert c["joint_horizon"]["at"]["1/1000000"]["not_established"] == ["certifier"]
    assert all(p["joint_horizon"]["vacuous"] for p in rec["monitoring_period"]["periods"].values())


def test_integrity_bounds_are_the_runs() -> None:
    assert R.integrity_bound(1024, 2, Fraction(1, 2)) == 299
    assert R.integrity_bound(1024, 64, Fraction(1, 100)) == 68
    at = _load("block-record-r1-window10")["legs"]["integrity"]["at"]
    assert (at["1/100"]["value"], at["1/1000000"]["value"]) == (68, 192)


def test_a_flipped_leg_verdict_is_refused() -> None:
    rec = copy.deepcopy(_load("block-record-r2-period0"))
    rec["legs"]["pouw"]["answer"]["verdict"] = "reject"
    assert set(R.check(rec)) == {"composed:verdict", "composed:legs", "composed:joint_horizon", "monitoring_period"}


def test_a_changed_error_is_refused() -> None:
    rec = copy.deepcopy(_load("block-record-r2-period0"))
    rec["legs"]["certifier"]["error"] = "1/1000000"
    assert "leg:certifier" in R.check(rec)


def test_another_server_or_an_empty_block_is_refused() -> None:
    rec = copy.deepcopy(_load("block-record-r1-window10"))
    rec["server_time"] = [{"server": "elsewhere", "steps": [0, 2]}]
    assert "server_time" in R.check(rec)
    rec["server_time"] = [{"server": rec["server"], "steps": [2, 2]}]
    assert "server_time" in R.check(rec)


def test_a_changed_monitoring_period_is_refused() -> None:
    rec = copy.deepcopy(_load("block-record-r1-window10"))
    rec["monitoring_period"]["periods"]["week"]["blocks"] += 1
    assert R.check(rec) == {"monitoring_period": "not the deployment's periods at the block's error sum"}


def _accepting_report() -> tuple[dict, dict]:
    """r2's period 0 with its wipe on its own server: every leg accepts there."""
    p0 = _load("block-record-r2-period0")
    profile = {"levels": [{"name": "unit", "size": 1024}], "drawn": {"unit": 2}}
    rows = {}
    for leg, key in R.RUNNER_KEYS.items():
        lg = p0["legs"][leg]
        q = {"integrity": {"bound": 299}, "pouw": {"work": {"credited": 262144, "credited_tiles": 128}},
             "pous": {"s_bits": 0, "certified": False, "eps": "0"}, "bandwidth": {"q_bytes": 6963200, "eps": lg["error"]},
             "certifier": {"K": 15764, "windows": 1, "eps": "1/100"}}[leg]
        rows[key] = {"answer": {**lg["answer"], "protocol": lg["protocol"], "profile": profile}, "quantity": q,
                     "delta": lg["error"], "owner": lg["owner"], "runs": lg["runs"], "verdict": "accept"}
    report = {"format": "verity/all-protocols-period/v1", "run": p0["run"], "period": 0, "node": p0["server"],
              "protocols": rows, "composed": None}
    w = p0["legs"]["wipe"]
    wipe = {"row": {"answer": {**w["answer"], "protocol": w["protocol"], "profile": {"k": w["quantity"]["k"]}}},
            "server": p0["server"], "runs": w["runs"], "source": "fixture"}
    return report, wipe


def test_the_wipe_is_its_threshold_at_its_reads() -> None:
    """The wipe's quantity is `verity/wipe/v0`'s `foreign_bits` at the `k` and `r` it publishes; the fixtures' wipe, from
    before it published them, carries its profile's `k` and no `r`."""
    assert _load("block-record-r2-period0")["legs"]["wipe"]["quantity"] == {
        "symbol": "foreign_bits", "value": 1 << 28, "unit": "bits", "k": 2152, "r": None, "M": 1 << 28, "certified": False}
    public = {"certified": True, "egress_mib": 0, "eps": f"1/{2 ** 128}", "foreign_bits": 39_721_768_854, "k": 5231,
              "memory_bits": 1_146_400_000_000, "r": 59}
    lg = R.leg("wipe", {"answer": {"protocol": "verity/wipe/v0", "verdict": "accept", "reason": None, "commitment": "00",
                                   "public": public, "profile": {"k": 5231}}}, server="n0", runs=["r"])
    assert (lg["quantity"]["value"], lg["quantity"]["k"], lg["quantity"]["r"]) == (39_721_768_854, 5231, 59)
    assert lg["error"] == f"1/{2 ** 128}" and all(at["holds"] for at in lg["at"].values())


def test_six_accepting_legs_on_one_server_compose_to_accept() -> None:
    report, wipe = _accepting_report()
    rec = R.from_period(report, steps=(4, 6), deployment=R.REFERENCE, extra={"wipe": wipe})
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert R.check(rec) == {}
    assert (rec["composed"]["verdict"], rec["composed"]["missing"], rec["composed"]["onsite"]) == ("accept", [], None)
    assert rec["legs"]["wipe"]["source"] == "fixture" and rec["composed"]["joint_horizon"]["outside_statement"] == ["wipe"]
    assert rec["server_time"] == [{"server": report["node"], "steps": [4, 6]}]
    week = rec["monitoring_period"]["periods"]["week"]
    assert week["blocks"] == 12500 * 1008 and week["per_block_error_for"]["1/100"] == str(Fraction(1, 100 * 12500 * 1008))


def test_a_report_of_another_format_is_refused() -> None:
    report, _ = _accepting_report()
    with pytest.raises(ValueError, match="not one of"):
        R.from_period({**report, "format": "verity/all-protocols-window/v0"}, steps=(0, 2), deployment=R.REFERENCE)
