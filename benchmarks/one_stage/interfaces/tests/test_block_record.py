"""The block record: the fixture records (built by `from-runs` from two all-protocols runs and a wipe run) fit their schema
and check clean, every derived field is refused when it is not its legs' or its answer's, the verdict is the Lean
onsite or remote check's answer, undecided when none ran, whatever the legs say, and remote's product composes sampled
proofs and PoUW alone."""
from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import jsonschema
import pytest

from verity.periods import block_count
from verity.protocols.reporting import PERIODS
from verity_e2e_demo import block_record as R

FIXTURES = Path(__file__).parent / "fixtures"
#: the census's deployments, which the block record's reference deployment is held to here (the record itself is public
#: and the census is the team's, so the record carries the entry and doesn't read the file)
CENSUS = Path(__file__).resolve().parents[3] / "census" / "deployments.json"
SCHEMA = json.loads((Path(R.__file__).parent / "schemas" / "block-record.v2.schema.json").read_text())
NAMES = ("block-record-r2-period0", "block-record-r1-window10", "block-record-b71a-w30")
ONSITE = {"protocol": "verity/onsite/v0", "verdict": "accept", "reason": None, "commitment": "ab" * 64, "public": None}


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
    assert (c["verdict"], c["answer"], c["missing"], c["elsewhere"]) == ("undecided", None, ["wipe"], ["wipe"])
    assert Fraction(c["joint_horizon"]["error_sum"]) == (Fraction(1, 2) + Fraction(1, 100) + Fraction(1, 10**6)
                                                         + Fraction(4194729, 2**62))
    assert c["joint_horizon"]["at"]["1/1000000"]["not_established"] == ["certifier"]
    assert all(p["joint_horizon"]["vacuous"] for p in rec["monitoring_period"]["periods"].values())


def test_integrity_bounds_are_the_profiles_count_bound() -> None:
    """The integrity line at δ is `profile.Stratified`'s count bound (Lean's `countBound`): the largest K with
    miss(K) ≥ δ, so at miss = δ exactly (one of 100 units sampled, 99 wrong) it is 99, where the copy this replaced said 98.
    A quantity with no strata, as the fixture's, is bounded as a uniform sample of its checked level."""
    integrity = _load("block-record-r1-window10")["legs"]["integrity"]
    at = integrity["at"]
    assert "strata" not in integrity["quantity"]
    assert (at["1/100"]["value"], at["1/1000000"]["value"]) == (68, 192)
    q = {"symbol": "integrity bound", "value": 0, "unit": "wrong units", "of": 100, "drawn": 1}
    assert R.at_deltas("integrity", q, Fraction(1, 2))["1/100"] == {"holds": True, "value": 99, "epsilon": "99/100",
                                                                    "by": "draw"}


def test_the_integrity_bound_counts_a_sampling_term_equal_to_delta() -> None:
    """Of 4 units with 2 sampled, 2 wrong ones escape with C(2, 2)/C(4, 2) = 1/6 exactly, so at δ = 1/6 the bound is 2
    (core's `Stratified.count_bound`: the largest K whose sampling term is at least δ), not 1."""
    assert R.integrity_bound(4, 2, Fraction(1, 6)) == 2
    assert R.integrity_bound(4, 2, Fraction(1, 6) + Fraction(1, 10**9)) == 1


def test_a_stratified_sample_is_bounded_by_its_strata_and_not_as_a_uniform_one() -> None:
    """An answer whose profile samples each stratum on its own is bounded by that sampling design. The fixture profile
    is a served run's: all 160 units of one template and 2 of another's 4. An incorrect proof block among the 160 is
    always caught; among the 4, one escapes with C(3, 2)/C(4, 2) = 1/2, two with C(2, 2)/C(4, 2) = 1/6 and three never.
    So at most 2 of the 164 are incorrect at δ = 1/100 and at 1/10^6, 1 only for δ above 1/6 and none above 1/2. Read
    as a uniform sample of 162 of 164, where one escapes with 1/82 and two with 1/13,366, the same counts gave 1 at
    1/100.
    The quantity carries the strata, so `check` recomputes the stratified bound, and refuses a record that dropped
    them. Strata that are not the quantity's own counts are refused."""
    profile = json.loads((FIXTURES / "integrity-profile-stratified.json").read_text())
    report, wipe = _accepting_report()
    row = report["protocols"][R.RUNNER_KEYS["integrity"]]
    row["answer"] = {**row["answer"], "profile": profile}
    rec = R.from_period(report, steps=(4, 6), deployment=R.REFERENCE, extra={"wipe": wipe})
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert R.check(rec) == {}
    integrity = rec["legs"]["integrity"]
    strata = [{"name": "RoPEHead_v1{D=64}", "size": 160, "k": 160}, {"name": "SiluMul_v2{I=8192}", "size": 4, "k": 2}]
    assert integrity["quantity"]["strata"] == strata and integrity["quantity"]["of"] == 164
    assert {d: (at["value"], at["epsilon"]) for d, at in integrity["at"].items()} == {
        "1/100": (2, "1/82"), "1/1000000": (2, "1/82")}
    sixth = Fraction(1, 6)
    assert [R.integrity_bound(164, 162, d, strata) for d in (sixth, sixth + Fraction(1, 10**9), Fraction(1, 2),
                                                               Fraction(1, 2) + Fraction(1, 10**9))] == [2, 1, 1, 0]
    assert [R.integrity_bound(164, 162, d) for d in (Fraction(1, 100), Fraction(1, 10**6))] == [1, 2]
    del integrity["quantity"]["strata"]
    assert list(R.check(rec).values()) == ["its certificates at the deltas are not its quantity's at its error"]
    with pytest.raises(ValueError, match="the strata hold 164 units and sample 162 of them, not the 161 of 164"):
        R.integrity_bound(164, 161, Fraction(1, 100), strata)
    row["answer"]["profile"] = {**profile, "levels": [{**profile["levels"][-1], "size": 165}]}
    with pytest.raises(ValueError, match="the strata hold 164 units and sample 162 of them, not the 162 of 165"):
        R.from_period(report, steps=(4, 6), deployment=R.REFERENCE, extra={"wipe": wipe})


def test_a_flipped_leg_verdict_is_refused() -> None:
    rec = copy.deepcopy(_load("block-record-r2-period0"))
    rec["legs"]["pouw"]["answer"]["verdict"] = "reject"
    assert set(R.check(rec)) == {"composed:legs", "composed:joint_horizon", "monitoring_period"}


def test_the_verdict_is_the_answers_and_nothing_else() -> None:
    """A record's verdict is its Lean answer's: an accepting onsite answer with a leg missing is accept, a rejecting one
    over six accepting legs is reject, and a verdict of the record's own, or an answer of another protocol or with a
    verdict that is not `Core.Verdict`'s, is refused."""
    rec = _load("block-record-r2-period0")
    assert rec["composed"]["missing"] == ["wipe"]
    for verdict in R.VERDICTS:
        a = {**ONSITE, "verdict": verdict, "reason": None if verdict == "accept" else "why"}
        c = R.composed(rec["server"], rec["legs"], a)
        assert (c["verdict"], c["answer"]) == (verdict, {k: a[k] for k in ("protocol", "verdict", "reason", "commitment")})
        mine = rec | {"composed": c}
        jsonschema.Draft202012Validator(SCHEMA).validate(mine)
        assert R.check(mine) == {}
    forged = copy.deepcopy(rec)
    forged["composed"]["verdict"] = "accept"
    assert R.check(forged) == {"composed:verdict": "not its answer's verdict"}
    for bad in ({**ONSITE, "protocol": "verity/pouw/period/v1"}, {**ONSITE, "verdict": "pending"}):
        with pytest.raises(ValueError, match="not an answer of"):
            R.composed(rec["server"], rec["legs"], bad)
        assert "composed:answer" in R.check(rec | {"composed": {**rec["composed"], "answer": bad}})


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


def test_pous_writes_its_state_bound_and_a_record_under_the_older_name_still_reads() -> None:
    """PoUS's quantity is written as `S0`, the verifier's `s_bits`, in a record built here as in the fixtures, and no
    other quantity has that symbol. The same record under `S_cert`, the quantity's older name, fits the schema and
    checks clean, and a symbol the schema doesn't list is refused."""
    row = {"answer": {}, "quantity": {"s_bits": 7, "certified": True, "eps": "1/8"}}
    assert R._quantity("pous", row) == ({"symbol": "S0", "value": 7, "unit": "bits", "certified": True},
                                        Fraction(1, 8), "quantity.eps")
    report, wipe = _accepting_report()
    built = R.from_period(report, steps=(4, 6), deployment=R.REFERENCE, extra={"wipe": wipe})
    validator = jsonschema.Draft202012Validator(SCHEMA)
    for written in (built, _load("block-record-r2-period0"), _load("block-record-b71a-w30")):
        text, new = json.dumps(written), '"symbol": "S0"'
        assert text.count(new) == 1 and "S_cert" not in text
        older = json.loads(text.replace(new, '"symbol": "S_cert"'))
        validator.validate(older)
        assert R.check(older) == {}
        assert not validator.is_valid(json.loads(text.replace(new, '"symbol": "S"')))


def test_six_accepting_legs_on_one_server_are_undecided_until_the_lean_check_answers() -> None:
    report, wipe = _accepting_report()
    rec = R.from_period(report, steps=(4, 6), deployment=R.REFERENCE, extra={"wipe": wipe})
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert R.check(rec) == {}
    assert (rec["composed"]["verdict"], rec["composed"]["missing"], rec["composed"]["answer"]) == ("undecided", [], None)
    decided = R.from_period(report, steps=(4, 6), deployment=R.REFERENCE, answer=ONSITE, extra={"wipe": wipe})
    assert R.check(decided) == {} and decided["composed"]["verdict"] == "accept"
    assert rec["legs"]["wipe"]["source"] == "fixture" and rec["composed"]["joint_horizon"]["outside_statement"] == ["wipe"]
    assert rec["server_time"] == [{"server": report["node"], "steps": [4, 6]}]
    week = rec["monitoring_period"]["periods"]["week"]
    assert week["blocks"] == 12500 * 1008 and week["per_block_error_for"]["1/100"] == str(Fraction(1, 100 * 12500 * 1008))


def test_remote_composes_spot_and_pouw_alone_its_other_legs_dropped() -> None:
    """Remote's product: sampled proofs and PoUW accepted on the block's server, no onsite leg run, compose with the onsite
    legs named dropped, neither missing nor in the error sum, and are undecided until the remote check answers; as
    onsite's the same legs miss four. The remote check's answer decides it, and an onsite answer is refused. A record of
    remote's that ran a leg remote drops, that names no product, or that names an unknown one is refused."""
    report, _ = _accepting_report()
    keep = {R.RUNNER_KEYS[leg] for leg in R.PRODUCTS["remote"]}
    report["protocols"] = {k: v for k, v in report["protocols"].items() if k in keep}
    rec = R.from_period(report, steps=(4, 6), deployment=R.REFERENCE, product="remote")
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert R.check(rec) == {}
    c = rec["composed"]
    assert (c["verdict"], c["answer"], c["product"], c["missing"], c["dropped"]) == (
        "undecided", None, "remote", [], ["bandwidth", "certifier", "pous", "wipe"])
    assert c["legs"] == {"integrity": "accept", "pouw": "accept"} and sorted(c["joint_horizon"]["terms"]) == ["integrity", "pouw"]
    remote = {**ONSITE, "protocol": "verity/remote/v0"}
    decided = R.from_period(report, steps=(4, 6), deployment=R.REFERENCE, answer=remote, product="remote")
    jsonschema.Draft202012Validator(SCHEMA).validate(decided)
    assert R.check(decided) == {} and decided["composed"]["verdict"] == "accept"
    with pytest.raises(ValueError, match="not an answer of the remote check"):
        R.from_period(report, steps=(4, 6), deployment=R.REFERENCE, answer=ONSITE, product="remote")
    with pytest.raises(ValueError, match="not an answer of the onsite check"):
        R.from_period(report, steps=(4, 6), deployment=R.REFERENCE, answer=remote)
    onsite = R.from_period(report, steps=(4, 6), deployment=R.REFERENCE)["composed"]
    assert (onsite["verdict"], onsite["missing"]) == ("undecided", ["bandwidth", "certifier", "pous", "wipe"])
    full, wipe = _accepting_report()
    with pytest.raises(ValueError, match="legs the remote product drops"):
        R.from_period(full, steps=(4, 6), deployment=R.REFERENCE, extra={"wipe": wipe}, product="remote")
    ran = copy.deepcopy(rec)
    ran["legs"]["pous"] = R.from_period(full, steps=(4, 6), deployment=R.REFERENCE)["legs"]["pous"]
    assert R.check(ran) == {"legs": "pous: legs the remote product drops, which the block ran"}
    unnamed = copy.deepcopy(rec)
    del unnamed["composed"]["product"], unnamed["composed"]["dropped"]
    assert set(R.check(unnamed)) == {"composed:missing"}
    unnamed = copy.deepcopy(decided)
    del unnamed["composed"]["product"], unnamed["composed"]["dropped"]
    assert set(R.check(unnamed)) == {"composed:answer"}
    unknown = copy.deepcopy(rec)
    unknown["composed"]["product"] = "offsite"
    assert R.check(unknown) == {"composed:product": "not one of ['onsite', 'remote']"}


def test_a_leg_carries_its_answers_placement_and_no_other_profile() -> None:
    """The certifier's answer names which host ran each party (``profile.placement``): its leg carries that and the
    record still fits its schema and checks clean; a leg whose profile names no placement carries no profile."""
    report, wipe = _accepting_report()
    placement = {"kind": "not-placed", "parties": {"network_certifier": "gate", "prover": "n0", "verifier": "n0"}}
    row = report["protocols"][R.RUNNER_KEYS["certifier"]]
    row["answer"] = {**row["answer"], "profile": {**row["answer"]["profile"], "placement": placement}}
    rec = R.from_period(report, steps=(4, 6), deployment=R.REFERENCE, extra={"wipe": wipe})
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert R.check(rec) == {}
    assert rec["legs"]["certifier"]["answer"]["profile"] == {"placement": placement}
    assert all("profile" not in lg["answer"] for k, lg in rec["legs"].items() if k != "certifier")


def test_a_report_of_another_format_is_refused() -> None:
    report, _ = _accepting_report()
    with pytest.raises(ValueError, match="not one of"):
        R.from_period({**report, "format": "verity/all-protocols-window/v0"}, steps=(0, 2), deployment=R.REFERENCE)


def test_the_reference_deployment_is_the_census_entry_the_fixtures_carry() -> None:
    census = [d for d in json.loads(CENSUS.read_text()) if d["role"] == "reference"]
    assert len(census) == 1 and R.REFERENCE == {k: v for k, v in census[0].items() if k != "sources"}
    for name in (*NAMES, "block-record-b71a-w30"):
        assert _load(name)["monitoring_period"]["deployment"] == R.REFERENCE


def test_a_period_s_blocks_are_core_s_block_count() -> None:
    window_s, machines = R.REFERENCE["window_s"], R.REFERENCE["machines"]
    periods = R.monitoring_period(R.REFERENCE, Fraction(1, 2**40))["periods"]
    assert [(name, p["seconds"]) for name, p in periods.items()] == list(PERIODS)
    for p in periods.values():
        assert p["blocks_per_machine"] == block_count(p["seconds"], window_s)
        assert p["blocks"] == block_count(p["seconds"], window_s, machines)
