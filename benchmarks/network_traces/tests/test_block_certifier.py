"""The block record's certifier entry (`block_certifier`): `block_leg` turns a period's accepted answer (the
fixture's period, a run's) into the block record's certifier leg, K held at each δ its exact eps is at most, and refuses
a rejected answer, another protocol, a source other than real or fixture, and no run; each leg fits
``verity/block-record/v2``'s schema, is `block_record.leg`'s of the runner's row, reproduces its fixture's certifier leg
from that leg's answer, and a fixture record carrying it checks clean.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import jsonschema
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
import block_certifier as BC  # noqa: E402

ANSWERS = json.loads((Path(__file__).parent / "fixtures" / "certifier-period-answers.json").read_text())
(ROWS_RUN,) = ANSWERS
E2E = ROOT / "benchmarks/one_stage/interfaces"
BLOCK_SCHEMA = E2E / "verity_e2e_demo" / "schemas" / "block-record.v2.schema.json"


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _block_leg(run: str, delta: str) -> dict:
    return BC.block_leg(ANSWERS[run]["answers"][delta], server=ANSWERS[run]["node"], runs=[run])


def _held(value: int) -> dict:
    return {"holds": True, "value": value, "epsilon": None, "by": "run"}


def _not_held(eps: str, delta: str) -> dict:
    return {"holds": False, "why": f"the run established it except with {eps}, more than {delta}"}


def test_a_period_answer_is_the_block_records_certifier_leg():
    rows = _block_leg(ROWS_RUN, "1/100")
    assert (rows["protocol"], rows["owner"], rows["server"], rows["source"], rows["runs"]) == (
        BC.PROTOCOL, "network-accounting", ANSWERS[ROWS_RUN]["node"], "real", [ROWS_RUN])
    assert rows["answer"] == {k: ANSWERS[ROWS_RUN]["answers"]["1/100"][k]
                              for k in ("verdict", "reason", "commitment", "public")}
    assert "scans" in rows["answer"]["public"] and rows["quantity"] == {"symbol": "K", "value": 12200606,
                                                                         "unit": "bits", "windows": 1}
    assert (rows["error"], rows["error_from"]) == ("1/50", "quantity.eps")
    assert rows["at"] == {"1/100": _not_held("1/50", "1/100"), "1/1000000": _not_held("1/50", "1/1000000")}
    rows6 = _block_leg(ROWS_RUN, "1/1000000")
    assert rows6["error"] == "1/500000" and rows6["at"] == {"1/100": _held(12202528),
                                                            "1/1000000": _not_held("1/500000", "1/1000000")}
    answer = ANSWERS[ROWS_RUN]["answers"]["1/100"]
    with pytest.raises(ValueError, match="certifies no K"):
        BC.block_leg(answer | {"verdict": "reject", "reason": "certificates: a link-window"}, server="n",
                     runs=[ROWS_RUN])
    with pytest.raises(ValueError, match="not verity/certifier/v0"):
        BC.block_leg(answer | {"protocol": "verity/sampled-proofs/one-stage/v1"}, server="n", runs=[ROWS_RUN])
    with pytest.raises(ValueError, match="real or fixture"):
        BC.block_leg(answer, server="n", runs=[ROWS_RUN], source="synthetic")
    with pytest.raises(ValueError, match="at least one run"):
        BC.block_leg(answer, server="n", runs=[])


def test_the_block_leg_fits_the_block_records_schema_its_fixture_and_its_runner_path():
    schema = json.loads(BLOCK_SCHEMA.read_text())
    B = _module("verity_e2e_demo_block_record", E2E / "verity_e2e_demo" / "block_record.py")
    assert [str(d) for d in BC.BLOCK_DELTAS] == [str(d) for d in B.DELTAS]
    leg_schema = jsonschema.Draft202012Validator({"$schema": schema["$schema"], "$defs": schema["$defs"],
                                                  "$ref": "#/$defs/leg"})
    for run, x in ANSWERS.items():
        for delta, a in x["answers"].items():
            got = BC.block_leg(a, server=x["node"], runs=[run])
            leg_schema.validate(got)
            row = {"answer": a, "quantity": a["public"], "owner": "network-accounting", "runs": [run]}
            assert B.leg("certifier", row, server=x["node"], runs=[run]) == got
    rec = json.loads((E2E / "tests" / "fixtures" / "block-record-r2-period0.json").read_text())
    fx = rec["legs"]["certifier"]
    assert BC.block_leg({**fx["answer"], "protocol": fx["protocol"]}, server=fx["server"], runs=fx["runs"]) == fx
    legs = rec["legs"] | {"certifier": BC.block_leg(ANSWERS[ROWS_RUN]["answers"]["1/1000000"], server=rec["server"],
                                                    runs=[ROWS_RUN])}
    composed = B.composed(rec["server"], legs, rec["composed"]["answer"])
    mine = rec | {"legs": legs, "composed": composed, "monitoring_period": B.monitoring_period(
        rec["monitoring_period"]["deployment"], Fraction(composed["joint_horizon"]["error_sum"]))}
    jsonschema.Draft202012Validator(schema).validate(mine)
    assert B.check(mine) == {}
    assert composed["joint_horizon"]["at"]["1/1000000"]["not_established"] == ["certifier"]
