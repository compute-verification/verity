"""The block record's certifier entry and the period check's answer (`block_certifier`):

* `block_leg` turns a period's accepted answer, clock form or rows form (the fixture's two periods, each a run's), into
  the block record's certifier leg, K held at each δ its exact eps is at most, and refuses a rejected answer, another
  protocol, a source other than real or fixture, and no run; each leg fits ``verity/block-record/v1``'s schema, is
  `block_record.leg`'s of the runner's row, reproduces its fixture's certifier leg from that leg's answer, and a fixture
  record carrying it checks clean;
* `check` hands ``network-period check`` the case's params (canonical JSON), commitment and evidence as its three files
  and reads its answer off the last line it prints, a rejection's too, and refuses another exit or no answer.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from verity.ml.circuits.codec import canonical_json

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import block_certifier as BC  # noqa: E402

ANSWERS = json.loads((Path(__file__).parent / "fixtures" / "certifier-period-answers.json").read_text())
CLOCK_RUN, ROWS_RUN = "r20261025-213746-7692", "r20261025-235229-305e"
E2E = ROOT / "examples" / "h200" / "interfaces"
BLOCK_SCHEMA = E2E / "verity_e2e_demo" / "schemas" / "block-record.v1.schema.json"


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


def test_a_period_answer_of_either_form_is_the_block_records_certifier_leg():
    clock = _block_leg(CLOCK_RUN, "1/100")
    assert (clock["protocol"], clock["owner"], clock["server"], clock["source"], clock["runs"]) == (
        BC.PROTOCOL, "network-accounting", ANSWERS[CLOCK_RUN]["node"], "real", [CLOCK_RUN])
    assert clock["answer"] == {k: ANSWERS[CLOCK_RUN]["answers"]["1/100"][k]
                               for k in ("verdict", "reason", "commitment", "public")}
    assert clock["quantity"] == {"symbol": "K", "value": 2178452, "unit": "bits", "windows": 1}
    assert (clock["error"], clock["error_from"]) == ("1/100", "quantity.eps")
    assert clock["at"] == {"1/100": _held(2178452), "1/1000000": _not_held("1/100", "1/1000000")}
    assert _block_leg(CLOCK_RUN, "1/1000000")["at"] == {"1/100": _held(2180068), "1/1000000": _held(2180068)}
    rows = _block_leg(ROWS_RUN, "1/100")
    assert "scans" in rows["answer"]["public"] and rows["quantity"]["value"] == 12200606
    assert rows["error"] == "1/50" and rows["at"] == {"1/100": _not_held("1/50", "1/100"),
                                                       "1/1000000": _not_held("1/50", "1/1000000")}
    rows6 = _block_leg(ROWS_RUN, "1/1000000")
    assert rows6["error"] == "1/500000" and rows6["at"] == {"1/100": _held(12202528),
                                                            "1/1000000": _not_held("1/500000", "1/1000000")}
    answer = ANSWERS[CLOCK_RUN]["answers"]["1/100"]
    with pytest.raises(ValueError, match="certifies no K"):
        BC.block_leg(ANSWERS[CLOCK_RUN]["tampered"], server="n", runs=[CLOCK_RUN])
    with pytest.raises(ValueError, match="not verity/certifier/v0"):
        BC.block_leg(answer | {"protocol": "verity/sampled-proofs/one-stage/v1"}, server="n", runs=[CLOCK_RUN])
    with pytest.raises(ValueError, match="real or fixture"):
        BC.block_leg(answer, server="n", runs=[CLOCK_RUN], source="synthetic")
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
    for run, delta in ((CLOCK_RUN, "1/100"), (ROWS_RUN, "1/1000000")):
        legs = rec["legs"] | {"certifier": BC.block_leg(ANSWERS[run]["answers"][delta], server=rec["server"],
                                                        runs=[run])}
        composed = B.compose(rec["server"], legs) | {"onsite": rec["composed"]["onsite"]}
        mine = rec | {"legs": legs, "composed": composed, "monitoring_period": B.monitoring_period(
            rec["monitoring_period"]["deployment"], Fraction(composed["joint_horizon"]["error_sum"]))}
        jsonschema.Draft202012Validator(schema).validate(mine)
        assert B.check(mine) == {}
        assert composed["joint_horizon"]["at"]["1/1000000"]["not_established"] == ["certifier"]


def test_check_hands_network_period_the_cases_files_and_blobs_and_reads_its_last_line(monkeypatch, tmp_path):
    case = {"params": {"windows": 1, "first": 0, "key": "ab"}, "commitment": b'{"audits":[]}'.hex(),
            "evidence": b"\x00\x01".hex()}
    accept, reject = {"verdict": "accept", "public": {"K": 7}}, {"verdict": "reject", "reason": "a window"}
    seen, said = [], {}

    def run(argv, **kw):
        seen.append((argv[:2], [Path(f).name for f in argv[2:]], [Path(f).read_bytes() for f in argv[2:]]))
        return subprocess.CompletedProcess(argv, said["rc"], said["out"], "the check's stderr")

    monkeypatch.setattr(BC, "subprocess", SimpleNamespace(run=run))
    for rc, answer in ((0, accept), (1, reject), (2, reject)):
        said.update(rc=rc, out=f"a progress line\n{json.dumps(answer)}\n")
        assert BC.check("network-period", case) == answer
    assert seen[0] == (["network-period", "check"], ["params.json", "commitment.bin", "evidence.json"],
                       [canonical_json(case["params"]), b'{"audits":[]}', b"\x00\x01"])
    for rc, out in ((3, json.dumps(accept)), (0, "\n")):
        said.update(rc=rc, out=out)
        with pytest.raises(RuntimeError, match=f"network-period check exited {rc}: the check's stderr"):
            BC.check("network-period", case)
    (blob := tmp_path / "w001-pub-1.bin").write_bytes(b"pub")
    said.update(rc=0, out=json.dumps(accept))
    assert BC.check("network-period", case | {"blobs": [blob]}) == accept
    assert seen[-1][1:] == (["params.json", "commitment.bin", "evidence.json", "w001-pub-1.bin"],
                            [canonical_json(case["params"]), b'{"audits":[]}', b"\x00\x01", b"pub"])
