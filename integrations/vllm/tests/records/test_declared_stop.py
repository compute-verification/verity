"""`records.declared_stop`: the workload Program's declared stop rule and LAG as `pipeline.global_program` and the global match's
mock Program write them, refused by name when a field they all write is missing, and read by the global match's chronology check
(`check.match.chronology`, G6) to the same document as the parsed JSON.

`data/global_program.json` is a real record: the global match's global Program of the regression workload
`llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager` (`global_match_global_program.json`,
`tests/regression/files.toml`, sha256 ec2f476bb5cf...), byte for byte.  The G6 runs are the global match's toy
(`tests.check.test_global_match`), with the Program's two blocks rewritten to each writer's shape and to each way G6 reads them.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.check import test_global_match as T
from tests.observe.test_fold_synthetic import memory  # noqa: F401 - fixture
from verity_vllm.check.match import global_match as gm
from verity_vllm.records.declared_stop import DeclaredStop, LagBlock, StopRule
from verity_vllm.records.fields import MissingField, RecordError

DATA = Path(__file__).parent / "data"
REAL = DATA / "global_program.json"
#: sha256 of the global match's outcomes over `_corpus` (`json.dumps(..., sort_keys=True, default=str)`, the run directory replaced), computed
#: with G6 on the parsed JSON before it read the record
OUTCOMES_SHA256 = "1a920268eff23cacdd3b4b51bec598d9070633c029861bca65ebdbf202cfe75a"
REQUIRED = ("stop_rule.eos_token_ids", "lag.value")
#: The global Program's two blocks as `pipeline.global_program` writes them (the real record's, with the toy's EOS id and the LAG to declare)
GP01_SOURCE = {"generation_config.json": [T.EOS], "config.json": [T.EOS]}


def _real() -> dict:
    return json.loads(REAL.read_text())


def _gp01(lag: int) -> dict:
    real = _real()
    sr = dict(real["stop_rule"], eos_token_ids=[T.EOS], source=GP01_SOURCE, lag=lag, eos_lag_rows=lag)
    return {"stop_rule": sr, "lag": dict(real["lag"], value=lag, async_scheduling=bool(lag))}


def test_the_real_record_loads():
    d = DeclaredStop.of(_real(), str(REAL))
    real = _real()
    assert d.where == str(REAL) and d.declared_lag == 1
    assert d.stop_rule == StopRule(eos_token_ids=[128001], ignore_eos=None, source={"generation_config.json": [128001], "config.json": [128001]},
                                   lag=1, eos_lag_rows=1, eos_lag_source=real["stop_rule"]["eos_lag_source"])
    assert d.lag == LagBlock(value=1, source=real["lag"]["source"])
    assert DeclaredStop.of(d, "elsewhere") is d


def test_the_mock_programs_load(tmp_path):
    wl = T.workload([T.R0], tmp_path / "workload.json")
    lagged = DeclaredStop.of(gm.mock_global_program(wl, [], eos_token_ids=[T.EOS], lag=1), "mock")
    assert lagged.declared_lag == 1 and lagged.stop_rule.eos_token_ids == [T.EOS] and lagged.lag == LagBlock(1, "mock_global_program(lag=...)")
    pre = DeclaredStop.of(gm.mock_global_program(wl, []), "mock")
    assert pre.lag is None and pre.declared_lag is None and pre.stop_rule == StopRule(None, None, None, None, None, None)


def test_a_program_without_the_blocks_declares_nothing():
    d = DeclaredStop.of({"stop_rule": None, "lag": None}, "gp")
    assert d.stop_rule is None and d.lag is None and d.declared_lag is None
    assert DeclaredStop.of({}, "gp") == d


@pytest.mark.parametrize("sr,lag", [({"eos_token_ids": [1], "lag": 0, "eos_lag_rows": 0}, 0), ({"eos_token_ids": [1], "eos_lag_rows": 1}, 1),
                                    ({"eos_token_ids": [1]}, None)])
def test_the_stop_rule_declares_the_lag_without_a_lag_block(sr, lag):
    assert DeclaredStop.of({"stop_rule": sr}, "gp").declared_lag == lag


@pytest.mark.parametrize("path", REQUIRED)
def test_a_missing_required_field_is_refused(path):
    doc = _real()
    block, field = path.split(".")
    del doc[block][field]
    with pytest.raises(MissingField) as e:
        DeclaredStop.of(doc, "the global Program")
    assert str(e.value) == f"the global Program: field {path!r} is missing: every writer of the record writes it"


def test_a_null_lag_value_is_refused():
    doc = _real()
    doc["lag"]["value"] = None
    with pytest.raises(MissingField, match="field 'lag.value' is missing"):
        DeclaredStop.of(doc, "the global Program")


@pytest.mark.parametrize("path,value,kind", [("stop_rule", [1], "list, not a Mapping"), ("lag", 1, "int, not a Mapping"),
                                             ("stop_rule.eos_token_ids", 128001, "int, not a list or tuple"),
                                             ("stop_rule.ignore_eos", "yes", "str, not a bool"), ("stop_rule.source", "config.json", "str, not a Mapping"),
                                             ("stop_rule.lag", "1", "str, not a int"), ("stop_rule.eos_lag_rows", 1.0, "float, not a int"),
                                             ("stop_rule.eos_lag_source", 1, "int, not a str"), ("lag.value", "1", "str, not a int"),
                                             ("lag.source", ["x"], "list, not a str")])
def test_a_mistyped_field_is_refused(path, value, kind):
    doc = _real()
    if "." in path:
        block, field = path.split(".")
        doc[block][field] = value
    else:
        doc[path] = value
    with pytest.raises(RecordError) as e:
        DeclaredStop.of(doc, "the global Program")
    assert str(e.value) == f"the global Program: field {path!r} is a {kind}"


def test_a_non_object_is_refused():
    with pytest.raises(RecordError, match="field 'stop_rule' cannot be read: the Program is a list, not a JSON object"):
        DeclaredStop.of([], "the global Program")


def _outcome(gp: dict, oracle: str) -> dict:
    res = gm.check(gp, oracle, run_x09=False, record_alternate=False)
    return {"verdict": res["verdict"], "failed": res["failed_checks"], "G6": T._check(res, "G6"), "requests": res["requests"],
            "lag": res.get("lag"), "alerts": res.get("alerts")}


def _corpus(tmp_path: Path) -> dict[str, dict]:
    """The global match over the Program's two blocks as each writer writes them and as each G6 path reads them: served to the cap, an EOS stop
    under a synchronous Program, an EOS stop under a LAG-1 Program; declared, undeclared, disagreeing with the record."""
    out: dict[str, dict] = {}

    def run(name, gp, oracle, **blocks):
        g = copy.deepcopy(gp)
        for k, v in blocks.items():
            if v is _POP:
                g.pop(k, None)
            else:
                g[k] = v
        out[name] = _outcome(g, oracle)

    gp, oracle = T.assemble([T.R0, T.R1], tmp_path / "cap")
    run("cap, as the mock writes it", gp, oracle)
    run("cap, as GP-01 writes it", gp, oracle, **_gp01(0))
    run("cap, LAG 1 declared on a synchronous record", gp, oracle, **_gp01(1))
    run("cap, LAG 2 declared", gp, oracle, lag={"value": 2, "source": "test"})
    run("cap, no stop rule", gp, oracle, stop_rule=None)

    gp, oracle = T.assemble([T.R0, T.R1_EOS], tmp_path / "eos")
    sr = gp["stop_rule"]
    run("eos, as the mock writes it", gp, oracle)
    run("eos, as GP-01 writes it", gp, oracle, **_gp01(0))
    run("eos, no stop rule", gp, oracle, stop_rule=None)
    run("eos, the stop rule not written", gp, oracle, stop_rule=_POP)
    run("eos, no EOS set", gp, oracle, stop_rule=dict(sr, eos_token_ids=None))
    run("eos, ignore_eos", gp, oracle, stop_rule={"eos_token_ids": [], "ignore_eos": True})
    run("eos, another EOS id", gp, oracle, stop_rule=dict(sr, eos_token_ids=[3]))
    run("eos, LAG only in the stop rule", gp, oracle, lag=None)
    run("eos, LAG only as eos_lag_rows", gp, oracle, lag=None, stop_rule={k: v for k, v in sr.items() if k != "lag"})
    run("eos, no LAG declared", gp, oracle, lag=None, stop_rule={"eos_token_ids": [T.EOS]})
    run("eos, a lag block with no source", gp, oracle, lag={"value": 0})
    run("eos, no LAG source anywhere", gp, oracle, lag={"value": 0}, stop_rule={"eos_token_ids": [T.EOS], "lag": 0, "eos_lag_rows": 0})

    gp, oracle = T.assemble([T.R0, T.R1_LAG], tmp_path / "lag", lag=1)
    T._set_async_scheduling(oracle, True)
    T._driver_stop_at(oracle, "r1", n_answer=2, eos_step=1)
    run("lag 1, as the mock writes it", gp, oracle)
    run("lag 1, as GP-01 writes it", gp, oracle, **_gp01(1))
    run("lag 1, LAG 0 declared", gp, oracle, **_gp01(0))
    run("lag 1, no LAG declared", gp, oracle, lag=None, stop_rule={"eos_token_ids": [T.EOS]})
    return out


_POP = object()


def test_g6_reads_the_record_to_the_same_document(memory, tmp_path):
    """Every outcome of the global match over `_corpus` (verdict, G6's check, the per-request account, the lag finding) is what it was when G6
    read the parsed JSON."""
    got = _corpus(tmp_path)
    assert got["cap, as the mock writes it"]["verdict"] == "PASS" and got["cap, as GP-01 writes it"]["verdict"] == "PASS"
    blob = json.dumps(got, sort_keys=True, default=str).replace(str(tmp_path), "<run>")
    assert hashlib.sha256(blob.encode()).hexdigest() == OUTCOMES_SHA256, {k: (v["verdict"], v["G6"]["problems"][:2]) for k, v in got.items()}


@pytest.mark.parametrize("block,value,field,why", [
    ("stop_rule", {"ignore_eos": False}, "stop_rule.eos_token_ids", "is missing: every writer of the record writes it"),
    ("stop_rule", "eos", "stop_rule", "is a str, not a Mapping"),
    ("lag", {"source": "test"}, "lag.value", "is missing: every writer of the record writes it"),
    ("lag", {"value": "0", "source": "test"}, "lag.value", "is a str, not a int")])
def test_g6_fails_a_refused_declaration_by_name(memory, tmp_path, block, value, field, why):
    gp, oracle = T.assemble([T.R0, T.R1_EOS], tmp_path)
    gp[block] = value
    res = gm.check(gp, oracle, run_x09=False, record_alternate=False)
    g6 = T._check(res, "G6")
    assert res["verdict"] == "FAIL" and "G6" in res["failed_checks"]
    want = (f"the Program's declared stop rule and LAG are refused, so G6 holds the record to no EOS set and no LAG: the global "
            f"Program: field {field!r} {why}")
    assert want in g6["problems"], g6["problems"]
    assert g6["stop_rule"]["eos_token_ids"] is None and g6["lag"]["declared"] is None
    assert any(p.startswith("r1: stopped after 2 < max_tokens 4 events but the Program declares no EOS set") for p in g6["problems"]), g6["problems"]
