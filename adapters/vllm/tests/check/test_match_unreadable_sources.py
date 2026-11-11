"""A Match record's execution source that is present but unreadable fails G1 by name; it is never read as "not attested" and passed.

`record_engine_facts` names an unreadable `capture/versions.json` under `unreadable` (absent when every location reads, so a readable
record's facts are what they were), and G1 fails with the error.  The verdict's `--engine-arg` values parse as JSON, else stay strings."""
from __future__ import annotations

import json
from pathlib import Path

from tests.check.test_global_match import R0, R1, _check, assemble
from tests.observe.test_fold_synthetic import memory  # noqa: F401 - `memory` is the fixture
from verity_vllm.check import verdict as V
from verity_vllm.check.match import declaration as GD
from verity_vllm.check.match import global_match as gm


def test_an_unreadable_versions_json_is_named(tmp_path):
    header = {"engine_kwargs": {"enforce_eager": True, "enable_prefix_caching": False}}
    assert "unreadable" not in GD.record_engine_facts(header, str(tmp_path))
    (tmp_path / "capture").mkdir()
    (tmp_path / "capture" / "versions.json").write_text(json.dumps({"engine_kwargs": {"enforce_eager": True}}))
    ok = GD.record_engine_facts(header, str(tmp_path))
    assert "unreadable" not in ok and ok["sources"]["enforce_eager"] == ["RunHeader.engine_kwargs", "capture/versions.json engine_kwargs"]
    (tmp_path / "capture" / "versions.json").write_text('{"engine_kwargs": {"enforce_eager": tr')
    bad = GD.record_engine_facts(header, str(tmp_path))
    assert list(bad["unreadable"]) == ["capture/versions.json"] and "JSONDecodeError" in bad["unreadable"]["capture/versions.json"]
    assert bad["engine"] == {"enforce_eager": True, "enable_prefix_caching": False}


def test_g1_fails_by_name_on_an_unreadable_versions_json(memory, tmp_path):
    gp, oracle = assemble([R0, R1], tmp_path)
    assert gm.check(gp, oracle)["verdict"] == "PASS"
    (Path(oracle) / "capture" / "versions.json").write_text("not json")
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and "G1" in res["failed_checks"], res["reasons"]
    assert any("capture/versions.json is present but unreadable (JSONDecodeError" in p for p in _check(res, "G1")["problems"])


def test_engine_args_parse_as_json_else_stay_strings(tmp_path):
    (tmp_path / "commit.log").write_text("$ python commit_delta.py --engine-arg enforce_eager=true --engine-arg kv_cache_dtype=fp8 "
                                         "--engine-arg max_model_len=4096 --engine-arg cfg=[1,2\n")
    es = V.execution_of_record(tmp_path)["engine_settings"]
    assert {k: es[k] for k in ("enforce_eager", "kv_cache_dtype", "max_model_len", "cfg")} == {
        "enforce_eager": True, "kv_cache_dtype": "fp8", "max_model_len": 4096, "cfg": "[1,2"}
