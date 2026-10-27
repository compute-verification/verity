"""PoUW's benchmarks' settings registry (benchmarks/pouw/settings.py): it loads by path, refuses an unknown Pearl-C or
``VERITY_POUW_*`` variable, every read in the files that run outside the tree (`STANDALONE`) is its declaration's, each
setting is declared whole, and a run's record carries its value settings and refuses a debug one."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from verity_config.settings import DebugOnRecord, UnknownSetting, of

POUW = Path(__file__).resolve().parents[1]
REG = of(POUW / "settings.py")


def test_loads_by_path_and_holds_the_runners_to_their_declarations():
    reg = of(POUW / "pearl_c_vllm" / "bench_arm.py")
    assert reg is of(POUW / "harness" / "verify.py") is of(POUW / "served_zk" / "rows_bench.py") is REG
    assert Path(reg.path) == POUW / "settings.py"
    assert reg.prefixes == ("PEARLC_", "PEARLC4_", "PEARL_C4_", "VERITY_POUW_")
    assert reg.standalone_problems() == []


def test_parses_as_the_readers_did(monkeypatch):
    reg = REG
    for name in ("PEARLC_SPLIT", "PEARLC_WHOLE_DEFER_A", "PEARLC_CAPTURE_SIZES", "SHAPES", "VARIANTS"):
        monkeypatch.delenv(name, raising=False)
    assert reg.PEARLC_SPLIT is True and reg.PEARLC_WHOLE_DEFER_A is True and reg.PEARLC_CAPTURE_SIZES is None
    assert reg.SHAPES == ("8192",) and reg.VARIANTS is None
    monkeypatch.setenv("PEARLC_SPLIT", "0")
    monkeypatch.setenv("PEARLC_CAPTURE_SIZES", "1,8")
    monkeypatch.setenv("SHAPES", "")
    monkeypatch.setenv("VARIANTS", "sm120")
    assert reg.PEARLC_SPLIT is False and reg.PEARLC_CAPTURE_SIZES == (1, 8)
    assert reg.SHAPES == () and reg.VARIANTS == ("sm120",)
    monkeypatch.setenv("PEARLC_GROUPED_PLACE", "base+tiles4")
    assert reg.PEARLC_GROUPED_PLACE == frozenset({"tiles4"}) and reg.debug_set() == ["PEARLC_GROUPED_PLACE"]


def test_refuses_an_unknown_pearl_c_or_pouw_variable():
    with pytest.raises(UnknownSetting, match="PEARLC_SPLITT"):
        REG.validate_environ({"PEARLC_SPLITT": "0", "PEARLC_SPLIT": "0", "POUW_ANYTHING": "1"})
    assert REG.unknown(["VERITY_POUW_H1_PIN", "VERITY_POUW_V31_LIB", "PATH"]) == []
    with pytest.raises(UnknownSetting, match="VERITY_POUW_TYPO"):
        REG.validate_environ(["VERITY_POUW_TYPO"])


@pytest.mark.parametrize("name", sorted(REG.own))
def test_each_setting_is_declared_whole(name):
    s = REG.own[name]
    assert s.name == name and s.doc.strip()
    assert s.default is None or not isinstance(s.type, type) or isinstance(s.default, s.type)


def test_a_read_parses_and_an_unset_or_empty_variable_is_its_default():
    assert REG.get("RASTER", {}) == 8
    assert REG.get("RASTER", {"RASTER": ""}) == 8
    assert REG.get("RASTER", {"RASTER": "4"}) == 4
    assert REG.get("REPS", {}) is None
    with pytest.raises(ValueError, match="RASTER='four'"):
        REG.get("RASTER", {"RASTER": "four"})


def test_an_undeclared_name_is_refused():
    with pytest.raises(AttributeError, match="NOT_A_SETTING"):
        REG.get("NOT_A_SETTING", {})
    with pytest.raises(AttributeError):
        REG.NOT_A_SETTING  # noqa: B018
    with pytest.raises(AttributeError):
        REG.put("NOT_A_SETTING", "1")


def test_the_record_carries_the_value_settings_with_their_digest():
    rec = REG.of_record({"REPS": "5", "SHAPE": "8192", "RESEARCH_RUN_ID": "r1", "UNRELATED": "x"})
    assert rec["values"] == {"REPS": "5", "SHAPE": "8192"}
    assert rec["digest"] == hashlib.sha512(json.dumps(rec["values"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_the_record_of_a_run_refuses_a_debug_setting():
    with pytest.raises(DebugOnRecord, match="STAMPS"):
        REG.of_record({"STAMPS": "1"})
    assert REG.of_record({"STAMPS": ""})["values"] == {}
