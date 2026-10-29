"""PoUW's benchmarks' settings registry (benchmarks/pouw/settings.py): it loads by path, refuses an unknown Pearl-C
variable, and every read in the files that run outside the tree (`STANDALONE`) is its declaration's."""
from __future__ import annotations

from pathlib import Path

import pytest
from verity_config.settings import UnknownSetting, of

POUW = Path(__file__).resolve().parents[1]


def test_loads_by_path_and_holds_the_runners_to_their_declarations():
    reg = of(POUW / "pearl_c_vllm" / "bench_arm.py")
    assert reg is of(POUW / "harness" / "verify.py") and Path(reg.path) == POUW / "settings.py"
    assert reg.prefixes == ("PEARLC_", "PEARLC4_", "PEARL_C4_")
    assert reg.standalone_problems() == []


def test_parses_as_the_readers_did(monkeypatch):
    reg = of(POUW / "settings.py")
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


def test_refuses_an_unknown_pearl_c_variable():
    reg = of(POUW / "settings.py")
    with pytest.raises(UnknownSetting, match="PEARLC_SPLITT"):
        reg.validate_environ({"PEARLC_SPLITT": "0", "PEARLC_SPLIT": "0", "POUW_ANYTHING": "1"})
