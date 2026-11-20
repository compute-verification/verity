"""PoUW's benchmarks' settings registry (benchmarks/pouw/settings.py): it loads by path, refuses an unknown Pearl-C
variable, and every read in the files that run outside the tree (`STANDALONE`) is its declaration's."""
from __future__ import annotations

from pathlib import Path

import pytest
from verity_config.settings import UnknownSetting, of

POUW = (Path(__file__).resolve().parents[3] / "benchmarks" / "pouw")


def test_loads_by_path_and_holds_the_runners_to_their_declarations():
    reg = of(POUW / "pearl_c_vllm" / "bench_arm.py")
    assert reg is of(POUW / "harness" / "verify.py") and Path(reg.path) == POUW / "settings.py"
    assert reg.prefixes == ("PEARLC_",)
    assert reg.standalone_problems() == []


def test_refuses_an_unknown_pearl_c_variable():
    reg = of(POUW / "settings.py")
    with pytest.raises(UnknownSetting, match="PEARLC_SPLITT"):
        reg.validate_environ({"PEARLC_SPLITT": "0", "PEARLC_SPLIT": "0", "POUW_ANYTHING": "1"})
