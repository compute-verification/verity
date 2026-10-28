"""Dither's settings registry (settings.py): it loads by path, refuses an unknown DITHER_, SBENCH_ or HB_ variable, and
every read in the files that run outside the tree (`STANDALONE`) is its declaration's."""
from __future__ import annotations

from pathlib import Path

import pytest
from verity_config.settings import UnknownSetting, of

DITHER = Path(__file__).resolve().parents[1]


def test_loads_by_path_and_holds_the_standalone_files_to_their_declarations():
    reg = of(DITHER / "sbench.py")
    assert reg is of(DITHER / "dither_so.py") and Path(reg.path) == DITHER / "settings.py"
    assert reg.prefixes == ("DITHER_", "SBENCH_", "HB_")
    assert reg.standalone_problems() == []


def test_parses_as_the_readers_did(monkeypatch):
    reg = of(DITHER / "settings.py")
    for name in ("DITHER_SO_GRAPH", "DITHER_A_LVF", "SBENCH_B", "SBENCH_SMQ_SWEEP", "DITHER_SO_TAIL"):
        monkeypatch.delenv(name, raising=False)
    assert reg.DITHER_SO_GRAPH is True and reg.DITHER_A_LVF is None and reg.SBENCH_B == 32
    assert reg.SBENCH_SMQ_SWEEP == () and reg.DITHER_SO_TAIL == "0"
    monkeypatch.setenv("DITHER_SO_GRAPH", "0")
    monkeypatch.setenv("DITHER_A_LVF", "")
    monkeypatch.setenv("SBENCH_SMQ_SWEEP", "47/64,,45/64")
    assert reg.DITHER_SO_GRAPH is False and reg.DITHER_A_LVF == "" and reg.SBENCH_SMQ_SWEEP == ("47/64", "45/64")
    monkeypatch.setenv("SBENCH_ARM_ENV", '{"dither-a-so4": {"DITHER_SO_FUSED": "1"}}')
    assert reg.SBENCH_ARM_ENV == {"dither-a-so4": {"DITHER_SO_FUSED": "1"}}
    with pytest.raises(KeyError, match="DITHER_H1C is not set"):
        reg.need("DITHER_H1C", {})


def test_refuses_an_unknown_dither_variable():
    reg = of(DITHER / "settings.py")
    with pytest.raises(UnknownSetting, match="DITHER_SO_FUSD"):
        reg.validate_environ({"DITHER_SO_FUSD": "1", "DITHER_SO_FUSED": "1", "H1U_JRC": "1"})
    assert reg.unknown({"SBENCH_B": "8", "HB_H1UU": "x"}) == ["HB_H1UU"]
