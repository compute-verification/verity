"""Dither's settings registry (settings.py): it loads by path, refuses an unknown DITHER_, SBENCH_ or HB_ variable, and
every read in the files that run outside the tree (`STANDALONE`) is its declaration's."""
from __future__ import annotations

from pathlib import Path

import pytest
from verity_config.settings import UnknownSetting, of

DITHER = (Path(__file__).resolve().parents[4] / "benchmarks" / "hash" / "dither")


def test_loads_by_path_and_holds_the_standalone_files_to_their_declarations() -> None:
    reg = of(DITHER / "dbw.py")
    assert reg is of(DITHER / "forms.py") and Path(reg.path) == DITHER / "settings.py"
    assert reg.prefixes == ("DITHER_", "SBENCH_", "HB_")
    assert reg.standalone_problems() == []


def test_refuses_an_unknown_dither_variable() -> None:
    reg = of(DITHER / "settings.py")
    with pytest.raises(UnknownSetting, match="DITHER_SO_FUSD"):
        reg.validate_environ({"DITHER_SO_FUSD": "1", "DITHER_SO_FUSED": "1", "H1U_JRC": "1"})
    assert reg.unknown({"SBENCH_B": "8", "HB_H1UU": "x"}) == ["HB_H1UU"]
