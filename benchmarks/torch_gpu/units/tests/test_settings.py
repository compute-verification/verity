"""The units' settings registry (settings.py): build.sh ships it beside every module that reads it, where it loads with no
pyproject.toml above it, and it refuses an unknown UNITS_ variable."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from verity_config.settings import UnknownSetting, of

UNITS = Path(__file__).resolve().parents[1]


def shipped() -> list[str]:
    m = re.search(r'cp "\$HERE"/\{([a-z_,]+)\}\.py "\$OUT/"', (UNITS / "build.sh").read_text())
    assert m, "build.sh copies the modules into the ship with one cp"
    return m.group(1).split(",")


def test_the_ship_carries_the_registry_of_every_module_that_reads_it():
    names = shipped()
    readers = [n for n in names if "of(__file__)" in (UNITS / f"{n}.py").read_text()]
    assert {"check", "bench", "lib"} <= set(readers) and "settings" in names


def test_a_shipped_module_finds_the_registry_beside_it(tmp_path, monkeypatch):
    ship = tmp_path / "ship"
    ship.mkdir()
    shutil.copy(UNITS / "settings.py", ship)
    reg = of(ship / "check.py")
    assert Path(reg.path) == ship / "settings.py" and reg is not of(UNITS / "check.py")
    monkeypatch.delenv("BATCHES", raising=False)
    monkeypatch.setenv("TIME_B", "8,32")
    assert reg.BATCHES == (1, 7, 128, 4096, 16384, 20480) and reg.TIME_B == (8, 32) and reg.UNITS_LIBCUDA == "libcuda.so.1"


def test_refuses_an_unknown_units_variable():
    reg = of(UNITS / "settings.py")
    with pytest.raises(UnknownSetting, match="UNITS_CUBN"):
        reg.validate_environ({"UNITS_CUBN": "x", "UNITS_CUBIN": "x", "SWEEP": "0"})
    assert reg.debug_set({"UNITS_ANY_GPU": "1"}) == ["UNITS_ANY_GPU"]
