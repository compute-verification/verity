"""attn_ref's settings registry (settings.py): build.sh ships it beside the modules that read it, where it loads with no
pyproject.toml above it, and it refuses an unknown ATTN_REF_ variable."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from verity_config.settings import UnknownSetting, of

ATTN_REF = Path(__file__).resolve().parents[1]


def test_the_ship_carries_the_registry_of_every_module_that_reads_it():
    m = re.search(r'^cp ((?:"\$HERE/[a-z_]+\.py" )+)"\$OUT/"$', (ATTN_REF / "build.sh").read_text(), re.M)
    assert m, "build.sh copies the modules into the ship with one cp"
    names = re.findall(r"([a-z_]+)\.py", m.group(1))
    readers = [n for n in names if "of(__file__)" in (ATTN_REF / f"{n}.py").read_text()]
    assert set(readers) == {"attn_ref", "check"} and "settings" in names


def test_a_shipped_module_finds_the_registry_beside_it(tmp_path, monkeypatch):
    ship = tmp_path / "ship"
    ship.mkdir()
    shutil.copy(ATTN_REF / "settings.py", ship)
    reg = of(ship / "check.py")
    assert Path(reg.path) == ship / "settings.py" and reg is not of(ATTN_REF / "check.py")
    monkeypatch.delenv("BATCHES", raising=False)
    monkeypatch.setenv("TIME_B", "4096")
    assert reg.BATCHES == (1, 7, 128, 20480) and reg.TIME_B == 4096 and reg.ATTN_EXPECT == "NVIDIA H200"


def test_refuses_an_unknown_attn_ref_variable():
    reg = of(ATTN_REF / "settings.py")
    with pytest.raises(UnknownSetting, match="ATTN_REF_CUBN"):
        reg.validate_environ({"ATTN_REF_CUBN": "x", "ATTN_REF_CUBIN": "x", "ATTN_BF16_CUBIN": "x"})
    assert reg.debug_set({"ATTN_ANY_GPU": "1"}) == ["ATTN_ANY_GPU"]
