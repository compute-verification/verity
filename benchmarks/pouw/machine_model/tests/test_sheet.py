"""`machine-model sheet`: a device's sheet.toml derives its DS census line for line, each per-clock peak reproduces the
figure its document prints, and the Lean module it names is what `sheet --lean` prints."""
from __future__ import annotations

import shutil

import pytest

from verity_machine_model.cli import main
from verity_machine_model.model import DEVICES
from verity_machine_model.sheet import ROOT, differences, figures, lean, reading, sheet

SHEETS = ["sm_90", "sm_100"]


@pytest.mark.parametrize("name", SHEETS)
def test_sheet_is_the_census(name):
    d = DEVICES / name
    assert differences(sheet(d), d) == []


@pytest.mark.parametrize("name", SHEETS)
def test_peaks_reproduce_the_figures(name):
    s = sheet(DEVICES / name)
    failed = [(f.type, float(f.value), float(f.dense), float(f.tol)) for f in figures(s) if not f.holds]
    assert not failed, f"{name}: a per-clock peak at its clock misses the document's figure: {failed}"
    assert all(c.value < s.fmax.value for c in s.clocks.values())


@pytest.mark.parametrize("name", SHEETS)
def test_sheet_prices_the_headline(name):
    """The headline's sheet reading of the device is the sheet's: its design priced on lines derived from sheet.toml
    is headline.py's ε, exactly."""
    r = reading(sheet(DEVICES / name), DEVICES / name)
    assert r.sheet == r.headline


@pytest.mark.parametrize("name", SHEETS)
def test_lean_is_generated(name):
    d = DEVICES / name
    s = sheet(d)
    path = ROOT / s.lean
    assert path.read_text() == lean(s, d), (f"{s.lean} isn't sheet.toml's: run `uv run machine-model --device {name} "
                                            f"sheet --lean > {s.lean}`")


def test_sheet_command_flags_a_changed_line(tmp_path, capsys):
    d = tmp_path / "sm_90"
    shutil.copytree(DEVICES / "sm_90", d)
    p = d / "census" / "DS-90.csv"
    p.write_text(p.read_text().replace("HGMMA_BF16_N128,DS_tensor,,,64,", "HGMMA_BF16_N128,DS_tensor,,,32,"))
    assert main(["--data", str(d), "sheet"]) == 1
    assert "HGMMA_BF16_N128 DS_tensor: census 32, sheet 64" in capsys.readouterr().out
