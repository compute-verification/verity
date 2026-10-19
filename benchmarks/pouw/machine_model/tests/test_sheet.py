"""`machine-model sheet`: a device's sheet.toml derives its DS census line for line, each per-clock peak reproduces the
figure its document prints, and the Lean module it names is what `sheet --lean` prints."""
from __future__ import annotations

import shutil

import pytest

from verity_machine_model import eps as E
from verity_machine_model.cli import main
from verity_machine_model.model import DEVICES
from verity_machine_model.sheet import ROOT, differences, figures, lean, moves, reading, sheet, uni

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
    assert r.fmax == r.sheet, (f"{name}: the sheet's ε now depends on reading A3 per clock (the H200's tensor clock, "
                               "the B200's measured peak: B200-PEAK-PER-CLOCK) rather than per second at f_max")


@pytest.mark.parametrize("name", SHEETS)
def test_tensor_lines_have_slack(name):
    """The headline's sheet ε doesn't move until every tensor line is 10% slower than the sheet's, so an A3 error under
    10% leaves it as it is. On DS-90 that slack is the li fold's margin: the cheater's 145 SMSP-cycles a MMA on
    DS_issue against DS_tensor's 128 (RT-HOPPER-2)."""
    s = sheet(DEVICES / name)
    m = moves(s, DEVICES / name, steps=10)
    assert m is None or m >= 1.10, (f"{name}: ε moves with every tensor line {m:.3f}× the sheet's, so A3's error now "
                                    "moves the headline (RT-HOPPER-2)")


@pytest.mark.parametrize("name", SHEETS)
def test_uni_issue_leaves_a_ds_vector(name):
    """A DS vector charges a uniform instruction its issue slot already (`UNI,DS_issue`, A2u), so `eps --uni-issue`
    leaves it as it is, its name too: a renamed vector loses its ε_skip `at` entry (RT-HOPPER-4)."""
    d = DEVICES / name
    pi = E.machine(sheet(d).vector, d)
    assert E.uniform_issue_charged(pi) and E.uniform_on_issue(pi, 0.5) is pi


@pytest.mark.parametrize("name", SHEETS)
def test_uni_prices_a2u_on_the_sheet(name):
    """`sheet --eps --uni` changes the uniform instruction's line, where `eps --uni-issue` leaves a DS vector as it is:
    at the sheet's own cost it is the headline, and with the issue slot free the cheater takes the uniform route as far
    as DS_uniform, the measured UP line, lets it (RT-MACHINE-4: without that row, 94% on the H200, 100% on the B200)."""
    s, d = sheet(DEVICES / name), DEVICES / name
    r = reading(s, d)
    assert uni(s, d, 1.0) == r.sheet
    free = uni(s, d, 0.0)
    assert r.sheet < free < 0.02, f"{name}: with a uniform instruction's issue slot free, ε is {free:.4%}"


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
