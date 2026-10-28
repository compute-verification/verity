"""`machine-model sheet`: a device's sheet.toml derives its DS census line for line, each per-clock peak reproduces the
figure its document prints, and the Lean module it names is what `sheet --lean` prints."""
from __future__ import annotations

import shutil

import pytest

from verity_machine_model import eps as E
from verity_machine_model.cli import main
from verity_machine_model.model import DEVICES
from verity_machine_model.sheet import ROOT, differences, figures, lean, moves, omega, reading, sheet, uni

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


def test_a_clock_above_fmax_is_refused(tmp_path):
    d = tmp_path / "sm_90"
    shutil.copytree(DEVICES / "sm_90", d)
    p = d / "sheet.toml"
    p.write_text(p.read_text().replace("mhz = 1830", "mhz = 1990"))
    with pytest.raises(SystemExit, match=r"\[clocks.tensor\] is 1990 MHz, above f_max's 1980"):
        sheet(d)


@pytest.mark.parametrize("name", SHEETS)
def test_omega_is_a6s(name):
    """Each tensor class's ω is its accumulator fragment's words, as A6 (`SheetWords`) lists them: mma.sync m16n8k32
    s32 128, wgmma m64n256k32 4096 a warp, tcgen05.mma 128 × 256 32768, and DMMA's two registers an f64 element, 128
    at m8n8k4 and 256 at m16n8k16."""
    w = omega(sheet(DEVICES / name))
    a6 = {"IMMA16832_S8": 128, "DMMA884": 128, "DMMA16816": 256,
          **({"IGMMA_S8_N256": 4096} if name == "sm_90" else {"UTCIMMA_M128_N256": 32768})}
    assert {k: w[k] for k in a6} == a6


@pytest.mark.parametrize("name", SHEETS)
def test_sheet_prices_the_headline(name):
    """The headline's sheet reading of the device is the sheet's: its design priced on lines derived from sheet.toml
    is headline.py's ε, exactly. A3 is read per clock at the clock each figure is quoted at (`SheetTensor`); the
    reading per second at f_max is a bound `sheet --eps` prints beside it, which the headline doesn't rest on."""
    r = reading(sheet(DEVICES / name), DEVICES / name)
    assert r.sheet == r.headline


@pytest.mark.parametrize("name", SHEETS)
def test_tensor_lines_have_slack(name):
    """The headline's sheet ε doesn't move until every tensor line is 2% slower than the sheet's, so an A3 error under
    2% leaves it as it is: the H200's tensor lines are measured at 0.01% of the sheet's, and the B200's per-clock peaks
    are its measured rates. On DS-90 that slack is the fold's margin (FOLD-OPT)."""
    s = sheet(DEVICES / name)
    m = moves(s, DEVICES / name, steps=10)
    assert m is None or m >= 1.02, (f"{name}: ε moves with every tensor line {m:.3f}× the sheet's, inside A3's "
                                    "measured error and its margin, so A3's error now moves the headline")


@pytest.mark.parametrize("name", SHEETS)
def test_uni_issue_leaves_a_ds_vector(name):
    """A DS vector charges a uniform instruction its issue slot already (`UNI,DS_issue`, A2u), so `eps --uni-issue`
    leaves it as it is, its name too: a renamed vector loses its ε_skip `at` entry."""
    d = DEVICES / name
    pi = E.machine(sheet(d).vector, d)
    assert E.uniform_issue_charged(pi) and E.uniform_on_issue(pi, 0.5) is pi


@pytest.mark.parametrize("name", SHEETS)
def test_uni_prices_a2u_on_the_sheet(name):
    """`sheet --eps --uni` changes the uniform instruction's line, where `eps --uni-issue` leaves a DS vector as it is:
    at the sheet's own cost it is the headline, and with the issue slot free the cheater takes the uniform route as far
    as DS_uniform, the measured UP line, lets it: about 1.15 points over the sheet's ε on each (2.007% on the H200,
    1.844% on the B200; without that row, 94% on the H200, 100% on the B200)."""
    s, d = sheet(DEVICES / name), DEVICES / name
    r = reading(s, d)
    assert uni(s, d, 1.0) == r.sheet
    free = uni(s, d, 0.0)
    assert r.sheet < free < r.sheet + 0.012, f"{name}: with a uniform instruction's issue slot free, ε is {free:.4%}"


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
