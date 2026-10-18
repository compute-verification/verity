"""kappa_count.py on the censuses in the tree: every kind each device's current epoch holds has a destination width, and
the creation caps κ = 65·d that `ClassPSlots` charges come out as the ISA's widths give them."""
from __future__ import annotations

import importlib.util
from fractions import Fraction
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "kappa_count.py"
spec = importlib.util.spec_from_file_location("kappa_count", SCRIPT)
kc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kc)


def test_every_device_classified(tmp_path):
    """The script runs over every priced device and writes its JSON; a census kind without a width would stop it."""
    assert kc.main(["--json", str(tmp_path / "kappa.json")]) == 0
    assert (tmp_path / "kappa.json").read_text(encoding="utf-8").startswith("[")


def test_sm120():
    r = kc.count("sm_120")
    assert r["kinds"]["LDG256"]["kappa"] == 520
    assert r["kinds"]["IMADW"]["kappa"] == 130
    assert r["kinds"]["IMMA16832"]["kappa"] is None
    assert (r["kappa_vector"], r["kappa_lane"], r["kappa_table"]) == (65, 130, 520)


def test_sm89_sm90():
    a, h = kc.count("sm_89"), kc.count("sm_90")
    assert (a["kappa_vector"], a["kappa_table"]) == (65, 260)
    assert (h["kappa_vector"], h["kappa_table"]) == (65, None)


def test_floor_and_refusal():
    assert kc.floor(130, 8) == 1 + Fraction(24, 130)
    assert kc.floor(128, 4) == Fraction(39, 32)
    assert kc.classify("LDSM8T_X2") == ("load", 4) and kc.classify("LDSM_X4T") == ("load", 4)
    with pytest.raises(SystemExit):
        kc.classify("NEWOP")
