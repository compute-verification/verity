"""`members_standin`'s window: the templates its members are spread over, without the audit (`members_standin.sh`
runs that on a node)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import members_standin as MS


def _templates(d: Path) -> list[int]:
    """Each template's units, in the program's order, as the window's ``circuit.json`` states them: the registration
    holds as many units, a stratum a template."""
    units = [n for _id, n in json.loads((d / "circuit.json").read_text())["templates"]]
    registered = json.loads((d / "registration.json").read_text())
    assert registered["population"] == sum(units)
    assert [sum(hi - lo for lo, hi in s["units"]) for s in registered["law"]["strata"]] == units
    return units


def test_two_templates_are_the_default_and_each_further_one_is_a_new_k():
    assert MS.ks_of(2) == MS.KS == (128, 256)
    ks = MS.ks_of(40)
    assert ks[:2] == MS.KS and len(set(ks)) == 40 and all(k % 32 == 0 and 128 <= k < 2 ** 17 for k in ks)
    assert MS.ks_of(1) == (128,) and len(MS.ks_of(4092)) == 4092
    for refused in (0, 4093):
        with pytest.raises(ValueError, match="templates"):
            MS.ks_of(refused)


def test_a_window_holds_the_members_asked_for_over_the_templates_asked_for(tmp_path: Path):
    """The member count stays as the templates grow, and the default window is two templates of half the units each,
    written the same whether or not the templates are named."""
    assert _templates(MS.window(tmp_path / "default", 12)) == [6, 6]
    assert _templates(MS.window(tmp_path / "two", 12, MS.ks_of(2))) == [6, 6]
    assert (tmp_path / "default" / "circuit.json").read_bytes() == (tmp_path / "two" / "circuit.json").read_bytes()
    assert _templates(MS.window(tmp_path / "three", 12, MS.ks_of(3))) == [4, 4, 4]
    assert _templates(MS.window(tmp_path / "five", 12, MS.ks_of(5))) == [3, 3, 2, 2, 2]
    with pytest.raises(ValueError, match="give at least 3"):
        MS.window(tmp_path / "none", 2, MS.ks_of(3))


def test_the_audit_gets_a_port_for_each_template(tmp_path: Path):
    win = MS.window(tmp_path / "w", 6, MS.ks_of(3))
    args = MS.audit_argv(win, tmp_path / "audit", tmp_path / "main.txt", tmp_path / "flock", tmp_path / "lean", 3)
    assert args[args.index("--ports") + 1] == "3"
    default = MS.audit_argv(win, tmp_path / "audit", tmp_path / "main.txt", tmp_path / "flock", tmp_path / "lean")
    assert default[default.index("--ports") + 1] == "2"
