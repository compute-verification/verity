"""`machine-model headline`: each family's row at both readings, from the design files and vectors."""

from __future__ import annotations

import json

import pytest
from verity_machine_model import headline as H
from verity_machine_model.cli import main
from verity_machine_model.eps import design
from verity_machine_model.model import DATA, device

# (ε_skip, ε_machine, ε_luck, ε, γ, the floor it needs, ε at p = 1): credit ♭, operands free, ε_mand 0 everywhere
PINS = {
    "sheet": {
        "sm_120": (0.004207, 0.001506, 0.000139, 0.005852, 2.2831, "half", None),
        "sm_90": (0.006428, 0.0, 0.000037, 0.006465, 1.7698, "half", 0.006850),
        "sm_100": (0.004063, 0.000631, 0.000026, 0.004720, 4.1719, "half", None),
        "sm_89": (0.001463, 0.006652, 0.000336, 0.008451, 2.9795, "half", None),
        "sm_80": (0.001463, 0.0, 0.0, 0.001463, 5.9347, "half", None),
    },
    "measured": {
        "sm_120": (0.003628, 0.001251, 0.000517, 0.005396, 2.1357, "issue", None),
        "sm_90": (0.004661, 0.000757, 0.000730, 0.006148, 1.7171, "issue", 0.011618),
        "sm_100": (0.004687, 0.004308, 0.000001, 0.008996, 4.0704, "half", None),
        "sm_89": (0.001470, 0.001126, 0.000468, 0.003065, 3.0144, "issue", None),
        "sm_80": (0.001470, 0.0, 0.0, 0.001470, 4.2263, "half", None),
    },
}


@pytest.mark.parametrize("reading", H.READINGS)
def test_each_familys_row_is_pinned(reading):
    """The headline every note, the canvas and Notion embed: a change to a design, a vector or the LP that moves a
    family's row shows here, with the row it moved."""
    rows = {x.family: x for x in H.rows(reading)}
    assert list(rows) == list(PINS[reading])
    for fam, (skip, machine, luck, core, gamma, floor, p1) in PINS[reading].items():
        x = rows[fam]
        got = (x.skip, x.machine, x.luck, x.core)
        assert got == pytest.approx((skip, machine, luck, core), abs=2e-6), (fam, got)
        assert x.mand == 0 and x.core == pytest.approx(x.skip + x.mand + x.machine + x.luck, abs=1e-12), fam
        assert x.gamma == pytest.approx(gamma, abs=2e-4) and x.floor == floor, fam
        assert (x.p1["core"] if x.p1 else None) == (None if p1 is None else pytest.approx(p1, abs=2e-6)), fam
        assert x.core < 0.01 and x.q == H.spec()["reading"][reading]["q"], fam


def test_conditions_name_the_floor_the_fillers_credit_and_the_p1_reading():
    """Decision 89's reading needs the 1.22-slot issue floor where ε_machine is above its floor at q* = ½, and states it
    as unproved in Lean; a filler's credit is the theorem its check meets (decision 99), and the H200's p = 1 reading
    stands beside it."""
    t = H.spec()
    m = {x.family: x for x in H.rows("measured")}
    assert m["sm_120"].conditions[0] == t["floor"]["issue"] and "Theorem C, unproved in Lean" in t["floor"]["issue"]
    assert m["sm_100"].conditions[0] == t["floor"]["half"] and m["sm_100"].threshold <= 0.5
    h = m["sm_90"]
    assert h.fillers == ("0.07 FFMA a word credited at p = 0 (Accounting.Guarantees.FillerLuck.PinnedFillBlocksDeficit)",)
    assert "at p = 1 (dither-2s-licpw-redux-p1-l32): ε 1.162%" in h.conditions
    assert any("FFMA filler check pending" in c for c in h.conditions)
    s = {x.family: x for x in H.rows("sheet")}
    assert all(x.floor == "half" for x in s.values()) and any(c.startswith("A2u") for c in s["sm_89"].conditions)


def test_cli(capsys):
    assert main(["headline", "--reading", "measured"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "## decision 89's measured reading (`measured`): q* = 1.22, operands free, credit ♭"
    head = next(x for x in out if x.startswith("| GPU"))
    assert all(H.TERMS[k] in head for k in ("skip", "mand", "machine", "luck", "gamma", "threshold"))
    assert any(x.startswith("| H200 | dither-2s-licpw-redux-l32 | DC-90-ri | 0.466% |") for x in out)
    assert any(x.startswith("- H200: the 1.22-slot issue floor") for x in out)
    assert main(["headline", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["terms"] == H.TERMS and list(d["readings"]) == list(H.READINGS)
    assert d["credit"] == "floor" and d["operands"] == "free" and "decision 100" in d["capacity"]
    r = d["readings"]["sheet"]["rows"][0]
    assert {k for k in H.TERMS} <= set(r) and r["vector"] == "DS-120" and r["core"] == pytest.approx(0.005852, abs=2e-6)


def test_an_unrecorded_skip_is_refused():
    """♭ moves with Π, so a reading on a vector whose ε_skip the design hasn't recorded is refused, with the fix."""
    d = design("dither-c-peel-fx1")
    assert H.skip_at(d, "VF3r-16", DATA) == d.skip["served"].eps
    assert H.skip_at(d, "DS-120", DATA) == d.skip["served"].at["DS-120"]
    with pytest.raises(SystemExit, match=r"no ε_skip at VF3q-16 .*at = \{ \"VF3q-16\" = … \}"):
        H.skip_at(d, "VF3q-16", DATA)


def test_a_p1_sibling_is_its_design_with_every_filler_at_p_1():
    sm90 = device("sm_90")
    d, p1 = design("dither-2s-licpw-redux-l32", sm90), design("dither-2s-licpw-redux-p1-l32", sm90)
    H._sibling(d, p1)
    assert [f.p for f in p1.filler] == [1.0] and p1.skip["served"].eps > d.skip["served"].eps   # ♭ drops the filler
    with pytest.raises(SystemExit, match="every filler at p = 1"):
        H._sibling(d, design("dither-2s-li-l32", sm90))
