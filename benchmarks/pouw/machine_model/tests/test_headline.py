"""`machine-model headline`: each family's row at both readings, from the design files and vectors."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from verity_machine_model import headline as H
from verity_machine_model.cli import main
from verity_machine_model.eps import design, eps, machine, uniform_issue_charged
from verity_machine_model.model import DATA, device

# (ε_skip, ε_machine, ε_luck, ε, γ, the floor it needs, ε at p = 1): credit ♭, operands free, ε_mand 0 everywhere
PINS = {
    "sheet": {
        "sm_120": (0.004207, 0.001506, 0.000139, 0.005852, 2.2831, "half", None),
        "sm_90": (0.008583, 0.0, 0.000015, 0.008598, 2.4063, "half", None),
        "sm_100": (0.006981, 0.0, 0.000001, 0.006982, 2.6251, "half", None),
        "sm_89": (0.001463, 0.006652, 0.000336, 0.008451, 2.9795, "half", None),
        "sm_80": (0.001463, 0.0, 0.0, 0.001463, 5.9347, "half", None),
    },
    "measured": {
        "sm_120": (0.003628, 0.001251, 0.000517, 0.005396, 2.1357, "issue", None),
        "sm_90": (0.005927, 0.003303, 0.000986, 0.010215, 2.3435, "issue", None),
        "sm_100": (0.008145, 0.0, 0.000002, 0.008147, 2.5611, "half", None),
        "sm_89": (0.001470, 0.001126, 0.000468, 0.003065, 3.0144, "issue", None),
        "sm_80": (0.001470, 0.0, 0.0, 0.001470, 4.2263, "half", None),
    },
}
#: rows at or over 1%, each with its reason
OVER_ONE: dict[tuple[str, str], str] = {
    ("measured", "sm_90"): "♭ folds by the built routes alone and the cheater by li-bulk's counted trimmed W = ∞ member "
                           "too, so ε_machine is 0.330% at q* = 1.22; ffill, which no check reads, is uncredited",
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
        assert (x.core < 0.01) == ((reading, fam) not in OVER_ONE), fam
        assert x.q == H.spec()["reading"][reading]["q"], fam


def test_conditions_name_the_floor_the_fillers_credit_and_the_checked_fold():
    """The measured reading needs the 1.22-slot issue floor where ε_machine is above its floor at q* = ½, and states it
    as unproved in Lean; a filler's credit is the theorem its check meets, so the H200's ffill, which no
    check reads, is credited at p = 1, and the B200's fold is the one PouwWarpCheck reads. The H200's
    sheet row holds only with A3 per clock at its figures' 1830 MHz, not per second at f_max."""
    t = H.spec()
    m = {x.family: x for x in H.rows("measured")}
    assert m["sm_120"].conditions[0] == t["floor"]["issue"] and "Theorem C, unproved in Lean" in t["floor"]["issue"]
    assert m["sm_100"].conditions[0] == t["floor"]["half"] and m["sm_100"].threshold <= 0.5
    h = m["sm_90"]
    assert h.design == "dither-2s-licpw-redux-p1-l32" and h.p1 is None
    assert h.fillers == ("0.0703125 FFMA+IMAD a word uncredited, p = 1 (decision 99)",)
    assert any("RT-LOGIC-12" in c for c in h.conditions) and any("RT-LOGIC-12" in c for c in m["sm_100"].conditions)
    s = {x.family: x for x in H.rows("sheet")}
    assert all(x.floor == "half" for x in s.values())
    assert any(c.startswith("A3 per SM clock") and "1830 MHz" in c and "f_max (1980 MHz" in c
               for c in s["sm_90"].conditions)


def test_a_sheet_row_that_prices_the_uniform_route_by_a2u_states_it():
    """No data sheet gives a uniform rate, so a DS vector that charges a uniform instruction on `DS_issue` prices the
    uniform route by A2u alone (in no vendor document): its row says so, computed, not hand-written per family."""
    a2u = H.spec()["reading"]["sheet"]["a2u"]
    s = {x.family: x for x in H.rows("sheet")}
    assert {f for f, x in s.items() if a2u in x.conditions} == {"sm_120", "sm_100", "sm_90", "sm_89"}
    assert not any(c.startswith("A2u") for c in s["sm_80"].conditions)   # DS-80 has no uniform line: route closed
    assert not any(c.startswith("A2u") for x in H.rows("measured") for c in x.conditions)


def test_without_a2u_the_sheet_uniform_route_is_free():
    """Why the row states A2u: DS-120 with the uniform classes off `DS_issue` prices the uniform route at nothing, and
    the cheater writes every word there."""
    dd = device("sm_120")
    d, pi = design(H.spec()["family"]["sm_120"]["design"], dd), machine("DS-120", dd)
    c = {k: ({r: (0.0 if r == "DS_issue" else v) for r, v in rs.items()} if k.startswith("UNI") else rs)
         for k, rs in pi.c.items()}
    free = eps(d, replace(pi, c=c), 0.5, dd, operands=H.OPERANDS, credit=H.CREDIT)
    assert uniform_issue_charged(pi) and not uniform_issue_charged(replace(pi, c=c))
    assert free.machine_eps == pytest.approx(1.0) and free.cheat.words["uniform"] == pytest.approx(1.0)


def test_cli(capsys):
    assert main(["headline", "--reading", "measured"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "## decision 89's measured reading (`measured`): q* = 1.22, operands free, credit ♭"
    head = next(x for x in out if x.startswith("| GPU"))
    assert all(H.TERMS[k] in head for k in ("skip", "mand", "machine", "luck", "gamma", "threshold"))
    assert any(x.startswith("| H200 | dither-2s-licpw-redux-p1-l32 | DC-90-ri | 0.593% |") for x in out)
    assert any(x.startswith("- H200: the 1.22-slot issue floor") for x in out)
    assert f"- {H.spec()['scope']}" in out and "ε is per node:" in H.spec()["scope"]
    assert main(["headline", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["terms"] == H.TERMS and list(d["readings"]) == list(H.READINGS)
    assert d["credit"] == "floor" and d["operands"] == "free" and "decision 100" in d["capacity"]
    assert "accounting-declared-device" in d["capacity"] and f"- {d['capacity']}" in out
    assert d["scope"] == H.spec()["scope"]
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


def test_sm120s_measured_row_names_how_it_differs_from_its_lean_statement():
    """The row reads VF3r-16 at the reading's 1.22 while `Sm120MeasuredHeadline` reads VF3s-16 at 9/8."""
    x = next(x for x in H.rows("measured") if x.family == "sm_120")
    assert x.vector == "VF3r-16" and x.q == 1.22
    assert any("Sm120MeasuredHeadline reads VF3s-16 at q = q_T = 9/8" in c for c in x.conditions)
