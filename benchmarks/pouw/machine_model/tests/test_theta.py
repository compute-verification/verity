"""θ (`theta.toml`): every rate sourced, every figure reproduced, and the derived d, exchange rate, B_RF and r_H."""
from __future__ import annotations

import shutil

import pytest

from verity_machine_model import theta as T
from verity_machine_model.cli import main
from verity_machine_model.model import DATA, device

SM120, SM90 = DATA, device("sm_90")


@pytest.mark.parametrize("data", [SM120, SM90], ids=["sm_120", "sm_90"])
def test_every_rate_is_sourced_and_every_figure_holds(data):
    t = T.theta(data)
    assert not T.check(t)
    fs = T.figures(t)
    assert fs and all(f.holds for f in fs), [f for f in fs if not f.holds]
    assert {w.rate for w in t.writers} <= set(t.rates)
    assert t.paths and all(p[k] for p in t.paths for k in ("format", "native", "exact", "isa"))
    assert any(p["format"].startswith("b1") for p in t.paths)
    assert len(t.fp8) >= 1 and all(x["source"] for x in t.fp8)


def test_d_by_route():
    d = lambda data, reading: {r.name: round(r.d, 2) for r in T.routes(T.theta(data), reading)}
    two, imad = T.ADDING[0], T.ADDING[2]
    assert d(SM90, "sheet")[two] == 32 and d(SM90, "sheet")[imad] == 64
    assert d(SM90, "measured")[two] == 32.5
    assert d(SM120, "sheet")[two] == 4 and d(SM120, "measured")[two] == 8
    assert min(min(d(x, r).values()) for x in (SM120, SM90) for r in T.READINGS) == 4


def test_strassen_saves_on_the_sheet_and_not_measured():
    for data, rates in ((SM90, (10, 20, 20, 20 / 3)), (SM120, (1.25, 2.5, 5, 5 / 3))):
        t = T.theta(data)
        s = T.strassen(t, "sheet", 32)
        assert s.tensor == pytest.approx(7 / 8)
        assert [s.rates[n] for n in T.ADDING] == pytest.approx(rates)
        m = T.strassen(t, "measured", 32)
        assert m.tensor > 1.7 and not m.rates
    with pytest.raises(SystemExit, match="L is even"):
        T.strassen(T.theta(SM90), "sheet", 31)


def test_b_rf():
    b = {(x, r): T.bound(T.theta(DATA if x == "sm_120" else device(x)), r) for x in ("sm_120", "sm_90")
         for r in (*T.READINGS, "measured only")}
    assert b["sm_90", "sheet"].words == pytest.approx(448) and b["sm_90", "sheet"].c == pytest.approx(4096 / 14336)
    assert b["sm_120", "sheet"].words == pytest.approx(352)
    assert b["sm_120", "measured"].words == pytest.approx(265.8, abs=0.1)
    assert b["sm_90", "measured only"].dropped and not b["sm_90", "sheet"].dropped
    assert set(b["sm_90", "measured"].filled) == {"smem.bcast", "dfma", "imad.wide"}
    assert all(0.05 < x.c < 0.6 for x in b.values())


def test_r_h_against_r_min():
    f = T.fold(T.theta(SM120), SM120)
    assert f.r_min == 3 and f.r_h == pytest.approx(3.2206, abs=1e-4)
    assert f.eps("sheet") == pytest.approx(f.slack["sheet"] / 1.25) and f.eps("measured") == 0
    h = T.fold(T.theta(SM90), SM90)
    assert h.r_h == h.r_min == 3 and h.status == "provisional"


def test_the_rules_refuse_an_unsourced_or_wrong_rate(tmp_path):
    shutil.copytree(SM90, tmp_path / "d", ignore=shutil.ignore_patterns("census", "vectors", "designs"))
    p = tmp_path / "d" / T.THETA
    text = p.read_text()
    p.write_text(text.replace('key = "alu.prmt"\nwhat = "PRMT (byte permute, __byte_perm)"\nunit = "lanes/SM/clk"\n',
                              'key = "alu.prmt"\nwhat = "PRMT (byte permute, __byte_perm)"\nunit = "lanes/SM/clk"\n'
                              'sheet = 64\n'))
    with pytest.raises(SystemExit, match="alu.prmt: a rate with a `sheet` value"):
        T.theta(tmp_path / "d")
    p.write_text(text.replace("figure = 1979\n", "figure = 1900\n", 1))
    assert not T.figures(T.theta(tmp_path / "d"))[0].holds


def test_the_command_and_the_render(capsys):
    assert main(["--device", "sm_90", "theta"]) == 0
    out = capsys.readouterr().out
    assert "4096 / 128 [2 · alu.iadd3 64] = 32.00" in out
    assert "7/8 · 4095.6 / 1995.7 = 1.7957 of naive: the level saves no tensor time" in out
    assert "r_H = 3.0000, r_min = 3" in out
    assert main(["theta", "--check"]) == 0
    assert "holds" in capsys.readouterr().out
    assert main(["render", "theta"]) == 0
    out = capsys.readouterr().out
    assert "| `tensor.s8.k16` |" in out and "### FP8 atoms" in out
    with pytest.raises(SystemExit, match="no theta.toml"):
        main(["--device", "sm_80", "theta"])
