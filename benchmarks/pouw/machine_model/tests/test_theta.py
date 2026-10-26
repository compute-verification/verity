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
    tma, ffma = T.ADDING[10], T.ADDING[4]
    assert d(SM90, "sheet")[tma] == 18.29 and d(SM90, "measured")[ffma] == 32.02
    assert min(d(SM90, "sheet").values()) == 18.29 and d(SM120, "sheet")[tma] == 4.57
    assert T.REDUCTIONS["nvls.red"] in d(SM90, "sheet") and T.REDUCTIONS["nvls.red"] not in d(SM120, "sheet")


def test_strassen_saves_on_the_sheet_and_not_measured():
    lanes = (10, 20, 20, 20 / 3, 10, 40, 40, 40, 40, 4096 / 3.93 * 10 / 32, 4096 / 224 * 10 / 32)
    for data, rates in ((SM90, lanes), (SM120, (1.25, 2.5, 5, 5 / 3, 2.5, 10, 10, 10, 10, 1024 / 224 * 10 / 32))):
        t = T.theta(data)
        s = T.strassen(t, "sheet", 32)
        assert s.tensor == pytest.approx(7 / 8)
        assert [s.rates[n] for n in T.ADDING if n in s.rates] == pytest.approx(rates)
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
    assert set(b["sm_90", "measured"].filled) == {"tensor.f16.k8.sync", "tensor.tf32.k4.sync", "smem.bcast", "dfma",
                                                  "imad.wide"}
    assert b["sm_90", "measured"].words == pytest.approx(447.9, abs=0.1)
    assert b["sm_90", "measured"].mix["mma.sync m16n8k8 f16.f32"] == pytest.approx(2)
    assert b["sm_90", "measured only"].words == pytest.approx(246.8, abs=0.1)
    assert all(0.05 < x.c < 0.6 for x in b.values())


def test_r_h_against_r_min_and_gamma():
    [f] = T.folds(T.theta(SM120), SM120)
    assert f.r_min == 3 and f.r_h == pytest.approx(3.2206, abs=1e-4)
    assert f.eps("sheet") == pytest.approx(f.slack["sheet"] / 1.25) and f.eps("measured") == 0
    assert f.gamma == pytest.approx(26.1221 / (128 * 32 * 4 / 1023.8))
    duo, li, loop = T.folds(T.theta(SM90), SM90)
    assert duo.r_h == duo.r_min == 3 and duo.status == "provisional" and duo.gamma is None
    assert (li.fold, li.r_h, li.r_min, li.gamma) == ("li", 1, 1, None)
    assert loop.gamma == pytest.approx(280.89 / 128.01, abs=1e-3) and loop.r_h == pytest.approx(1.0547, abs=1e-4)


def test_every_unit_has_a_verdict_and_the_bounded_ones_a_share():
    for data, texture, absent in ((SM90, 16 / 4096, "rt"), (SM120, 16 / 1024, "nvlink.reduce")):
        t = T.theta(data)
        assert set(t.units) == set(T.UNITS) and t.units[absent].verdict == "absent"
        assert T.free(t, "sheet") == pytest.approx({"tensor.s8.k32": texture})
        assert all(u.gap for u in t.units.values() if u.verdict == "open")
    rt = T.theta(SM120).units["rt"]
    assert rt.verdict == "open" and T.share(T.theta(SM120), rt, "sheet") == pytest.approx(386.2 / 1024)
    for data, cps in ((SM90, 4.79e9), (SM120, 7.13e9)):
        t = T.theta(data)
        host = t.units["host.cpu"]
        assert host.verdict == "open" and "κ" in host.gap
        assert T.share(t, host, "measured") == pytest.approx(64e9 / 128 / cps, rel=2e-3)


def test_sha512_against_the_public_figures():
    by = lambda data: {(s.public.card.split(" (")[0], s.public.form): s for s in T.sha(T.theta(data))}
    rtx, h = by(SM120), by(SM90)
    server = rtx["RTX PRO 6000 Blackwell Server Edition", "generic"]
    assert server.per_clk == pytest.approx(7.894e9 / (188 * 2.43e9)) and server.c == 1
    assert rtx["RTX PRO 6000 Blackwell Server Edition", "structured"].c == pytest.approx(0.878, abs=1e-3)
    assert rtx["RTX PRO 6000 Blackwell Workstation Edition", "structured"].c == pytest.approx(0.832, abs=1e-3)
    assert h["H100 SXM 80GB HBM3", "generic"].c == 1
    assert h["H100 SXM 80GB HBM3", "structured"].c == pytest.approx(0.8847, abs=1e-3)


def test_the_rules_refuse_a_missing_unit_or_an_unbounded_one(tmp_path):
    shutil.copytree(SM90, tmp_path / "d", ignore=shutil.ignore_patterns("census", "vectors", "designs"))
    p = tmp_path / "d" / T.THETA
    text = p.read_text()
    p.write_text(text.replace('[[unit]]\nkey = "copy"', '[[ignored]]\nkey = "copy"'))
    with pytest.raises(SystemExit, match="no \\[\\[unit\\]\\] for copy"):
        T.theta(tmp_path / "d")
    p.write_text(text.replace('bound = 16\nagainst = "tensor.s8.k32"\n', ""))
    with pytest.raises(SystemExit, match="unit texture: a bounded unit gives `bound`"):
        T.theta(tmp_path / "d")


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
    assert "r_H = 3.0000, r_min = 3" in out and "the least r_min here is li's 1" in out
    assert "280.89 / 128.01 = 2.194" in out
    assert "texture: texture filtering: 4 texture units a SM; ≤ 16 against tensor.s8.k32: 0.39% / 0.39%" in out
    assert "c_sha ≤ 0.885" in out
    assert "    mma.sync m16n8k32 s8: 128.0 / 84.0\n" in out and "    mma.sync m16n8k4 tf32: 256.0 / 256.0*\n" in out
    assert main(["theta", "--check"]) == 0
    assert "holds" in capsys.readouterr().out
    assert main(["render", "theta"]) == 0
    out = capsys.readouterr().out
    assert "| `tensor.s8.k16` |" in out and "### FP8 atoms" in out
    with pytest.raises(SystemExit, match="no theta.toml"):
        main(["--device", "sm_80", "theta"])
