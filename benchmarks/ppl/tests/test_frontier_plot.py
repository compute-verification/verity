import json
import math
import subprocess

import pytest

import frontier_plot
from test_report import results


def fake_eps(core: dict, packed: float = 0.004, floor: float = 0.03) -> dict:
    """An EPS.json in which design `label` has core ε core[label] with the table route closed or at QSTAR's measured
    route, `packed` on SPEC's packed route and `floor` on either floor."""
    fk = frontier_plot.flag_key
    fixed = {fk(frontier_plot.PACKED): packed, fk(frontier_plot.FLOOR): floor, fk(frontier_plot.FLOOR_B2): floor}
    return {"settings": "", "designs": {
        label: {"design": design, "gamma": 1.948, "vector": "VF3n-16",
                "by_flags": {"closed": core.get(label, floor), fk(frontier_plot.MEASURED): core.get(label, floor),
                             **fixed}}
        for label, _, design, _ in frontier_plot.DESIGNS}}


def test_eps_runs_each_design_once_per_flag_set_at_full_checking(monkeypatch, tmp_path):
    seen = []

    def run(cmd, **_):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"core": 0.002, "gamma": 1.948,
                                                               "terms": {"skip": 0.001}, "vector": "VF3n-16"}), "")

    monkeypatch.setattr(frontier_plot.subprocess, "run", run)
    out = tmp_path / "eps.json"
    assert frontier_plot.eps(str(out)) == 0
    assert len({tuple(c) for c in seen}) == len(seen)
    assert all(cmd[:2] == ["machine-model", "eps"] and "--checking" in cmd and cmd[cmd.index("--checking") + 1] == "full"
               for cmd in seen)
    doc = json.loads(out.read_text())
    assert set(doc["designs"]["Dither-2S (A)"]["by_flags"]) == {
        "closed", *(frontier_plot.flag_key(f) for f in (frontier_plot.MEASURED, frontier_plot.PACKED,
                                                        frontier_plot.FLOOR))}
    assert set(doc["designs"]["H1, U ±16"]["by_flags"]) == {
        "closed", *(frontier_plot.flag_key(f) for f in (frontier_plot.MEASURED, frontier_plot.PACKED,
                                                        frontier_plot.FLOOR))}


def test_shares_split_a_tokens_committed_words_by_kind():
    a = frontier_plot.shares("llama-3.1-8b", 2048, frontier_plot.DESIGNS[1][3])
    h1 = frontier_plot.shares("llama-3.1-8b", 2048, frontier_plot.DESIGNS[7][3])
    assert set(a) == {"linears", "qk", "pv"} and set(h1) == {"linears", "digits", "qk", "pv"}
    assert sum(a.values()) == pytest.approx(1) and sum(h1.values()) == pytest.approx(1)
    assert 0.015 < a["pv"] < 0.02 and 0.005 < h1["digits"] < 0.01
    assert frontier_plot.shares("llama-3.1-70b", 2048, frontier_plot.DESIGNS[1][3])["pv"] < a["pv"]
    assert frontier_plot.shares("llama-3.1-8b", 8192, frontier_plot.DESIGNS[1][3])["pv"] > a["pv"]
    ep = fake_eps({"Dither-2S (A)": 0.001, "H1, U ±16": 0.002})
    assert frontier_plot.core(ep, "Dither-2S (A)", "builds charged", "llama-3.1-8b", 2048) == pytest.approx(
        (1 - a["pv"]) * 0.004 + a["pv"] * 0.03)
    assert frontier_plot.core(ep, "H1, U ±16", "builds charged", "llama-3.1-8b", 2048) == pytest.approx(
        h1["linears"] * 0.002 + h1["qk"] * 0.004 + (h1["digits"] + h1["pv"]) * 0.03)
    assert frontier_plot.core(ep, "Dither-2S (A)", "closed", "llama-3.1-8b", 2048) == pytest.approx(0.001)


def test_the_frontier_is_the_pareto_minimal_points_by_eps():
    assert frontier_plot.frontier([(0.5, 1.0), (0.1, 2.0), (0.2, 2.5), (0.3, 0.9), (0.6, 0.95)]) == [
        (0.1, 2.0), (0.3, 0.9)]


def test_rows_and_table_mark_designs_that_meet_every_target(tmp_path):
    seeds = lambda base: {s: [base + 0.001 * s] * 16 for s in range(3)}  # noqa: E731
    b8 = results({"bf16": ("", {0: [2.0] * 16}), "adopted-bd-lm": ("int8-all", seeds(2.007)),
                  "co32c-u16": ("int8-all", seeds(2.008)), "co32c": ("int8-all", seeds(2.015)),
                  "side-lin-act": ("int8-all", seeds(2.0))},
                 point=("llama-3.1-8b", "wikitext2", 2048), run_id="r20261017-000000-aaaa")
    b70 = results({"bf16": ("", {0: [2.0] * 16}), "adopted-bd-lm": ("int8-all", seeds(2.016)),
                   "co32c-u16": ("int8-all", {0: [2.025] * 16})},
                  point=("llama-3.1-70b", "c4", 2048), run_id="r20261017-000000-bbbb")
    failed = results({"bf16": ("", {0: [2.0] * 16}), "co32c": ("int8-all", seeds(2.0))}, passed=False)
    paths = []
    for n, r in enumerate((b8, b70, failed)):
        paths.append(tmp_path / f"{n}.json")
        paths[-1].write_text(json.dumps(r))
    rs = frontier_plot.rows([str(p) for p in paths])
    assert {k[3] for k in rs} == {"adopted-bd-lm", "co32c-u16", "co32c"}
    u16 = rs[("llama-3.1-8b", "wikitext2", 2048, "co32c-u16")]
    assert u16["inc"] == pytest.approx(100 * (sum(math.exp(0.008 + 0.001 * s) for s in range(3)) / 3 - 1))
    assert u16["seeds"] == 3
    assert u16["inc_min"] == pytest.approx(100 * (math.exp(0.008) - 1))
    ep = fake_eps({"Dither-2S (A)": 0.004, "H1, U ±16": 0.002, "Dither-C (co32c)": 0.002})
    t = frontier_plot.table(rs, ep)
    lines = {line.split(" | ")[0].strip("| "): line for line in t.splitlines() if line.startswith("| ")}
    charged = " / ".join(f"{100 * frontier_plot.core(ep, 'Dither-2S (A)', 'builds charged', m, 2048):.3f}"
                         for m in ("llama-3.1-8b", "llama-3.1-70b"))
    assert lines["Dither-2S (A)"].endswith(
        f"| 0.400 / 0.400% ✓ | 0.400 / 0.400% ✓ | {charged}% ✓ | 3.000 / 3.000% |")
    assert "core ε, closed (8B 2K / 70B 2K)" in t
    assert "✓" not in lines["H1, U ±16"] and "(1 seed), r20261017-000000-bbbb" in lines["H1, U ±16"]
    assert "✓" not in lines["Dither-C (co32c)"] and "| – |" in lines["Dither-C (co32c)"]
    assert lines["H1, U ±8†"].split(" | ")[2:4] == ["–", "–"]
    assert t.rstrip().endswith("(±16 holds to x = 9.48).")
    pytest.importorskip("matplotlib")
    png = tmp_path / "f.png"
    frontier_plot.plot(rs, fake_eps({}), str(png))
    assert png.stat().st_size > 0
