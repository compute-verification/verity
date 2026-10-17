import json
import math
import subprocess

import pytest

import frontier_plot
from test_report import results


def fake_eps(core: dict) -> dict:
    """An EPS.json in which design `label` has core ε core[label] under every reading."""
    return {"settings": "", "readings": {reading: {label: {"core": core.get(label, 0.03), "gamma": 1.948, "skip": 0.0,
                                                           "vector": "VF3n-16", "cmd": ""}
                                                   for label, _, _, _ in frontier_plot.DESIGNS}
                                         for reading in frontier_plot.READINGS}}


def test_eps_runs_every_design_under_every_reading_at_full_checking(monkeypatch, tmp_path):
    seen = []

    def run(cmd, **_):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"core": 0.002, "gamma": 1.948,
                                                               "terms": {"skip": 0.001}, "vector": "VF3n-16"}), "")

    monkeypatch.setattr(frontier_plot.subprocess, "run", run)
    out = tmp_path / "eps.json"
    assert frontier_plot.eps(str(out)) == 0
    assert len(seen) == len(frontier_plot.DESIGNS) * len(frontier_plot.READINGS)
    assert all(cmd[:2] == ["machine-model", "eps"] and "--checking" in cmd and cmd[cmd.index("--checking") + 1] == "full"
               for cmd in seen)
    doc = json.loads(out.read_text())
    assert doc["readings"]["measured"]["Dither-2S (A)"]["cmd"].endswith("--lookup-q 1.906 --lookup-bytes 8")
    assert "--lookups" not in doc["readings"]["closed"]["H1, U ±16"]["cmd"]


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
    t = frontier_plot.table(rs, fake_eps({"Dither-2S (A)": 0.004, "H1, U ±16": 0.002, "Dither-C (co32c)": 0.002}))
    lines = {line.split(" | ")[0].strip("| "): line for line in t.splitlines() if line.startswith("| ")}
    assert lines["Dither-2S (A)"].endswith("| 0.400% ✓ | 0.400% ✓ | 0.400% ✓ | 0.400% ✓ |")
    assert "0.200% ✓" not in lines["H1, U ±16"] and "(1 seed), r20261017-000000-bbbb" in lines["H1, U ±16"]
    assert "0.200% ✓" not in lines["Dither-C (co32c)"] and "| – |" in lines["Dither-C (co32c)"]
    assert lines["H1, U ±8†"].split(" | ")[2:4] == ["–", "–"]
    assert t.rstrip().endswith("(±16 holds to x = 9.48).")
    pytest.importorskip("matplotlib")
    png = tmp_path / "f.png"
    frontier_plot.plot(rs, fake_eps({}), str(png))
    assert png.stat().st_size > 0
