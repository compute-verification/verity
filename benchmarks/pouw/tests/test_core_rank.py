"""`core_rank/rank.py` refuses a row that leaves a target uncited or unplaced, and pools a shared gap over each period."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("core_rank", POUW / "core_rank" / "rank.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

PERIOD = {"world_flop_year": 1e30, "dc_gpus": 1000, "gpu_flops": 1e15, "mfu": 0.5, "gates_per_flop": 100,
          "sources": ["dummy"]}


def row(**over) -> dict:
    r = {"id": "R1", "pos": 1, "verdict": "ranked", "design": "d", "catalogs": ["c"], "assumption": "a", "rank": 3,
         "lean": ["proved"], "lean_ids": "x", "vllm": "not served", "why": "w", "cites": ["s"], "share": 0.01,
         "share_basis": "work bound"}
    for t in R.TARGETS:
        r |= {t: "?", f"{t}_card": ["none"], f"{t}_status": "open"}
    return r | over


def test_a_complete_table_renders_and_pools_its_share(tmp_path):
    rows = tmp_path / "rows.toml"
    rows.write_text('[period]\nworld_flop_year = 1e30\ndc_gpus = 1000\ngpu_flops = 1e15\nmfu = 0.5\ngates_per_flop = 100\n'
                    'sources = ["dummy"]\n[[row]]\n' + "\n".join(f"{k} = {json.dumps(v)}" for k, v in row().items()))
    assert R.main(["--rows", str(rows), "--out", str(tmp_path / "out")]) == 0
    got = json.loads((tmp_path / "out" / "ranking.json").read_text())["per_period"][0]
    assert got["world_year"] == pytest.approx(0.01 * 1e30 * 100)
    assert got["world_week"] == pytest.approx(0.01 * 1e30 * 100 * 7 / 365)
    assert got["datacenter_year"] == pytest.approx(0.01 * 1000 * 1e15 * 0.5 * 365 * 86400 * 100)
    md = (tmp_path / "out" / "ranking.md").read_text()
    assert "| 1 | R1 | d | 3: a | ? [none; open] |" in md


@pytest.mark.parametrize(("over", "says"), [
    ({"t1": ""}, "R1: 't1' is empty; write '?' where no number exists"),
    ({"t2_card": ["H200"]}, "R1: 't2_card' must be a non-empty list from"),
    ({"rank": 6}, "R1: rank 6 is not an integer 0..5"),
    ({"lean": ["sorry"]}, "R1: Lean status ['sorry'] not in"),
    ({"pos": 2}, "ranked positions [2] must be 1..1"),
    ({"share": 0.01, "share_basis": ""}, "R1: 'share' must be a fraction in [0, 1] with a 'share_basis'"),
])
def test_an_incomplete_row_is_refused_by_name(over, says):
    assert any(e.startswith(says) for e in R.check([row(**over)], PERIOD)), R.check([row(**over)], PERIOD)


def test_a_period_without_its_sources_is_refused():
    assert R.check([row()], {k: v for k, v in PERIOD.items() if k != "sources"}) == ["[period] lacks 'sources'"]
