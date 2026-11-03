"""te_core.console cuts a panel to the panels route's limits, the first cut that fits, and names the cut in its note."""
import json

import pytest

pytest.importorskip("matplotlib")

from te_core import console

P1 = ["start", "start_id", "size", "saturation", "curve", "policy", "basis", "target_pct", "compute_gates",
      "compute_frac", "loss", "loss_lo", "loss_hi", "record", "checkpoint", "method", "family"]


def p1(points: int, band: int) -> dict:
    rows = []
    for curve, policies in (("honest", [""]), ("covert", ["c=4096w k=10kB"]),
                            ("covert-policy", [f"c1e1{i}/k1e{i}" for i in range(band)])):
        for pol in policies:
            for i in range(points):
                loss = 1.2 * (1 - 0.0001 * i)
                rows.append(["34M", "s1", 4.5e7, 3.42, curve, pol, "target", None, 1e17 * (1 + i), None, loss, None,
                             None, f"record-{i:06d}", i, "ref-optim/muon", "full"])
    return {"id": "training/p1-loss-vs-compute", "title": "t", "kind": "table", "updated_at": "now", "source": "s",
            "note": "x" * 2000, "columns": P1, "rows": rows}


def test_a_small_panel_is_kept_whole_with_the_short_note():
    q = console.cut(p1(10, 2))
    assert len(q["rows"]) == 40 and "loss_lo" not in q["columns"]
    assert q["note"].startswith(console.SHORT["p1-loss-vs-compute"]) and len(q["note"]) <= console.NOTE_MAX


def test_a_large_panel_takes_the_first_cut_that_fits():
    q = console.cut(p1(2000, 9))
    assert len(console.body(q)) <= console.BODY_MAX and len(q["note"]) <= console.NOTE_MAX
    rows = [dict(zip(q["columns"], r)) for r in q["rows"]]
    assert "apart" in q["note"] and {r["curve"] for r in rows} >= {"honest", "covert"}
    ends = [r for r in rows if r["curve"] == "honest"]
    assert ends[0]["loss"] == pytest.approx(1.2) and ends[-1]["record"] == "record-001999"


def test_write_cuts_every_panel(tmp_path):
    from te_core.plots import NAMES
    for n in NAMES:
        p = p1(5, 1) | {"id": f"training/{n}"}
        (tmp_path / f"{n}.json").write_text(json.dumps(p))
    made = console.write(tmp_path, tmp_path / "console")
    assert sorted(made) == sorted(NAMES) and all((tmp_path / "console" / f"{n}.json").exists() for n in NAMES)
