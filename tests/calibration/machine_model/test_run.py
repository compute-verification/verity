"""`run.py`: every family's ε at every reading under the metric's one id, the RTX PRO 6000's sheet row last and bare."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import run
from verity_bench import kit
from verity_machine_model import headline as H


def test_each_row_is_a_member_of_the_one_metric_and_the_sheet_row_is_its_point(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(kit, "host_stamp", lambda: {"commit": None})
    (row,) = tomllib.loads((Path(run.__file__).parent / "metrics.toml").read_text())["metric"]
    assert run.main(["--out", str(tmp_path)]) == 0
    capsys.readouterr()
    ms = json.loads((tmp_path / "result.json").read_text())["measurements"]
    rows = [x for r in H.READINGS for x in H.rows(r)]
    assert {m["name"] for m in ms} == {row["id"]} and {m["unit"] for m in ms} == {row["unit"]}
    assert {m["mark"] for m in ms} == {"modeled"} and len(ms) == len(rows) + 1
    assert len({m["member"] for m in ms[:-1]}) == len(rows) and "member" not in ms[-1]
    own = H.row(*run.OWN)
    assert ms[-1]["value"] == round(100 * own.core, 6) and own.gpu == "RTX PRO 6000"
    h200 = next(m for m in ms if m.get("member") == "h200-measured")
    assert h200["value"] == round(100 * H.row("sm_90", "measured").core, 6)
