"""`verity bench machine_model` (`benchmarks/machine_model/run.py`): the sm_120 headline's core ε at both readings goes
into the run's result under the id `metrics.toml` declares, modeled, the data sheet's reading last and with no member."""
from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

from verity_bench import kit
from verity_machine_model import headline as H

DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "machine_model"
_spec = importlib.util.spec_from_file_location("machine_model_run", DIR / "run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_run_writes_the_sm120_headline_under_its_metrics_id(tmp_path, monkeypatch, capsys):
    for name in ("RESEARCH_RUN_ID", "RESEARCH_RUN_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OUT", str(tmp_path))
    monkeypatch.setattr(kit, "host_stamp", lambda: {"commit": None})
    (row,) = tomllib.loads((DIR / "metrics.toml").read_text())["metric"]
    sheet, measured = H.row("sm_120", "sheet"), H.row("sm_120", "measured")
    assert run.main([]) == 0
    head = capsys.readouterr().out.splitlines()[0]
    first, last = json.loads((tmp_path / "result.json").read_text())["measurements"]
    assert row["id"] == run.METRIC and head.startswith(f"machine_model: ε {100 * sheet.core:.3f}% for the RTX PRO 6000")
    assert (last["name"], last["value"], last["unit"], last["mark"], last.get("member")) == (
        row["id"], sheet.core, row["unit"], "modeled", None)
    assert (first["name"], first["value"], first["mark"], first["member"]) == (row["id"], measured.core, "modeled",
                                                                              "measured")
    assert json.loads((tmp_path / "machine_model.json").read_text())["value"] == sheet.core
