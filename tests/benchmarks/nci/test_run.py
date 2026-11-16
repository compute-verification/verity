"""`verity bench nci` (`benchmarks/nci/run.py`): the slowdown statement's first figure goes into the run's result under
the id `metrics.toml` declares, modeled, as the run's one measurement."""
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str((Path(__file__).resolve().parent.parent.parent.parent / "benchmarks" / "nci")))

import run  # noqa: E402
import statement as S  # noqa: E402
import tiers as T  # noqa: E402
from verity_bench import kit  # noqa: E402


def test_run_writes_the_statements_first_figure_under_its_metrics_id(tmp_path, monkeypatch, capsys):
    for name in ("RESEARCH_RUN_ID", "RESEARCH_RUN_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OUT", str(tmp_path))
    monkeypatch.setattr(kit, "host_stamp", lambda: {"commit": None})
    (row,) = tomllib.loads((Path(run.__file__).parent / "metrics.toml").read_text())["metric"]
    want = S.slowdown(T.DEFAULTS | S.BEST, "T1")
    assert run.main([]) == 0
    assert capsys.readouterr().out.startswith(f"nci: {S.num(want)}x, the slowdown at the best plausible PoUW design, T1")
    (m,) = json.loads((tmp_path / "result.json").read_text())["measurements"]
    assert (m["name"], m["value"], m["unit"], m["mark"]) == (row["id"], want, row["unit"], "modeled")
