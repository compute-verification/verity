"""`machine-model`'s commands, and adding a measurement as data: a row, its census lines, an epoch and a vector."""
from __future__ import annotations

import json
import shutil

from verity_machine_model.cli import main
from verity_machine_model.model import DATA, vector
from verity_machine_model.resolve import resolve

RUN = "r20261016-000000-0000"


def test_solve_prints_the_numbers_and_the_check(capsys):
    assert main(["solve", "VF3j-16"]) == 0
    out = capsys.readouterr().out
    assert "τ 0.0610   τ at 600 W 0.0477   U/F_low 1.1563   margin 2.000" in out
    assert "(reproduced)" in out and "1306 attacks in 13 families at X ≥ 1.0: pass" in out


def test_solve_defaults_to_the_current_vector(capsys):
    assert main(["solve", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["vector"] == "VF3j" and round(d["tau"], 4) == 0.0634 and d["failures"] == [] and d["unreproduced"] == []


def test_list_census_attacks(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "VF3i-strict-r8" in out and "superseded by VF3i-strict" in out
    assert main(["census", "VF3j", "--kind", "FFMA"]) == 0
    assert capsys.readouterr().out.startswith("FFMA: ")
    assert main(["attacks", "VF3j-16", "--design", "U±64, F±1", "-n", "1"]) == 0
    assert "X 2.000" in capsys.readouterr().out


def _epoch(data, rows=()):
    """README "Add a measurement": the epoch VF3k after the last, and a vector VF3k on VF3j's rows and credits plus `rows`."""
    m = (data / "model.toml").read_text().replace('"VF3j", "VF3j-8"]', '"VF3j", "VF3j-8", "VF3k"]')
    (data / "model.toml").write_text(m)
    t = (data / "vectors" / "VF3j.toml").read_text()
    t = t.replace('name = "VF3j"', 'name = "VF3k"').replace('epoch = "VF3j"', 'epoch = "VF3k"')
    t = t.replace("rows = [\n", "rows = [\n" + "".join(f'    "{r}",\n' for r in rows)).split("[published]")[0]
    (data / "vectors" / "VF3k.toml").write_text(t)


def test_a_measurement_is_data(tmp_path, capsys):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    with open(data / "rows.csv", "a") as f:
        f.write(f'ROW_TEST,measured,{RUN},"a test row: FFMA alone at its F rate"\n')
    with open(data / "census.csv", "a") as f:
        f.write(f"FFMA,ROW_TEST,3.885,,,,VF3k,,{RUN}\n")
    _epoch(data, ["ROW_TEST"])
    new, old = resolve("VF3k", data), resolve("VF3j", data)
    assert new.failures == [] and len(new.vector.rows) == len(old.vector.rows) + 1
    assert "ROW_TEST" in new.census["FFMA"] and "ROW_TEST" not in old.census["FFMA"]
    assert (old.tau, old.margin) == (resolve("VF3j").tau, resolve("VF3j").margin)
    assert "LDSM_X1" in old.census and "JI3e_0" not in new.census["LDSM_X1"], "VF3j-8's lines close at VF3j-8"
    assert vector("VF3k", data).published == {}
    assert main(["--data", str(data), "solve", "VF3k"]) == 0
    assert "published: nothing yet" in capsys.readouterr().out


def test_a_refit_closes_the_old_line(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    c = (data / "census.csv").read_text()
    old_line = "FFMA,F,3.885,,,,,,note:20261015T0608Z-draft-census-sm120\n"
    assert c.count(old_line) == 1
    c = c.replace(old_line, old_line.replace(",,note:", ",VF3j-8,note:")) + f"FFMA,F,3.9,,,,VF3k,,{RUN}\n"
    (data / "census.csv").write_text(c)
    _epoch(data)
    new, old = resolve("VF3k", data), resolve("VF3j", data)
    assert new.census["FFMA"]["F"] == new.model.cycle_u * 4 / 3.9
    assert old.census["FFMA"]["F"] == old.model.cycle_u * 4 / 3.885
    assert (old.tau, old.margin) == (resolve("VF3j").tau, resolve("VF3j").margin) and new.failures == []
