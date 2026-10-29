"""`machine-model`'s commands, and adding a measurement as data: a row, its census lines, an epoch and a vector."""
from __future__ import annotations

import json
import shutil

import pytest

from verity_machine_model.adopt import adopt
from verity_machine_model.cli import main
from verity_machine_model.model import DATA, census, model, vector
from verity_machine_model.resolve import resolve

RUN = "r20261016-000000-0000"


def test_solve_prints_the_numbers_and_the_check(capsys):
    assert main(["solve", "VF3j-16"]) == 0
    out = capsys.readouterr().out
    assert "τ 0.0610   τ at 600 W 0.0477   U/F_low 1.1563   margin 2.000" in out
    assert "(reproduced)" in out and "1306 attacks in 13 families at X ≥ 1.0: pass" in out
    assert "superseded by VF3k-16" in out
    assert "(π₃₂): every root-writing kind at ≥ 32 MACs a word: 84 kinds, holds (not split out at VF3j-8: ATOME32P)" in out
    assert "a UR word at the 32 lanes it serves) at ≥ 16 MACs a word: 62 kinds, fails: UNI 2.00 a word < 16;" in out
    assert "copy routes, MACs a word (price/measured), cheapest first: UNI64 1.00/" in out


# VF3n-16's MMA forms, each its own kind from VF3n-16: an earlier vector doesn't split them out
MMA_FORMS = ["HMMA16816", "HMMA16816F16", "HMMA1684T", "HMMA1688T", "HMMASP16832", "IMMA16816", "IMMA16832", "IMMA8816", "IMMASP16832", "IMMASP16864", "OMMASF16864", "QMMA16832", "QMMASF16832", "QMMASP16864"]


# the loads VF3q-16 split out of the census's unpriced ones: an earlier vector doesn't
VF3Q_LOADS = ["LDS64", "LDG64", "LDG256", "LDG128NA", "LDG128CG", "LDG256CG"]


def test_solve_defaults_to_the_current_vector(capsys):
    assert main(["solve", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["vector"] == model().current and d["failures"] == [] and d["unreproduced"] == []
    assert main(["solve", "VF3r-16", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert round(d["tau"], 4) == 0.0556 and d["failures"] == [] and d["unreproduced"] == []
    assert (round(d["tau_600w"], 4), round(d["uf_low"], 4), d["margin"]) == (0.0435, 1.1563, 2.0)
    assert d["floors"] == {"pi32": {"kinds": 85, "below": [], "absent": []},
                           "loads16": {"kinds": 15, "below": [], "absent": []},
                           "copy16": {"kinds": 83, "below": ["UNI64 1.00 a word < 16"], "absent": []}}
    assert d["copies"][0]["kind"] == "UNI64"
    uni = next(c for c in d["copies"] if c["kind"] == "UNI")
    assert uni["words"] == 32 and round(uni["price"], 2) == 29.59 and uni["measured"] > 32
    assert uni["what"].startswith("UR fan-out")
    assert main(["solve", "VF3p-16", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["floors"]["copy16"]["absent"] == VF3Q_LOADS
    assert main(["solve", "VF3l-16", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["floors"]["copy16"]["absent"] == MMA_FORMS + VF3Q_LOADS


def test_list_census_attacks(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "VF3i-strict-r8" in out and "superseded by VF3i-strict" in out
    assert main(["census", "VF3j", "--kind", "FFMA"]) == 0
    assert capsys.readouterr().out.startswith("FFMA: ")
    assert main(["attacks", "VF3j-16", "--design", "U±64, F±1", "-n", "1"]) == 0
    assert "X 2.000" in capsys.readouterr().out


def _epoch(data, rows=()):
    """README "Add a measurement": the epoch TEST after the last, and a vector TEST on the current vector's rows and
    credits plus `rows`."""
    m = model(data)
    last = m.epochs[-1]
    (data / "model.toml").write_text((data / "model.toml").read_text().replace(f'"{last}"]', f'"{last}", "TEST"]'))
    t = (data / "vectors" / f"{m.current}.toml").read_text()
    t = t.replace(f'name = "{m.current}"', 'name = "TEST"').replace(f'epoch = "{vector(m.current, data).epoch}"',
                                                                    'epoch = "TEST"')
    t = t.replace("rows = [\n", "rows = [\n" + "".join(f'    "{r}",\n' for r in rows)).split("[published]")[0]
    (data / "vectors" / "TEST.toml").write_text(t)
    return m.current, last


def test_a_measurement_is_data(tmp_path, capsys):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    with open(data / "rows.csv", "a") as f:
        f.write(f'ROW_TEST,measured,{RUN},"a test row: FFMA alone at its F rate"\n')
    _census(data, "TEST", f"FFMA,ROW_TEST,3.885,,,,TEST,,{RUN}")
    cur, _ = _epoch(data, ["ROW_TEST"])
    new, old = resolve("TEST", data), resolve(cur, data)
    assert new.failures == [] and len(new.vector.rows) == len(old.vector.rows) + 1
    assert "ROW_TEST" in new.census["FFMA"] and "ROW_TEST" not in old.census["FFMA"]
    assert (old.tau, old.margin) == (resolve(cur).tau, resolve(cur).margin)
    assert "JI3e_0" in resolve("VF3m-16", data).census["LDSM_X1"] and "JI3e_0" not in resolve("VF3j", data).census["LDSM_X1"]
    assert "JI3n_0" in resolve("VF3n-16", data).census["LDSM_X1"]
    assert "JI3o_0" in resolve("VF3p-16", data).census["LDSM_X1"]
    assert "JI3q_0" in resolve("VF3q-16", data).census["LDSM_X1"]
    assert "JI3s_0" in resolve("VF3r-16", data).census["LDSM_X1"]
    assert "JI3t_0" in resolve("VF3s-16", data).census["LDSM_X1"]
    assert not {"JI3e_0", "JI3n_0", "JI3o_0", "JI3q_0", "JI3s_0"} & set(resolve("VF3s-16", data).census["LDSM_X1"])
    assert vector("TEST", data).published == {}
    assert main(["--data", str(data), "solve", "TEST"]) == 0
    assert "published: nothing yet" in capsys.readouterr().out


def test_a_refit_closes_the_old_line(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    first = data / "census" / "VF.csv"
    c = first.read_text()
    old_line = "FFMA,F,3.885,,,,VF,,@census-sm120\n"
    assert c.count(old_line) == 1
    last = model(data).epochs[-1]
    first.write_text(c.replace(old_line, old_line.replace(",,@", f",{last},@")))
    _census(data, "TEST", f"FFMA,F,3.9,,,,TEST,,{RUN}")
    cur, _ = _epoch(data)
    new, old = resolve("TEST", data), resolve(cur, data)
    assert new.census["FFMA"]["F"] == new.model.cycle_u * 4 / 3.9
    assert old.census["FFMA"]["F"] == old.model.cycle_u * 4 / 3.885
    assert (old.tau, old.margin) == (resolve(cur).tau, resolve(cur).margin) and new.failures == []


def _census(data, epoch, *lines):
    """README "Add a measurement": the census lines from `epoch`, in its own file."""
    (data / "census" / f"{epoch}.csv").write_text("kind,row,rate,noise,cycles,u,since,until,runs\n"
                                                  + "".join(f"{x}\n" for x in lines))


def _probe(path, lines):
    """A probe's census.csv: each of `lines` is kind,row,rate,noise,cycles,u, measured by RUN."""
    path.write_text("kind,row,rate,noise,cycles,u,since,until,runs\n" + "".join(f"{x},,,{RUN}\n" for x in lines))
    return path


def test_adopt_keeps_the_faster_line(tmp_path, capsys):
    """README "Adopt a probe's census": each (kind, row) at the faster of its line and the measured one."""
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    cur, last = _epoch(data)
    before = census(model(data), last, data)
    probe = _probe(tmp_path / "census.csv", ["FFMA,F,3.9,0.02,,", "FFMA,issue,0.5,,,", "IMAD,ROW_ATOMS32,1.0,,,"])
    assert main(["--data", str(data), "adopt", str(probe), "TEST"]) == 0
    out = capsys.readouterr().out
    assert "kept (the measurement is slower)" in out and "2 of 3 lines adopted at TEST, 1 kept" in out
    m = model(data)
    after = census(m, "TEST", data)
    assert after["FFMA"]["F"] == m.cycle_u * 4 / (3.9 * 1.02) < before["FFMA"]["F"]
    assert after["FFMA"]["issue"] == before["FFMA"]["issue"] and census(m, last, data) == before
    assert "ROW_ATOMS32" in after["IMAD"] and "ROW_ATOMS32" not in before["IMAD"]
    assert [x.split(",")[:2] for x in (data / "census" / "TEST.csv").read_text().splitlines()[1:]] == [
        ["FFMA", "F"], ["IMAD", "ROW_ATOMS32"]]
    assert f"FFMA,F,3.885,,,,VF,{last},@census-sm120" in (data / "census" / "VF.csv").read_text()
    runs = {r: x for r, x in (line.split(",")[:3:2] for line in (data / "rows.csv").read_text().splitlines()[1:])}
    assert runs["F"].endswith(RUN) and runs["ROW_ATOMS32"].endswith(RUN) and RUN not in runs["issue"]
    with pytest.raises(SystemExit, match="already has a line from TEST"):
        adopt(probe, "TEST", data)


def test_adopt_refuses(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    last = model(data).epochs[-1]
    with pytest.raises(SystemExit, match="append 'NEXT' to model.toml's epochs first"):
        adopt(_probe(tmp_path / "a.csv", ["FFMA,F,3.9,,,"]), "NEXT", data)
    with pytest.raises(SystemExit, match="two lines for FFMA on F"):
        adopt(_probe(tmp_path / "b.csv", ["FFMA,F,3.9,,,", "FFMA,F,4.0,,,"]), last, data)
    with pytest.raises(SystemExit, match="add the row to rows.csv first"):
        adopt(_probe(tmp_path / "c.csv", ["FFMA,ROW_NONE,3.9,,,"]), last, data)
    (tmp_path / "d.csv").write_text(f"kind,row,rate,noise,cycles,u,since,until,runs\nFFMA,F,3.9,,,,VF,,{RUN}\n")
    with pytest.raises(SystemExit, match="a measured line has no epochs"):
        adopt(tmp_path / "d.csv", last, data)
