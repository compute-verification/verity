"""`machine-model adopt`: a probe's census merged at the last epoch, each (kind, row) at its faster line."""
from __future__ import annotations

import shutil

import pytest

from verity_machine_model.adopt import adopt
from verity_machine_model.cli import main
from verity_machine_model.model import DATA, census, model, vector

RUN = "r20261016-000000-0000"


def _epoch(data, rows=()):
    """DESIGN.md "Add a measurement": the epoch TEST after the last, and a vector TEST on the current vector's rows and
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


def _probe(path, lines):
    """A probe's census.csv: each of `lines` is kind,row,rate,noise,cycles,u, measured by RUN."""
    path.write_text("kind,row,rate,noise,cycles,u,since,until,runs\n" + "".join(f"{x},,,{RUN}\n" for x in lines))
    return path


def test_adopt_keeps_the_faster_line(tmp_path, capsys):
    """DESIGN.md "Adopt a probe's census": each (kind, row) at the faster of its line and the measured one."""
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
