"""`machine-model solve`: a vector re-solved with its check, and adding a measurement as data: a row, its census lines,
an epoch and a vector."""
from __future__ import annotations

import json
import shutil

import pytest

from verity_machine_model.cli import main
from verity_machine_model.model import DATA, device, devices, model, vector
from verity_machine_model.resolve import resolve

RUN = "r20261016-000000-0000"


def test_solve_prints_the_numbers_and_the_check(capsys):
    assert main(["--device", "sm_120", "solve", "VF3j-16"]) == 0
    out = capsys.readouterr().out
    assert "τ 0.0610   U/F_low 1.1563   margin 2.000" in out
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
    assert main(["--device", "sm_120", "solve", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["vector"] == model().current and d["failures"] == [] and d["unreproduced"] == []
    assert main(["--device", "sm_120", "solve", "VF3r-16", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert round(d["tau"], 4) == 0.0556 and d["failures"] == [] and d["unreproduced"] == []
    assert (round(d["uf_low"], 4), d["margin"]) == (1.1563, 2.0)
    assert d["floors"] == {"pi32": {"kinds": 85, "below": [], "absent": []},
                           "loads16": {"kinds": 15, "below": [], "absent": []},
                           "copy16": {"kinds": 83, "below": ["UNI64 1.00 a word < 16"], "absent": []}}
    assert d["copies"][0]["kind"] == "UNI64"
    uni = next(c for c in d["copies"] if c["kind"] == "UNI")
    assert uni["words"] == 32 and round(uni["price"], 2) == 29.59 and uni["measured"] > 32
    assert uni["what"].startswith("UR fan-out")
    assert main(["--device", "sm_120", "solve", "VF3p-16", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["floors"]["copy16"]["absent"] == VF3Q_LOADS
    assert main(["--device", "sm_120", "solve", "VF3l-16", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["floors"]["copy16"]["absent"] == MMA_FORMS + VF3Q_LOADS


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
    """DESIGN.md "Add a measurement": the census lines from `epoch`, in its own file."""
    (data / "census" / f"{epoch}.csv").write_text("kind,row,rate,noise,cycles,u,since,until,runs\n"
                                                  + "".join(f"{x}\n" for x in lines))


def test_solve_says_a_device_has_no_price_vector():
    """DESIGN.md "Devices": a device directory that no vector is solved on leaves out the price LP, so `solve` says so
    and exits instead of failing inside it."""
    unpriced = [d for d in devices() if not model(device(d)).adopted]
    assert unpriced
    for d in unpriced:
        with pytest.raises(SystemExit, match=f"^{d}: no price vector is solved on this device"):
            main(["--device", d, "solve"])
