"""`machine-model render` prints every table of a data directory, and the README holds none: a table there is a line
every new vector, design or census row edits, and two changes that each add one conflict on it."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from verity_machine_model import eps as E
from verity_machine_model.cli import main
from verity_machine_model.commands.render import SECTIONS, render
from verity_machine_model.model import device, devices, model, rows, vectors

README = Path(__file__).resolve().parents[1] / "README.md"
TABLE_ROW = re.compile(r"\s*\|.*\|\s*$")
CELL = re.compile(r"(?<!\\)\|")


def test_the_readme_holds_no_table():
    text = README.read_text()
    found = [f"README.md:{i}: {line}" for i, line in enumerate(text.splitlines(), 1) if TABLE_ROW.match(line)]
    assert not found, ("a table of the data belongs in `machine-model render` (verity_machine_model/commands/"
                       "render.py, one section a table, read from data/), and the README names the section:\n"
                       + "\n".join(found))
    assert "machine-model render" in text


@pytest.mark.parametrize("name", devices())
def test_render_tabulates_the_whole_directory(name):
    d = device(name)
    out = render(d)
    assert out.startswith("# `machine-model render`: data/")
    tables = [t for t in re.split(r"\n\n+", out) if t.startswith("| ")]
    assert tables
    for t in tables:
        lines = t.splitlines()
        assert re.fullmatch(r"\|(---\|)+", lines[1])
        assert {len(CELL.findall(line)) for line in lines} == {lines[1].count("|")}, t.splitlines()[0]
    assert all(f"| `{n}` |" in out for n in vectors(d))
    assert all(f"| `{n}` |" in out for n in E.designs(d))
    assert all(f"| `{n}` |" in out for n in E.profiles(d))
    assert all(f"| `{r}` |" in out for r in rows(d))
    assert f"| `{model(d).current}` |" in out and out.count("| current |") == 1


def test_render_prints_the_sections_named(capsys):
    assert main(["render", "vectors", "floors"]) == 0
    out = capsys.readouterr().out
    assert re.findall(r"^## (.*)", out, re.M) == [SECTIONS["vectors"][0], SECTIONS["floors"][0]]
    assert main(["--device", "sm_90", "render", "designs"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# `machine-model render`: data/devices/sm_90/") and "| `dither-c-l32` |" in out
    with pytest.raises(SystemExit, match="no section nope"):
        main(["render", "nope"])
