"""benchmarks/pouw's contents come from `contents/`, one file an entry, and the README holds no table: a table there is
a line every new benchmark edits, and two PRs that each add one conflict on it."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pouw_contents", POUW / "contents.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

PATH_COLUMNS = {"paths": ("path",), "filler": ("files",), "kernels": ("files", "includes")}


def test_the_readme_holds_no_table():
    text = (POUW / "README.md").read_text()
    found = [f"README.md:{i}: {line}" for i, line in enumerate(text.splitlines(), 1)
             if re.match(r"\s*\|.*\|\s*$", line)]
    assert not found, ("an entry is a file in contents/TABLE/ (contents.py prints the tables), and the README points "
                       "to it:\n" + "\n".join(found))
    assert "contents.py" in text


def paths(table: str, cell: str):
    """Each backticked path a cell names; one without a `/` is beside the last that had one (the filler's in
    filler/)."""
    base = here = POUW / "filler" if table == "filler" else POUW
    for tok in re.findall(r"`([^`]+)`", cell):
        p = (base if "/" in tok else here) / tok
        if "/" in tok:
            here = p if tok.endswith("/") else p.parent
        yield tok, p


@pytest.mark.parametrize("table", list(C.TABLES))
def test_every_entry_names_paths_the_tree_has(table):
    es = C.entries(table)
    assert es and set(PATH_COLUMNS) == set(C.TABLES)
    missing = [f"{tok} ({e[col][:60]})" for e in es for col in PATH_COLUMNS[table] for tok, p in paths(table, e[col])
               if not (any(p.parent.glob(p.name)) if "*" in tok else p.exists())]
    assert not missing, f"contents/{table}/ names paths the tree doesn't have: {missing}"
    assert C.table(table).count("\n") == len(es) + 3


def test_contents_refuses_an_unknown_table():
    with pytest.raises(SystemExit, match="no table nope"):
        C.main(["nope"])
