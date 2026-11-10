"""`layout_maps.py`'s rule on a toy assignment: where a file goes, what a file the data doesn't know does, and which rows
a map gets. The maps of the real tree are checked by the script itself each time it writes them."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MOVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOVE))
spec = importlib.util.spec_from_file_location("layout_maps_for_tests", MOVE / "layout_maps.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

DATA = {
    "parts": ["verifier", "prover", "device"],
    "components": [
        {"id": "pouw", "to": "verity/protocols/pouw/verifier", "part": "verifier",
         "paths": ["verity/core/protocols/pouw/"]},
        {"id": "ops", "to": "verity/definitions", "part": "",
         "paths": ["verity/ml/circuits/ops/", "verity/ml/gates/a.py", "verity/ml/gates/b.circuit-check.py"]},
        {"id": "poub", "to": "verity/protocols/poub/verifier", "part": "verifier", "paths": ["verity/ml/gates/b.py"]},
        {"id": "pous", "to": "verity/protocols/pous/device", "part": "device", "paths": ["verity/core/devices/c/"]},
    ],
    "archived": ["benchmarks/old/"],
    "stays": ["tools/t.py", "verity/core/devices/__init__.py", "verity/ml/gates/__init__.py"],
}
TRACKED = ["benchmarks/old/run.py", "tools/t.py", "verity/core/devices/__init__.py", "verity/core/devices/c/d.py",
           "verity/core/protocols/pouw/audit.py", "verity/core/protocols/pouw/lean/lakefile.toml",
           "verity/ml/circuits/ops/x.py", "verity/ml/gates/__init__.py", "verity/ml/gates/a.py", "verity/ml/gates/b.py",
           "verity/ml/gates/b.circuit-check.py"]


@pytest.fixture(autouse=True)
def no_rows_of_the_real_tree(monkeypatch):
    for table, empty in (("EXCEPT", []), ("EXCEPT_MOVE", {}), ("KEEP", {}), ("NEW", {}), ("DROP", {})):
        monkeypatch.setattr(G, table, empty)


def plan(tracked=TRACKED, data=DATA, suites=()):
    """What `G.plan` gives: {file: (new path, component, move)}, with the emptied `__init__.py`s under "emptied"."""
    said = []
    dest, emptied, _ = G.plan(data, G.place(data, set(tracked), said.append), said.append, suites)
    return {**dest, "emptied": emptied}, said


def test_a_file_goes_to_its_components_directory_under_the_name_it_has_today():
    dest, _ = plan()
    assert dest["verity/core/protocols/pouw/audit.py"] == ("verity/protocols/pouw/verifier/audit.py", "pouw", "pouw")
    assert dest["verity/core/protocols/pouw/lean/lakefile.toml"][0] == "verity/protocols/pouw/verifier/lean/lakefile.toml"
    assert dest["verity/ml/circuits/ops/x.py"] == ("verity/definitions/ops/x.py", "ops", "definitions")
    assert dest["verity/core/devices/c/d.py"] == ("verity/protocols/pous/device/c/d.py", "pous", "pous")
    assert dest["benchmarks/old/run.py"] == ("archive/benchmarks/old/run.py", "archived", "archive")
    assert "tools/t.py" not in dest


def test_a_file_or_a_component_named_in_keep_stays(monkeypatch):
    monkeypatch.setattr(G, "KEEP", {"verity/ml/gates/a.py": "a file that stays", "pous": "a component that stays"})
    dest, _ = plan()
    assert "verity/ml/gates/a.py" not in dest and "verity/core/devices/c/d.py" not in dest
    assert dest["verity/ml/circuits/ops/x.py"][0] == "verity/definitions/ops/x.py"


def test_a_sidecar_goes_with_its_module_and_an_emptied_package_levels_init_stays():
    dest, said = plan()
    assert dest["verity/ml/gates/b.py"][0] == "verity/protocols/poub/verifier/gates/b.py"
    assert dest["verity/ml/gates/b.circuit-check.py"] == ("verity/protocols/poub/verifier/gates/b.circuit-check.py", "ops", "poub")
    assert any(s.startswith("with its module: verity/ml/gates/b.circuit-check.py") for s in said)
    # each stays and every other file under it leaves: no row, and it is named for the commit that deletes it
    assert dest["emptied"] == ["verity/core/devices/__init__.py", "verity/ml/gates/__init__.py"]
    assert "verity/core/devices/__init__.py" not in dest and "verity/ml/gates/__init__.py" not in dest
    kept, _ = plan([*TRACKED, "verity/ml/gates/stays.py"], {**DATA, "stays": [*DATA["stays"], "verity/ml/gates/stays.py"]})
    assert kept["emptied"] == ["verity/core/devices/__init__.py"]
    # archived while a module of its package is not: a row would rename the package, and every mention left, to `archive.…`
    archived = {**DATA, "stays": ["tools/t.py", "verity/ml/gates/__init__.py"], "archived": [*DATA["archived"], "verity/core/devices/__init__.py"]}
    assert plan(data=archived)[0]["emptied"] == dest["emptied"]


def test_a_suite_moves_whole_with_its_pyproject():
    data = {**DATA, "stays": [*DATA["stays"], "tools/pyproject.toml"],
            "components": [*DATA["components"], {"id": "kernels", "to": "verity/definitions", "part": "",
                                                 "paths": ["verity/core/protocols/pouw/tests/test_kernel.py"]},
                           {"id": "t", "to": "verity/definitions", "part": "", "paths": ["tools/tests/test_t.py"]}]}
    tracked = [*TRACKED, "verity/core/protocols/pouw/pyproject.toml", "verity/core/protocols/pouw/tests/test_kernel.py",
               "verity/core/protocols/pouw/tests/fixtures/k.json", "tools/pyproject.toml", "tools/tests/test_t.py"]
    suites = [("verity/core/protocols/pouw/pyproject.toml", ["verity/core/protocols/pouw/tests"]), ("tools/pyproject.toml", ["tools/tests"])]
    alone, _ = plan(tracked, data)
    assert alone["verity/core/protocols/pouw/tests/test_kernel.py"][0] == "verity/definitions/pouw/tests/test_kernel.py"
    dest, said = plan(tracked, data, suites)
    assert dest["verity/core/protocols/pouw/tests/test_kernel.py"] == ("verity/protocols/pouw/verifier/tests/test_kernel.py", "kernels", "pouw")
    assert dest["verity/core/protocols/pouw/tests/fixtures/k.json"][0] == "verity/protocols/pouw/verifier/tests/fixtures/k.json"
    assert ("with its suite: 1 file of kernels goes to verity/protocols/pouw/verifier/tests, not verity/definitions/pouw/tests") in said
    assert "tools/tests/test_t.py" in alone and "tools/tests/test_t.py" not in dest      # its suite stays, and so does it


def test_a_file_the_data_does_not_know_goes_with_its_directory_or_stops_the_script_by_name():
    dest, said = plan([*TRACKED, "verity/core/protocols/pouw/new.py", "verity/ml/circuits/ops/tests/test_new.py"])
    assert dest["verity/core/protocols/pouw/new.py"][0] == "verity/protocols/pouw/verifier/new.py"
    assert dest["verity/ml/circuits/ops/tests/test_new.py"][0] == "verity/definitions/ops/tests/test_new.py"
    assert not any(s.startswith("new since") for s in said)        # a directory that goes one way takes them along
    with pytest.raises(SystemExit, match=r"verity/ml/gates/new\.py: its directory's files are ops, poub, stays"):
        plan([*TRACKED, "verity/ml/gates/new.py"])
    alone = {**DATA, "stays": ["tools/t.py"]}
    dest, said = plan(["tools/t.py", "tools/u.py"], alone)
    assert "tools/u.py" not in dest and said[0] == ("new since the data was cut: tools/u.py goes with the other files of "
                                                    "its directory (stays)")
    dest, said = plan(["tools/t.py"], alone)
    assert "dropped, benchmarks/old/ is no longer tracked: a path of archived" in said


def test_a_whole_directory_is_one_row_and_a_shared_one_is_a_row_a_file():
    dirs = {"": ([], ["v"]), "v": ([], ["p", "g"]), "v/p": (["a.py"], ["lean"]), "v/p/lean": (["l.toml"], []),
            "v/g": (["a.py", "b.py"], [])}
    mine = {"v/p/a.py": "n/p/a.py", "v/p/lean/l.toml": "n/p/lean/l.toml", "v/g/a.py": "n/defs/a.py"}
    assert G.compress(mine, dirs) == {"v/p/": "n/p/", "v/g/a.py": "n/defs/a.py"}
    assert G.compress({**mine, "v/p/a.py": "n/q/a.py"}, dirs) == {"v/p/a.py": "n/q/a.py", "v/p/lean/": "n/p/lean/",
                                                                   "v/g/a.py": "n/defs/a.py"}
    assert G.compress({**mine, "v/p/a.py": "n/q/a.py"}, dirs, own=["v/p/a.py"]) == {
        "v/p/": "n/p/", "v/p/a.py": "n/q/a.py", "v/g/a.py": "n/defs/a.py"}
    assert G.compress({**mine, "v/p/lean/l.toml": "b/lean/l.toml"}, dirs, own=["v/p/lean/"]) == {
        "v/p/": "n/p/", "v/p/lean/": "b/lean/", "v/g/a.py": "n/defs/a.py"}
