"""`suite_check.py` and `gone_dirs.py` on a toy tree: a map that moves a suite whole passes both, and each way a map
can go wrong is named with its path. Both read a commit and the maps' rows, so the toy tree is a commit and nothing in
it moves."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

MOVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOVE))


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", MOVE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOVED, SUITES, GONE = load("moved"), load("suite_check"), load("gone_dirs")

ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
TREE = {
    "tests/test_repository.py": 'UNSUITED_TEST_DIRS = ("old/kernels/",)\n',
    "old/kernels/test_gpu.py": "def test_gpu():\n    pass\n",
    "old/pkg/pyproject.toml": '[project]\nname = "pkg"\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
                              '[tool.verity.tests]\ninputs_only = true\ninputs = ["old/lib"]\n',
    "old/pkg/tests/conftest.py": "",
    "old/pkg/tests/test_a.py": "from old.lib import mod\n\n\ndef test_a():\n    assert mod\n",
    "old/pkg/tests/test_b.py": "def test_b():\n    pass\n",
    "old/pkg/tests/test_lister.py": "NAMES = []\n",                # named as a test file, defines no test
    "old/lib/__init__.py": "",
    "old/lib/mod.py": "from . import deep\n",
    "old/lib/deep.py": "",
    "app/pyproject.toml": '[project]\nname = "app"\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
    "app/tests/test_app.py": "def test_app():\n    pass\n",
    "docs.md": "Read old/lib and old/lib/mod.py, then every old/lib/*.py. The suite is old/pkg.\n",
}
ROOTS = '[roots]\nold = { old = "old" }\nnew = { new = "new" }\n'
WHOLE = ROOTS + '[paths]\n"old/pkg/" = "new/pkg/"\n"old/lib/" = "new/lib/"\n"old/kernels/" = "new/kernels/"\n'
#: the library file by file, one of them elsewhere
PARTS = ('"old/pkg/" = "new/pkg/"\n"old/kernels/" = "new/kernels/"\n"old/lib/__init__.py" = "new/lib/__init__.py"\n'
         '"old/lib/mod.py" = "new/lib/mod.py"\n"old/lib/deep.py" = "new/elsewhere/deep.py"\n')


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for name, text in TREE.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "toy"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, env={**os.environ, **ENV})
    return tmp_path


def moved(repo: Path, *maps: str, deleted: tuple[str, ...] = ()):
    files = []
    for i, text in enumerate(maps):
        files.append(repo / f"map{i}.toml")
        files[-1].write_text(text)
    return MOVED.Moved([str(f) for f in files], "HEAD", repo, deleted)


def suites(repo: Path, *maps: str) -> dict:
    m = moved(repo, *maps)
    try:
        return SUITES.check(m)
    finally:
        m.close()


def gone(repo: Path, *maps: str, deleted: tuple[str, ...] = ()) -> dict:
    m = moved(repo, *maps, deleted=deleted)
    try:
        return GONE.check(m)
    finally:
        m.close()


def test_a_suite_moved_whole_passes(repo: Path) -> None:
    r = suites(repo, WHOLE)
    assert (r["orphans"], r["parted"], r["inputs"]) == ([], [], [])
    assert gone(repo, WHOLE)["count"] == 0


def test_the_tree_as_it_is_passes(repo: Path) -> None:
    r = suites(repo)
    assert (r["orphans"], r["parted"], r["inputs"]) == ([], [], [])


def test_a_test_file_sent_elsewhere_is_in_no_suite_and_parted(repo: Path) -> None:
    rows = ('"old/pkg/pyproject.toml" = "new/pkg/pyproject.toml"\n"old/pkg/tests/conftest.py" = "new/pkg/tests/conftest.py"\n'
            '"old/pkg/tests/test_a.py" = "new/pkg/tests/test_a.py"\n"old/pkg/tests/test_b.py" = "new/other/tests/test_b.py"\n'
            '"old/pkg/tests/test_lister.py" = "new/other/tests/test_lister.py"\n"old/lib/" = "new/lib/"\n')
    r = suites(repo, ROOTS + "[paths]\n" + rows)
    assert [o["lands"] for o in r["orphans"]] == ["new/other/tests/test_b.py"]       # the lister defines no test
    assert {(x["test"], x["with_its_suite"]) for x in r["parted"]} == {
        ("old/pkg/tests/test_b.py", "new/pkg/tests/test_b.py"), ("old/pkg/tests/test_lister.py", "new/pkg/tests/test_lister.py")}


def test_a_suite_left_behind_by_its_tests_is_parted(repo: Path) -> None:
    r = suites(repo, ROOTS + '[paths]\n"old/pkg/tests/" = "new/pkg/tests/"\n')
    assert len(r["orphans"]) == 2 and len(r["parted"]) == 4
    assert {x["suite_lands"] for x in r["parted"]} == {"old/pkg/pyproject.toml"}


def test_an_unsuited_directory_follows_its_row(repo: Path) -> None:
    r = suites(repo, ROOTS + '[paths]\n"old/kernels/" = "new/kernels/"\n')
    assert r["orphans"] == []
    r = suites(repo, ROOTS + '[paths]\n"old/kernels/test_gpu.py" = "new/kernels/test_gpu.py"\n')
    assert [o["lands"] for o in r["orphans"]] == ["new/kernels/test_gpu.py"]          # nothing rewrites the exemption


def test_an_imported_module_outside_the_inputs_is_named(repo: Path) -> None:
    r = suites(repo, ROOTS + "[paths]\n" + PARTS + '[tokens]\n"old/lib/" = "new/lib/"\n')
    assert (r["orphans"], r["parted"]) == ([], [])
    assert [(x["module"], x["lands"], x["imported_by"], x["holds_it_after"]) for x in r["inputs"]] == [
        ("old/lib/deep.py", "new/elsewhere/deep.py", "old/lib/mod.py", "new/elsewhere")]


def test_an_import_inside_a_function_is_told_from_one_a_test_file_runs(repo: Path) -> None:
    rows = ROOTS + "[paths]\n" + PARTS + '[tokens]\n"old/lib/" = "new/lib/"\n'
    assert [(x["module"], x["at_import"]) for x in suites(repo, rows)["inputs"]] == [("old/lib/deep.py", True)]
    (repo / "old/lib/mod.py").write_text("def later():\n    from . import deep\n    return deep\n")
    subprocess.run(["git", "commit", "-qam", "lazy"], cwd=repo, check=True, env={**os.environ, **ENV})
    r = suites(repo, rows)
    assert [(x["module"], x["at_import"]) for x in r["inputs"]] == [("old/lib/deep.py", False)]
    assert r["reached"] == 2 and r["outside_today"] == []


def test_a_file_that_does_not_parse_is_named_and_one_that_is_not_text_reads_as_none(repo: Path) -> None:
    assert suites(repo, WHOLE)["unparsed"] == {}
    (repo / "old/lib/mod.py").write_text("from . import (\n")
    (repo / "old/lib/blob.bin").write_bytes(b"\xff\xfe\x00 not text")
    for args in (["add", "-A"], ["commit", "-q", "-m", "broken"]):
        subprocess.run(["git", *args], cwd=repo, check=True, env={**os.environ, **ENV})
    r = suites(repo, WHOLE)
    assert list(r["unparsed"]) == ["old/lib/mod.py"] and r["unparsed"]["old/lib/mod.py"].startswith("SyntaxError: ")
    assert r["reached"] == 1                    # the module itself, and nothing it would have imported
    m = moved(repo, WHOLE)
    try:
        assert m.read("old/lib/blob.bin") is None and m.read("docs.md") == TREE["docs.md"]
    finally:
        m.close()


def test_an_archived_module_a_test_that_stays_imports_is_named_once(repo: Path) -> None:
    rows = PARTS.replace("new/elsewhere/deep.py", "archive/old/lib/deep.py")
    r = suites(repo, ROOTS + "[paths]\n" + rows + '[tokens]\n"old/lib/" = "new/lib/"\n')
    assert r["inputs"] == []
    assert [(x["module"], x["imported_by"], x["suites"]) for x in r["loads_archive"]] == [
        ("old/lib/deep.py", "old/lib/mod.py", ["old/pkg/pyproject.toml"])]


def test_a_test_file_goes_to_archive_with_the_code_it_tests(repo: Path) -> None:
    for name, text in {"old/lib/extra.py": "", "old/pkg/tool.py": "",
                       "old/pkg/tests/test_extra.py": "from old.lib import extra\n\n\ndef test_extra():\n    assert extra\n",
                       "old/pkg/tests/test_tool.py": 'TOOL = "tool.py"                # run by path\n\n\ndef test_tool():\n    pass\n'}.items():
        (repo / name).write_text(text)
    for args in (["add", "-A"], ["commit", "-q", "-m", "more"]):
        subprocess.run(["git", *args], cwd=repo, check=True, env={**os.environ, **ENV})
    tests = ('"old/pkg/tests/test_extra.py" = "archive/old/pkg/tests/test_extra.py"\n'
             '"old/pkg/tests/test_tool.py" = "archive/old/pkg/tests/test_tool.py"\n')
    code = '"old/lib/extra.py" = "archive/old/lib/extra.py"\n"old/pkg/tool.py" = "archive/old/pkg/tool.py"\n'
    r = suites(repo, WHOLE + tests + code)
    assert (r["orphans"], r["parted"], r["inputs"], r["loads_archive"]) == ([], [], [], [])
    assert [x["test"] for x in r["archived_with_code"]] == ["old/pkg/tests/test_extra.py", "old/pkg/tests/test_tool.py"]
    r = suites(repo, WHOLE + tests)                                 # archived alone: the code they test stays
    assert r["orphans"] == [] and r["archived_with_code"] == []
    assert [x["lands"] for x in r["parted"]] == ["archive/old/pkg/tests/test_extra.py", "archive/old/pkg/tests/test_tool.py"]


def test_a_test_file_goes_into_another_suite_with_the_code_it_tests(repo: Path) -> None:
    test = ROOTS + '[paths]\n"old/pkg/tests/test_a.py" = "app/tests/test_a.py"\n'
    alone = suites(repo, test)
    assert alone["orphans"] == [] and [x["test"] for x in alone["parted"]] == ["old/pkg/tests/test_a.py"]
    r = suites(repo, test + '"old/lib/mod.py" = "app/app/mod.py"\n')
    assert r["orphans"] == r["parted"] == []
    assert [(x["test"], x["into"]) for x in r["sent_with_code"]] == [("old/pkg/tests/test_a.py", "app/pyproject.toml")]
    # it is the other suite's now: what it imports, and its own suite's inputs covered, must be in that suite's inputs
    assert [(x["suite"], x["module"]) for x in r["inputs"]] == [("app/pyproject.toml", "old/lib/deep.py")]


def test_a_directory_emptied_by_file_rows_is_counted_until_a_tokens_row_names_it(repo: Path) -> None:
    r = gone(repo, ROOTS + "[paths]\n" + PARTS)
    assert r["count"] == 3 and r["one_segment"] == ["old"]
    assert [(d["directory"], sorted(d["where"])) for d in r["directories"]] == [
        ("old/lib", ["docs.md:1", "docs.md:1", "old/pkg/pyproject.toml:7"])]         # the directory and the glob, not mod.py
    assert r["directories"][0]["files_go_to"] == ["new/lib (2)", "new/elsewhere (1)"]
    assert gone(repo, ROOTS + "[paths]\n" + PARTS + '[tokens]\n"old/lib/" = "new/lib/"\n')["count"] == 0


def test_a_package_level_whose_init_a_later_commit_deletes_is_gone(repo: Path) -> None:
    rows = ROOTS + "[paths]\n" + PARTS.replace('"old/lib/__init__.py" = "new/lib/__init__.py"\n', "")
    assert gone(repo, rows)["count"] == 0                           # `old/lib` still holds its `__init__.py`
    r = gone(repo, rows, deleted=("old/lib/__init__.py",))
    assert (r["count"], [d["directory"] for d in r["directories"]]) == (3, ["old/lib"])


def test_a_file_a_drop_row_deletes_is_gone(repo: Path) -> None:
    rows = ROOTS + '[drop]\npaths = ["old/kernels/test_gpu.py", "old/pkg/tests/test_b.py"]\n[paths]\n"old/pkg/tests/test_b.py" = "x/test_b.py"\n'
    assert suites(repo, rows)["orphans"] == suites(repo, rows)["parted"] == []
    r = gone(repo, rows)
    assert [(d["directory"], d["where"]) for d in r["directories"]] == [("old/kernels", ["tests/test_repository.py:1"])]


def test_a_tokens_row_to_a_directory_that_holds_nothing_fails(repo: Path) -> None:
    r = gone(repo, ROOTS + "[paths]\n" + PARTS + '[tokens]\n"old/lib/" = "new/library/"\n')
    assert r["tokens_to_nothing"] == ["old/lib/ = new/library/"]


def test_maps_apply_in_order_and_an_excluded_file_is_not_read(repo: Path) -> None:
    first = ROOTS + '[paths]\n"old/pkg/" = "new/pkg/"\n"old/kernels/" = "new/kernels/"\n'
    second = ROOTS + "[paths]\n" + PARTS.split("\n", 2)[2] + '[rewrite]\nexclude = ["docs.md"]\n'
    assert gone(repo, first, second)["count"] == 3                  # the first map rewrites docs.md, so it is read
    assert gone(repo, second)["directories"][0]["where"] == ["old/pkg/pyproject.toml:7"]


def test_the_commands_exit_1_with_the_paths(repo: Path, capsys: pytest.CaptureFixture) -> None:
    (repo / "bad.toml").write_text(ROOTS + "[paths]\n" + PARTS)
    (repo / "good.toml").write_text(WHOLE)
    for module in (SUITES, GONE):
        m = MOVED.Moved
        module.Moved = lambda maps, ref, deleted=(), m=m: m(maps, "HEAD", repo, deleted)
        try:
            assert module.main([str(repo / "good.toml")]) == 0
            assert module.main([str(repo / "bad.toml")]) == 1
        finally:
            module.Moved = m
    out = capsys.readouterr().out
    assert "old/lib/deep.py -> new/elsewhere/deep.py" in out and "old/pkg/pyproject.toml:7" in out
