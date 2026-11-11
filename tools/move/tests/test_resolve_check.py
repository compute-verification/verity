"""`resolve_check.py`, on a toy Lean tree that `rename.py` moves by a word map of one `[[words_in]]` row: a declaration the
row renames and a file outside the row still spells is an issue, named by its file and line; the move with that file in
the row has none, and neither has a file's own declaration of the same spelling."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("resolve_check_under_test", HERE.parent / "resolve_check.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

TREE = {
    "Sec/lakefile.toml": 'name = "sec"\n',
    "Sec/Game.lean": "namespace Game\n\n/-- The period's run. -/\ndef periodRun (n : Nat) : Nat := n\n\nend Game\n",
    "Sec/Two.lean": "import Game\n\nnamespace Two\nopen Game\n\ntheorem two : periodRun 2 = 2 := rfl\n\nend Two\n",
    "Sec/TwoReg.lean": "import Game\n\nnamespace TwoReg\nopen Game\n\ntheorem reg : periodRun 3 = 3 := rfl\n\nend TwoReg\n",
    # its own `periodRun`, which no row renames
    "Sec/Pous.lean": "namespace Pous\n\ndef periodRun : Nat := 0\n\ntheorem own : periodRun = 0 := rfl\n\nend Pous\n",
    # a name under no namespace, and a file that spells nothing else of it
    "Sec/Grid.lean": "def horizon : Nat := 4\n",
    "Sec/Ticks.lean": "import Grid\n\ntheorem ticks : horizon = 4 := rfl\n",
}


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def moved(tmp_path: Path, words: dict[str, str], files: list[str]) -> tuple[Path, str, str]:
    """The toy tree and the move of a map whose one row renames ``words`` in ``files`` alone, as `rename.py` writes it: the
    tree, the move's parent and its commit."""
    root = tmp_path / "tree"
    for p, t in TREE.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(t)
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "toy")
    row = ", ".join(f"{json.dumps(old)} = {json.dumps(new)}" for old, new in words.items())
    word_map = tmp_path / "map.toml"
    word_map.write_text(f"[words]\n\n[[words_in]]\nfiles = {json.dumps(files)}\nwords = {{ {row} }}\n")
    subprocess.run([sys.executable, str(HERE.parent / "rename.py"), "--map", str(word_map), "--root", str(root)],
                   check=True, capture_output=True, text=True)
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "move")
    return root, git(root, "rev-parse", "HEAD~1").strip(), git(root, "rev-parse", "HEAD").strip()


def test_a_name_its_row_renames_and_a_file_outside_the_row_spells_is_an_issue(tmp_path, capsys):
    root, parent, commit = moved(tmp_path, {"periodRun": "monitoringPeriodRun"}, ["Sec/Game.lean", "Sec/Two.lean"])
    assert "def monitoringPeriodRun (n : Nat)" in (root / "Sec/Game.lean").read_text()
    assert "theorem two : monitoringPeriodRun 2 = 2" in (root / "Sec/Two.lean").read_text()
    assert (root / "Sec/TwoReg.lean").read_text() == TREE["Sec/TwoReg.lean"]            # outside the row: as it was
    assert M.main([parent, commit, "--root", str(root)]) == 1
    assert capsys.readouterr().out == (
        "4 of 6 files compared identifier by identifier\n"
        "Sec/TwoReg.lean:6: periodRun -> periodRun in TwoReg: want Game.monitoringPeriodRun, got None (unchanged file)\n"
        "1 issue(s)\n")
    assert M.main([parent, commit, "--root", str(root), "--json"]) == 1
    assert json.loads(capsys.readouterr().out) == [
        {"file": "Sec/TwoReg.lean", "line": 6, "base_tok": "periodRun", "head_tok": "periodRun", "ns": "TwoReg",
         "base": "Game.periodRun", "want": "Game.monitoringPeriodRun", "got": None, "projs": [], "changed_file": False}]


def test_the_move_with_that_file_in_the_row_has_no_issue(tmp_path, capsys):
    root, parent, commit = moved(tmp_path, {"periodRun": "monitoringPeriodRun"},
                                 ["Sec/Game.lean", "Sec/Two.lean", "Sec/TwoReg.lean"])
    assert "theorem reg : monitoringPeriodRun 3 = 3" in (root / "Sec/TwoReg.lean").read_text()
    assert (root / "Sec/Pous.lean").read_text() == TREE["Sec/Pous.lean"]    # its own declaration stands, and is no issue
    assert M.main([parent, commit, "--root", str(root)]) == 0
    assert capsys.readouterr().out == "4 of 6 files compared identifier by identifier\n0 issue(s)\n"
    assert M.main([parent, commit, "--root", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_a_file_that_spells_only_the_old_name_is_compared(tmp_path, capsys):
    """A renamed name under no namespace shares no component with its new name, so the file that still spells it is found
    by the old name alone."""
    root, parent, commit = moved(tmp_path, {"horizon": "gridTicks"}, ["Sec/Grid.lean"])
    assert (root / "Sec/Grid.lean").read_text() == "def gridTicks : Nat := 4\n"
    assert M.main([parent, commit, "--root", str(root)]) == 1
    assert capsys.readouterr().out == (
        "2 of 6 files compared identifier by identifier\n"
        "Sec/Ticks.lean:3: horizon -> horizon in <root>: want gridTicks, got None (unchanged file)\n"
        "1 issue(s)\n")
