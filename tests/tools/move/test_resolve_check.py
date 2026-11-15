"""`resolve_check.py` on toy Lean trees. Moved by `rename.py` under a word map of one `[[words_in]]` row: a declaration the
row renames and a file outside the row still spells is an issue, named by its file and line; the move with that file in
the row has none, and neither has a file's own declaration of the same spelling. On two commits written by hand: the
scopes the scan holds (`namespace`, `section`, `mutual` and each form of `end`), the imports it compares, the files it
can't pair, `_root_` and `--no-imports`."""
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
LAKE = {"Sec/lakefile.toml": 'name = "sec"\n'}
GAME = "namespace Game\n\ndef periodRun (n : Nat) : Nat := n\n\nend Game\n"
RENAMED = GAME.replace("periodRun", "monitoringPeriodRun")
MISSED = "periodRun -> periodRun in {}: want Game.monitoringPeriodRun, got None (unchanged file)"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout


def committed(root: Path, message: str) -> str:
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD").strip()


def moved(tmp_path: Path, words: dict[str, str], files: list[str]) -> tuple[Path, str, str]:
    """The toy tree and the move of a map whose one row renames ``words`` in ``files`` alone, as `rename.py` writes it: the
    tree, the move's parent and its commit."""
    root = tmp_path / "tree"
    for p, t in TREE.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(t)
    git(root, "init", "-q")
    parent = committed(root, "toy")
    row = ", ".join(f"{json.dumps(old)} = {json.dumps(new)}" for old, new in words.items())
    word_map = tmp_path / "map.toml"
    word_map.write_text(f"[words]\n\n[[words_in]]\nfiles = {json.dumps(files)}\nwords = {{ {row} }}\n")
    subprocess.run([sys.executable, str(HERE.parent / "rename.py"), "--map", str(word_map), "--root", str(root)],
                   check=True, capture_output=True, text=True)
    return root, parent, committed(root, "move")


def issues(tmp_path: Path, capsys, before: dict[str, str], after: dict[str, str | None], *flags: str) -> list[str]:
    """The issues the check prints of a move written by hand: ``before``'s files are the parent, and the commit holds
    ``after``'s over them (None: the file deleted)."""
    root = tmp_path / "tree"
    root.mkdir(parents=True)
    git(root, "init", "-q")
    commits = []
    for files in (LAKE | before, after):
        for p, t in files.items():
            (root / p).parent.mkdir(parents=True, exist_ok=True)
            if t is None:
                (root / p).unlink()
            else:
                (root / p).write_text(t)
        commits.append(committed(root, "toy"))
    rc = M.main([*commits, "--root", str(root), *flags])
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == f"{len(lines) - 2} issue(s)" and rc == (1 if len(lines) > 2 else 0)
    return lines[1:-1]


def held(text: str) -> dict[str, str]:
    """The namespace the scan holds at each identifier of ``text`` that starts with `at`."""
    scan = M.Scan(text, tokens=True)
    return {word: ".".join(scan.ctxs[ctx][0]) for _, word, ctx, _ in scan.tokens if word.startswith("at")}


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


def test_the_end_of_a_mutual_block_closes_nothing_of_the_namespace(tmp_path, capsys):
    """The file outside the row names `periodRun` after a `mutual` block, inside the namespace that declares it."""
    out = ("import Game\n\nnamespace Game\n\nmutual\ndef even : Nat → Bool\n  | 0 => true\n  | n + 1 => odd n\n"
           "def odd : Nat → Bool\n  | 0 => false\n  | n + 1 => even n\nend\n\ntheorem late : periodRun 1 = 1 := rfl\n\n"
           "end Game\n")
    assert issues(tmp_path, capsys, {"Sec/Game.lean": GAME, "Sec/Out.lean": out}, {"Sec/Game.lean": RENAMED}) == [
        "Sec/Out.lean:14: " + MISSED.format("Game")]
    assert held(out.replace("periodRun", "atLate")) == {"atLate": "Game"}


def test_an_end_closes_one_scope_for_each_component_it_names(tmp_path, capsys):
    """`end Er` after `namespace Game.Deep.Er` leaves `Game.Deep` open, where `periodRun` is still `Game`'s; `end
    Game.Deep` then closes both, and the same word after it names nothing at either commit."""
    out = ("import Game\n\nnamespace Game.Deep.Er\n\ndef helper : Nat := 0\n\nend Er\n\ntheorem mid : periodRun 1 = 1 := rfl\n\n"
           "end Game.Deep\n\ntheorem outside : periodRun 2 = 2 := rfl\n")
    assert issues(tmp_path, capsys, {"Sec/Game.lean": GAME, "Sec/Out.lean": out}, {"Sec/Game.lean": RENAMED}) == [
        "Sec/Out.lean:9: " + MISSED.format("Game.Deep")]
    text = "namespace A.B.C.D\nexample := atD\nend D\nexample := atC\nend B.C\nexample := atA\nend A\nexample := atRoot\n"
    assert held(text) == {"atD": "A.B.C.D", "atC": "A.B.C", "atA": "A", "atRoot": ""}


def test_an_end_that_names_nothing_closes_a_section_and_nothing_of_the_namespace():
    text = ("namespace A.B\n\nsection\nexample := atSection\nend\n\nexample := atFirst\n\nnoncomputable section\n"
            "example := atNoncomputable\nend\n\nexample := atSecond\n\nsection S\nexample := atNamed\nend S\n\n"
            "example := atThird\n\nend A.B\n\nexample := atRoot\n")
    assert held(text) == {"atSection": "A.B", "atFirst": "A.B", "atNoncomputable": "A.B", "atSecond": "A.B",
                          "atNamed": "A.B", "atThird": "A.B", "atRoot": ""}


def test_an_end_that_names_nothing_closes_a_mutual_and_nothing_of_the_namespace():
    text = ("namespace A.B\n\nmutual\ndef f : Nat := atFirst\ndef g : Nat := f\nend\n\nexample := atSecond\n\n"
            "section\nmutual\ndef h : Nat := atThird\nend\nexample := atFourth\nend\n\nexample := atFifth\n\nend B\n\n"
            "example := atSixth\n\nend A\n")
    assert held(text) == {"atFirst": "A.B", "atSecond": "A.B", "atThird": "A.B", "atFourth": "A.B", "atFifth": "A.B",
                          "atSixth": "A"}
    assert M.Scan(text).decls == ["A.B.f", "A.B.g", "A.B.h"]


def test_a_section_is_one_whatever_stands_before_the_word():
    """Lean's section header: an attribute, `public`, `noncomputable`, `meta`. The `end` of each closes it and no namespace."""
    for header in ("@[expose] public section", "public section", "public noncomputable section", "public meta section",
                   "@[expose] public noncomputable section Named"):
        end = "end Named" if header.endswith("Named") else "end"
        text = f"namespace A\n\n{header}\nexample := atInside\n{end}\n\nexample := atAfter\n\nend A\n\nexample := atRoot\n"
        assert held(text) == {"atInside": "A", "atAfter": "A", "atRoot": ""}, header


def test_an_end_that_names_no_open_scope_closes_nothing_and_is_an_issue(tmp_path, capsys):
    """An `end` whose name is not the scope's, and an `end` alone inside a namespace: Lean refuses both, so the scan has
    missed what they close. It keeps the namespace, and says in a file it compares that it has lost the place."""
    out = "import Game\n\nnamespace Game\n\nend Other\n\nend\n\ntheorem late : periodRun 1 = 1 := rfl\n\nend Game\n\nend Game\n"
    lost = "an `end` that closes no scope the scan holds open: it has lost this file's namespaces"
    assert issues(tmp_path, capsys, {"Sec/Game.lean": GAME, "Sec/Out.lean": out}, {"Sec/Game.lean": RENAMED}) == [
        f"Sec/Out.lean:5: {lost}", f"Sec/Out.lean:7: {lost}", "Sec/Out.lean:9: " + MISSED.format("Game"),
        f"Sec/Out.lean:13: {lost}"]
    assert M.Scan(out).stray == [5, 7, 13]
    assert held(out.replace("periodRun", "atLate")) == {"atLate": "Game"}


def test_an_import_of_a_module_the_move_renamed_is_an_issue_where_it_was_left(tmp_path, capsys):
    """The move renames the module `Game` to `Match` and its import in one file; the other, which the move did not change,
    still imports `Game`, which is no file now. Its names are as they were, so only the import tells."""
    two = "import Game\n\nopen Game\n\ntheorem two : periodRun 2 = 2 := rfl\n"
    before = {"Sec/Game.lean": GAME, "Sec/Two.lean": two, "Sec/Out.lean": "-- outside the row\nimport Game\n"}
    after = {"Sec/Game.lean": None, "Sec/Match.lean": GAME, "Sec/Two.lean": two.replace("import Game", "import Match")}
    assert issues(tmp_path, capsys, before, after) == [
        "Sec/Out.lean:2: import Game -> Game: want Sec/Match.lean, got None (unchanged file)"]


def test_an_import_that_now_reads_another_package_s_module_of_the_name_is_an_issue(tmp_path, capsys):
    before = {"Sec/Game.lean": GAME, "Sec/Out.lean": "import Game\n", "Other/lakefile.toml": 'name = "other"\n',
              "Other/Game.lean": "def other : Nat := 0\n"}
    after = {"Sec/Game.lean": None, "Sec/Match.lean": GAME}
    assert issues(tmp_path, capsys, before, after) == [
        "Sec/Out.lean:1: import Game -> Game: want Sec/Match.lean, got Other/Game.lean (unchanged file)"]


def test_a_lean_file_with_none_paired_to_it_at_the_other_commit_is_an_issue(tmp_path, capsys):
    """A file the move deletes and a file it adds, too unlike for git to pair as a rename."""
    before = {"Sec/Game.lean": GAME, "Sec/Pous.lean": TREE["Sec/Pous.lean"]}
    after = {"Sec/Pous.lean": None, "Sec/Fresh.lean": "/-- Nothing of the file that went. -/\ndef fresh : Nat := 1\n"}
    assert issues(tmp_path, capsys, before, after) == [
        "Sec/Fresh.lean: a Lean file at the commit with none paired to it at the parent",
        "Sec/Pous.lean: a Lean file at the parent with none paired to it at the commit"]


def test_a_move_that_renames_and_changes_more_files_than_git_s_rename_limit_pairs_each_one(tmp_path, capsys):
    """Past its default limit git skips the inexact rename detection, and each file a move renamed and changed is then
    one deleted and one added. The move renames 1,100 files and the word `periodRun` in each, and leaves the word in one
    line of one of them: every file is paired, and that line is the one issue."""
    def text(i: int, first: str, second: str) -> str:
        return (f"import Game\n\n/-- File {i:04} of the run. Its own lines, {i * 7919} and {i * 104729}, are as the move\n"
                f"found them, so the file is most like itself. -/\nnamespace Run{i:04}\nopen Game\n\n"
                f"theorem first{i:04} : {first} {i} = {i} := rfl\n\ntheorem second{i:04} : {second} {i} = {i} := rfl\n\n"
                f"end Run{i:04}\n")
    renamed, left = range(1100), 707
    before = {"Sec/Game.lean": GAME} | {f"Sec/Run{i:04}.lean": text(i, "periodRun", "periodRun") for i in renamed}
    after = {"Sec/Game.lean": RENAMED} | {f"Sec/Run{i:04}.lean": None for i in renamed} | {
        f"Sec/Span{i:04}.lean": text(i, "monitoringPeriodRun", "periodRun" if i == left else "monitoringPeriodRun")
        for i in renamed}
    assert issues(tmp_path, capsys, before, after) == [
        "Sec/Span0707.lean:10: periodRun -> periodRun in Run0707: want Game.monitoringPeriodRun, got None"]


def test_a_file_whose_imports_declarations_or_identifiers_differ_in_number_is_an_issue(tmp_path, capsys):
    two = "import Game\n\nopen Game\n\ntheorem two : periodRun 2 = 2 := rfl\n"
    before = {"Sec/Game.lean": GAME, "Sec/Two.lean": two, "Sec/Three.lean": "import Game\n"}
    after = {"Sec/Game.lean": GAME + "\ndef more : Nat := 1\n", "Sec/Two.lean": two.replace(":= rfl", ":= id rfl"),
             "Sec/Three.lean": "import Game\nimport Two\n"}
    assert issues(tmp_path, capsys, before, after) == [
        "Sec/Game.lean: 1 declarations at the parent, 2 at the commit: they can't be paired",
        "Sec/Game.lean: 5 identifiers at the parent, 7 at the commit: they can't be paired",
        "Sec/Three.lean: 1 imports at the parent, 2 at the commit: they can't be paired",
        "Sec/Two.lean: 3 identifiers at the parent, 4 at the commit: they can't be paired"]


def test_root_names_a_declaration_from_the_root(tmp_path, capsys):
    """Inside `Pous`, `horizon` is `Pous`'s own and `_root_.horizon` the one the move renames; `def _root_.Grid.span`
    declares `Grid.span`, not `Pous.Grid.span`."""
    pous = ("import Grid\n\nnamespace Pous\n\ndef horizon : Nat := 0\n\ntheorem own : horizon = 0 := rfl\n\n"
            "theorem wide : _root_.horizon = 4 := rfl\n\ndef _root_.Grid.span : Nat := 8\n\nend Pous\n")
    assert M.Scan(pous).decls == ["Pous.horizon", "Pous.own", "Pous.wide", "Grid.span"]
    before = {"Sec/Grid.lean": "def horizon : Nat := 4\n", "Sec/Pous.lean": pous}
    assert issues(tmp_path, capsys, before, {"Sec/Grid.lean": "def gridTicks : Nat := 4\n"}) == [
        "Sec/Pous.lean:9: _root_.horizon -> _root_.horizon in Pous: want gridTicks, got None (unchanged file)"]


def test_no_imports_counts_a_declaration_the_file_s_imports_do_not_reach(tmp_path, capsys):
    """`Loose.lean` opens `Game` and imports nothing: by its imports `periodRun` named nothing before the move either."""
    before = {"Sec/Game.lean": GAME,
              "Sec/Loose.lean": "namespace Loose\nopen Game\n\ntheorem loose : periodRun 1 = 1 := rfl\n\nend Loose\n"}
    assert issues(tmp_path, capsys, before, {"Sec/Game.lean": RENAMED}) == []
    assert issues(tmp_path / "anywhere", capsys, before, {"Sec/Game.lean": RENAMED}, "--no-imports") == [
        "Sec/Loose.lean:4: " + MISSED.format("Loose")]
