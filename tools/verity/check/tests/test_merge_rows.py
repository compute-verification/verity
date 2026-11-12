"""`tools/verity/check/merge_rows.py`: a Markdown table's rows merged under their first cells, as a function and as git's
merge driver under the command the tree registers (`research.clone.RECORD_DRIVER`)."""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

from research.clone import RECORD_DRIVER

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("merge_rows_under_test", ROOT / "tools" / "verity" / "check" / "merge_rows.py")
MR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MR)

HEAD = ["| Directory | What it measures | Status |\n", "| --- | --- | --- |\n"]
A, B, C, D = (f"| `{c}/` | what {c} measures | maintained |\n" for c in "abcd")
PAGE = "# Index\n\nEvery directory has one row.\n\n{}\nThe rules follow.\n"


def test_two_prs_that_each_add_a_row_at_one_place_merge_in_ours_order():
    assert MR.rows([A], [A, B], [A, C]) == [A, C, B]
    assert MR.rows([A, D], [A, B, D], [A, C, D]) == [A, C, B, D]
    assert MR.rows([], [B], [C]) == [B, C]


def test_a_row_one_side_changed_removed_or_renamed_takes_that_change():
    changed = "| `a/` | what a measures now | frozen |\n"
    assert MR.rows([A, B], [A, B, C], [changed, B]) == [changed, B, C]
    assert MR.rows([A, B], [A, B, C], [B]) == [B, C]
    assert MR.rows([A, B], [changed, B], [A, D]) == [changed, D], "theirs renamed b/ to d/, which stays in its place"
    assert MR.rows([A, B, C], [A, B, C, D], [A, changed.replace("`a/`", "`x/`"), B]) == [
        A, changed.replace("`a/`", "`x/`"), B, D], "theirs dropped c/ and put x/ above b/, which is no rename of b/"
    assert MR.rows([A, B, C], [A, D, C], [A, B, changed.replace("`a/`", "`c/`")]) == [A, D, changed.replace("`a/`", "`c/`")]


def test_a_row_set_two_ways_and_a_line_that_is_no_row_stay_gits_conflict():
    ours, theirs = "| `a/` | ours' text | maintained |\n", "| `a/` | theirs' text | maintained |\n"
    assert MR.rows([A], [ours], [theirs]) is None, "changed on both sides, differently"
    assert MR.rows([A, B], [B], [theirs, B]) is None, "removed on one side and changed on the other"
    assert MR.rows([], [ours], [theirs]) is None, "added on both sides, differently"
    assert MR.rows([A], [A, B, B], [A, C]) is None, "a first cell one side holds twice"
    assert MR.rows([A], [A, "Some prose.\n"], [A, C]) is None
    assert MR.rows([A], [A, "- an item\n"], [A, C]) is None
    assert MR.rows([A], [A, B.rstrip("\n")], [A, C]) is None, "a last row with no newline can't move"


def test_a_row_both_sides_rewrote_under_a_new_first_cell_is_not_kept_twice():
    """A table whose first cell is prose: each side edits that cell, so the base's row is gone on both and each brings a
    row the other lacks.  Keeping both would hold the row twice."""
    base, ours, theirs = ("| The watcher {} | each loop |\n".format(s) for s in ("runs", "runs every loop", "runs and reports"))
    assert MR.rows([base], [ours], [theirs]) is None
    assert MR.rows([base, A], [ours, A], [ours, A, C]) == [ours, A, C], "the same rewrite on both sides is one row"


def test_a_row_only_one_side_moved_stays_gits_conflict():
    """A move is a change.  The result is in ours' order, so a row theirs alone moved would stay where ours has it with no one
    told; one ours alone moved would be kept, and lost when the same two changes merge on the other branch."""
    assert MR.rows([A, B], [A, B, C], [B, A]) is None, "theirs reorders while ours adds"
    assert MR.rows([A, B], [B, A], [A, B, C]) is None, "ours reorders while theirs adds"
    assert MR.rows([A, B, C], [B, A, C], [A, C, B]) is None, "both reorder, differently"
    assert MR.rows([], [A, B], [B, A]) is None, "both add the same rows, in different orders"
    assert MR.rows([A, B, C], [A, B], [C, A, B]) is None, "theirs moved the row ours removed"
    assert MR.rows([A, B, C], [C, A, B, D], [A, B]) is None, "ours moved the row theirs removed"


def test_a_move_both_sides_made_the_same_way_merges():
    assert MR.rows([A, B], [B, A, C], [B, A]) == [B, A, C], "and ours adds a row"
    assert MR.rows([A, B], [B, A], [B, A, C]) == [B, A, C], "and theirs adds a row"
    assert MR.rows([A, B, D], [B, A], [B, A, D, C]) == [B, A, C], "and ours removes a row neither moved"


def test_a_conflict_that_holds_the_row_under_a_header_stays_gits_conflict():
    """One side gave every row a new column while the other added a row, or each side added a table at one place: merged by
    rows, the first is a table with a short row and the second one table under two headers."""
    wider = [row[:-1] + f" {cell} |\n" for row, cell in zip([*HEAD, A], ("Owner", ":-:", "top"))]
    assert MR.rows([*HEAD, A], wider, [*HEAD, A, C]) is None
    assert MR.rows([], [*HEAD, A], ["| Name | What it holds | Unit |\n", HEAD[1], C]) is None
    assert MR.rows([A], [A, B], [A, "| - | - | - |\n"]) is None, "a row of dashes alone reads as the row under a header"


def _sides(tmp_path: Path, base: list[str], ours: list[str], theirs: list[str]) -> tuple[Path, Path, Path]:
    paths = []
    for name, table in (("base", base), ("ours", ours), ("theirs", theirs)):
        (p := tmp_path / name).write_text(PAGE.format("".join(HEAD + table)))
        paths.append(p)
    return tuple(paths)


def test_the_driver_writes_the_merge_to_ours_and_leaves_gits_markers_where_it_refuses(tmp_path):
    base, ours, theirs = _sides(tmp_path, [A], [A, B], [A, C])
    assert MR.main([str(base), str(ours), str(theirs), "benchmarks/README.md"]) == 0
    assert ours.read_text() == PAGE.format("".join(HEAD + [A, B, C]))
    base, ours, theirs = _sides(tmp_path, [A], [A.replace("what a", "all a")], [A.replace("what a", "how a"), C])
    assert MR.main([str(base), str(ours), str(theirs)]) == 1
    text = ours.read_text()
    assert "<<<<<<< ours\n" in text and ">>>>>>> theirs\n" in text and text.count("| `a/` |") == 3


def test_the_driver_leaves_a_conflict_that_holds_half_of_a_move_to_git(tmp_path):
    """Theirs moves a/ below b/ while ours adds c/ at the end: git takes a/ out above with no conflict, and the conflict
    holds the two rows added below b/, one of them a row the base holds above.  And ours moves a/ below b/ while theirs
    removes it and adds c/: merged by rows, the page would hold the row theirs removed."""
    base, ours, theirs = _sides(tmp_path, [A, B], [A, B, C], [B, A])
    assert MR.main([str(base), str(ours), str(theirs)]) == 1
    assert ours.read_text().count("<<<<<<< ours\n") == 1
    base, ours, theirs = _sides(tmp_path, [A, B], [B, A], [B, C])
    assert MR.main([str(base), str(ours), str(theirs)]) == 1
    assert ours.read_text().count("<<<<<<< ours\n") == 1


def test_the_driver_leaves_a_row_the_table_would_hold_twice_to_git(tmp_path):
    """Each side adds c/, at different places: git puts theirs' above with no conflict, and the conflict holds ours'."""
    base, ours, theirs = _sides(tmp_path, [A, B], [A, C], [C, A])
    assert MR.main([str(base), str(ours), str(theirs)]) == 1
    text = ours.read_text()
    assert text.count("<<<<<<< ours\n") == 1 and text.count("| `c/` |") == 2


def test_the_driver_leaves_rows_each_side_moved_its_own_way_as_gits_conflict(tmp_path):
    """Ours swaps a/ and b/ while theirs moves b/ below c/: the conflict holds ours' swap beside theirs' b/ removed.  Merged
    by rows, the page would be in theirs' order, and ours' move dropped with no one told."""
    base, ours, theirs = _sides(tmp_path, [A, B, C], [B, A, C], [A, C, B])
    assert MR.main([str(base), str(ours), str(theirs)]) == 1
    assert ours.read_text().count("<<<<<<< ours\n") == 1
    base, ours, theirs = _sides(tmp_path, [A, B, C], [A, C, B], [B, A, C])
    assert MR.main([str(base), str(ours), str(theirs)]) == 1, "and with the sides swapped"


def test_prose_both_sides_rewrote_stays_a_conflict_beside_rows_that_merge(tmp_path):
    base, ours, theirs = _sides(tmp_path, [A], [A, B], [A, C])
    ours.write_text(ours.read_text().replace("The rules follow.", "The rules follow, as ours states them."))
    theirs.write_text(theirs.read_text().replace("The rules follow.", "The rules follow, as theirs states them."))
    assert MR.main([str(base), str(ours), str(theirs)]) == 1
    text = ours.read_text()
    assert "".join(HEAD + [A, B, C]) in text and text.count("<<<<<<< ours\n") == 1 and "as theirs states them" in text


def test_a_file_whose_own_lines_look_like_markers_is_left_to_git(tmp_path):
    """The conflicts found must be the ones git counts: a page that quotes a conflict is git's text, markers and all."""
    quoted = "<<<<<<< ours\n| `x/` | quoted | data |\n||||||| base\n=======\n>>>>>>> theirs\n"
    base, ours, theirs = _sides(tmp_path, [A], [A, B], [A, C])
    for p in (base, ours, theirs):
        p.write_text(p.read_text() + quoted)
    assert MR.main([str(base), str(ours), str(theirs)]) == 1
    assert ours.read_text().count("<<<<<<< ours\n") == 2


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "HOME": str(repo.parent),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=check, capture_output=True, text=True,
                          env=env)


def test_git_merges_two_branches_that_each_add_a_row_with_the_registered_driver(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "check").mkdir(parents=True)
    (repo / "tools" / "verity" / "check" / "merge_rows.py").write_text((ROOT / "tools/verity/check/merge_rows.py").read_text())
    (repo / ".gitattributes").write_text("*.md merge=markdown-rows\n")
    m = repo / "benchmarks" / "README.md"
    m.parent.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    for name, table in (("main", [A]), ("a", [A, B]), ("b", [A, C])):
        if name != "main":
            _git(repo, "checkout", "-q", "-b", name, "main")
        m.write_text(PAGE.format("".join(HEAD + table)))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)
    _git(repo, "checkout", "-q", "a")
    # no driver set yet: git's own text merge
    _git(repo, "merge", "--no-edit", "b", check=False)
    assert _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split() == ["benchmarks/README.md"]
    _git(repo, "merge", "--abort")
    for k, v in RECORD_DRIVER.items():
        if k.startswith("merge.markdown-rows."):
            _git(repo, "config", k, v)
    _git(repo, "merge", "--no-edit", "b")
    assert m.read_text() == PAGE.format("".join(HEAD + [A, B, C]))
