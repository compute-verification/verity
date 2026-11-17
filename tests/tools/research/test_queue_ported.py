"""Drive's merges once the tree is on Lean's module system (`research.queue.ported`): git runs the `.lean` merge driver only on
a file both sides changed, so a `.lean` file either side added before the port stays unported; the merged tree's codemod
(`MODULES`, a stand-in here) runs over the whole merged tree, and what it rewrites is committed into the merge commit with each
pin recording it; a codemod that fails refuses the merge.  A tree holding the codemod is on the module system only when its
roots (`PORTED_ROOTS`) are modules as its header reader (`HEADERS`, a stand-in too) reads them."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research.queue import HEADERS, MODULES, PORTED_ROOTS
from test_queue import (EXFIL, POUS, TRUSTED, UP, World, _exfil, _git, _move_main, _no_worktree_left, _pin_main, _pinned,
                        _preflight, _preflight_main, _trusted)


def _stub(argv: Path, then: str = "") -> str:
    """The codemod in miniature, run at the root over the whole tree: each tracked `.lean` file that isn't a module gets
    `module` first and its imports made public (`_port`); it writes the arguments it was given to ``argv``, then runs
    ``then``."""
    return (f"import subprocess, sys\nfrom pathlib import Path\nPath({str(argv)!r}).write_text(' '.join(sys.argv[1:]))\n"
            "for f in subprocess.run(['git', 'ls-files', '*.lean'], capture_output=True, text=True, check=True).stdout.split():\n"
            "    p = Path(f)\n    s = p.read_text()\n"
            "    if not s.startswith('module'):\n        p.write_text('module\\n\\n' + s.replace('import ', 'public import '))\n"
            + then)


def _port(src: str) -> str:
    return "module\n\n" + src.replace("import ", "public import ")


#: the header reader in miniature: a file's header is a module's when the file starts with `module`
HEADERS_STUB = ("from typing import NamedTuple\n\n\nclass Header(NamedTuple):\n    module: bool\n\n\n"
                "def header(src):\n    return Header(src.startswith('module'))\n")
ROOT = "import Init\n"


def _tooling(argv: Path, then: str = "", unported: tuple[str, ...] = ()) -> dict[str, str]:
    """The port's tooling (the stand-in codemod, `_stub`, and header reader) and its roots, each a module but ``unported``."""
    return {MODULES: _stub(argv, then), HEADERS: HEADERS_STUB,
            **{r: ROOT if r in unported else _port(ROOT) for r in PORTED_ROOTS}}


def _show(world: World, ref: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=world.origin, check=True, capture_output=True, text=True).stdout


#: the stand-in's lakefile step: a package's lakefile without the option takes it
LAKE = ("p = Path('Lib/lakefile.toml')\nif p.exists() and 'proofsInPublic' not in p.read_text():\n"
        "    p.write_text(p.read_text() + 'backward.proofsInPublic = true\\n')\n")


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def test_the_ports_own_merge_ports_the_lean_checked_added_below_it(world, tmp_path):
    """The port's merge adds none of the `.lean` files `checked` added since the port's base, and git's merge leaves them as
    written: the codemod, run over the whole merged tree with no arguments, ports them and gives a package's lakefile the
    option, into the merge commit, with the port's head as its second parent; the log names each file.  A merge onto that
    tree changes nothing."""
    argv = tmp_path / "argv"
    port = world.change({**_tooling(argv, LAKE), "Lib/Port.lean": "module\n\npublic import Init\n"})
    old = "-- added on checked since the port's base\nimport Lib.Port\n"
    main = _move_main(world, {"Lib/Old.lean": old, "Lib/lakefile.toml": 'name = "lib"\n'})
    world.gh.put(1, port)
    world.gh.put(2, world.change({"b.txt": "b2\n"}))
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1, 2], {n: e.why for n, e in entries.items()}
    one = members[0].commit
    assert _show(world, one, "Lib/Old.lean") == _port(old)
    assert _show(world, one, "Lib/lakefile.toml") == 'name = "lib"\nbackward.proofsInPublic = true\n'
    assert _show(world, one, "Lib/Port.lean") == "module\n\npublic import Init\n" and argv.read_text() == ""
    assert _git(world.origin, "rev-parse", f"{one}^1", f"{one}^2").split() == [main, port]
    assert (f"research queue: #1's merged tree holds Lean made before the module system: Lib/Old.lean, Lib/lakefile.toml "
            f"ported by `{MODULES}`" in world.logs), world.logs
    assert [s for s in world.logs if "module system" in s and "#2" in s] == []
    assert _show(world, "refs/heads/next", "Lib/Old.lean") == _port(old)


def test_a_pr_adding_lean_made_before_the_port_above_a_ported_base_gets_it_ported_in_its_merge(world, tmp_path):
    """What the codemod rewrites is committed into the merge commit, with the PR's head as its second parent and its message,
    and the log names each file; a module the PR adds, and a merge adding no Lean, change nothing."""
    argv = tmp_path / "argv"
    _move_main(world, {**_tooling(argv), "Lib/Old.lean": "module\n\npublic import Init\n"})
    main = _git(world.origin, "rev-parse", "main")
    new, done = "-- made before the port\nimport Lib.Old\n\ndef x := 1\n", "module\n\npublic import Lib.Old\n"
    world.gh.put(1, head := world.change({"Lib/New.lean": new, "Lib/Done.lean": done, "a.txt": "a1\n"}))
    world.gh.put(2, world.change({"b.txt": "b2\n"}))
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1, 2], {n: e.why for n, e in entries.items()}
    one = members[0].commit
    assert _show(world, one, "Lib/New.lean") == _port(new) and _show(world, one, "Lib/Done.lean") == done
    assert _show(world, one, "a.txt") == "a1\n" and _show(world, one, "Lib/Old.lean") == "module\n\npublic import Init\n"
    assert argv.read_text() == ""
    assert (f"research queue: #1's merged tree holds Lean made before the module system: Lib/New.lean ported by `{MODULES}`"
            in world.logs), world.logs
    assert [s for s in world.logs if "module system" in s and "#2" in s] == []
    assert _git(world.origin, "rev-parse", f"{one}^1", f"{one}^2").split() == [main, head]
    assert _git(world.origin, "log", "-1", "--format=%B", one) == f"Queue: #1 change 1\n\nPR: #1\nHead: {head}\nOwner: lane-1"
    assert _show(world, "refs/heads/next", "Lib/New.lean") == _port(new)


def test_a_merge_changing_nothing_the_codemod_reads_onto_a_ported_base_runs_no_codemod(world, tmp_path):
    """The codemod's run over the whole tree takes about 20 s on drive's control pod, and a sync merges hundreds of times: a
    merge that changes no `.lean` file, Lake file or the port's tooling onto a tree on the module system takes that tree's
    porting as it is, and only the merge that brings Lean runs the codemod."""
    argv, runs = tmp_path / "argv", tmp_path / "runs"
    _move_main(world, {**_tooling(argv, f"open({str(runs)!r}, 'a').write('x')\n"),
                       "Lib/Old.lean": "module\n\npublic import Init\n"})
    new = "import Lib.Old\n"
    world.gh.put(1, world.change({"a.txt": "a1\n"}))
    world.gh.put(2, world.change({"Lib/New.lean": new}))
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1, 2], {n: e.why for n, e in entries.items()}
    assert runs.read_text() == "x", world.logs
    assert _show(world, members[1].commit, "Lib/New.lean") == _port(new)


def test_a_merge_made_again_onto_the_same_tree_takes_its_recorded_porting_without_a_codemod_run(world, tmp_path):
    """A rebuild of `next` in a later tick (a new process: no codemod run cached) merges each member again; one merged onto
    the tree it had takes its recorded merge, its porting and pins too, and runs no codemod; the log names the ported files
    as the merge did."""
    from research import queue as Q

    argv, runs = tmp_path / "argv", tmp_path / "runs"
    _move_main(world, {**_tooling(argv, f"open({str(runs)!r}, 'a').write('x')\n"),
                       "Lib/Old.lean": "module\n\npublic import Init\n"})
    world.gh.put(1, world.change({"Lib/New.lean": "import Lib.Old\n"}))
    members, _ = world.queue().sync()
    said = [s for s in world.logs if "module system" in s]
    assert runs.read_text() == "x" and said
    Q._codemod.cache_clear()
    _git(world.origin, "update-ref", "-d", "refs/heads/next")
    again, _ = world.queue().sync()
    assert runs.read_text() == "x" and [s for s in world.logs if "module system" in s] == said * 2
    assert _git(world.origin, "rev-parse", f"{again[0].commit}^{{tree}}") == _git(world.origin, "rev-parse",
                                                                                   f"{members[0].commit}^{{tree}}")


def test_each_pin_recording_a_ported_file_records_its_ported_digest_and_keeps_every_other_byte(world, tmp_path):
    """A pin records a Lean source's digest (PoUS's TRUSTED.sha256, a JSON pin with vectors after it), so each file the
    codemod ports, the PR's or one below it, moves its record with it."""
    tree = _pin_main(world)
    _move_main(world, _tooling(tmp_path / "argv"))
    pr = {**tree, f"{POUS}/New.lean": "import Old\n", "y.lean": "import x\n"}
    paths, exfil = ["lean-toolchain", "lakefile.toml", "Old.lean", UP, "New.lean"], ["security/Proofs/lakefile.toml", "x.lean", "y.lean"]
    world.gh.put(1, _pinned(world, pr, paths, exfil))
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1], entries[1].why
    merged = {**pr, **{f: _port(pr[f]) for f in (f"{POUS}/New.lean", f"{POUS}/Old.lean", "x.lean", "y.lean")}}
    assert _show(world, members[0].commit, f"{POUS}/New.lean") == merged[f"{POUS}/New.lean"]
    assert _show(world, members[0].commit, f"{POUS}/Old.lean") == merged[f"{POUS}/Old.lean"]
    assert _show(world, members[0].commit, TRUSTED) == _trusted(paths, merged)
    assert _show(world, members[0].commit, EXFIL) == _exfil(exfil, merged)
    assert (f"research queue: #1's merged tree holds Lean made before the module system: {POUS}/New.lean, {POUS}/Old.lean, "
            f"x.lean, y.lean ported by `{MODULES}`, and the pins {TRUSTED}, {EXFIL} rebuilt" in world.logs), world.logs


def test_a_pr_made_before_the_port_deleting_a_file_the_port_only_ported_merges_with_it_deleted(world, tmp_path):
    """Git runs no merge driver on a modify/delete, so a PR made before the port that deletes a `.lean` file conflicts with
    the port's rewrite of it.  Where the tip's text is what its codemod makes of the merge base's, the merge takes the
    deletion, as git's did before the port, and the log names it; a file the tip changed beyond porting it still conflicts."""
    from research.queue import conflicting_files

    old = "-- made before the port\nimport Init\n"
    _move_main(world, {"Lib/Gone.lean": old, "Lib/Edited.lean": old})
    world.gh.put(1, one := world.change({"Lib/Gone.lean": None, "a.txt": "a1\n"}))
    world.gh.put(2, world.change({"Lib/Edited.lean": None}))
    main = _move_main(world, {**_tooling(tmp_path / "argv"), "Lib/Gone.lean": _port(old),
                              "Lib/Edited.lean": _port(old) + "\ndef y := 2\n"})
    assert conflicting_files(world.dev, main, one) == []
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1], {n: e.why for n, e in entries.items()}
    assert _git(world.origin, "ls-tree", "--name-only", members[0].commit, "--", "Lib/Gone.lean") == ""
    assert _show(world, members[0].commit, "a.txt") == "a1\n"
    assert ("research queue: #1 deletes Lib/Gone.lean, which `next` changed only by porting it to the module system: deleted "
            "in the merge" in world.logs), world.logs
    assert entries[2].state == "refused" and entries[2].why == "conflicts with `next` in Lib/Edited.lean", entries[2].why


def test_the_port_merged_onto_a_tip_that_deleted_a_file_it_only_ported_merges_with_it_deleted(world, tmp_path):
    """The mirror: a head carrying the port merges onto a tip that, since the port's base, deleted a `.lean` file the head
    changed only by porting it.  Where the head's text is what its own codemod makes of the merge base's, the merge takes the
    tip's deletion and the log names it; a head that changed the file beyond porting it still conflicts."""
    from research.queue import conflicting_files

    old = "-- made before the port\nimport Init\n"
    _move_main(world, {"Lib/Gone.lean": old, "Lib/Edited.lean": old})
    tooling = _tooling(tmp_path / "argv")
    world.gh.put(1, one := world.change({**tooling, "Lib/Gone.lean": _port(old), "a.txt": "a1\n"}))
    world.gh.put(2, world.change({**tooling, "Lib/Edited.lean": _port(old) + "\ndef y := 2\n"}))
    main = _move_main(world, {"Lib/Gone.lean": None, "Lib/Edited.lean": None})
    assert conflicting_files(world.dev, main, one) == []
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1], {n: e.why for n, e in entries.items()}
    assert _git(world.origin, "ls-tree", "--name-only", members[0].commit, "--", "Lib/Gone.lean") == ""
    assert _show(world, members[0].commit, "a.txt") == "a1\n"
    assert ("research queue: `next` deletes Lib/Gone.lean, which #1 changed only by porting it to the module system: deleted "
            "in the merge" in world.logs), world.logs
    assert entries[2].state == "refused" and entries[2].why == "conflicts with `next` in Lib/Edited.lean", entries[2].why


def _criss_cross(world: World, old: str, edited: str) -> tuple[str, str]:
    """Two commits whose merge bases are two (a criss-cross history): ``a`` edits `Lib/Gone.lean` from ``old`` to ``edited``,
    ``b`` leaves it, and each side merges the other; their texts of the file are ``edited``."""
    base = world.change({"Lib/Gone.lean": old})
    a = world.change({"Lib/Gone.lean": edited, "a.txt": "a1\n"}, base=base)
    b = world.change({"b.txt": "b1\n"}, base=base)
    out = []
    for ours, theirs in ((a, b), (b, a)):
        _git(world.dev, "checkout", "-q", "--detach", ours)
        _git(world.dev, "merge", "-q", "--no-ff", "--no-edit", theirs)
        out.append(_git(world.dev, "rev-parse", "HEAD"))
    return out[0], out[1]


@pytest.mark.parametrize("deleter", ["tip", "head"])
def test_a_criss_cross_merge_takes_the_deletion_when_one_merge_base_ports_to_the_porting_sides_text(world, tmp_path, deleter):
    """A criss-cross history has more than one merge base, here two with different texts of the file.  The deletion is taken
    when the porting side's text is the port of one base's text, whichever side deleted it (`port_deletions`), and `taken`
    says which side that was."""
    from research.queue import merged_tree, taken

    old, edited = "-- made before the port\nimport Init\n", "-- edited before the port\nimport Init\n"
    ab, ba = _criss_cross(world, old, edited)
    porter = world.change({**_tooling(tmp_path / "argv"), "Lib/Gone.lean": _port(edited)}, base=ab)
    deleting = world.change({"Lib/Gone.lean": None}, base=ba)
    tip, head = (deleting, porter) if deleter == "tip" else (porter, deleting)
    assert len(_git(world.dev, "merge-base", "--all", tip, head).split()) == 2
    tree, files, _ = merged_tree(world.dev, tip, head)
    assert tree and files == ["Lib/Gone.lean"], files
    assert _git(world.dev, "ls-tree", "--name-only", tree, "--", "Lib/Gone.lean") == ""
    assert _git(world.dev, "show", f"{tree}:a.txt") == "a1" and _git(world.dev, "show", f"{tree}:b.txt") == "b1"
    assert taken(world.dev, tree, head, files) == (([], [], files) if deleter == "tip" else ([], files, []))


@pytest.mark.parametrize("deleter", ["tip", "head"])
def test_a_deletion_of_a_file_the_other_side_changed_beyond_porting_still_conflicts(world, tmp_path, deleter):
    """A porting side's text that no merge base's text ports to, a change beyond porting it, keeps the modify/delete conflict
    in either direction: the deletion would drop that change."""
    from research.queue import conflicting_files, merged_tree

    old, edited = "-- made before the port\nimport Init\n", "-- edited before the port\nimport Init\n"
    ab, ba = _criss_cross(world, old, edited)
    porter = world.change({**_tooling(tmp_path / "argv"), "Lib/Gone.lean": _port(edited) + "\ndef y := 2\n"}, base=ab)
    deleting = world.change({"Lib/Gone.lean": None}, base=ba)
    tip, head = (deleting, porter) if deleter == "tip" else (porter, deleting)
    assert merged_tree(world.dev, tip, head)[:2] == (None, ["Lib/Gone.lean"])
    assert conflicting_files(world.dev, tip, head) == ["Lib/Gone.lean"]


def test_a_base_without_the_port_merges_lean_files_as_the_pr_wrote_them(world):
    """Before the port lands the tree has no codemod, and the merge is git's alone."""
    new = "import Lib.Old\n"
    world.gh.put(1, world.change({"Lib/New.lean": new}))
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1], entries[1].why
    assert _show(world, members[0].commit, "Lib/New.lean") == new
    assert [s for s in world.logs if "module system" in s] == []


@pytest.mark.parametrize("unported", [PORTED_ROOTS, PORTED_ROOTS[1:]], ids=["no-root-ported", "one-root-unported"])
def test_a_tree_holding_the_codemod_before_the_port_merges_lean_files_as_the_pr_wrote_them(world, tmp_path, unported):
    """The port's tooling may land before the port: a tree holding the codemod whose roots aren't all modules isn't on the
    module system, so the codemod, which would port the whole tree there, never runs, and the merge is git's alone."""
    argv = tmp_path / "argv"
    _move_main(world, _tooling(argv, unported=unported))
    new = "import Lib.Old\n"
    world.gh.put(1, world.change({"Lib/New.lean": new}))
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1], entries[1].why
    assert _show(world, members[0].commit, "Lib/New.lean") == new
    assert all(_show(world, members[0].commit, r) == ROOT for r in unported)
    assert not argv.exists() and [s for s in world.logs if "module system" in s] == []


def test_a_header_reader_that_fails_refuses_the_merge(world, tmp_path):
    """Whether the tree is on the module system is the tree's header reader's to say: when it fails, the merge is left out with
    its words, not merged as git's."""
    _move_main(world, {**_tooling(tmp_path / "argv"), HEADERS: "import sys\nsys.exit('headers: broken')\n"})
    world.gh.put(1, world.change({"Lib/New.lean": "import Lib.Old\n"}))
    members, entries = world.queue().sync()
    assert members == [] and entries[1].state == "refused", entries[1].why
    assert entries[1].why == f"`{HEADERS}` failed to read the headers of {', '.join(PORTED_ROOTS)} (exit 1): headers: broken"


def test_a_codemod_that_fails_refuses_the_merge(world, tmp_path):
    """The PR is left out with the codemod's own words, as a conflict is (the state's `conflicts`, which drive posts to its
    owner), and the members above it merge."""
    fail = ("if Path('Lib/New.lean').exists():\n"
            "    print('Lib/New.lean: its header can\\'t be read: port it by hand', file=sys.stderr)\n    sys.exit(1)\n")
    _move_main(world, _tooling(tmp_path / "argv", fail))
    world.gh.put(1, world.change({"Lib/New.lean": "import Lib.Old\n"}))
    world.gh.put(2, world.change({"b.txt": "b2\n"}))
    q = world.queue()
    members, entries = q.sync()
    why = (f"`{MODULES}` failed on the merged tree (exit 1): Lib/New.lean: its header can't be read: port it by hand; port by "
           "hand what it names")
    assert [m.pr for m in members] == [2] and entries[1].state == "refused" and entries[1].why == why, entries[1].why
    assert q.load_state()["conflicts"]["1"]["why"] == why
    assert _show(world, "refs/heads/next", "b.txt") == "b2\n"


@pytest.mark.parametrize("then", ["", "if Path('pkg/New.lean').exists():\n    sys.exit(1)\n"], ids=["ports", "fails"])
def test_the_preflight_judges_the_merge_with_its_lean_ported_as_the_sync_ports_it(world, tmp_path, capsys, then):
    """A head marked after the port is checked, its Lean build included, on the tree drive's sync would merge, and a codemod
    that fails refuses the mark with its reason."""
    from research.queue import MarkOnly, StoreLabels, ready

    _preflight_main(world, _tooling(tmp_path / "argv", then))
    world.gh.put(2, h := world.change({"pkg/New.lean": "import Lib.Old\n"}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == (1 if then else 0)
    err = capsys.readouterr().err
    if then:
        assert f"  - `{MODULES}` failed on the merged tree (exit 1)" in err, err
    else:
        said = next(s for s in world.logs if "preflight: #2" in s)
        assert "nothing refused" in said and f"Lean made before the module system: pkg/New.lean ported by `{MODULES}`" in said, said
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h) is (not then) and _no_worktree_left(world)
