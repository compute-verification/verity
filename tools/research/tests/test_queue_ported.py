"""Drive's merges once the tree has Lean's module system (`research.queue.ported`): git runs the `.lean` merge driver only on a
file both sides changed, so each `.lean` file a merge adds that isn't a module takes the port from the merged tree's codemod
(`MODULES`, a stand-in here), committed into the merge commit with each pin recording it; a codemod that fails, or rewrites a
file the merge doesn't add, refuses the merge."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research.queue import MODULE_HEADER, MODULES
from test_queue import (EXFIL, POUS, TRUSTED, UP, World, _exfil, _git, _move_main, _no_worktree_left, _pin_main, _pinned,
                        _preflight, _preflight_main, _trusted)


def _stub(argv: Path, then: str = "") -> str:
    """The codemod in miniature, run at the root on the files it names: each that isn't a module gets `module` first and its
    imports made public (`_port`); it writes the paths it was given to ``argv``, then runs ``then``."""
    return (f"import sys\nfrom pathlib import Path\nPath({str(argv)!r}).write_text(' '.join(sys.argv[1:]))\n"
            "for f in sys.argv[1:]:\n    p = Path(f)\n    s = p.read_text()\n"
            "    if not s.startswith('module'):\n        p.write_text('module\\n\\n' + s.replace('import ', 'public import '))\n"
            + then)


def _port(src: str) -> str:
    return "module\n\n" + src.replace("import ", "public import ")


def _show(world: World, ref: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=world.origin, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


@pytest.mark.parametrize("src, ported", [
    ("module\n\npublic import A\n", True),
    ("-- made before\n/-! a doc\n- an item -/\n\nmodule\npublic import A\n", True),
    ("module", True),
    ("/- a /- nested -/ comment -/\nmodule\n", False),          # the codemod decides, and leaves a module as it is
    ("/- a /- nested -/ module -/\nimport A\n", False),          # `module` inside the outer comment
    ("import A\n", False),
    ("-- module\nimport A\n", False),
    ("modules := 1\n", False),
    ("", False),
])
def test_a_header_reads_as_ported_only_when_module_opens_it(src, ported):
    assert bool(MODULE_HEADER.match(src)) is ported


def test_a_pr_adding_lean_made_before_the_port_above_a_ported_base_gets_it_ported_in_its_merge(world, tmp_path):
    """Only the `.lean` files the merge adds that aren't modules go to the codemod, run at the root of a worktree of the merged
    tree; what it rewrites is committed into the merge commit, with the PR's head as its second parent and its message, and
    the log names each file.  A merge adding no such file runs nothing."""
    argv = tmp_path / "argv"
    _move_main(world, {MODULES: _stub(argv), "Lib/Old.lean": "module\n\npublic import Init\n"})
    main = _git(world.origin, "rev-parse", "main")
    new, done = "-- made before the port\nimport Lib.Old\n\ndef x := 1\n", "/- ported -/\nmodule\n\npublic import Lib.Old\n"
    world.gh.put(1, head := world.change({"Lib/New.lean": new, "Lib/Done.lean": done, "a.txt": "a1\n"}))
    world.gh.put(2, world.change({"b.txt": "b2\n"}))
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [1, 2], {n: e.why for n, e in entries.items()}
    one = members[0].commit
    assert _show(world, one, "Lib/New.lean") == _port(new) and _show(world, one, "Lib/Done.lean") == done
    assert _show(world, one, "a.txt") == "a1\n" and _show(world, one, "Lib/Old.lean") == "module\n\npublic import Init\n"
    assert argv.read_text() == "Lib/New.lean"
    assert (f"research queue: #1 adds Lean made before the module system: Lib/New.lean ported by `{MODULES}` in the merged tree"
            in world.logs), world.logs
    assert [s for s in world.logs if "module system" in s and "#2" in s] == []
    assert _git(world.origin, "rev-parse", f"{one}^1", f"{one}^2").split() == [main, head]
    assert _git(world.origin, "log", "-1", "--format=%B", one) == f"Queue: #1 change 1\n\nPR: #1\nHead: {head}\nOwner: lane-1"
    assert _show(world, "refs/heads/next", "Lib/New.lean") == _port(new)


def test_each_pin_recording_a_ported_file_records_its_ported_digest_and_keeps_every_other_byte(world, tmp_path):
    """A pin records a Lean source's digest (PoUS's TRUSTED.sha256, a JSON pin with vectors after it), so a file the merge ports
    moves its record with it; a file `main` holds, unported or not, keeps its own."""
    tree = _pin_main(world)
    _move_main(world, {MODULES: _stub(tmp_path / "argv")})
    pr = {**tree, f"{POUS}/New.lean": "import Old\n", "y.lean": "import x\n"}
    paths, exfil = ["lean-toolchain", "lakefile.toml", "Old.lean", UP, "New.lean"], ["Security/Proofs/lakefile.toml", "x.lean", "y.lean"]
    world.gh.put(1, _pinned(world, pr, paths, exfil))
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1], entries[1].why
    merged = {**pr, f"{POUS}/New.lean": _port("import Old\n"), "y.lean": _port("import x\n")}
    assert _show(world, members[0].commit, f"{POUS}/New.lean") == merged[f"{POUS}/New.lean"]
    assert _show(world, members[0].commit, TRUSTED) == _trusted(paths, merged)
    assert _show(world, members[0].commit, EXFIL) == _exfil(exfil, merged)
    assert (f"research queue: #1 adds Lean made before the module system: {POUS}/New.lean, y.lean ported by `{MODULES}` in "
            f"the merged tree, and the pins {TRUSTED}, {EXFIL} rebuilt" in world.logs), world.logs


def test_a_base_without_the_port_merges_lean_files_as_the_pr_wrote_them(world):
    """Before the port lands the tree has no codemod, and the merge is git's alone."""
    new = "import Lib.Old\n"
    world.gh.put(1, world.change({"Lib/New.lean": new}))
    members, entries = world.queue().sync()
    assert [m.pr for m in members] == [1], entries[1].why
    assert _show(world, members[0].commit, "Lib/New.lean") == new
    assert [s for s in world.logs if "module system" in s] == []


@pytest.mark.parametrize("then, why", [
    ("print('Lib/New.lean: its header can\\'t be read: port it by hand', file=sys.stderr)\nsys.exit(1)\n",
     f"`{MODULES}` failed on the `.lean` files its merge adds that aren't modules, Lib/New.lean (exit 1): Lib/New.lean: its "
     f"header can't be read: port it by hand; run `python3 {MODULES}` on its head and port by hand what it leaves"),
    ("Path('a.txt').write_text('rewritten\\n')\n",
     f"`{MODULES}`, run on the `.lean` files its merge adds that aren't modules, rewrote a.txt, which the merge doesn't add: "
     f"run `python3 {MODULES}` on its head"),
], ids=["fails", "rewrites-a-file-not-added"])
def test_a_codemod_that_fails_or_rewrites_a_file_the_merge_doesnt_add_refuses_the_merge(world, tmp_path, then, why):
    """The PR is left out with the codemod's own words, as a conflict is (the state's `conflicts`, which drive posts to its
    owner), and the members above it merge."""
    _move_main(world, {MODULES: _stub(tmp_path / "argv", then)})
    world.gh.put(1, world.change({"Lib/New.lean": "import Lib.Old\n"}))
    world.gh.put(2, world.change({"b.txt": "b2\n"}))
    q = world.queue()
    members, entries = q.sync()
    assert [m.pr for m in members] == [2] and entries[1].state == "refused" and entries[1].why == why, entries[1].why
    assert q.load_state()["conflicts"]["1"]["why"] == why
    assert _show(world, "refs/heads/next", "a.txt") == "a\n"


@pytest.mark.parametrize("then", ["", "sys.exit(1)\n"], ids=["ports", "fails"])
def test_the_preflight_judges_the_merge_with_its_lean_ported_as_the_sync_ports_it(world, tmp_path, capsys, then):
    """A head marked after the port is checked, its Lean build included, on the tree drive's sync would merge, and a codemod
    that fails refuses the mark with its reason."""
    from research.queue import MarkOnly, StoreLabels, ready

    _preflight_main(world, {MODULES: _stub(tmp_path / "argv", then)})
    world.gh.put(2, h := world.change({"pkg/New.lean": "import Lib.Old\n"}), ready=False)
    assert ready(world.gh, world.store, 2, "lane-2", MarkOnly(), preflight=_preflight(world)) == (1 if then else 0)
    err = capsys.readouterr().err
    if then:
        assert f"  - `{MODULES}` failed on the `.lean` files its merge adds that aren't modules, pkg/New.lean (exit 1)" in err, err
    else:
        said = next(s for s in world.logs if "preflight: #2" in s)
        assert "nothing refused" in said and f"Lean made before the module system: pkg/New.lean ported by `{MODULES}`" in said, said
    assert StoreLabels(world.store, log=world.logs.append).ready(2, h) is (not then) and _no_worktree_left(world)
