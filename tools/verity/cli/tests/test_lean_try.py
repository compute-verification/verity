"""`verity lean try FILE` without `--on`: `lake env lean FILE` in the directory of the nearest lakefile above FILE, with
its package's Lean options and lean's exit status passed through, against a stand-in `lake` on PATH, so no Lean
toolchain is needed.  With or without
`--on`, it takes FILE from the checkout FILE is in (or the current directory's), not the one the CLI's code lives in."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import verity_cli


def stand_in_lake(tmp_path: Path, monkeypatch, rc: int) -> Path:
    """A `lake` that writes its working directory and arguments to the file it returns, and exits with ``rc``."""
    called, bin_ = tmp_path / "called", tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "lake").write_text(f'#!/bin/sh\n{{ pwd -P; echo "$@"; }} > "{called}"\nexit {rc}\n')
    (bin_ / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_), prepend=os.pathsep)
    return called


def test_try_runs_lake_env_lean_in_the_nearest_package_and_passes_its_status(tmp_path, monkeypatch, capsys):
    (tmp_path / "lakefile.toml").write_text('name = "Outer"\n')
    pkg = tmp_path / "Outer" / "Inner"
    (pkg / "Inner" / "Area").mkdir(parents=True)
    (pkg / "lakefile.lean").write_text("")
    file = pkg / "Inner" / "Area" / "File.lean"
    file.write_text("theorem t : True := trivial\n")
    called = stand_in_lake(tmp_path, monkeypatch, 3)
    assert verity_cli.main(["lean", "try", str(file)]) == 3
    assert called.read_text().splitlines() == [str(pkg.resolve()), "env lean .lake/axiom-probe/Inner/Area/File.lean"]
    assert (f"verity lean try: lake env lean .lake/axiom-probe/Inner/Area/File.lean, Inner/Area/File.lean with the axiom "
            f"probe after its last line (in {pkg.resolve()})") in capsys.readouterr().out
    assert not (pkg / ".lake" / "axiom-probe" / "Inner" / "Area" / "File.lean").exists()


def test_try_gives_lean_its_packages_lean_options(tmp_path, monkeypatch, capsys):
    """`lake build` gives each module its package's `[leanOptions]` and `lake env lean` doesn't, so the try passes them."""
    (tmp_path / "lakefile.toml").write_text('name = "P"\n\n[leanOptions]\nautoImplicit = false\n')
    (tmp_path / "T.lean").write_text("theorem t : True := trivial\n")
    called = stand_in_lake(tmp_path, monkeypatch, 0)
    assert verity_cli.main(["lean", "try", str(tmp_path / "T.lean")]) == 0
    assert called.read_text().splitlines()[1] == "env lean -Dweak.autoImplicit=false .lake/axiom-probe/T.lean"
    assert ("verity lean try: lake env lean -Dweak.autoImplicit=false .lake/axiom-probe/T.lean, T.lean with the axiom probe"
            in capsys.readouterr().out)


def test_try_ends_with_the_axiom_probes_line_when_lean_exits_0(tmp_path, monkeypatch, capsys):
    (tmp_path / "lakefile.toml").write_text('name = "P"\n')
    (tmp_path / "T.lean").write_text("theorem t : True := sorry\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "lake").write_text("#!/bin/sh\necho \"'t' depends on axioms: [sorryAx]\"\n")
    (bin_ / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_), prepend=os.pathsep)
    assert verity_cli.main(["lean", "try", str(tmp_path / "T.lean")]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == (
        "verity lean try: T.lean: axiom probe: 1 of its 1 theorems use axioms beyond propext, Classical.choice and "
        "Quot.sound: t (sorryAx)")


def test_try_names_a_theorem_lean_fails_on_and_prints_no_axioms_answer(tmp_path, monkeypatch, capsys):
    """`#print axioms` answers "does not depend on any axioms" for a theorem Lean refused (here a `rfl` out of recursion
    depth, as Lean prints it), beside theorems it accepts: the try exits with lean's status, its last line names the
    theorem as failed, and it prints no answer, so no line of it can be read as clean for any theorem."""
    (tmp_path / "lakefile.toml").write_text('name = "P"\n')
    (tmp_path / "T.lean").write_text("theorem clean : True := trivial\n\ntheorem refused : way 100000 = 100000 := rfl\n\n"
                                     "theorem after : True := trivial\n")
    said = [".lake/axiom-probe/T.lean:3:0: error: maximum recursion depth has been reached",
            "use `set_option maxRecDepth <num>` to increase limit",
            "'clean' does not depend on any axioms", "'refused' does not depend on any axioms",
            "'after' does not depend on any axioms"]
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "lake").write_text("#!/bin/sh\ncat <<'EOF'\n" + "\n".join(said) + "\nEOF\nexit 1\n")
    (bin_ / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_), prepend=os.pathsep)
    assert verity_cli.main(["lean", "try", str(tmp_path / "T.lean")]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[1:] == [*said[:2], "verity lean try: T.lean: axiom probe: 1 of its 3 theorems failed, with Lean's error "
                       "above on its own lines: refused; no axioms are shown for a file lean refuses"]


def test_try_refuses_a_file_in_no_package_and_remote_options_without_on(tmp_path, monkeypatch, capsys):
    called = stand_in_lake(tmp_path, monkeypatch, 0)
    loose = tmp_path / "Loose.lean"
    loose.write_text("")
    with pytest.raises(SystemExit, match="Loose.lean is not a .lean file in a Lake package"):
        verity_cli.main(["lean", "try", str(loose)])
    with pytest.raises(SystemExit):
        verity_cli.main(["lean", "try", str(loose), "--pool", "P"])
    assert "--pool and --as go with --on" in capsys.readouterr().err
    assert not called.exists()


def _git(cwd: Path, *argv: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *argv], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def checkouts(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    (main / "tools" / "verity" / "lean").mkdir(parents=True)
    (main / "tools" / "verity" / "lean" / "lean_changed.py").write_text("")
    (main / "A.lean").write_text("-- main's\n")
    _git(main, "init", "-q")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "base")
    _git(main, "worktree", "add", "-q", "-b", "lane", str(tmp_path / "wt"))
    (tmp_path / "wt" / "B.lean").write_text("-- the worktree's only\n")
    return main, tmp_path / "wt"


def test_ships_the_worktree_the_file_is_in(checkouts, monkeypatch):
    main, wt = checkouts
    monkeypatch.chdir(main)
    assert verity_cli._tree(str(wt / "A.lean")) == wt.resolve()


def test_ships_the_current_directorys_checkout(checkouts, monkeypatch):
    main, wt = checkouts
    monkeypatch.chdir(wt)
    assert verity_cli._tree("A.lean") == wt.resolve()
    monkeypatch.chdir(wt / "tools")
    assert verity_cli._tree("B.lean") == wt.resolve()


def test_snapshot_of_a_dirty_worktree_holds_its_edits(checkouts, tmp_path, monkeypatch):
    _, wt = checkouts
    for k in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(k, "t@t")
    shipped = verity_cli._snapshot(wt.resolve(), tmp_path)
    assert (shipped / "B.lean").read_text() == "-- the worktree's only\n"
    _git(wt, "worktree", "remove", "--force", str(shipped))


def test_refuses_a_checkout_that_isnt_verity(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    monkeypatch.chdir(other)
    with pytest.raises(SystemExit, match="not a Verity checkout"):
        verity_cli._tree("X.lean")


def test_try_on_auto_fetches_the_run_from_the_machine_it_landed_on(checkouts, tmp_path, monkeypatch, capsys):
    """`--on auto` names no machine and no pool: `research run --on auto` places the run, and its launch line says where, which
    is where the run's logs are fetched from."""
    _, wt = checkouts
    for k in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(k, "t@t")
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path / "runs"))
    monkeypatch.chdir(wt)
    launched, fetched, real_run, real_popen = [], [], subprocess.run, subprocess.Popen

    class Launch:
        stderr = iter(["research run --on auto: spilled to m-7's research pool, from . (a free slot)\n",
                       "research: run r1 launched on m-7 (p) pid=1 -> /r\n"])

        def wait(self):
            return 0

    def popen(argv, **kw):
        if argv[3:4] != ["run"]:
            return real_popen(argv, **kw)
        launched.append(argv)
        return Launch()

    def run(argv, **kw):
        if argv[3:5] == ["fetch", "r1"]:
            fetched.append(argv[5:])
            return subprocess.CompletedProcess(argv, 0)
        return real_run(argv, **kw)

    monkeypatch.setattr(verity_cli.subprocess, "Popen", popen)
    monkeypatch.setattr(verity_cli.subprocess, "run", run)
    assert verity_cli.main(["lean", "try", "A.lean", "--on", "auto"]) == 0
    argv = launched[0]
    assert argv[argv.index("--on") + 1] == "auto" and "--pool" not in argv
    assert fetched == [["--on", "m-7", "--all"]]
    assert "verity lean try: run r1 on m-7: exit 0" in capsys.readouterr().out


def test_local_try_elaborates_the_current_worktrees_file(checkouts, tmp_path, monkeypatch):
    main, wt = checkouts
    for tree in (main, wt):
        (tree / "lakefile.toml").write_text('name = "P"\n')
    called = stand_in_lake(tmp_path, monkeypatch, 0)
    monkeypatch.chdir(wt / "tools")
    assert verity_cli.main(["lean", "try", "A.lean"]) == 0
    assert called.read_text().splitlines()[0] == str(wt.resolve())


def test_on_refuses_a_file_outside_every_checkout_before_queueing(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch.lean"
    scratch.write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verity_cli, "_snapshot", lambda *a, **k: pytest.fail("shipped a tree to queue a run"))
    with pytest.raises(SystemExit, match="put FILE in a checkout"):
        verity_cli.main(["lean", "try", str(scratch), "--on", "vy-nebius-cpu-1"])
