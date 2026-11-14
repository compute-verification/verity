"""`evict_src` and a git worktree made from a tree: one that may still exist keeps the tree, and a registration whose worktree is
gone keeps nothing.  Git keeps the registration of a worktree removed with `rm -rf` in the tree's `.git/worktrees` for months,
so a tree that once had a worktree would otherwise never go."""
import os
import shutil
from pathlib import Path

import pytest

from research.store import evict as EV
from test_evict_src import RUNPOD_NGINX, _age, _git, _runpod_proxy, fake_proc, kept, node, run  # noqa: F401  (node is a fixture)


def worktree(n, sha: str, name: str = "wt", *, gone: bool = True) -> Path:
    """A worktree of the tree `sha`, as `git worktree add` makes one, and with `gone` removed as `rm -rf` removes one: its
    registration stays in the tree's `.git/worktrees`."""
    wt = n.tmp / name
    _git(n.src / sha, "worktree", "add", "-q", "--detach", str(wt), sha)
    if gone:
        shutil.rmtree(wt)
    _age(n.src / sha, 5)
    return wt


def entry(n, sha: str, name: str = "wt") -> Path:
    return n.src / sha / ".git" / "worktrees" / name


def went(rep: dict) -> list[str]:
    return [d["tree"] for d in rep["deleted"]]


def tail(path: Path | str) -> str:
    """A path as a reason spells it."""
    return str(path)[-120:]


def test_a_git_worktree_made_from_a_tree_keeps_it(node):
    """ci-bundle.sh and train.sh `git worktree add` from a checkout: its git data lives in the tree, its processes elsewhere."""
    wt = worktree(node, node.a, gone=False)
    rep = run(node)
    assert kept(rep)[node.a] == [f"{EV.WORKTREE}: {tail(wt)} is there"] and went(rep) == [node.b]


def test_registrations_whose_worktrees_are_gone_keep_no_tree(node):
    worktree(node, node.a)
    worktree(node, node.a, "wt2")
    assert sorted(os.listdir(entry(node, node.a).parent)) == ["wt", "wt2"], "git still lists both"
    rep = run(node)
    assert rep["error"] is None and went(rep) == [node.a, node.b] and not (node.src / node.a).exists()


def test_one_worktree_that_is_there_keeps_the_tree_whatever_the_others_are(node):
    worktree(node, node.a)
    wt = worktree(node, node.a, "wt2", gone=False)
    assert kept(run(node))[node.a] == [f"{EV.WORKTREE}: {tail(wt)} is there"]


def _locked(n, monkeypatch):
    wt = worktree(n, n.a, gone=False)
    _git(n.src / n.a, "worktree", "lock", str(wt))
    shutil.rmtree(wt)
    return "wt is locked"


def _relative(n, monkeypatch):
    worktree(n, n.a)
    (entry(n, n.a) / "gitdir").write_text("../../../../../wt/.git\n")
    return "wt's gitdir is not an absolute path to a worktree's .git: '../../../../../wt/.git'"


def _not_a_git_file(n, monkeypatch):
    wt = worktree(n, n.a)
    (entry(n, n.a) / "gitdir").write_text(f"{wt}\n")
    return f"wt's gitdir is not an absolute path to a worktree's .git: {tail(wt)!r}"


def _two_lines(n, monkeypatch):
    wt = worktree(n, n.a)
    (entry(n, n.a) / "gitdir").write_text(f"{wt}/.git\n{n.tmp}/elsewhere/.git\n")
    return "wt's gitdir is not an absolute path to a worktree's .git: "


def _no_gitdir(n, monkeypatch):
    worktree(n, n.a)
    (entry(n, n.a) / "gitdir").unlink()
    return "cannot tell whether wt's worktree is gone: FileNotFoundError"


def _unreadable(n, monkeypatch):
    wt, real = worktree(n, n.a), os.lstat

    def lstat(p, *a, **k):
        if str(p) == str(wt):
            raise PermissionError(13, "Permission denied", str(p))
        return real(p, *a, **k)

    monkeypatch.setattr(EV.os, "lstat", lstat)
    return "cannot tell whether wt's worktree is gone: PermissionError"


def _under_a_file(n, monkeypatch):
    """ENOTDIR says no more than that this path reaches nothing from here."""
    worktree(n, n.a)
    (n.tmp / "wt").write_text("a file where the worktree's directory was")
    (entry(n, n.a) / "gitdir").write_text(f"{n.tmp}/wt/inner/.git\n")
    return "cannot tell whether wt's worktree is gone: NotADirectoryError"


def _named(n, monkeypatch):
    wt = worktree(n, n.a)
    fake_proc(n.proc, 300, cwd=f"{wt}/tools")
    return f"{tail(wt)} is gone, but process 300 (p300) uses it"


def _named_as_it_resolves(n, monkeypatch):
    """A process's cwd, open files and maps are resolved paths, whatever spelling the registration holds."""
    worktree(n, n.a)
    (n.tmp / "real").mkdir()
    os.symlink(n.tmp / "real", n.tmp / "link")
    (entry(n, n.a) / "gitdir").write_text(f"{n.tmp}/link/wt/.git\n")
    fake_proc(n.proc, 301, fds=(f"{os.path.realpath(n.tmp / 'real')}/wt/README.md (deleted)",))
    return f"{tail(n.tmp / 'link' / 'wt')} is gone, but process 301 (p301) uses it"


def _processes_unread(n, monkeypatch):
    wt, real, calls = worktree(n, n.a), EV.proc_uses, []

    def unreadable(pattern, proc, *rest):
        calls.append(1)
        if len(calls) == 2:                      # the reading for a's worktree; the pass's own came first
            raise EV.Unestablished("2: /proc/9 is unreadable")
        return real(pattern, proc, *rest)

    monkeypatch.setattr(EV, "proc_uses", unreadable)
    return f"whether a process names one that is gone (e.g. {tail(wt)}) is unknown: /proc/9 is unreadable"


@pytest.mark.parametrize("setup", [_locked, _relative, _not_a_git_file, _two_lines, _no_gitdir, _unreadable, _under_a_file, _named,
                                   _named_as_it_resolves, _processes_unread], ids=lambda f: f.__name__.strip("_"))
def test_a_registration_whose_worktree_may_still_exist_keeps_the_tree_and_the_pass_goes_on(node, monkeypatch, setup):
    """A registration is dead only when it is not locked, its `gitdir` is one absolute path to a worktree's `.git`, that
    worktree's directory is not there (ENOENT) and no live process names it; each case here fails one of those, with every
    other one met."""
    why = setup(node, monkeypatch)
    rep = run(node)
    assert rep["error"] is None and went(rep) == [node.b]
    assert [w[:len(EV.WORKTREE) + 2 + len(why)] for w in kept(rep)[node.a]] == [f"{EV.WORKTREE}: {why}"]
    assert (node.src / node.a / EV.READY_FILE).exists() and not [n for n in os.listdir(node.src) if n.startswith(EV.ASIDE)]


def test_a_worktrees_directory_that_cannot_be_listed_keeps_the_tree(node, monkeypatch):
    worktree(node, node.a)
    real = os.listdir

    def listdir(p="."):
        if str(p) == str(entry(node, node.a).parent):
            raise PermissionError(13, "Permission denied")
        return real(p)

    monkeypatch.setattr(EV.os, "listdir", listdir)
    rep = run(node)
    assert kept(rep)[node.a] == [f"{EV.WORKTREE}: cannot list {entry(node, node.a).parent}: Permission denied"] and went(rep) == [node.b]


def test_a_process_that_appears_in_a_gone_worktree_after_the_rename_moves_the_tree_back(node, monkeypatch):
    """The worktrees are read again once the tree is aside, as everything else is."""
    wt, real, calls = worktree(node, node.a), EV.dir_uses, []

    def late(dirs, proc, *rest):
        calls.append(sorted(dirs))
        if len(calls) == 2:
            fake_proc(proc, 302, cmdline=f"python3 {wt}/tools/x.py")
        return real(dirs, proc, *rest)

    monkeypatch.setattr(EV, "dir_uses", late)
    rep = run(node)
    assert calls == [[str(wt)]] * 2
    assert kept(rep)[node.a] == [f"{EV.WORKTREE}: {tail(wt)} is gone, but process 302 (p302) uses it", "2: moved back"]
    assert (node.src / node.a / EV.READY_FILE).exists() and went(rep) == [node.b]


def test_the_processes_are_read_for_a_gone_worktree_only_when_nothing_else_keeps_its_tree(node, monkeypatch):
    """A tree its own changes keep is checked every pass, and each reading of a node's processes takes seconds."""
    worktree(node, node.a)
    (node.src / node.a / "notes.txt").write_text("no commit holds this")
    _age(node.src / node.a, 5)
    calls = []
    monkeypatch.setattr(EV, "dir_uses", lambda dirs, *rest: calls.append(dirs) or {})
    rep = run(node)
    assert kept(rep)[node.a] == ["1: not recoverable on the node: 1 local change(s) no commit holds, e.g. '? notes.txt'"]
    assert calls == [] and went(rep) == [node.b]


def test_a_dry_run_says_a_tree_with_only_gone_worktrees_would_go_and_deletes_nothing(node):
    worktree(node, node.a)
    rep = run(node, dry_run=True)
    assert went(rep) == [node.a, node.b] and (node.src / node.a / EV.READY_FILE).exists()
    assert sorted(os.listdir(entry(node, node.a).parent)) == ["wt"], "a pass never prunes a registration"


UNNAMED, NAMED = "/a-top-directory-no-configuration-names/wt", "/no-such-top-directory/wt"
NAMES_ONE_TOP = RUNPOD_NGINX + "# root /no-such-top-directory/x;\n"


@pytest.mark.parametrize("conf, worktrees, kept_a", [
    (RUNPOD_NGINX, [UNNAMED, NAMED], False),
    (NAMES_ONE_TOP, [UNNAMED], False),
    (NAMES_ONE_TOP, [NAMED], True),
    (NAMES_ONE_TOP, [UNNAMED, NAMED], True),
], ids=["conf names neither", "conf names another's top directory", "conf names its top directory", "conf names one of two"])
def test_an_unreadable_proxy_worker_is_passed_over_only_when_its_configuration_cannot_name_a_gone_worktree(
        node, monkeypatch, conf, worktrees, kept_a):
    """The pass reads every process but RunPod's proxy worker, which it passes over because the proxy's configuration cannot name
    a tree.  A worktree is somewhere else, so the same holds of it only when that configuration names nothing above it either,
    whatever it names above the tree's other worktrees; when it might, the tree stays and the pass goes on."""
    for i, wt in enumerate(worktrees):
        worktree(node, node.a, f"wt{i}")
        (entry(node, node.a, f"wt{i}") / "gitdir").write_text(f"{wt}/.git\n")
    rep = run(node, nginx_conf=_runpod_proxy(node, monkeypatch, conf=conf))
    assert rep["error"] is None and went(rep) == ([node.b] if kept_a else [node.a, node.b])
    if kept_a:
        why, = kept(rep)[node.a]
        assert why.startswith(f"{EV.WORKTREE}: whether a process names one that is gone (e.g. {worktrees[0]}) is unknown: ")
        assert "/40 is unreadable" in why
