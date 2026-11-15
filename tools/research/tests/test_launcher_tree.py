"""Under `RESEARCH_LAUNCHER_TREE` the launcher runs nothing of the tree it ships (research.launcher).

Each test launches from a tripwire tree: every program the launcher used to run from it, every git hook and fsmonitor
command its git directory names, and a filter its `.gitattributes` selects, touch a file under `marks`. With the variable
set no mark appears; unset, the launcher runs the tree's copies as before, and the marks show each tripwire is live."""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import sys

import pytest

from research import asker, at_main, auto, cli, launcher, remote, suite_size

ENV = "RESEARCH_LAUNCHER_TREE"
LAUNCHER_PY = "tools/research/src/research/launcher.py"
SETUP_GIT = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t",
             "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
SUITES = ["uv", "run", "tools/check/suites.py", "--changed", "main", "--quick"]
RECORDS = ["python3", "tools/verity/lean/lean_changed.py", "--records", "security"]


def trip(marks: Path, name: str) -> str:
    return f"import pathlib\npathlib.Path({str(marks)!r}, {name!r}).touch()\n"


def write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def script(path: Path, marks: Path, name: str, tail: str = "") -> Path:
    path.write_text(f"#!/bin/sh\ntouch {marks / name}\n{tail}")
    path.chmod(0o755)
    return path


def git(at: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=at, env=SETUP_GIT, check=True, capture_output=True, text=True).stdout.strip()


def checkout(at: Path) -> None:
    """``at`` committed, with its HEAD as its own origin/main."""
    for args in (["init", "-q", "-b", "lane"], ["add", "-A"], ["commit", "-q", "-m", "tree"], ["update-ref", "refs/remotes/origin/main", "HEAD"]):
        git(at, *args)


@pytest.fixture(autouse=True)
def unchecked():
    getattr(launcher, "_CHECKED", {}).clear()


@pytest.fixture
def trees(tmp_path, monkeypatch):
    """(source, trusted, marks): the source a committed git checkout whose programs, hooks, fsmonitor and filter each mark
    `marks`, with the global git config defining the filter its `.gitattributes` names; the trusted tree a clean checkout
    at its origin/main with stubs of the same programs that mark nothing."""
    marks, source, trusted = tmp_path / "marks", tmp_path / "source", tmp_path / "trusted"
    marks.mkdir()
    write(source, {
        "infra/cluster/src/cluster/__init__.py": "",
        "infra/cluster/src/cluster/submit.py": "",
        "infra/cluster/src/cluster/__main__.py": trip(marks, "cluster-submit") + "print('{\"refused\": \"source\"}')\n",
        "infra/cluster/descriptions/nebius.toml": '[[shares]]\nproject = "verity"\nnode = "vy-nebius-2"\ncpus = "0-1"\n',
        "cluster/__init__.py": trip(marks, "cluster-shadow"),
        "cluster/__main__.py": "",
        "research/__init__.py": trip(marks, "research-shadow"),
        "tools/check/suites.py": trip(marks, "suites"),
        "tools/check/slot.py": "TRIPWIRE_SLOT = 1\n",
        "tools/check/lean_slot.py": "TRIPWIRE_LEAN_SLOT = 1\n",
        cli.LEAN_AUDIT: trip(marks, "lean-audit") + "def url_file(dest):\n    return None\n",
        LAUNCHER_PY: trip(marks, "source-launcher"),
        ".gitattributes": "* filter=trip\n",
    })
    checkout(source)
    hooks = source / ".git" / "hooks"
    script(hooks / "pre-push", marks, "pre-push")
    script(hooks / "post-checkout", marks, "post-checkout")
    fsmonitor = script(tmp_path / "fsmonitor.sh", marks, "fsmonitor", "exit 1\n")
    subprocess.run(["git", "config", "core.fsmonitor", str(fsmonitor)], cwd=source, env=SETUP_GIT, check=True)
    filt = script(tmp_path / "filter.sh", marks, "filter", "exec cat\n")
    (tmp_path / "gitconfig").write_text(f'[filter "trip"]\n\tsmudge = {filt}\n\tclean = {filt}\n')
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    write(trusted, {
        "infra/cluster/src/cluster/__init__.py": "",
        "infra/cluster/src/cluster/submit.py": "",
        "infra/cluster/src/cluster/__main__.py": "print('{\"machine\": \"vy-nebius-2\", \"why\": \"trusted\"}')\n",
        "infra/cluster/descriptions/nebius.toml": '[[shares]]\nproject = "verity"\nnode = "vy-nebius-2"\ncpus = "0-47"\n',
        "tools/check/slot.py": "TRUSTED_SLOT = 1\n",
        "tools/check/lean_slot.py": "TRUSTED_LEAN_SLOT = 1\n",
        cli.LEAN_AUDIT: "import json, pathlib\nSHARED = 'lean-artifacts/'\ndef url_file(dest):\n"
                        "    p = pathlib.Path(dest) / 'lean-deps-urls.json'\n    p.write_text(json.dumps({'k': 'trusted'}))\n    return p\n",
        LAUNCHER_PY: "",
    })
    checkout(trusted)
    return source, trusted, marks


def marked(marks: Path) -> list[str]:
    return sorted(p.name for p in marks.iterdir())


def test_the_queues_client_is_the_trusted_trees_run_outside_the_source(trees, monkeypatch):
    """`cluster submit` (`remote.queue_placement`) is the trusted tree's, and its `python -m` doesn't start in the source,
    whose top-level `cluster` would come first on its sys.path. Unset, the source's client places the job."""
    source, trusted, marks = trees
    spec = {"project": "verity", "kind": "agent-test", "node": "vy-nebius-2"}
    monkeypatch.chdir(source)
    monkeypatch.setenv(ENV, str(trusted))
    assert remote.queue_placement(source, "r1", spec)["why"] == "trusted"
    assert marked(marks) == []
    monkeypatch.delenv(ENV)
    monkeypatch.chdir(trusted)
    with pytest.raises(remote.Refused, match="the queue refuses the job: source"):
        remote.queue_placement(source, "r2", spec)
    assert marked(marks) == ["cluster-submit"]


def test_a_changed_suites_run_is_not_listed_by_the_sources_suites_py_and_takes_the_kinds_size(trees, monkeypatch, capsys):
    """suites.py `--list` would import the source's code, so under the variable it doesn't run and `--on auto` sizes the run
    as its kind (None). Unset, the source's suites.py lists the suites."""
    source, trusted, marks = trees
    monkeypatch.setenv(ENV, str(trusted))
    assert suite_size.size(source, SUITES, None) is None
    assert marked(marks) == []
    assert f"not listed under ${ENV}" in capsys.readouterr().err
    monkeypatch.delenv(ENV)
    suite_size.size(source, SUITES, None)
    assert marked(marks) == ["fsmonitor", "suites"], "the source's fsmonitor command is `git ls-files`'s"


def test_a_lean_records_runs_urls_come_from_the_trusted_trees_lean_audit(trees, monkeypatch):
    source, trusted, marks = trees
    monkeypatch.setenv(ENV, str(trusted))
    urls = cli.lean_records_urls(source, RECORDS)
    try:
        assert urls is not None and json.loads(urls.read_text()) == {"k": "trusted"}
    finally:
        if urls:
            shutil.rmtree(urls.parent)
    assert marked(marks) == []
    monkeypatch.delenv(ENV)
    assert cli.lean_records_urls(source, RECORDS) is None
    assert marked(marks) == ["lean-audit"]


def test_the_pool_probe_carries_the_trusted_trees_slot_programs_and_share(trees, monkeypatch):
    """`auto.read` sends each machine slot.py and lean_slot.py to run as `python3 -`, and node 2's share from the
    description: the trusted tree's, never the source's."""
    source, trusted, _ = trees
    sent: list[bytes] = []

    def probe(argv, data):
        sent.append(data)
        return subprocess.CompletedProcess(argv, 0, b"{}", b"")

    monkeypatch.setenv(ENV, str(trusted))
    auto.read(source, 4, reach=lambda name: ["ssh", name], run=probe, node_only=True, lean=True)
    monkeypatch.delenv(ENV)
    auto.read(source, 4, reach=lambda name: ["ssh", name], run=probe, node_only=True, lean=True)
    under, unset = sent
    assert b"TRUSTED_SLOT" in under and b"TRUSTED_LEAN_SLOT" in under and b"SHARE = '0-47'" in under
    assert b"TRIPWIRE" not in under
    assert b"TRIPWIRE_SLOT" in unset and b"TRIPWIRE_LEAN_SLOT" in unset and b"SHARE = '0-1'" in unset


def ship(source: Path, root: Path, expect: type[BaseException] = remote.SourceNotReady) -> None:
    """`remote.ship_source` of ``source`` as far as it goes with no machine: the probe finds no tree, the commit is pushed
    over `ext::` into a bare repository here (the "ssh" is `sh -c`), the machine's checkout fails, and the archive stream's
    feed runs into nothing before the machine refuses it."""
    def ssh_run(ssh, argv, *, stdin=None, timeout=120):
        text = " ".join(argv)
        if remote.READY_FILE in text and "ABSENT" in text:
            return subprocess.CompletedProcess(argv, 0, b"ABSENT\n", b"")
        if "git init -q --bare" in text:
            return subprocess.run(argv, capture_output=True, env=SETUP_GIT)
        return subprocess.CompletedProcess(argv, 0 if argv[0] == "rm" else 1, b"", b"no machine")

    def ssh_stream(ssh, argv, feed, *, timeout):
        feed(io.BytesIO())
        return subprocess.CompletedProcess(argv, 1, b"", b"no machine")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(remote, "ssh_run", ssh_run)
        mp.setattr(remote, "ssh_stream", ssh_stream)
        with pytest.raises(expect):
            remote.ship_source(["sh", "-c"], str(root), source)


def asked(source: Path) -> None:
    """The asker's reads of the shipped commit: its `auto.READS` files (`asker.files`) and a checkout for a twin (`asker.tree`)."""
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=source, env=SETUP_GIT,
                            capture_output=True, text=True, check=True).stdout.strip()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, env=SETUP_GIT, capture_output=True, text=True,
                         check=True).stdout.strip()
    with asker.files(common, sha, auto.READS), asker.tree(common, sha):
        pass


def test_git_on_the_shipped_tree_runs_no_hook_fsmonitor_or_configured_filter(trees, tmp_path, monkeypatch):
    """Shipping the commit (`git status`, `archive`, `push`) and the asker's reads of it (`archive`, `worktree add`) run none
    of the source's hooks or fsmonitor command, nor the filter its `.gitattributes` names from the global config."""
    source, trusted, marks = trees
    monkeypatch.setenv(ENV, str(trusted))
    ship(source, tmp_path / "machine")
    asked(source)
    assert marked(marks) == []
    monkeypatch.delenv(ENV)
    shutil.rmtree(tmp_path / "machine")
    ship(source, tmp_path / "machine")
    asked(source)
    assert marked(marks) == ["filter", "fsmonitor", "post-checkout", "pre-push"]


def test_research_is_not_re_executed_from_the_sources_working_directory(trees, tmp_path, monkeypatch):
    """`research.at_main` re-executes `python -m research` in the working directory, a launch's source, which would come first
    on the new process's sys.path and run the source's top-level `research`: under the variable it re-executes nothing."""
    source, trusted, marks = trees
    monkeypatch.delenv(at_main.OPT_OUT, raising=False)
    monkeypatch.setattr(at_main, "behind", lambda repo, end=None: ("0" * 40, 1))
    monkeypatch.setattr(at_main, "tree", lambda repo, main, cache, end=None: tmp_path / "main-tree")
    monkeypatch.setattr(at_main, "prune", lambda cache, keep, end: None)
    monkeypatch.chdir(source)

    def execve(exe, argv, env):
        subprocess.run(argv, executable=exe, cwd=Path.cwd(), env=env, capture_output=True)

    repo = tmp_path / "install"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setenv(ENV, str(trusted))
    at_main.reexec(["--version"], repo, tmp_path / "cache", execve)
    assert marked(marks) == []
    monkeypatch.delenv(ENV)
    at_main.reexec(["--version"], repo, tmp_path / "cache", execve)
    assert marked(marks) == ["research-shadow"]


def _nested_trusted(source, trusted, tmp_path):
    shutil.copytree(trusted, source / "nested", symlinks=True)
    return str(source / "nested"), source


def _around(source, trusted, tmp_path):
    shutil.copytree(source, trusted / "lane", symlinks=True)
    return str(trusted), trusted / "lane"


def _link_into(source, trusted, tmp_path):
    (tmp_path / "link").symlink_to(source)
    return str(tmp_path / "link"), source


def _branch(source, trusted, tmp_path):
    git(trusted, "commit", "-q", "--allow-empty", "-m", "a lane's commit")
    return str(trusted), source


def _dirty(source, trusted, tmp_path):
    (trusted / "tools/check/slot.py").write_text("MODIFIED = 1\n")
    return str(trusted), source


def _untracked(source, trusted, tmp_path):
    (trusted / "infra/cluster/src/cluster/extra.py").write_text("")
    return str(trusted), source


def _no_launcher(source, trusted, tmp_path):
    (trusted / LAUNCHER_PY).unlink()
    return str(trusted), source


def _not_a_checkout(source, trusted, tmp_path):
    shutil.rmtree(trusted / ".git")
    return str(trusted), source


#: (value and the launch's source, from (source, trusted, tmp_path); what the refusal says)
UNTRUSTED = {
    "relative": (lambda s, t, p: (".", s), "is not an absolute path"),
    "relative-to-a-good-tree": (lambda s, t, p: ("../trusted", s), "is not an absolute path"),
    "blank": (lambda s, t, p: ("", s), "is blank, and set"),
    "whitespace": (lambda s, t, p: ("  ", s), "is blank, and set"),
    "missing": (lambda s, t, p: (str(p / "nowhere"), s), "is not an existing directory"),
    "empty-dir": (lambda s, t, p: (str(p), s), "lacks what the launcher runs from it: tools/research/src/research/launcher.py"),
    "no-launcher-py": (_no_launcher, f"lacks what the launcher runs from it: {LAUNCHER_PY}"),
    "not-a-checkout": (_not_a_checkout, "is not a git checkout with an origin/main"),
    "the-source": (lambda s, t, p: (str(s), s), "is the --source tree"),
    "a-symlink-to-the-source": (_link_into, "is the --source tree"),
    "inside-the-source": (_nested_trusted, "is inside the --source tree"),
    "around-the-source": (_around, "is around the --source tree"),
    "on-a-branch": (_branch, "is stale: its HEAD"),
    "a-modified-slot-py": (_dirty, "is stale: what the launcher runs from it is changed or untracked there (tools/check/slot.py)"),
    "an-untracked-module": (_untracked, "is stale: what the launcher runs from it is changed or untracked there"),
}


@pytest.mark.parametrize("case", sorted(UNTRUSTED))
def test_an_untrusted_value_is_refused_before_anything_runs_and_nothing_falls_back_to_the_source(case, trees, tmp_path,
                                                                                                    monkeypatch, capsys):
    """Each value the variable can't be trusted with refuses `research run` in one line naming the variable, the problem and
    the cure, exit 2, before placement (which here marks `launched`); and every reader of the tree refuses it too, so no
    path runs the source's programs in its place: no tripwire marks."""
    source, trusted, marks = trees
    value, src = UNTRUSTED[case][0](source, trusted, tmp_path)
    said = UNTRUSTED[case][1]
    monkeypatch.chdir(src)
    monkeypatch.setenv(ENV, value)
    monkeypatch.setattr(cli, "_auto_placement", lambda ns, opts: (marks / "launched").touch() or "placed")
    assert cli._cmd_run(["--on", "auto", "--source", str(src), "--", *SUITES], []) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"research: ${ENV}={value!r} {said}") and err.count("\n") == 1, err
    assert "absolute path" in err or "unset" in err
    refused = pytest.raises(SystemExit, match=rf"\${ENV}=")
    spec = {"project": "verity", "kind": "agent-test", "node": "vy-nebius-2"}
    for call in (lambda: remote.queue_placement(src, "r1", spec), lambda: suite_size.size(src, SUITES, None),
                 lambda: cli.lean_records_urls(src, RECORDS),
                 lambda: auto.read(src, 4, reach=lambda n: ["ssh", n], run=lambda a, d: (marks / "probe").touch(), node_only=True)):
        with refused:
            call()
    ship(src, tmp_path / "machine", SystemExit)
    monkeypatch.setattr(at_main, "behind", lambda repo, end=None: ("0" * 40, 1))
    monkeypatch.setattr(at_main, "tree", lambda repo, main, cache, end=None: tmp_path / "main-tree")
    monkeypatch.setattr(at_main, "prune", lambda cache, keep, end: None)
    (tmp_path / "install" / ".git").mkdir(parents=True)
    at_main.reexec(["--version"], tmp_path / "install", tmp_path / "cache", lambda exe, argv, env: (marks / "reexec").touch())
    assert marked(marks) == []


def test_unset_at_main_re_executes_origin_mains_copy_exactly_as_before(tmp_path, monkeypatch):
    """The queue's loop runs `research` through `at_main` every tick: with the variable absent, a checkout behind origin/main
    re-executes origin/main's copy with the same interpreter, argv and environment as before the variable existed. It
    passes on the code before this change too; it is here so no later change moves it."""
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(at_main.OPT_OUT, raising=False)
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    wt = tmp_path / "cache" / "wt"
    monkeypatch.setattr(at_main, "behind", lambda repo, end=None: ("a" * 40, 3))
    monkeypatch.setattr(at_main, "tree", lambda repo, main, cache, end=None: wt)
    monkeypatch.setattr(at_main, "prune", lambda cache, keep, end: None)
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    calls = []
    at_main.reexec(["queue", "vqueue", "--once"], tmp_path / "repo", tmp_path / "cache", lambda *a: calls.append(a))
    want = {**os.environ, at_main.OPT_OUT: "1", "PYTHONPATH": f"{wt / 'tools/research/src'}{os.pathsep}/elsewhere"}
    [(exe, argv, got)] = calls
    assert (exe, argv) == (sys.executable, [sys.executable, "-m", "research", "queue", "vqueue", "--once"])
    differ = sorted(k for k in got.keys() | want.keys() if got.get(k) != want.get(k))
    assert differ == []
    monkeypatch.setenv(at_main.OPT_OUT, "1")
    at_main.reexec(["status"], tmp_path / "repo", tmp_path / "cache", lambda *a: calls.append(a))
    assert len(calls) == 1
