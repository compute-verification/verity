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

import pytest

from research import asker, at_main, auto, cli, remote, suite_size

ENV = "RESEARCH_LAUNCHER_TREE"
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


@pytest.fixture
def trees(tmp_path, monkeypatch):
    """(source, trusted, marks): the source a committed git checkout whose programs, hooks, fsmonitor and filter each mark
    `marks`, with the global git config defining the filter its `.gitattributes` names; the trusted tree stubs of the same
    programs that mark nothing."""
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
        ".gitattributes": "* filter=trip\n",
    })
    for args in (["init", "-q", "-b", "lane"], ["add", "-A"], ["commit", "-q", "-m", "tripwire"]):
        subprocess.run(["git", *args], cwd=source, env=SETUP_GIT, check=True, capture_output=True)
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
    })
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


def ship(source: Path, root: Path) -> None:
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
        with pytest.raises(remote.SourceNotReady):
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
