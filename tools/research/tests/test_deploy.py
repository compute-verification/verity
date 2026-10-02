"""`research deploy`: each node's installed files against `main` (ok, stale, ahead, unknown, missing, restart, strays), and an
install only from a pushed commit, atomic, keeping owner and mode, with the old file kept and a record line. The node is this
machine: a runner that runs the probe and the installer locally, over files under a temporary directory."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research import deploy

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
FUTURE = 4102444800    # 2100-01-01: an mtime after any process here started


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=ENV).stdout.strip()


def _commit(repo: Path, files: dict[str, str], msg: str) -> str:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def local(node: str, argv: list[str], stdin: bytes | None) -> subprocess.CompletedProcess:
    argv = argv[2:] if argv[:2] == ["sudo", "-n"] else argv
    return subprocess.run([sys.executable if argv[0] == "python3" else argv[0], *argv[1:]], input=stdin, capture_output=True)


@pytest.fixture
def world(tmp_path):
    origin, repo, node = tmp_path / "origin.git", tmp_path / "repo", tmp_path / "node"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _git(repo, "remote", "add", "origin", str(origin))
    v1 = _commit(repo, {"src/a.py": "a1\n", "src/b.sh": "b1\n", "src/t/x.yaml": "x1\n"}, "v1")
    _commit(repo, {"src/a.py": "a2\n", "src/c.py": "c1\n"}, "v2")
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "checkout", "-q", "-b", "fix")
    _commit(repo, {"src/c.py": "c-fix\n"}, "a hotfix")
    _git(repo, "push", "-q", "origin", "fix")
    _git(repo, "checkout", "-q", "main")
    (node / "bin").mkdir(parents=True)
    (node / "tree").mkdir()
    manifest = tmp_path / "deploy.toml"
    manifest.write_text(f"""
[nodes.n]
record = "{node}/research/deploy/record.jsonl"
managed = ["{node}/bin/*"]

[[file]]
path = "{node}/bin/a.py"
src = "src/a.py"
nodes = ["n"]
run = ["tmux:a"]

[[file]]
path = "{node}/bin/b"
src = "src/b.sh"
nodes = ["n"]

[[file]]
path = "{node}/bin/c.py"
src = "src/c.py"
nodes = ["n"]

[[file]]
path = "{node}/bin/unit.service"
by = "src/setup.sh"
nodes = ["n"]

[[tree]]
path = "{node}/tree"
src = "src/t"
nodes = ["n"]
""")
    return {"repo": repo, "node": node, "manifest": manifest, "v1": v1, "nodes": deploy.load(manifest)}


def _states(rep: dict) -> dict[str, str]:
    return {Path(e["path"]).name: e["state"] for e in rep["entries"]}


def test_drift_names_each_files_state_and_the_strays(world):
    node, repo = world["node"], world["repo"]
    (node / "bin" / "a.py").write_text("a1\n")              # main's older version
    (node / "bin" / "b").write_text("b1\n")                 # main's
    (node / "bin" / "c.py").write_text("c-fix\n")           # the pushed hotfix main lacks
    (node / "bin" / "a.py.bak-20261001T0000Z").write_text("a0\n")
    (node / "tree" / "x.yaml.prev").write_text("x0\n")
    rep = deploy.drift(world["nodes"]["n"], "origin/main", local, repo)
    assert _states(rep) == {"a.py": "stale", "b": "ok", "c.py": "ahead", "unit.service": "missing", "x.yaml": "missing"}
    by = {Path(e["path"]).name: e for e in rep["entries"]}
    assert "fix" in by["c.py"]["detail"] and "not in origin/main" in by["c.py"]["detail"]
    assert world["v1"][:12] in by["a.py"]["detail"]
    assert sorted(Path(p).name for p in rep["strays"]) == ["a.py.bak-20261001T0000Z", "x.yaml.prev"]
    assert not deploy.clean(rep)
    text = deploy.render("n", "origin/main", rep)
    assert "stale    " in text and "1 files of the tree" in text and "stray    " in text

    (node / "bin" / "c.py").write_text("edited by hand\n")
    assert _states(deploy.drift(world["nodes"]["n"], "origin/main", local, repo))["c.py"] == "unknown"


def test_a_process_that_started_before_its_file_was_installed_needs_a_restart(world):
    a = world["node"] / "bin" / "a.py"
    a.write_text("a2\n")
    holder = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()", str(a)], stdin=subprocess.PIPE)
    try:
        os.utime(a, (FUTURE, FUTURE))
        rep = deploy.drift(world["nodes"]["n"], "origin/main", local, world["repo"])
    finally:
        holder.stdin.close()
        holder.wait()
    e = next(e for e in rep["entries"] if e["path"] == str(a))
    assert e["state"] == "ok" and "restart tmux:a" in e["restart"]


def test_install_refuses_an_unpushed_commit_and_installs_a_pushed_one_atomically_with_a_record(world, capsys):
    node, repo, n = world["node"], world["repo"], world["nodes"]["n"]
    b = node / "bin" / "b"
    b.write_text("b0\n")
    b.chmod(0o750)
    inode = b.stat().st_ino
    _commit(repo, {"src/a.py": "local only\n"}, "unpushed")
    with pytest.raises(RuntimeError, match="on no origin branch: push it first"):
        deploy.install(n, "HEAD", "lane-x", [], local, repo=repo)

    done = deploy.install(n, "origin/fix", "lane-x", [], local, repo=repo)
    assert {Path(f["path"]).name: f["changed"] for f in done} == {"a.py": True, "b": True, "c.py": True, "x.yaml": True}
    assert (node / "bin" / "c.py").read_text() == "c-fix\n" and b.read_text() == "b1\n"
    assert b.stat().st_mode & 0o777 == 0o750 and b.stat().st_ino != inode           # kept its mode; a new inode, renamed over
    assert [p.read_text() for p in (node / "bin" / ".deploy-prev").glob("b.*")] == ["b0\n"]
    assert (node / "tree" / "x.yaml").read_text() == "x1\n"
    rec = [json.loads(ln) for ln in (node / "research" / "deploy" / "record.jsonl").read_text().splitlines()]
    fix = _git(repo, "rev-parse", "origin/fix")
    assert {r["path"] for r in rec} == {f["path"] for f in done} and {(r["commit"], r["ref"], r["by"]) for r in rec} == {(fix, "origin/fix", "lane-x")}

    rep = deploy.drift(n, "origin/main", local, repo)
    c = next(e for e in rep["entries"] if e["path"].endswith("c.py"))
    assert c["state"] == "ahead" and "by lane-x from origin/fix" in c["detail"]

    assert all(f["changed"] is False for f in deploy.install(n, "origin/fix", "lane-x", [], local, repo=repo))
    assert deploy.main(["install", "n", str(node / "bin" / "c.py"), "--ref", "origin/main", "--by", "lane-x", "--no-fetch"],
                       run=local, manifest=world["manifest"], repo=repo) == 0
    assert "installed     " in capsys.readouterr().out and (node / "bin" / "c.py").read_text() == "c1\n"
    with pytest.raises(RuntimeError, match="not in deploy.toml for n: nope"):
        deploy.install(n, "origin/main", "lane-x", ["nope"], local, repo=repo)


def test_drift_is_clean_once_everything_is_mains_and_nothing_strays(world, capsys):
    node, n = world["node"], world["nodes"]["n"]
    deploy.install(n, "origin/main", "lane-x", [], local, repo=world["repo"])
    (node / "bin" / "unit.service").write_text("[Unit]\n")
    rep = deploy.drift(n, "origin/main", local, world["repo"])
    assert deploy.clean(rep), rep
    assert deploy.main(["drift", "--json", "--no-fetch"], run=local, manifest=world["manifest"], repo=world["repo"]) == 0
    assert json.loads(capsys.readouterr().out)["n"]["entries"]


def test_the_manifest_loads_and_every_source_is_a_file_of_the_repository():
    nodes = deploy.load()
    assert {"vy-nebius-1", "vy-nebius-2"} <= set(nodes)
    for node in nodes.values():
        assert node.record.startswith("/") and node.managed
        for e in node.entries:
            if e.src:
                assert (deploy.REPO / e.src).is_dir() if e.tree else (deploy.REPO / e.src).is_file(), e.src
            assert e.path.startswith("/") and all(r.split(":", 1)[0] in ("tmux", "systemd", "user-systemd") for r in e.run)


def test_a_manifest_entry_needs_exactly_one_of_src_and_by_and_a_known_node(tmp_path):
    m = tmp_path / "deploy.toml"
    m.write_text('[nodes.n]\nrecord = "/r"\n[[file]]\npath = "/p"\nsrc = "s"\nby = "b"\nnodes = ["n"]\n')
    with pytest.raises(ValueError, match="exactly one of src and by"):
        deploy.load(m)
    m.write_text('[nodes.n]\nrecord = "/r"\n[[file]]\npath = "/p"\nsrc = "s"\nnodes = ["m"]\n')
    with pytest.raises(ValueError, match="has no \\[nodes.m\\]"):
        deploy.load(m)
