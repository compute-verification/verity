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
    _git(repo, "checkout", "-q", "fix")
    _commit(repo, {"src/t/new.py": "a PR adds it\n"}, "a PR's new file")
    _git(repo, "push", "-q", "origin", "fix")
    _git(repo, "checkout", "-q", "main")
    (node / "tree" / "new.py").write_text("a PR adds it\n")
    rep = deploy.drift(world["nodes"]["n"], "origin/main", local, repo)
    assert _states(rep) == {"a.py": "stale", "b": "ok", "c.py": "ahead", "unit.service": "missing", "x.yaml": "missing", "new.py": "ahead"}
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

    def no_cwd(name: str, argv: list[str], stdin: bytes | None) -> subprocess.CompletedProcess:
        # another user's process (root's, under sudo): /proc/PID/cwd is unreadable
        hide = "import os\n_rl = os.readlink\ndef _no(p):\n    if p.endswith('/cwd'): raise PermissionError(p)\n    return _rl(p)\nos.readlink = _no\n"
        return local(name, [*argv[:2], hide + argv[2], *argv[3:]] if argv[2] == deploy.PROBE else argv, stdin)

    holder = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()", str(a)], stdin=subprocess.PIPE)
    try:
        os.utime(a, (FUTURE, FUTURE))
        reps = [deploy.drift(world["nodes"]["n"], "origin/main", run, world["repo"]) for run in (local, no_cwd)]
    finally:
        holder.stdin.close()
        holder.wait()
    for rep in reps:
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


def test_a_dry_run_says_what_would_change_and_an_install_names_only_what_still_runs_old_code(world, capsys):
    node, repo, n = world["node"], world["repo"], world["nodes"]["n"]
    deploy.install(n, "origin/main", "lane-x", [], local, repo=repo)
    _commit(repo, {"src/a.py": "a3\n", "src/c.py": "c3\n"}, "v3")
    _git(repo, "push", "-q", "origin", "main")
    dry = deploy.install(n, "origin/main", "lane-x", [], local, dry=True, repo=repo)
    assert {Path(f["path"]).name: f["changed"] for f in dry} == {"a.py": None, "b": False, "c.py": None, "x.yaml": False}
    assert (node / "bin" / "a.py").read_text() == "a2\n"

    def installed_later(name: str, argv: list[str], stdin: bytes | None) -> subprocess.CompletedProcess:
        r = local(name, argv, stdin)
        if argv[:2] == ["sudo", "-n"]:
            for f in json.loads(stdin)["files"]:
                os.utime(f["path"], (FUTURE, FUTURE))
        return r

    holder = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()", str(node / "bin" / "a.py")], stdin=subprocess.PIPE)
    try:
        assert deploy.main(["install", "n", "--by", "lane-x", "--no-fetch"], run=installed_later, manifest=world["manifest"], repo=repo) == 0
    finally:
        holder.stdin.close()
        holder.wait()
    out = capsys.readouterr().out
    assert "restart when safe: tmux:a (" in out and (node / "bin" / "c.py").read_text() == "c3\n"


def test_attic_moves_only_strays_none_a_process_runs_and_records_where_each_went(world, capsys):
    node, n = world["node"], world["nodes"]["n"]
    deploy.install(n, "origin/main", "lane-x", [], local, repo=world["repo"])
    (node / "bin" / "unit.service").write_text("[Unit]\n")
    bak, one_off = node / "bin" / "a.py.bak-20261001T0611Z", node / "bin" / "probe-once.sh"
    bak.write_text("a0\n")
    one_off.write_text("echo\n")
    old_tree = node / "tree" / "jobs.bak-20260930T2115Z"
    old_tree.mkdir()
    (old_tree / "x.yaml").write_text("x0\n")
    with pytest.raises(RuntimeError, match="not a stray of n"):
        deploy.attic(n, "origin/main", "lane-x", [str(node / "bin" / "a.py")], False, local, repo=world["repo"])
    holder = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()", str(one_off)], stdin=subprocess.PIPE)
    try:
        with pytest.raises(RuntimeError, match="a process runs"):
            deploy.attic(n, "origin/main", "lane-x", [str(one_off)], False, local, repo=world["repo"])
    finally:
        holder.stdin.close()
        holder.wait()
    assert deploy.attic(n, "origin/main", "lane-x", [], True, local, dry=True, repo=world["repo"]) == [(str(bak), None),
                                                                                                       (str(old_tree / "x.yaml"), None)]

    assert deploy.main(["attic", "n", str(one_off), "--backups", "--by", "lane-x", "--no-fetch"], run=local,
                       manifest=world["manifest"], repo=world["repo"]) == 0
    assert "restore one with `sudo mv DEST PATH`" in capsys.readouterr().out
    assert not bak.exists() and not one_off.exists() and not old_tree.exists() and (node / "tree").is_dir()
    rec = [r for ln in (node / "research" / "deploy" / "record.jsonl").read_text().splitlines() if "attic" in (r := json.loads(ln))]
    assert {r["path"] for r in rec} == {str(bak), str(one_off), str(old_tree / "x.yaml")} and all(r["by"] == "lane-x" for r in rec)
    assert Path(next(r["attic"] for r in rec if r["path"] == str(bak))).read_text() == "a0\n"
    assert deploy.clean(deploy.drift(n, "origin/main", local, world["repo"]))


def test_drift_is_clean_once_everything_is_mains_and_nothing_strays(world, capsys):
    node, n = world["node"], world["nodes"]["n"]
    deploy.install(n, "origin/main", "lane-x", [], local, repo=world["repo"])
    (node / "bin" / "unit.service").write_text("[Unit]\n")
    rep = deploy.drift(n, "origin/main", local, world["repo"])
    assert deploy.clean(rep), rep
    assert deploy.main(["drift", "--json", "--no-fetch"], run=local, manifest=world["manifest"], repo=world["repo"]) == 0
    assert json.loads(capsys.readouterr().out)["n"]["entries"]


def test_an_entry_whose_source_the_ref_lacks_is_absent_and_a_ref_deploys_its_own_manifest(world):
    node, repo, manifest = world["node"], world["repo"], world["manifest"]
    _git(repo, "checkout", "-q", "fix")
    _commit(repo, {"src/d.py": "d1\n"}, "a new script")
    _git(repo, "push", "-q", "origin", "fix")
    _git(repo, "checkout", "-q", "main")
    manifest.write_text(manifest.read_text() + f'\n[[file]]\npath = "{node}/bin/d.py"\nsrc = "src/d.py"\nnodes = ["n"]\n')
    n = deploy.load(manifest)["n"]
    d = next(e for e in deploy.drift(n, "origin/main", local, repo)["entries"] if e["path"].endswith("d.py"))
    assert d["state"] == "absent" and "origin/main lacks src/d.py" in d["detail"]
    with pytest.raises(RuntimeError, match="lacks the source of src/d.py"):
        deploy.install(n, "origin/main", "lane-x", ["src/d.py"], local, repo=repo)
    done = deploy.install(n, "origin/main", "lane-x", [], local, repo=repo)
    assert [f["changed"] for f in done if f["path"].endswith("d.py")] == ["absent"] and not (node / "bin" / "d.py").exists()
    deploy.install(n, "origin/fix", "lane-x", ["src/d.py"], local, repo=repo)
    assert (node / "bin" / "d.py").read_text() == "d1\n"

    assert deploy.manifest_at("origin/main", repo) is None
    rel = deploy.MANIFEST.relative_to(deploy.REPO)
    _commit(repo, {str(rel): "[nodes.m]\nrecord = '/r'\n"}, "a manifest")
    assert set(deploy.load(text=deploy.manifest_at("HEAD", repo))) == {"m"}


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
