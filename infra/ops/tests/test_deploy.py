"""`research deploy`: each node's installed files against `main` (ok, stale, ahead, unknown, missing, held, restart, strays), and
an install only from a pushed commit, atomic, keeping owner and mode, with the old file kept and a record line, restarting the
units it names; and the checked-in manifest's own rules (units only, an owner each, units' files installed beside them). The node
is this machine: a runner that runs the probe and the installer locally, over files under a temporary directory."""
from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from verity_ops import deploy

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
run = ["systemd:a"]

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
        assert e["state"] == "ok" and "restart systemd:a" in e["restart"]


def test_a_file_installed_from_a_commit_main_has_since_merged_and_changed_is_stale_not_ahead(world, monkeypatch):
    """Node 1's custody tree ran a store/local.py from 6 Oct that drift called `ahead` (its record named the commit) for two days
    after main had merged that commit and changed the file (#1402), because the stale search reads only main's last HISTORY
    commits and the record was taken at its word."""
    node, repo, n = world["node"], world["repo"], world["nodes"]["n"]
    deploy.install(n, "origin/fix", "lane-x", [str(node / "bin" / "c.py")], local, apply=True, repo=repo)
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge fix", "origin/fix")
    _commit(repo, {"src/c.py": "c3\n"}, "main changes c.py")
    _commit(repo, {"src/b.sh": "b2\n"}, "and moves on")
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "push", "-q", "origin", "--delete", "fix")
    _git(repo, "fetch", "-q", "--prune", "origin")
    monkeypatch.setattr(deploy, "HISTORY", 1)
    c = next(e for e in deploy.drift(n, "origin/main", local, repo)["entries"] if e["path"].endswith("c.py"))
    assert c["state"] == "stale" and "which origin/main holds" in c["detail"]

    (node / "bin" / "c.py").write_text("c-fix\n")
    (node / "research" / "deploy" / "record.jsonl").unlink()
    c = next(e for e in deploy.drift(n, "origin/main", local, repo)["entries"] if e["path"].endswith("c.py"))
    assert c["state"] == "stale", "no record: the recent search finds the blob in main's own history"


def test_install_refuses_an_unpushed_commit_and_installs_a_pushed_one_atomically_with_a_record(world, capsys):
    node, repo, n = world["node"], world["repo"], world["nodes"]["n"]
    b = node / "bin" / "b"
    b.write_text("b0\n")
    b.chmod(0o750)
    inode = b.stat().st_ino
    _commit(repo, {"src/a.py": "local only\n"}, "unpushed")
    with pytest.raises(deploy.Refused, match="on no origin branch: push it first"):
        deploy.install(n, "HEAD", "lane-x", [], local, apply=True, repo=repo)
    assert deploy.main(["install", "n", "--ref", "HEAD", "--by", "lane-x", "--apply", "--no-fetch"], run=local,
                       manifest=world["manifest"], repo=repo) == 3
    assert "push it first" in capsys.readouterr().err

    done = deploy.install(n, "origin/fix", "lane-x", [], local, apply=True, repo=repo)["files"]
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

    assert all(f["changed"] is False for f in deploy.install(n, "origin/fix", "lane-x", [], local, apply=True, repo=repo)["files"])
    assert deploy.main(["install", "n", "--ref", "origin/main", "--by", "lane-x", str(node / "bin" / "c.py"), "--apply",
                        "--no-fetch"], run=local, manifest=world["manifest"], repo=repo) == 0
    assert "installed     " in capsys.readouterr().out and (node / "bin" / "c.py").read_text() == "c1\n"
    assert deploy.main(["install", "n", "--by", "lane-x", "--ref", "origin/fix", str(node / "bin" / "c.py"), "--apply", "--no-fetch"],
                       run=local, manifest=world["manifest"], repo=repo) == 0
    assert (node / "bin" / "c.py").read_text() == "c-fix\n"
    with pytest.raises(SystemExit):
        deploy.main(["install", "n", "--by", "lane-x", str(node / "bin" / "c.py"), "--bogus", "--no-fetch"],
                    run=local, manifest=world["manifest"], repo=repo)
    with pytest.raises(deploy.Refused, match="not in deploy.toml for n: nope"):
        deploy.install(n, "origin/main", "lane-x", ["nope"], local, apply=True, repo=repo)


def test_a_dry_run_says_what_would_change_and_an_install_names_only_what_still_runs_old_code(world, capsys):
    node, repo, n = world["node"], world["repo"], world["nodes"]["n"]
    deploy.install(n, "origin/main", "lane-x", [], local, apply=True, repo=repo)
    _commit(repo, {"src/a.py": "a3\n", "src/c.py": "c3\n"}, "v3")
    _git(repo, "push", "-q", "origin", "main")
    dry = deploy.install(n, "origin/main", "lane-x", [], local, repo=repo)["files"]
    assert {Path(f["path"]).name: f["changed"] for f in dry} == {"a.py": None, "b": False, "c.py": None, "x.yaml": False}
    assert deploy.main(["install", "n", "--by", "lane-x", "--no-fetch"], run=local, manifest=world["manifest"], repo=repo) == 0
    assert "nothing changed: --apply installs and restarts" in capsys.readouterr().out
    assert (node / "bin" / "a.py").read_text() == "a2\n"

    def installed_later(name: str, argv: list[str], stdin: bytes | None) -> subprocess.CompletedProcess:
        r = local(name, argv, stdin)
        if argv[:2] == ["sudo", "-n"]:
            for f in json.loads(stdin)["files"]:
                os.utime(f["path"], (FUTURE, FUTURE))
        return r

    holder = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()", str(node / "bin" / "a.py")], stdin=subprocess.PIPE)
    try:
        assert deploy.main(["install", "n", "--by", "lane-x", "--apply", "--no-fetch"], run=installed_later,
                           manifest=world["manifest"], repo=repo) == 0
    finally:
        holder.stdin.close()
        holder.wait()
    out = capsys.readouterr().out
    # the holder runs in no unit a.py's entry names, so the path is what to restart
    assert f"restart when safe: {node / 'bin' / 'a.py'} (" in out and (node / "bin" / "c.py").read_text() == "c3\n"


def test_installing_a_changed_unit_reloads_systemd_once_and_an_unchanged_one_does_not(world, monkeypatch):
    node, repo, manifest = world["node"], world["repo"], world["manifest"]
    monkeypatch.setattr(deploy, "UNITS", f"{node}/units/")
    _commit(repo, {"src/x.service": "[Service]\nAllowedCPUs=0-7\n", "src/x.timer": "[Timer]\n"}, "units")
    _git(repo, "push", "-q", "origin", "main")
    manifest.write_text(manifest.read_text() + "".join(f'\n[[file]]\npath = "{node}/units/x.{k}"\nsrc = "src/x.{k}"\nnodes = ["n"]\n'
                                                       for k in ("service", "timer")))
    n, calls, rc = deploy.load(manifest)["n"], [], [0]

    def systemd(name: str, argv: list[str], stdin: bytes | None) -> subprocess.CompletedProcess:
        if "systemctl" in argv:
            calls.append(argv)
            return subprocess.CompletedProcess(argv, rc[0], b"", b"" if rc[0] == 0 else b"Access denied")
        return local(name, argv, stdin)

    deploy.install(n, "origin/main", "lane-x", [], systemd, repo=repo)
    assert calls == [] and not (node / "units" / "x.service").exists()                         # a dry run
    deploy.install(n, "origin/main", "lane-x", [], systemd, apply=True, repo=repo)
    assert calls == [["sudo", "-n", "systemctl", "daemon-reload"]] and (node / "units" / "x.service").read_text().endswith("0-7\n")
    deploy.install(n, "origin/main", "lane-x", [], systemd, apply=True, repo=repo)
    _commit(repo, {"src/c.py": "c4\n"}, "not a unit")
    _git(repo, "push", "-q", "origin", "main")
    deploy.install(n, "origin/main", "lane-x", [], systemd, apply=True, repo=repo)
    assert len(calls) == 1

    _commit(repo, {"src/x.timer": "[Timer]\nOnUnitActiveSec=5min\n"}, "a timer change")
    _git(repo, "push", "-q", "origin", "main")
    rc[0] = 1
    with pytest.raises(RuntimeError, match=r"installed .*x\.timer, but `systemctl daemon-reload` failed \(exit 1\): Access denied"):
        deploy.install(n, "origin/main", "lane-x", [], systemd, apply=True, repo=repo)


def test_attic_moves_only_strays_none_a_process_runs_and_records_where_each_went(world, capsys):
    node, n = world["node"], world["nodes"]["n"]
    deploy.install(n, "origin/main", "lane-x", [], local, apply=True, repo=world["repo"])
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
    assert deploy.attic(n, "origin/main", "lane-x", [], True, local, repo=world["repo"]) == [(str(bak), None),
                                                                                             (str(old_tree / "x.yaml"), None)]
    assert deploy.main(["attic", "n", "--backups", "--by", "lane-x", "--no-fetch"], run=local, manifest=world["manifest"],
                       repo=world["repo"]) == 0
    assert "nothing moved: --apply moves them" in capsys.readouterr().out and bak.exists()
    assert deploy.main(["attic", "n", str(bak.with_name("nope")), "--by", "lane-x", "--no-fetch"], run=local,
                       manifest=world["manifest"], repo=world["repo"]) == 3

    with pytest.raises(SystemExit):
        deploy.main(["attic", "n", "--by", "lane-x", str(one_off), "--bogus", "--no-fetch"], run=local, manifest=world["manifest"],
                    repo=world["repo"])
    assert deploy.main(["attic", "n", "--backups", "--by", "lane-x", str(one_off), "--apply", "--no-fetch"], run=local,
                       manifest=world["manifest"], repo=world["repo"]) == 0
    assert "restore one with `sudo mv DEST PATH`" in capsys.readouterr().out
    assert not bak.exists() and not one_off.exists() and not old_tree.exists() and (node / "tree").is_dir()
    rec = [r for ln in (node / "research" / "deploy" / "record.jsonl").read_text().splitlines() if "attic" in (r := json.loads(ln))]
    assert {r["path"] for r in rec} == {str(bak), str(one_off), str(old_tree / "x.yaml")} and all(r["by"] == "lane-x" for r in rec)
    assert Path(next(r["attic"] for r in rec if r["path"] == str(bak))).read_text() == "a0\n"
    assert deploy.clean(deploy.drift(n, "origin/main", local, world["repo"]))


def test_a_missing_file_of_a_required_tree_is_drift(world):
    # red team T: the console imports cluster.usage from its tree, and drift called a console without it clean
    node = world["node"]
    deploy.install(world["nodes"]["n"], "origin/main", "lane-x", [], local, apply=True, repo=world["repo"])
    (node / "bin" / "unit.service").write_text("[Unit]\n")
    (node / "tree" / "x.yaml").unlink()
    assert deploy.clean(deploy.drift(world["nodes"]["n"], "origin/main", local, world["repo"]))
    text = world["manifest"].read_text().replace('src = "src/t"\n', 'src = "src/t"\nrequired = true\n')
    rep = deploy.drift(deploy.load(text=text)["n"], "origin/main", local, world["repo"])
    assert not deploy.clean(rep)
    assert f"  missing  {node}/tree/x.yaml  (in a required tree, not installed: install the tree)" in deploy.render(
        "n", "origin/main", rep).splitlines()
    for bad in (text.replace("required = true", 'required = "yes"'), text.replace('src = "src/c.py"\n', 'src = "src/c.py"\nrequired = true\n')):
        with pytest.raises(ValueError, match="`required` is a tree's"):
            deploy.load(text=bad)
    shipped = deploy.load()["vy-nebius-1"].entries
    assert next(e for e in shipped if e.path == "/workspace/research/console/cluster").required


def test_drift_is_clean_once_everything_is_mains_and_nothing_strays(world, capsys):
    node, n = world["node"], world["nodes"]["n"]
    deploy.install(n, "origin/main", "lane-x", [], local, apply=True, repo=world["repo"])
    (node / "bin" / "unit.service").write_text("[Unit]\n")
    rep = deploy.drift(n, "origin/main", local, world["repo"])
    assert deploy.clean(rep), rep
    assert deploy.main(["drift", "--json", "--no-fetch"], run=local, manifest=world["manifest"], repo=world["repo"]) == 0
    assert json.loads(capsys.readouterr().out)["n"]["entries"]
    assert deploy.main(["drift", "--node", "zz", "--node", "n", "--no-fetch"], run=local, manifest=world["manifest"],
                       repo=world["repo"]) == 2                                   # a repeated --on adds, never replaces
    err = capsys.readouterr().err                                                  # --node: the old spelling, a hidden alias
    assert "no node 'zz'" in err and err.count("research deploy: --node is now --on") == 1


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
    with pytest.raises(deploy.Refused, match="lacks the source of src/d.py"):
        deploy.install(n, "origin/main", "lane-x", ["src/d.py"], local, apply=True, repo=repo)
    done = deploy.install(n, "origin/main", "lane-x", [], local, apply=True, repo=repo)["files"]
    assert [f["changed"] for f in done if f["path"].endswith("d.py")] == ["absent"] and not (node / "bin" / "d.py").exists()
    deploy.install(n, "origin/fix", "lane-x", ["src/d.py"], local, apply=True, repo=repo)
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
            assert e.path.startswith("/") and all(r.split(":", 1)[0] in ("systemd", "user-systemd") for r in e.run)


def test_every_switched_unit_is_one_deploy_installs_on_its_node():
    """The lock-out switches each tmux loop to its unit: every unit it starts, and every file of its own, is one deploy.toml installs."""
    import importlib.util

    from research.pods import registry

    lockout = Path(registry.__file__).resolve().parent / "nebius" / "lockout"
    spec = importlib.util.spec_from_file_location("vy_lockout", lockout / "lockout.py")
    LO = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("vy_lockout", LO)
    spec.loader.exec_module(LO)
    nodes = deploy.load()
    for node, loops in LO.SWITCH.items():
        paths = {e.path for e in nodes[node].entries}
        for lp in loops:
            for u in {lp.unit, LO.service_of(lp.unit)}:
                assert f"/etc/systemd/system/{LO.unit_file(u)}" in paths, (node, u)
    srcs = {e.src for n in nodes.values() for e in n.entries if e.src}
    files = [p for p in lockout.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "README.md"]
    assert files and not [p for p in files if p.relative_to(deploy.REPO).as_posix() not in srcs], "deploy.toml installs every file"


def test_the_manifest_runs_nothing_in_tmux_and_each_entry_has_an_owner_handle():
    """A tmux session is restarted by hand, typed into, and gone after a reboot: what deploy.toml installs runs under a unit."""
    m = tomllib.loads(deploy.MANIFEST.read_text())
    entries = m["file"] + m["tree"]
    assert not [(e["path"], r) for e in entries for r in e.get("run", []) if not r.startswith(("systemd:", "user-systemd:"))]
    registry = json.loads((deploy.REPO / ".agents/skills/using-slack/registry.json").read_text())
    handles = {h["handle"] for h in registry["handles"]}
    assert not [(e["path"], e.get("owner")) for e in entries if e.get("owner") not in handles]


PODS = deploy.MANIFEST.parent
ROOT_UNITS = {"vy-write-probe@.service": "reads every thread's /proc/PID/task/TID/io, as its tmux pane did under sudo -n"}


def _keys(path: Path) -> list[str]:
    return [ln.split("=", 1)[0] for ln in path.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def test_each_unit_is_installed_with_the_environment_files_and_scripts_it_names_on_its_nodes():
    nodes = deploy.load()
    srcs = {e.src for n in nodes.values() for e in n.entries if e.src}
    files = sorted(p.relative_to(deploy.REPO).as_posix() for d in ("units", "env") for p in (PODS / d).iterdir())
    assert files and not [f for f in files if f not in srcs]
    seen = 0
    for node in nodes.values():
        paths = {e.path for e in node.entries if not e.tree}
        trees = [e.path.rstrip("/") + "/" for e in node.entries if e.tree]
        for e in node.entries:
            if not (e.src or "").startswith(f"{PODS.relative_to(deploy.REPO).as_posix()}/units/"):
                continue
            text, base = (deploy.REPO / e.src).read_text(), Path(e.path).name
            assert "Restart=on-failure" in text and (base in ROOT_UNITS or "User=research" in text or
                                                     e.path.startswith(deploy.USER_UNITS)), e.path
            units = [deploy.unit_name(r) for r in e.run]
            assert units and all(u and deploy.unit_file(u) == e.path for u in units), (e.path, e.run)
            for u in units:
                name = u.split(":", 1)[1].rsplit(".", 1)[0]
                inst = name.partition("@")[2]
                resolve = lambda s: s.replace("%N", name).replace("%i", inst)
                envs = [resolve(ln.split("=", 1)[1].lstrip("-")) for ln in text.splitlines() if ln.startswith("EnvironmentFile=")]
                scripts = [resolve(s) for ln in text.splitlines() if ln.startswith(("ExecStart=", "ExecStartPre="))
                           for s in re.findall(r"/workspace/[\w./@-]+\.(?:py|sh)\b", ln)]
                assert scripts, u
                for f in [x for x in envs if x != "/etc/environment"] + scripts:
                    assert f in paths or f.startswith(tuple(trees)), f"{node.name}: {u} reads {f}, which deploy.toml doesn't install there"
                seen += 1
    assert seen >= 12


def test_environment_files_hold_settings_never_credentials_nor_what_dispatch_env_sets():
    for f in sorted((PODS / "env").iterdir()):
        for k in _keys(f):
            assert re.fullmatch(r"[A-Z_][A-Z0-9_]*", k), (f.name, k)
            assert not re.search(r"TOKEN|SECRET|PASS|CRED|AUTH|(^|_)KEY($|_)", k), (f.name, k)
    assert _keys(PODS / "dispatch-n1.env")
    assert not set(_keys(PODS / "env" / "vy-node1-dispatch.env")) & set(_keys(PODS / "dispatch-n1.env"))


FAKE_CGROUP = """import builtins, io
def open(p, *a, _o=builtins.open, **k):
    return io.StringIO({cg!r}) if str(p).endswith("/cgroup") else _o(p, *a, **k)
"""


def _units(world, monkeypatch, active=("s.service", "u.service"), cgroup: list[str] | None = None):
    """The world with units: s (its unit, its environment file, and s.py its process runs), t (stopped), u (a user unit), the
    timer s.timer and a setup script's g.service, all named by s.env's entry; node n's `quiet` is that {node}/quiet exists. The
    probe sees every process in the cgroup `cgroup[0]`, when given."""
    node, repo, manifest = world["node"], world["repo"], world["manifest"]
    for k, d in (("UNITS", "units"), ("USER_UNITS", "user"), ("ENVS", "env")):
        monkeypatch.setattr(deploy, k, f"{node}/{d}/")
    _commit(repo, {"src/s.service": "[Service]\n", "src/t.service": "[Service]\n", "src/u.service": "[Service]\n",
                   "src/s.timer": "[Timer]\n", "src/s.env": "A=1\n", "src/s.py": "s1\n"}, "units")
    _git(repo, "push", "-q", "origin", "main")
    entries = [("units/s.service", "s.service", ["systemd:s"]), ("units/t.service", "t.service", ["systemd:t"]),
               ("user/u.service", "u.service", ["user-systemd:u"]), ("units/s.timer", "s.timer", ["systemd:s.timer"]),
               ("env/s.env", "s.env", ["systemd:s", "systemd:s.timer", "systemd:g"]), ("bin/s.py", "s.py", ["systemd:s"])]
    text = manifest.read_text().replace("\nmanaged = ", f'\nquiet = "test -e {node}/quiet"\nmanaged = ', 1)
    text += "".join(f'\n[[file]]\npath = "{node}/{p}"\nsrc = "src/{s}"\nnodes = ["n"]\nrun = {json.dumps(r)}\n' for p, s, r in entries)
    manifest.write_text(text + f'\n[[file]]\npath = "{node}/units/g.service"\nby = "src/setup.sh"\nnodes = ["n"]\n')
    calls: list[list[str]] = []

    def run(name: str, argv: list[str], stdin: bytes | None) -> subprocess.CompletedProcess:
        if "systemctl" in argv:
            calls.append(argv)
            down = "is-active" in argv and argv[-1] not in active
            return subprocess.CompletedProcess(argv, 3 if down else 0, b"", b"")
        if cgroup and len(argv) > 2 and argv[2] == deploy.PROBE:
            argv = [*argv[:2], FAKE_CGROUP.format(cg=f"0::{cgroup[0]}\n") + argv[2], *argv[3:]]
        r = local(name, argv, stdin)
        if argv[:2] == ["sudo", "-n"] and stdin and "files" in (req := json.loads(stdin)):
            for f in req["files"]:
                os.utime(f["path"], (FUTURE, FUTURE))
        return r
    return deploy.load(manifest)["n"], run, calls


def _restarts(node: Path) -> list[dict]:
    return [r for ln in (node / "research" / "deploy" / "record.jsonl").read_text().splitlines() if "unit" in (r := json.loads(ln))]


def test_install_restarts_each_running_service_whose_unit_or_environment_changed_once_the_node_is_quiet(world, monkeypatch, capsys):
    node, repo = world["node"], world["repo"]
    n, run, calls = _units(world, monkeypatch)
    (node / "quiet").write_text("")
    dry = deploy.install(n, "origin/main", "lane-x", [], run, repo=repo)
    assert {d["unit"]: d["state"] for d in dry["restarts"]} == {"systemd:s.service": "would restart",
                                                               "systemd:t.service": "not running",
                                                               "user-systemd:u.service": "would restart"}
    assert not [c for c in calls if "is-active" not in c or "sudo" in c] and not (node / "units" / "s.service").exists()

    calls.clear()
    assert deploy.main(["install", "n", "--by", "lane-x", "--apply", "--no-fetch"], run=run, manifest=world["manifest"],
                       repo=repo) == 0
    assert calls == [["sudo", "-n", "systemctl", "daemon-reload"], ["systemctl", "--user", "daemon-reload"],
                     ["systemctl", "is-active", "--quiet", "s.service"],
                     ["sudo", "-n", "systemctl", "try-restart", "s.service"],
                     ["systemctl", "is-active", "--quiet", "t.service"],
                     ["systemctl", "--user", "is-active", "--quiet", "u.service"],
                     ["systemctl", "--user", "try-restart", "u.service"]]               # never s.timer, g.service, nor t's start
    out = capsys.readouterr().out
    assert "restarted     systemd:s.service" in out and "not running   systemd:t.service" in out and "it stays stopped" in out
    head = _git(repo, "rev-parse", "origin/main")
    assert [(r["unit"], r["restart"], r["why"], r["commit"], r["by"]) for r in _restarts(node)] == [
        ("systemd:s.service", "restarted", [f"{node}/env/s.env", f"{node}/units/s.service"], head, "lane-x"),
        ("user-systemd:u.service", "restarted", [f"{node}/user/u.service"], head, "lane-x")]

    (node / "quiet").unlink()
    _commit(repo, {"src/s.env": "A=2\n"}, "a setting")
    _git(repo, "push", "-q", "origin", "main")
    calls.clear()
    done = deploy.install(n, "origin/main", "lane-x", [], run, apply=True, repo=repo)
    assert done["restarts"] == [{"unit": "systemd:s.service", "why": [f"{node}/env/s.env"], "state": "not quiet"}]
    assert calls == [] and (node / "env" / "s.env").read_text() == "A=2\n" and len(_restarts(node)) == 2


def test_install_restarts_a_service_whose_process_runs_the_old_script_and_not_one_outside_it(world, monkeypatch):
    node, repo, cgroup = world["node"], world["repo"], [""]
    (node / "quiet").write_text("")
    n, run, calls = _units(world, monkeypatch, cgroup=cgroup)
    deploy.install(n, "origin/main", "lane-x", [], run, apply=True, repo=repo)
    for cgroup[0], restarted in (("/system.slice/s.service", True), ("/user.slice/user-1000.slice/session-3.scope", False)):
        _commit(repo, {"src/s.py": f"s-{restarted}\n"}, "a script change")
        _git(repo, "push", "-q", "origin", "main")
        holder = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()", str(node / "bin" / "s.py")],
                                  stdin=subprocess.PIPE)
        try:
            done = deploy.install(n, "origin/main", "lane-x", [], run, apply=True, repo=repo)
        finally:
            holder.stdin.close()
            holder.wait()
        assert [(d["unit"], d["why"], d["state"]) for d in done["restarts"]] == (
            [("systemd:s.service", [f"{node}/bin/s.py"], "restarted")] if restarted else [])
        s = next(f for f in done["files"] if f["path"].endswith("s.py"))
        assert s["run"] == ["systemd:s.service" if restarted else str(node / "bin" / "s.py")]


def test_a_held_entry_is_reported_and_skipped_and_refused_when_named(world, capsys):
    node, repo, manifest = world["node"], world["repo"], world["manifest"]
    _commit(repo, {"src/h.py": "h1\n"}, "a held script")
    _git(repo, "push", "-q", "origin", "main")
    manifest.write_text(manifest.read_text() + f'\n[[file]]\npath = "{node}/bin/h.py"\nsrc = "src/h.py"\nnodes = ["n"]\n'
                                               f'hold = "its owner\'s yes pending"\n')
    n = deploy.load(manifest)["n"]
    done = deploy.install(n, "origin/main", "lane-x", [], local, apply=True, repo=repo)
    assert [(f["changed"], f["hold"]) for f in done["files"] if f["path"].endswith("h.py")] == [("held", "its owner's yes pending")]
    assert not (node / "bin" / "h.py").exists()
    (node / "bin" / "unit.service").write_text("[Unit]\n")
    rep = deploy.drift(n, "origin/main", local, repo)
    h = next(e for e in rep["entries"] if e["path"].endswith("h.py"))
    assert h["state"] == "held" and h["detail"] == "its owner's yes pending (not installed)" and deploy.clean(rep)
    with pytest.raises(deploy.Refused, match="held in deploy.toml"):
        deploy.install(n, "origin/main", "lane-x", ["src/h.py"], local, apply=True, repo=repo)
    assert deploy.main(["install", "n", "src/h.py", "--by", "lane-x", "--apply", "--no-fetch"], run=local, manifest=manifest,
                       repo=repo) == 3
    assert "A PR that drops its `hold` comes first" in capsys.readouterr().err and not (node / "bin" / "h.py").exists()

    manifest.write_text(f'[nodes.n]\nrecord = "/r"\n[[file]]\npath = "/p"\nby = "s"\nnodes = ["n"]\nhold = "x"\n')
    with pytest.raises(ValueError, match="is held, which only an entry install copies"):
        deploy.load(manifest)


def test_restarts_follow_the_changed_file_and_the_services_running_it():
    n = deploy.Node("n", "/r", [], [deploy.Entry(f"{deploy.UNITS}s.service", "src/s.service", run=["systemd:s"]),
                                    deploy.Entry(f"{deploy.UNITS}h@.service", "src/h@.service", run=["systemd:h@a"], hold="x"),
                                    deploy.Entry("/bin/s.py", "src/s.py", run=["systemd:s", "systemd:h@a", "systemd:s.timer"])])
    py = {"/bin/s.py": n.entries[2]}
    assert deploy.restarts(n, py, {"/bin/s.py": ["systemd:s.service", "systemd:h@a.service"]}) == {"systemd:s.service": ["/bin/s.py"]}
    assert deploy.restarts(n, py, {"/bin/s.py": [None, "systemd:other.service"]}) == {}
    assert deploy.unit_name("systemd:h@a") == "systemd:h@a.service" and deploy.unit_file("systemd:h@a.service") == f"{deploy.UNITS}h@.service"
    assert deploy.unit_name("tmux:s") is None and deploy.unit_name("user-systemd:p.timer") == "user-systemd:p.timer"


def test_a_manifest_entry_needs_exactly_one_of_src_and_by_and_a_known_node(tmp_path):
    m = tmp_path / "deploy.toml"
    m.write_text('[nodes.n]\nrecord = "/r"\n[[file]]\npath = "/p"\nsrc = "s"\nby = "b"\nnodes = ["n"]\n')
    with pytest.raises(ValueError, match="exactly one of src and by"):
        deploy.load(m)
    m.write_text('[nodes.n]\nrecord = "/r"\n[[file]]\npath = "/p"\nsrc = "s"\nnodes = ["m"]\n')
    with pytest.raises(ValueError, match="has no \\[nodes.m\\]"):
        deploy.load(m)


# the Lean sandbox, its setup and the daily cache units (test_nebius_lean_sandbox.py tests what they do)
_N = "tools/research/src/research/pods/nebius"
LEAN_CACHE = {"/usr/local/sbin/vy-lean-build-setup": f"{_N}/lean_build_setup.sh", "/usr/local/bin/vy-lean-sandbox": f"{_N}/lean_sandbox.py",
              "/usr/local/lib/vy-lean-cache/run-daily": "tools/verity/lean/run_daily.py",
              "/usr/local/lib/vy-lean-cache/cache.py": "tools/verity/lean/cache.py",
              "/etc/systemd/system/vy-lean-cache-daily.service": f"{_N}/vy-lean-cache-daily.service",
              "/etc/systemd/system/vy-lean-cache-daily.timer": f"{_N}/vy-lean-cache-daily.timer"}


def test_deploy_puts_the_sandbox_its_setup_and_the_daily_units_on_node_1_only_and_starts_nothing():
    nodes = deploy.load()
    n1 = {e.path: e for e in nodes["vy-nebius-1"].entries}
    for path, src in LEAN_CACHE.items():
        assert n1[path].src == src and n1[path].run == [], path
        assert any(fnmatch.fnmatch(path, m) for m in nodes["vy-nebius-1"].managed), path
    assert not set(LEAN_CACHE) & {e.path for e in nodes["vy-nebius-2"].entries}
    for src in (deploy.REPO / _N / "lean_build_setup.sh", deploy.REPO / _N / "lean_sandbox.py", deploy.REPO / "tools/verity/lean/run_daily.py"):
        assert os.access(src, os.X_OK), src                                         # installed with the repository's mode
