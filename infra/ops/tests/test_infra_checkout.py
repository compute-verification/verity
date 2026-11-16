"""The control pod's infra checkout (infra/control/infra_checkout.py) puts infra's main on a node as the node's login, with
no sudo, so a node whose lock-out took root from research still gets each commit: a whole tree swapped in by renaming a
symlink, a malformed commit refused, two trees kept."""
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def module():
    stub, had = types.ModuleType("research.remote"), sys.modules.get("research.remote")
    sys.modules["research.remote"] = stub
    try:
        spec = importlib.util.spec_from_file_location("infra_checkout", ROOT / "infra" / "control" / "infra_checkout.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if had is None:
            sys.modules.pop("research.remote")
        else:
            sys.modules["research.remote"] = had
    return mod


C = module()


@pytest.fixture
def node(tmp_path):
    """A node reached by running its commands here: its own home, and a sudo that refuses as a locked node's does."""
    home, bin_ = tmp_path / "home", tmp_path / "bin"
    home.mkdir()
    bin_.mkdir()
    (bin_ / "sudo").write_text("#!/bin/sh\necho 'sudo: a password is required' >&2\nexit 1\n")
    (bin_ / "sudo").chmod(0o755)
    return types.SimpleNamespace(home=home, env={"HOME": str(home), "PATH": f"{bin_}:/usr/bin:/bin"})


def tree(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def put(node, commit: str, files: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", "-c", C.INSTALL, "install", commit], input=tree(files), capture_output=True, env=node.env)


def test_a_node_takes_a_whole_tree_as_its_login_and_keeps_two(node):
    link, trees = node.home / ".research" / "infra", node.home / ".research" / "infra-trees"
    commits = [c * 40 for c in "abc"]
    for i, c in enumerate(commits):
        r = put(node, c, {"AGENTS.md": f"tree {c[0]}\n"})
        assert r.returncode == 0 and r.stdout.decode().strip() == c, r.stderr.decode()
        assert link.is_symlink() and link.resolve() == (trees / c).resolve(), "the link names the whole new tree"
        assert (link / "AGENTS.md").read_text() == f"tree {c[0]}\n" and (link / ".commit").read_text() == f"{c}\n"
        os.utime(trees / c, (1_000_000_000 + 100 * i,) * 2)
    assert sorted(p.name for p in trees.iterdir()) == commits[1:], "each node keeps its two newest trees"
    for bad in ("not-a-commit", "a" * 39, ""):
        r = put(node, bad, {"x": "y"})
        assert r.returncode == 2 and b"not a commit" in r.stderr
    assert link.resolve() == (trees / commits[2]).resolve(), "a refused commit changes nothing"


def test_the_loop_installs_on_a_node_that_refuses_sudo(node, tmp_path, monkeypatch):
    repo = tmp_path / "infra"
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=node.env)
    (repo / "AGENTS.md").write_text("infra's rules\n")
    subprocess.run([*git, "add", "AGENTS.md"], check=True, env=node.env)
    subprocess.run([*git, "commit", "-qm", "c"], check=True, env=node.env)
    commit = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True, env=node.env).stdout.strip()

    def ssh_run(ssh, argv, timeout):
        return subprocess.run(argv, capture_output=True, env=node.env)

    def ssh_stream(ssh, argv, feed, timeout):
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=node.env)
        feed(p.stdin)
        out, err = p.communicate()
        return types.SimpleNamespace(returncode=p.returncode, stdout=out, stderr=err)

    monkeypatch.setattr(C, "CHECKOUT", repo)
    monkeypatch.setattr(C, "pod_main", lambda: commit)
    monkeypatch.setattr(C, "ssh_to", lambda name, ms: ["ssh", name])
    monkeypatch.setattr(C.remote, "load_machines", lambda: {"n": {"provider": "ssh", "user": "research"}}, raising=False)
    monkeypatch.setattr(C.remote, "ssh_run", ssh_run, raising=False)
    monkeypatch.setattr(C.remote, "ssh_stream", ssh_stream, raising=False)

    state = C.once(tmp_path / "state.json", [])
    assert state["nodes"]["n"] == {"commit": commit, "error": ""}, state["nodes"]["n"]
    assert (node.home / ".research" / "infra" / "AGENTS.md").read_text() == "infra's rules\n"
    assert C.once(tmp_path / "state.json", [])["nodes"]["n"]["error"] == "", "a node at the pod's commit reads as current"
