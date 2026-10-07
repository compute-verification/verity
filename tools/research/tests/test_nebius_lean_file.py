"""vy-lean-file (`pods/nebius/lean_file.py`) and `research lean-file`: what it refuses, which package, dependencies and warm build a
file gets, and a whole check through a stand-in vy-lean-sandbox and `lake`; the patch the agent's side ships and the run it
launches."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from research import cli

NEBIUS = Path(__file__).resolve().parents[1] / "src/research/pods/nebius"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


F = _load("nebius_lean_file", NEBIUS / "lean_file.py")
SHA = "a" * 40
MANIFEST = json.dumps({"packages": [{"name": "mathlib", "type": "git"}]})


def _git(d: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(d), *args], capture_output=True, text=True, check=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                               "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull}).stdout.strip()


def _pkg(d: Path, toml: str = "", manifest: str = '{"packages": []}') -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "lakefile.toml").write_text(toml)
    (d / "lake-manifest.json").write_text(manifest)
    (d / "lean-toolchain").write_text("leanprover/lean4:v4.34.0\n")


@pytest.mark.parametrize("sha, file, why", [
    ("abc", "A/B.lean", "full commit SHA"),
    (SHA, "/etc/B.lean", "relative"),
    (SHA, "A/../../B.lean", "relative"),
    (SHA, "A/B.py", ".lean file"),
])
def test_it_refuses_a_short_sha_and_a_path_that_isnt_a_lean_file_of_the_repository(sha, file, why):
    with pytest.raises(F.Refused, match=why):
        F.check_args(sha, file)


def test_a_file_is_in_its_innermost_package_which_brings_the_packages_it_requires_by_path(tmp_path):
    _pkg(tmp_path / "verity/Security")
    _pkg(tmp_path / "verity/Security/Proofs", '[[require]]\nname = "security"\npath = ".."\n\n'
                                              '[[require]]\nname = "flock"\npath = "../../../backends/flock"\n\n'
                                              '[[require]]\nname = "mathlib"\ngit = "https://x"\n')
    _pkg(tmp_path / "backends/flock")
    assert F.package(tmp_path, "verity/Security/Proofs/Core/X.lean") == "verity/Security/Proofs"
    assert F.package(tmp_path, "verity/Security/Definitions/X.lean") == "verity/Security"
    assert F.requires(tmp_path, "verity/Security/Proofs") == ["verity/Security/Proofs", "backends/flock", "verity/Security"]
    with pytest.raises(F.Refused, match="no Lake package"):
        F.package(tmp_path, "tools/X.lean")


def test_the_publishers_build_is_copied_only_for_the_same_manifest_and_toolchain_and_never_while_it_builds(tmp_path):
    tree, pub = tmp_path / "tree", tmp_path / "pub"
    for t in (tree, pub):
        _pkg(t / "a")
        _pkg(t / "b")
    (pub / "a/.lake/build").mkdir(parents=True)
    (pub / "b/.lake/build").mkdir(parents=True)
    (pub / "b/lean-toolchain").write_text("leanprover/lean4:v4.33.0\n")
    n = F.Node(publisher=pub)
    got, why = F.warm(n, tree, ["a", "b"], active=lambda unit: False)
    assert got == [[str(pub / "a/.lake/build"), str(tree / "a/.lake/build")]] and why.startswith("cold: b")
    assert F.warm(n, tree, ["a"], active=lambda unit: unit == n.publisher_unit)[0] == []


FAKE_SANDBOX = r'''
import json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(args) + "\n")
if args[0] == "checkout":
    tree = args[args.index("--tree") + 1]
    subprocess.run(["git", "clone", "-q", os.environ["FAKE_REPO"], tree], check=True)
    subprocess.run(["git", "-C", tree, "checkout", "-q", "--detach", args[1]], check=True)
    for lf in subprocess.run(["git", "-C", tree, "ls-files", "*lakefile.toml"], capture_output=True, text=True).stdout.split():
        os.mkdir(os.path.join(tree, os.path.dirname(lf), ".lake"))
    sys.exit(0)
cmd = args[args.index("--") + 1:]
sys.exit(subprocess.run(cmd, env={**os.environ, "PATH": os.environ["FAKE_BIN"] + ":" + os.environ["PATH"]}).returncode)
'''

FAKE_LAKE = r'''#!/bin/sh
if [ "$1" = lean ]; then echo "built the imports of $2"; exit 0; fi
if grep -q sorry "$3"; then echo "$3:1:0: error: no"; exit 1; fi
echo "elaborated $3"
'''


@pytest.fixture
def node(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _pkg(repo / "verity/Security", manifest=MANIFEST)
    (repo / "verity/Security/Definitions").mkdir()
    (repo / "verity/Security/Definitions/A.lean").write_text("theorem a : True := trivial\n")
    _git(tmp_path, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "x")
    sha = _git(repo, "rev-parse", "HEAD")
    deps = tmp_path / "deps"
    (deps / F.deps_key(repo / "verity/Security") / "mathlib").mkdir(parents=True)
    pub = tmp_path / "publisher"
    _pkg(pub / "verity/Security", manifest=MANIFEST)
    (pub / "verity/Security/.lake/build/lib").mkdir(parents=True)
    (pub / "verity/Security/.lake/build/lib/A.olean").write_text("olean")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/lake").write_text(FAKE_LAKE)
    (tmp_path / "bin/lake").chmod(0o755)
    (tmp_path / "sandbox.py").write_text(FAKE_SANDBOX)
    for k, v in {"FAKE_LOG": tmp_path / "sandbox.log", "FAKE_REPO": repo, "FAKE_BIN": tmp_path / "bin"}.items():
        monkeypatch.setenv(k, str(v))
    n = F.Node(jobs=tmp_path / "jobs", deps=deps, publisher=pub, sandbox=[sys.executable, str(tmp_path / "sandbox.py")],
               python=sys.executable)
    return n, sha, repo, tmp_path


def _calls(tmp_path: Path) -> list[list[str]]:
    return [json.loads(ln) for ln in (tmp_path / "sandbox.log").read_text().splitlines()]


def test_a_check_elaborates_the_file_in_a_tree_of_its_own_with_the_warm_deps_read_only_and_a_copy_of_the_warm_build(node):
    n, sha, repo, tmp = node
    out = tmp / "run/out"
    out.parent.mkdir()
    rc = F.check(n, sha, "verity/Security/Definitions/A.lean", None, out, geteuid=lambda: 0, active=lambda u: False)
    assert rc == F.OK
    checkout, run = _calls(tmp)
    tree = checkout[checkout.index("--tree") + 1]
    assert checkout[:2] == ["checkout", sha] and Path(tree).parent == n.jobs and not Path(tree).exists()
    deps = n.deps / F.deps_key(repo / "verity/Security")
    assert run[run.index("--ro") + 1] == f"{deps}:{tree}/verity/Security/.lake/packages"
    assert run[run.index("--tree") + 1] == tree and "LAKE_ARTIFACT_CACHE=false" in run and run[run.index("--by") + 1] == "lean-file"
    spec = json.loads(run[-1])
    assert spec["warm"] == [[str(n.publisher / "verity/Security/.lake/build"), f"{tree}/verity/Security/.lake/build"]]
    assert spec["file"] == "Definitions/A.lean" and spec["package"] == f"{tree}/verity/Security"
    assert (out / "elaboration.txt").read_text() == "elaborated Definitions/A.lean\n"
    summary = json.loads((out / "lean-file.json").read_text())
    assert [s["step"] for s in summary["steps"]] == [f"warm {n.publisher}/verity/Security/.lake/build", "imports", "elaborate"]
    assert all(s["rc"] == 0 for s in summary["steps"]) and summary["package"] == "verity/Security"
    assert "built the imports of Definitions/A.lean" in (out / "lean.log").read_text()


def test_the_patch_is_applied_to_the_fresh_tree_and_an_error_in_the_file_fails_the_check(node):
    n, sha, repo, tmp = node
    (repo / "verity/Security/Definitions/B.lean").write_text("theorem b : False := sorry\n")
    _git(repo, "add", "-N", "verity/Security/Definitions/B.lean")
    patch = tmp / "p.patch"
    patch.write_text(_git(repo, "diff") + "\n")
    out = tmp / "out"
    rc = F.check(n, sha, "verity/Security/Definitions/B.lean", patch, out, geteuid=lambda: 0, active=lambda u: True)
    assert rc == F.FAILED
    assert "error: no" in (out / "elaboration.txt").read_text()
    summary = json.loads((out / "lean-file.json").read_text())
    assert summary["patch"]["bytes"] == patch.stat().st_size and "cold" in summary and summary["warm"] == []


def test_without_warm_deps_for_its_manifest_nothing_runs_and_the_tree_is_removed(node):
    n, sha, _repo, tmp = node
    for d in n.deps.iterdir():
        d.rename(d.with_name("other"))
    out = tmp / "out"
    with pytest.raises(F.Refused, match="no fetched dependencies"):
        F.check(n, sha, "verity/Security/Definitions/A.lean", None, out, geteuid=lambda: 0, active=lambda u: False)
    assert [c[0] for c in _calls(tmp)] == ["checkout"] and list(n.jobs.iterdir()) == []
    assert "no fetched dependencies" in json.loads((out / "lean-file.json").read_text())["refused"]


def test_it_refuses_to_run_as_anyone_but_root(tmp_path):
    with pytest.raises(F.Refused, match="run as root"):
        F.check(F.Node(jobs=tmp_path), SHA, "A.lean", None, tmp_path / "out", geteuid=lambda: 1000)


# ---- the agent's side

def test_the_patch_carries_tracked_and_untracked_changes_and_leaves_the_index_alone(tmp_path):
    repo = tmp_path / "repo"
    _git(tmp_path, "init", "-q", str(repo))
    (repo / "A.lean").write_text("a\n")
    _git(repo, "add", "A.lean")
    _git(repo, "commit", "-qm", "x")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "A.lean").write_text("a2\n")
    (repo / "B.lean").write_text("b\n")
    patch = cli.lean_file_patch(repo, base).decode()
    assert "+a2" in patch and "+++ b/B.lean" in patch
    assert _git(repo, "diff", "--cached", "--name-only") == "" and "?? B.lean" in _git(repo, "status", "--porcelain")


def test_lean_file_launches_vy_lean_file_through_research_run_on_node_1_with_the_patch_sent(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _git(tmp_path, "init", "-q", str(repo))
    (repo / "A.lean").write_text("a\n")
    _git(repo, "add", "A.lean")
    _git(repo, "commit", "-qm", "x")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "A.lean").write_text("a2\n")
    seen = {}

    def run(argv):
        i = argv.index("--send") + 1
        seen.update(argv=argv, patch=Path(argv[i]).read_text())
        return 0

    monkeypatch.setattr(cli, "cmd_run", run)
    monkeypatch.chdir(repo)
    assert cli.main(["lean-file", "A.lean", "--rev", base]) == 0
    argv = seen["argv"]
    assert argv[:4] == ["--on", "vy-nebius-1", "--project", "verity"] and "+a2" in seen["patch"]
    assert argv[argv.index("--declared-output") + 1] == "out/*"
    assert argv[argv.index("--") + 1:] == ["bash", "-c", cli.LEAN_FILE_CMD, "vy-lean-file", base, "A.lean", cli.LEAN_FILE_PATCH]


def test_the_workload_hands_vy_lean_file_the_run_dirs_paths(tmp_path):
    fake = tmp_path / "sudo"
    fake.write_text('#!/bin/sh\nshift; printf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin", "RESEARCH_RUN_DIR": "/r"}
    for patch, want in (("lean-file.patch", ["--patch", "/r/inputs/lean-file.patch"]), ("", [])):
        r = subprocess.run(["bash", "-c", cli.LEAN_FILE_CMD, "vy-lean-file", SHA, "A.lean", patch], capture_output=True,
                           text=True, env=env, check=False)
        assert r.stdout.split() == ["/usr/local/bin/vy-lean-file", SHA, "A.lean", "--out", "/r/out", *want]
