"""A part's pass is reused on any node for the same inputs (`check.py --plan`, `research merge --verdicts`), so no key names the node:
a suite's (`suites.key`, and its test modules' `module_key` and `traced_key`), check's whole steps' (`STEP_KEYS`) and a Lean package's
(`lean_audit.key`), each computed for one tree on two simulated nodes, are equal, and a change to what the key declares moves it. The
two nodes differ in everything that names a node: the tree's absolute path, the host, the home, the PATH, the temporary and runtime
directories, the check slot's test cache, the kernel release, a GPU node whose GPUs the runner hides against a node with none,
rustup's `cargo` on the PATH against only installed, and a Lean build output one node built and the other didn't. circuit-check's
key is held to the same in its own suite (`tools/verity/circuit_check/tests/test_circuit_check.py`)."""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]

TREE = {
    ".gitignore": ".lake/\n__pycache__/\n",
    "pyproject.toml": '[tool.uv.workspace]\nmembers = ["pkg"]\n',
    "pkg/pyproject.toml": ('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n[tool.verity.tests]\n'
                           'inputs = ["lean"]\nartifacts = ["lean/tool/.lake/build/bin/tool"]\n'),
    "pkg/tests/test_a.py": "def test_a():\n    pass\n",
    "pkg/tests/test_b.py": "def test_b():\n    pass\n",
    "lean/tool/lakefile.toml": 'name = "tool"\n\n[[require]]\nname = "lib"\npath = "../lib"\n',
    "lean/tool/lake-manifest.json": json.dumps({"packages": [{"name": "lib", "type": "path", "dir": "../lib"}]}),
    "lean/tool/lean-toolchain": "leanprover/lean4:v4.0.0\n",
    "lean/tool/lean-audit.json": "{}",
    "lean/tool/Main.lean": "def main : IO Unit := pure ()\n",
    "lean/lib/lakefile.toml": 'name = "lib"\n',
    "lean/lib/Lib.lean": "def x := 1\n",
    "tools/verity/lean/lean_audit.py": "",
    "tools/verity/check/common.py": "",
    "verity/ml/flock/check_build.sh": "set -e\nRUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.98.1}\n",
    "verity/ml/flock/live/src/lib.rs": "fn a() {}\n",
    "verity/ml/flock/verifier/ci.py": "print('agree')\n",
    "verity/core/service/flock/lakefile.toml": 'name = "flock_verifier"\n',
}


def _load(name: str, path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, m)                  # dataclasses resolve their module by name
    spec.loader.exec_module(m)
    return m


def _exe(p: Path, text: str = "#!/bin/sh\n") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    p.chmod(0o755)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=root, check=True,
                   capture_output=True)


class Node:
    """One node with the tree checked out at a path of its own: ``gpu``, a GPU node whose GPUs the runner hides
    (`CUDA_VISIBLE_DEVICES=''`), else one with no `nvidia-smi`; ``cargo``, rustup's `cargo` on the PATH, else only installed."""

    def __init__(self, tmp: Path, name: str, where: str, gpu: bool, cargo: bool, built: bool):
        base = tmp / name
        self.name, self.root, self.home, self.bin = name, base / where, base / "home", base / "bin"
        for f, text in TREE.items():
            (self.root / f).parent.mkdir(parents=True, exist_ok=True)
            (self.root / f).write_text(text)
        if built:
            _exe(self.root / "lean/tool/.lake/build/bin/tool", f"#!/bin/sh\n# built on {name}\n")
        self.bin.mkdir(parents=True)
        os.symlink(shutil.which("git"), self.bin / "git")       # only what the node has on its PATH, never this machine's
        if gpu:
            _exe(self.bin / "nvidia-smi")
        _exe((self.bin if cargo else self.home / ".cargo" / "bin") / "cargo")
        _exe(self.home / ".cargo" / "bin" / "rustc", '#!/bin/sh\necho "rustc $RUSTUP_TOOLCHAIN (x86_64-unknown-linux-gnu)"\n')
        self.env = {"PATH": str(self.bin), "HOME": str(self.home), "HOSTNAME": name, "TMPDIR": str(base / "tmp"),
                    "LANG": "C.UTF-8", "XDG_RUNTIME_DIR": f"/run/user/{1000 + len(name)}", "UV_CACHE_DIR": str(self.home / ".cache/uv"),
                    "VIRTUAL_ENV": str(self.root / ".venv"), "CARGO_HOME": str(self.home / ".cargo"),
                    "VERITY_TEST_CACHE": str(self.home / f".cache/verity/tests-slot-{name}"), "CHECK_SLOT": name,
                    "SSH_CONNECTION": f"10.0.0.{len(name)} 22", **({"CUDA_VISIBLE_DEVICES": ""} if gpu else {})}
        _git(self.root, "init", "-q")
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "-m", "tree")

    def write(self, path: str, text: str) -> None:
        (self.root / path).write_text(text)
        _git(self.root, "commit", "-q", "-am", path)

    @contextmanager
    def on(self, monkeypatch, **env):
        """This process as a process on the node: its environment, and its kernel release and host name."""
        with monkeypatch.context() as m:
            m.setattr(os, "environ", {**self.env, **env})
            m.setattr(platform, "release", lambda: f"6.{len(self.name)}.0-{self.name}")
            m.setattr(platform, "node", lambda: self.name)
            m.setattr(platform, "platform", lambda *a, **k: f"Linux-6.{len(self.name)}.0-{self.name}-x86_64")
            yield


def _nodes(tmp_path) -> tuple[Node, Node]:
    return (Node(tmp_path, "vy-nebius-1", "workspace/research/src/0123abcd", gpu=True, cargo=True, built=True),
            Node(tmp_path, "vy-nebius-cpu-11", "w/trees/verity", gpu=False, cargo=False, built=False))


def _suite_keys(node: Node, monkeypatch, **env) -> dict[str, str]:
    """The suite's key, and its first test module's per-test keys, as `suites.py`'s `main` and `PerTest` compute them on ``node``."""
    S = _load(f"suites_on_{node.name.replace('-', '_')}", HERE / "suites.py", monkeypatch)
    monkeypatch.setattr(S, "ROOT", node.root)
    with node.on(monkeypatch, **env):
        S.cargo_on_path()
        e = S.environment()
        pkgs, tree, lock = S.members(), S.worktree(), S.locked()
        [s] = S.discover()
        paths = S.inputs_of(s, pkgs)
        ids = S.input_ids(tree, sorted(set(paths) | set(S.RUNNER)), S.markdown(s), S.dropped(s, paths))
        deps = S.resolved(S.roots(s, pkgs), lock, s.path)
        own = S.own_ids(tree, s.path)
        mods = S.Modules(s, {f: (node.root / f).read_text() for f in own if f.endswith(".py")})
        m = "pkg/tests/test_a.py"
        keep = mods.reach(m)
        return {"suite": S.key(s, ids, e, [], deps), "module": S.module_key(s, m, keep, mods, ids, own, e, [], deps),
                "traced": S.traced_key(s, m, keep, mods, ids, own, e, [], deps)}


def test_a_suites_keys_are_the_same_on_every_node_and_move_with_what_they_declare(tmp_path, monkeypatch):
    a, b = _nodes(tmp_path)
    one = _suite_keys(a, monkeypatch)
    assert _suite_keys(b, monkeypatch) == one, "the same tree, toolchains and switches on another node"
    assert _suite_keys(b, monkeypatch, VERITY_SLOW="1") != one, "a switch tests read"
    assert _suite_keys(a, monkeypatch, CUDA_VISIBLE_DEVICES="0") != one, "a run that may use a GPU runs the GPU tests"
    b.write("lean/lib/Lib.lean", "def x := 2\n")
    two = _suite_keys(b, monkeypatch)
    assert two["suite"] != one["suite"], "a source of the build output the tests run"
    b.write("pkg/tests/test_a.py", "def test_a():\n    assert True\n")
    assert all(_suite_keys(b, monkeypatch)[k] != two[k] for k in one), "a test module"


def test_a_lake_build_output_is_keyed_by_what_its_build_reads(tmp_path, monkeypatch):
    """`built_from`: a Lake package's output by its tree and those of the packages it requires by path, built or not; by its bytes
    when its build may read anything else."""
    a, _ = _nodes(tmp_path)
    S = _load("suites_lake_sources", HERE / "suites.py", monkeypatch)
    monkeypatch.setattr(S, "ROOT", a.root)
    out = "lean/tool/.lake/build/bin/tool"
    assert S.lake_sources("lean/tool") == ["lean/lib", "lean/tool"]
    assert S.built_from(out).startswith("built from ")
    for lakefile, manifest in [('name = "tool"\n[[require]]\nname = "mathlib"\n', {"packages": []}),
                               ('name = "tool"\n[[require]]\nname = "x"\npath = "../../.."\n', {"packages": []})]:
        (a.root / "lean/tool/lakefile.toml").write_text(lakefile)
        (a.root / "lean/tool/lake-manifest.json").write_text(json.dumps(manifest))
        assert S.lake_sources("lean/tool") is None, lakefile
    (a.root / "lean/tool/lakefile.toml").write_text('name = "tool"\n[[require]]\nname = "mathlib"\n')
    (a.root / "lean/tool/lake-manifest.json").write_text(json.dumps({"packages": [{"name": "mathlib", "type": "git", "rev": "abc"}]}))
    assert S.lake_sources("lean/tool") == ["lean/tool"], "a git require is fetched at the revision the manifest pins"
    (a.root / "lean/tool/lakefile.lean").write_text("")
    S._SOURCES.clear()
    assert S.lake_sources("lean/tool") is None and S.built_from(out) is None


def _step_keys(node: Node, check, monkeypatch, **env) -> dict[str, str | None]:
    monkeypatch.setattr(check, "ROOT", node.root)
    monkeypatch.setattr(check, "tracked_files", lambda root=None: check.COMMON.tracked_files(node.root))
    with node.on(monkeypatch, **env):
        e = check.env()
        return {name: f(e) for name, f in check.STEP_KEYS.items()}


def test_checks_step_keys_are_the_same_on_every_node_and_move_with_what_they_declare(tmp_path, monkeypatch):
    check = _load("check_on_nodes", HERE / "check.py", monkeypatch)
    a, b = _nodes(tmp_path)
    one = _step_keys(a, check, monkeypatch)
    assert all(one.values()), one
    assert _step_keys(b, check, monkeypatch) == one, "the same tree and toolchains on another node"
    assert all(v != one[k] for k, v in _step_keys(b, check, monkeypatch, VERITY_SLOW="1").items()), "a switch the tests read"
    b.write("verity/ml/flock/verifier/ci.py", "print('disagree')\n")
    two = _step_keys(b, check, monkeypatch)
    assert two["lean-agreement"] != one["lean-agreement"] and two["preflight-lints"] != one["preflight-lints"]
    assert two["flock-rust-tests"] == one["flock-rust-tests"], "not an input of the Rust tests"
    _exe(b.home / ".cargo" / "bin" / "rustc", "#!/bin/sh\necho 'rustc 1.99.0 (x86_64-unknown-linux-gnu)'\n")
    assert _step_keys(b, check, monkeypatch)["flock-rust-tests"] != one["flock-rust-tests"], "the toolchain the script pins"


def test_a_lean_packages_key_is_the_same_on_every_node_and_moves_with_its_sources(tmp_path, monkeypatch):
    LA = _load("lean_audit_on_nodes", ROOT / "tools" / "verity" / "lean" / "lean_audit.py", monkeypatch)
    a, b = _nodes(tmp_path)

    def key(node: Node) -> str:
        monkeypatch.setattr(LA, "ROOT", node.root)
        with node.on(monkeypatch):
            return LA.key("lean/tool", LA.COMMON.tracked_files(node.root), LA.FLAGS)

    one = key(a)
    assert key(b) == one
    b.write("lean/lib/Lib.lean", "def x := 2\n")
    assert key(b) != one, "a package it requires by path"
