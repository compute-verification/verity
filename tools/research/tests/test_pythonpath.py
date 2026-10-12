"""A shipped tree's PYTHONPATH comes from its workspace declaration (research.pythonpath), never a hand list: two silent failures came
from a list that left out packages/verity/src."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research import pythonpath as P
from research import remote
from research.cli import main
from waits import wait_runner

REPO = Path(__file__).resolve().parents[3]
SRC = str(Path(remote.__file__).resolve().parents[1])
PY = sys.executable
REAL_ROOTS = ["tools/research/src", "infra/cluster/src", "infra/ops", "archive/sp1/python", "benchmarks/numerical/python", "integrations/vllm",
              "verity/ml/flock/python", "tools/verity/circuit_check/src", "tools/verity/cli", "verity/core/difftest/python"]


def test_this_repository_resolves_every_workspace_package_inside_the_tree(tmp_path):
    d = P.import_roots_dir(REPO)
    assert d["roots"] == REAL_ROOTS and d["missing"] == [] and d["shim"] == ["verity"] and d["source"] == "pyproject.toml [tool.uv.workspace]"
    names = ["research", "cluster", "verity_ops", "verity", "verity.core.protocols.sampled_proofs", "verity_vllm", "verity_sp1",
             "verity_numerical", "verity_flock", "circuit_check", "verity.ml.circuits", "verity.ml.kernels", "verity.core",
             "verity_difftest"]
    probe = "import importlib.util as u, sys; [print(n, u.find_spec(n).origin) for n in sys.argv[1:]]"
    # -S: no site-packages, so an editable install elsewhere cannot answer; find_spec locates without importing (no torch)
    r = subprocess.run([PY, "-S", "-c", probe, *names], cwd=tmp_path, env={"PYTHONPATH": P.pythonpath(str(REPO), d)}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    found = dict(line.split(" ", 1) for line in r.stdout.splitlines())
    assert sorted(found) == sorted(names)
    assert all(Path(o).resolve().is_relative_to(REPO) for o in found.values()), found


IN_GIT = (REPO / ".git").exists()           # a shipped tree (`run --source`) is a git archive: no history to read


@pytest.mark.skipif(not IN_GIT, reason="shipped tree: no .git")
def test_a_git_revision_gives_the_same_roots_without_a_checkout():
    assert P.import_roots_git(REPO, "HEAD")["roots"] == REAL_ROOTS
    with pytest.raises(LookupError):
        P.import_roots_git(REPO, "no-such-rev-xyz")


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_layouts_globs_exclude_and_missing(tmp_path):
    t = tmp_path / "tree"
    _write(t, "pyproject.toml", '[tool.uv.workspace]\nmembers = ["libs/*", "flat", "st_dir", "st_find", "gone"]\nexclude = ["libs/skip"]\n')
    _write(t, "libs/a/pyproject.toml", '[tool.hatch.build.targets.wheel]\npackages = ["src/a"]\n')
    _write(t, "libs/a/src/a/__init__.py", "")
    _write(t, "libs/b/pyproject.toml", "[project]\nname='b'\n")
    _write(t, "libs/b/src/b/__init__.py", "")
    _write(t, "libs/skip/pyproject.toml", "")
    _write(t, "flat/pyproject.toml", "[project]\nname='flat'\n")
    _write(t, "flat/flatpkg/__init__.py", "")
    _write(t, "st_dir/pyproject.toml", '[tool.setuptools]\npackage-dir = {"" = "lib"}\n')
    _write(t, "st_dir/lib/x/__init__.py", "")
    _write(t, "st_find/pyproject.toml", '[tool.setuptools.packages.find]\nwhere = ["code"]\n')
    _write(t, "st_find/code/y/__init__.py", "")
    d = P.import_roots_dir(t)
    assert d["members"] == ["libs/a", "libs/b", "flat", "st_dir", "st_find", "gone"]
    assert d["roots"] == ["libs/a/src", "libs/b/src", "flat", "st_dir/lib", "st_find/code"]
    assert d["missing"] == ["gone"], "a declared member that is absent is reported, not dropped silently"
    assert P.pythonpath("/w/src/s", d, extra="/mine:/w/src/s/flat") == \
        "/w/src/s/libs/a/src:/w/src/s/libs/b/src:/w/src/s/flat:/w/src/s/st_dir/lib:/w/src/s/st_find/code:/mine"
    assert P.import_roots_dir(tmp_path)["roots"] == [] and P.import_roots_dir(tmp_path)["source"].startswith("none")


def test_a_package_split_across_members_is_found_in_each_and_one_found_nowhere_raises(tmp_path):
    t = tmp_path / "tree"
    _write(t, "pyproject.toml", '[tool.uv.workspace]\nmembers = ["upper", "lower", "other"]\n')
    for member, module in (("upper", "ml"), ("lower", "ir")):
        _write(t, f"{member}/pyproject.toml", '[tool.hatch.build.targets.wheel]\npackages = ["src/core"]\n')
        _write(t, f"{member}/src/core/{module}/__init__.py", "")
    _write(t, "other/pyproject.toml", "[project]\nname='other'\n")
    _write(t, "other/src/other/__init__.py", "")
    _write(t, "uv.lock", "")
    _write(t, "upper/uv.lock", "")
    assert P.package_dirs(t, "core") == ["upper/src/core", "lower/src/core"]
    assert P.package_dirs(t, "other") == ["other/src/other"]
    with pytest.raises(LookupError, match="'gone'"):
        P.package_dirs(t, "gone")
    with pytest.raises(LookupError):
        P.package_dirs(tmp_path, "core")
    assert P.workspace_root(t / "lower" / "src" / "core" / "ir" / "__init__.py") == t.resolve()
    assert P.package_paths(t, "core", "**", "ir/**", "ml/__init__.py") == ["upper/src/core/**", "lower/src/core/**", "lower/src/core/ir/**",
                                                                          "upper/src/core/ml/__init__.py"]
    with pytest.raises(LookupError, match="'tc/\\*\\*'"):
        P.package_paths(t, "core", "ir/**", "tc/**")
    assert P.workspace_root(t / "upper") == t.resolve(), "a member's own pyproject.toml declares no workspace, lock or not"
    with pytest.raises(LookupError):
        P.workspace_root(tmp_path)


def test_a_workspace_root_that_is_a_package_is_reached_through_a_shim_never_the_root(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "SHIM_BASE", tmp_path / "shims")
    t = tmp_path / "tree"
    _write(t, "pyproject.toml", '[build-system]\nrequires = ["hatchling"]\n[tool.hatch.build.targets.wheel]\npackages = ["core", "src/extra"]\n'
                                '[tool.uv.workspace]\nmembers = ["tools/one"]\n')
    _write(t, "core/__init__.py", "")
    _write(t, "core/ir/__init__.py", "")
    _write(t, "src/extra/__init__.py", "")
    _write(t, "kernels/__init__.py", "")                         # a grouping directory with a PyPI name: must not import from the tree
    _write(t, "tools/one/pyproject.toml", '[tool.hatch.build.targets.wheel]\npackages = ["one"]\n')
    _write(t, "tools/one/one/__init__.py", "")
    d = P.import_roots_dir(t)
    assert (d["shim"], d["roots"], d["missing"]) == (["core"], ["src", "tools/one"], [])
    assert P.package_dirs(t, "core") == ["core"] and P.package_dirs(t, "one") == ["tools/one/one"]
    path = P.pythonpath(str(t), d, extra="/mine", shim_base=tmp_path / "shims").split(":")
    shim = Path(path[0])
    assert path[1:] == [str(t / "src"), str(t / "tools/one"), "/mine"] and str(t) not in path
    assert shim.parent == tmp_path / "shims" and os.listdir(shim) == ["core"] and (shim / "core").resolve() == t / "core"
    probe = "import importlib.util as u, sys; print([bool(u.find_spec(n)) for n in sys.argv[1:]])"
    r = subprocess.run([PY, "-S", "-c", probe, "core", "core.ir", "extra", "one", "kernels"], cwd=tmp_path,
                       env={"PYTHONPATH": ":".join(path)}, capture_output=True, text=True)
    assert r.stdout.strip() == "[True, True, True, True, False]", r.stderr
    assert P.shim_dir(t, ["core"], tmp_path / "shims") == str(shim)                       # the same tree shares one shim
    assert P.shim_packages(shim, t) == ["core"] and P.shim_packages(shim, t / "src") is None and P.shim_packages(t, t) is None
    (shim / "core").unlink()
    (shim / "core").symlink_to(tmp_path)
    assert P.shim_packages(shim, t) is None and P.shim_packages(tmp_path / "absent", t) is None
    assert P.shim_dir(t, ["core"], tmp_path / "shims") == str(shim) and (shim / "core").resolve() == t / "core", "a stale link is rebuilt"
    assert sorted(os.listdir(tmp_path / "shims")) == [shim.name]
    assert str(t) not in P.tree_pythonpath(t).split(":"), "a render of a root-package tree does not add the root either"
    _write(t, "pyproject.toml", '[build-system]\nrequires = ["hatchling"]\n[tool.uv]\npackage = false\n[tool.uv.workspace]\nmembers = ["tools/one"]\n')
    assert P.import_roots_dir(t)["shim"] == [] and P.tree_pythonpath(t).split(":")[-1] == str(t)     # a virtual root: as before
    _write(t, "pyproject.toml", '[build-system]\nrequires = ["setuptools"]\n[tool.uv.workspace]\nmembers = ["tools/one"]\n')
    assert P.import_roots_dir(t)["missing"][0].startswith(". (the workspace root is a package")


def test_launch_request_sets_the_trees_pythonpath_appends_the_users_and_records_it(tmp_path):
    root = tmp_path / "root"
    sha = "d" * 40
    src = root / "src" / sha
    _write(src, "pyproject.toml", '[tool.uv.workspace]\nmembers = ["pkgs/one"]\n')
    _write(src, "pkgs/one/pyproject.toml", '[tool.hatch.build.targets.wheel]\npackages = ["src/one"]\n')
    _write(src, "pkgs/one/src/one/__init__.py", "WHO = 'tree'\n")
    (src / remote.READY_FILE).write_text(json.dumps({"schema": remote.READY_SCHEMA, "identity": sha, "archive_sha256": "0" * 64}))
    mine = tmp_path / "mine"
    _write(mine, "mymod.py", "M = 1\n")
    (root / "requests" / "r").mkdir(parents=True)
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": "r", "runs_root": str(root / "runs"), "tool": SRC, "python": PY, "cwd": str(src),
           "env": {"PYTHONPATH": str(mine)}, "harness_args": ["--stage", "t", "--no-sampler"],
           "command": [PY, "-c", "import os, one, mymod; print(one.WHO, mymod.M); print(os.environ['PYTHONPATH'])"], "exclusive": False,
           "cost": {"provider": "runpod", "pod_id": "p", "rate_usd_per_hour": 0.1, "rate_observed_utc": "t", "currency": "USD", "reason": None},
           "remote": {"machine": "m", "pod_id": "p", "root": str(root), "source_sha": sha, "source_dir": str(src)}}
    (root / "requests" / "r" / "request.json").write_text(json.dumps(req))
    assert remote.launch_request(root / "requests" / "r" / "request.json")["started"]
    run_dir = root / "runs" / "r"
    wait_runner(run_dir)
    out = (run_dir / "stdout.log").read_text().splitlines()
    assert out[0] == "tree 1", (run_dir / "stderr.log").read_text()
    assert out[1].split(":") == [SRC, str(src / "pkgs/one/src"), str(mine)], "tool, then the tree's roots, then the user's --env PYTHONPATH"
    rem = json.loads((run_dir / "job.json").read_text())["run"]["remote"]
    assert rem["pythonpath"] == [str(src / "pkgs/one/src"), str(mine)]
    assert rem["pythonpath_derived"] == {"roots": ["pkgs/one/src"], "missing": [], "source": "pyproject.toml [tool.uv.workspace]"}


def test_research_env_prints_the_export_line(capsys):
    assert main(["env", "--source", str(REPO)]) == 0
    line = capsys.readouterr().out.strip()
    assert line == "export PYTHONPATH=" + ":".join([P.shim_dir(REPO, ["verity"]), *(str(REPO / r) for r in REAL_ROOTS)])
    if not IN_GIT:
        return
    assert main(["env", "--source", str(REPO), "--rev", "HEAD", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["roots"] == REAL_ROOTS
