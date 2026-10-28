"""Source identity manifest + PRECHECK_FAIL_SOURCE_IDENTITY (torch-free).

A Commit under --gpu-tree without the tree variable (VERITY_TREE) can import the core package from the pod's DEFAULT tree (another
release) while `verity_vllm` runs the release tree and versions.json names the release.  These tests reproduce the leak -- a SECOND
tree on sys.path ahead of the tree of record -- and require the precheck to fail BY NAME, plus the positive (one tree), the shipped-tree
sha fallback (RESEARCH_SOURCE_SHA), and the runner-side refusals (row_pod.sh / canary.sh with a foreign $VERITY_TREE).

Layout under test (Verity): the tree of record is the REPO ROOT; `verity` from the root, `verity_vllm` from integrations/vllm,
`research` from tools/research/src.  REPO must be a git checkout: the positive precheck, the requested-sha mismatch and the
`.git`-before-markers resolution read its commit.  A `research run --source` tree has no `.git`, so on a pod run the suite from a clone
of the shipped commit (`git clone <root>/git/verity.git T && git -C T checkout $RESEARCH_SOURCE_SHA`), not from the shipped tree.

    cd integrations/vllm && PYTHONPATH=.:../..:../../tools/research/src python -m pytest -q tests/pipeline/test_source_identity.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research import pythonpath as rpp
from research.telemetry import source_identity as gsid
from verity_vllm import config as vconfig, settings as vsettings
from verity_vllm.pipeline import cli as vcli
from verity_vllm.pipeline import research_tools
from verity_vllm.pipeline import source_identity as sid

INTEGRATION = Path(__file__).resolve().parents[2]  # <repo>/integrations/vllm
REPO = INTEGRATION.parents[1]  # <repo>
PY = sys.executable
CORE_ENTRIES = [os.path.dirname(d) for d in research_tools.CORE_DIRS]   # each sys.path entry that ships core, relative to the repo root
PP = os.pathsep.join([str(INTEGRATION), *(str(REPO / e) for e in CORE_ENTRIES), str(REPO / "tools" / "research" / "src")])


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("VERITY_TREE", "PYTHONPATH", "RESEARCH_SOURCE_SHA")}
    env.update(extra)
    return env


def _run(*args: str, cwd: Path = INTEGRATION, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([PY, "-m", "verity_vllm.pipeline.cli", "source-identity", *args], cwd=cwd, env=_env(PYTHONPATH=PP, **env),
                          capture_output=True, text=True)


def _other_tree(tmp_path: Path, name: str = "other", commit: str = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef") -> Path:
    """A second tree carrying only the core package `verity` (+ EXPORT.json naming another commit) -- the default tree of tp's finding."""
    other = tmp_path / name
    (other / "verity").mkdir(parents=True)
    (other / "verity" / "__init__.py").write_text("LEAKED = True\n")
    (other / "EXPORT.json").write_text(json.dumps({"commit": commit, "branch": "other", "dirty_patch_sha256": "0" * 64}))
    return other


# ---- positive ------------------------------------------------------------------------------------------------------------------------------------

def test_module_layout_is_verity():
    assert sid.TREE == REPO and sid.normalise_tree(None) == os.path.realpath(REPO)
    assert set(sid.PACKAGES) == {"verity", "verity_vllm", "research"}
    assert "vllm-poc" not in sid.PACKAGES and "telemetry" not in sid.PACKAGES
    assert all((REPO / m).exists() for m in sid.ROOT_MARKERS)


@pytest.mark.parametrize("tree_arg", [None, "repo", "integration"])
def test_precheck_passes_from_one_tree(tmp_path, tree_arg):
    """`--tree` takes the repo root or the integration root (what row_pod.sh passes) and normalises to the repo root."""
    out = tmp_path / "source_identity.json"
    args = ["--out", str(out)]
    if tree_arg == "repo":
        args += ["--tree", str(REPO)]
    elif tree_arg == "integration":
        args += ["--tree", str(INTEGRATION)]
    r = _run(*args)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("SOURCE-IDENTITY-OK")
    doc = json.loads(out.read_text())
    assert doc["ok"] is True and doc["fail_class"] is None and doc["schema"] == gsid.SCHEMA
    assert doc["requested_root"] == os.path.realpath(REPO)
    assert set(doc["packages"]) == set(sid.PACKAGES)
    for name in ("verity", "verity_vllm", "research"):
        p = doc["packages"][name]
        assert p["repo_root"] == os.path.realpath(REPO), name
        assert p["resolved_path"].startswith(os.path.realpath(REPO)), name
        assert p["git_sha"] == doc["requested"]["git_sha"], name
        assert p["sys_path_entry"] == p["expected_sys_path_entry"], name
    assert doc["packages"]["verity"]["sys_path_entry"] == os.pathsep.join(os.path.realpath(REPO / e) for e in CORE_ENTRIES)
    assert doc["packages"]["research"]["sys_path_entry"] == os.path.realpath(REPO / "tools" / "research" / "src")
    assert doc["sys_path0"] is not None


@pytest.mark.parametrize("env_form", ["repo", "integration"])
def test_precheck_passes_with_matching_verity_tree_and_sha(env_form):
    sha = gsid.tree_identity(str(REPO))["git_sha"]
    assert sha
    r = _run("--sha", sha, VERITY_TREE=str(REPO if env_form == "repo" else INTEGRATION))
    assert r.returncode == 0, r.stderr


# ---- the leak, reproduced: a second tree ahead on sys.path --------------------------------------------------------------------------------------

def test_negative_second_tree_on_sys_path_fails_by_name(tmp_path):
    """tp's vector: `sys.path.insert(0, <default tree>)` while the release tree's verity_vllm runs.  Here the second tree is sys.path[0]
    (the interpreter's cwd under -m) and the tree of record follows on PYTHONPATH ->
    `verity` resolves from the second tree, verity_vllm from the tree of record: PRECHECK_FAIL_SOURCE_IDENTITY naming verity."""
    other = _other_tree(tmp_path)
    out = tmp_path / "source_identity.json"
    r = _run("--tree", str(REPO), "--out", str(out), cwd=other)
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert r.stderr.startswith(gsid.FAIL_CLASS), r.stderr
    assert "verity:" in r.stderr and str(other) in r.stderr and "deadbeef" in r.stderr
    doc = json.loads(out.read_text())     # the record is written even on failure
    assert doc["ok"] is False and doc["fail_class"] == "PRECHECK_FAIL_SOURCE_IDENTITY"
    names = {m["package"] for m in doc["mismatches"]}
    assert names == {"verity"}, names
    m = doc["mismatches"][0]
    assert m["repo_root"] == os.path.realpath(other) and m["git_sha"].startswith("deadbeef")
    assert m["resolved_path"] == os.path.realpath(other / "verity" / "__init__.py")
    # the packages that DID resolve from the tree of record are still recorded as such
    assert doc["packages"]["verity_vllm"]["repo_root"] == os.path.realpath(REPO)
    assert doc["sys_path0"] == str(other) or doc["sys_path0"] == os.path.realpath(other)     # the record names where the leak came from


def test_negative_mixed_tree_copy_of_verity_vllm_elsewhere_is_refused_by_name(tmp_path):
    """the mixed-tree case for an integration package: a COPY of verity_vllm (the vLLM IR half) somewhere else on sys.path,
    ahead of the tree of record.  The copy is byte-identical to ours -- identity is WHERE it resolves from, not what it contains.
    (cwd is neutral: under -m, sys.path[0] is the cwd and would otherwise put the tree of record ahead of PYTHONPATH.)"""
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(INTEGRATION / "verity_vllm", elsewhere / "verity_vllm", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (elsewhere / "EXPORT.json").write_text(json.dumps({"commit": "c0ffee" * 6 + "c0ff", "branch": "stale-copy", "dirty_patch_sha256": "0" * 64}))
    out = tmp_path / "source_identity.json"
    r = subprocess.run([PY, "-m", "verity_vllm.pipeline.cli", "source-identity", "--tree", str(INTEGRATION), "--out", str(out)], cwd=tmp_path,
                       env=_env(PYTHONPATH=f"{elsewhere}{os.pathsep}{PP}"), capture_output=True, text=True)
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert r.stderr.startswith(gsid.FAIL_CLASS) and "verity_vllm:" in r.stderr and str(elsewhere) in r.stderr, r.stderr
    doc = json.loads(out.read_text())
    assert {m["package"] for m in doc["mismatches"]} == {"verity_vllm"}
    m = doc["mismatches"][0]
    assert m["repo_root"] == os.path.realpath(elsewhere) and m["git_sha"].startswith("c0ffee") and m["sys_path_entry"] == os.path.realpath(elsewhere)
    assert m["expected_sys_path_entry"] == os.path.realpath(INTEGRATION)
    assert doc["packages"]["verity_vllm"]["repo_root"] == os.path.realpath(elsewhere)   # the precheck ran from the copy too


def test_negative_in_process_sys_path_insert_fails_by_name(tmp_path):
    """The literal former line -- `sys.path.insert(0, REPO)` with REPO = another tree -- executed in-process before the precheck."""
    other = _other_tree(tmp_path)
    probe = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(other)!r})",
        "from verity_vllm.pipeline import source_identity as sid",
        "try:",
        "    sid.precheck('commit')",
        "except sid.SourceIdentityError as e:",
        "    print('RAISED', str(e).splitlines()[0]); raise SystemExit(3)",
        "print('NOT-RAISED'); raise SystemExit(0)",
    ])
    r = subprocess.run([PY, "-c", probe], cwd=INTEGRATION, env=_env(PYTHONPATH=PP), capture_output=True, text=True)
    assert r.returncode == 3 and r.stdout.startswith("RAISED PRECHECK_FAIL_SOURCE_IDENTITY"), (r.stdout, r.stderr)


def test_negative_requested_sha_mismatch_fails_by_name():
    r = _run("--sha", "0" * 40)
    assert r.returncode == 3 and r.stderr.startswith(gsid.FAIL_CLASS) and "git_sha != requested" in r.stderr


def test_negative_verity_tree_naming_another_tree_is_refused(tmp_path):
    other = _other_tree(tmp_path)
    r = _run(VERITY_TREE=str(other))
    assert r.returncode == 3 and r.stderr.startswith(gsid.FAIL_CLASS) and "VERITY_TREE" in r.stderr, r.stderr


def test_research_must_import_as_research_telemetry_not_be_repaired(tmp_path):
    """No sys.path hack for `research`: without tools/research/src on the path the import fails with a message naming the fix."""
    r = subprocess.run([PY, "-c", "import verity_vllm.pipeline.source_identity"], cwd=INTEGRATION,
                       env=_env(PYTHONPATH=os.pathsep.join([str(INTEGRATION), *(str(REPO / e) for e in CORE_ENTRIES)])), capture_output=True, text=True)
    if r.returncode == 0:
        pytest.skip("`research` is importable from site-packages in this interpreter; the missing-package path cannot be exercised here")
    assert "tools/research/src" in r.stderr and "verity-vllm row" in r.stderr, r.stderr
    assert "sys.path.append" not in Path(sid.__file__).read_text() and "sys.path.insert" not in Path(sid.__file__).read_text()


def _from_harness_tool(rec: dict) -> bool:
    """`research` resolved from the harness's own snapshot (`research run` puts `<root>/tool/<RESEARCH_TOOL_SNAPSHOT>` first on
    PYTHONPATH), not from the tree under test."""
    snap = os.environ.get("RESEARCH_TOOL_SNAPSHOT")
    path = rec.get("resolved_path")
    return bool(snap and path) and any(p.name == snap and p.parent.name == "tool" for p in Path(path).parents)


def test_only_researchs_copy_in_the_harness_tool_snapshot_is_left_out(monkeypatch):
    shipped = {"resolved_path": "/workspace/research/tool/ab12/research/__init__.py"}
    monkeypatch.delenv("RESEARCH_TOOL_SNAPSHOT", raising=False)
    assert not _from_harness_tool(shipped)
    monkeypatch.setenv("RESEARCH_TOOL_SNAPSHOT", "ab12")
    assert _from_harness_tool(shipped)
    assert not _from_harness_tool({"resolved_path": "/elsewhere/ab12/research/__init__.py"})
    assert not _from_harness_tool({"resolved_path": str(REPO / "tools" / "research" / "src" / "research" / "__init__.py")})
    assert not _from_harness_tool({"resolved_path": None})


def test_in_process_check(tmp_path):
    other = _other_tree(tmp_path)
    ids = sid.resolve(str(REPO))
    ids = {k: v for k, v in ids.items() if not (k == "research" and _from_harness_tool(v))}
    assert gsid.check(ids, str(REPO)) == []
    leaked = dict(ids["verity"], resolved_path=str(other / "verity" / "__init__.py"), repo_root=str(other), git_sha="deadbeef")
    mm = gsid.check({**ids, "verity": leaked}, str(REPO))
    assert [m["package"] for m in mm] == ["verity"]
    with pytest.raises(gsid.SourceIdentityError, match="PRECHECK_FAIL_SOURCE_IDENTITY"):
        gsid.assert_identities({**ids, "verity": leaked}, str(REPO))


def test_repo_root_resolution_prefers_export_json_then_git_then_markers(tmp_path):
    shipped = tmp_path / "shipped"
    (shipped / "integrations" / "vllm" / "verity_vllm" / "ops").mkdir(parents=True)
    (shipped / "EXPORT.json").write_text(json.dumps({"commit": "a" * 40, "dirty_patch_sha256": "b" * 64, "branch": "x"}))
    assert gsid.repo_root_of(shipped / "integrations" / "vllm" / "verity_vllm" / "ops" / "f.py") == str(shipped.resolve())
    ident = gsid.tree_identity(str(shipped))
    assert ident == {"git_sha": "a" * 40, "git_sha_source": "EXPORT.json", "dirty_patch_sha256": "b" * 64, "branch": "x"}
    assert gsid.repo_root_of(Path(sid.__file__)) == os.path.realpath(REPO)   # .git (worktree file or dir) at the REPO root, not integrations/vllm
    assert gsid.tree_identity(str(REPO))["git_sha_source"] == "git"
    # a `git archive` export: no .git, no EXPORT.json -> Verity's markers find the repo root; the sha is not the tree's to give
    bare = tmp_path / "bare"
    for rel in sid.ROOT_MARKERS:
        (bare / rel).parent.mkdir(parents=True, exist_ok=True)
        (bare / rel).write_text("")
    assert gsid.repo_root_of(bare / "integrations" / "vllm" / "verity_vllm" / "__init__.py", sid.ROOT_MARKERS) == str(bare.resolve())
    assert gsid.tree_identity(str(bare))["git_sha"] is None and "neither" in gsid.tree_identity(str(bare))["git_sha_source"]
    assert sid.normalise_tree(bare / "integrations" / "vllm") == str(bare.resolve()) and sid.normalise_tree(bare) == str(bare.resolve())


# ---- a shipped tree (git archive: no .git): the sha comes from --sha or RESEARCH_SOURCE_SHA ------------------------------------------------------

def _shipped_tree(tmp_path: Path) -> Path:
    """<root>/src/<sha>/ as `research run --source` lays it out: the repo root markers and the workspace declaration (every member's
    pyproject.toml), stub packages at Verity's locations, and the REAL source_identity modules (generic + project, with the verity_vllm.config
    it locates the tree through and the typed records that reads, the research.pythonpath that names core's directories, and the CLI with the
    settings it reads) so the precheck under test runs from the shipped tree itself."""
    t = tmp_path / "src" / ("f" * 40)
    for rel in (*research_tools.CORE_DIRS, "integrations/vllm/verity_vllm", "integrations/vllm/verity_vllm/pipeline", "integrations/vllm/tests/pipeline",
                "tools/research/src/research/telemetry"):
        (t / rel).mkdir(parents=True)
        (t / rel / "__init__.py").write_text("")
    (t / "tools" / "research" / "src" / "research" / "__init__.py").write_text("")
    for member in ["", *rpp.import_roots_dir(REPO)["members"]]:
        (t / member).mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / member / "pyproject.toml", t / member / "pyproject.toml")
    (t / "uv.lock").write_text("")
    shutil.copy(gsid.__file__, t / "tools" / "research" / "src" / "research" / "telemetry" / "source_identity.py")
    shutil.copy(rpp.__file__, t / "tools" / "research" / "src" / "research" / "pythonpath.py")
    shutil.copy(sid.__file__, t / "integrations" / "vllm" / "verity_vllm" / "pipeline" / "source_identity.py")
    shutil.copy(vconfig.__file__, t / "integrations" / "vllm" / "verity_vllm" / "config.py")
    shutil.copy(vcli.__file__, t / "integrations" / "vllm" / "verity_vllm" / "pipeline" / "cli.py")
    shutil.copy(vsettings.__file__, t / "integrations" / "vllm" / "verity_vllm" / "settings.py")
    shutil.copytree(INTEGRATION / "verity_vllm" / "records", t / "integrations" / "vllm" / "verity_vllm" / "records",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    assert not (t / ".git").exists() and not (t / "EXPORT.json").exists()
    return t


def _shipped_pp(t: Path) -> str:
    return os.pathsep.join([str(t / "integrations" / "vllm"), *(str(t / e) for e in CORE_ENTRIES), str(t / "tools" / "research" / "src")])


def test_shipped_tree_takes_its_sha_from_research_source_sha(tmp_path):
    """`research run --source` executes from a git-archive export: no .git, no EXPORT.json.  The runner exports RESEARCH_SOURCE_SHA
    (research.result.ENV_SOURCE_SHA); the record carries it for every package of the tree, naming the source.  Without it the packages
    resolve (ok) with git_sha None -- the tree has no sha to give and none is invented."""
    t = _shipped_tree(tmp_path)
    integ = t / "integrations" / "vllm"
    out = tmp_path / "sid.json"
    r = subprocess.run([PY, "-m", "verity_vllm.pipeline.cli", "source-identity", "--tree", str(integ), "--out", str(out)], cwd=integ,
                       env=_env(PYTHONPATH=_shipped_pp(t), RESEARCH_SOURCE_SHA="f" * 40), capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
    doc = json.loads(out.read_text())
    assert doc["requested_root"] == str(t.resolve()) and doc["requested"]["git_sha"] == "f" * 40 and doc["requested"]["git_sha_source"] == "RESEARCH_SOURCE_SHA"
    for name in ("verity", "verity_vllm", "research"):
        assert doc["packages"][name]["repo_root"] == str(t.resolve()) and doc["packages"][name]["git_sha"] == "f" * 40, name
        assert doc["packages"][name]["git_sha_source"] == "RESEARCH_SOURCE_SHA", name
    assert "f" * 9 in r.stdout and doc["RESEARCH_SOURCE_SHA"] == "f" * 40
    # --sha wins over the environment and is checked against it: a differing env sha is not a package mismatch (env is only a fallback)
    r = subprocess.run([PY, "-m", "verity_vllm.pipeline.cli", "source-identity", "--tree", str(integ), "--sha", "e" * 40, "--out", str(out)], cwd=integ,
                       env=_env(PYTHONPATH=_shipped_pp(t)), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc["packages"]["verity"]["git_sha"] == "e" * 40 and doc["packages"]["verity"]["git_sha_source"] == "--sha"
    # no declaration at all: ok, sha None, source says why
    r = subprocess.run([PY, "-m", "verity_vllm.pipeline.cli", "source-identity", "--tree", str(integ), "--out", str(out)], cwd=integ,
                       env=_env(PYTHONPATH=_shipped_pp(t)), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc["packages"]["verity"]["git_sha"] is None and "neither" in doc["packages"]["verity"]["git_sha_source"]
    assert r.stdout.startswith("SOURCE-IDENTITY-OK verity=-")


def test_shipped_tree_still_refuses_a_foreign_package(tmp_path):
    """A shipped tree with the sha declared: a `verity` that resolves from a checkout elsewhere (here: THIS repo) is refused by name."""
    t = _shipped_tree(tmp_path)
    integ = t / "integrations" / "vllm"
    pp = os.pathsep.join([*(str(REPO / e) for e in CORE_ENTRIES), _shipped_pp(t)])
    r = subprocess.run([PY, "-m", "verity_vllm.pipeline.cli", "source-identity", "--tree", str(integ)], cwd=integ,
                       env=_env(PYTHONPATH=pp, RESEARCH_SOURCE_SHA="f" * 40), capture_output=True, text=True)
    assert r.returncode == 3 and r.stderr.startswith(gsid.FAIL_CLASS) and "verity:" in r.stderr and os.path.realpath(REPO) in r.stderr, r.stderr


def test_a_shipped_tree_with_core_split_across_members_passes_and_a_foreign_portion_is_refused(tmp_path):
    """Core as a namespace (PEP 420) that two members ship: the record expects and finds both members' entries, and a third portion
    from another tree is refused by name.  `-S` keeps this interpreter's own regular `verity` from shadowing the namespace."""
    t = _shipped_tree(tmp_path)
    for d in research_tools.CORE_DIRS:
        shutil.rmtree(t / d)
    (t / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["core/upper", "core/lower", "integrations/vllm", "tools/research"]\n')
    for member, module in (("core/upper", "ml"), ("core/lower", "ir")):
        (t / member / "src" / "verity" / module).mkdir(parents=True)
        (t / member / "src" / "verity" / module / "__init__.py").write_text("")
        (t / member / "pyproject.toml").write_text('[tool.hatch.build.targets.wheel]\npackages = ["src/verity"]\n')
    integ = t / "integrations" / "vllm"
    entries = [os.path.realpath(t / m / "src") for m in ("core/upper", "core/lower")]
    pp = os.pathsep.join([str(integ), *entries, str(t / "tools" / "research" / "src")])
    out = tmp_path / "sid.json"
    argv = [PY, "-S", "-m", "verity_vllm.pipeline.cli", "source-identity", "--tree", str(integ), "--out", str(out)]
    r = subprocess.run(argv, cwd=integ, env=_env(PYTHONPATH=pp, RESEARCH_SOURCE_SHA="f" * 40), capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
    core = json.loads(out.read_text())["packages"]["verity"]
    assert core["expected_sys_path_entry"] == core["sys_path_entry"] == os.pathsep.join(entries)
    assert [p["repo_root"] for p in core["portions"]] == [str(t.resolve())] * 2
    other = _other_tree(tmp_path)
    (other / "verity" / "__init__.py").unlink()
    r = subprocess.run(argv, cwd=integ, env=_env(PYTHONPATH=f"{pp}{os.pathsep}{other}", RESEARCH_SOURCE_SHA="f" * 40), capture_output=True, text=True)
    assert r.returncode == 3 and r.stderr.startswith(gsid.FAIL_CLASS), (r.stdout, r.stderr)
    assert "namespace portion(s) outside the requested tree" in r.stderr and os.path.realpath(other / "verity") in r.stderr, r.stderr


# ---- runners set VERITY_TREE from their own tree ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [[PY, "-m", "verity_vllm.pipeline.cli", "row", "run", "some-row", "ROLE", "org/repo", "rev"],
                                  ["bash", "verity_vllm/ops/canary.sh", "r18-test"],
                                  ["bash", "verity_vllm/ops/compiled_commit.sh", "some-row", "ROLE"],
                                  ["bash", "verity_vllm/ops/stoch_negatives.sh", "some-row", "ROLE", "unused-out"],
                                  ["bash", "verity_vllm/ops/stoch_negative_n3.sh", "some-row", "ROLE", "org/repo", "rev", "unused-out"]])
def test_runners_refuse_a_foreign_verity_tree_before_doing_anything(tmp_path, argv):
    other = tmp_path / "other"
    other.mkdir()
    r = subprocess.run(argv, cwd=INTEGRATION, env=_env(VERITY_TREE=str(other), PATH=os.environ.get("PATH", ""), PYTHONPATH=PP),
                       capture_output=True, text=True)
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert "PRECHECK_FAIL_SOURCE_IDENTITY" in r.stderr and str(other) in r.stderr
    assert not list(tmp_path.glob("**/stages.txt"))


@pytest.mark.parametrize("script", ["canary.sh", "compiled_commit.sh", "stoch_negatives.sh", "stoch_negative_n3.sh"])
def test_runner_scripts_export_verity_tree_from_their_own_tree_and_have_no_default_path(script):
    text = (INTEGRATION / "verity_vllm" / "ops" / script).read_text()
    assert 'cd "${VERITY_TREE:-' not in text and 'export VERITY_TREE=$TREE_SELF' in text
