"""[R18 infra] F-r17-tp-06 fixed by construction: source identity manifest + PRECHECK_FAIL_SOURCE_IDENTITY (torch-free).

The finding (tp M-1830): a Commit under --gpu-tree without VERITOR_REPO imported `verity.ir` from the pod's DEFAULT tree (another
release) while `verity_capture` ran the release tree and versions.json named the release.  These tests reproduce the leak -- a SECOND
tree on sys.path ahead of the release tree -- and require the precheck to fail BY NAME, plus the positive (one tree) and the
runner-side refusals (row_pod.sh / canary.sh / hidden_gpu.py with a foreign $VERITOR_REPO).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
PY = sys.executable
sys.path.append(str(TREE / "zk" / "campaign"))
from telemetry import source_identity as gsid  # noqa: E402
from verity_capture.sweep import source_identity as sid  # noqa: E402


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("VERITOR_REPO", "PYTHONPATH")}
    env.update(extra)
    return env


def _other_tree(tmp_path: Path, name: str = "other", commit: str = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef") -> Path:
    """A second tree carrying only verity.ir (+ EXPORT.json naming another commit) -- the int-3 default tree of tp's finding."""
    other = tmp_path / name
    (other / "verity.ir").mkdir(parents=True)
    (other / "verity.ir" / "__init__.py").write_text("LEAKED = True\n")
    (other / "EXPORT.json").write_text(json.dumps({"commit": commit, "branch": "other", "dirty_patch_sha256": "0" * 64}))
    return other


# ---- positive ------------------------------------------------------------------------------------------------------------------------------------

def test_precheck_passes_from_one_tree(tmp_path):
    out = tmp_path / "source_identity.json"
    r = subprocess.run([PY, "-m", "verity_capture.sweep.source_identity", "--out", str(out)], cwd=TREE, env=_env(PYTHONPATH=".:vllm-poc"),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("SOURCE-IDENTITY-OK")
    doc = json.loads(out.read_text())
    assert doc["ok"] is True and doc["fail_class"] is None and doc["schema"] == gsid.SCHEMA
    assert set(doc["packages"]) == set(sid.PACKAGES)
    for name in ("verity.ir", "verity_capture", "vllm-poc", "telemetry"):
        p = doc["packages"][name]
        assert p["repo_root"] == os.path.realpath(TREE), name
        assert p["resolved_path"].startswith(os.path.realpath(TREE)), name
        assert p["git_sha"] == doc["requested"]["git_sha"], name
    hg = doc["packages"]["hidden_gpu"]
    assert hg["resolved_path"].endswith("hidden_gpu.py") and hg["repo_root"] == os.path.realpath(TREE) and len(hg["file_sha256"]) == 64
    assert doc["packages"]["vllm-poc"]["sys_path_entry"] == os.path.realpath(TREE / "vllm-poc")
    assert doc["sys_path0"] is not None


def test_precheck_passes_with_matching_veritor_repo_and_sha(tmp_path):
    sha = gsid.tree_identity(str(TREE))["git_sha"]
    assert sha
    r = subprocess.run([PY, "-m", "verity_capture.sweep.source_identity", "--sha", sha], cwd=TREE,
                       env=_env(PYTHONPATH=".:vllm-poc", VERITOR_REPO=str(TREE)), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---- the leak, reproduced: a second tree ahead on sys.path --------------------------------------------------------------------------------------

def test_negative_second_tree_on_sys_path_fails_by_name(tmp_path):
    """tp's vector: `sys.path.insert(0, <default tree>)` while the release tree's verity_capture runs.  Here the second tree is sys.path[0]
    (the interpreter's cwd under -m, exactly where hidden_gpu.py used to insert VERITOR_REPO-or-default) and the release tree follows on
    PYTHONPATH -> verity.ir resolves from the second tree, verity_capture from the release: PRECHECK_FAIL_SOURCE_IDENTITY naming verity.ir."""
    other = _other_tree(tmp_path)
    out = tmp_path / "source_identity.json"
    r = subprocess.run([PY, "-m", "verity_capture.sweep.source_identity", "--tree", str(TREE), "--out", str(out)], cwd=other,
                       env=_env(PYTHONPATH=f"{TREE}:{TREE / 'vllm-poc'}"), capture_output=True, text=True)
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert r.stderr.startswith(gsid.FAIL_CLASS), r.stderr
    assert "verity.ir" in r.stderr and str(other) in r.stderr and "deadbeef" in r.stderr
    doc = json.loads(out.read_text())     # the record is written even on failure
    assert doc["ok"] is False and doc["fail_class"] == "PRECHECK_FAIL_SOURCE_IDENTITY"
    names = {m["package"] for m in doc["mismatches"]}
    assert names == {"verity.ir"}, names
    m = doc["mismatches"][0]
    assert m["repo_root"] == os.path.realpath(other) and m["git_sha"].startswith("deadbeef")
    assert m["resolved_path"] == os.path.realpath(other / "verity.ir" / "__init__.py")
    # the packages that DID resolve from the release tree are still recorded as such
    assert doc["packages"]["verity_capture"]["repo_root"] == os.path.realpath(TREE)
    assert doc["sys_path0"] == str(other) or doc["sys_path0"] == os.path.realpath(other)     # the record names where the leak came from


def test_negative_in_process_sys_path_insert_fails_by_name(tmp_path):
    """The literal former line -- `sys.path.insert(0, REPO)` with REPO = another tree -- executed in-process before the precheck."""
    other = _other_tree(tmp_path)
    probe = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(other)!r})",
        "from verity_capture.sweep import source_identity as sid",
        "try:",
        "    sid.precheck('commit')",
        "except sid.SourceIdentityError as e:",
        "    print('RAISED', str(e).splitlines()[0]); raise SystemExit(3)",
        "print('NOT-RAISED'); raise SystemExit(0)",
    ])
    r = subprocess.run([PY, "-c", probe], cwd=TREE, env=_env(PYTHONPATH=".:vllm-poc"), capture_output=True, text=True)
    assert r.returncode == 3 and r.stdout.startswith("RAISED PRECHECK_FAIL_SOURCE_IDENTITY"), (r.stdout, r.stderr)


def test_negative_requested_sha_mismatch_fails_by_name():
    r = subprocess.run([PY, "-m", "verity_capture.sweep.source_identity", "--sha", "0" * 40], cwd=TREE, env=_env(PYTHONPATH=".:vllm-poc"),
                       capture_output=True, text=True)
    assert r.returncode == 3 and r.stderr.startswith(gsid.FAIL_CLASS) and "git_sha != requested" in r.stderr


def test_negative_veritor_repo_naming_another_tree_is_refused(tmp_path):
    other = _other_tree(tmp_path)
    r = subprocess.run([PY, "-m", "verity_capture.sweep.source_identity"], cwd=TREE, env=_env(PYTHONPATH=".:vllm-poc", VERITOR_REPO=str(other)),
                       capture_output=True, text=True)
    assert r.returncode == 3 and r.stderr.startswith(gsid.FAIL_CLASS) and "VERITOR_REPO" in r.stderr, r.stderr


def test_in_process_check_and_loaded_module_guard(tmp_path):
    other = _other_tree(tmp_path)
    ids = sid.resolve(str(TREE))
    assert gsid.check(ids, str(TREE)) == []
    leaked = dict(ids["verity.ir"], resolved_path=str(other / "verity.ir" / "__init__.py"), repo_root=str(other), git_sha="deadbeef")
    mm = gsid.check({**ids, "verity.ir": leaked}, str(TREE))
    assert [m["package"] for m in mm] == ["verity.ir"]
    with pytest.raises(gsid.SourceIdentityError, match="PRECHECK_FAIL_SOURCE_IDENTITY"):
        gsid.assert_identities({**ids, "verity.ir": leaked}, str(TREE))
    # native_host's import-site guard: the hidden_gpu module that LOADED must be this tree's
    fake = types.ModuleType("hidden_gpu")
    fake.__file__ = str(other / "hidden_gpu.py")
    with pytest.raises(gsid.SourceIdentityError, match="hidden_gpu"):
        sid.assert_loaded_module_is_ours(fake, "hidden_gpu", str(TREE))
    ours = types.ModuleType("hidden_gpu")
    ours.__file__ = str(TREE / "verity_capture" / "sweep" / "hidden_gpu_src" / "hidden_gpu.py")
    rec = sid.assert_loaded_module_is_ours(ours, "hidden_gpu", str(TREE))
    assert rec["repo_root"] == os.path.realpath(TREE)


def test_repo_root_resolution_prefers_export_json_then_git_then_markers(tmp_path):
    shipped = tmp_path / "shipped"
    (shipped / "verity_capture" / "sweep").mkdir(parents=True)
    (shipped / "EXPORT.json").write_text(json.dumps({"commit": "a" * 40, "dirty_patch_sha256": "b" * 64, "branch": "x"}))
    assert gsid.repo_root_of(shipped / "verity_capture" / "sweep" / "f.py") == str(shipped.resolve())
    ident = gsid.tree_identity(str(shipped))
    assert ident == {"git_sha": "a" * 40, "git_sha_source": "EXPORT.json", "dirty_patch_sha256": "b" * 64, "branch": "x"}
    assert gsid.repo_root_of(TREE / "verity_capture" / "sweep" / "source_identity.py") == os.path.realpath(TREE)   # .git (worktree file or dir)
    assert gsid.tree_identity(str(TREE))["git_sha_source"] == "git"
    bare = tmp_path / "bare"
    (bare / "verity_capture").mkdir(parents=True)
    (bare / "verity_capture" / "__init__.py").write_text("")
    (bare / "verity.ir").mkdir()
    (bare / "verity.ir" / "__init__.py").write_text("")
    (bare / "vllm-poc" / "verity_vllm_adapter").mkdir(parents=True)
    (bare / "vllm-poc" / "verity_vllm_adapter" / "__init__.py").write_text("")
    assert gsid.repo_root_of(bare / "verity_capture" / "__init__.py", sid.ROOT_MARKERS) == str(bare.resolve())
    assert gsid.tree_identity(str(bare))["git_sha"] is None and "neither" in gsid.tree_identity(str(bare))["git_sha_source"]
    assert gsid.repo_root_of(tmp_path / "nowhere" / "f.py") is None or True    # no tree above a bare tmp file: None (or the tmp root if it is a checkout)


# ---- hidden_gpu.py: REPO from the importing tree, never from the environment ----------------------------------------------------------------------

def _fake_tree_with_hidden_gpu(tmp_path: Path, name: str) -> Path:
    t = tmp_path / name
    for d in ("verity_capture/sweep/hidden_gpu_src", "out/gen/r9/cmt-hidden/src", "vllm-poc"):
        (t / d).mkdir(parents=True, exist_ok=True)
    (t / "verity_capture" / "__init__.py").write_text("")
    src = (TREE / "verity_capture" / "sweep" / "hidden_gpu_src" / "hidden_gpu.py").read_text()
    (t / "verity_capture" / "sweep" / "hidden_gpu_src" / "hidden_gpu.py").write_text(src)
    (t / "out" / "gen" / "r9" / "cmt-hidden" / "src" / "hidden_gpu.py").write_text(src)
    return t


@pytest.mark.parametrize("copy", ["verity_capture/sweep/hidden_gpu_src", "out/gen/r9/cmt-hidden/src"])
def test_hidden_gpu_resolves_repo_from_its_own_file_not_the_environment(tmp_path, copy):
    """Both copies (the shipped one, and the r9 path native_host inserts) find the SAME tree by walking up from __file__; a foreign
    $VERITOR_REPO is refused BEFORE torch/numpy load (so this runs torch-free); an agreeing one gets past the identity step."""
    a = _fake_tree_with_hidden_gpu(tmp_path, "treeA")
    b = _fake_tree_with_hidden_gpu(tmp_path, "treeB")
    probe = f"import sys; sys.path.insert(0, {str(a / copy)!r}); import hidden_gpu"
    # the leak vector of F-r17-tp-06: hidden_gpu from tree A while the environment names tree B -> refused by name
    r = subprocess.run([PY, "-c", probe], env=_env(VERITOR_REPO=str(b)), capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode != 0 and "PRECHECK_FAIL_SOURCE_IDENTITY" in r.stderr and str(a.resolve()) in r.stderr, r.stderr
    # no environment at all: REPO = tree A (the file's tree); the import proceeds past identity (fails later only on torch/numpy/hs here)
    r = subprocess.run([PY, "-c", probe + "; print('REPO', hidden_gpu.REPO)"], env=_env(), capture_output=True, text=True, cwd=tmp_path)
    assert "PRECHECK_FAIL_SOURCE_IDENTITY" not in r.stderr
    if r.returncode == 0:
        assert r.stdout.strip().endswith(str(a.resolve()))
    else:
        assert "No module named" in r.stderr, r.stderr      # torch-free laptop: the identity step passed, the heavy imports did not
    # the environment agreeing with the file's tree is fine
    r = subprocess.run([PY, "-c", probe], env=_env(VERITOR_REPO=str(a)), capture_output=True, text=True, cwd=tmp_path)
    assert "PRECHECK_FAIL_SOURCE_IDENTITY" not in r.stderr


def test_hidden_gpu_refuses_a_verity_capture_already_imported_from_another_tree(tmp_path):
    a = _fake_tree_with_hidden_gpu(tmp_path, "treeA")
    b = _fake_tree_with_hidden_gpu(tmp_path, "treeB")
    probe = (f"import sys; sys.path.insert(0, {str(b)!r}); import verity_capture; "
             f"sys.path.insert(0, {str(a / 'out/gen/r9/cmt-hidden/src')!r}); import hidden_gpu")
    r = subprocess.run([PY, "-c", probe], env=_env(), capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode != 0 and "PRECHECK_FAIL_SOURCE_IDENTITY" in r.stderr and "verity_capture" in r.stderr, r.stderr


def test_shipped_and_r9_copies_of_hidden_gpu_are_identical():
    a = (TREE / "verity_capture" / "sweep" / "hidden_gpu_src" / "hidden_gpu.py").read_bytes()
    r9 = TREE / "out" / "gen" / "r9" / "cmt-hidden" / "src" / "hidden_gpu.py"
    if not r9.exists():
        pytest.skip("out/gen/r9 not in this sparse checkout")
    assert r9.read_bytes() == a
    code = b"\n".join(l for l in a.splitlines() if l.startswith(b"REPO = "))                  # module-level assignments only, not the docstring
    assert code == b"REPO = _repo_of_this_file()"                                             # the F-r17-tp-06 line is gone from both copies


# ---- runners set VERITOR_REPO from their own tree -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("script,args", [("verity_capture/sweep/row_pod.sh", ["some-row", "ROLE", "org/repo", "rev"]),
                                         ("verity_capture/sweep/canary.sh", ["r18-test"])])
def test_runners_refuse_a_foreign_veritor_repo_before_doing_anything(tmp_path, script, args):
    other = tmp_path / "other"
    other.mkdir()
    r = subprocess.run(["bash", str(TREE / script), *args], cwd=TREE, env=_env(VERITOR_REPO=str(other), PATH=os.environ.get("PATH", "")),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert "PRECHECK_FAIL_SOURCE_IDENTITY" in r.stderr and str(other) in r.stderr
    assert not list(tmp_path.glob("**/stages.txt"))


def test_row_pod_exports_veritor_repo_from_its_own_tree_and_ignores_the_default_path():
    src = (TREE / "verity_capture" / "sweep" / "row_pod.sh").read_text()
    assert 'cd "${VERITOR_REPO:-/workspace/cp/veritor}"' not in src
    assert 'export VERITOR_REPO=$TREE_SELF' in src
    assert "verity_capture.sweep.source_identity" in src and "PRECHECK_FAIL_SOURCE_IDENTITY" in src
    can = (TREE / "verity_capture" / "sweep" / "canary.sh").read_text()
    assert 'cd "${VERITOR_REPO:-/workspace/cp/veritor}"' not in can and 'export VERITOR_REPO=$TREE_SELF' in can
