"""RELEASE.json names the tree the ship INSTALLED, enforced at the writer
(explicit tree, refused unless it is the cwd), the reader (tree-local record first) and the ship (read-back check).  Torch-free."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm.pipeline import release_json as rj

HERE = Path(__file__).resolve().parents[2] / "verity_vllm" / "ops"
EXPORT = {"commit": "a2dabe49314ec6d85d1d62096551cf76751c7ea0", "branch": "lane/r16-int", "dirty_patch_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "dirty_patch_bytes": 0, "exported_utc": "2026-09-20T10:08:00Z"}
HG = "HIDDEN-GPU-OK 1111 /root/.cache/torch_extensions/hidden_gpu_tree/hidden_gpu_tree.so sm_89"
TP = "FA2-TAP-OK 2222 /workspace/cp/fa2/build/matReq/verity_fa2_matReq.so arch=8.9 hdims=64,96,128 mask=14 layout=v8"
NC = "3333 /workspace/cp/nc_build-r16-int-3/verity_native_collect.so"
TP3 = "FA3-TAP-OK 4444 /workspace/cp/fa2/build/fa3_matReq/verity_fa3_matReq.so hdims=64,128 mask=6 layout=v9"


def _rec(tree: Path, tag="r16-int-3", cwd=None, **kw):
    return rj.release_record(tag=tag, export=EXPORT, tree=str(tree), cwd=str(cwd or tree), artifacts=rj.parse_artifacts(HG, TP, NC),
                             env={"python": "3.12"}, gpu={"name": "L40S"}, installed_utc="2026-09-20T10:08:55Z", **kw)


def test_record_tree_is_the_installed_tree_never_the_default(tmp_path):
    own = tmp_path / "r16-int-3"; own.mkdir()
    rel = _rec(own)
    assert rel["tree"] == os.path.realpath(str(own))
    assert rel["schema"] == rj.SCHEMA and rel["tag"] == "r16-int-3" and rel["commit"] == EXPORT["commit"]
    assert set(rel["artifacts"]) == {"hidden_gpu_tree.so", "verity_fa2_matReq.so", "verity_native_collect.so"}


def test_record_refuses_a_tree_that_is_not_the_cwd(tmp_path):
    own = tmp_path / "r16-int-3"; own.mkdir()
    other = tmp_path / "shared"; other.mkdir()
    with pytest.raises(ValueError, match="refusing to write RELEASE.json"):
        _rec(own, cwd=other)


def test_record_resolves_symlinked_trees(tmp_path):
    own = tmp_path / "r16-int-3"; own.mkdir()
    link = tmp_path / "current"; link.symlink_to(own)
    rel = _rec(link, cwd=own)   # REMOTE_TREE given through a symlink, cwd is the real dir: same tree
    assert rel["tree"] == os.path.realpath(str(own))


def test_parse_artifacts_fa3_optional_and_fail_closed():
    arts = rj.parse_artifacts(HG, TP, NC, TP3)
    assert arts["verity_fa3_matReq.so"] == {"sha256": "4444", "path": "/workspace/cp/fa2/build/fa3_matReq/verity_fa3_matReq.so", "hdims": "64,128", "mask": "6", "layout": "v9"}
    assert arts["verity_fa2_matReq.so"]["arch"] == "8.9" and arts["hidden_gpu_tree.so"]["arch"] == "sm_89"
    assert "verity_fa3_matReq.so" not in rj.parse_artifacts(HG, TP, NC, "")
    with pytest.raises(ValueError):
        rj.parse_artifacts("HIDDEN-GPU-FAIL nvcc", TP, NC)
    with pytest.raises(ValueError):
        rj.parse_artifacts(HG, "FA2-TAP-FAIL build", NC)


def test_m0477_shape_does_not_name_the_own_tree():
    """A record with the own tree's tag and commit whose tree field is the pod's shared tree names the shared tree."""
    bad = {"tag": "r16-int-3", "commit": EXPORT["commit"], "tree": "/workspace/cp/shared"}
    assert rj.release_names_tree(bad, "/workspace/cp/shared")
    assert not rj.release_names_tree(bad, "/workspace/cp/r16-int-3")
    legacy = {"tag": "r15-int-6", "commit": "x"}          # a record without a tree field names no tree
    assert not rj.release_names_tree(legacy, "/workspace/cp/shared") and not rj.release_names_tree(legacy, "/workspace/cp/lane-moe")
    assert not rj.release_names_tree({}, "/workspace/cp/shared")


def test_write_then_check_pod_wide_and_tree_local(tmp_path):
    own = tmp_path / "r16-int-3"; own.mkdir()
    pod = tmp_path / "RELEASE.json"
    paths = rj.write_record(_rec(own), str(own), str(pod))
    assert paths == [str(pod), os.path.join(os.path.realpath(str(own)), "RELEASE.json")]
    assert json.load(open(pod))["tree"] == os.path.realpath(str(own)) == json.load(open(own / "RELEASE.json"))["tree"]
    assert rj.check_installed(str(own), "r16-int-3", EXPORT["commit"][:9], str(pod)) == []
    # the ship's read-back check catches a pod-wide record that names another tree
    json.dump({**json.load(open(pod)), "tree": "/workspace/cp/shared"}, open(pod, "w"))
    probs = rj.check_installed(str(own), "r16-int-3", EXPORT["commit"][:9], str(pod))
    assert len(probs) == 1 and "pod-wide" in probs[0] and "the record names another tree" in probs[0]
    # wrong tag / commit / missing file are problems too
    assert any("tag" in p for p in rj.check_installed(str(own), "r16-int-4", EXPORT["commit"][:9], str(own / "RELEASE.json")))
    assert any("commit" in p for p in rj.check_installed(str(own), "r16-int-3", "deadbeef0", str(own / "RELEASE.json")))
    assert any("missing" in p for p in rj.check_installed(str(own), "r16-int-3", EXPORT["commit"][:9], str(tmp_path / "nope.json")))


def _tree_with_export(tmp_path: Path, name: str, export=EXPORT) -> Path:
    t = tmp_path / name; t.mkdir()
    json.dump(export, open(t / "EXPORT.json", "w"))
    return t


def test_release_line_two_trees_on_one_pod(tmp_path):
    """The pod's shared tree = int-2, /workspace/cp/r16-int-3 = int-3 own tree.  The pod-wide record is the LAST ship
    (int-3); each tree still cites ITS release through the tree-local record, and neither cites the other's."""
    exp2 = {**EXPORT, "commit": "30486e648" + "0" * 31, "branch": "lane/r16-int"}
    base = _tree_with_export(tmp_path, "shared", exp2)
    own = _tree_with_export(tmp_path, "r16-int-3")
    pod = tmp_path / "RELEASE.json"
    rel2 = rj.release_record(tag="r16-int-2", export=exp2, tree=str(base), cwd=str(base), artifacts=rj.parse_artifacts(HG, TP, NC), env={}, gpu={}, installed_utc="t")
    rj.write_record(rel2, str(base), str(pod))
    rj.write_record(_rec(own), str(own), str(pod))          # the later own-tree ship overwrites the pod-wide file
    assert json.load(open(pod))["tag"] == "r16-int-3"
    line_own, src_own = rj.release_line(str(own), pod_path=str(pod))
    line_base, src_base = rj.release_line(str(base), pod_path=str(pod))
    assert line_own.startswith("r16-int-3|commit=a2dabe493 ") and src_own == os.path.join(os.path.realpath(str(own)), "RELEASE.json")
    assert line_base.startswith("r16-int-2|commit=30486e648 ") and src_base == os.path.join(os.path.realpath(str(base)), "RELEASE.json")
    assert f"tree={os.path.realpath(str(own))}" in line_own and f"tree={os.path.realpath(str(base))}" in line_base


def test_release_line_pod_wide_fallback_and_unreleased(tmp_path):
    own = _tree_with_export(tmp_path, "r16-int-3")
    pod = tmp_path / "RELEASE.json"
    # pod-wide only (a ship that wrote no tree-local record): applies when it names this tree
    json.dump({**_rec(own), "tree": os.path.realpath(str(own))}, open(pod, "w"))
    line, src = rj.release_line(str(own), pod_path=str(pod))
    assert line.startswith("r16-int-3|") and src == str(pod)
    # pod-wide names the shared tree -> this tree is UNRELEASED, by name
    json.dump({**_rec(own), "tree": "/workspace/cp/shared"}, open(pod, "w"))
    line, src = rj.release_line(str(own), pod_path=str(pod))
    assert line.startswith("UNRELEASED|commit=a2dabe493 ") and "is the tree /workspace/cp/shared, not this one" in line and src == "-"
    # a record with no tree field cites no tree
    json.dump({k: v for k, v in _rec(own).items() if k != "tree"}, open(pod, "w"))
    line, src = rj.release_line(str(own), pod_path=str(pod))
    assert line.startswith("UNRELEASED|") and "is the tree None, not this one" in line and src == "-"
    # a tree-local record whose commit no longer matches EXPORT (ship.sh over the tree) never cites the stale release
    rj.write_record(_rec(own), str(own), str(pod))
    json.dump({**EXPORT, "commit": "f" * 40}, open(own / "EXPORT.json", "w"))
    line, src = rj.release_line(str(own), pod_path=str(pod))
    assert line.startswith("UNRELEASED|commit=fffffffff ") and "!= this tree's EXPORT" in line
    # no records at all
    lane = _tree_with_export(tmp_path, "lane-moe")
    line, src = rj.release_line(str(lane), pod_path=str(tmp_path / "none.json"))
    assert line.startswith("UNRELEASED|") and "not installed here by pod_release.sh" in line and src == "-"


def test_native_collect_build_dir_per_tree():
    assert rj.native_collect_build_dir("/workspace/cp/shared") == "/workspace/cp/nc_build-shared"
    assert rj.native_collect_build_dir("/workspace/cp/shared/") == "/workspace/cp/nc_build-shared"
    assert rj.native_collect_build_dir("/workspace/cp/r16-int-3") == "/workspace/cp/nc_build-r16-int-3"
    assert rj.native_collect_build_dir("/workspace/cp/r17-int-1/") == "/workspace/cp/nc_build-r17-int-1"


def test_cli_check_and_line_as_a_process(tmp_path):
    """The row reads its release line through `verity-vllm release-json line` in a child process: exit codes and output as documented."""
    own = _tree_with_export(tmp_path, "r16-int-3")
    pod = tmp_path / "RELEASE.json"
    rj.write_record(_rec(own), str(own), str(pod))
    py = [sys.executable, "-m", "verity_vllm.pipeline.cli", "release-json"]
    cwd = HERE.parents[1]
    r = subprocess.run(py + ["check", "--tree", str(own), "--tag", "r16-int-3", "--commit", "a2dabe493", "--pod-path", str(pod)], capture_output=True, text=True,
                       cwd=cwd)
    assert r.returncode == 0 and r.stdout.startswith("RELEASE-TREE-OK r16-int-3 a2dabe493 tree=")
    r = subprocess.run(py + ["check", "--tree", str(own), "--tag", "r16-int-4", "--commit", "a2dabe493", "--pod-path", str(pod)], capture_output=True, text=True,
                       cwd=cwd)
    assert r.returncode == 1 and r.stdout.startswith("RELEASE-TREE-FAIL")
    r = subprocess.run(py + ["line", str(own), "--pod-path", str(pod)], capture_output=True, text=True, cwd=cwd)
    line, src = r.stdout.rstrip("\n").split("\t")
    assert r.returncode == 0 and line.startswith("r16-int-3|commit=a2dabe493 ") and src.endswith("/r16-int-3/RELEASE.json")
    r = subprocess.run(py + ["nc-build", "/workspace/cp/r16-int-3"], capture_output=True, text=True, cwd=cwd)
    assert r.stdout.strip() == "/workspace/cp/nc_build-r16-int-3"


def test_the_row_reads_the_release_through_the_module(tmp_path, monkeypatch):
    """The row's release line: from `release-json line` (tree-local first), the record USED is the one copied into the evidence, and a
    released tree takes the release's collector build dir unless the caller set NATIVE_COLLECT_BUILD."""
    from tests.pipeline.test_row import _row
    rel = tmp_path / "RELEASE.json"
    rel.write_text(json.dumps({"tag": "r16-int-3", "build_dirs": {"native_collect": str(tmp_path / "nc_rel")}}))

    def prepared(*flags, environ=None):
        r = _row(tmp_path / "sweep", "--native-collect-default", str(tmp_path / "nc_default"), *flags, environ=environ)
        monkeypatch.setattr(r, "release_line", lambda fallback: (f"r16-int-3|commit=a2dabe493 tree={r.tree}", str(rel)))
        for step in ("precheck_identity", "precheck_target", "admission", "install_signal_forwarding"):
            monkeypatch.setattr(r, step, lambda: None)

        def shape():
            r.B = r.I = r.O = r.LP = r.T = 1
            r.ck = r.tgt = r.hidden_so = r.code9 = ""
        monkeypatch.setattr(r, "shape", shape)
        r.prepare()
        return r
    r = prepared()
    assert json.loads((Path(r.d) / "release.json").read_text()) == json.loads(rel.read_text())
    assert r.ncb == r.child_env["NATIVE_COLLECT_BUILD"] == str(tmp_path / "nc_rel")
    assert json.loads((Path(r.d) / "tree.json").read_text())["release_line"].startswith("r16-int-3|commit=a2dabe493 ")
    for pinned in (prepared("--native-collect-build", str(tmp_path / "nc_own")),
                   prepared(environ={"NATIVE_COLLECT_BUILD": str(tmp_path / "nc_own")})):
        assert pinned.ncb == pinned.child_env["NATIVE_COLLECT_BUILD"] == str(tmp_path / "nc_own")
