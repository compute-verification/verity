"""[R17 rel, exec M-0477 / M-0709 (R16)] RELEASE.json names the tree the ship INSTALLED -- the own-tree bug class, fixed at the writer
(explicit tree, refused unless it is the cwd), the reader (tree-local record first) and the ship (read-back check).  Torch-free."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm.harness import release_json as rj

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
    assert rel["tree"] != rj.DEFAULT_TREE
    assert rel["schema"] == rj.SCHEMA and rel["tag"] == "r16-int-3" and rel["commit"] == EXPORT["commit"]
    assert set(rel["artifacts"]) == {"hidden_gpu_tree.so", "verity_fa2_matReq.so", "verity_native_collect.so"}


def test_record_refuses_a_tree_that_is_not_the_cwd(tmp_path):
    own = tmp_path / "r16-int-3"; own.mkdir()
    other = tmp_path / "veritor"; other.mkdir()
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
    """The R16 record as measured (exec M-0477 08:06Z / M-0709 10:10Z): tag + commit of the own tree, tree field = the default tree."""
    bad = {"tag": "r16-int-3", "commit": EXPORT["commit"], "tree": "/workspace/cp/veritor"}
    assert rj.release_names_tree(bad, "/workspace/cp/veritor")
    assert not rj.release_names_tree(bad, "/workspace/cp/r16-int-3")
    legacy = {"tag": "r15-int-6", "commit": "x"}          # pre-R16 record without a tree field: the default tree only
    assert rj.release_names_tree(legacy, "/workspace/cp/veritor") and not rj.release_names_tree(legacy, "/workspace/cp/lane-moe")
    assert not rj.release_names_tree({}, "/workspace/cp/veritor")


def test_write_then_check_pod_wide_and_tree_local(tmp_path):
    own = tmp_path / "r16-int-3"; own.mkdir()
    pod = tmp_path / "RELEASE.json"
    paths = rj.write_record(_rec(own), str(own), str(pod))
    assert paths == [str(pod), os.path.join(os.path.realpath(str(own)), "RELEASE.json")]
    assert json.load(open(pod))["tree"] == os.path.realpath(str(own)) == json.load(open(own / "RELEASE.json"))["tree"]
    assert rj.check_installed(str(own), "r16-int-3", EXPORT["commit"][:9], str(pod)) == []
    # the ship's read-back check catches the M-0477 shape: a pod-wide record that names another tree
    json.dump({**json.load(open(pod)), "tree": "/workspace/cp/veritor"}, open(pod, "w"))
    probs = rj.check_installed(str(own), "r16-int-3", EXPORT["commit"][:9], str(pod))
    assert len(probs) == 1 and "pod-wide" in probs[0] and "M-0477" in probs[0]
    # wrong tag / commit / missing file are problems too
    assert any("tag" in p for p in rj.check_installed(str(own), "r16-int-4", EXPORT["commit"][:9], str(own / "RELEASE.json")))
    assert any("commit" in p for p in rj.check_installed(str(own), "r16-int-3", "deadbeef0", str(own / "RELEASE.json")))
    assert any("missing" in p for p in rj.check_installed(str(own), "r16-int-3", EXPORT["commit"][:9], str(tmp_path / "nope.json")))


def _tree_with_export(tmp_path: Path, name: str, export=EXPORT) -> Path:
    t = tmp_path / name; t.mkdir()
    json.dump(export, open(t / "EXPORT.json", "w"))
    return t


def test_release_line_two_trees_on_one_pod(tmp_path):
    """cp-r1 10:08Z: /workspace/cp/veritor = int-2, /workspace/cp/r16-int-3 = int-3 own tree.  The pod-wide record is the LAST ship
    (int-3); each tree still cites ITS release through the tree-local record, and neither cites the other's."""
    exp2 = {**EXPORT, "commit": "30486e648" + "0" * 31, "branch": "lane/r16-int"}
    base = _tree_with_export(tmp_path, "veritor", exp2)
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
    # pod-wide only (a pre-R17 ship wrote no tree-local record): applies when it names this tree
    json.dump({**_rec(own), "tree": os.path.realpath(str(own))}, open(pod, "w"))
    line, src = rj.release_line(str(own), pod_path=str(pod))
    assert line.startswith("r16-int-3|") and src == str(pod)
    # the M-0477 shape: pod-wide names the default tree -> this tree is UNRELEASED, by name
    json.dump({**_rec(own), "tree": "/workspace/cp/veritor"}, open(pod, "w"))
    line, src = rj.release_line(str(own), pod_path=str(pod))
    assert line.startswith("UNRELEASED|commit=a2dabe493 ") and "is the tree /workspace/cp/veritor, not this one" in line and src == "-"
    # a tree-local record whose commit no longer matches EXPORT (lane ship.sh over the tree) never cites the stale release
    rj.write_record(_rec(own), str(own), str(pod))
    json.dump({**EXPORT, "commit": "f" * 40}, open(own / "EXPORT.json", "w"))
    line, src = rj.release_line(str(own), pod_path=str(pod))
    assert line.startswith("UNRELEASED|commit=fffffffff ") and "!= this tree's EXPORT" in line
    # no records at all
    lane = _tree_with_export(tmp_path, "lane-moe")
    line, src = rj.release_line(str(lane), pod_path=str(tmp_path / "none.json"))
    assert line.startswith("UNRELEASED|") and "not installed here by pod_release.sh" in line and src == "-"


def test_native_collect_build_dir_per_tree():
    assert rj.native_collect_build_dir("/workspace/cp/veritor") == "/workspace/cp/nc_build"
    assert rj.native_collect_build_dir("/workspace/cp/veritor/") == "/workspace/cp/nc_build"
    assert rj.native_collect_build_dir("/workspace/cp/r16-int-3") == "/workspace/cp/nc_build-r16-int-3"
    assert rj.native_collect_build_dir("/workspace/cp/r17-int-1/") == "/workspace/cp/nc_build-r17-int-1"


def test_cli_check_and_line_with_system_python(tmp_path):
    """row_pod / pod_release call the file with the pod's system python3: stdlib only, exit codes as documented."""
    own = _tree_with_export(tmp_path, "r16-int-3")
    pod = tmp_path / "RELEASE.json"
    rj.write_record(_rec(own), str(own), str(pod))
    py = [sys.executable, str(HERE / ".." / "harness" / "release_json.py")]
    r = subprocess.run(py + ["check", "--tree", str(own), "--tag", "r16-int-3", "--commit", "a2dabe493", "--pod-path", str(pod)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("RELEASE-TREE-OK r16-int-3 a2dabe493 tree=")
    r = subprocess.run(py + ["check", "--tree", str(own), "--tag", "r16-int-4", "--commit", "a2dabe493", "--pod-path", str(pod)], capture_output=True, text=True)
    assert r.returncode == 1 and r.stdout.startswith("RELEASE-TREE-FAIL")
    r = subprocess.run(py + ["line", str(own), "--pod-path", str(pod)], capture_output=True, text=True)
    line, src = r.stdout.rstrip("\n").split("\t")
    assert r.returncode == 0 and line.startswith("r16-int-3|commit=a2dabe493 ") and src.endswith("/r16-int-3/RELEASE.json")
    r = subprocess.run(py + ["nc-build", "/workspace/cp/r16-int-3"], capture_output=True, text=True)
    assert r.stdout.strip() == "/workspace/cp/nc_build-r16-int-3"


def test_pod_release_sh_writes_through_the_module_and_checks():
    """The shell side: no constant tree, the module gets --tree "$TREE", the record is re-read after the ship, the pod-side steps run
    under the per-pod lock with the per-tree collector build dir."""
    s = (HERE / "pod_release.sh").read_text()
    assert '"tree": "/workspace/cp/veritor"' not in s and "os.path.realpath(os.getcwd())" not in s
    assert re.search(r'release_json write --tag "\$TAG" --tree "\$TREE"', s)
    assert re.search(r"release_json\.py check --tree '\$TREE' --tag '\$TAG' --commit '\$HEAD9'", s)
    assert "RELEASE-TREE-OK" in s and "RELEASE-FAIL $TAG $POD: RELEASE.json does not name the installed tree" in s
    assert re.search(r"flock -w \$\{RELEASE_LOCK_WAIT_S:-3600\} /workspace/cp/release\.lock bash -s", s)
    assert re.search(r'NC_BUILD=\$\(python3 verity_vllm/harness/release_json\.py nc-build "\$TREE"\)', s)
    assert "NATIVE_COLLECT_BUILD=${NC_BUILD:-/workspace/cp/nc_build}" in s
    assert '--nc-build "$NATIVE_COLLECT_BUILD"' in s
    # the TREE the pod side sees is the REMOTE_TREE the laptop side resolved; readiness uses the same collector build dir
    assert re.search(r"TREE='\$TREE' NC_BUILD='\$NC_BUILD'", s)
    assert 'REMOTE_ENV="${REMOTE_ENV:-} NATIVE_COLLECT_BUILD=$NC_BUILD"' in s
    assert subprocess.run(["bash", "-n", str(HERE / "pod_release.sh")]).returncode == 0


def test_row_pod_sh_reads_the_release_through_the_module():
    """row_pod's REL-09 line: from release_json.py (tree-local first), the record USED is the one copied into the evidence, and a
    released tree takes the release's collector build dir unless the caller set NATIVE_COLLECT_BUILD."""
    s = (HERE / "row_pod.sh").read_text()
    assert 'json.load(open("/workspace/cp/RELEASE.json"))' not in s      # no direct pod-wide read any more
    assert re.search(r'REL_OUT=\$\(python3 verity_vllm/harness/release_json\.py line "\$TREE_HERE"', s)
    assert 'cp "$REL_SRC" "$D/release.json"' in s and 'cp /workspace/cp/RELEASE.json "$D/release.json"' not in s
    assert "NCB_SET=${NATIVE_COLLECT_BUILD:-}" in s and "get('build_dirs') or {}).get('native_collect')" in s
    assert subprocess.run(["bash", "-n", str(HERE / "row_pod.sh")]).returncode == 0


def test_pod_release_busy_pattern_names_the_stage_processes_launched_outside_row_pod():
    """[R18 infra, F-r18-infra-2, M-0105] pod_release.sh busy(): a Match stage run outside row_pod.sh from the default tree (run_config +
    its m1_capture / fold workers, cb_a.global_match, cb_a.compare) is row-like -- cp-r18-l40s 22:57Z the int-2 ship landed under x09's
    fold the moment the capture freed the GPU.  The pgrep -f pattern is applied to real command lines of that shape."""
    src = (HERE / "pod_release.sh").read_text()
    m = re.search(r"pgrep -f '([^']+)' \| grep -v -w", src)
    assert m, "busy() pgrep pattern not found"
    pat = re.compile(m.group(1))
    row_like = [
        "bash verity_vllm/ops/row_pod.sh smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager B0 wl.json /d build",
        "/workspace/venv-py312/bin/python -m verity_vllm.harness.run_config --case MISTRAL7B --workload w.json --out /o/match --phase all",
        "/workspace/venv-py312/bin/python -m verity_vllm.harness.run_config --worker fold --log /o/match/capture/log.jsonl.gz",
        "/workspace/venv-cu129/bin/python -m verity_vllm.observe.m1_capture --case MISTRAL7B",
        "/workspace/venv-py312/bin/python -m verity_vllm.check.global_match --gprog p.json --workload w.json",
        "/workspace/venv-cu129/bin/python -m verity_vllm.harness.commit_delta --case B0 --workload w.json",
    ]
    for cmd in row_like:
        assert pat.search(cmd), cmd
    for cmd in ["/usr/bin/python3 zk/campaign/tele sample --attempt-dir /a", "nvidia-smi --query-gpu=memory.used --format=csv", "sshd: root@notty"]:
        assert not pat.search(cmd), cmd
