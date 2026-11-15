"""RELEASE.json of a shipped worker package (schema verity-sweep/release/v1) -- writer, reader and the release line.

The `tree` field names the tree THIS ship installed.  A RELEASE.json naming the pod's shared default tree for a release installed as its
OWN tree makes a row from the own tree read UNRELEASED by row_pod's release rule ("pod release ... is the tree <default tree>, not this
one") and lets a row from the default tree cite the release falsely, so the field is never a constant and there is no default tree:
a record names its tree or none.  Rules:

  * the writer takes the tree EXPLICITLY (the REMOTE_TREE the ship installed into) and refuses when it is not the process cwd;
  * the record is written pod-wide (/workspace/cp/RELEASE.json, the latest ship on the pod) AND tree-local (<tree>/RELEASE.json), so a
    second tree on the pod never erases the first tree's citation;
  * `check` re-reads the written file after the ship and fails the release (RELEASE-FAIL) when tree / tag / commit do not name the
    installed tree -- the ship never ends with a RELEASE.json that no row can cite;
  * `line` is row_pod's release line: the tree-local record when it names this tree, else the pod-wide one, else
    UNRELEASED -- a record applies only when its `tree` == realpath(cwd) AND its commit + dirty-patch digest equal this tree's EXPORT.json.

Stdlib only at import time; `write` imports torch/vllm lazily for the env identity.

  verity-vllm release-json write --tag <tag> --tree <tree> --tp "FA2-TAP-OK ..." --nc "<sha> <path>" [--tp3 "FA3-TAP-OK ..."] [--nc-build <dir>]
  verity-vllm release-json check --tree <tree> --tag <tag> --commit <sha>      -> RELEASE-TREE-OK ... | RELEASE-TREE-FAIL <why> (exit 1)
  verity-vllm release-json line <tree>                                          -> "<release line>\t<record path or ->"
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field

from verity_vllm.config import option

SCHEMA = "verity-sweep/release/v1"
POD_RELEASE_JSON = "/workspace/cp/RELEASE.json"
DEFAULT_NC_BUILD = "/workspace/cp/nc_build"
TREE_LOCAL_NAME = "RELEASE.json"


def _real(p: str) -> str:
    return os.path.realpath(p)


def native_collect_build_dir(tree: str) -> str:
    """The collector JIT build dir of a tree: one directory per tree (/workspace/cp/nc_build-<name>) so two ships / two trees
    never build in one directory."""
    t = _real(tree)
    return f"{DEFAULT_NC_BUILD}-{os.path.basename(t.rstrip('/')) or 'tree'}"


def parse_artifacts(tp: str, nc: str, tp3: str = "") -> dict:
    """The two (three on cc 9.x) built artifacts from the pod-side OK lines."""
    tp_f, nc_f = tp.split(), nc.split()
    if len(tp_f) < 3 or not tp_f[0].startswith("FA2-TAP-OK"):
        raise ValueError(f"fa2 tap line not OK: {tp!r}")
    if len(nc_f) < 2:
        raise ValueError(f"native collector line malformed: {nc!r}")
    arts = {"verity_fa2_matReq.so": {"sha256": tp_f[1], "path": tp_f[2], **dict(kv.split("=", 1) for kv in tp_f[3:] if "=" in kv)},
            # source_digest = the key of the JIT dir the .so was built in (native_jit); bindings = the pybind names checked
            "verity_native_collect.so": {"sha256": nc_f[0], "path": nc_f[1], **dict(kv.split("=", 1) for kv in nc_f[2:] if "=" in kv)}}
    if tp3:
        f3 = tp3.split()
        if len(f3) < 3 or not f3[0].startswith("FA3-TAP-OK"):
            raise ValueError(f"fa3 tap line not OK: {tp3!r}")
        arts["verity_fa3_matReq.so"] = {"sha256": f3[1], "path": f3[2], **dict(kv.split("=", 1) for kv in f3[3:] if "=" in kv)}
    return arts


def release_record(*, tag: str, export: dict, tree: str, cwd: str, artifacts: dict, env: dict, gpu: dict,
                   installed_utc: str | None = None, build_dirs: dict | None = None) -> dict:
    """The RELEASE.json record.  `tree` is the tree the ship installed into (REMOTE_TREE); it must be the directory this runs in --
    a mismatch is refused, never written (fail-closed)."""
    t, c = _real(tree), _real(cwd)
    if t != c:
        raise ValueError(f"release tree {tree} (realpath {t}) is not the installing cwd {cwd} (realpath {c}) -- refusing to write RELEASE.json")
    for k in ("commit", "branch", "dirty_patch_sha256", "dirty_patch_bytes", "exported_utc"):
        if k not in export:
            raise ValueError(f"EXPORT.json lacks {k}")
    rel = {"schema": SCHEMA, "tag": tag, "commit": export["commit"], "branch": export["branch"], "dirty_patch_sha256": export["dirty_patch_sha256"],
           "dirty_patch_bytes": export["dirty_patch_bytes"], "shipped_utc": export["exported_utc"],
           "installed_utc": installed_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tree": t, "artifacts": artifacts, "env": env, "gpu": gpu}
    if build_dirs:
        rel["build_dirs"] = dict(build_dirs)
    return rel


def release_names_tree(rel: dict, tree: str) -> bool:
    """row_pod's rule: a RELEASE.json applies to a tree only when its `tree` is that tree.  A record without a tree field (the
    older pod-wide form) names no tree."""
    if not rel or not rel.get("tree"):
        return False
    return _real(rel["tree"]) == _real(tree)


def release_matches_export(rel: dict, export: dict) -> bool:
    c, d = export.get("commit") or "", export.get("dirty_patch_sha256") or ""
    return bool(rel) and rel.get("commit") == c and (rel.get("dirty_patch_sha256") or "") == d


def write_record(rel: dict, tree: str, pod_path: str = POD_RELEASE_JSON) -> list[str]:
    """Pod-wide (latest ship on the pod) + tree-local; tmp+rename so a reader never sees a torn file.  Returns the paths written."""
    out = []
    for p in (pod_path, os.path.join(_real(tree), TREE_LOCAL_NAME)):
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rel, f, indent=1)
        os.replace(tmp, p)
        out.append(p)
    return out


def check_installed(tree: str, tag: str, commit: str, pod_path: str = POD_RELEASE_JSON) -> list[str]:
    """After the ship: the pod-wide AND the tree-local record must name this tree, this tag and this commit (prefix match on the
    commit: the ship knows 9 hex digits).  Returns the problems (empty == OK)."""
    problems = []
    t = _real(tree)
    for label, p in (("pod-wide", pod_path), ("tree-local", os.path.join(t, TREE_LOCAL_NAME))):
        if not os.path.exists(p):
            problems.append(f"{label} {p}: missing")
            continue
        try:
            rel = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            problems.append(f"{label} {p}: unreadable ({e})")
            continue
        if not release_names_tree(rel, t):
            problems.append(f"{label} {p}: tree {rel.get('tree')!r} != installed tree {t} (the record names another tree)")
        if rel.get("tag") != tag:
            problems.append(f"{label} {p}: tag {rel.get('tag')!r} != {tag}")
        rc = str(rel.get("commit") or "")
        if not (rc.startswith(commit) or commit.startswith(rc)) or not rc:
            problems.append(f"{label} {p}: commit {rc[:9] or None} != {commit[:9]}")
    return problems


def release_line(tree: str, export_path: str | None = None, pod_path: str = POD_RELEASE_JSON) -> tuple[str, str]:
    """row_pod's release line + the record path it came from ('-' when UNRELEASED).  Tree-local record first (the tree whose
    code runs), then the pod-wide one; each applies only when it names this tree and matches its EXPORT.json."""
    t = _real(tree)
    exp_p = export_path or os.path.join(t, "EXPORT.json")
    exp = json.load(open(exp_p)) if os.path.exists(exp_p) else {}
    c, d = exp.get("commit") or "", exp.get("dirty_patch_sha256") or ""
    why = []
    for p in (os.path.join(t, TREE_LOCAL_NAME), pod_path):
        if not os.path.exists(p):
            why.append(f"no {p}")
            continue
        try:
            rel = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            why.append(f"{p} unreadable ({e})")
            continue
        if not release_names_tree(rel, t):
            why.append(f"{p}: release {rel.get('tag')} {str(rel.get('commit'))[:9]} is the tree {rel.get('tree')}, not this one")
            continue
        if not release_matches_export(rel, exp):
            why.append(f"{p}: release {rel.get('tag')} {str(rel.get('commit'))[:9]} dirty {str(rel.get('dirty_patch_sha256') or '')[:12]} != this tree's EXPORT {c[:9]} dirty {d[:12]}")
            continue
        return f"{rel['tag']}|commit={c[:9]} tree={t} branch={exp.get('branch')} dirty={d[:12]}", p
    return f"UNRELEASED|commit={c[:9] or 'unknown'} tree={t} branch={exp.get('branch')} dirty={d[:12]} ({'; '.join(why)}; not installed here by pod_release.sh)", "-"


def _env_identity() -> tuple[dict, dict, str]:
    import torch  # pod only
    try:
        import vllm
        vllm_v = vllm.__version__
    except Exception as e:  # noqa: BLE001
        vllm_v = f"unavailable: {e}"
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    p = torch.cuda.get_device_properties(0)
    env = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "vllm": vllm_v, "nvcc": nvcc,
           "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(), "venv": sys.prefix}
    gpu = {"name": p.name, "compute_capability": f"{p.major}.{p.minor}", "num_sms": p.multi_processor_count, "total_mem_gb": round(p.total_memory / 2**30, 1)}
    return env, gpu, freeze


@dataclass(frozen=True, kw_only=True)
class WriteOptions:
    "write /workspace/cp/RELEASE.json + <tree>/RELEASE.json (pod side, cwd must be the tree)"

    tag: str = option("--tag")
    tree: str = option("--tree")
    tp: str = option("--tp")
    nc: str = option("--nc")
    tp3: str = option("--tp3", default="")
    nc_build: str | None = option("--nc-build", default=None,
                                  help="collector JIT build dir used by this ship (recorded under build_dirs)")
    out: str = option("--out", default=POD_RELEASE_JSON)
    export: str = option("--export", default="EXPORT.json")
    freeze_out: str | None = option("--freeze-out", default=None,
                                    help="write pip freeze here (default /workspace/cp/release-logs/pip_freeze_<tag>.txt)")
    cmd: str = field(default="write", init=False)


@dataclass(frozen=True, kw_only=True)
class CheckOptions:
    "after the ship: both records name this tree / tag / commit"

    tree: str = option("--tree")
    tag: str = option("--tag")
    commit: str = option("--commit")
    pod_path: str = option("--pod-path", default=POD_RELEASE_JSON)
    cmd: str = field(default="check", init=False)


@dataclass(frozen=True, kw_only=True)
class LineOptions:
    "row_pod's release line for <tree>"

    tree: str = option("tree")
    pod_path: str = option("--pod-path", default=POD_RELEASE_JSON)
    cmd: str = field(default="line", init=False)


@dataclass(frozen=True, kw_only=True)
class NcBuildOptions:
    "the collector JIT build dir of <tree> (one per tree)"

    tree: str = option("tree")
    cmd: str = field(default="nc-build", init=False)


COMMANDS = {"write": WriteOptions, "check": CheckOptions, "line": LineOptions, "nc-build": NcBuildOptions}


def main(a: WriteOptions | CheckOptions | LineOptions | NcBuildOptions) -> int:
    if a.cmd == "nc-build":
        print(native_collect_build_dir(a.tree))
        return 0
    if a.cmd == "write":
        export = json.load(open(a.export))
        env, gpu, freeze = _env_identity()
        arts = parse_artifacts(a.tp, a.nc, a.tp3)
        bd = {"native_collect": a.nc_build} if a.nc_build else None
        rel = release_record(tag=a.tag, export=export, tree=a.tree, cwd=os.getcwd(), artifacts=arts, env=env, gpu=gpu, build_dirs=bd)
        paths = write_record(rel, a.tree, a.out)
        fo = a.freeze_out or f"/workspace/cp/release-logs/pip_freeze_{a.tag}.txt"
        os.makedirs(os.path.dirname(fo), exist_ok=True)
        open(fo, "w").write(freeze)
        print("  RELEASE.json written:", json.dumps({"tag": rel["tag"], "commit": rel["commit"], "tree": rel["tree"]}), "->", " + ".join(paths))
        return 0
    if a.cmd == "check":
        probs = check_installed(a.tree, a.tag, a.commit, a.pod_path)
        if probs:
            print("RELEASE-TREE-FAIL " + " | ".join(probs))
            return 1
        print(f"RELEASE-TREE-OK {a.tag} {a.commit[:9]} tree={_real(a.tree)}")
        return 0
    if a.cmd == "line":
        line, src = release_line(a.tree, pod_path=a.pod_path)
        print(f"{line}\t{src}")
        return 0
    return 2
