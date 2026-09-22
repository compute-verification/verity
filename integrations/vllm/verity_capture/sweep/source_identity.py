"""Verity source identity + precheck (F-r17-tp-06 fixed by construction; BRIEF R18 Phase 0).

    python -m verity_capture.sweep.source_identity [--tree ROOT] [--sha SHA] [--out FILE] [--stage NAME]
        exit 0  every important package (verity.ir, verity_capture, vllm-poc, hidden_gpu, telemetry) resolves from ROOT
        exit 3  PRECHECK_FAIL_SOURCE_IDENTITY -- the first line of stderr names the class, the following lines each mismatch

The requested tree ROOT defaults to THIS FILE'S tree (never an environment default): the runner that imports this module IS the tree
that must run.  `VERITOR_REPO`, when set, must name the same tree -- a caller that exports another tree is the mixed-tree case and is
refused by name.  `--out` writes the manifest record (schema research-infra/source-identity/v1); the same record goes into
versions.json (commit_delta) and the attempt's job.json (tele init --source-identity).

In-process: `precheck(stage)` is called by the Commit (commit_delta.main, before the engine is built -> before any inference) and by
native_host._gpu_ext_load after `import hidden_gpu` (the module that actually loaded must be this tree's).  Both raise
telemetry.source_identity.SourceIdentityError, whose message starts with PRECHECK_FAIL_SOURCE_IDENTITY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# this file: <tree>/verity_capture/sweep/source_identity.py -> the tree is parents[2]; stated once, derived from __file__, never from env
TREE = Path(__file__).resolve().parents[2]
ROOT_MARKERS = ("verity_capture/__init__.py", "verity.ir/__init__.py", "verity_vllm_adapter/__init__.py")
_ZK = TREE / "zk" / "campaign"
if str(_ZK) not in sys.path:
    sys.path.append(str(_ZK))    # the telemetry package of THIS tree (appended: shadows nothing; a foreign `telemetry` earlier on the path is caught by the check below)
from telemetry import source_identity as _sid  # noqa: E402

FAIL_CLASS = _sid.FAIL_CLASS
SourceIdentityError = _sid.SourceIdentityError

# hidden_gpu is imported by native_host._gpu_ext_load from <tree>/out/gen/r9/cmt-hidden/src (materialised by pod_hidden_gpu.sh from the
# shipped copy verity_capture/sweep/hidden_gpu_src); both are candidates, the r9 path first because that is what native_host inserts
HIDDEN_GPU_CANDIDATES = ("out/gen/r9/cmt-hidden/src/hidden_gpu.py", "verity_capture/sweep/hidden_gpu_src/hidden_gpu.py")
# vllm-poc is a sys.path ENTRY (PYTHONPATH=.), not a package: its identity is the adapter package inside it
VLLM_POC_PACKAGE = "verity_vllm_adapter"
PACKAGES = ("verity.ir", "verity_capture", "vllm-poc", "hidden_gpu", "telemetry")


def requested_tree(tree: str | os.PathLike[str] | None = None) -> str:
    """The tree this job must run from: the explicit argument, else THIS file's tree.  $VERITOR_REPO is checked against it, never used
    as the default -- a differing value is the mixed-tree case (F-r17-tp-06) and fails by name."""
    root = os.path.realpath(os.fspath(tree) if tree else str(TREE))
    env = os.environ.get("VERITOR_REPO")
    if env and os.path.realpath(env) != root:
        raise SourceIdentityError([{"package": "VERITOR_REPO", "resolved_path": os.path.realpath(env), "repo_root": os.path.realpath(env),
                                    "git_sha": _sid.tree_identity(os.path.realpath(env)).get("git_sha"),
                                    "why": f"$VERITOR_REPO names another tree than the code that runs ({root})"}], root)
    return root


def resolve(tree: str | os.PathLike[str] | None = None) -> dict[str, dict[str, Any]]:
    """Identity of every package in PACKAGES as the CURRENT interpreter resolves it (sys.modules first, then find_spec on sys.path)."""
    root = os.path.realpath(os.fspath(tree) if tree else str(TREE))
    ids: dict[str, dict[str, Any]] = {}
    ids["verity.ir"] = _sid.resolve_module("verity.ir", ROOT_MARKERS)
    ids["verity_capture"] = _sid.resolve_module("verity_capture", ROOT_MARKERS)
    vp = _sid.resolve_module(VLLM_POC_PACKAGE, ROOT_MARKERS)
    vp["package"] = "vllm-poc"
    vp["import_name"] = VLLM_POC_PACKAGE
    if vp.get("resolved_path"):
        # <tree>/verity_vllm_adapter/__init__.py -> the vllm-poc directory is the sys.path entry that resolved it
        vp["sys_path_entry"] = str(Path(vp["resolved_path"]).parent.parent)
    ids["vllm-poc"] = vp
    ids["hidden_gpu"] = _sid.resolve_file("hidden_gpu", [os.path.join(root, c) for c in HIDDEN_GPU_CANDIDATES], ROOT_MARKERS)
    ids["telemetry"] = _sid.resolve_module("telemetry", ROOT_MARKERS)
    return ids


def manifest(tree: str | os.PathLike[str] | None = None, *, sha: str | None = None, strict_env: bool = True) -> dict[str, Any]:
    """The record + verdict (no raise).  `ok` False carries fail_class PRECHECK_FAIL_SOURCE_IDENTITY and the mismatches."""
    try:
        root = requested_tree(tree) if strict_env else os.path.realpath(os.fspath(tree) if tree else str(TREE))
        env_mm: list[dict[str, Any]] = []
    except SourceIdentityError as e:
        root, env_mm = e.requested_root, e.mismatches
    ids = resolve(root)
    mm = env_mm + _sid.check(ids, root, requested_sha=sha, optional=("hidden_gpu",))
    doc = _sid.document(ids, root, mm)
    doc["VERITOR_REPO"] = os.environ.get("VERITOR_REPO")
    doc["PYTHONPATH"] = os.environ.get("PYTHONPATH")
    return doc


def precheck(stage: str, tree: str | os.PathLike[str] | None = None, *, sha: str | None = None, out: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Assert (raise SourceIdentityError) that every package resolves from the requested tree; writes the record to `out` first when
    given, so the failing document is on disk beside the class name.  Called BEFORE any inference."""
    doc = manifest(tree, sha=sha)
    doc["stage"] = stage
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    if not doc["ok"]:
        raise SourceIdentityError(doc["mismatches"], doc["requested_root"])
    return doc


def assert_loaded_module_is_ours(mod: Any, name: str, tree: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """After an import that a sys.path insertion made possible (native_host: `import hidden_gpu`), the module that LOADED must live in
    the requested tree -- the by-construction guard at the import site itself."""
    root = os.path.realpath(os.fspath(tree) if tree else str(TREE))
    f = getattr(mod, "__file__", None)
    rec = {"package": name, "resolved_path": os.path.realpath(f) if f else None, "repo_root": _sid.repo_root_of(f, ROOT_MARKERS) if f else None, "loaded": True}
    rec.update(_sid.tree_identity(rec["repo_root"]))
    mm = _sid.check({name: rec}, root)
    if mm:
        raise SourceIdentityError(mm, root)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=None, help="the requested tree (default: this file's tree; $VERITOR_REPO must agree)")
    ap.add_argument("--sha", default=None, help="the requested commit; every package's git_sha must equal it")
    ap.add_argument("--out", default=None, help="write the manifest record here (also on failure)")
    ap.add_argument("--stage", default="precheck")
    ap.add_argument("--print", action="store_true", help="print the record to stdout")
    a = ap.parse_args(argv)
    try:
        doc = precheck(a.stage, a.tree, sha=a.sha, out=a.out)
    except SourceIdentityError as e:
        print(str(e), file=sys.stderr, flush=True)
        if a.out:
            print(f"[source_identity] record: {a.out}", file=sys.stderr, flush=True)
        return 3
    if a.print:
        print(json.dumps(doc, indent=1, default=str))
    pk = doc["packages"]
    print("SOURCE-IDENTITY-OK " + " ".join(f"{k}={('' if v.get('git_sha') else '-')}{(v.get('git_sha') or '')[:9]}" for k, v in pk.items())
          + f" tree={doc['requested_root']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
