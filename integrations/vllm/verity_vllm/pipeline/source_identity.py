"""Verity source identity + precheck: the job runs the tree it names, and a mixed tree is refused by construction.

    verity-vllm source-identity [--tree ROOT] [--sha SHA] [--out FILE] [--stage NAME]
        exit 0  every important package (verity, verity_vllm, hidden_gpu, research) resolves from ROOT
        exit 3  PRECHECK_FAIL_SOURCE_IDENTITY -- the first line of stderr names the class, the following lines each mismatch

The tree of record is the VERITY REPO ROOT: the checkout (or the shipped `git archive`, <root>/src/<sha>/) holding `.`,
`tools/research/src` and `integrations/vllm`.  It defaults to THIS FILE'S tree (never an environment default): the runner that imports this
module IS the tree that must run.  `--tree` accepts the repo root or the integration root (`integrations/vllm`, what row_pod.sh passes) and
normalises to the repo root.  `VERITY_TREE`, when set, must name the same tree (either form) -- a caller that exports another tree is the
mixed-tree case and is refused by name.  `--out` writes the manifest record (schema research-infra/source-identity/v1); the same record goes
into versions.json (commit_delta) and the attempt's job.json (tele init --source-identity).

A shipped tree has no .git and no EXPORT.json, so its sha comes from `--sha` or, failing that, from $RESEARCH_SOURCE_SHA (set by the
`research` runner from request.json's tree_sha; see research.result.ENV_SOURCE_SHA).  Packages that resolve from the requested tree but
whose tree carries no sha of its own are recorded with that sha (git_sha_source names where it came from).

In-process: `precheck(stage)` is called by the Commit (commit_delta.main, before the engine is built -> before any inference), and
`assert_loaded_module_is_ours` after an import a sys.path insertion made possible (the module that actually loaded must be this tree's).
Both raise research.telemetry.source_identity.SourceIdentityError, whose message starts with PRECHECK_FAIL_SOURCE_IDENTITY.

Import layout: the generic half is `research.telemetry.source_identity` and must be importable as such -- on a pod row_pod.sh exports
PYTHONPATH=<integration root>:<repo>:<repo>/tools/research/src.  No sys.path is manipulated here: a `research` that
cannot be imported is a broken environment and is reported as one, not repaired silently (that repair is how mixed trees happen).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import ROOT as INTEGRATION
from verity_vllm.config import option

try:
    from research.pythonpath import package_dirs as _package_dirs
    from research.telemetry import source_identity as _sid
except ImportError as _e:  # pragma: no cover - environment defect, reported by name
    raise ImportError("verity_vllm.pipeline.source_identity needs the `research` package of the tree of record on sys.path "
                      "(PYTHONPATH must include <repo>/tools/research/src; verity-vllm row exports it); nothing is appended here on purpose: "
                      f"{_e}") from _e

INTEGRATION_REL = Path("integrations") / "vllm"
# INTEGRATION is <repo>/integrations/vllm; the repo root is derived from the package location, never from env
TREE = INTEGRATION.parent.parent
ROOT_MARKERS = ("pyproject.toml", "uv.lock", "integrations/vllm/verity_vllm/__init__.py")

FAIL_CLASS = _sid.FAIL_CLASS
SourceIdentityError = _sid.SourceIdentityError
tree_identity = _sid.tree_identity

# hidden_gpu resolves from <integration>/out/gen/r9/cmt-hidden/src (materialised by pod_hidden_gpu.sh from the shipped copy
# verity_vllm/commit/committer/hidden_gpu_src) or from that shipped copy; both are candidates, the r9 path first
HIDDEN_GPU_CANDIDATES = ("integrations/vllm/out/gen/r9/cmt-hidden/src/hidden_gpu.py", "integrations/vllm/verity_vllm/commit/committer/hidden_gpu_src/hidden_gpu.py")
# package name -> (import name, the sys.path entries it must resolve from, relative to the repo root); core's are the requested tree's
# workspace roots that ship it (research.pythonpath.package_dirs)
PACKAGE_ROOTS = {
    "verity_vllm": ("verity_vllm", ("integrations/vllm",)),
    "research": ("research", ("tools/research/src",)),                 # the runner/telemetry package of the tree of record
}
PACKAGES = ("verity", "verity_vllm", "hidden_gpu", "research")
ENV_SOURCE_SHA = "RESEARCH_SOURCE_SHA"


def _is_repo_root(p: Path) -> bool:
    return all((p / m).exists() for m in ROOT_MARKERS)


def normalise_tree(tree: str | os.PathLike[str] | None) -> str:
    """Realpath of the repo root named by `tree`: the repo root itself, or its integration root (`<repo>/integrations/vllm`, what
    row_pod.sh passes as $TREE_HERE).  Anything else is returned as given (realpath'd) and will fail the check by name."""
    root = Path(os.path.realpath(os.fspath(tree) if tree else str(TREE)))
    if not _is_repo_root(root) and root.name == INTEGRATION_REL.name and root.parent.name == INTEGRATION_REL.parent.name and _is_repo_root(root.parents[1]):
        return str(root.parents[1])
    return str(root)


def code_identity(root: str | os.PathLike[str]) -> dict[str, Any]:
    """The code the integration executes, by content: the canonical sha256 of the manifest {repo path: sha256} of the repository at
    `root` over `research_tools.CLOSURE` (the integration plus Verity core), the files the store records as the research Tools' closure.
    The one code identity: the hot Commit worker's key and the TP Commit's tree of record.  Raises FileNotFoundError when `root` holds
    no file under one of the closure's directories, the integration or a directory of core (not a repository root: a manifest
    missing one would be a key that ignores it)."""
    from verity_vllm.pipeline.research_tools import VLLM_BUILD

    files = VLLM_BUILD.closure_manifest(root)
    for part in (g.removesuffix("**") for g in VLLM_BUILD.closure):
        if not any(p.startswith(part) for p in files):
            raise FileNotFoundError(f"code identity: no {part} file under {os.fspath(root)!r} (not a repository root)")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(canonical).hexdigest(), "files": len(files)}


def requested_tree(tree: str | os.PathLike[str] | None = None) -> str:
    """The tree this job must run from: the explicit argument, else THIS file's tree.  $VERITY_TREE is checked against it, never used
    as the default -- a differing value is the mixed-tree case and fails by name."""
    root = normalise_tree(tree)
    env = os.environ.get("VERITY_TREE")
    if env and normalise_tree(env) != root:
        raise SourceIdentityError([{"package": "VERITY_TREE", "resolved_path": os.path.realpath(env), "repo_root": os.path.realpath(env),
                                    "git_sha": _sid.tree_identity(os.path.realpath(env)).get("git_sha"),
                                    "why": f"$VERITY_TREE names another tree than the code that runs ({root})"}], root)
    return root


def _fill_sha(rec: dict[str, Any], root: str, sha: str | None, source: str | None) -> None:
    """A package of the requested tree whose tree has no sha of its own (a git archive: no .git, no EXPORT.json) takes the declared one."""
    if sha and rec.get("git_sha") is None and rec.get("repo_root") and os.path.realpath(rec["repo_root"]) == root:
        rec["git_sha"] = sha
        rec["git_sha_source"] = source


def declared_sha(sha: str | None = None) -> tuple[str | None, str | None]:
    """(sha, where it came from): `--sha` first, then $RESEARCH_SOURCE_SHA (the research runner's declaration of the shipped tree)."""
    if sha:
        return sha, "--sha"
    env = os.environ.get(ENV_SOURCE_SHA)
    if env:
        return env, ENV_SOURCE_SHA
    return None, None


def resolve(tree: str | os.PathLike[str] | None = None, *, sha: str | None = None) -> dict[str, dict[str, Any]]:
    """Identity of every package in PACKAGES as the CURRENT interpreter resolves it (sys.modules first, then find_spec on sys.path).
    `sha` is the caller's explicit declaration (--sha); absent, $RESEARCH_SOURCE_SHA is the fallback."""
    root = normalise_tree(tree)
    try:
        core = tuple(os.path.dirname(d) for d in _package_dirs(root, "verity"))
    except LookupError:
        core = ()                   # not a workspace: no entry to expect (the verdict is check()'s, on repo_root)
    roots = {"verity": ("verity", core), **PACKAGE_ROOTS}           # core: verity.ml.circuits, verity.ml.circuits.silicon, verity_sp1.proofs
    sha, sha_src = declared_sha(sha)
    ids: dict[str, dict[str, Any]] = {}
    for name in PACKAGES:
        if name == "hidden_gpu":
            rec = _sid.resolve_file("hidden_gpu", [os.path.join(root, c) for c in HIDDEN_GPU_CANDIDATES], ROOT_MARKERS)
        else:
            import_name, rels = roots[name]
            rec = _sid.resolve_module(import_name, ROOT_MARKERS)
            rec["package"] = name
            rec["import_name"] = import_name
            rec["expected_sys_path_entry"] = os.pathsep.join(os.path.normpath(os.path.join(root, rel)) for rel in rels)
            if rec.get("resolved_path"):
                # <entry>/<import_name>/__init__.py, or each portion <entry>/<import_name> of a namespace -> the sys.path entries that resolved it
                portions = [p["path"] for p in rec.get("portions") or ()]
                rec["sys_path_entry"] = (os.pathsep.join(str(Path(p).parent) for p in portions) if portions
                                         else str(Path(rec["resolved_path"]).parent.parent))
        _fill_sha(rec, root, sha, sha_src)
        ids[name] = rec
    return ids


def manifest(tree: str | os.PathLike[str] | None = None, *, sha: str | None = None, strict_env: bool = True) -> dict[str, Any]:
    """The record + verdict (no raise).  `ok` False carries fail_class PRECHECK_FAIL_SOURCE_IDENTITY and the mismatches."""
    try:
        root = requested_tree(tree) if strict_env else normalise_tree(tree)
        env_mm: list[dict[str, Any]] = []
    except SourceIdentityError as e:
        root, env_mm = e.requested_root, e.mismatches
    ids = resolve(root, sha=sha)                  # resolve() applies the same --sha / $RESEARCH_SOURCE_SHA precedence
    sha, sha_src = declared_sha(sha)
    mm = env_mm + _sid.check(ids, root, requested_sha=sha, optional=("hidden_gpu",))
    doc = _sid.document(ids, root, mm)
    if doc.get("requested") is not None and sha and doc["requested"].get("git_sha") is None:
        doc["requested"]["git_sha"] = sha            # the requested tree's own sha, when the tree cannot say it (git archive)
        doc["requested"]["git_sha_source"] = sha_src
    doc["VERITY_TREE"] = os.environ.get("VERITY_TREE")
    doc["PYTHONPATH"] = os.environ.get("PYTHONPATH")
    doc[ENV_SOURCE_SHA] = os.environ.get(ENV_SOURCE_SHA)
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
    """After an import that a sys.path insertion made possible (`import hidden_gpu` from the r9 directory), the module that LOADED must live in
    the requested tree -- the by-construction guard at the import site itself."""
    root = normalise_tree(tree)
    f = getattr(mod, "__file__", None)
    rec = {"package": name, "resolved_path": os.path.realpath(f) if f else None, "repo_root": _sid.repo_root_of(f, ROOT_MARKERS) if f else None, "loaded": True}
    rec.update(_sid.tree_identity(rec["repo_root"]))
    mm = _sid.check({name: rec}, root)
    if mm:
        raise SourceIdentityError(mm, root)
    return rec


@dataclass(frozen=True, kw_only=True)
class Options:
    tree: str | None = option("--tree", default=None,
                              help="the requested tree: the Verity repo root or its integrations/vllm (default: this file's tree; $VERITY_TREE must agree)")
    sha: str | None = option("--sha", default=None,
                             help="the requested commit; every package's git_sha must equal it (default: $RESEARCH_SOURCE_SHA when set)")
    out: str | None = option("--out", default=None, help="write the manifest record here (also on failure)")
    stage: str = option("--stage", default="precheck")
    print: bool = option("--print", default=False, action="store_true", help="print the record to stdout")


def main(a: Options) -> int:
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
