"""Verity source identity + precheck (F-r17-tp-06 fixed by construction; BRIEF R18 Phase 0, adapted to the Verity tree in R19).

    python -m verity_vllm.harness.source_identity [--tree ROOT] [--sha SHA] [--out FILE] [--stage NAME]
        exit 0  every important package (verity, verity_vllm, hidden_gpu, research) resolves from ROOT
        exit 3  PRECHECK_FAIL_SOURCE_IDENTITY -- the first line of stderr names the class, the following lines each mismatch

The tree of record is the VERITY REPO ROOT: the checkout (or the shipped `git archive`, <root>/src/<sha>/) holding `packages/verity/src`,
`tools/research/src` and `integrations/vllm`.  It defaults to THIS FILE'S tree (never an environment default): the runner that imports this
module IS the tree that must run.  `--tree` accepts the repo root or the integration root (`integrations/vllm`, what row_pod.sh passes) and
normalises to the repo root.  `VERITOR_REPO`, when set, must name the same tree (either form) -- a caller that exports another tree is the
mixed-tree case and is refused by name.  `--out` writes the manifest record (schema research-infra/source-identity/v1); the same record goes
into versions.json (commit_delta) and the attempt's job.json (tele init --source-identity).

A shipped tree has no .git and no EXPORT.json, so its sha comes from `--sha` or, failing that, from $RESEARCH_SOURCE_SHA (set by the
`research` runner from request.json's tree_sha; see research.result.ENV_SOURCE_SHA).  Packages that resolve from the requested tree but
whose tree carries no sha of its own are recorded with that sha (git_sha_source names where it came from).

In-process: `precheck(stage)` is called by the Commit (commit_delta.main, before the engine is built -> before any inference) and by
native_host._gpu_ext_load after `import hidden_gpu` (the module that actually loaded must be this tree's).  Both raise
research.telemetry.source_identity.SourceIdentityError, whose message starts with PRECHECK_FAIL_SOURCE_IDENTITY.

Import layout: the generic half is `research.telemetry.source_identity` and must be importable as such -- on a pod row_pod.sh exports
PYTHONPATH=<integration root>:<repo>/packages/verity/src:<repo>/tools/research/src.  No sys.path is manipulated here: a `research` that
cannot be imported is a broken environment and is reported as one, not repaired silently (that repair is how mixed trees happen).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from research.telemetry import source_identity as _sid
except ImportError as _e:  # pragma: no cover - environment defect, reported by name
    raise ImportError("verity_vllm.harness.source_identity needs the `research` package of the tree of record on sys.path "
                      "(PYTHONPATH must include <repo>/tools/research/src; row_pod.sh exports it); nothing is appended here on purpose: "
                      f"{_e}") from _e

# this file: <repo>/integrations/vllm/verity_vllm/harness/source_identity.py -> the repo root is parents[4]
# (sweep -> verity_vllm -> integrations/vllm -> integrations -> repo); stated once, derived from __file__, never from env
TREE = Path(__file__).resolve().parents[4]
INTEGRATION_REL = Path("integrations") / "vllm"
ROOT_MARKERS = ("pyproject.toml", "packages/verity/src/verity/__init__.py", "integrations/vllm/verity_vllm/__init__.py")

FAIL_CLASS = _sid.FAIL_CLASS
SourceIdentityError = _sid.SourceIdentityError

# hidden_gpu is imported by native_host._gpu_ext_load from <integration>/out/gen/r9/cmt-hidden/src (materialised by pod_hidden_gpu.sh from
# the shipped copy verity_vllm/acquire/hidden_gpu_src); both are candidates, the r9 path first because that is what native_host inserts
HIDDEN_GPU_CANDIDATES = ("integrations/vllm/out/gen/r9/cmt-hidden/src/hidden_gpu.py", "integrations/vllm/verity_vllm/acquire/hidden_gpu_src/hidden_gpu.py")
# package name -> (import name, the sys.path entry it must resolve from, relative to the repo root)
PACKAGE_ROOTS = {
    "verity": ("verity", "packages/verity/src"),                       # core: verity.ir, verity.ml.tc, verity.verification
    "verity_vllm": ("verity_vllm", "integrations/vllm"),
    "research": ("research", "tools/research/src"),                    # the runner/telemetry package of the tree of record
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


def requested_tree(tree: str | os.PathLike[str] | None = None) -> str:
    """The tree this job must run from: the explicit argument, else THIS file's tree.  $VERITOR_REPO is checked against it, never used
    as the default -- a differing value is the mixed-tree case (F-r17-tp-06) and fails by name."""
    root = normalise_tree(tree)
    env = os.environ.get("VERITOR_REPO")
    if env and normalise_tree(env) != root:
        raise SourceIdentityError([{"package": "VERITOR_REPO", "resolved_path": os.path.realpath(env), "repo_root": os.path.realpath(env),
                                    "git_sha": _sid.tree_identity(os.path.realpath(env)).get("git_sha"),
                                    "why": f"$VERITOR_REPO names another tree than the code that runs ({root})"}], root)
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
    sha, sha_src = declared_sha(sha)
    ids: dict[str, dict[str, Any]] = {}
    for name in PACKAGES:
        if name == "hidden_gpu":
            rec = _sid.resolve_file("hidden_gpu", [os.path.join(root, c) for c in HIDDEN_GPU_CANDIDATES], ROOT_MARKERS)
        else:
            import_name, rel = PACKAGE_ROOTS[name]
            rec = _sid.resolve_module(import_name, ROOT_MARKERS)
            rec["package"] = name
            rec["import_name"] = import_name
            rec["expected_sys_path_entry"] = os.path.join(root, rel)
            if rec.get("resolved_path"):
                # <entry>/<import_name>/__init__.py -> the sys.path entry that resolved it
                rec["sys_path_entry"] = str(Path(rec["resolved_path"]).parent.parent)
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
    doc["VERITOR_REPO"] = os.environ.get("VERITOR_REPO")
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
    """After an import that a sys.path insertion made possible (native_host: `import hidden_gpu`), the module that LOADED must live in
    the requested tree -- the by-construction guard at the import site itself."""
    root = normalise_tree(tree)
    f = getattr(mod, "__file__", None)
    rec = {"package": name, "resolved_path": os.path.realpath(f) if f else None, "repo_root": _sid.repo_root_of(f, ROOT_MARKERS) if f else None, "loaded": True}
    rec.update(_sid.tree_identity(rec["repo_root"]))
    mm = _sid.check({name: rec}, root)
    if mm:
        raise SourceIdentityError(mm, root)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=None, help="the requested tree: the Verity repo root or its integrations/vllm (default: this file's tree; $VERITOR_REPO must agree)")
    ap.add_argument("--sha", default=None, help="the requested commit; every package's git_sha must equal it (default: $RESEARCH_SOURCE_SHA when set)")
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
