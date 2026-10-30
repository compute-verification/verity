"""Source identity of the packages a job imports -- the generic half (stdlib only; project-independent).

[R18 infra, F-r17-tp-06 / COORD M-1848 (2)] A Commit under --gpu-tree without the tree variable (now VERITY_TREE) imported `verity_ir` from the pod's DEFAULT tree
while `verity_vllm` ran the release and versions.json named the release commit (tp M-1830).  The fix by construction has three parts and
this module is the evidence half: for every important imported package record

    resolved_path   the file/directory Python actually resolves the name to (importlib.util.find_spec, NOT an import -- nothing runs)
    repo_root       the source tree that file lives in (the nearest ancestor holding EXPORT.json or .git, else the ancestor holding
                    every `root_markers` entry)
    git_sha         that tree's commit: EXPORT.json `commit` (a shipped tree; `dirty_patch_sha256` beside it) or `git rev-parse HEAD`
    file_sha256     for a single-file module (hidden_gpu): the source bytes = the build identity of its JIT extension's driver

plus `sys_path0` = sys.path[0] of the resolving interpreter, and `check(...)` compares every repo_root with the REQUESTED tree: a
mismatch is `PRECHECK_FAIL_SOURCE_IDENTITY` (the failure class the classifier reads from the precheck record), raised BEFORE any inference.

The project half (which names, which root markers, the CLI) lives with the project (Verity: verity_vllm/pipeline/source_identity.py).
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "research-infra/source-identity/v1"
FAIL_CLASS = "PRECHECK_FAIL_SOURCE_IDENTITY"


class SourceIdentityError(RuntimeError):
    """Raised by check(): message starts with FAIL_CLASS and names every mismatching package."""

    def __init__(self, mismatches: list[dict[str, Any]], requested_root: str):
        self.mismatches = mismatches
        self.requested_root = requested_root
        lines = [f"{FAIL_CLASS}: {len(mismatches)} package(s) do not resolve from the requested tree {requested_root}"]
        for m in mismatches:
            lines.append(f"  {m['package']}: resolved_path={m.get('resolved_path')} repo_root={m.get('repo_root')} "
                         f"git_sha={m.get('git_sha')} -- {m['why']}")
        super().__init__("\n".join(lines))


def _realpath(p: str | os.PathLike[str]) -> str:
    return os.path.realpath(os.fspath(p))


def repo_root_of(path: str | os.PathLike[str], root_markers: Iterable[str] = ()) -> str | None:
    """Nearest ancestor of `path` that is a source tree: holds EXPORT.json (a shipped tree) or .git (a checkout/worktree); failing both,
    the nearest ancestor holding every entry of `root_markers` (relative paths such as 'verity_vllm/observe/__init__.py')."""
    p = Path(_realpath(path))
    if p.is_file():
        p = p.parent
    markers = list(root_markers)
    fallback: str | None = None
    for anc in [p, *p.parents]:
        if (anc / "EXPORT.json").is_file() or (anc / ".git").exists():
            return str(anc)
        if fallback is None and markers and all((anc / m).exists() for m in markers):
            fallback = str(anc)
    return fallback


def tree_identity(root: str | os.PathLike[str] | None) -> dict[str, Any]:
    """{git_sha, git_sha_source, dirty_patch_sha256, branch} of a source tree: EXPORT.json first (the shipped record), then git."""
    out: dict[str, Any] = {"git_sha": None, "git_sha_source": None, "dirty_patch_sha256": None, "branch": None}
    if root is None:
        out["git_sha_source"] = "no repo root"
        return out
    exp = Path(root) / "EXPORT.json"
    if exp.is_file():
        try:
            d = json.loads(exp.read_text())
            out.update(git_sha=d.get("commit"), git_sha_source="EXPORT.json", dirty_patch_sha256=d.get("dirty_patch_sha256"), branch=d.get("branch"))
            return out
        except Exception as e:  # noqa: BLE001
            out["git_sha_source"] = f"EXPORT.json unreadable: {e}"
    if (Path(root) / ".git").exists():
        try:
            sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
            br = subprocess.run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
            out.update(git_sha=sha or None, git_sha_source="git", branch=br or None)
            return out
        except Exception as e:  # noqa: BLE001
            out["git_sha_source"] = f"git failed: {e}"
    if out["git_sha_source"] is None:
        out["git_sha_source"] = "neither EXPORT.json nor .git at the repo root"
    return out


def _sha256_file(p: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def resolve_module(name: str, root_markers: Iterable[str] = (), path: list[str] | None = None) -> dict[str, Any]:
    """Where `name` WOULD be imported from on `path` (default sys.path) -- find_spec only, the module is not executed.  A module already
    in sys.modules is reported from its __file__ (that IS what runs) with `loaded: true`.  A namespace package (PEP 420) is every
    directory it spans: `portions` lists each with its repo_root, `resolved_path` is the first, and `repo_root` is their one tree, or
    None when they come from more than one."""
    rec: dict[str, Any] = {"package": name, "resolved_path": None, "repo_root": None, "loaded": False, "reason": None}
    origin: str | None = None
    portions: list[str] = []
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__file__", None):
        origin = mod.__file__
        rec["loaded"] = True
    else:
        try:
            spec = importlib.util.find_spec(name) if path is None else importlib.machinery.PathFinder.find_spec(name, path)
        except (ImportError, ValueError) as e:
            spec = None
            rec["reason"] = f"find_spec failed: {e}"
        if spec is None:
            rec["reason"] = rec["reason"] or "not importable on sys.path"
            return rec
        origin = spec.origin
        if (origin is None or origin == "namespace") and spec.submodule_search_locations:
            portions = [_realpath(p) for p in spec.submodule_search_locations]
            origin = portions[0]
    if not origin or origin in ("built-in", "frozen"):
        rec["reason"] = f"no file origin ({origin})"
        return rec
    rp = _realpath(origin)
    rec["resolved_path"] = rp
    root = repo_root_of(rp, root_markers)
    if portions:
        rec["portions"] = [{"path": p, "repo_root": repo_root_of(p, root_markers)} for p in portions]
        if len({p["repo_root"] for p in rec["portions"]}) > 1:
            root = None
            rec["reason"] = "namespace portions from more than one tree"
    rec["repo_root"] = root
    rec.update(tree_identity(root))
    if os.path.isfile(rp) and not os.path.basename(rp) == "__init__.py":
        rec["file_sha256"] = _sha256_file(rp)
    return rec


def resolve_file(name: str, candidates: Iterable[str], root_markers: Iterable[str] = ()) -> dict[str, Any]:
    """Identity of a single-file module that is NOT on sys.path until its importer inserts it (hidden_gpu): the first existing candidate
    path is the source of record; file_sha256 = the build identity of what its importer will JIT-compile from."""
    rec: dict[str, Any] = {"package": name, "resolved_path": None, "repo_root": None, "loaded": False, "reason": None, "candidates": [str(c) for c in candidates]}
    mod = sys.modules.get(name)
    chosen: str | None = None
    if mod is not None and getattr(mod, "__file__", None):
        chosen = mod.__file__
        rec["loaded"] = True
    else:
        for c in rec["candidates"]:
            if os.path.isfile(c):
                chosen = c
                break
    if chosen is None:
        rec["reason"] = "no candidate file exists"
        return rec
    rp = _realpath(chosen)
    rec["resolved_path"] = rp
    root = repo_root_of(rp, root_markers)
    rec["repo_root"] = root
    rec.update(tree_identity(root))
    rec["file_sha256"] = _sha256_file(rp)
    return rec


def _same_tree(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and _realpath(a) == _realpath(b)


def check(identities: dict[str, dict[str, Any]], requested_root: str, *, requested_sha: str | None = None,
          optional: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Every identity's repo_root must be the requested tree (realpath-equal); when `requested_sha` is given, every recorded git_sha must
    equal it too.  Returns the mismatches (empty = pass).  A package in `optional` that is absent (resolved_path None) is not a mismatch --
    a package that resolves ELSEWHERE always is."""
    req = _realpath(requested_root)
    opt = set(optional)
    out: list[dict[str, Any]] = []
    for name, rec in identities.items():
        if rec.get("resolved_path") is None:
            if name in opt:
                continue
            out.append({**rec, "why": f"not resolvable ({rec.get('reason')})"})
            continue
        foreign = [p["path"] for p in rec.get("portions") or () if not _same_tree(p["repo_root"], req)]
        if foreign:
            out.append({**rec, "why": f"namespace portion(s) outside the requested tree {req}: {', '.join(foreign)}"})
            continue
        if not _same_tree(rec.get("repo_root"), req):
            out.append({**rec, "why": f"repo_root != requested tree {req}"})
            continue
        if requested_sha and rec.get("git_sha") and rec["git_sha"] != requested_sha:
            out.append({**rec, "why": f"git_sha != requested {requested_sha}"})
    return out


def assert_identities(identities: dict[str, dict[str, Any]], requested_root: str, **kw: Any) -> None:
    mm = check(identities, requested_root, **kw)
    if mm:
        raise SourceIdentityError(mm, _realpath(requested_root))


def document(identities: dict[str, dict[str, Any]], requested_root: str | None, mismatches: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The manifest record (job.json `source_identity` / versions.json `source_identity` / <attempt>/source_identity.json)."""
    req = _realpath(requested_root) if requested_root else None
    return {
        "schema": SCHEMA,
        "requested_root": req,
        "requested": tree_identity(req) if req else None,
        "sys_path0": sys.path[0] if sys.path else None,
        "sys_executable": sys.executable,
        "cwd": os.getcwd(),
        "packages": identities,
        "ok": (not mismatches) if mismatches is not None else None,
        "fail_class": FAIL_CLASS if mismatches else None,
        "mismatches": mismatches or [],
    }
