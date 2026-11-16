"""Source identity of the packages a job imports (stdlib only)."""
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
    except OSError as e:
        print(f"source identity: {p} not hashed ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return None


def resolve_module(name: str, root_markers: Iterable[str] = (), path: list[str] | None = None) -> dict[str, Any]:
    """find_spec only: the module is not executed."""
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
