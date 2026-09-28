"""A source tree's Python import roots, derived from its workspace declaration -- never a hand-written PYTHONPATH list.

Hand lists for shipped trees failed silently twice (a missing `packages/verity/src`: `verity` not importable -> one lane crashed, and a
night earlier an incomplete manifest looked like a result).  The tree already says what its packages are:

    <tree>/pyproject.toml   [tool.uv.workspace] members = [...] (globs allowed), exclude = [...]
    <member>/pyproject.toml the member's build config names where its import packages live:
                              hatch       [tool.hatch.build.targets.wheel] packages = ["src/research", "python/verity_sp1", "verity_vllm"]
                                          -> the parent of each package path (src/, python/, the member itself)
                              setuptools  [tool.setuptools] package-dir = {"" = "src"}; [tool.setuptools.packages.find] where = ["src"]
                            with none of these: <member>/src when it exists, else <member> (flat layout)

    <tree>/pyproject.toml   when it has a [build-system] (the workspace root is itself a package, as `verity` is after the layout
                            move): its hatch packages too, but a package directly at the root is reached through a shim directory
                            holding a symlink to it (`shim`), never the root itself: on PYTHONPATH the root would make every top-level
                            directory importable (`kernels`, `tools`, `benchmarks`, ... are PyPI names; transformers imports `kernels`)

`import_roots_dir(tree)` reads a directory; `import_roots_git(git_dir, rev)` reads a commit without checking it out (`research env
--rev`).  Roots are relative to the tree, in member order, de-duplicated; a member or
root that does not exist is reported in `missing`, never silently dropped.  `research env --source <tree>` prints the export line;
a `research run --on M --source <tree>` sets it for the runner and the workload (the user's `--env PYTHONPATH=...` is appended) and
records it in job.json run.remote.pythonpath."""
from __future__ import annotations

import fnmatch
import hashlib
import os
import posixpath
import secrets
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Callable

Reader = Callable[[str], "str | None"]


def _toml(read: Reader, path: str) -> dict:
    text = read(path)
    if text is None:
        return {}
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}


def _member_roots(read: Reader, is_dir: Callable[[str], bool], member: str) -> list[str]:
    doc = _toml(read, posixpath.join(member, "pyproject.toml"))
    tool = doc.get("tool") or {}
    if (tool.get("uv") or {}).get("package") is False:      # a virtual member: no import packages, nothing for PYTHONPATH
        return []
    pkgs = (((tool.get("hatch") or {}).get("build") or {}).get("targets") or {}).get("wheel", {}).get("packages")
    if pkgs:
        return [posixpath.normpath(posixpath.join(member, posixpath.dirname(p.rstrip("/")) or ".")) for p in pkgs]
    st = tool.get("setuptools") or {}
    if isinstance(st.get("package-dir"), dict) and "" in st["package-dir"]:
        return [posixpath.normpath(posixpath.join(member, st["package-dir"][""]))]
    where = ((st.get("packages") or {}).get("find") or {}).get("where") if isinstance(st.get("packages"), dict) else None
    if where:
        return [posixpath.normpath(posixpath.join(member, w)) for w in where]
    src = posixpath.join(member, "src")
    return [src if is_dir(src) else member]


def import_roots(read: Reader, is_dir: Callable[[str], bool], dirs: Callable[[], list[str]]) -> dict:
    """{"members": [...], "roots": [...], "missing": [...], "source": what declared them}.  `read(relpath)` -> text or None, `is_dir(relpath)`,
    `dirs()` -> every directory of the tree (relative; used only to expand member globs)."""
    top = _toml(read, "pyproject.toml")
    ws = ((top.get("tool") or {}).get("uv") or {}).get("workspace") or {}
    pats, excl = list(ws.get("members") or []), list(ws.get("exclude") or [])
    members: list[str] = []
    all_dirs: list[str] | None = None
    for pat in pats:
        if any(c in pat for c in "*?["):
            all_dirs = dirs() if all_dirs is None else all_dirs
            hits = sorted(d for d in all_dirs if d.count("/") == pat.count("/") and fnmatch.fnmatchcase(d, pat))   # `*` stays in one segment
        else:
            hits = [pat.rstrip("/")]
        for h in hits:
            if h not in members and not any(fnmatch.fnmatchcase(h, e) for e in excl):
                members.append(h)
    roots: list[str] = []
    missing: list[str] = []
    shim: list[str] = []
    if "build-system" in top and ((top.get("tool") or {}).get("uv") or {}).get("package") is not False:
        pkgs = (((top.get("tool") or {}).get("hatch") or {}).get("build") or {}).get("targets", {}).get("wheel", {}).get("packages")
        if not pkgs:
            missing.append(". (the workspace root is a package: name its packages in [tool.hatch.build.targets.wheel] packages)")
        for p in (posixpath.normpath(p.rstrip("/")) for p in pkgs or []):
            parent = posixpath.dirname(p)
            target, into = (p, shim) if parent in ("", ".") else (parent, roots)
            if not is_dir(target):
                missing.append(target)
            elif target not in into:
                into.append(target)
    for m in members:
        if not is_dir(m):
            missing.append(m)
            continue
        for r in _member_roots(read, is_dir, m):
            if not is_dir(r):
                missing.append(r)
            elif r not in roots:
                roots.append(r)
    return {"members": members, "roots": roots, "missing": missing, "shim": shim,
            "source": "pyproject.toml [tool.uv.workspace]" if pats else "none (no [tool.uv.workspace] members in the tree's pyproject.toml)"}


def import_roots_dir(tree: Path | str) -> dict:
    t = Path(tree)

    def read(rel: str) -> str | None:
        try:
            return (t / rel).read_text()
        except OSError:
            return None

    def dirs() -> list[str]:
        return [p.relative_to(t).as_posix() for p in t.rglob("*") if p.is_dir() and ".git" not in p.parts]

    return import_roots(read, lambda rel: (t / rel).is_dir(), dirs)


def import_roots_git(git_dir: Path | str, rev: str) -> dict:
    """The same derivation for commit `rev` of the repository at `git_dir`, read with `git show` / `git ls-tree` (no checkout)."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=git_dir, capture_output=True, text=True)

    listing = git("ls-tree", "-r", "-d", "--name-only", rev)
    if listing.returncode != 0:
        raise LookupError(f"git ls-tree {rev}: {listing.stderr.strip()}")
    tree_dirs = set(listing.stdout.split("\n")) - {""}

    def read(rel: str) -> str | None:
        r = git("show", f"{rev}:{rel}")
        return r.stdout if r.returncode == 0 else None

    return import_roots(read, lambda rel: rel in ("", ".") or rel in tree_dirs, lambda: sorted(tree_dirs))


#: where `shim_dir` keeps one shim per tree when the caller names no other place
SHIM_BASE = Path.home() / ".research" / "cache" / "pyshim"


def shim_dir(root: str | os.PathLike[str], packages: list[str], base: str | os.PathLike[str] | None = None) -> str:
    """A directory holding one symlink per root-level package (`verity -> <root>/verity`), keyed by the tree's resolved path under
    `base`: the import path to the workspace root's own packages that leaves the root's other top-level directories unimportable.
    Built aside and renamed in, so runs of one tree share it, and rebuilt when its links no longer match."""
    tree = Path(root).resolve()
    want = {posixpath.basename(p): str(tree / p) for p in packages}
    d = Path(base if base is not None else SHIM_BASE).expanduser() / hashlib.sha256(str(tree).encode()).hexdigest()[:16]

    def matches() -> bool:
        try:
            return sorted(os.listdir(d)) == sorted(want) and all(os.readlink(d / n) == t for n, t in want.items())
        except OSError:
            return False

    if matches():
        return str(d)
    tmp = d.with_name(f"{d.name}.tmp-{secrets.token_hex(4)}")
    tmp.mkdir(parents=True)
    for n, t in want.items():
        (tmp / n).symlink_to(t, target_is_directory=True)
    try:
        if d.exists() or d.is_symlink():
            old = d.with_name(f"{d.name}.old-{secrets.token_hex(4)}")
            os.rename(d, old)
            shutil.rmtree(old, ignore_errors=True)
        os.rename(tmp, d)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not matches():                                       # another run's rename won the race; anything else is a real failure
            raise
    return str(d)


def pythonpath(root: str, derived: dict, extra: str | None = None, shim_base: str | os.PathLike[str] | None = None) -> str:
    """`root`-absolute roots joined with ':', then `extra` (the user's own PYTHONPATH) appended; the tree's shim (`shim_dir`, under
    `shim_base`) first when the workspace root is itself a package."""
    parts = [shim_dir(root, derived["shim"], shim_base)] if derived.get("shim") else []
    parts += [posixpath.join(root, r) if r not in ("", ".") else root for r in derived["roots"]]
    if extra:
        parts += [p for p in extra.split(":") if p and p not in parts]
    return ":".join(parts)


def tree_pythonpath(tree: Path | str) -> str:
    """The PYTHONPATH a command run from a source tree needs: the tree's workspace import roots (what `research env --source` prints),
    then, in a tree whose root is not a package, the tree root itself (modules addressed from the top, `backends.direct...`).  The
    steward's render and the parity tool's render both use it, so a new workspace member reaches them without a hand-kept list."""
    root = str(Path(tree))
    derived = import_roots_dir(root)
    return pythonpath(root, derived, extra=None if derived["shim"] else root)

