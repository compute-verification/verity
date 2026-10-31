"""Verity source identity + precheck: the job runs the tree it names, and a mixed tree is refused by construction.

    verity-vllm source-identity [--tree ROOT] [--sha SHA] [--out FILE] [--stage NAME]
        exit 0  every important package (verity, verity_vllm, hidden_gpu) resolves from ROOT
        exit 3  PRECHECK_FAIL_SOURCE_IDENTITY -- the first line of stderr names the class, the following lines each mismatch

For every package the record holds

    resolved_path   the file/directory Python resolves the name to (importlib.util.find_spec, not an import: nothing runs)
    repo_root       the source tree that file lives in (the nearest ancestor holding EXPORT.json or .git, else the ancestor holding
                    every ROOT_MARKERS entry)
    git_sha         that tree's commit: EXPORT.json `commit` (a shipped tree; `dirty_patch_sha256` beside it) or `git rev-parse HEAD`
    file_sha256     for a single-file module (hidden_gpu): the source bytes, the build identity of its JIT extension's driver

and `check` compares every repo_root with the requested tree.

The tree of record is the VERITY REPO ROOT: the checkout (or the shipped `git archive`, <root>/src/<sha>/) holding `.` and
`integrations/vllm`.  It defaults to THIS FILE'S tree (never an environment default): the runner that imports this
module IS the tree that must run.  `--tree` accepts the repo root or the integration root (`integrations/vllm`, what row_pod.sh passes) and
normalises to the repo root.  `VERITY_TREE`, when set, must name the same tree (either form) -- a caller that exports another tree is the
mixed-tree case and is refused by name.  `--out` writes the manifest record (schema research-infra/source-identity/v1); the same record goes
into versions.json (commit_delta) and the attempt's job.json (tele init --source-identity).

A shipped tree has no .git and no EXPORT.json, so its sha comes from `--sha` or, failing that, from $RESEARCH_SOURCE_SHA (set by the
`research` runner from request.json's tree_sha; see research.result.ENV_SOURCE_SHA).  Packages that resolve from the requested tree but
whose tree carries no sha of its own are recorded with that sha (git_sha_source names where it came from).

In-process: `precheck(stage)` is called by the Commit (commit_delta.main, before the engine is built -> before any inference), and
`assert_loaded_module_is_ours` after an import a sys.path insertion made possible (the module that actually loaded must be this tree's).
Both raise SourceIdentityError, whose message starts with PRECHECK_FAIL_SOURCE_IDENTITY.  No sys.path is manipulated here: a package
that cannot be imported is a broken environment and is reported as one, not repaired silently (that repair is how mixed trees happen).

`code_identity` is the other identity: the code the integration executes, by content, over `closure()`.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from verity_vllm.config import ROOT as INTEGRATION
from verity_vllm.config import option
from verity_vllm.records.source_identity import SCHEMA

INTEGRATION_REL = Path("integrations") / "vllm"
# INTEGRATION is <repo>/integrations/vllm; the repo root is derived from the package location, never from env
TREE = INTEGRATION.parent.parent
ROOT_MARKERS = ("pyproject.toml", "uv.lock", "integrations/vllm/verity_vllm/__init__.py")

FAIL_CLASS = "PRECHECK_FAIL_SOURCE_IDENTITY"


def core_dirs(tree: str | os.PathLike[str]) -> list[str]:
    """Where `tree` ships Verity core, relative to it: each package its root pyproject.toml builds whose import name is `verity`.
    Raises LookupError when it names none (not a Verity tree)."""
    try:
        doc = tomllib.loads((Path(tree) / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise LookupError(f"{os.fspath(tree)}: no readable pyproject.toml ({e})") from None
    pkgs = doc.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {}).get("packages", [])
    dirs = [p.rstrip("/") for p in pkgs if PurePosixPath(p.rstrip("/")).name == "verity"]
    if not dirs:
        raise LookupError(f"{os.fspath(tree)}/pyproject.toml builds no package `verity`")
    return dirs


def closure() -> list[str]:
    """The code the integration executes, as repository-relative globs: itself and Verity core (`core_dirs` of this file's tree),
    with the catalog, the kernels and the verifiers in it.  Read on call, so a copy of the package outside a tree still imports and
    its precheck names the mismatch."""
    return ["integrations/vllm/**", *(f"{d}/**" for d in core_dirs(TREE))]


class SourceIdentityError(RuntimeError):
    """Raised by precheck(): message starts with FAIL_CLASS and names every mismatching package."""

    def __init__(self, mismatches: list[dict[str, Any]], requested_root: str):
        self.mismatches = mismatches
        self.requested_root = requested_root
        lines = [f"{FAIL_CLASS}: {len(mismatches)} package(s) do not resolve from the requested tree {requested_root}"]
        for m in mismatches:
            lines.append(f"  {m['package']}: resolved_path={m.get('resolved_path')} repo_root={m.get('repo_root')} "
                         f"git_sha={m.get('git_sha')} -- {m['why']}")
        super().__init__("\n".join(lines))


def repo_root_of(path: str | os.PathLike[str], root_markers: Iterable[str] = ()) -> str | None:
    """Nearest ancestor of `path` that is a source tree: holds EXPORT.json (a shipped tree) or .git (a checkout/worktree); failing both,
    the nearest ancestor holding every entry of `root_markers` (relative paths such as 'verity_vllm/observe/__init__.py')."""
    p = Path(os.path.realpath(path))
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
        except (OSError, ValueError, AttributeError) as e:
            out["git_sha_source"] = f"EXPORT.json unreadable: {e}"
    if (Path(root) / ".git").exists():
        try:
            sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
            br = subprocess.run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
            out.update(git_sha=sha or None, git_sha_source="git", branch=br or None)
            return out
        except (OSError, subprocess.SubprocessError) as e:
            out["git_sha_source"] = f"git failed: {e}"
    if out["git_sha_source"] is None:
        out["git_sha_source"] = "neither EXPORT.json nor .git at the repo root"
    return out


def _sha256_file(p: str | os.PathLike[str]) -> str | None:
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
        except ValueError as e:     # a module in sys.modules whose __spec__ is None
            spec = None
            rec["reason"] = f"find_spec failed: {e}"
        if spec is None:
            rec["reason"] = rec["reason"] or "not importable on sys.path"
            return rec
        origin = spec.origin
        if (origin is None or origin == "namespace") and spec.submodule_search_locations:
            portions = [os.path.realpath(p) for p in spec.submodule_search_locations]
            origin = portions[0]
    if not origin or origin in ("built-in", "frozen"):
        rec["reason"] = f"no file origin ({origin})"
        return rec
    rp = os.path.realpath(origin)
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
        chosen = next((c for c in rec["candidates"] if os.path.isfile(c)), None)
    if chosen is None:
        rec["reason"] = "no candidate file exists"
        return rec
    rp = os.path.realpath(chosen)
    rec["resolved_path"] = rp
    root = repo_root_of(rp, root_markers)
    rec["repo_root"] = root
    rec.update(tree_identity(root))
    rec["file_sha256"] = _sha256_file(rp)
    return rec


def _same_tree(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and os.path.realpath(a) == os.path.realpath(b)


def check(identities: dict[str, dict[str, Any]], requested_root: str, *, requested_sha: str | None = None,
          optional: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Every identity's repo_root must be the requested tree (realpath-equal); when `requested_sha` is given, every recorded git_sha must
    equal it too.  Returns the mismatches (empty = pass).  A package in `optional` that is absent (resolved_path None) is not a mismatch --
    a package that resolves ELSEWHERE always is."""
    req = os.path.realpath(requested_root)
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


def document(identities: dict[str, dict[str, Any]], requested_root: str | None, mismatches: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The manifest record (job.json `source_identity` / versions.json `source_identity` / <attempt>/source_identity.json)."""
    req = os.path.realpath(requested_root) if requested_root else None
    return {
        "schema": SCHEMA,
        "requested_root": req,
        "requested": tree_identity(req) if req else None,
        "sys_path0": sys.path[0] if sys.path else None,
        "sys_executable": sys.executable,
        "packages": identities,
        "ok": (not mismatches) if mismatches is not None else None,
        "fail_class": FAIL_CLASS if mismatches else None,
        "mismatches": mismatches or [],
    }


# hidden_gpu resolves from <integration>/out/gen/r9/cmt-hidden/src (materialised by pod_hidden_gpu.sh from the shipped copy
# verity_vllm/commit/committer/hidden_gpu_src) or from that shipped copy; both are candidates, the r9 path first
HIDDEN_GPU_CANDIDATES = ("integrations/vllm/out/gen/r9/cmt-hidden/src/hidden_gpu.py", "integrations/vllm/verity_vllm/commit/committer/hidden_gpu_src/hidden_gpu.py")
# package name -> (import name, the sys.path entries it must resolve from, relative to the repo root); core's are the parents of the
# requested tree's `core_dirs`
PACKAGE_ROOTS = {
    "verity_vllm": ("verity_vllm", ("integrations/vllm",)),
}
PACKAGES = ("verity", "verity_vllm", "hidden_gpu")
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


def glob_re(pat: str) -> re.Pattern[str]:
    """Repo-relative glob -> regex: `**` crosses directories, `*` and `?` do not."""
    out, i = "", 0
    while i < len(pat):
        c = pat[i]
        if pat.startswith("**", i):
            if pat.startswith("**/", i):
                out += "(?:.*/)?"
                i += 3
                continue
            out += ".*"
            i += 2
            continue
        out += "[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c)
        i += 1
    return re.compile("^" + out + "$")


def _ignore_rules(root: Path) -> list[tuple[re.Pattern[str], bool]]:
    rules = [(re.compile(r"^(?:.*/)?\.git$"), True), (re.compile(r"^(?:.*/)?__pycache__$"), True)]
    try:
        lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    except OSError:
        return rules
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        dir_only = s.endswith("/")
        s = s.rstrip("/")
        if s.startswith("/"):
            s = s[1:]
        elif "/" not in s:
            s = "**/" + s
        rules.append((glob_re(s), dir_only))
    return rules


def _in_git_tree(root: Path) -> bool:
    if shutil.which("git") is None:
        return False
    try:
        r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and r.stdout.strip() == b"true"


def list_source_files(root: str | os.PathLike[str]) -> list[str]:
    """Relative posix paths of the files that make up the tree at `root`: git's tracked and untracked-not-ignored files in a git tree,
    else every file the root's .gitignore leaves."""
    root = Path(root)
    if _in_git_tree(root):
        try:
            r = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root, capture_output=True, timeout=60)
            if r.returncode == 0:
                return sorted({p for p in r.stdout.decode("utf-8", errors="replace").split("\0") if p and (root / p).is_file()})
        except (OSError, subprocess.TimeoutExpired):
            pass
    rules = _ignore_rules(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        dirnames[:] = [d for d in sorted(dirnames) if not any(
            rx.match(d if rel_dir == "." else f"{rel_dir}/{d}") for rx, _ in rules)]
        for f in sorted(filenames):
            rel = f if rel_dir == "." else f"{rel_dir}/{f}"
            if not any(rx.match(rel) and not dir_only for rx, dir_only in rules) and (root / rel).is_file():
                out.append(rel)
    return sorted(out)


def closure_manifest(root: str | os.PathLike[str]) -> dict[str, str]:
    """{repo path: sha256} of every file of the tree at `root` that `closure()` matches, sorted by path."""
    pats = [glob_re(p) for p in closure()]
    out = {}
    for rel in list_source_files(root):
        if any(rx.match(rel) for rx in pats) and (h := _sha256_file(Path(root) / rel)) is not None:
            out[rel] = h
    return dict(sorted(out.items()))


def code_identity(root: str | os.PathLike[str]) -> dict[str, Any]:
    """The code the integration executes, by content: the canonical sha256 of `closure_manifest(root)` (the integration plus Verity
    core), the files `research run` records as the vLLM row stages' closure.  The one code identity: the hot Commit worker's key and the
    TP Commit's tree of record.  Raises FileNotFoundError when `root` holds no file under one of the closure's directories, the
    integration or a directory of core (not a repository root: a manifest missing one would be a key that ignores it)."""
    files = closure_manifest(root)
    for part in (g.removesuffix("**") for g in closure()):
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
                                    "git_sha": tree_identity(os.path.realpath(env)).get("git_sha"),
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
        core = tuple(os.path.dirname(d) for d in core_dirs(root))
    except LookupError:
        core = ()                   # not a workspace: no entry to expect (the verdict is check()'s, on repo_root)
    roots = {"verity": ("verity", core), **PACKAGE_ROOTS}           # core: verity.ml.circuits, verity.ml.circuits.silicon, verity_sp1.proofs
    sha, sha_src = declared_sha(sha)
    ids: dict[str, dict[str, Any]] = {}
    for name in PACKAGES:
        if name == "hidden_gpu":
            rec = resolve_file("hidden_gpu", [os.path.join(root, c) for c in HIDDEN_GPU_CANDIDATES], ROOT_MARKERS)
        else:
            import_name, rels = roots[name]
            rec = resolve_module(import_name, ROOT_MARKERS)
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
    mm = env_mm + check(ids, root, requested_sha=sha, optional=("hidden_gpu",))
    doc = document(ids, root, mm)
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
    rec = {"package": name, "resolved_path": os.path.realpath(f) if f else None, "repo_root": repo_root_of(f, ROOT_MARKERS) if f else None, "loaded": True}
    rec.update(tree_identity(rec["repo_root"]))
    mm = check({name: rec}, root)
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
