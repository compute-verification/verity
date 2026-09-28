"""Tool declarations (store README §6.1): a small Python object around an existing command.

    TOOL = Tool(name="tc_probe", version="1", command=["python", "benchmarks/tc_probe/tc_probe.py"],
                closure=["benchmarks/tc_probe/**", ...], key_params=[...], key_conditions=["hw.family"],
                param_artifacts=["replay_golden"], caches=["nvcc"], produces=["tc-evidence/v1", "run-files/v1"],
                parse=None, conditions=None)

closure_manifest(root)   repo-relative globs -> {path: sha256} of every matching file (git-tracked + untracked-not-ignored when
                         `root` is a git tree; a plain walk honouring the root .gitignore otherwise, e.g. a shipped `git archive`)
resolve_params(argv, root)  -> (params, key_params_resolved): `params` = parse(argv) (or a generic --flag parse) + `argv`;
                         key params that name paths (`param_artifacts`) are replaced by CONTENT identity: an `art:` id when the
                         path is a fetched artifact (a `.art-id` marker beside it), else `sha256:<hex>` of the file or of the
                         canonical tree listing.  The pathname never enters the derivation.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ART_MARKER = ".art-id"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_listing(root: Path) -> list[dict[str, Any]]:
    """The §2 tree payload listing: [{path, sha256, bytes}] sorted by path, files only, marker excluded."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name == ART_MARKER:
                continue
            p = Path(dirpath) / name
            if not p.is_file():
                continue
            out.append({"path": p.relative_to(root).as_posix(), "sha256": _sha256_file(p), "bytes": p.stat().st_size})
    out.sort(key=lambda f: f["path"])
    return out


def tree_sha256(root: Path) -> str:
    from .canon import sha256

    return sha256(tree_listing(root))


def _read_marker(d: Path) -> str | None:
    try:
        line = (d / ART_MARKER).read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    return line if line.startswith("art:") else None


def content_identity(value: Any, source_root: Path | None = None) -> Any:
    """Path-valued param -> `art:<id>` (fetched artifact) or `sha256:<hex>` (file, or canonical tree listing).  None when the
    path does not exist (the derivation then says "missing", never the pathname).  Non-string values pass through."""
    if not isinstance(value, str) or not value:
        return value
    p = Path(value).expanduser()
    if not p.is_absolute():
        cands = [Path(source_root) / p] if source_root else []
        cands.append(Path.cwd() / p)
        p = next((c for c in cands if c.exists()), cands[0])
    if p.is_dir():
        art = _read_marker(p)
        return art if art else "sha256:" + tree_sha256(p)
    if p.is_file():
        art = _read_marker(p.parent)
        if art and [e.name for e in p.parent.iterdir() if e.name != ART_MARKER] == [p.name]:
            return art                      # a fetched single-file artifact
        return "sha256:" + _sha256_file(p)
    return None


# ---- globs and the fallback ignore rules ---------------------------------------------------------------------------------------------------

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


def _ignored(rel: str, is_dir: bool, rules: list[tuple[re.Pattern[str], bool]]) -> bool:
    return any(rx.match(rel) and (is_dir or not dir_only) for rx, dir_only in rules)


def list_source_files(root: Path) -> list[str]:
    """Relative posix paths of the files that make up the tree at `root`."""
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
        keep = []
        for d in sorted(dirnames):
            rel = d if rel_dir == "." else f"{rel_dir}/{d}"
            if not _ignored(rel, True, rules):
                keep.append(d)
        dirnames[:] = keep
        for f in sorted(filenames):
            rel = f if rel_dir == "." else f"{rel_dir}/{f}"
            if not _ignored(rel, False, rules) and (root / rel).is_file():
                out.append(rel)
    return sorted(out)


def _in_git_tree(root: Path) -> bool:
    if shutil.which("git") is None:
        return False
    try:
        r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and r.stdout.strip() == b"true"


# ---- the generic argv parse ----------------------------------------------------------------------------------------------------------------

_NUM = re.compile(r"^-\d")


def parse_argv(argv: list[str], command: list[str] | None = None) -> dict[str, Any]:
    """`--k v` / `--k=v` / `--flag` -> {k: v|True}; dashes become underscores; positionals under `_positional`.  Values stay
    strings (a Tool wanting types supplies `parse`).  The leading `command` (matched by basename) or argv[0] is skipped."""
    toks = list(argv)
    n = 0
    if command and len(toks) >= len(command) and all(Path(a).name == Path(b).name for a, b in zip(toks, command)):
        n = len(command)
    elif toks:
        n = 1
    out: dict[str, Any] = {}
    pos: list[str] = []
    i = n
    while i < len(toks):
        t = toks[i]
        if t == "--":
            pos.extend(toks[i + 1:])
            break
        if t.startswith("-") and not _NUM.match(t):
            key, sep, val = t.lstrip("-").partition("=")
            key = key.replace("-", "_")
            if sep:
                out[key] = val
            elif i + 1 < len(toks) and (not toks[i + 1].startswith("-") or _NUM.match(toks[i + 1])):
                out[key] = toks[i + 1]
                i += 1
            else:
                out[key] = True
        else:
            pos.append(t)
        i += 1
    if pos:
        out["_positional"] = pos
    return out


# ---- the declaration -----------------------------------------------------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    version: str | None = None
    command: list[str] = field(default_factory=list)
    closure: list[str] = field(default_factory=list)          # repo-relative globs
    key_params: list[str] = field(default_factory=list)
    key_conditions: list[str] = field(default_factory=list)
    param_artifacts: list[str] = field(default_factory=list)  # path-valued params: identity = content, never pathname
    caches: list[str] = field(default_factory=list)           # names for --scratch NAME
    produces: list[str] = field(default_factory=list)
    parse: Callable[[list[str]], dict] | None = None
    conditions: Callable[[dict], dict] | None = None          # extra keys, recorded under tool.<name>.<key>
    spec: str | None = None                                   # `dotted.module:NAME` this declaration was loaded from (set by the registry)
    #: `research merge`: result steps (``result.json`` ``steps[].step``) that must have passed when a merge changes a path under
    #: one of their prefixes, less the paths a `!` pattern names (`merge.rule_touches`); an attempt where such a step was skipped
    #: does not admit that merge
    merge_requires: dict[str, list[str]] = field(default_factory=dict)
    #: `research run --on`: an argv run on the machine in the shipped tree before the run is claimed (`remote.run_preflight`);
    #: exit 0 admits the run, anything else refuses it with the preflight's stderr as the reason
    preflight: list[str] = field(default_factory=list)
    #: `research merge --push`: an argv started in the target's checkout, detached, with the landed commit appended, once a merge
    #: the tool gated is on the remote (`merge.after_land`); it outlives the merge and never changes its result
    after_land: list[str] = field(default_factory=list)

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}

    def closure_manifest(self, source_root: Path | str) -> dict[str, str]:
        root = Path(source_root)
        if not self.closure:
            return {}
        pats = [glob_re(p) for p in self.closure]
        out = {}
        for rel in list_source_files(root):
            if any(rx.match(rel) for rx in pats):
                try:
                    out[rel] = _sha256_file(root / rel)
                except OSError:
                    continue
        return dict(sorted(out.items()))

    def resolve_params(self, argv: list[str], source_root: Path | str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        argv = list(argv)
        params = dict(self.parse(argv)) if self.parse else parse_argv(argv, self.command)
        params["argv"] = argv
        return params, self.resolve_key_params(params, source_root)

    def resolve_key_params(self, params: dict[str, Any], source_root: Path | str | None = None) -> dict[str, Any]:
        root = Path(source_root) if source_root else None
        key: dict[str, Any] = {}
        for k in self.key_params:
            if k not in params:
                continue
            v = params[k]
            key[k] = content_identity(v, root) if k in self.param_artifacts else v
        return key

    def tool_conditions(self, env: dict[str, Any]) -> dict[str, Any]:
        if self.conditions is None:
            return {}
        try:
            extra = self.conditions(env) or {}
        except Exception:  # noqa: BLE001  a tool's probe must not stop the attempt
            return {}
        return {f"tool.{self.name}.{k}": v for k, v in extra.items()}
