"""The private files no run record, no `research data put` and no `research fetch --all` copies: by name (the tree's one list,
TREE_NAMES, in the tree `research` runs from: the floor), and in code what a name can't say: where a file lies, what lies beside it,
or (`contains`) its text.  A lane adds a name to TREE_NAMES, or a rule to PLACES, in a reviewed change; a bare name goes in only
where no public file has it.  For one run, custody adds the names its statement (`private-paths/v1` at PATHS_FILE, which the lanes'
launchers write) and its shipped tree's TREE_NAMES; nothing removes a rule at run time.  Custody lists what it refused
(store.custody); a put refuses the whole payload."""
from __future__ import annotations

import fnmatch
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

#: `contains` reads only a file of its suffix at most this big; any other file is never opened
SCAN_MAX_BYTES = 4 << 20
#: custody's listing of the private files it refused (REFUSED_SCHEMA: `refused`, [{path, rule, owner}]), in the run dir and its record
REFUSED_FILE = "custody-refused.json"
REFUSED_SCHEMA = "research/custody-refused/v1"
#: the tree's one list of private names, by its path from a tree's root: {"schema": NAMES_SCHEMA, "kept_apart": [prefixes of a
#: directory's or file's name], "private": [globs on a file's own name], "why": {entry: [owner, reason]}}, read with the meaning
#: its own matcher (`private_names.py` beside it) gives it, which this module can't import
TREE_NAMES = "adapters/vllm/verity_vllm/pipeline/private_names.json"
NAMES_SCHEMA = "verity/private-names/v1"
#: the roots the floor's TREE_NAMES is looked for under: the tree this file is in, laid out as the repository (a checkout, or a
#: deployed copy carrying the list at its path), then a tool snapshot's directory (remote.tool_tarball puts the list there)
FLOOR_ROOTS = tuple(Path(__file__).resolve().parents[i] for i in (5, 2) if i < len(Path(__file__).resolve().parents))
#: the run's statement at its run dir's root: {"schema": PATHS_SCHEMA, "private": [absolute directories outside the run dir],
#: "names": [globs no file or directory of the run's may have; a trailing `/` names a directory]}
PATHS_FILE = "private-paths.json"
PATHS_SCHEMA = "private-paths/v1"


@dataclass(frozen=True)
class Rule:
    what: str                                           # the rule as custody prints it and records it
    test: Callable[[PurePosixPath, Path], bool]         # (the path's components, the file on disk) -> private
    reason: str = ""
    owner: str = ""                                     # a lane's handle; empty for a rule a run's statement or tree adds
    reads: bool = False                                 # opens the file (`contains`), never through a link custody doesn't follow


def name(glob: str, reason: str, owner: str) -> Rule:
    """A file whose own name matches `glob`."""
    return Rule(f"name {glob}", lambda p, _: fnmatch.fnmatchcase(p.name, glob), reason, owner)


def under(glob: str, reason: str, owner: str) -> Rule:
    """Every file under a directory, at any depth, whose name matches `glob`."""
    return Rule(f"under {glob}/", lambda p, _: any(fnmatch.fnmatchcase(x, glob) for x in p.parts[:-1]), reason, owner)


def beside(file: str, siblings: tuple[str, ...], reason: str, owner: str) -> Rule:
    """A file named `file` in a directory that also holds one of `siblings`."""
    return Rule(f"{file} beside {' or '.join(siblings)}",
                lambda p, f: p.name == file and any(os.path.lexists(f.parent / s) for s in siblings), reason, owner)


def under_beside(suffix: str, sibling: str, reason: str, owner: str) -> Rule:
    """Every file under a directory `<R>{suffix}` that has `<R>{sibling}` beside it."""

    def test(p: PurePosixPath, f: Path) -> bool:
        up = f.parent
        for x in reversed(p.parts[:-1]):
            if x.endswith(suffix) and len(x) > len(suffix) and os.path.lexists(up.parent / (x[:-len(suffix)] + sibling)):
                return True
            up = up.parent
        return False

    return Rule(f"under <R>{suffix}/ beside <R>{sibling}", test, reason, owner)


def contains(text: str, suffix: str, reason: str, owner: str) -> Rule:
    """A regular file named `*suffix`, at most SCAN_MAX_BYTES, whose bytes hold `text`."""
    needle = text.encode()

    def test(p: PurePosixPath, f: Path) -> bool:
        if not p.name.endswith(suffix):
            return False
        try:
            fd = os.open(f, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            return False
        with os.fdopen(fd, "rb") as fh:
            st = os.fstat(fh.fileno())
            return stat.S_ISREG(st.st_mode) and st.st_size <= SCAN_MAX_BYTES and needle in fh.read(SCAN_MAX_BYTES + 1)

    return Rule(f"*{suffix} holding {text}", test, reason, owner, reads=True)


class PathsUnread(ValueError):
    """A run's statement or a private-names list that can't be read as one: nothing of the run is copied while its private paths
    aren't known."""


def kept_apart(prefix: str, reason: str, owner: str) -> Rule:
    """Every file whose path has a component (a directory's name or its own) starting with `prefix`."""
    return Rule(f"kept apart {prefix}*", lambda p, _: any(x.startswith(prefix) for x in p.parts), reason, owner)


def names_of(doc: Any, where: str, label: str = "") -> tuple[Rule, ...]:
    """The rules of a NAMES_SCHEMA document, `kept_apart` then `private`, each with its `why` when the document gives one and its
    printed rule after `label`; PathsUnread naming `where` when it is not one."""
    def strs(k: str) -> bool:
        return isinstance(doc.get(k), list) and all(isinstance(g, str) and g for g in doc[k])

    if not isinstance(doc, dict) or doc.get("schema") != NAMES_SCHEMA or not strs("kept_apart") or not strs("private") \
            or not isinstance(doc.get("why", {}), dict):
        raise PathsUnread(f"{where}: not {{\"schema\": \"{NAMES_SCHEMA}\", \"kept_apart\": [prefixes], \"private\": [globs]}}")
    why = doc.get("why", {})

    def of(g: str) -> tuple[str, str]:
        w = why.get(g)
        return (str(w[1]), str(w[0])) if isinstance(w, list) and len(w) == 2 else ("", "")

    rules = [*(kept_apart(g, *of(g)) for g in doc["kept_apart"]), *(name(g, *of(g)) for g in doc["private"])]
    return tuple(Rule(label + r.what, r.test, r.reason, r.owner) for r in rules)


def _read(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise PathsUnread(f"{p}: {type(e).__name__}: {e}") from None


def floor_text() -> str:
    """The floor's TREE_NAMES as text: the first under FLOOR_ROOTS; PathsUnread when none holds it."""
    if "NAMES_TEXT" in globals():
        return globals()["NAMES_TEXT"]
    for root in FLOOR_ROOTS:
        if (root / TREE_NAMES).is_file():
            try:
                return (root / TREE_NAMES).read_text()
            except OSError as e:
                raise PathsUnread(f"{root / TREE_NAMES}: {type(e).__name__}: {e}") from None
    raise PathsUnread(f"no {TREE_NAMES} under {' or '.join(map(str, FLOOR_ROOTS))}, the tree research runs from: its private names "
                      "aren't known")


def _floor() -> tuple[tuple[Rule, ...], str | None]:
    try:
        text = floor_text()
        try:
            doc = json.loads(text)
        except ValueError as e:
            raise PathsUnread(f"the floor's {TREE_NAMES}: {type(e).__name__}: {e}") from None
        return names_of(doc, f"the floor's {TREE_NAMES}"), None
    except PathsUnread as e:
        return (Rule(f"every file: {e}", lambda p, _: True),), str(e)


# NAMES_TEXT is set beside this module's source where it runs without its directory (remote.PRIVATE_PROGRAM).  A floor that can't be
# read refuses every file, and run_rules and check_payload raise its PathsUnread.
NAMES, FLOOR_UNREAD = _floor()

PLACES: tuple[Rule, ...] = (
    under_beside(".salts", ".json", "a registration's salt directory (index.json and <i>.bin) beside the registration",
                 "@compute-accounting"),
    under("pass-[0-9][0-9][0-9][0-9][0-9]*", "a Pearl+ pass's retain directory, refused whole", "@compute-accounting"),
    beside("manifest.json", ("salts", "salts_t"), "a Pearl+ pass manifest beside its salts", "@compute-accounting"),
    contains("hm96_salt_key", ".json", "a Pearl+ pass manifest holding the prover's salt key", "@compute-accounting"),
)
RULES: tuple[Rule, ...] = (*NAMES, *PLACES)


def _named(glob: str, what: str) -> Rule:
    """A path whose own name or any directory's name matches `glob`; with a trailing `/`, only a directory's."""
    g = glob.rstrip("/")
    if glob.endswith("/"):
        return Rule(what, lambda p, _: any(fnmatch.fnmatchcase(x, g) for x in p.parts[:-1]))
    return Rule(what, lambda p, _: any(fnmatch.fnmatchcase(x, g) for x in p.parts))


def _inside(d: str, what: str) -> Rule:
    return Rule(what, lambda p, f: os.path.realpath(f) == d or os.path.realpath(f).startswith(d.rstrip("/") + "/"))


def shipped_names(run_dir: Path) -> tuple[Path | None, str | None]:
    """(the run's shipped tree's TREE_NAMES, None) when job.json's run.remote.source_dir names a tree holding it; (None, why) when it
    names a tree that is missing or holds none, whose names custody then doesn't apply; (None, None) for a run with no shipped tree."""
    try:
        job = json.loads((Path(run_dir) / "job.json").read_text())
        src = ((job.get("run") or {}).get("remote") or {}).get("source_dir")
    except (OSError, ValueError, AttributeError):
        return None, None
    if not isinstance(src, str) or not src:
        return None, None
    if not Path(src).is_dir():
        return None, f"the run's shipped tree {src} is missing, so only the floor's private names applied"
    if not os.path.lexists(t := Path(src) / TREE_NAMES):
        return None, f"the run's shipped tree {src} holds no {TREE_NAMES}, so only the floor's private names applied"
    return t, None


def run_rules(run_dir: Path) -> tuple[Rule, ...]:
    """RULES, then the run's: its statement's (PATHS_FILE) names and directories, and its shipped tree's TREE_NAMES.  PathsUnread,
    naming the file, when the floor's list, the statement or the tree's list is there and is not its schema, or the floor's is not
    there."""
    if FLOOR_UNREAD:
        raise PathsUnread(FLOOR_UNREAD)
    out: list[Rule] = []
    p = Path(run_dir) / PATHS_FILE
    if os.path.lexists(p):
        doc = _read(p)
        if not (isinstance(doc, dict) and doc.get("schema") == PATHS_SCHEMA and isinstance(doc.get("private"), list)
                and isinstance(doc.get("names"), list) and all(isinstance(d, str) and os.path.isabs(d) for d in doc["private"])
                and all(isinstance(g, str) and g.strip("/") for g in doc["names"])):
            raise PathsUnread(f"{p}: not {{\"schema\": \"{PATHS_SCHEMA}\", \"private\": [absolute directories], "
                              f"\"names\": [globs]}}")
        out += [_named(g, f"{PATHS_FILE}: name {g}") for g in doc["names"]]
        out += [_inside(os.path.realpath(d), f"{PATHS_FILE}: inside {d}") for d in doc["private"]]
    if (t := shipped_names(run_dir)[0]) is not None:
        out += names_of(_read(t), str(t), "the run's tree: ")
    return (*RULES, *out)


def rule_for(rel: str, path: Path, *, follow: bool = False, rules: tuple[Rule, ...] = RULES) -> Rule | None:
    """The first of `rules` that makes `path` (named `rel`) private, else None.  A symlink is private when its own name or the path
    it resolves to is; its target is read for `contains` only with `follow` (a put, which stores a link's target by content)."""
    rp = PurePosixPath(rel)
    if not os.path.islink(path):
        return next((r for r in rules if r.test(rp, path)), None)
    target = Path(os.path.realpath(path))
    # a link to a directory is private when what lies under it would be: tested as a child of it
    names = [(rp, path), (PurePosixPath(target.as_posix()), target)]
    if target.is_dir():
        names += [(p / "-", f / "-") for p, f in names]
    return next((r for p, f in names for r in rules if (follow or not r.reads) and r.test(p, f)), None)


def refusal(rel: str, rule: Rule, *, link: bool = False) -> dict[str, Any]:
    return {"path": rel, "rule": rule.what, **({"owner": rule.owner} if rule.owner else {}), **({"link": True} if link else {})}


def scan(root: Path, *, links_only: bool = False, rules: tuple[Rule, ...] | None = None) -> list[dict[str, Any]]:
    """Every private file and every symlink (to a file or a directory, never followed) under `root` that `rules` (by default
    `run_rules(root)`) refuse, by path relative to `root`, in path order; with `links_only`, the symlinks alone."""
    root = Path(root)
    rules = run_rules(root) if rules is None else rules
    out = []
    for base, dirs, files in os.walk(root):
        dirs.sort()
        for n in sorted([*files, *(d for d in dirs if os.path.islink(os.path.join(base, d)))]):
            p = Path(base) / n
            if links_only and not p.is_symlink():
                continue
            rel = p.relative_to(root).as_posix()
            if rule := rule_for(rel, p, rules=rules):
                out.append(refusal(rel, rule, link=p.is_symlink()))
    return sorted(out, key=lambda r: r["path"])


class PrivateFiles(ValueError):
    """A payload that holds a private file: nothing of it is stored."""

    def __init__(self, refused: list[dict[str, Any]]):
        self.refused = refused
        super().__init__(f"{len(refused)} private file(s) refused, nothing stored: " + summary(refused)
                         + ". A private file is never stored; a lane that needs one stored changes research/store/private.py in a "
                         "reviewed change")


def summary(refused: list[dict[str, Any]], n: int = 3) -> str:
    shown = ", ".join(f"{r['path']} ({', '.join(x for x in (r.get('rule'), r.get('owner')) if x)})" for r in refused[:n])
    return shown + (f" and {len(refused) - n} more" if len(refused) > n else "")


def _paths_root(f: Path) -> Path | None:
    return next((d for d in f.parents if os.path.lexists(d / PATHS_FILE)), None)


#: the fixture registry at a repository's root, and its map of the registered paths that are public on purpose although they hold
#: private-named files (a toy circuit's prover store on made-up inputs): {"toys": {registered path: reason}}
REGISTRY = "artifacts.json"
TOYS = "toys"


def toy(path: Path) -> str | None:
    """The reason `path` may hold private-named files: it is an entry of the REGISTRY at its repository's root (the nearest
    enclosing directory holding `.git`) and that registry's TOYS map marks it; else None.  No other REGISTRY is read."""
    path = Path(os.path.abspath(path))
    repo = next((d for d in path.parents if os.path.lexists(d / ".git")), None)
    if repo is None or not (repo / REGISTRY).is_file():
        return None
    try:
        doc = json.loads((repo / REGISTRY).read_text())
    except (OSError, ValueError) as e:
        print(f"research: {repo / REGISTRY} was not read ({type(e).__name__}), so {path.name} is taken as no toy", file=sys.stderr)
        return None
    rel = path.relative_to(repo).as_posix()
    toys = doc.get(TOYS) if isinstance(doc, dict) else None
    registered = {v.rstrip("/") for k, v in doc.items() if k.startswith("art:") and isinstance(v, str)} if isinstance(doc, dict) else set()
    why = toys.get(rel) if isinstance(toys, dict) and rel in registered else None
    return why if isinstance(why, str) and why.strip() else None


def check_payload(files: list[Path], root: Path | None = None) -> None:
    """PrivateFiles when any of the files a put would store is private, by its absolute path or, for a symlink, by what it
    resolves to (read: the put stores its target's bytes), or by the run rules (`run_rules`) of its nearest directory holding a
    PATHS_FILE; PathsUnread when the floor's list or that directory's run rules can't be read.  A payload `root` that is a
    registered toy (`toy`) is stored whole, and says so."""
    if root is not None and (why := toy(root)) is not None:
        print(f"research: {root} is a registered toy fixture ({REGISTRY} {TOYS}: {why}): stored with any private-named files it holds",
              file=sys.stderr)
        return
    if FLOOR_UNREAD:
        raise PathsUnread(FLOOR_UNREAD)
    bad, ruled = [], {}
    for f in files:
        absolute = Path(os.path.abspath(f))
        rel = absolute.as_posix()
        if (root_dir := _paths_root(absolute)) is not None:
            if root_dir not in ruled:
                ruled[root_dir] = run_rules(root_dir)[len(RULES):]
            if r := rule_for(absolute.relative_to(root_dir).as_posix(), absolute, follow=True, rules=ruled[root_dir]):
                bad.append(refusal(rel, r, link=absolute.is_symlink()))
                continue
        if rule := rule_for(rel, absolute, follow=True):
            bad.append(refusal(rel, rule, link=absolute.is_symlink()))
    if bad:
        raise PrivateFiles(bad)


def listed(run_dir: Path) -> list[dict[str, Any]]:
    """The refusals the run dir's REFUSED_FILE lists (none when it is absent or unreadable)."""
    p = Path(run_dir) / REFUSED_FILE
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        print(f"research: {p} unread ({type(e).__name__}: {e}), so the private files custody refused are not shown", file=sys.stderr)
        return []
    got = doc.get("refused") if isinstance(doc, dict) and doc.get("schema") == REFUSED_SCHEMA else None
    return [r for r in got if isinstance(r, dict)] if isinstance(got, list) else []


__all__ = ["FLOOR_ROOTS", "FLOOR_UNREAD", "NAMES", "NAMES_SCHEMA", "PATHS_FILE", "PATHS_SCHEMA", "PLACES", "REFUSED_FILE",
           "REFUSED_SCHEMA", "REGISTRY", "RULES", "SCAN_MAX_BYTES", "TOYS", "TREE_NAMES", "PathsUnread", "PrivateFiles", "Rule", "beside",
           "check_payload", "contains", "floor_text", "kept_apart", "listed", "name", "names_of", "refusal", "rule_for", "run_rules",
           "scan", "shipped_names", "summary", "toy", "under", "under_beside"]
