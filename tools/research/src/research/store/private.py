"""The private files no run record and no `research data put` stores, one rule a line: each refused by where it lies, its name, or
(`contains`) its text, with the reason and the owning lane's handle.  A lane adds a line to RULES in a reviewed change; nothing
overrides the list at run time.  Custody lists what it refused (store.custody); a put refuses the whole payload."""
from __future__ import annotations

import fnmatch
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

#: `contains` reads only a file of its suffix at most this big; any other file is never opened
SCAN_MAX_BYTES = 4 << 20
#: custody's listing of the private files it refused (REFUSED_SCHEMA: `refused`, [{path, rule, owner}]), in the run dir and its record
REFUSED_FILE = "custody-refused.json"
REFUSED_SCHEMA = "research/custody-refused/v1"


@dataclass(frozen=True)
class Rule:
    what: str                                           # the rule as custody prints it and records it
    test: Callable[[PurePosixPath, Path], bool]         # (the path's components, the file on disk) -> private
    reason: str
    owner: str
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


RULES: tuple[Rule, ...] = (
    name("keys.bin", "a vLLM store dump's per-step salt keys", "@vllm"),
    name("weights.key", "the weights registration's signing key", "@vllm"),
    name("salts", "a Pearl+ pass's fresh hm96 salts", "@compute-accounting"),
    name("salts_t", "a Pearl+ pass's tile salts", "@compute-accounting"),
    name("*.salts", "hm96 salts beside their rows (a Pearl+ pass's y/<index>.salts, a table's layer salts)", "@compute-accounting"),
    under("*.salts", "a registration's salt directory (index.json and <i>.bin)", "@compute-accounting"),
    under("pass-[0-9][0-9][0-9][0-9][0-9]*", "a Pearl+ pass's retain directory, refused whole", "@compute-accounting"),
    beside("manifest.json", ("salts", "salts_t"), "a Pearl+ pass manifest beside its salts", "@compute-accounting"),
    contains("hm96_salt_key", ".json", "a Pearl+ pass manifest holding the prover's salt key", "@compute-accounting"),
)


def rule_for(rel: str, path: Path, *, follow: bool = False) -> Rule | None:
    """The first rule that makes `path` (named `rel`) private, else None.  A symlink is private when its own name or the path it
    resolves to is; its target is read for `contains` only with `follow` (a put, which stores a link's target by content)."""
    rp = PurePosixPath(rel)
    if not os.path.islink(path):
        return next((r for r in RULES if r.test(rp, path)), None)
    target = Path(os.path.realpath(path))
    # a link to a directory is private when what lies under it would be: tested as a child of it
    names = [(rp, path), (PurePosixPath(target.as_posix()), target)]
    if target.is_dir():
        names += [(p / "-", f / "-") for p, f in names]
    return next((r for p, f in names for r in RULES if (follow or not r.reads) and r.test(p, f)), None)


def refusal(rel: str, rule: Rule, *, link: bool = False) -> dict[str, Any]:
    return {"path": rel, "rule": rule.what, "owner": rule.owner, **({"link": True} if link else {})}


def scan(root: Path, *, links_only: bool = False) -> list[dict[str, Any]]:
    """Every private file and every symlink (to a file or a directory, never followed) under `root` that RULES refuse, by path
    relative to `root`, in path order; with `links_only`, the symlinks alone."""
    root = Path(root)
    out = []
    for base, dirs, files in os.walk(root):
        dirs.sort()
        for n in sorted([*files, *(d for d in dirs if os.path.islink(os.path.join(base, d)))]):
            p = Path(base) / n
            if links_only and not p.is_symlink():
                continue
            rel = p.relative_to(root).as_posix()
            if rule := rule_for(rel, p):
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
    shown = ", ".join(f"{r['path']} ({r['rule']}, {r['owner']})" for r in refused[:n])
    return shown + (f" and {len(refused) - n} more" if len(refused) > n else "")


def check_payload(files: list[Path]) -> None:
    """PrivateFiles when any of the files a put would store is private, by its absolute path or, for a symlink, by what it
    resolves to (read: the put stores its target's bytes)."""
    bad = []
    for f in files:
        absolute = Path(os.path.abspath(f))
        if rule := rule_for(absolute.as_posix(), absolute, follow=True):
            bad.append(refusal(absolute.as_posix(), rule, link=absolute.is_symlink()))
    if bad:
        raise PrivateFiles(bad)


def listed(run_dir: Path) -> list[dict[str, Any]]:
    """The refusals the run dir's REFUSED_FILE lists (none when it is absent or unreadable)."""
    try:
        doc = json.loads((Path(run_dir) / REFUSED_FILE).read_text())
    except (OSError, ValueError):
        return []
    got = doc.get("refused") if isinstance(doc, dict) and doc.get("schema") == REFUSED_SCHEMA else None
    return [r for r in got if isinstance(r, dict)] if isinstance(got, list) else []


__all__ = ["REFUSED_FILE", "REFUSED_SCHEMA", "RULES", "SCAN_MAX_BYTES", "listed", "PrivateFiles", "Rule", "beside", "check_payload", "contains", "name", "refusal", "rule_for",
           "scan", "summary", "under"]
