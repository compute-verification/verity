"""`move-check`: certify that BASE..HEAD is a pure move, so the lander may schedule it first in a quiet window, alone in its
train (Daniel, 4 Oct). It relaxes no gate: a certified move still goes through `check`.

    uv run python tools/check/move_check.py BASE HEAD [--map OLD=NEW ...] [--json]

Every path git sees changed must be one of:
- a rename with byte-identical content and mode (git's 100% rename);
- an edit, in place or with a rename, to an allowlisted path-bearing file (`PATH_FILES`, `PATH_NAMES`, a skill's `SKILL.md`)
  that leaves it unchanged once each old path in it is rewritten to its new one, a quoted relative path (`".."`,
  `"../level3"`) being resolved against the file's own directory first. This covers each `lean-audit.json`, whose pins
  are then unchanged;
- an edit to a `*.py` file whose changed lines only rewrite the module of an `import M` / `from M import` line, from where
  a renamed module was to where it is now.
Anything else is refused, with its first offending hunk. Exit 0 certified, 1 refused, 2 on a usage or git error.

The path map comes from the exact renames. Each rename contributes the shortest prefix pair (old directory, new directory)
whose whole subtree in BASE moved there file by file. When only `protocols/pous` moved, `protocols -> security_proofs` is
never inferred. `--map OLD=NEW` adds an entry, and a dotted one (no `/`) also maps a module.
"""
from __future__ import annotations

import argparse
import bisect
import difflib
import json
import posixpath
import re
import subprocess
import sys

PATH_FILES = {"tools/check/ci.toml", "tests/test_lean_packages.py", "AGENTS.md", "fixtures/artifacts.json"}
PATH_NAMES = {"pyproject.toml", "lakefile.lean", "lakefile.toml", "lake-manifest.json", "lean-audit.json", ".gitignore"}
#: files whose quoted relative paths name another directory of the tree (a Lake `require`, a manifest's `dir`, a uv source)
RELATIVE_NAMES = {"pyproject.toml", "lakefile.lean", "lakefile.toml", "lake-manifest.json"}
IMPORT = re.compile(r"^(\s*from\s+)([\w.]+)(\s+import\b.*)$|^(\s*import\s+)([\w.]+)(.*)$")
RELATIVE = re.compile(r"""(["'])(\.\.?(?:/[^"'\s]*)?)\1""")
HUNK_LINES = 8


def git(*args: str, cwd: str | None = None) -> bytes:
    return subprocess.run(["git", "-c", "diff.renameLimit=0", *args], cwd=cwd, check=True, capture_output=True).stdout


def kind(path: str) -> str | None:
    name = posixpath.basename(path)
    if path in PATH_FILES or name in PATH_NAMES or (path.startswith(".agents/skills/") and name == "SKILL.md"):
        return "path"
    return "import" if path.endswith(".py") else None


def raw_diff(base: str, head: str, cwd: str | None) -> list[dict]:
    """`git diff --raw -z -M100%`, parsed: only byte-identical renames are paired; a moved file with an edit is a delete and an
    add, which `certify` pairs through the path map."""
    fields = git("diff", "--raw", "-z", "-M100%", "--no-abbrev", base, head, cwd=cwd).split(b"\0")
    out, i = [], 0
    while i < len(fields) and fields[i]:
        src_mode, dst_mode, src_sha, dst_sha, status = fields[i].decode()[1:].split(" ")
        n = 2 if status[0] in "RC" else 1
        paths = [p.decode("utf-8", "surrogateescape") for p in fields[i + 1:i + 1 + n]]
        out.append({"status": status[0], "score": int(status[1:] or 0), "old": paths[0], "new": paths[-1],
                    "modes": (src_mode, dst_mode), "shas": (src_sha, dst_sha)})
        i += 1 + n
    return out


def derive_map(renames: list[tuple[str, str]], base_files: list[str], deleted: set[str], added: set[str]) -> dict[str, str]:
    """Each exact rename's shortest (old prefix, new prefix), sharing the rest of the path, whose subtree all moved there: each
    file in it renamed there, or deleted with its counterpart there added (a moved file with an edit, judged on its own)."""
    moved = dict(renames)
    sound: dict[tuple[str, str], bool] = {}

    def there(f: str, target: str) -> bool:
        return moved.get(f) == target or (f in deleted and target in added)

    def whole(p_old: str, p_new: str) -> bool:
        if (p_old, p_new) not in sound:
            lo = bisect.bisect_left(base_files, p_old + "/")
            hi = bisect.bisect_left(base_files, p_old + "0")         # "0" sorts right after "/"
            sound[(p_old, p_new)] = all(there(f, p_new + f[len(p_old):]) for f in base_files[lo:hi])
        return sound[(p_old, p_new)]

    out: dict[str, str] = {}
    for old, new in renames:
        o, n = old.split("/"), new.split("/")
        k = 0
        while k < min(len(o), len(n)) - 1 and o[-1 - k] == n[-1 - k]:
            k += 1
        for keep in range(k, 0, -1):
            p_old, p_new = "/".join(o[:-keep]), "/".join(n[:-keep])
            if whole(p_old, p_new):
                out[p_old] = p_new
                break
        else:
            out[old] = new
    return out


def substitute(text: str, pmap: dict[str, str]) -> str:
    if not pmap:
        return text
    olds = sorted(pmap, key=len, reverse=True)
    pattern = re.compile(r"(?:(?<![\w.\-/])|(?<=^/))(" + "|".join(map(re.escape, olds)) + r")(?![\w.\-])", re.M)
    return pattern.sub(lambda m: pmap[m.group(1)], text)


def map_path(p: str, pmap: dict[str, str]) -> str:
    for old in sorted(pmap, key=len, reverse=True):
        if p == old or p.startswith(old + "/"):
            return pmap[old] + p[len(old):]
    return p


def resolve_relative(text: str, path: str, pmap: dict[str, str]) -> str:
    """Each quoted relative path, as `<root>/` and the repository path it names from `path`'s directory, mapped by `pmap`."""
    here = posixpath.dirname(path)

    def repl(m: re.Match) -> str:
        p = posixpath.normpath(posixpath.join(here, m.group(2)))
        return m.group(0) if p.startswith("..") else f"{m.group(1)}<root>/{map_path(p, pmap)}{m.group(1)}"
    return RELATIVE.sub(repl, text)


class Modules:
    """Whether a renamed module justifies rewriting `import A` to `import B`."""

    def __init__(self, renames: list[tuple[str, str]], extra: dict[str, str]):
        self.files = [(self._mod(o), self._mod(n)) for o, n in renames if o.endswith(".py") and n.endswith(".py")]
        self.extra = extra

    @staticmethod
    def _mod(path: str) -> list[str]:
        parts = path[:-3].split("/")
        return parts[:-1] if parts[-1] == "__init__" else parts

    def moved(self, a: str, b: str) -> bool:
        for old, new in self.extra.items():
            if (a == old or a.startswith(old + ".")) and b == new + a[len(old):]:
                return True
        A, B = a.split("."), b.split(".")
        for o, n in self.files:
            for r in range(min(len(o), len(n)) + 1):
                if r and o[-r] != n[-r]:
                    break
                oo, nn = o[:len(o) - r], n[:len(n) - r]
                if len(oo) >= len(A) and len(nn) >= len(B) and oo[len(oo) - len(A):] == A and nn[len(nn) - len(B):] == B:
                    return True
        return False


def import_rewrite(old: str, new: str, modules: Modules) -> bool:
    mo, mn = IMPORT.match(old), IMPORT.match(new)
    if not (mo and mn):
        return False
    go, gn = mo.groups(), mn.groups()
    i = 0 if go[0] is not None else 3
    if (gn[0] is not None) != (i == 0):
        return False
    return go[i] == gn[i] and go[i + 2] == gn[i + 2] and modules.moved(go[i + 1], gn[i + 1])


def hunk(a: list[str], b: list[str], i1: int, i2: int, j1: int, j2: int) -> str:
    lines = [f"@@ -{i1 + 1},{i2 - i1} +{j1 + 1},{j2 - j1} @@"]
    lines += ["-" + x for x in a[i1:i2][:HUNK_LINES]] + ["+" + x for x in b[j1:j2][:HUNK_LINES]]
    return "\n".join(lines)


def first_hunk(old_text: str, new_text: str, old_path: str, new_path: str, pmap: dict[str, str], modules: Modules,
               k: str | None) -> str | None:
    """The first hunk not explained by the move, in the file's own lines, or None when every changed line is."""
    a, b = old_text.splitlines(), new_text.splitlines()
    na, nb = a, b
    if k == "path":
        rel = posixpath.basename(new_path) in RELATIVE_NAMES
        na_text = resolve_relative(old_text, old_path, pmap) if rel else old_text
        na = (na_text if old_text == new_text else substitute(na_text, pmap)).splitlines()
        nb = (resolve_relative(new_text, new_path, {}) if rel else new_text).splitlines()
    py = new_path.endswith(".py")
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, na, nb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if py and tag == "replace" and i2 - i1 == j2 - j1 and all(
                import_rewrite(x, y, modules) for x, y in zip(na[i1:i2], nb[j1:j2])):
            continue
        return hunk(a, b, i1, i2, j1, j2)
    return None


def blob(sha: str, cwd: str | None) -> str | None:
    try:
        return git("cat-file", "blob", sha, cwd=cwd).decode("utf-8")
    except UnicodeDecodeError:
        return None


def certify(base: str, head: str, extra: dict[str, str] | None = None, cwd: str | None = None) -> dict:
    extra = extra or {}
    entries = raw_diff(base, head, cwd)
    exact = [(e["old"], e["new"]) for e in entries
             if e["status"] == "R" and e["score"] == 100 and e["modes"][0] == e["modes"][1]]
    base_files = sorted(p.decode("utf-8", "surrogateescape") for p in git("ls-tree", "-r", "-z", "--name-only", base, cwd=cwd).split(b"\0") if p)
    dels = {e["old"]: e for e in entries if e["status"] == "D"}
    adds = {e["new"]: e for e in entries if e["status"] == "A"}
    pmap = {**derive_map(exact, base_files, set(dels), set(adds)), **extra}
    for old, d in list(dels.items()):
        new = map_path(old, pmap)
        if new != old and new in adds:
            a = adds.pop(new)
            entries.remove(a)
            d.update(status="R", score=0, new=new, modes=(d["modes"][0], a["modes"][1]), shas=(d["shas"][0], a["shas"][1]))
    modules = Modules(exact, {o: n for o, n in extra.items() if "/" not in o})
    report = {"base": base, "head": head, "map": dict(sorted(pmap.items())), "renames": [], "edits": [], "refusals": []}

    def refuse(e: dict, reason: str, h: str | None = None) -> None:
        report["refusals"].append({"path": e["new"], **({"old": e["old"]} if e["old"] != e["new"] else {}),
                                   "status": e["status"], "reason": reason, "hunk": h})

    for e in entries:
        st, old, new = e["status"], e["old"], e["new"]
        if st not in "RM":
            refuse(e, {"A": "added", "D": "deleted", "C": "copied", "T": "type changed"}.get(st, f"status {st}"))
            continue
        if e["modes"][0] != e["modes"][1]:
            refuse(e, f"mode {e['modes'][0]} -> {e['modes'][1]}")
            continue
        k = kind(new)
        if k is None or kind(old) != k:
            if st == "R" and e["score"] == 100:
                report["renames"].append({"old": old, "new": new})
            else:
                a, b = blob(e["shas"][0], cwd), blob(e["shas"][1], cwd)
                refuse(e, "edited, not an allowlisted path-only file",
                       None if a is None or b is None else first_hunk(a, b, old, new, {}, modules, None))
            continue
        a, b = blob(e["shas"][0], cwd), blob(e["shas"][1], cwd)
        if a is None or b is None:
            refuse(e, "not UTF-8 text")
            continue
        h = first_hunk(a, b, old, new, pmap, modules, k)
        if h is not None:
            refuse(e, "relative path now resolves elsewhere" if a == b else f"edit beyond {'path strings' if k == 'path' else 'import lines'}", h)
        elif a == b:
            report["renames"].append({"old": old, "new": new})
        else:
            report["edits"].append({"path": new, **({"old": old} if old != new else {}), "kind": k})
    report["certified"] = not report["refusals"]
    return report


def summary(r: dict) -> str:
    span = f"{r['base']}..{r['head']}"
    counts = f"{len(r['renames'])} renames, {len(r['edits'])} path-only edits"
    if r["certified"]:
        return f"move-check: certified {span} is a pure move ({counts})"
    return f"move-check: refused {span}: {len(r['refusals'])} paths are not part of a pure move ({counts})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="move-check", description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW", help="an extra path (or dotted module) mapping")
    ap.add_argument("--json", action="store_true", help="print the report as JSON; the summary goes to stderr")
    args = ap.parse_args(argv)
    extra = {}
    for m in args.map:
        old, sep, new = m.partition("=")
        if not (sep and old and new):
            ap.error(f"--map {m!r}: expected OLD=NEW")
        extra[old.strip("/")] = new.strip("/")
    try:
        r = certify(args.base, args.head, extra)
    except subprocess.CalledProcessError as e:
        print(f"move-check: git {' '.join(e.cmd[3:])}: {e.stderr.decode(errors='replace').strip()}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(r, indent=2))
        print(summary(r), file=sys.stderr)
    else:
        for f in r["refusals"]:
            print(f"{f['path']}: {f['reason']}" + (f" (from {f['old']})" if "old" in f else ""))
            if f["hunk"]:
                print("    " + f["hunk"].replace("\n", "\n    "))
        print(summary(r))
    return 0 if r["certified"] else 1


if __name__ == "__main__":
    sys.exit(main())
