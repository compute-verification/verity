#!/usr/bin/env python3
"""`tools/move/flock_base.py STEP`: what follows the modules of Flock's Lean package that moved to the base packages.

The base packages are the ones Flock's package requires by path; a module `ROOT.M` of one of them was `Flock.M` (all
but ``NEW``), and `Flock/M.lean` re-exports it while an importer still names it there. Nothing here names a base
package, its directory or its root: they are read from the lakefiles, so renaming one (`rename.py`, `layout.py`) leaves
this script as it is.

    private   each `open private … from Flock.M` names the module M became. A private declaration's name holds its
              module, and a re-export holds no declaration, so these lines cannot wait for `rewrite`
    moves     the map `tools/verity/lean/check.py --update --moved` takes, on stdout: each moved module's private names
    rewrite   every import of `Flock.M`, and every `layers` entry of a lock that allows it, names the base module, and the
              re-exports are deleted; `--dry-run` counts the lines by directory and writes nothing

Stdlib and git, run from the repository root. `private` and `rewrite` refuse a dirty worktree, and a second run changes
nothing.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

FLOCK = "verity/protocols/flock/verifier"
OLD = "Flock"
#: the base packages' modules that never were modules of Flock's package
NEW = {"Pow2"}
IMPORT = re.compile(r"^((?:(?:public|private|meta)[ \t]+)*import[ \t]+(?:all[ \t]+)?)" + OLD + r"\.(\w+)[ \t]*$", re.M)
#: between two words of one command: blanks, or a break onto an indented line
GAP = r"(?:[ \t]+|[ \t]*\n[ \t]+)"
PRIVATE = re.compile(r"^(open[ \t]+private(?:" + GAP + r"(?!from\b)\S+)*" + GAP + r"from[ \t]+)" + OLD + r"\.(\w+)[ \t]*$", re.M)


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def moved(root: Path) -> dict[str, str]:
    """{M: the module `Flock.M` became}, from the packages Flock's lakefile requires by path."""
    out = {}
    for req in tomllib.loads((root / FLOCK / "lakefile.toml").read_text()).get("require", []):
        if "path" not in req:
            continue
        package = (root / FLOCK / req["path"]).resolve()
        for lib in tomllib.loads((package / "lakefile.toml").read_text()).get("lean_lib", []):
            for f in sorted((package / lib["name"]).glob("*.lean")):
                if f.stem not in NEW:
                    out[f.stem] = f"{lib['name']}.{f.stem}"
    return out


def lean_files(root: Path) -> list[str]:
    return [f for f in git(root, "ls-files", "--", "*.lean").split("\n") if f]


def clean(root: Path) -> None:
    if git(root, "status", "--porcelain"):
        sys.exit("flock_base: the worktree has changes; commit or stash them first")


def private(root: Path) -> list[tuple[str, str, str]]:
    """Rewrites each `open private … from Flock.M` of a moved M; (file, old module, new module) for each."""
    names, out = moved(root), []
    for f in lean_files(root):
        text = (root / f).read_text()
        hits = [m[2] for m in PRIVATE.finditer(text) if m[2] in names]
        if hits:
            (root / f).write_text(PRIVATE.sub(lambda m: m[1] + names[m[2]] if m[2] in names else m[0], text))
            out += [(f, f"{OLD}.{m}", names[m]) for m in hits]
    return out


def moves(root: Path) -> dict[str, str]:
    return {f"_private.{OLD}.{m}": f"_private.{new}" for m, new in sorted(moved(root).items())}


def layers(lock: dict, names: dict[str, str]) -> int:
    """Renames, in a lock's `layers`, each allowed import and exception that is a moved module; how many."""
    n = 0
    for rule in lock.get("layers", {}).values():
        for key in ("imports", "except"):
            for i, x in enumerate(rule.get(key, [])):
                if x.startswith(OLD + ".") and x.split(".", 1)[1] in names:
                    rule[key][i] = names[x.split(".", 1)[1]]
                    n += 1
    return n


def rewrite(root: Path, dry: bool = False) -> tuple[Counter, list[str]]:
    """Each import of a moved module names it where it is, the locks' `layers` allow it there, and the re-exports go.
    ({directory: lines}, the re-exports deleted)."""
    names = moved(root)
    forwarders = [f"{FLOCK}/{OLD}/{m}.lean" for m in sorted(names) if (root / FLOCK / OLD / f"{m}.lean").exists()]
    lines: Counter = Counter()
    for f in lean_files(root):
        if f in forwarders:
            continue
        text = (root / f).read_text()
        n = sum(m[2] in names for m in IMPORT.finditer(text))
        if n:
            lines[f.rsplit("/", 1)[0]] += n
            if not dry:
                (root / f).write_text(IMPORT.sub(lambda m: m[1] + names[m[2]] if m[2] in names else m[0], text))
    for f in git(root, "ls-files", "--", "*lean-audit.json").split("\n"):
        if not f:
            continue
        lock = json.loads((root / f).read_text())
        n = layers(lock, names)
        if n:
            lines[f] += n
            if not dry:
                sys.path.insert(0, str(root / "tools/verity/lean"))
                import check  # the lock's own layout

                (root / f).write_text(check.layout(lock) + "\n")
    if not dry and forwarders:
        git(root, "rm", "-q", "--", *forwarders)
    return lines, forwarders


def main(argv: list[str]) -> int:
    root = Path.cwd()
    step, dry = (argv + [""])[0], "--dry-run" in argv
    if step == "moves":
        print(json.dumps(moves(root), indent=1))
    elif step == "private":
        clean(root)
        for f, old, new in private(root):
            print(f"{f}: from {old} -> from {new}")
    elif step == "rewrite":
        if not dry:
            clean(root)
        lines, forwarders = rewrite(root, dry)
        for d, n in sorted(lines.items()):
            print(f"{n:5d}  {d}")
        print(f"{sum(lines.values())} line(s) in {len(lines)} place(s); {len(forwarders)} re-export(s) "
              f"{'to delete' if dry else 'deleted'}")
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
