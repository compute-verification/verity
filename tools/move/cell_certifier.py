#!/usr/bin/env python3
"""The cell certifier move: the network certifier's cells-on-a-schedule relay (design v3) is the cell certifier.

    python3 tools/move/cell_certifier.py              rename and move, then stage it all; prints what it changed
    python3 tools/move/cell_certifier.py --dry-run    print what it would change, and change nothing
    python3 tools/move/cell_certifier.py --avoid REF  leave each line REF changes as it was (and the line on either side),
                                                      so REF merges after it without a conflict; prints what stands

What it does, on the tree as it is (so it reruns on a later main, and `restack.py --script tools/move/cell_certifier.py`
carries a branch across it):
1. moves the relay's modules, the benchmarks that run it and their tests (`MOVES`), each from its old place if it is there
   (`warden_chain.py`, which code still being written imports by that name, stays for now);
2. renames their module names (`warden_rows`, `warden_serve`, ...) and the relay's class names (`WardenParams`, and in
   Python `Warden`) where they stand as names, in every tracked text file `EXCLUDE` doesn't cover;
3. in the moved Python and shell files, renames the prose (comments, docstrings, `help=` strings) from "warden" to
   "cell certifier", outside code in backticks, flags, file names, paths and domain tags
   (`verity-network-warden/.../vN`, a format's, stay);
4. appends the moved paths to research's `moves.json` once, as `rename.py` does.
The Lean (`*.lean`, the locks) is not this move's: its names move with the Lean's own rename. A second run changes nothing.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter
from pathlib import Path

NC = "verity/core/protocols/network_certifier"
NT = "benchmarks/network_traces"
MOVES = {
    f"{NC}/warden.py": f"{NC}/cell_certifier.py",
    f"{NC}/warden_rows.py": f"{NC}/cell_certifier_rows.py",
    f"{NC}/tests/test_network_certifier_warden.py": f"{NC}/tests/test_network_certifier_cell_certifier.py",
    f"{NC}/tests/test_network_certifier_warden_adversarial.py":
        f"{NC}/tests/test_network_certifier_cell_certifier_adversarial.py",
    f"{NC}/tests/test_network_certifier_warden_rows.py": f"{NC}/tests/test_network_certifier_cell_certifier_rows.py",
    f"{NC}/tests/test_network_certifier_warden_rows_phase11.py":
        f"{NC}/tests/test_network_certifier_cell_certifier_rows_phase11.py",
    f"{NC}/tests/test_network_certifier_warden_rows_replay.py":
        f"{NC}/tests/test_network_certifier_cell_certifier_rows_replay.py",
    f"{NT}/warden_serve.py": f"{NT}/cell_certifier_serve.py",
    f"{NT}/warden_zk.py": f"{NT}/cell_certifier_zk.py",
    f"{NT}/warden_zk.sh": f"{NT}/cell_certifier_zk.sh",
    f"{NT}/tests/test_warden_serve.py": f"{NT}/tests/test_cell_certifier_serve.py",
    f"{NT}/tests/test_warden_serve_phase10.py": f"{NT}/tests/test_cell_certifier_serve_phase10.py",
    f"{NT}/tests/test_warden_zk.py": f"{NT}/tests/test_cell_certifier_zk.py",
    "benchmarks/network_e2e/traces/warden_probe.py": "benchmarks/network_e2e/traces/cell_certifier_probe.py",
    "benchmarks/network_e2e/traces/tests/test_warden_probes.py":
        "benchmarks/network_e2e/traces/tests/test_cell_certifier_probes.py",
    "benchmarks/one_stage/a5_warden.py": "benchmarks/one_stage/a5_cell_certifier.py",
    "benchmarks/one_stage/warden_replay.py": "benchmarks/one_stage/cell_certifier_replay.py",
    "benchmarks/one_stage/tests/test_a5_warden.py": "benchmarks/one_stage/tests/test_a5_cell_certifier.py",
    "benchmarks/one_stage/tests/test_warden_replay.py": "benchmarks/one_stage/tests/test_cell_certifier_replay.py",
    "benchmarks/live_challenger/tests/test_auditor_warden.py":
        "benchmarks/live_challenger/tests/test_auditor_cell_certifier.py",
}
MOVED = set(MOVES) | set(MOVES.values())

# (pattern, replacement), one pass, the first alternative that matches at a place wins. A name is bounded by `_B`, `_E`.
_B, _E = r"(?<![A-Za-z0-9_])", r"(?![A-Za-z0-9_])"
NAMES = [
    (_B + r"test_network_certifier_warden", "test_network_certifier_cell_certifier"),
    (_B + r"test_warden_serve", "test_cell_certifier_serve"),
    (_B + r"test_warden_zk" + _E, "test_cell_certifier_zk"),
    (_B + r"test_warden_probes" + _E, "test_cell_certifier_probes"),
    (_B + r"test_warden_replay" + _E, "test_cell_certifier_replay"),
    (_B + r"test_a5_warden" + _E, "test_a5_cell_certifier"),
    (_B + r"test_auditor_warden" + _E, "test_auditor_cell_certifier"),
    (_B + r"network_traces_warden_", "network_traces_cell_certifier_"),
    (_B + r"warden_rows" + _E, "cell_certifier_rows"),
    (_B + r"warden_serve" + _E, "cell_certifier_serve"),
    (_B + r"warden_zk" + _E, "cell_certifier_zk"),
    (_B + r"warden_probe" + _E, "cell_certifier_probe"),
    (_B + r"warden_replay" + _E, "cell_certifier_replay"),
    (_B + r"a5_warden" + _E, "a5_cell_certifier"),
    (r"network_certifier\.warden" + _E, "network_certifier.cell_certifier"),
    (r"network_certifier/warden" + _E, "network_certifier/cell_certifier"),
    (r"network_certifier import warden" + _E, "network_certifier import cell_certifier"),
    (_B + r"warden as W" + _E, "cell_certifier as W"),
    (r"(?<![\w/.-])warden\.py\b", "cell_certifier.py"),
    (_B + r"WardenParams" + _E, "CellCertifierParams"),
    (_B + r"warden\.Warden" + _E, "cell_certifier.CellCertifier"),
]
#: The class, in Python files only.
CLASS = [(_B + r"Warden" + _E, "CellCertifier")]
#: Rules of one file, before `NAMES`: a file that binds the module to the name `warden` and uses it by that name, which
#: stays its own, and the serve script's own name in its log lines and scratch directories.
_SERVE = [(r'(?<=")warden-serve(?=[: -])', "cell-certifier-serve"), (r'(?<=")warden-publisher"', 'cell-certifier-publisher"')]
BOUND = {"benchmarks/network_e2e/traces/corpus.py":
         [(r"network_certifier import warden" + _E, "network_certifier import cell_certifier as warden")],
         f"{NT}/warden_serve.py": _SERVE, f"{NT}/cell_certifier_serve.py": _SERVE}
PROSE = [
    (r"(?<![\w-])warden-serve\b", "cell-certifier-serve"), (r"(?<![\w-])warden-zk\b", "cell-certifier-zk"),
    (r"(?<![\w-])warden-form\b", "cell-certifier-form"), (r"(?<![\w-])warden-signed\b", "cell-certifier-signed"),
    (r"\bAn active warden\b", "A cell certifier"), (r"\ban active warden\b", "a cell certifier"),
    (r"\bThe active warden\b", "The cell certifier"), (r"\bthe active warden\b", "the cell certifier"),
    (r"\bactive warden\b", "cell certifier"),
    (r"\bwardens\b", "cell certifiers"), (r"\bWardens\b", "Cell certifiers"),
    (r"\bwarden\b", "cell certifier"),
]
#: What prose never renames: code in backticks, compounds (`warden-run`, `warden.json`), paths, flags and domain tags.
KEEP = re.compile(r"``[^`]*``|`[^`\n]*`|[\w./-]*warden[\w-]*(?:[./][\w./-]+)|[\w-]+-warden\b|"
                  r"warden-(?!(?:serve|zk|form|signed)\b)[\w-]+|[\w.-]+/warden\b|warden/[\w./-]*|--[\w-]+")
#: The module in backticks, alone or with one of its names, which prose renames.
TICKED = [(r"`warden`", "`cell_certifier`"), (r"`warden_rows`", "`cell_certifier_rows`"),
          *((rf"`warden\.{n}" + _E, f"`cell_certifier.{n}") for n in ("cells", "Shim", "audit", "cell", "read_cell"))]

EXCLUDE = ["archive/**", "tools/move/**", "**/CHANGELOG.md", ".agents/skills/friction/SKILL.md", "**/lean-audit.json",
           "**/*.lean", "**/*.patch", "uv.lock", "**/*.lock", "**/TRUSTED.sha256", "**/lake-manifest.json"]
MOVES_JSON = "tools/research/src/research/store/moves.json"


def git(root: Path, *a: str) -> str:
    return subprocess.run(["git", *a], cwd=root, check=True, capture_output=True, text=True).stdout


def excluded(p: str) -> bool:
    return any(fnmatch.fnmatchcase(p, g) or (g.startswith("**/") and fnmatch.fnmatchcase(p, g[3:])) for g in EXCLUDE)


def rules(table: list[tuple[str, str]]):
    pattern = re.compile("|".join(f"(?P<g{i}>{r})" for i, (r, _) in enumerate(table)))

    def sub(text: str, counts: Counter) -> str:
        def one(m: re.Match) -> str:
            i = int(m.lastgroup[1:])
            counts[table[i]] += 1
            return table[i][1]
        return pattern.sub(one, text)
    return sub


_NAMES, _CLASS, _PROSE, _TICKED = rules(NAMES), rules(CLASS), rules(PROSE), rules(TICKED)
_BOUND = {p: rules(t) for p, t in BOUND.items()}


def prose(text: str, counts: Counter) -> str:
    """`text`, a comment, docstring or help string, with its prose renamed outside `KEEP`'s spans."""
    text = _TICKED(text, counts)
    out, last = [], 0
    for m in KEEP.finditer(text):
        out += [_PROSE(text[last:m.start()], counts), m.group(0)]
        last = m.end()
    return "".join(out) + _PROSE(text[last:], counts)


def py_prose(src: str, counts: Counter) -> str:
    """`src` with `prose` applied to its comments, its docstrings (a string that starts a statement) and its `help=`,
    `description=` and `doc=` strings, each with the strings implicitly concatenated after it."""
    starts = [0]
    for line in src.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    spans, sig, run = [], [], False
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT:
            spans.append((t.start, t.end))
            continue
        if t.type == tokenize.NL:
            continue
        if t.type == tokenize.STRING and (
                not sig or sig[-1].type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
                or (len(sig) >= 2 and sig[-2].type == tokenize.NAME and sig[-2].string in ("help", "description", "doc")
                    and sig[-1].string == "=")
                or (run and sig[-1].type == tokenize.STRING)):
            spans.append((t.start, t.end))
            sig.append(t)
            run = True
            continue
        sig.append(t)
        run = False
    out, pos = [], 0
    for (sr, sc), (er, ec) in spans:
        a, b = starts[sr - 1] + sc, starts[er - 1] + ec
        out += [src[pos:a], prose(src[a:b], counts)]
        pos = b
    return "".join(out) + src[pos:]


def sh_prose(src: str, counts: Counter) -> str:
    return "".join(prose(line, counts) if line.lstrip().startswith("#") and not line.startswith("#!") else line
                   for line in src.splitlines(keepends=True))


def avoided(root: Path, ref: str) -> dict[str, set[int]]:
    """For each path of HEAD's tree, the lines (from 1) that `ref` changes since its merge base with HEAD, and the line on
    either side of each (git conflicts on adjacent changes), with a path HEAD has moved since then read at its place."""
    base = git(root, "merge-base", "HEAD", ref).strip()
    moved = {}
    for line in git(root, "diff", "--name-status", "-M50%", base, "HEAD").splitlines():
        status, *paths = line.split("\t")
        if status.startswith("R"):
            moved[paths[0]] = paths[1]
    out: dict[str, set[int]] = {}
    path = None
    for line in git(root, "diff", "-U0", "--no-renames", base, ref).splitlines():
        if line.startswith("--- "):
            path = None if line[4:] == "/dev/null" else moved.get(line[6:], line[6:])
        elif line.startswith("@@") and path is not None:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? ", line)
            a, n = int(m[1]), int(m[2] if m[2] is not None else 1)
            out.setdefault(path, set()).update(range(max(1, a - 1), a + max(n, 1) + 1))
    return out


def record(root: Path, moves: dict[str, str]) -> str | None:
    doc, entry = json.loads((root / MOVES_JSON).read_text()), {"paths": dict(sorted(moves.items()))}
    if not moves or entry in doc["moves"]:
        return None
    doc["moves"].append(entry)
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--avoid", default=None, metavar="REF")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root = (a.root or Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())).resolve()
    if not a.dry_run and git(root, "status", "--porcelain").strip():
        raise SystemExit(f"{root}: the worktree isn't clean")
    files = [p for p in git(root, "ls-files", "-z").split("\0") if p]
    moves = {o: n for o, n in MOVES.items() if (root / o).exists()}
    avoid = avoided(root, a.avoid) if a.avoid else {}
    counts, texts, standing = Counter(), {}, Counter()
    for p in files:
        if excluded(p) or not (root / p).is_file() or (root / p).is_symlink():
            continue
        try:
            old = (root / p).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "arden" not in old:
            continue
        new = _BOUND[p](old, counts) if p in _BOUND else old
        new = _NAMES(new, counts)
        if p.endswith(".py"):
            new = _CLASS(new, counts)
        if p in MOVED and p.endswith(".py"):
            new = py_prose(new, counts)
        elif p in MOVED and p.endswith(".sh"):
            new = sh_prose(new, counts)
        if p in avoid and new != old:
            before, after = old.splitlines(keepends=True), new.splitlines(keepends=True)
            for i in sorted(avoid[p]):
                if i <= len(before) and before[i - 1] != after[i - 1]:
                    after[i - 1] = before[i - 1]
                    standing[p, i] += 1
            new = "".join(after)
        if new != old:
            texts[p] = new
    if (t := record(root, moves)) is not None:
        texts[MOVES_JSON] = t
    print(f"cell certifier move at {git(root, 'rev-parse', 'HEAD').strip()}{' (dry run)' if a.dry_run else ''}")
    print(f"  files rewritten: {len(texts)}, files moved: {len(moves)}")
    for (r, n), c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {c:5d}  {r!r} -> {n!r}")
    if standing:
        print(f"  lines left as they were, which {a.avoid} changes: {len(standing)}")
        for p, i in sorted(standing):
            print(f"    {p}:{i}")
    if a.dry_run:
        return 0
    for p, t in texts.items():
        (root / p).write_text(t, encoding="utf-8")
    for o, n in moves.items():
        git(root, "mv", o, n)
    git(root, "add", "-A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
