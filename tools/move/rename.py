#!/usr/bin/env python3
"""The rename move: rename every word a word map names, in the tree's paths, modules, Lean names and prose.

    python3 tools/move/rename.py --map MAP              rename and move, then stage it all; prints what it changed
    python3 tools/move/rename.py --map MAP --dry-run    print what it would change, and change nothing
    python3 tools/move/rename.py --map MAP --moves      print the declaration renames that the Lean locks' records need
                                                        ({old: new}), the map `tools/verity/lean/check.py --update --moved`
                                                        takes; run it on the tree you re-record, since a branch's records
                                                        hold names no other lock has

What it does, on the tree as it is (so it reruns on a later main, and `restack.py --script tools/move/rename.py` carries a
branch across it):
1. renames each [words] entry in every tracked text file the map doesn't exclude, except inside what [keep] keeps (a domain
   tag, a format id, an id) and, in the files a [keep_in] glob matches, the words it lists;
2. moves each tracked file whose path the same words rename, with `layout.py`'s machinery (`layout.Run.move`); a module's
   name is its path's, so the first step has already renamed every import of it;
3. in a `lean-audit.json`, renames module names and prose but not the records the Lean check's `--update` writes (`guarantees`,
   `reads`, `dependencies`), so that `--update --moved` compares the build with them under their old names;
4. rewrites a `TRUSTED.sha256` line whose file the move changed, when it held that file's digest before, and likewise a
   Lake manifest's digest in a JSON pin of `tools/verity/check/pins.py`: a move changes a manifest only in where its path
   packages sit, which no pinned output reads (a pin of any other file the move changes is left for its command);
5. appends its renamed paths to research's `moves.json` once, as `layout.py` does, each moved path up to the end of each
   run of renamed parts (`.../warden/`, not each file under it, and `.../warden/tests/test_warden.py` besides), so that a
   branch carried across the move appends the same entry unless it adds a path that renames a part below a renamed one.
It refuses a dirty worktree, and a second run changes nothing.
"""
from __future__ import annotations

import argparse
import bisect
import fnmatch
import hashlib
import importlib.util
import json
import os
import posixpath
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout as L  # noqa: E402

POLICY = "lean-audit.json"
RECORDS = ("guarantees", "reads", "dependencies")   # what the Lean check's `--update` writes from the build
PINS = "TRUSTED.sha256"
PIN = re.compile(r"^([0-9a-f]{64})  (.+)$", re.M)
PINS_PY = "tools/verity/check/pins.py"              # the pins of Lean sources, some of them JSON (`json_pins`)
MANIFEST = "lake-manifest.json"
MOVES = L.MOVES                                     # research's record of each move's renamed paths


def _match(path: str, glob: str) -> bool:
    return fnmatch.fnmatchcase(path, glob) or (glob.startswith("**/") and fnmatch.fnmatchcase(path, glob[3:]))


def _word(w: str, whole: bool = False) -> str:
    """`w` as it stands as a word or as one part of an identifier, or (``whole``) as a whole word, and a namespace (``X.``)
    only at the start of a name (the rules at the top of `tests/fixtures/rename_map.toml`)."""
    if whole:
        left = (r"(?<![\w.])" if w.endswith(".") else r"(?<!\w)") if re.match(r"\w", w[0]) else ""
        right = r"(?!\w)" if re.match(r"\w", w[-1]) else ""
    else:
        if re.search(r"[ ./]", w):
            left = r"(?<![\w-])"
        elif w[0].islower():
            left = r"(?<![A-Za-z0-9])"
        else:
            left = r"(?<![A-Z])"
        right = r"(?![a-z])" if w[-1].islower() else r"(?![A-Za-z])"
    return left + re.escape(w) + right


class Words:
    def __init__(self, data: dict):
        self.words: dict[str, str] = dict(data["words"])
        self.scoped: list[tuple[list[str], dict[str, str]]] = [(s["files"], dict(s["words"]))
                                                              for s in data.get("words_in", [])]
        for table in (self.words, *(t for _, t in self.scoped)):
            if bad := [o for o, n in table.items() if not o or not n or o == n]:
                raise SystemExit(f"the word map: entries that rename nothing: {bad}")
        self.exclude = data.get("rewrite", {}).get("exclude", [])
        keep = data.get("keep", {})
        self.keep = [re.compile(p) for p in keep.get("patterns", [])]
        self.keep += [re.compile(re.escape(s)) for s in keep.get("strings", [])]
        self.keep_in: dict[str, list[str]] = data.get("keep_in", {})
        self._rules: dict[tuple[int, ...], tuple[re.Pattern, list[str], list[str]]] = {}

    def excluded(self, path: str) -> bool:
        return any(_match(path, g) for g in self.exclude)

    def rules(self, path: str = "") -> tuple[re.Pattern, list[str], list[str]]:
        """The words for file `path`: those of each [[words_in]] whose files match it (whole words, first), then [words].
        The pattern has one group per rule, and the first list holds each group's new word."""
        which = tuple(i for i, (globs, _) in enumerate(self.scoped) if path and any(_match(path, g) for g in globs))
        if which not in self._rules:
            scoped: dict[str, str] = {}
            for i in which:
                for k, v in self.scoped[i][1].items():
                    scoped.setdefault(k, v)
            rules = [(_word(w, whole=True), n) for w, n in scoped.items()] + [(_word(w), n) for w, n in self.words.items()]
            first = "".join(sorted({re.escape(w[0]) for w in {**scoped, **self.words}}))
            pattern = f"(?=[{first}])(?:" + "|".join(f"({r})" for r, _ in rules) + ")" if rules else "(?!)"
            self._rules[which] = (re.compile(pattern),
                                  [n for _, n in rules], [w.lower() for w in {**scoped, **self.words}])
        return self._rules[which]

    def sub(self, text: str, path: str = "", keep: list[tuple[int, int]] = (), counts: Counter | None = None,
            kept: Counter | None = None, protect: bool = True) -> str:
        """`text` (of file `path`) with every word renamed, except inside the spans [keep] (and `keep`) cover and the words
        [keep_in] lists for `path`. ``counts`` and ``kept`` count the words renamed (by old and new) and the words left
        standing; without ``protect``, [keep] keeps nothing (a path)."""
        pattern, new, lower = self.rules(path)
        low = text.lower()
        if not any(w in low for w in lower) or pattern.search(text) is None:
            return text
        spans = sorted([*keep, *((m.start(), m.end()) for k in self.keep if protect for m in k.finditer(text))])
        starts, reach, far = [], [], 0
        for s, e in spans:                                          # reach[i]: the furthest end of spans[:i + 1]
            starts.append(s)
            far = max(far, e)
            reach.append(far)
        skip = {w for g, ws in self.keep_in.items() if path and _match(path, g) for w in ws}

        def one(m: re.Match) -> str:
            w = m.group(0)
            i = bisect.bisect_left(starts, m.end()) - 1               # the last span starting before the match ends
            if w in skip or (i >= 0 and reach[i] > m.start()):
                if kept is not None:
                    kept[w] += 1
                return w
            if counts is not None:
                counts[w, new[m.lastindex - 1]] += 1
            return new[m.lastindex - 1]
        return pattern.sub(one, text)

    def path(self, p: str) -> str:
        return self.sub(p, p, protect=False)

    def name(self, n: str, path: str = "") -> str:
        """A Lean or Python dotted name, as it is renamed where it stands in file `path`: not where [keep] keeps it (a
        namespace that shares a module's name)."""
        return self.sub(n, path)


def record_spans(text: str) -> list[tuple[int, int]]:
    """The spans of a `lean-audit.json`'s top-level values that the Lean check's `--update` writes (`RECORDS`)."""
    dec, gap, spans = json.JSONDecoder(), re.compile(r"[\s,]*"), []
    i = gap.match(text, text.index("{") + 1).end()
    while text[i] != "}":
        key, i = json.decoder.scanstring(text, i + 1)
        i = re.compile(r"\s*:\s*").match(text, i).end()
        _, end = dec.raw_decode(text, i)
        if key in RECORDS:
            spans.append((i, end))
        i = gap.match(text, end).end()
    return spans


def renamed(name: str, moved: dict[str, str]) -> str:
    """`tools/verity/lean/check.py`'s `renamed`: `name` with its longest moved prefix replaced."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        if (p := ".".join(parts[:i])) in moved:
            return ".".join([moved[p], *parts[i:]])
    return name


def module_files(files: list[str]) -> dict[str, str]:
    """Each Lean module's file ({module: path}): a module is its file's path below its package's source directory, so it is
    each dotted tail of a `.lean` path that no other file's path ends in."""
    tails: dict[str, list[str]] = {}
    for p in files:
        if p.endswith(".lean"):
            parts = p[:-len(".lean")].split("/")
            for i in range(len(parts)):
                tails.setdefault(".".join(parts[i:]), []).append(p)
    return {m: ps[0] for m, ps in tails.items() if len(ps) == 1}


def lock_moves(root: Path, words: Words, files: list[str]) -> dict[str, str]:
    """The declaration renames the locks' records need: for each guarantee and read definition whose name the words rename
    where it is declared (a read definition in its module's file, a guarantee in its lock), its first renamed prefix, and
    the full name too when that prefix alone would rename it otherwise; none implied by another. A prefix with such a name
    under it is listed as its names in full instead, since `check.py --moved` reads no move under another move's new name
    that maps back elsewhere (its `move_errors`)."""
    want: dict[str, str] = {}
    where = module_files(files)
    for p in files:
        if posixpath.basename(p) != POLICY or words.excluded(p):
            continue
        d = json.loads((root / p).read_text())
        names = dict.fromkeys(d.get("guarantees", {}), p)
        names |= {n: where.get(m, p) for m, g in d.get("reads", {}).items() for n in g.get("definitions", {})}
        for n, f in sorted(names.items()):
            if words.excluded(f):
                continue
            if (new := words.name(n, f)) != n and want.setdefault(n, new) != new:
                raise SystemExit(f"{n} is renamed {want[n]} in one lock and {new} in {p}")
    want = dict(sorted(want.items()))
    moves: dict[str, str] = {}
    for old, new in want.items():
        a, b = old.split("."), new.split(".")
        i = next(i for i in range(len(a)) if a[i] != b[i])
        moves.setdefault(".".join(a[:i + 1]), ".".join(b[:i + 1]))
    for old, new in want.items():
        if renamed(old, moves) != new:
            moves[old] = new
    while nested := sorted({p for p, q in moves.items() for o, n in moves.items()
                            if n.startswith(q + ".") and o != p + n[len(q):]}):
        if nested[0] in want:
            raise SystemExit(f"{nested[0]} is renamed {moves[nested[0]]} and a name under it is renamed elsewhere: "
                             "`check.py --moved` can't read both back, so rename them in two moves")
        del moves[nested[0]]
        moves |= {o: n for o, n in want.items() if o.startswith(nested[0] + ".")}
    for old in sorted(moves, key=lambda k: -k.count(".")):
        rest = {k: v for k, v in moves.items() if k != old}
        if renamed(old, rest) == moves[old]:
            del moves[old]
    return dict(sorted(moves.items()))


def repin(root: Path, words: Words, pins: str, texts: dict[str, str], dest: dict[str, str]) -> str | None:
    """The `TRUSTED.sha256` at `pins` after the move: each line whose file the move changes or moves, and whose digest was
    that file's, gets the file's new digest and path. None when no line changes."""
    text = (root / pins).read_text()
    here = posixpath.dirname(pins)

    def line(m: re.Match) -> str:
        digest, rel = m.groups()
        target = posixpath.normpath(posixpath.join(here, rel))
        try:
            before = (root / target).read_bytes()
        except OSError:
            return m.group(0)
        after = texts[target].encode() if target in texts else before
        if hashlib.sha256(before).hexdigest() != digest or (after == before and target not in dest):
            return m.group(0)
        return f"{hashlib.sha256(after).hexdigest()}  {words.path(rel)}"
    new = PIN.sub(line, text)
    return None if new == text else new


def json_pins(root: Path) -> list[tuple[str, str, str]]:
    """The pins `tools/verity/check/pins.py` lists that record {path: digest} under a JSON key, as (pin, key, the directory
    its paths are below); none in a tree without it."""
    if not (root / PINS_PY).is_file():
        return []
    spec = importlib.util.spec_from_file_location("rename_move_pins", root / PINS_PY)
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)       # its dataclass looks its module up
    spec.loader.exec_module(mod)
    return [(p.path, p.key, p.under) for p in mod.PINS if p.key is not None]


def repin_manifests(root: Path, words: Words, pin: str, key: str, under: str, texts: dict[str, str]) -> str | None:
    """The JSON pin at `pin` after the move: each Lake manifest it records under `key` whose digest it held, and which the
    move changes, gets the manifest's new digest. None when no digest changes."""
    before = (root / pin).read_text()
    text = new = texts.get(pin, before)
    for rel, digest in json.loads(before).get(key, {}).items():
        target = posixpath.normpath(posixpath.join(under, rel))
        if posixpath.basename(target) != MANIFEST or target not in texts:
            continue
        if hashlib.sha256((root / target).read_bytes()).hexdigest() != digest:
            continue
        written = rel if words.excluded(pin) else words.sub(rel, pin)
        after = hashlib.sha256(texts[target].encode()).hexdigest()
        new = re.sub(rf'("{re.escape(written)}"\s*:\s*"){digest}"', lambda m, h=after: f'{m[1]}{h}"', new)
    return None if new == text else new


def renamed_paths(moves: list[tuple[str, str]]) -> dict[str, str]:
    """Each moved path up to the end of each of its runs of renamed parts ({old: new}; a directory ends in "/"), so that
    every file under one renamed directory gives the same entries, a path below it that renames another part
    (`warden/tests/test_warden.py`) is recorded besides, since `move_check.py` and research's `tools_registry` read the
    longest entry that holds a path, and a swap (`Properties/Proofs/` to `Proofs/Properties/`) is recorded whole."""
    out: dict[str, str] = {}
    for o, n in moves:
        a, b = o.split("/"), n.split("/")
        spans = [(o, n)] if len(a) != len(b) else []
        i = 0 if len(a) == len(b) else len(a)
        while i < len(a):
            if a[i] != b[i]:
                while i + 1 < len(a) and a[i + 1] != b[i + 1]:
                    i += 1
                end = "/" if i < len(a) - 1 else ""
                spans.append(("/".join(a[:i + 1]) + end, "/".join(b[:i + 1]) + end))
            i += 1
        for old, new in spans:
            if out.setdefault(old, new) != new:
                raise SystemExit(f"{old} would move to both {out[old]} and {new}")
    return dict(sorted(out.items()))


def record(root: Path, moves: list[tuple[str, str]]) -> str | None:
    """research's `moves.json` with this move's `renamed_paths` appended, or None when it holds them already or nothing
    moves."""
    if not moves:
        return None
    if not (root / MOVES).is_file():
        raise SystemExit(f"{MOVES}: research's record of the moves is missing")
    doc, entry = json.loads((root / MOVES).read_text()), {"paths": renamed_paths(moves)}
    if entry in doc["moves"]:
        return None
    doc["moves"].append(entry)
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", type=Path, required=True, help="the word map (`tests/fixtures/rename_map.toml` is an example)")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--moves", action="store_true", help="print the declaration renames this tree's locks need, for the Lean check's `--update --moved`")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    root = (a.root or Path(L.git(HERE, "rev-parse", "--show-toplevel").strip())).resolve()
    words = Words(tomllib.loads(a.map.read_text()))
    files = [p for p in L.git(root, "ls-files", "-z").split("\0") if p]
    if a.moves:
        print(json.dumps(lock_moves(root, words, files), indent=1))
        return 0
    if not a.dry_run and not a.allow_dirty and L.git(root, "status", "--porcelain").strip():
        raise SystemExit(f"{root}: the worktree isn't clean")
    run = L.Run(root, L.Layout({"roots": {"old": {}, "new": {}}, "paths": {}}), a.dry_run)
    run.old_of = {}
    present = [p for p in files if os.path.lexists(root / p)]
    moves = [(p, n) for p in present if not words.excluded(p) and (n := words.path(p)) != p]
    dest = dict(moves)
    landing = Counter(dest.get(p, p) for p in present)
    if clash := sorted(n for n, c in landing.items() if c > 1):
        raise SystemExit("two files would land at one path:\n  " + "\n  ".join(clash[:50]))
    folded = Counter(dest.get(p, p).lower() for p in present)
    if fold := sorted(n for n in dest.values() if folded[n.lower()] > 1):
        raise SystemExit("paths that would differ only in case:\n  " + "\n  ".join(fold[:50]))

    counts, kept, texts = Counter(), Counter(), {}
    for p in present:
        if words.excluded(p) or os.path.islink(root / p):
            continue
        lock = posixpath.basename(p) == POLICY
        text = (root / p).read_text() if lock else run._read(p)      # a lock is larger than `layout`'s text limit
        if text is None:
            continue
        keep = record_spans(text) if lock else []
        here = Counter()
        new = words.sub(text, p, keep, counts, here)
        kept.update({(p, w): n for w, n in here.items()})
        if new != text:
            texts[p] = new
    pins = {p: t for p in present if posixpath.basename(p) == PINS and (t := repin(root, words, p, texts, dest)) is not None}
    tracked = set(present)
    pins |= {p: t for p, key, under in json_pins(root)
             if p in tracked and (t := repin_manifests(root, words, p, key, under, texts)) is not None}
    recorded = {MOVES: t} if (t := record(root, moves)) is not None else {}

    print(f"rename move at {L.git(root, 'rev-parse', 'HEAD').strip()}{' (dry run)' if a.dry_run else ''}")
    print(f"  files rewritten: {len(texts)}, files moved: {len(moves)}, pins repinned: {len(pins)}")
    if recorded:
        print(f"  {MOVES}: the renamed paths recorded, {len(renamed_paths(moves))} of them")
    for (w, new), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {n:5d}  {w!r} -> {new!r}")
    if kept:
        print(f"  left standing ([keep], [keep_in] or a lock's records): {sum(kept.values())}")
        for (p, w), n in sorted(kept.items()):
            print(f"    {n:5d}  {w!r} in {p}")
    if a.dry_run:
        for o, n in moves:
            print(f"    move {o} -> {n}")
        for p in sorted(pins):
            print(f"    repin {p}")
        for o, n in renamed_paths(moves).items() if recorded else ():
            print(f"    record {o} -> {n}")
        return 0
    for p, t in {**texts, **pins, **recorded}.items():
        with open(root / p, "w", encoding="utf-8", newline="") as f:
            f.write(t)
    run.move(moves)
    L.git(root, "add", "-A")
    if moves:
        subprocess.run(["git", "--literal-pathspecs", "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul"],
                       cwd=root, input="\0".join(n for _, n in moves).encode(), check=True, capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
