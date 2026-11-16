"""Counts of exact spellings in the tracked text files of a ref.

A site of a spelling is the spelling itself, case kept, with no letter, digit or underscore touching it on a side where
the spelling begins or ends with one: `windowOf` is not a site of `window`, `window of` is not a site of `windowOf`.
`substring=True` is top's earlier reading (`str.count`, any touching allowed), kept to reproduce top_new_side.tsv.
Files left out: under tools/move/, tools/sweepgap/, GLOSSARY.md, binary (a NUL in the first 8000 bytes), over 50 MB.
"""
from __future__ import annotations

import re
import subprocess
from collections import Counter, defaultdict

SKIP = ("tools/move/", "tools/sweepgap/")
TOK = re.compile(r"[A-Za-z0-9_]+")
PLAIN = re.compile(r"[A-Za-z0-9_]+")


class Tree:
    def __init__(self, ref: str, checkout: str = "."):
        self.ref, self.checkout = ref, checkout
        out = subprocess.run(["git", "-C", checkout, "ls-tree", "-r", "-z", "--long", ref], capture_output=True).stdout
        self.files: list[tuple[str, str, int]] = []
        for ent in out.split(b"\0"):
            if not ent:
                continue
            meta, path = ent.split(b"\t", 1)
            mode, typ, oid, size = meta.split()
            path = path.decode("utf-8", "replace")
            if typ != b"blob" or mode == b"160000" or path == "GLOSSARY.md" or path.startswith(SKIP):
                continue
            if int(size) > 50_000_000:
                continue
            self.files.append((path, oid.decode(), int(size)))
        self._p = None

    def _proc(self):
        if self._p is None:
            self._p = subprocess.Popen(["git", "-C", self.checkout, "cat-file", "--batch"], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE)
        return self._p

    def read(self, oid: str) -> str | None:
        p = self._proc()
        p.stdin.write(oid.encode() + b"\n")
        p.stdin.flush()
        head = p.stdout.readline().split()
        n = int(head[2])
        data = p.stdout.read(n)
        p.stdout.read(1)
        if b"\0" in data[:8000]:
            return None
        return data.decode("utf-8", "replace")

    def texts(self):
        for path, oid, _ in self.files:
            t = self.read(oid)
            if t is not None:
                yield path, oid, t


def rx(s: str) -> re.Pattern:
    left = r"(?<![A-Za-z0-9_])" if re.match(r"\w", s[0]) else ""
    right = r"(?![A-Za-z0-9_])" if re.match(r"\w", s[-1]) else ""
    return re.compile(left + re.escape(s) + right)


def count(tree: Tree, spellings, substring: bool = False, only_files: dict[str, set[str] | None] | None = None):
    """{spelling: Counter(path -> sites)} for the spellings. `only_files[s]`, when given and not None, limits s to those paths."""
    spellings = sorted(set(spellings))
    out: dict[str, Counter] = {s: Counter() for s in spellings}
    if substring:
        for path, _, t in tree.texts():
            for s in spellings:
                if s in t:
                    out[s][path] = t.count(s)
        return out
    plain = {s for s in spellings if PLAIN.fullmatch(s)}
    phrase = [s for s in spellings if s not in plain]
    pieces = {s: TOK.findall(s) for s in phrase}
    need = {p for ps in pieces.values() for p in ps}
    have: dict[str, set[str]] = defaultdict(set)
    oid_of = {}
    for path, oid, t in tree.texts():
        oid_of[path] = oid
        c = Counter(TOK.findall(t))
        for tok, n in c.items():
            if tok in plain:
                out[tok][path] = n
            if tok in need:
                have[tok].add(path)
    for s in phrase:
        ps = pieces[s]
        cand = set.intersection(*(have.get(p, set()) for p in ps)) if ps else set()
        if only_files and only_files.get(s) is not None:
            cand &= only_files[s]
        r = rx(s)
        for path in sorted(cand):
            t = tree.read(oid_of[path])
            n = len(r.findall(t)) if t else 0
            if n:
                out[s][path] = n
    if only_files:
        for s in plain:
            f = only_files.get(s)
            if f is not None:
                out[s] = Counter({p: n for p, n in out[s].items() if p in f})
    return out


def sites(tree: Tree, spelling: str, paths):
    """[(path, line number, line text)] of the exact spelling in the given paths."""
    r = rx(spelling)
    oid_of = {p: o for p, o, _ in tree.files}
    res = []
    for path in sorted(paths):
        t = tree.read(oid_of[path])
        if not t:
            continue
        for i, line in enumerate(t.split("\n"), 1):
            if r.search(line):
                res.append((path, i, line))
    return res
