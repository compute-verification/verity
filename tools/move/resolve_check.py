#!/usr/bin/env python3
"""`tools/move/resolve_check.py PARENT COMMIT [--no-imports] [--json] [--root DIR]`: whether a move left every Lean name
and import resolving to what it resolved to before, renamed as the move renames it.

A word map's `[[words_in]]` row renames a word only in the files it lists. Where the word is a Lean declaration or a
module and a file outside the row spells it too, the move renames the declaration and leaves that file naming nothing.
Such a move is still a pure move: `move_check.py` judges each file's text against the map, never what a name resolves
to, and the file is found only when it is built. This check reads the two commits and reports each such name. It reads
source text and elaborates nothing, so it reports some names that are fine and misses others (below): an issue is where
to look before the build, and the Lean build stays the authority. Exit 0 when it finds none, 1 naming each (`--json`:
as JSON).

What it reads, from git's objects alone (no checkout, no build, no Lean):
- every tracked `.lean` file at PARENT and at COMMIT, a file paired with itself across git's rename detection, which
  runs with no limit on the number of files (a file the move renamed and changed past git's similarity threshold is
  unpaired, and so an issue), and each `lakefile.toml`'s `srcDir`, from which a file's module name follows;
- in a file, outside comments and strings: its imports; where they start a line, its `namespace`, `section`, `mutual`,
  `end` and `open` commands and the name each `def`, `theorem`, `lemma`, `abbrev`, `structure`, `inductive`, `class`,
  `instance`, `opaque` or `axiom` declares; and every other identifier.

Its scopes are Lean's. `namespace A.B` and `section A.B` open one scope a component; a `section` that names none and a
`mutual` open one with no name. `end A.B` closes one scope for each component it names and `end` alone closes one with
no name, so the `end` of a `section` or a `mutual` takes nothing of the namespace around it. An `end` that doesn't name
what the scan holds open closes nothing.

It resolves an identifier in the order Lean resolves a global name: under the innermost enclosing namespace that declares
it, then as written, then under an opened namespace, then with trailing components dropped as projections; a declaration
counts where the file's imports reach its file (`--no-imports`: anywhere in the tree, which reports more). A file's
declarations are paired between the commits in order, which gives each name's new name. An issue is then:
- an import, each file's being paired in order, that resolved to a file at PARENT and at COMMIT doesn't resolve to that
  file where the move put it: `FILE:LINE: import OLD -> NEW: want FILE, got FILE`;
- in a file the move changed or that spells a component of a name only one commit declares, an identifier, the file's
  being paired in order, whose resolution at COMMIT is not its resolution at PARENT under its new name: `FILE:LINE:
  OLD -> NEW in NAMESPACE: want NAME, got NAME`;
- in such a file, an `end` that closed nothing: the scan has lost the file's namespaces, and its names with them;
- a Lean file at one commit with none paired to it at the other, and a file whose imports, declarations or identifiers
  can't be paired in order.

What it cannot see:
- anything Lean elaborates. A local (a binder, a `variable`, a pattern's name) is read as a global of its spelling, so a
  local that spells a renamed declaration can be reported where nothing is wrong;
- a name no such line declares: a structure's field, a constructor, a `where` function, an indented declaration, what
  `deriving`, a macro or an attribute makes, an `export` or an alias. A rename of one of those that misses a file is
  not reported;
- a scope command that doesn't start its line, an `open` inside a term or a tactic block, and what `hiding`, `renaming`
  or a parenthesised list leaves out of one;
- a declaration or a module outside the tree (Lean's, Mathlib's): a name or an import that resolves to one resolves to
  nothing at both commits;
- two declarations of one full name in two packages, which are one name to it;
- a name in a comment, a docstring or a string, and every file that is not Lean. `move_check.py` and the map hold those.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
IDENT = r"(?:[^\W\d]|_)[\w'!?\u2080-\u209c]*"
NAME = rf"{IDENT}(?:\.{IDENT})*"
TOKEN = re.compile(rf"(?<![\w.'!?]){NAME}")
DECL = re.compile(
    r"^(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|public|noncomputable|partial|unsafe|nonrec|scoped|local|meta)\s+)*"
    rf"(def|theorem|lemma|abbrev|structure|inductive|class(?:\s+inductive)?|instance|opaque|axiom)\s+({NAME})", re.M)
SCOPE = re.compile(
    rf"^(?:@\[[^\]]*\]\s*)?(?:(?:public|noncomputable|meta)\s+)*(namespace|section|mutual|end)\b\s*({NAME})?$")
IMPORT = re.compile(rf"^(?:public\s+)?(?:meta\s+)?import\s+(?:all\s+)?({NAME})", re.M)
KEYWORDS = frozenset("""def theorem lemma abbrev structure inductive class instance opaque axiom where with fun let in if then
else match do have show from by at only for return namespace section end open import public private protected module
noncomputable partial unsafe variable universe set_option attribute local scoped macro syntax notation infix infixl
infixr prefix postfix elab deriving extends example calc mut termination_by decreasing_by Type Prop Sort true false""".split())


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def texts(root: Path, commit: str, paths: list[str]) -> dict[str, str]:
    """Each of ``paths``' text at ``commit``, read through one `git cat-file --batch`."""
    out: dict[str, str] = {}
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p:
        for path in paths:
            p.stdin.write(f"{commit}:{path}\n".encode())
            p.stdin.flush()
            header = p.stdout.readline().split()
            if header[-1] == b"missing":
                continue
            out[path] = p.stdout.read(int(header[-1])).decode()
            p.stdout.read(1)
        p.stdin.close()
    return out


def lean_files(root: Path, commit: str) -> list[str]:
    return [p for p in git(root, "ls-tree", "-r", "-z", "--name-only", commit).split("\0") if p.endswith(".lean")]


def changes(root: Path, base: str, head: str) -> tuple[dict[str, str], set[str]]:
    """What ``head`` changed of ``base``: {a renamed file's old path: its new one}, and every changed file's path at ``head``."""
    renames: dict[str, str] = {}
    changed: set[str] = set()
    # -l0: past its default limit git skips the inexact detection, and a file renamed and changed is one deleted, one added
    fields = iter(git(root, "diff", "-M", "-l0", "-z", "--name-status", base, head).split("\0"))
    for status in fields:
        if not status:
            continue
        path = next(fields)
        if status[0] in "RC":
            old, path = path, next(fields)
            if status[0] == "R":
                renames[old] = path
        changed.add(path)
    return renames, changed


def strip(text: str) -> str:
    """``text`` with its comments and strings blanked, every newline kept."""
    out, i, n = [], 0, len(text)
    depth = 0
    while i < n:
        c = text[i]
        if depth:
            if text.startswith("/-", i):
                depth += 1
                out.append("  ")
                i += 2
            elif text.startswith("-/", i):
                depth -= 1
                out.append("  ")
                i += 2
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
        elif text.startswith("/-", i):
            depth = 1
            out.append("  ")
            i += 2
        elif text.startswith("--", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append("".join("\n" if ch == "\n" else " " for ch in text[i:j + 1]))
            i = j + 1
        elif c == "'" and i + 2 < n and (text[i + 2] == "'" or (text[i + 1] == "\\" and i + 3 < n and text[i + 3] == "'")) \
                and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_'!?")):
            j = text.find("'", i + 1 if text[i + 1] != "\\" else i + 3)
            out.append(" " * (j - i + 1))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


class Scan:
    """One file: its imports as (line, module), the full names it declares and the line of each `end` that closed nothing;
    with ``tokens``, each identifier as (line, text, context, whether it is the name a declaration declares), a context
    being the namespace and the opens in force there; without, the words it spells."""

    def __init__(self, text: str, tokens: bool = False):
        code = strip(text)
        self.imports = [(code.count("\n", 0, m.start()) + 1, m.group(1)) for m in IMPORT.finditer(code)]
        self.decls: list[str] = []
        self.stray: list[int] = []
        self.ctxs: list[tuple[tuple[str, ...], tuple]] = []
        self.tokens: list[tuple[int, str, int, bool]] = []
        self._scan(code, tokens)
        self.words = None if tokens else set(re.findall(IDENT, code))

    def _scan(self, code: str, tokens: bool) -> None:
        # each open scope, one a name component as Lean opens them: its name (None: a `section` that names none, a
        # `mutual`), whether it is a namespace, and how many opens were made in it
        scopes: list[tuple[str | None, bool, int]] = []
        ns: list[str] = []
        opens: list[tuple[tuple[str, ...], str]] = []   # each open: the namespace it stands in and the name it opens
        for ln, line in enumerate(code.split("\n")):
            s, top = line.strip(), line[:1] not in (" ", "\t")
            m = SCOPE.match(s)
            if m and (top or m.group(1) == "end"):
                kind, parts = m.group(1), m.group(2).split(".") if m.group(2) else [None]
                if kind == "namespace" and parts != [None]:
                    scopes.extend((part, True, 0) for part in parts)
                    ns.extend(parts)
                elif kind == "section":
                    scopes.extend((part, False, 0) for part in parts)
                elif kind == "mutual":
                    scopes.append((None, False, 0))
                elif kind == "end" and top:
                    if [name for name, _, _ in scopes[len(scopes) - len(parts):]] != parts:
                        self.stray.append(ln + 1)   # it names no scope this scan saw open: it closes none
                        continue
                    for _ in parts:
                        _, namespace, added = scopes.pop()
                        if namespace:
                            ns.pop()
                        if added:
                            del opens[len(opens) - added:]
                continue
            mo = re.match(r"^open\s+(.*)$", s)
            if mo and top:
                body = re.sub(r"\([^)]*\)", " ", re.sub(r"\bin\s*$", "", mo.group(1)))
                body = re.sub(r"\b(?:scoped|private)\b", " ", re.split(r"\b(?:hiding|renaming)\b", body)[0])
                names = re.findall(NAME, body)
                opens.extend((tuple(ns), nm) for nm in names)
                if scopes:
                    scopes[-1] = (*scopes[-1][:2], scopes[-1][2] + len(names))
                continue
            if re.match(r"^(?:public\s+)?(?:meta\s+)?import\b", s):
                continue
            dm = DECL.match(line)
            if dm:
                name = dm.group(2)
                self.decls.append(name.removeprefix("_root_.") if name.startswith("_root_.") else ".".join([*ns, name]))
            if not tokens:
                continue
            ctx = (tuple(ns), tuple(opens))
            if not self.ctxs or self.ctxs[-1] != ctx:
                self.ctxs.append(ctx)
            for tm in TOKEN.finditer(line):
                if tm.group(0).split(".")[0] not in KEYWORDS:
                    self.tokens.append((ln, tm.group(0), len(self.ctxs) - 1, dm is not None and tm.span() == dm.span(2)))


class Tree:
    """A commit's Lean files: which file each module is, what each file's imports reach, and which files declare a name."""

    def __init__(self, root: Path, commit: str, everywhere: bool):
        self.everywhere = everywhere
        self.scans = {p: Scan(t) for p, t in texts(root, commit, lean_files(root, commit)).items()}
        lakefiles = [p for p in git(root, "ls-tree", "-r", "-z", "--name-only", commit).split("\0")
                     if p.endswith(("lakefile.toml", "lakefile.lean")) and "/fixtures/" not in p]
        roots = sorted({os.path.dirname(p) for p in lakefiles}, key=len, reverse=True)
        sources = set()         # each package's source directory, below which a file's path is its module's name
        for p, text in texts(root, commit, [p for p in lakefiles if p.endswith(".toml")]).items():
            m = re.search(r'^srcDir\s*=\s*"([^"]*)"', text.split("\n[")[0], re.M)
            src = os.path.normpath(os.path.join(os.path.dirname(p) or ".", m.group(1) if m else "."))
            sources.add("" if src == "." else src)
        sources = sorted(sources, key=len, reverse=True)
        self.module_of: dict[str, list[str]] = defaultdict(list)
        self.root_of: dict[str, str] = {}
        for p in self.scans:
            self.root_of[p] = next((r for r in roots if p.startswith(r + "/") or r == ""), "")
            src = next((r for r in sources if r == "" or p.startswith(r + "/")), "")
            self.module_of[(p[len(src) + 1:] if src else p)[:-len(".lean")].replace("/", ".")].append(p)
        self._reach: dict[str, frozenset[str]] = {}
        self.namespaces: set[str] = set()
        self.decl_files: dict[str, set[str]] = defaultdict(set)
        for p, scan in self.scans.items():
            for d in scan.decls:
                self.decl_files[d].add(p)
                parts = d.split(".")
                self.namespaces.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))

    def module(self, p: str, name: str) -> list[str]:
        """The files ``p``'s import of the module ``name`` reads (none: no file of the tree is that module): a module two
        packages define is the one of ``p``'s own package where there is one."""
        files = self.module_of.get(name, [])
        if len(files) > 1:
            files = [f for f in files if self.root_of[f] == self.root_of[p]] or files
        return files

    def imported(self, p: str) -> list[str]:
        """The files ``p`` imports."""
        return [f for _, name in self.scans[p].imports for f in self.module(p, name)]

    def reach(self, p: str) -> frozenset[str]:
        """``p`` and every file its imports reach."""
        if p not in self._reach:
            seen, stack = {p}, [p]
            while stack:
                for r in self.imported(stack.pop()):
                    if r not in seen:
                        seen.add(r)
                        stack.append(r)
            self._reach[p] = frozenset(seen)
        return self._reach[p]

    def visible(self, name: str, files: frozenset[str]) -> bool:
        declared = self.decl_files.get(name)
        return bool(declared) and (self.everywhere or not declared.isdisjoint(files))

    def resolve(self, t: str, ns: tuple[str, ...], opens: tuple, files: frozenset[str]) -> tuple[str | None, tuple[str, ...]]:
        """The declaration the identifier ``t`` names where ``ns`` and ``opens`` are in force and ``files`` are imported
        (several under opened namespaces: joined by `|`), and the components left over as projections; (None, ()) for none."""
        if t.startswith("_root_."):
            t, ns, opens = t[len("_root_."):], (), ()
        opened = []
        for at, nm in opens:
            for i in range(len(at), -1, -1):
                if (cand := ".".join([*at[:i], nm])) in self.namespaces:
                    opened.append(cand)
                    break
        parts, projections = t.split("."), []
        while parts:
            ident = ".".join(parts)
            for i in range(len(ns), 0, -1):
                if self.visible(cand := ".".join([*ns[:i], ident]), files):
                    return cand, tuple(projections)
            if self.visible(ident, files):
                return ident, tuple(projections)
            if hits := sorted({f"{o}.{ident}" for o in opened if self.visible(f"{o}.{ident}", files)}):
                return "|".join(hits), tuple(projections)
            projections.insert(0, parts.pop())
        return None, ()


def check(root: Path, base: str, head: str, everywhere: bool = False) -> tuple[list[dict], int, int]:
    """The issues of the move ``base``..``head``, how many files it compared identifier by identifier, and of how many."""
    renames, changed = changes(root, base, head)
    B, H = Tree(root, base, everywhere), Tree(root, head, everywhere)
    issues: list[dict] = []
    new_name: dict[str, str] = {}
    paired = {bp: hp for bp in B.scans if (hp := renames.get(bp, bp)) in H.scans}
    issues += [{"file": bp, "kind": "file", "at": "parent"} for bp in B.scans if bp not in paired]
    issues += [{"file": hp, "kind": "file", "at": "commit"} for hp in H.scans.keys() - set(paired.values())]
    for bp, hp in paired.items():
        bs, hs = B.scans[bp], H.scans[hp]
        if len(bs.imports) != len(hs.imports):
            issues.append({"file": hp, "kind": "imports", "base": len(bs.imports), "head": len(hs.imports)})
        else:
            for (_, was), (line, now) in zip(bs.imports, hs.imports):
                want, got = sorted(renames.get(f, f) for f in B.module(bp, was)), sorted(H.module(hp, now))
                if want and got != want:
                    issues.append({"file": hp, "line": line, "kind": "import", "base_import": was, "head_import": now,
                                   "want": want, "got": got, "changed_file": hp in changed})
        if len(bs.decls) != len(hs.decls):
            issues.append({"file": hp, "kind": "declarations", "base": len(bs.decls), "head": len(hs.decls)})
            continue
        new_name.update(zip(bs.decls, hs.decls))

    def moved(name: str | None) -> str | None:
        return None if name is None else "|".join(sorted(new_name.get(x, x) for x in name.split("|")))

    # only a file the move changed, or one that spells a component of a name only one commit declares, can resolve otherwise
    one_side = {d for s in B.scans.values() for d in s.decls} ^ {d for s in H.scans.values() for d in s.decls}
    parts = {c for d in one_side for c in d.split(".")}
    chosen = sorted(bp for bp, hp in paired.items() if hp in changed or not parts.isdisjoint(H.scans[hp].words))
    btexts = texts(root, base, chosen)
    htexts = texts(root, head, [paired[bp] for bp in chosen])
    for bp in chosen:
        hp = paired[bp]
        bs, hs = Scan(btexts.pop(bp), tokens=True), Scan(htexts.pop(hp), tokens=True)
        issues += [{"file": hp, "line": line, "kind": "end"} for line in hs.stray]
        if len(bs.tokens) != len(hs.tokens):
            issues.append({"file": hp, "kind": "identifiers", "base": len(bs.tokens), "head": len(hs.tokens)})
            continue
        bfiles, hfiles = B.reach(bp), H.reach(hp)
        for (_, bt, bci, bdecl), (hl, ht, hci, hdecl) in zip(bs.tokens, hs.tokens):
            if bdecl or hdecl:
                continue
            was = B.resolve(bt, *bs.ctxs[bci], bfiles)
            now = H.resolve(ht, *hs.ctxs[hci], hfiles)
            want = (moved(was[0]), tuple(ht.split(".")[len(ht.split(".")) - len(was[1]):]) if was[1] else ())
            if now != want:
                issues.append({"file": hp, "line": hl + 1, "base_tok": bt, "head_tok": ht, "ns": ".".join(hs.ctxs[hci][0]),
                               "base": was[0], "want": want[0], "got": now[0], "projs": list(now[1]),
                               "changed_file": hp in changed})
    issues.sort(key=lambda issue: (issue["file"], issue.get("line", 0)))
    return issues, len(chosen), len(B.scans)


def said(issue: dict) -> str:
    kind, where = issue.get("kind"), f"{issue['file']}:{issue.get('line')}"
    unchanged = "" if issue.get("changed_file", True) else " (unchanged file)"
    if kind == "file":
        other = "commit" if issue["at"] == "parent" else "parent"
        return f"{issue['file']}: a Lean file at the {issue['at']} with none paired to it at the {other}"
    if kind == "end":
        return f"{where}: an `end` that closes no scope the scan holds open: it has lost this file's namespaces"
    if kind == "import":
        return (f"{where}: import {issue['base_import']} -> {issue['head_import']}: want {' '.join(issue['want'])}, "
                f"got {' '.join(issue['got']) or None}{unchanged}")
    if kind:
        return f"{issue['file']}: {issue['base']} {kind} at the parent, {issue['head']} at the commit: they can't be paired"
    return (f"{where}: {issue['base_tok']} -> {issue['head_tok']} in {issue['ns'] or '<root>'}: "
            f"want {issue['want']}, got {issue['got']}{unchanged}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", metavar="PARENT", help="the commit the move ran on")
    ap.add_argument("head", metavar="COMMIT", help="the move's commit")
    ap.add_argument("--no-imports", action="store_true",
                    help="count a declaration wherever the tree declares it, not only where the file's imports reach it")
    ap.add_argument("--json", action="store_true", help="print the issues as JSON, and nothing else")
    ap.add_argument("--root", type=Path, default=None, help="the repository (default: this file's)")
    a = ap.parse_args(argv)
    root = (a.root or Path(git(HERE, "rev-parse", "--show-toplevel").strip())).resolve()
    issues, compared, files = check(root, a.base, a.head, a.no_imports)
    if a.json:
        print(json.dumps(issues, indent=1))
        return 1 if issues else 0
    print(f"{compared} of {files} files compared identifier by identifier")
    for issue in issues:
        print(said(issue))
    print(f"{len(issues)} issue(s)")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
