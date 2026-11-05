#!/usr/bin/env python3
"""Apply a split map's operations, in order, to a tree: move declarations out of a Lean module into a new one
(kind = "split"), or a module's declarations to the top of another (kind = "merge"), and point the importers at the result.

    python3 leansplit.py ROOT --map MAP.toml [--first N | --ops 0,3,7] [--dry-run] [--check-roundtrip]

A split: `source` and `new` (module names), `decls` (full declaration names, auxiliaries allowed), `imports` (the in-tree
modules whose constants they use), `source_imports_new` (the source's remaining declarations use them), `importers` (the
other modules using one of them, which import `new` beside the source's import, in its form), `drop_source` (importers
that use nothing else of the source and stop importing it) and `uses` (for each of those, the in-tree modules its
constants use: each one it no longer reaches is imported directly). A merge: `module`, `into`. After every operation, each
module of the map's `[repair]` (module = [modules its constants use]) imports those its imports no longer reach, and each
module of `root`'s closure the tactic and notation modules its imports reached before and don't now, whose atoms its text
holds; a module outside it, which the constant graph doesn't cover, imports back all it reached before.

A moved declaration takes its docstring, attributes, `set_option … in` and `open … in` prefixes and the comments directly
above it, and the new module replays the sections, namespaces, `open`s, `variable`s and global `set_option`s in force
where it stood. The new module imports `imports`, the source's untracked imports (Mathlib, Std, ArkLib), and each in-tree
module of the source's closure defining a tactic or notation whose atom a moved declaration's text holds; it keeps the
source's header style (`module` and `public import` under the module system). The new module, and each importer that stops
importing the source, leaves out of its `open` lines the in-tree namespaces none of its imports declares any more, and the
new module the `local` notations its declarations don't use (`prune`). Text-level and refusing: a declaration it
can't place (an unnamed instance, a name it doesn't find, one inside `mutual`) stops it with the name, before anything is
written. It refuses a dirty worktree, and a second run changes nothing.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

AUX = re.compile(r"^(_.*|proof_\d+|match_\d+(_\d+)?|eq_\d+|eq_def|induct|induct_unfolding|fun_cases|mutual_induct|rec|"
                 r"recOn|casesOn|noConfusion|noConfusionType|below|brecOn|binductionOn|ibelow|mk|sizeOf_spec|inj|injEq|"
                 r"congr_simp|ext|ext_iff|spec_\d+|unfold|hcongr_\d+|splitter|else_eq|eq_iff|sizeOf|ctorIdx|toCtorIdx|"
                 r"ofNat|proof_\d+_\d+|inj_\d+)$")
ATTRS = r"(?:@\[[^\]]*\]\s*)*"
MODS = r"(?:(?:private|protected|noncomputable|partial|unsafe|nonrec|public|meta)\s+)*"
DECL = re.compile("^" + ATTRS + MODS + r"(theorem|lemma|def|abbrev|instance|structure|class|inductive|opaque|axiom|"
                  r"irreducible_def|example)\b(?:\s*\(priority\s*:=\s*[^)]*\))?\s*([^\s:({\[⦃]+)?")
OPENER = re.compile("^" + ATTRS + r"(?:(?:noncomputable|public)\s+)*(namespace|section|mutual)\b[ \t]*(\S*)")
END = re.compile(r"^end\b[ \t]*(\S*)")
CONTEXT = re.compile(r"^(open|variable|universe|set_option|attribute\s*\[local|local\s|export|include|omit)\b")
IN_PREFIX = re.compile(r"^(set_option|open)\s.*\sin\s*$")
CONT = re.compile(r"^(deriving(?!\s+instance)|where|with|termination_by|decreasing_by)\b")
IMPORT = re.compile(r"^(public\s+)?(meta\s+)?import\s+(all\s+)?(\S+)")
SYNTAX_DEF = re.compile(r'^(?:scoped |local )?(?:elab|macro|syntax|notation|infix[lr]?|prefix|postfix)\b[^"\n]*"([^"]+)"',
                        re.M)
OPEN = re.compile(r"^open(\s+scoped)?((?:[ \t]+[^\s()]+)+)[ \t]*$")
OPEN_ONLY = re.compile(r"^open[ \t]+([^\s()]+)[ \t]*\(([^()]*)\)[ \t]*$")
LOCAL_SYNTAX = re.compile(r"^local\s+(?:elab|macro|syntax|notation|infix[lr]?|prefix|postfix)\b")
TOKEN = re.compile(r"[\w'!?]+")
BRACKETS = {"(": ")", "{": "}", "[": "]", "⦃": "⦄"}


def nl_lines(text: str) -> list[str]:
    """The lines of `text`, split at `\\n` only: Lean source may hold other line separators inside a line."""
    return [ln for ln in re.split(r"(?<=\n)", text) if ln]


def owner(n: str) -> str:
    parts = n.split(".")
    while len(parts) > 1 and AUX.match(parts[-1]):
        parts.pop()
    return ".".join(parts)


def top_lines(text: str) -> list[bool]:
    """For each line, whether it starts outside every comment and string."""
    out, depth, in_str, i, n, bol = [], 0, False, 0, len(text), True
    while True:
        if bol:
            out.append(depth == 0 and not in_str)
            bol = False
        if i >= n:
            return out
        c = text[i]
        if c == "\n":
            bol, i = True, i + 1
        elif in_str:
            i += 2 if c == "\\" and text[i + 1:i + 2] != "\n" else 1
            in_str = c != '"' if c != "\\" else in_str
        elif depth:
            if text.startswith("/-", i):
                depth, i = depth + 1, i + 2
            elif text.startswith("-/", i):
                depth, i = depth - 1, i + 2
            else:
                i += 1
        elif text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/-", i):
            depth, i = 1, i + 2
        elif c == "'" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_'!?.₀₁₂₃₄₅₆₇₈₉")):
            close = text.find("'", i + 3) if text[i + 1:i + 2] == "\\" else (i + 2 if text[i + 2:i + 3] == "'" else -1)
            i = close + 1 if 0 < close - i <= 10 and "\n" not in text[i:close] else i + 1
        else:
            in_str = c == '"'
            i += 1


class Unit:
    """A top-level command's lines, the blank lines after it, and for a declaration its prefixes, name and context."""

    def __init__(self, lines: list[str], start: int):
        self.start = start
        k = len(lines)
        while k > 1 and not lines[k - 1].strip():
            k -= 1
        self.lines, self.blank = lines[:k], lines[k:]
        self.head = self.lines[0].strip()
        self.prefix: list[Unit] = []
        self.name: str | None = None
        self.decl = False
        self.in_mutual = False
        self.ctx: tuple = ()

    def own(self) -> str:
        return "".join(self.lines)

    def text(self) -> str:
        return "".join(p.own() + "".join(p.blank) for p in self.prefix) + self.own()

    def full(self) -> str:
        return self.text() + "".join(self.blank)


def split_header(text: str) -> tuple[str, str]:
    """The header (`module`, imports, and the comments and blank lines among them) and the rest."""
    lines = nl_lines(text)
    top = top_lines(text)
    h = 0
    for k, ln in enumerate(lines):
        s = ln.strip()
        if top[k] and (s in ("module", "prelude") or IMPORT.match(s)):
            h = k + 1
        elif top[k] and s and not s.startswith("--") and not (s.startswith("/-") and s[2:3] not in ("-", "!")):
            break
    return "".join(lines[:h]), "".join(lines[h:])


def units_of(body: str) -> list[Unit]:
    lines = nl_lines(body)
    top = top_lines(body)
    out, cur, start = [], [], 0
    for k, ln in enumerate(lines):
        cont = not top[k] or not ln.strip() or ln[:1] in " \t|)]}·" or CONT.match(ln)
        if not cont and ln.startswith("--"):
            # a comment inside a command's body, when the next line of code is indented
            j = next((j for j in range(k + 1, len(lines)) if lines[j].strip() and not lines[j].startswith("--")), None)
            cont = j is not None and lines[j][:1] in " \t"
        if cont and cur:
            cur.append(ln)
            continue
        if cur:
            out.append(Unit(cur, start))
        cur, start = [ln], k
    if cur:
        out.append(Unit(cur, start))
    return out


def is_prefix(u: Unit) -> bool:
    h = u.head
    if h.startswith("/-!"):
        return False
    if h.startswith("/-") or h.startswith("--") or IN_PREFIX.match(h):
        return True
    return h.startswith("@[") and not DECL.match(h)


def annotate(units: list[Unit]) -> tuple[list[Unit], list[tuple]]:
    """The units with prefixes attached to their declarations, each declaration named and given its context; and the
    scopes still open at the end. A context is a tuple of frames, each (opener line, context line, …), the first
    frame's opener empty."""
    out: list[Unit] = []
    pending: list[Unit] = []
    stack: list[list[str]] = [[""]]
    names: list[list[str]] = [[]]
    for u in units:
        if is_prefix(u) and not u.blank:
            pending.append(u)
            continue
        m = DECL.match(u.head)
        if m:
            u.prefix, pending = pending, []
            u.decl = True
            kind, nm = m.group(1), m.group(2)
            ns = [c for f in names for c in f]
            if nm and kind != "example":
                u.name = nm[len("_root_."):] if nm.startswith("_root_.") else ".".join([*ns, nm])
            u.in_mutual = any(OPENER.match(f[0]) and OPENER.match(f[0]).group(1) == "mutual" for f in stack[1:])
            u.ctx = tuple(tuple(f) for f in stack)
            out.append(u)
            continue
        out.extend(pending)
        pending = []
        o, e = OPENER.match(u.head), END.match(u.head)
        if o and not u.head.startswith("/-"):
            stack.append([u.own()])
            names.append(o.group(2).split(".") if o.group(1) == "namespace" and o.group(2) else [])
        elif e and len(stack) > 1:
            stack.pop()
            names.pop()
        elif CONTEXT.match(u.head):
            stack[-1].append(u.own())
        out.append(u)
    out.extend(pending)
    return out, [tuple(f) for f in stack]


def closer(frame: tuple) -> str | None:
    """The `end` line closing a frame, or None for one that stays open to the end of the file."""
    o = OPENER.match(frame[0].strip())
    if not o or o.group(1) == "mutual":
        return None
    if o.group(1) == "section" and not o.group(2) and "public" in frame[0]:
        return None
    return f"end{(' ' + o.group(2)) if o.group(2) else ''}\n"


def emit(units: list[Unit]) -> str:
    """The units, each under its context: scopes opened, extended and closed between them as their contexts differ."""
    out: list[str] = []
    cur: tuple = (("",),)
    for u in units:
        ctx = u.ctx
        k = 0
        while k < min(len(cur), len(ctx)) and cur[k] == ctx[k]:
            k += 1
        if k == len(cur) - 1 and k < len(ctx) and ctx[k][:len(cur[k])] == cur[k]:
            out.extend(c if c.endswith("\n") else c + "\n" for c in ctx[k][len(cur[k]):])
            k += 1
        else:
            for f in reversed(cur[k:]):
                if (c := closer(f)) is not None:
                    out.append("\n" + c)
        for f in ctx[k:]:
            if f[0]:
                out.append("\n" + f[0])
            out.extend(c if c.endswith("\n") else c + "\n" for c in f[1:])
        cur = ctx
        out.append("\n" + u.text())
    for f in reversed(cur[1:]):
        if (c := closer(f)) is not None:
            out.append("\n" + c)
    return "".join(out)


def imports_of(header: str) -> list[str]:
    return [m.group(4) for ln in nl_lines(header) if (m := IMPORT.match(ln.strip()))]


def module_style(header: str) -> bool:
    return any(ln.strip() == "module" for ln in nl_lines(header))


def add_import(text: str, mod: str, after: str | None) -> str:
    header, body = split_header(text)
    if mod in imports_of(header):
        return text
    line = f"{'public ' if module_style(header) else ''}import {mod}\n"
    hl = nl_lines(header)
    idx = None
    for k, ln in enumerate(hl):
        m = IMPORT.match(ln.strip())
        if m:
            idx = k
            if after and m.group(4) == after:
                line = f"{m.group(1) or ''}{m.group(2) or ''}import {m.group(3) or ''}{mod}\n"
                break
    hl.insert(len(hl) if idx is None else idx + 1, line)
    return "".join(hl) + body


def drop_import(text: str, mod: str) -> str:
    header, body = split_header(text)
    keep = [ln for ln in nl_lines(header) if not ((m := IMPORT.match(ln.strip())) and m.group(4) == mod)]
    return "".join(keep) + body


def flagged(header: str, mod: str | None = None) -> list[str]:
    """The header's `import all` and `meta import` lines (of `mod` alone, given one), as written: a module that takes
    declarations from this one needs them as they are, since a plain import neither exposes a body nor runs at
    compile time."""
    return [ln.strip() for ln in nl_lines(header)
            if (m := IMPORT.match(ln.strip())) and (m.group(2) or m.group(3)) and mod in (None, m.group(4))]


def add_line(text: str, line: str) -> str:
    """`text` with the import `line` after its last import, unless it holds that line already."""
    header, body = split_header(text)
    hl = nl_lines(header)
    if line in (ln.strip() for ln in hl):
        return text
    idx = max((k for k, ln in enumerate(hl) if IMPORT.match(ln.strip())), default=len(hl) - 1)
    hl.insert(idx + 1, line + "\n")
    return "".join(hl) + body


def retarget(line: str, mod: str) -> str:
    m = IMPORT.match(line)
    return f"{m.group(1) or ''}{m.group(2) or ''}import {m.group(3) or ''}{mod}"


def binders(text: str) -> list[str] | None:
    """A `variable` command's binder groups, or None when it holds anything else (a comment, an `in`)."""
    rest, out, k = text[len("variable"):], [], 0
    while k < len(rest):
        if rest[k].isspace():
            k += 1
            continue
        close = BRACKETS.get(rest[k])
        if close is None:
            return None
        stack, j = [close], k + 1
        while stack and j < len(rest):
            c = rest[j]
            if c in BRACKETS:
                stack.append(BRACKETS[c])
            elif c == stack[-1]:
                stack.pop()
            j += 1
        if stack:
            return None
        out.append(rest[k:j])
        k = j
    return out


def kept_binders(groups: list[str], used: set[str]) -> list[bool]:
    """Which of `groups` (every `variable` binder group of a file, in order) a declaration whose text holds the tokens
    `used` can read: an anonymous one (an instance), and a named one whose name it or a kept group's type names, in any
    `variable` command. A named variable is included only where it is named, so the rest change nothing but must still
    elaborate."""
    def names(g):
        head = g[1:-1].split(":", 1)
        return TOKEN.findall(head[0]) if len(head) == 2 and g[0] != "[" else []
    keep = [not names(g) or bool(set(names(g)) & used) for g in groups]
    while True:
        need = {t for g, k in zip(groups, keep) if k for t in TOKEN.findall(g.split(":", 1)[-1])}
        more = [k or bool(set(names(g)) & need) for g, k in zip(groups, keep)]
        if more == keep:
            return keep
        keep = more


def prune(text: str, known: set[str], avail: set[str], used: str | None = None) -> tuple[str, list[str]]:
    """`text` without what it can no longer resolve: each namespace of a top-level `open` line (`open N`, `open N in`,
    `open N (names)`) that is declared in the tree (`known`) but by none of the modules it now imports (`avail`), as
    Lean resolves the name from the namespaces open there and those opened before it; a namespace no in-tree module
    declares (Mathlib's) stays. With `used` (the text of the
    declarations it holds), each named `variable` binder neither `used`, an `include` or `omit` nor a kept binder names
    goes too, and so does a `local` notation or macro whose atom nothing kept holds: not `used`, a kept binder, another
    command or a kept notation. Returns the text and what went."""
    header, body = split_header(text)
    units = units_of(body)
    toks = set(TOKEN.findall(used)) if used is not None else set()
    toks |= {t for u in units if re.match(r"(include|omit)\s", u.head) for t in TOKEN.findall(u.own())}
    var: dict[int, list[str]] = {}
    for k, u in enumerate(units if used is not None else []):
        if re.match(r"variable\s", u.head):
            if (gs := binders(u.own().strip())) is None:
                toks |= set(TOKEN.findall(u.own()))
            else:
                var[k] = gs
    flags = iter(kept_binders([g for k in sorted(var) for g in var[k]], toks))
    kept = {k: [g for g in var[k] if next(flags)] for k in sorted(var)}
    syn = {k: a.group(1).strip() for k, u in enumerate(units if used is not None else [])
           if LOCAL_SYNTAX.match(u.head) and (a := SYNTAX_DEF.search(u.own())) and a.group(1).strip()}
    ctx = (used or "") + "".join(" ".join(g) for g in kept.values()) + "".join(
        u.own() for k, u in enumerate(units) if k not in var and not LOCAL_SYNTAX.match(u.head))
    live: set[int] = set()
    while (more := {k for k, a in syn.items() if a in ctx + "".join(units[j].own() for j in live)}) != live:
        live = more
    out, ns, gone, opened = [header], [[]], [], set()

    def spaces(n: str) -> set[str]:
        cur = [c for f in ns for c in f]
        return {".".join(cur[:j] + [n]) for j in range(len(cur) + 1)} | {f"{p}.{n}" for p in opened}

    for k, u in enumerate(units):
        o, e = OPENER.match(u.head), END.match(u.head)
        if o and not u.head.startswith("/-"):
            ns.append(o.group(2).split(".") if o.group(1) == "namespace" and o.group(2) else [])
        elif e and len(ns) > 1:
            ns.pop()
        line = u.own().rstrip("\n") if len(u.lines) == 1 else ""
        m, only = OPEN.match(line), OPEN_ONLY.match(line)
        words = m.group(2).split() if m else []
        tail = words[-1:] if words[-1:] == ["in"] else []
        if m and not {"in", "hiding", "renaming"} & set(words[:len(words) - len(tail)]):
            keep = []
            for n in words[:len(words) - len(tail)]:
                cands = spaces(n)
                if cands & known and not cands & avail:
                    gone.append(f"open {n}")
                else:
                    keep.append(n)
                    if not tail:
                        opened |= cands & (known | avail) or {n}
            if not keep:
                continue
            if len(keep) + len(tail) < len(words):
                out.append(f"open{m.group(1) or ''} {' '.join(keep + tail)}\n" + "".join(u.blank))
                continue
        elif only:
            cands = spaces(only.group(1))
            if cands & known and not cands & avail:
                gone.append(f"open {only.group(1)} ({only.group(2).strip()})")
                continue
        elif k in syn and k not in live:
            gone.append(u.head)
            continue
        elif k in var and len(kept[k]) < len(var[k]):
            gone += [g for g in var[k] if g not in kept[k]]
            if kept[k]:
                out.append("variable " + " ".join(kept[k]) + "\n" + "".join(u.blank))
            continue
        out.append(u.full())
    return "".join(out), gone


class Tree:
    def __init__(self, root: Path):
        self.root = root
        self.mods: dict[str, Path] = {}
        for pkg in ("Security", "verity/core/service/flock"):
            for p in sorted((root / pkg).rglob("*.lean")):
                if ".lake" not in p.parts:
                    self.mods.setdefault(".".join(p.relative_to(root / pkg).with_suffix("").parts), p)
        self.files: dict[Path, str | None] = {}
        self._imports: dict[str, list[str]] = {}
        self._atoms: dict[str, list[str]] = {}
        self._spaces: dict[str, set[str]] = {}

    def read(self, m: str) -> str:
        p = self.mods[m]
        if p not in self.files:
            self.files[p] = p.read_text()
        t = self.files[p]
        assert t is not None, f"{m} was merged away"
        return t

    def write(self, m: str, text: str | None) -> None:
        self.files[self.mods[m]] = text
        self._imports.pop(m, None)
        self._atoms.pop(m, None)
        self._spaces.pop(m, None)

    def imports(self, m: str) -> list[str]:
        if m not in self._imports:
            self._imports[m] = imports_of(split_header(self.read(m))[0])
        return self._imports[m]

    def atoms(self, m: str) -> list[str]:
        """The atoms of the tactics and notations `m` defines for its importers (not its `local` ones)."""
        if m not in self._atoms:
            self._atoms[m] = [a.group(1).strip() for a in SYNTAX_DEF.finditer(self.read(m))
                              if a.group(1).strip() and not a.group(0).startswith("local ")]
        return self._atoms[m]

    def spaces(self, mods) -> set[str]:
        """The namespaces the modules `mods` declare into: each proper prefix of a declaration's name."""
        out: set[str] = set()
        for m in mods:
            if m not in self._spaces:
                self._spaces[m] = {".".join(p[:k]) for n in declared(self, m) for p in [n.split(".")]
                                   for k in range(1, len(p))}
            out |= self._spaces[m]
        return out

    def closure(self, m: str) -> list[str]:
        seen: dict[str, None] = {}
        todo = [m]
        while todo:
            x = todo.pop()
            if x in seen or x not in self.mods or self.files.get(self.mods[x], "") is None:
                continue
            seen[x] = None
            todo.extend(self.imports(x))
        return list(seen)


def find(units: list[Unit], decls: list[str], where: str, log: list[str]) -> list[Unit]:
    """The units declaring `decls`. A name outside every namespace the file declares into is an auxiliary Lean realized
    here for another module's constant (`Except.map.eq_1`): skipped, since any module that needs it realizes it."""
    by_name = {u.name: u for u in units if u.name}
    spaces = {".".join(u.name.split(".")[:k]) for u in units if u.name for k in range(1, u.name.count(".") + 1)}
    want: list[Unit] = []
    missing = []
    for d in decls:
        parts = owner(d).split(".")
        hit = next((by_name[".".join(parts[:k])] for k in range(len(parts), 0, -1) if ".".join(parts[:k]) in by_name),
                   None)
        if hit is None and not any(".".join(parts[:k]) in spaces for k in range(1, len(parts))):
            log.append(f"  {where}: {d} is realized here for a constant declared elsewhere; not moved")
        elif hit is None:
            missing.append(d)
        elif hit.in_mutual:
            raise SystemExit(f"{where}: {d} is inside `mutual`")
        elif hit not in want:
            want.append(hit)
    if missing:
        raise SystemExit(f"{where}: not found: {missing}")
    return sorted(want, key=lambda u: u.start)


def declared(tree: Tree, m: str) -> set[str]:
    return {u.name for u in annotate(units_of(split_header(tree.read(m))[1]))[0] if u.name}


def syntax_from(tree: Tree, text: str, candidates: list[str]) -> list[str]:
    """The modules of `candidates` defining a tactic or notation whose atom `text` holds."""
    return [m for m in candidates if any(a in text for a in tree.atoms(m))]


def split(tree: Tree, op: dict, log: list[str]) -> None:
    src, new = op["source"], op["new"]
    header, body = split_header(tree.read(src))
    units, _ = annotate(units_of(body))
    if new in tree.mods:
        if not {u.name for u in units if u.name} & {owner(d) for d in op["decls"]}:
            log.append(f"split {src}: {new} holds its declarations already")
            return
        raise SystemExit(f"{src} -> {new}: {new} exists")
    moved = find(units, op["decls"], f"{src} -> {new}", log)
    private = [u.name for u in moved if re.search(r"\bprivate\s", u.head)]
    if private and (op.get("source_imports_new") or op.get("importers")):
        log.append(f"  {src} -> {new}: private and used outside the new module: {private}")
    style = module_style(header)
    imps = [i for i in op.get("imports", []) if i in tree.mods and i != src]
    ext = op.get("external_imports") or [i for i in imports_of(header) if i not in tree.mods]
    imps += [i for i in ext if i not in imps]
    text = "".join(u.text() for u in moved)
    closure = [m for m in tree.closure(src) if m != src]
    imps += [m for m in syntax_from(tree, text, closure) if m not in imps]
    shown = ", ".join(f"`{u.name.split('.')[-1]}`" for u in moved[:6]) + (", …" if len(moved) > 6 else "")
    out = ("module\n\n" if style else "") + "".join(f"{'public ' if style else ''}import {i}\n" for i in imps)
    out += "".join(ln + "\n" for ln in flagged(header))
    out += f"\n/-! {shown}: the declarations of `{src}` that its importers read without the rest of it. -/\n"
    out += emit(moved)
    known = tree.spaces([src, *closure])
    tree.mods[new] = tree.mods[src].with_name(new.split(".")[-1] + ".lean")
    tree.write(new, out)
    out, gone = prune(out, known, tree.spaces(tree.closure(new)), text)
    if gone:
        tree.write(new, out)
        log.append(f"  {new}: left out {', '.join(gone)}")
    rest = header + re.sub(r"\n{3,}", "\n\n", "".join(u.full() for u in units if u not in moved))
    tree.write(src, add_import(rest, new, None) if op.get("source_imports_new") else rest)
    for imp in op.get("importers", []):
        if imp in tree.mods:
            t = add_import(tree.read(imp), new, src)
            for ln in flagged(split_header(t)[0], src):
                t = add_line(t, retarget(ln, new))
            tree.write(imp, t)
        else:
            log.append(f"  {new}: importer {imp} isn't in the tree")
    uses = op.get("uses", {})
    for imp in op.get("drop_source", []):
        seen = tree.spaces(tree.closure(imp))
        t = drop_import(tree.read(imp), src)
        tree.write(imp, t)
        reach = set(tree.closure(imp))
        lost = [d for d in uses.get(imp, []) if d in tree.mods and d not in reach]
        lost += [m for m in syntax_from(tree, t, [m for m in closure if m not in reach]) if m not in lost]
        for d in lost:
            if d not in reach:
                t = add_import(t, d, new)
                reach |= set(tree.closure(d))
        tree.write(imp, t)
        t, gone = prune(t, seen, tree.spaces(tree.closure(imp)))
        if gone:
            tree.write(imp, t)
            log.append(f"  {imp}: left out {', '.join(gone)}")
    log.append(f"split {src}: {len(moved)} declarations to {new} "
               f"({tree.mods[new].relative_to(tree.root)}), {len(imps)} imports, "
               f"{len(op.get('drop_source', []))} of {len(op.get('importers', []))} importers drop {src.split('.')[-1]}")


def repair(tree: Tree, before: dict[str, list[str]], ops: list[dict], table: dict[str, list[str]], root: str | None,
           log: list[str]) -> None:
    """Each module whose imports no longer reach one it reads imports it directly. In `root`'s closure (all of the tree
    without one), what it reads is the map's `[repair]` (what the constant graph says its constants use) and each tactic
    or notation module its imports reached before the operations, whose atom its text holds; outside it, which the graph
    doesn't cover, every module its imports reached before and every new module split from one of them. Bottom-up, so a
    module sees what its imports gained, and never an import that would close a cycle."""
    origin = {o["new"]: o["source"] for o in ops if o["kind"] == "split"}

    def first(d):
        while d in origin and d not in before:
            d = origin[d]
        return d

    memo: dict[str, frozenset] = {}
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

    def old(m):
        if m not in memo:
            s = {m}
            for i in before.get(m, ()):
                s |= old(i)
            memo[m] = frozenset(s)
        return memo[m]

    alive = [m for m in tree.mods if tree.files.get(tree.mods[m], "") is not None]
    covered = old(root) if root in before else set(before)
    syn = {m for m in alive if tree.atoms(m)}
    added = 0
    for m in sorted(alive, key=lambda x: (len(old(x)) if x in before else 0, x)):
        reach = set(tree.closure(m))
        want = [d for d in table.get(m, []) if d not in reach]
        lost = old(m) - reach if m in before else set()
        if m in covered:
            text = tree.read(m)
            want += [s for s in sorted(lost & syn) if any(a in text for a in tree.atoms(s))]
        elif m in before:
            lost |= {d for d in origin if first(d) in old(m)}
            want += sorted(lost, key=lambda d: (-len(old(first(d))), d))
        t = tree.read(m)
        for d in want:
            if (d in tree.mods and d not in reach and tree.files.get(tree.mods[d], "") is not None
                    and m not in tree.closure(d)):
                t = add_import(t, d, None)
                reach |= set(tree.closure(d))
                added += 1
                log.append(f"  repair: {m} imports {d}")
        if t != tree.read(m):
            tree.write(m, t)
    log.append(f"repair: {added} imports, where a module's imports no longer reach what it reads")


def merge(tree: Tree, op: dict, log: list[str]) -> None:
    mod, into = op["module"], op["into"]
    if mod not in tree.mods:
        log.append(f"merge {mod} into {into}: merged already")
        return
    h1, b1 = split_header(tree.read(mod))
    units1, open1 = annotate(units_of(b1))
    t = tree.read(into)
    for i in imports_of(h1):
        if i != into:
            t = add_import(t, i, None)
    for ln in flagged(h1):
        if IMPORT.match(ln).group(4) != into:
            t = add_line(t, ln)
    t = drop_import(t, mod)
    h2, b2 = split_header(t)
    # every scope `mod` leaves open closes before `into`'s own text, a public section included
    closing = "".join("\n" + (closer(f) or "end\n") for f in reversed(open1[1:])
                      if not OPENER.match(f[0].strip()) or OPENER.match(f[0].strip()).group(1) != "mutual")
    b1 = b1.rstrip("\n") + "\n" + closing
    if len(open1[0]) > 1:
        b1 = "section\n\n" + b1 + "\nend\n"
    tree.write(into, h2 + b1 + "\n" + b2.lstrip("\n"))
    for m in list(tree.mods):
        p = tree.mods[m]
        if m in (mod, into) or tree.files.get(p, "") is None:
            continue
        txt = tree.read(m)
        if mod in imports_of(split_header(txt)[0]):
            t = add_import(txt, into, mod)
            for ln in flagged(split_header(t)[0], mod):
                t = add_line(t, retarget(ln, into))
            tree.write(m, drop_import(t, mod))
    tree.write(mod, None)
    log.append(f"merge {mod} into {into}: {sum(u.decl for u in units1)} declarations")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--map")
    ap.add_argument("--first", type=int)
    ap.add_argument("--ops")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-roundtrip", action="store_true", help="parse every file and check it reassembles")
    a = ap.parse_args()
    tree = Tree(Path(a.root).resolve())
    if a.check_roundtrip:
        bad = 0
        for m, p in tree.mods.items():
            header, body = split_header(tree.read(m))
            units, _ = annotate(units_of(body))
            if header + "".join(u.full() for u in units) != tree.read(m):
                bad += 1
                print(f"roundtrip differs: {m}")
        print(f"{len(tree.mods)} files, {bad} differ")
        return 1 if bad else 0
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=tree.root, capture_output=True, text=True)
    if not a.dry_run and dirty.returncode == 0 and dirty.stdout.strip():
        raise SystemExit(f"{tree.root}: the worktree has changes; commit or stash them first")
    spec = tomllib.loads(Path(a.map).read_text())
    ops = spec["op"]
    idx = [int(x) for x in a.ops.split(",")] if a.ops else list(range(len(ops)))[:a.first]
    before = {m: tree.imports(m) for m in tree.mods}
    log: list[str] = []
    for i in idx:
        (split if ops[i]["kind"] == "split" else merge)(tree, ops[i], log)
        log[-1] = f"{i}: {log[-1]}"
    repair(tree, before, [ops[i] for i in idx], spec.get("repair", {}) if len(idx) == len(ops) else {},
           spec.get("root"), log)
    print("\n".join(log))
    if not a.dry_run:
        for p, txt in tree.files.items():
            if txt is None:
                p.unlink(missing_ok=True)
            elif not p.exists() or p.read_text() != txt:
                p.write_text(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
