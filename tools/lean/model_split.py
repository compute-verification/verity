"""Separate a Lean package's spec from its proofs inside the package, declaration by declaration, as a pure move.

    python tools/lean/model_split.py PKG --decls DECLS [--spec PREFIX ...] [--apply] [--json]

The spec is every declaration the lock's guarantees read (its `reads`), every declaration of a module under
`<root>.Protocol`, `<root>.Assumptions`, `<root>.Guarantees` or a `--spec` prefix, and everything those use, theorems
included (a definition's proof terms name them). A module whose declarations are all spec stays as it is. A module that
holds both (mixed) keeps its name and its other declarations, and its spec declarations move, in their order and with the
file's namespaces, sections, `open`s and `variable`s, to the module `<root>.Protocol.<rest>`, which it imports. Every
declaration keeps its name, so the lock's guarantees stay byte for byte and its `reads` change only in the module names
they are keyed by. A private declaration that a spec declaration of its module uses keeps the declarations that use it
beside it (in the spec). `DECLS` is `Decls.lean`'s output for the package's build (a JSON file, or gzipped).

The spec's modules import only the spec and the package's dependencies: each import of a proof module is replaced by
the spec and dependency modules that proof module imports, followed through, and each import of a mixed module by its
spec half. A proof module whose closure loses a module that way imports it directly, so every proof sees what it saw.
A module that declares syntax (a tactic) and imports only the spec is the spec's tool. `example`s and `#` commands stay
with the proofs. The lock gets a `layers` rule per spec module (outside `--spec`), which lets it import only the spec,
the package's dependencies and the libraries, and its `reads` keyed by the new module names.

It refuses, printing why, when a declaration shares a line with another command, when ranges overlap, or when a module
under a spec name holds proofs. Without `--apply` it prints the plan and changes nothing; with `--apply` it writes the
files and the lock and `git add`s them. The package's own path move is `split.py`'s, a step of its own.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC_NAMES = ("Protocol", "Assumptions", "Guarantees")
PROTOCOL = "Protocol"
IMPORT = re.compile(r"^(?:public\s+)?(?:meta\s+)?import\s+(.*)$")
PREFIX = re.compile(r"^(omit|open|set_option|include|variable|attribute|unseal|seal)\b")
IN_END = re.compile(r"(^|\s)in\s*$")
SYNTAX = re.compile(r"^(?:@\[[^\]]*\]\s*)?(syntax|macro|elab|macro_rules|elab_rules|declare_syntax_cat|notation|infixl?|infixr|"
                    r"prefix|postfix)\b")
PROOF_CMD = re.compile(r"^(?:@\[[^\]]*\]\s*)?(example\b|#[a-z_]+\b)")
NOTATION = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:(local|scoped)\s+)?(syntax|macro|elab|macro_rules|elab_rules|notation3?|infixl?|"
                      r"infixr|prefix|postfix)\b")
IDENT = re.compile(r"[^\s()\[\]{},:]+")
_ID = r"[^\W\d][\w'!?\u2080-\u209c\u1d62-\u1d6a]*"
NAME = re.compile(rf"@?{_ID}(?:\.{_ID})*")
SIMP_ATTR = re.compile(r"@\[[^\]]*\bsimp\b[^\]]*\]")
RFL_PROOF = re.compile(r":=\s*(?:by\s+(?:exact\s+)?)?\(?\s*(?:rfl|Iff\.rfl|HEq\.rfl|Eq\.refl\s*_?)\s*\)?\s*$")
SIMP_SET = re.compile(r"(?<![\w.'])(?:simp|simp_all|simpa|dsimp|simp_arith|norm_num|field_simp|aesop|grind)[?!]?(?![\w'])"
                      r"(?!\s+only\b)")


def _audit():
    spec = importlib.util.spec_from_file_location("lean_audit_for_model_split", HERE / "audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _audit()


class Refused(Exception):
    pass


def load_decls(path: Path) -> dict:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def header(code: list[str]) -> list[int]:
    """The indices of the import lines of a file's header (in its comment-blanked lines)."""
    out = []
    for i, line in enumerate(code):
        s = line.strip()
        if not s or s in ("module", "prelude"):
            continue
        if not IMPORT.match(s):
            break
        out.append(i)
    return out


def imports_of(code: list[str]) -> list[str]:
    return [m for i in header(code) for m in IMPORT.match(code[i].strip()).group(1).split()]


def user_name(n: str) -> str:
    """A private name's user-facing name (`_private.M.0.X.y` is `X.y`)."""
    if not n.startswith("_private."):
        return n
    parts = n.split(".")
    for i in range(2, len(parts)):
        if parts[i].isdigit():
            return ".".join(parts[i + 1:])
    return n


def prefixes(n: str) -> list[str]:
    parts = user_name(n).split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


class Module:
    """A package module: its source, its declaration commands (from the decls facts) and its other commands."""

    def __init__(self, name: str, path: Path, decls: list[dict] | None):
        self.name, self.path = name, path
        self.text = path.read_text()
        self.lines = self.text.split("\n")
        self.code = A.strip_comments(self.text).split("\n")
        self.imports = imports_of(self.code)
        self.decls = decls or []
        self.cmds: list[dict] = []
        self.of: dict[str, int] = {}
        self.free: list[dict] = []
        ranged = sorted((d for d in self.decls if d["r"]), key=lambda d: (d["r"][0], d["r"][1], -d["r"][2], -d["r"][3], d["n"]))
        for d in ranged:
            l0, c0, l1, c1 = d["r"]
            if self.cmds and (l0, c0) >= self.cmds[-1]["start"] and (l1, c1) <= self.cmds[-1]["end"]:
                self.cmds[-1]["names"].append(d["n"])
            elif self.cmds and (l0, c0) < self.cmds[-1]["end"]:
                raise Refused(f"{name}: {d['n']}'s range {d['r']} overlaps {self.cmds[-1]['names'][0]}'s")
            else:
                self.cmds.append({"start": (l0, c0), "end": (l1, c1), "names": [d["n"]]})
            self.of[d["n"]] = len(self.cmds) - 1
        for d in self.decls:
            if d["n"] in self.of:
                continue
            p = d["n"]
            while "." in p:
                p = p.rsplit(".", 1)[0]
                if p in self.of:
                    self.of[d["n"]] = self.of[p]
                    break
            else:
                self.free.append(d)
        self.spans = [self.span(c) for c in self.cmds]
        self.proof_spans = self.other_spans(PROOF_CMD)

    def span(self, c: dict) -> tuple[int, int]:
        """The 0-based lines [first, last] a declaration command owns: its range, the `… in` commands before it, and the
        `--` comments right above them."""
        (l0, c0), (l1, c1) = c["start"], c["end"]
        s, e = l0 - 1, l1 - 1
        if self.code[s][:c0].strip() or self.code[e][c1:].strip():
            raise Refused(f"{self.name}:{l0}: {c['names'][0]} shares a line with another command")
        k = s
        while True:
            j = k - 1
            while j >= 0 and not self.code[j].strip():
                j -= 1
            if j < 0 or not IN_END.search(self.code[j].rstrip()):
                break
            i = j
            while i > 0 and (not self.code[i] or self.code[i][0].isspace()):
                i -= 1
            if not PREFIX.match(self.code[i]):
                break
            k = i
        while k > 0 and self.lines[k - 1].startswith("--") and not self.code[k - 1].strip():
            k -= 1
        return k, e

    def other_spans(self, pat: re.Pattern) -> list[tuple[int, int]]:
        """The lines of each top-level command `pat` matches (an `example`, a `#` command), from its doc comment to the
        next command."""
        out = []
        for i, line in enumerate(self.code):
            if not pat.match(line):
                continue
            j = i + 1
            while j < len(self.code) and (not self.code[j].strip() or self.code[j][0].isspace()) and \
                    (not self.lines[j].strip() or self.lines[j][0].isspace()):
                j += 1
            e = j - 1
            while e > i and not self.lines[e].strip():
                e -= 1
            b = i
            if i > 0 and self.lines[i - 1].rstrip().endswith("-/") and not self.code[i - 1].strip():
                k = i - 1
                while k > 0 and not self.code[k].strip() and not self.lines[k].lstrip().startswith("/--"):
                    k -= 1
                if self.lines[k].lstrip().startswith("/--") and not self.code[k].strip():
                    b = k
            out.append((b, e))
        return out


def commands_of(lines: list[str], code: list[str], i: int) -> int:
    """The last line of the top-level command starting on line ``i`` (its indented continuation lines)."""
    j = i + 1
    while j < len(code) and code[j].strip() and code[j][0].isspace():
        j += 1
    return j - 1


def scopes(code: list[str]) -> list[tuple[list[str], list[str]]]:
    """For each line, the namespace it is in (its components) and the namespaces opened in the scopes around it."""
    stack: list[tuple[str, list[str], list[str]]] = [("file", [], [])]
    out = []
    for line in code:
        ns = [c for k, comps, _ in stack if k == "ns" for c in comps]
        out.append((ns, [o for *_, opens in stack for o in opens]))
        s = line.strip()
        if m := re.match(r"^namespace\s+(\S+)", line):
            stack.append(("ns", m.group(1).split("."), []))
        elif re.match(r"^(noncomputable\s+)?section\b", line):
            stack.append(("sec", [], []))
        elif m := re.match(r"^end\b\s*(\S*)", line):
            # `end A.B` closes the scopes `namespace A.B` (one) or `namespace A` and `namespace B` (two) opened.
            want = len(m.group(1).split(".")) if m.group(1) and stack[-1][0] == "ns" else 1
            while want > 0 and len(stack) > 1:
                want -= max(len(stack.pop()[1]), 1)
        elif re.match(r"^open\s", line) and not IN_END.search(s):
            stack[-1][2].extend(open_items(s)[0])
    return out


def binder_groups(body: str) -> list[tuple[str, list[str], list[str]]]:
    """A `variable` command's bracketed binder groups: each one's text, the names it binds and the identifiers its type
    names."""
    groups, depth, start = [], 0, 0
    for k, ch in enumerate(body):
        if ch in "({[⦃":
            if depth == 0:
                start = k
            depth += 1
        elif ch in ")}]⦄" and depth:
            depth -= 1
            if depth == 0:
                inner = body[start + 1:k]
                head, sep, ty = inner.partition(" : ")
                if not sep:
                    head, ty = ("", inner) if body[start] == "[" else (inner, "")
                groups.append((body[start:k + 1], head.split(), IDENT.findall(ty)))
    return groups


def binders(body: str) -> tuple[list[str], list[str]]:
    """A `variable` command's binder names, and the identifiers its binders' types name."""
    groups = binder_groups(body)
    return [n for _, ns, _ in groups for n in ns], [r for *_, rs in groups for r in rs]


def wrap(head: str, parts: list[str], width: int = 120) -> str:
    """``head`` and ``parts`` joined by spaces, broken before a part that would pass ``width``, continued indented."""
    lines = [head]
    for k, part in enumerate(parts):
        if k and len(lines[-1]) + 1 + len(part) > width:
            lines.append("  " + part)
        else:
            lines[-1] += " " + part
    return "\n".join(lines)


def open_items(cmd: str) -> tuple[list[str], str]:
    """An `open` command's namespaces, and the form (`scoped `/plain, explicit names and `hiding` aside). Batteries'
    `open private … from M` opens none."""
    body = re.sub(r"\s+in\s*$", "", cmd.strip())[len("open"):].strip()
    if body.startswith("private "):
        return [], ""
    scoped = body.startswith("scoped ")
    if scoped:
        body = body[len("scoped "):]
    if "hiding" in body.split() or "renaming" in body.split() or "(" in body:
        return [body.split()[0]], "scoped " if scoped else ""
    return body.split(), "scoped " if scoped else ""


def candidates(ns: list[str], opens: list[str], t: str) -> list[str]:
    """The names ``t`` may stand for in namespace ``ns`` with ``opens`` open (each relative to an enclosing namespace)."""
    if t.startswith("_root_."):
        return [t[len("_root_."):]]
    return [".".join(ns[:k] + [o, t] if o else ns[:k] + [t]) for o in [""] + opens for k in range(len(ns), -1, -1)]


def plan(pkg: Path, decls: dict, spec_prefixes: list[str]) -> dict:
    pkg = pkg.resolve()
    policy = A.load(pkg)
    files = A.lean_modules(pkg)
    roots = policy["roots"]
    if len(roots) != 1:
        raise Refused(f"{pkg}: one root expected, the lock has {roots}")
    root = roots[0]
    spec_pre = [f"{root}.{n}" for n in SPEC_NAMES] + list(spec_prefixes)
    facts = decls["modules"]
    mods = {m: Module(m, pkg / f, (facts.get(m) or {}).get("decls")) for m, f in sorted(files.items())}
    where: dict[str, tuple[str, int]] = {}
    free_uses: dict[str, list[str]] = {}
    for m, mod in mods.items():
        for n, i in mod.of.items():
            where[n] = (m, i)
        for d in mod.free:
            free_uses[d["n"]] = d["u"]
    uses_of = {d["n"]: d["u"] for mod in mods.values() for d in mod.decls}
    memo: dict[str, frozenset] = {}

    def resolve(u: str, seen: frozenset = frozenset()) -> frozenset:
        if u in where:
            return frozenset([where[u]])
        if u in memo:
            return memo[u]
        if u not in free_uses or u in seen:
            return frozenset()
        r = frozenset().union(*(resolve(x, seen | {u}) for x in free_uses[u]))
        memo[u] = r
        return r

    members: dict[tuple[str, int], list[str]] = defaultdict(list)
    for n, c in where.items():
        members[c].append(n)
    deps: dict[tuple[str, int], set] = {}
    for m, mod in mods.items():
        for i in range(len(mod.cmds)):
            deps[(m, i)] = set().union(*(resolve(u) for n in members[(m, i)] for u in uses_of.get(n, []))) - {(m, i)}
    # What a command's text names counts too: `simp` leaves an rfl lemma out of the term it builds, and a notation is gone
    # once parsed. A name resolves as Lean would, through the command's namespaces and `open`s, a private one in its module.
    by_user: dict[str, list[str]] = defaultdict(list)
    for n in where:
        by_user[user_name(n)].append(n)
    named: dict[tuple[str, int], set] = defaultdict(set)
    scopes_of = {m: scopes(mod.code) for m, mod in mods.items()}
    for m, mod in mods.items():
        memo_names: dict[tuple, frozenset] = {}
        for i, (s, e) in enumerate(mod.spans):
            ns, opens = scopes_of[m][s]
            opens = opens + [it for k in range(s, e + 1) if mod.code[k].startswith("open ") and IN_END.search(mod.code[k].rstrip())
                             for it in open_items(mod.code[k])[0]]
            for t in set(NAME.findall("\n".join(mod.code[s:e + 1]))):
                key = (tuple(ns), tuple(opens), t)
                if key not in memo_names:
                    parts = t.lstrip("@").split(".")
                    got = set()
                    for p in (".".join(parts[:k]) for k in range(len(parts), 0, -1)):
                        got |= {n for c in candidates(ns, opens, p) for n in by_user.get(c, ())
                                if not n.startswith("_private.") or where[n][0] == m}
                    memo_names[key] = frozenset(got)
                named[(m, i)] |= memo_names[key]
            deps[(m, i)] |= {where[n] for n in named[(m, i)]} - {(m, i)}
    # An rfl `@[simp]` theorem is in no term `simp` builds, nor named when `simp` runs on its default set: a command that
    # does depends on each one it sees, earlier in its module or in a package module it imports.
    kinds = {d["n"]: d["k"] for mod in mods.values() for d in mod.decls}
    simp_rfl: dict[str, list[tuple[int, tuple[str, int]]]] = defaultdict(list)
    for m, mod in mods.items():
        for i, c in enumerate(mod.cmds):
            s, e = mod.spans[i]
            text = " ".join(x.strip() for x in mod.code[s:e + 1])
            if any(kinds.get(n) == "theorem" for n in c["names"]) and SIMP_ATTR.search(text) and RFL_PROOF.search(text):
                simp_rfl[m].append((s, (m, i)))
    pmemo: dict[str, frozenset] = {}

    def package_closure(m: str) -> frozenset:
        if m not in pmemo:
            pmemo[m] = frozenset()
            pmemo[m] = frozenset(x for i in mods[m].imports if i in mods for x in ({i} | package_closure(i)))
        return pmemo[m]

    for m, mod in mods.items():
        seen = {c for q in package_closure(m) for _, c in simp_rfl.get(q, ())}
        for i, (s, e) in enumerate(mod.spans):
            if SIMP_SET.search("\n".join(mod.code[s:e + 1])):
                deps[(m, i)] |= (seen | {c for k, c in simp_rfl.get(m, ()) if k < s}) - {(m, i)}
    by_parent: dict[str, set] = defaultdict(set)
    for n, c in where.items():
        by_parent[A.parent(n)].add(c)
    seeds = set()
    for m, rec in policy.get("reads", {}).items():
        if m not in mods:
            continue
        for n in rec.get("definitions", {}):
            if not by_parent.get(n):
                raise Refused(f"the lock reads {n} in {m}, which the decls facts do not declare: are they from this tree's build?")
            seeds |= by_parent[n]
    for m, mod in mods.items():
        if A.under(spec_pre, m):
            seeds |= {(m, i) for i, c in enumerate(mod.cmds) if any(kinds[n] != "theorem" for n in c["names"])}
    private_users: dict[tuple[str, int], set] = defaultdict(set)
    for m, mod in mods.items():
        for i in range(len(mod.cmds)):
            for u in {u for n in members[(m, i)] for u in uses_of.get(n, [])} | named[(m, i)]:
                if u.startswith("_private.") and u in where and where[u][0] == m and where[u] != (m, i):
                    private_users[where[u]].add((m, i))
    # A notation, macro or syntax: a local one is copied into both halves of its module (its constants are the module's
    # own), any other goes with the spec of a module that has some.
    syntax_cmd = {(m, i): NOTATION.match(next((x for x in mod.code[c["start"][0] - 1:c["end"][0]] if x.strip()), ""))
                  for m, mod in mods.items() for i, c in enumerate(mod.cmds)}
    syntax_cmd = {c: mt.group(1) == "local" for c, mt in syntax_cmd.items() if mt}
    spec: set = set()
    todo = sorted(seeds)
    while todo:
        while todo:
            c = todo.pop()
            if c not in spec:
                spec.add(c)
                todo.extend(sorted((deps[c] | private_users[c]) - spec))
        has_spec = {m for m, _ in spec}
        todo = sorted(c for c, local in syntax_cmd.items() if not local and c[0] in has_spec and c not in spec)
    side = {}
    for m, mod in mods.items():
        s = {i for i in range(len(mod.cmds)) if (m, i) in spec}
        p = set(range(len(mod.cmds))) - s
        side[m] = "spec" if s and not p and not mod.proof_spans else "mixed" if s else "proofs"
        if A.under(spec_pre, m):
            if p or mod.proof_spans:
                raise Refused(f"{m} is the spec's by its name but holds {len(p)} proof declaration(s) and "
                              f"{len(mod.proof_spans)} example(s)")
            side[m] = "spec"
    protocol = {m: f"{root}.{PROTOCOL}.{m[len(root) + 1:]}" for m in mods if side[m] == "mixed"}
    if clash := sorted(set(protocol.values()) & set(mods)):
        raise Refused(f"the spec halves' names are taken: {', '.join(clash)}")
    # A tool (a module declaring syntax) whose package imports are all spec is the spec's too.
    changed = True
    while changed:
        changed = False
        for m, mod in mods.items():
            if side[m] == "proofs" and any(SYNTAX.match(x) for x in mod.code) and \
                    all(side.get(i) in ("spec", "tool") for i in mod.imports if i in mods):
                side[m] = "tool"
                changed = True
    spec_mods = sorted([m for m in mods if side[m] in ("spec", "tool")] + list(protocol.values()))
    ext_imports = decls["imports"]

    # The spec's imports: a proof module's import is replaced by what it imports, followed through.
    fmemo: dict[str, list[str]] = {}

    def f(i: str) -> list[str]:
        if i not in mods:
            return [] if i == "Init" else [i]
        if side[i] in ("spec", "tool"):
            return [i]
        if side[i] == "mixed":
            return [protocol[i]]
        if i not in fmemo:
            fmemo[i] = []
            fmemo[i] = list(dict.fromkeys(x for j in mods[i].imports for x in f(j)))
        return fmemo[i]

    new_imports: dict[str, list[str]] = {}
    for m in sorted(mods):
        if side[m] in ("spec", "tool", "mixed"):
            key = protocol.get(m, m)
            new_imports[key] = list(dict.fromkeys(x for i in mods[m].imports for x in f(i)))
    def closure_of(graph):
        """The modules a package module reaches through `graph`, its dependencies' modules as leaves (never followed)."""
        memo: dict[str, frozenset] = {}

        def cl(m: str) -> frozenset:
            if m not in memo:
                memo[m] = frozenset()
                memo[m] = frozenset(x for i in graph(m) for x in ({i} | cl(i)))
            return memo[m]
        return cl

    spec_closure = closure_of(lambda m: new_imports.get(m, []))
    for m in sorted(new_imports):
        orig = set(mods[m].imports) if m in mods else set(mods[next(k for k, v in protocol.items() if v == m)].imports)
        imps = new_imports[m]
        new_imports[m] = [i for i in imps if i in orig or not any(i in spec_closure(j) for j in imps if j != i)]
    spec_closure = closure_of(lambda m: new_imports.get(m, []))
    # Proof modules see what they saw: each import the split takes from a closure comes back directly.
    proof_imports = {m: list(mods[m].imports) + ([protocol[m]] if m in protocol else [])
                     for m in mods if side[m] in ("proofs", "mixed")}
    oclosure = closure_of(lambda m: mods[m].imports if m in mods else [])
    ncl: dict[str, frozenset] = {}

    def reach(imps: list[str]) -> set:
        return set().union(set(imps), *(ncl.get(i) or (spec_closure(i) if i in new_imports else frozenset()) for i in imps))

    order, seen = [], set()

    def visit(m: str) -> None:
        if m in seen or m not in mods:
            return
        seen.add(m)
        for i in mods[m].imports:
            visit(i)
        order.append(m)

    for m in sorted(mods):
        visit(m)
    added: dict[str, list[str]] = {}
    for m in order:
        if m not in proof_imports:
            continue
        want = {x for x in oclosure(m) if x in mods and side[x] in ("proofs", "mixed")}
        miss = sorted(want - reach(proof_imports[m]))
        top = [x for x in miss if not any(x in oclosure(y) for y in miss if y != x)]
        if top:
            added[m] = top
            proof_imports[m] = proof_imports[m] + top
        ncl[m] = frozenset(reach(proof_imports[m]))

    # Commands by side, and the `open`s and `attribute`s a half can't keep.
    spec_consts = {n for m, mod in mods.items() for n, i in mod.of.items() if (m, i) in spec or side[m] in ("spec", "tool")}
    pkg_consts = set(where) | set(free_uses)
    # `open N` needs `N` registered where it stands: by a `namespace` command or a declaration under `N`, earlier in its
    # module or in a module it imports. What each line of a spec module (or a spec half) registers:
    ns_events: dict[str, list[tuple[int, set, bool]]] = {}
    for m, mod in mods.items():
        if side[m] == "proofs":
            continue
        ev = []
        for i, line in enumerate(mod.code):
            if mm := re.match(r"^namespace\s+(\S+)", line):
                ps = scopes_of[m][i][0] + mm.group(1).split(".")
                ev.append((i, {".".join(ps[:k]) for k in range(1, len(ps) + 1)}, False))
        ev += [(s, {p for n in members[(m, i)] for p in prefixes(n)}, True) for i, (s, _) in enumerate(mod.spans)
               if (m, i) in spec or side[m] in ("spec", "tool")]
        ns_events[protocol.get(m, m)] = ev
    imported_ns = {k: set().union(*(x for j in spec_closure(k) for _, x, _ in ns_events.get(j, ()))) for k in ns_events}

    def registered(key: str, i: int) -> set:
        return imported_ns[key].union(*(x for k, x, _ in ns_events[key] if k < i))

    if "namespaces" not in decls:
        raise Refused("the decls facts list no `namespaces`: rerun Decls.lean")
    # The split keeps every dependency a module's closure had, so what dependencies register stays registered.
    dep_ns = {n for d, ns in decls["namespaces"].items() if d not in mods for n in ns}

    def open_keep(key: str, m: str, i: int, items: list[str]) -> list[str]:
        """The items of line ``i``'s `open` that spec module ``key`` keeps: all but one no namespace registered there
        matches (through the enclosing namespaces and those open, as Lean resolves it). Nothing the spec names comes
        through it, since the module of what it names would register it; it refuses when a spec declaration later in
        the file is under it."""
        here, (ns, opens) = registered(key, i), scopes_of[m][i]
        later = set().union(*(x for k, x, decl in ns_events[key] if decl and k > i))
        keep = []
        for it in items:
            cands = set(candidates(ns, opens + keep, it))
            if cands & here or cands & dep_ns:
                keep.append(it)
            elif hit := sorted(cands & later):
                raise Refused(f"{m}:{i + 1}: `open {it}` stands before the first spec declaration under {hit[0]}")
        return keep

    def open_edit(m: str, i: int, keep: list[str], drop: set, edits: dict) -> None:
        e = commands_of(mods[m].lines, mods[m].code, i)
        body = " ".join(x.strip() for x in mods[m].code[i:e + 1])
        if not keep:
            drop.update(range(i, e + 1))
        else:
            edits[i] = f"open {open_items(body)[1]}{' '.join(keep)}{' in' if IN_END.search(body) else ''}"
            drop.update(range(i + 1, e + 1))

    whole_edits = {}
    for m, mod in sorted(mods.items()):
        if side[m] not in ("spec", "tool"):
            continue
        drop, edits = set(), {}
        for i, line in enumerate(mod.code):
            if line.startswith("open "):
                items = open_items(" ".join(x.strip() for x in mod.code[i:commands_of(mod.lines, mod.code, i) + 1]))[0]
                if len(keep := open_keep(m, m, i, items)) != len(items):
                    open_edit(m, i, keep, drop, edits)
        if drop or edits:
            whole_edits[m] = {"drop": sorted(drop), "edits": edits}
    halves = {}
    warnings = []
    for m, mod in sorted(mods.items()):
        if side[m] != "mixed":
            continue
        sc = scopes(mod.code)
        local_names = {n for i, line in enumerate(mod.code) if line.startswith("variable")
                       for n in binders(" ".join(x.strip() for x in mod.code[i:commands_of(mod.lines, mod.code, i) + 1]))[0]}
        gone: set[str] = set()
        spec_drop, proof_drop = set(), set()
        for i, (s, e) in enumerate(mod.spans):
            if syntax_cmd.get((m, i)) and all(n in spec_consts for n in named[(m, i)] if where[n] != (m, i)):
                continue
            (spec_drop if (m, i) not in spec else proof_drop).update(range(s, e + 1))
        for s, e in mod.proof_spans:
            spec_drop.update(range(s, e + 1))
        edits: dict[int, str | None] = {}
        for i, line in enumerate(mod.code):
            if i in spec_drop:
                continue
            if line.startswith("attribute ["):
                e = commands_of(mod.lines, mod.code, i)
                body = " ".join(x.strip() for x in mod.code[i:e + 1])
                local = re.match(r"^attribute\s*\[\s*local\b", body)
                targets = [n for n in IDENT.findall(body[body.index("]") + 1:]) if n != "in"]
                ns, opens = sc[i]
                sides = set()
                for t in targets:
                    for c in candidates(ns, opens, t):
                        if c in where:
                            sides.add("spec" if c in spec_consts else "proofs")
                            break
                if not local and sides == {"spec"}:
                    proof_drop.update(range(i, e + 1))
                elif sides and "spec" not in sides or (sides == {"spec", "proofs"} and local):
                    spec_drop.update(range(i, e + 1))
                elif sides == {"spec", "proofs"}:
                    raise Refused(f"{m}:{i + 1}: one `attribute` names spec and proof declarations: {body}")
            elif line.startswith("open "):
                e = commands_of(mod.lines, mod.code, i)
                body = " ".join(x.strip() for x in mod.code[i:e + 1])
                items = open_items(body)[0]
                ns = sc[i][0]
                keep = open_keep(protocol[m], m, i, items)
                if len(keep) == len(items):
                    if "(" in body:
                        got = re.search(r"\((.*)\)", body)
                        names = got.group(1).split() if got else []
                        cands = [".".join(ns[:k] + [items[0]]) for k in range(len(ns), -1, -1)]
                        ok = [n for n in names if not any(f"{c}.{n}" in pkg_consts for c in cands)
                              or any(f"{c}.{n}" in spec_consts for c in cands)]
                        if len(ok) != len(names):
                            if ok:
                                edits[i] = re.sub(r"\(.*\)", "(" + " ".join(ok) + ")", " ".join(mod.lines[i:e + 1]).strip())
                                spec_drop.update(range(i + 1, e + 1))
                            else:
                                spec_drop.update(range(i, e + 1))
                    continue
                open_edit(m, i, keep, spec_drop, edits)
            elif line.startswith("variable") and not IN_END.search(line):
                # A binder whose type names a proof's constant, or a binder left out before it, is left out of the spec
                # half; the rest stay, since a spec declaration may use them (an unused `variable` binder is not one
                # of a declaration's parameters).
                e = commands_of(mod.lines, mod.code, i)
                groups = binder_groups(" ".join(x.strip() for x in mod.code[i:e + 1])[len("variable"):])
                ns, opens = sc[i]
                kept, left = [], []
                for text, names, refs in groups:
                    bad = next((r for r in refs if r.lstrip("@").split(".")[0] in gone), None)
                    for t in (r.lstrip("@") for r in refs if r.lstrip("@") not in local_names):
                        if bad:
                            break
                        c = next((c for c in candidates(ns, opens, t) if c in where), None)
                        if c and c not in spec_consts:
                            bad = c
                    if bad:
                        gone |= set(names)
                        left.append((names, bad))
                    else:
                        kept.append(text)
                if left:
                    if kept:
                        edits[i] = wrap("variable", kept)
                        spec_drop.update(range(i + 1, e + 1))
                    else:
                        spec_drop.update(range(i, e + 1))
                    warnings.append(f"{m}:{i + 1}: the spec half leaves out the `variable` binders " +
                                    ", ".join(f"{' '.join(bs) or '[instance]'} (naming {b})" for bs, b in left) +
                                    ", a proof's or one left out")
            elif line.startswith("include ") and gone:
                e = commands_of(mod.lines, mod.code, i)
                names = " ".join(x.strip() for x in mod.code[i:e + 1]).split()[1:]
                if keep := [n for n in names if n not in gone]:
                    if len(keep) != len(names):
                        edits[i] = "include " + " ".join(keep)
                        spec_drop.update(range(i + 1, e + 1))
                else:
                    spec_drop.update(range(i, e + 1))
        halves[m] = {"spec_drop": sorted(spec_drop), "proof_drop": sorted(proof_drop), "spec_edits": edits}
    layers = {}
    deps_roots = sorted({i.split(".")[0] for m in spec_mods for i in new_imports.get(m, []) if i not in mods
                         and i not in protocol.values()})
    rule_imports = sorted(set(deps_roots) | {f"{root}.{PROTOCOL}"} |
                          {m for m in spec_mods if not m.startswith(f"{root}.{PROTOCOL}.") and not A.under(spec_pre, m)} |
                          {p for p in spec_pre if any(A.under([p], m) for m in spec_mods)})
    why = ("the spec: the definitions the guarantees read and what they use; it imports only the spec, the package's "
           "dependencies and the libraries, never a proof (model_split.py)")
    proof_names = [m for m in mods if side[m] in ("proofs", "mixed")]
    for key in sorted({f"{root}.{PROTOCOL}"} | {m for m in spec_mods if not m.startswith(f"{root}.{PROTOCOL}.")
                                              and not A.under(spec_pre, m)}):
        if key == f"{root}.{PROTOCOL}" and not protocol:
            continue
        layers[key] = {"imports": rule_imports, "except": sorted(p for p in proof_names if A.under([key], p)), "why": why}
    reads = policy.get("reads", {})
    new_reads = dict(sorted(((protocol.get(k, k), v) for k, v in reads.items())))
    lock = dict(policy)
    lock["reads"] = new_reads
    lock["layers"] = dict(sorted({**policy.get("layers", {}), **layers}.items()))
    counts = defaultdict(int)
    for m, mod in mods.items():
        for i, c in enumerate(mod.cmds):
            key = "spec" if (m, i) in spec or side[m] in ("spec", "tool") else "proofs"
            kinds = {d["k"] for d in mod.decls if d["n"] in c["names"]}
            counts[f"{key} {'theorem' if kinds == {'theorem'} else 'definition'} commands"] += 1
            if side[m] == "mixed" and (m, i) in spec:
                counts["moved commands"] += 1
                counts[f"moved {'theorem' if kinds == {'theorem'} else 'definition'} commands"] += 1
                counts["moved constants"] += len(c["names"])
    ext_spec = sorted({i for m in spec_mods for i in new_imports.get(m, []) if i not in mods and i not in protocol.values()})
    return {"package": str(pkg), "root": root, "side": side, "protocol": protocol, "new_imports": new_imports,
            "proof_imports": {m: v for m, v in proof_imports.items() if v != mods[m].imports}, "added": added,
            "halves": halves, "whole_edits": whole_edits, "lock": lock, "layers": layers, "warnings": warnings, "counts": dict(sorted(counts.items())),
            "spec_modules": spec_mods, "dependency_imports": ext_spec, "files": {m: files[m] for m in mods}}


def without(lines: list[str], drop: set[int], edits: dict[int, str]) -> list[str]:
    """``lines`` without the lines ``drop`` (a removed command takes one blank line around it with it), ``edits`` applied."""
    out = []
    for i, line in enumerate(lines):
        if i in drop:
            continue
        if not line.strip() and out and not out[-1].strip() and (i - 1 in drop or i + 1 in drop):
            continue
        out.append(edits.get(i, line))
    while out and not out[-1].strip() and len(out) > 1 and not out[-2].strip():
        out.pop()
    return out


def with_imports(lines: list[str], code: list[str], imports: list[str]) -> list[str]:
    """``lines`` with ``imports`` as its header: the imports it keeps where they were, with the comments between them, and
    the others after them."""
    idx = header(code)
    head, rest = (lines[:idx[0]], lines[idx[-1] + 1:]) if idx else ([], lines)
    body, done = [], set()
    for k in range(idx[0], idx[-1] + 1) if idx else ():
        if k in idx:
            mine = [i for i in IMPORT.match(code[k].strip()).group(1).split() if i in imports and i not in done]
            body += [f"import {i}" for i in mine]
            done |= set(mine)
        elif lines[k].strip():
            body.append(lines[k])
    body += [f"import {i}" for i in imports if i not in done]
    if not body and not any(x.strip() for x in head):
        while rest and not rest[0].strip():
            rest = rest[1:]
        return rest
    return head + body + rest


def render(p: dict) -> dict[str, str]:
    """{path relative to the package: new text} for every file the split writes."""
    pkg = Path(p["package"])
    out = {}
    root = p["root"]
    for m, rel in p["files"].items():
        side = p["side"][m]
        mod_text = (pkg / rel).read_text()
        lines, code = mod_text.split("\n"), A.strip_comments(mod_text).split("\n")
        if side == "mixed":
            h = p["halves"][m]
            spec_lines = without(lines, set(h["spec_drop"]), {int(k): v for k, v in h["spec_edits"].items()})
            spec_code = without(code, set(h["spec_drop"]), {int(k): v for k, v in h["spec_edits"].items()})
            sp = p["protocol"][m]
            out["/".join(sp.split(".")) + ".lean"] = "\n".join(with_imports(spec_lines, spec_code, p["new_imports"][sp]))
            proof_lines = without(lines, set(h["proof_drop"]), {})
            proof_code = without(code, set(h["proof_drop"]), {})
            out[rel] = "\n".join(with_imports(proof_lines, proof_code, p["proof_imports"].get(m, imports_of(code))))
            edited = {j for k in h["spec_edits"] for j in range(int(k), commands_of(lines, code, int(k)) + 1)}
            lost = Counter(x for k, x in enumerate(lines) if k not in set(header(code)) | edited and x.strip()) - \
                Counter(spec_lines + proof_lines)
            if lost:
                raise Refused(f"{m}: the split would lose {sum(lost.values())} line(s), such as {next(iter(lost))!r}")
        elif side in ("spec", "tool"):
            w = p["whole_edits"].get(m)
            if w:
                drop, edits = set(w["drop"]), {int(k): v for k, v in w["edits"].items()}
                lines, code = without(lines, drop, edits), without(code, drop, edits)
            if w or p["new_imports"][m] != imports_of(code):
                out[rel] = "\n".join(with_imports(lines, code, p["new_imports"][m]))
        elif m in p["proof_imports"]:
            out[rel] = "\n".join(with_imports(lines, code, p["proof_imports"][m]))
    return dict(sorted(out.items()))


def apply(p: dict) -> list[str]:
    pkg = Path(p["package"])
    notes = []
    for rel, text in render(p).items():
        f = pkg / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        new = not f.exists()
        f.write_text(text)
        notes.append(f"{'wrote' if new else 'rewrote'}: {rel}")
    A.dump(pkg, p["lock"])
    notes.append(f"rewrote: {A.POLICY}")
    subprocess.run(["git", "add", "-A", "--", "."], cwd=pkg, check=True, capture_output=True)
    return notes


def describe(p: dict) -> str:
    side = p["side"]
    n = defaultdict(int)
    for s in side.values():
        n[s] += 1
    lines = [f"model split of {p['package']} (root {p['root']})",
             f"  modules: {n['spec']} spec, {n['tool']} tools, {n['mixed']} mixed (spec halves under "
             f"{p['root']}.{PROTOCOL}), {n['proofs']} proofs",
             "  " + ", ".join(f"{k} {v}" for k, v in p["counts"].items()),
             f"  spec imports from dependencies: {len(p['dependency_imports'])} modules "
             f"({', '.join(sorted({i.split('.')[0] for i in p['dependency_imports']}))})",
             f"  proof modules given imports their closure lost: {len(p['added'])}",
             f"  lock: reads keyed by {len(p['protocol'])} new module names; layers rules added: {', '.join(p['layers']) or 'none'}"]
    lines += [f"  warning: {w}" for w in p["warnings"]]
    lines += [f"  spec {m}: an `open` loses a namespace only the proofs register, line(s) " +
              ", ".join(str(k + 1) for k in sorted({int(k) for k in w["edits"]} | {k for k in w["drop"] if k - 1 not in w["drop"]}))
              for m, w in sorted(p["whole_edits"].items())]
    lines += [f"  mixed {m} -> {p['protocol'][m]}" for m in sorted(p["protocol"])]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=Path)
    ap.add_argument("--decls", type=Path, required=True, help="Decls.lean's output for the package's build")
    ap.add_argument("--spec", nargs="*", default=[], help="module prefixes whose declarations are all the spec's")
    ap.add_argument("--apply", action="store_true", help="write the files and the lock (else print the plan)")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    a = ap.parse_args(argv)
    pkg = a.package if a.package.is_absolute() else Path.cwd() / a.package
    try:
        p = plan(pkg, load_decls(a.decls), a.spec)
    except Refused as e:
        print(f"model_split: refused: {e}", file=sys.stderr)
        return 1
    print(json.dumps(p, indent=1, ensure_ascii=False, default=list) if a.json else describe(p))
    if a.apply:
        for note in apply(p):
            print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
