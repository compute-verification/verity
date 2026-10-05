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
IDENT = re.compile(r"[^\s()\[\]{},:]+")


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
        """The lines of each top-level command `pat` matches (an `example`, a `#` command), to the next command."""
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
            out.append((i, e))
        return out


def commands_of(lines: list[str], code: list[str], i: int) -> int:
    """The last line of the top-level command starting on line ``i`` (its indented continuation lines)."""
    j = i + 1
    while j < len(code) and code[j].strip() and code[j][0].isspace():
        j += 1
    return j - 1


def scopes(code: list[str]) -> list[tuple[list[str], list[str]]]:
    """For each line, the namespace it is in (its components) and the namespaces opened in the scopes around it."""
    stack: list[tuple[str, list[str], list[str]]] = []
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
            want = len(m.group(1).split(".")) if m.group(1) and stack and stack[-1][0] == "ns" else 1
            for _ in range(min(want, len(stack))):
                stack.pop()
        elif re.match(r"^open\s", line) and not IN_END.search(s):
            if stack:
                stack[-1][2].extend(open_items(s)[0])
    return out


def binders(body: str) -> tuple[list[str], list[str]]:
    """A `variable` command's binder names, and the identifiers its binders' types name."""
    names, refs, depth, start = [], [], 0, 0
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
                names += head.split()
                refs += IDENT.findall(ty)
    return names, refs


def open_items(cmd: str) -> tuple[list[str], str]:
    """An `open` command's namespaces, and the form (`scoped `/plain, explicit names and `hiding` aside)."""
    body = re.sub(r"\s+in\s*$", "", cmd.strip())[len("open"):].strip()
    scoped = body.startswith("scoped ")
    if scoped:
        body = body[len("scoped "):]
    if "hiding" in body.split() or "renaming" in body.split() or "(" in body:
        return [body.split()[0]], "scoped " if scoped else ""
    return body.split(), "scoped " if scoped else ""


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

    deps: dict[tuple[str, int], set] = {}
    for m, mod in mods.items():
        for i, c in enumerate(mod.cmds):
            deps[(m, i)] = set().union(*(resolve(u) for n in c["names"] for u in uses_of.get(n, []))) - {(m, i)}
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
    kinds = {d["n"]: d["k"] for mod in mods.values() for d in mod.decls}
    for m, mod in mods.items():
        if A.under(spec_pre, m):
            seeds |= {(m, i) for i, c in enumerate(mod.cmds) if any(kinds[n] != "theorem" for n in c["names"])}
    private_users: dict[tuple[str, int], set] = defaultdict(set)
    for m, mod in mods.items():
        for i, cmd in enumerate(mod.cmds):
            for n in cmd["names"]:
                for u in uses_of.get(n, []):
                    if u.startswith("_private.") and u in where and where[u][0] == m and where[u] != (m, i):
                        private_users[where[u]].add((m, i))
    spec: set = set()
    todo = sorted(seeds)
    while todo:
        c = todo.pop()
        if c not in spec:
            spec.add(c)
            todo.extend(sorted((deps[c] | private_users[c]) - spec))
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
    full_ns = {p for n in pkg_consts for p in prefixes(n)}
    spec_ns = {p for n in spec_consts for p in prefixes(n)}
    for m, mod in mods.items():
        sc = scopes(mod.code)
        for i, line in enumerate(mod.code):
            if mm := re.match(r"^namespace\s+(\S+)", line):
                full = ".".join(sc[i][0] + mm.group(1).split("."))
                ps = full.split(".")
                names = {".".join(ps[:k]) for k in range(1, len(ps) + 1)}
                full_ns |= names
                if side[m] != "proofs":
                    spec_ns |= names
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
                    for c in [".".join(ns[:k] + [t]) for k in range(len(ns), -1, -1)] + [f"{o}.{t}" for o in opens]:
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
                items, form = open_items(body)
                ns = sc[i][0]
                keep = []
                for it in items:
                    cands = [".".join(ns[:k] + [it]) for k in range(len(ns), -1, -1)]
                    found = [c for c in cands if c in full_ns]
                    if not found or any(c in spec_ns for c in cands) or not all(A.under([root], c) for c in found):
                        keep.append(it)
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
                if not keep:
                    spec_drop.update(range(i, e + 1))
                else:
                    tail = " in" if IN_END.search(body) else ""
                    edits[i] = f"open {form}{' '.join(keep)}{tail}"
                    spec_drop.update(range(i + 1, e + 1))
            elif line.startswith("variable") and not IN_END.search(line):
                e = commands_of(mod.lines, mod.code, i)
                names, refs = binders(" ".join(x.strip() for x in mod.code[i:e + 1])[len("variable"):])
                ns, opens = sc[i]
                bad = None
                for t in (r.lstrip("@") for r in refs if r.lstrip("@") not in local_names):
                    c = next((c for c in [".".join(ns[:k] + [t]) for k in range(len(ns), -1, -1)] +
                              [f"{o}.{t}" for o in opens] if c in where), None)
                    if c and c not in spec_consts:
                        bad = c
                        break
                if bad:
                    spec_drop.update(range(i, e + 1))
                    gone |= set(names)
                    warnings.append(f"{m}:{i + 1}: the spec half leaves out a `variable` whose type names {bad}, a proof's")
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
            "halves": halves, "lock": lock, "layers": layers, "warnings": warnings, "counts": dict(sorted(counts.items())),
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
            if p["new_imports"][m] != imports_of(code):
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
