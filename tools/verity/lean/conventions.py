"""`conventions.py`: the form of the repository's Lean today, measured: naming, structure, organization, statement form.

    python tools/verity/lean/conventions.py [--json FILE] [--top N] [PACKAGE_DIR...]

Reads the sources of every Lake package that has a `lean-audit.json` (default: all of them under the repository, the
auditor's own fixtures aside) as text, with comments stripped as `check.py` strips them, and their locks, and prints counts
and the worst offenders: module path against namespace, where guarantees and assumptions live and how they are named,
casing, theorem name forms, retired words, file sizes, heartbeat overrides with and without a reason, imports, docstrings
on trusted declarations, duplicated statements, and how many guarantees have honest and nontrivial lemmas beside them.
It builds nothing and refuses nothing: it is the survey behind the Lean standards (LEAN-69), and what a check of one of
them would count. Text, not the elaborated environment: a declaration a macro writes is invisible, and a definition's
result is read from its header (`: Prop`), so a casing count is a lower bound.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("lean_check", HERE / "check.py")
check = sys.modules.setdefault(_spec.name, importlib.util.module_from_spec(_spec))
_spec.loader.exec_module(check)

#: the module roots of Security's trees; `Proofs.Properties` is the proofs of the properties
TREES = {"Definitions": "definitions", "Models": "definitions", "Proofs": "proofs", "Properties": "properties"}
#: namespaces of dependencies a package extends for dot notation (`Finset.sum_foo`), not its own
DEPENDENCY_ROOTS = {
    "Array", "BitVec", "Bool", "ByteArray", "Char", "Complex", "Decidable", "ENNReal", "EReal", "Equiv", "Fin", "Finite",
    "Finset", "Finsupp", "Fintype", "Function", "HashMap", "Int", "Lean", "List", "MeasureTheory", "Multiset", "NNReal",
    "Nat", "Option", "Ordinal", "PMF", "Polynomial", "Prod", "ProbabilityTheory", "Quotient", "Rat", "Real", "Set",
    "Std", "String", "Subtype", "Sum", "UInt16", "UInt32", "UInt64", "UInt8", "USize", "Vector", "ZMod", "Matrix",
    "OracleComp", "OracleSpec", "Sym2", "SimpleGraph", "Filter", "Cardinal", "Mathlib", "Encodable", "Prop", "Pi", "Sigma",
    "Except", "IO", "Thunk", "Ordering", "Nat.Prime", "Rel", "Relation", "Setoid", "Quot", "Order", "Monotone", "Antitone",
    "StateT", "ExceptT", "ReaderT", "Id", "Bitwise", "Float", "Subarray", "ByteSlice", "SProd", "Module", "Submodule",
    "LinearMap", "Mathlib.Meta", "Tactic", "Mul", "Add", "Fintype.card", "WithTop", "WithBot", "ENat",
    "Probability", "Dist", "Metric", "Nonempty", "HEq", "Eq", "Ne", "Iff", "And", "Or", "Not", "Exists",
}
#: what a declaration is, by its keyword
TYPE_KINDS = {"structure", "inductive", "class", "class inductive"}
DEF_KINDS = {"def", "abbrev", "opaque"}
THEOREM_KINDS = {"theorem", "lemma"}

DECL = re.compile(
    r"^(?P<indent>[ \t]*)(?:@\[[^\]\n]*\][ \t]*)*(?P<mods>(?:(?:private|protected|noncomputable|nonrec|partial|unsafe|"
    r"scoped|local)[ \t]+)*)(?P<kw>theorem|lemma|def|abbrev|structure|class[ \t]+inductive|class|inductive|instance|"
    r"opaque|axiom|example)\b(?P<rest>.*)$", re.M)
NAME = re.compile(r"[ \t]*(?:\(priority[^)]*\)[ \t]*)?(?P<name>[^\s:({\[⦃]+)")
SCOPE = re.compile(r"^[ \t]*(?:(?P<nc>noncomputable)[ \t]+)?(?P<kw>namespace|section|end|mutual)\b[ \t]*(?P<name>[^\s]*)",
                   re.M)
IMPORT = re.compile(r"^import[ \t]+([\w.«»']+)", re.M)
BANNED_CODE = re.compile(r"\b(sorry|native_decide)\b|#exit\b")
UNINFORMATIVE = re.compile(r"^(?:aux|helper|key|main|claim|step|lemma|thm|foo|bar|test|tmp|temp|h|this|goal|fact)\d*'*$|"
                           r"_aux\d*'*$|_helper\d*'*$")
#: Glossary terms retired on 25–26 Oct (ARCHITECTURE.md's Glossary, glossary/README.md in CMP-150's PR 1b)
RETIRED_PROSE = {
    "leg": re.compile(r"\blegs?\b", re.I),
    "warden": re.compile(r"\bwardens?\b", re.I),
    "tap": re.compile(r"\btaps?\b", re.I),
    "physical block": re.compile(r"\bphysical[ -]blocks?\b", re.I),
    "provider name": re.compile(r"\b(?:nebius|coreweave|aws|gcp|azure|lambda ?labs|runpod)\b", re.I),
    "proof unit": re.compile(r"\bproof[ -]units?\b", re.I),
    "isolation/replay/verification unit": re.compile(r"\b(?:isolation|replay|verification)[ -]units?\b", re.I),
}
#: the same words as an identifier's components (`legBound`, `warden_rows`, `NebiusHost`)
RETIRED_WORDS = {"leg": {"leg", "legs"}, "warden": {"warden", "wardens"}, "tap": {"tap", "taps"},
                 "provider name": {"nebius", "coreweave", "aws", "gcp", "azure"}}
#: the prefixes a block takes (Glossary, "Block": Daniel, 26 Oct)
BLOCK_PREFIXES = {"proof", "sampling", "server-time", "time", "serving", "server", "isolation", "replay", "inference",
                  "multiplication", "check", "flock", "c-flock", "flock's", "c-flock's"}
PROSE_BLOCK = re.compile(r"(?:\b([\w'-]+)[ \t]+)?\bblocks?\b", re.I)
WORDS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


@dataclass
class Decl:
    kind: str
    name: str
    full: str
    line: int
    doc: str | None
    result: str  # prop, type, data or unknown
    signature: str
    noncomputable: bool


@dataclass
class File:
    package: str
    rel: str  # from the repository's root
    module: str
    tree: str
    area: str
    lines: int
    imports: list[str]
    module_doc: bool
    decls: list[Decl] = field(default_factory=list)
    noncomputable_sections: int = 0
    banned: list[str] = field(default_factory=list)
    prose: str = ""


def packages(root: Path) -> list[Path]:
    """Every directory under ``root`` with a `lean-audit.json`, the auditor's fixtures and `.lake` aside."""
    out = []
    for p in sorted(root.rglob("lean-audit.json")):
        rel = p.relative_to(root).as_posix()
        if ".lake/" in rel or rel.startswith("tools/verity/lean/tests/"):
            continue
        out.append(p.parent)
    return out


def sources(pkg: Path, nested: list[Path]) -> list[Path]:
    out = []
    for p in sorted(pkg.rglob("*.lean")):
        if ".lake" in p.parts or any(n != pkg and n.is_relative_to(pkg) and p.is_relative_to(n) for n in nested):
            continue
        out.append(p)
    return out


def tree_and_area(pkg_rel: str, module: str) -> tuple[str, str]:
    parts = module.split(".")
    if pkg_rel.startswith("Security"):
        tree = TREES.get(parts[0], "other")
        rest = parts[1:]
        if parts[:2] == ["Proofs", "Properties"]:
            tree, rest = "proofs/properties", parts[2:]
        return tree, (rest[0] if rest else "(root)")
    return "code", pkg_rel.removeprefix("verity/core/")


def result_class(header: str) -> str:
    """What a declaration's header says its result is: `prop`, `type`, `data`, or `unknown` with no `:` at depth 0."""
    depth, colon = 0, -1
    for i, c in enumerate(header):
        if c in "([{⦃⟨":
            depth += 1
        elif c in ")]}⦄⟩":
            depth -= 1
        elif c == ":" and depth == 0 and header[i + 1:i + 2] != "=":
            colon = i
    if colon < 0:
        return "unknown"
    ty = header[colon + 1:].strip()
    last = re.split(r"→|->", ty)[-1].strip()
    if last == "Prop":
        return "prop"
    if re.match(r"(Type|Sort)\b", last):
        return "type"
    return "data"


HEADER_END = re.compile(r":=|\bwhere\b|^[ \t]*\||\n(?=\S)", re.M)
COMMENT = re.compile(r"/-[\s\S]*?-/|--[^\n]*")


def header_of(code: str, start: int) -> str:
    """The declaration's signature: from after its name to its `:=`, `where`, `|` or the next line at column 0."""
    window = code[start:start + 4000]
    m = HEADER_END.search(window)
    return " ".join(window[:m.start() if m else len(window)].split())


def doc_before(raw: str, pos: int) -> str | None:
    """The docstring (`/-- … -/`) right before a declaration at ``pos``, attributes and modifiers between them aside."""
    head = raw[:pos].rstrip()
    while True:
        m = re.search(r"@\[[^\]]*\]\s*$", head)
        if not m:
            break
        head = head[:m.start()].rstrip()
    if not head.endswith("-/"):
        return None
    start = head.rfind("/--")
    if start < 0 or "-/" in head[start + 3:-2]:
        return None
    return " ".join(head[start + 3:-2].split())


def src_dir(pkg: Path) -> Path:
    """The directory a package's module names are relative to: its lakefile's `srcDir` (`Security/Proofs`: `..`)."""
    lakefile = pkg / "lakefile.toml"
    if lakefile.exists():
        m = re.search(r'^srcDir\s*=\s*"([^"]*)"', lakefile.read_text(), re.M)
        if m:
            return (pkg / m.group(1)).resolve()
    return pkg


def parse(pkg: Path, pkg_rel: str, path: Path) -> File:
    raw = path.read_text(errors="replace")
    code = check.strip_comments(raw)
    rel_pkg = path.relative_to(pkg).as_posix()
    module = path.relative_to(src_dir(pkg)).as_posix().removesuffix(".lean").replace("/", ".")
    tree, area = tree_and_area(pkg_rel, module)
    f = File(pkg_rel, f"{pkg_rel}/{rel_pkg}", module, tree, area, raw.count("\n") + (not raw.endswith("\n")),
             IMPORT.findall(code), "/-!" in raw)
    f.prose = "\n".join(COMMENT.findall(raw))
    f.banned = [f"{code.count(chr(10), 0, m.start()) + 1}:{m.group(0)}" for m in BANNED_CODE.finditer(code)]
    # scopes and declarations in the order they appear; a scope is (keyword, name, noncomputable)
    events = [(m.start(), "scope", m) for m in SCOPE.finditer(code)] + [(m.start(), "decl", m) for m in DECL.finditer(code)]
    stack: list[tuple[str, str, bool]] = []
    for pos, what, m in sorted(events, key=lambda e: e[0]):
        if what == "scope":
            kw, name = m.group("kw"), m.group("name")
            if kw == "namespace":
                stack.append(("namespace", name, False))
            elif kw in ("section", "mutual"):
                nc = bool(m.group("nc")) and kw == "section"
                stack.append((kw, name, nc))
                f.noncomputable_sections += nc
            elif kw == "end" and stack:
                if name:
                    # `end A.B` closes `namespace A.B`, or the namespaces `A` and `B` opened one by one
                    want = name.split(".")
                    while want and stack:
                        _, n, _ = stack.pop()
                        got = n.split(".") if n else []
                        if not got or want[-len(got):] != got:
                            break
                        want = want[:-len(got)]
                else:
                    stack.pop()
            continue
        kw = " ".join(m.group("kw").split())
        rest = m.group("rest")
        nm = NAME.match(rest)
        name = nm.group("name") if nm and kw != "example" else ""
        ns = ".".join(n for k, n, _ in stack if k == "namespace" and n)
        full = name.removeprefix("_root_.") if name.startswith("_root_.") else ".".join(x for x in (ns, name) if x)
        sig = header_of(code, m.start("rest") + (nm.end() if name else 0))
        nc = "noncomputable" in m.group("mods") or any(x for _, _, x in stack)
        f.decls.append(Decl(kw, name, full, code.count("\n", 0, pos) + 1, doc_before(raw, line_start(raw, pos)),
                            "prop" if kw in THEOREM_KINDS else result_class(sig), sig, nc))
    return f


def line_start(raw: str, pos: int) -> int:
    """The start of the first attribute line above the declaration at ``pos`` (`@[simp]` on its own line)."""
    s = raw.rfind("\n", 0, pos) + 1
    while s > 0:
        prev = raw.rfind("\n", 0, s - 1) + 1
        if raw[prev:s].strip().startswith("@["):
            s = prev
        else:
            break
    return s


def words(name: str) -> list[str]:
    return [w.lower() for part in re.split(r"[._']", name) for w in WORDS.findall(part)]


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def dist(xs: list[int]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"n": len(s), "sum": sum(s), "p50": int(statistics.median(s)), "p90": q(0.9), "p99": q(0.99), "max": s[-1]}


def namespace_class(f: File, d: Decl) -> str:
    """How a declaration's namespace sits against its module's path: `path` (one is a prefix of the other, the tree's root
    module aside), `area` (the same area root, then apart), `extension` (a dependency's namespace, for dot notation),
    `none` (the root namespace) or `foreign` (another of the repository's roots)."""
    ns = d.full.split(".")[:-1]
    if not ns:
        return "none"
    mod = f.module.split(".")
    cands = [mod]
    if f.tree in ("definitions", "proofs", "properties"):
        cands.append(mod[1:])
    if f.tree == "proofs/properties":
        cands += [mod[1:], mod[2:]]
    for c in cands:
        k = min(len(c), len(ns))
        if k and c[:k] == ns[:k]:
            return "path"
    area = f.area if f.tree != "code" else mod[0]
    if ns[0] == area or (f.tree == "proofs/properties" and ns[0] == "Properties"):
        return "area"
    if ns[0] in DEPENDENCY_ROOTS or ".".join(ns[:2]) in DEPENDENCY_ROOTS:
        return "extension"
    return "foreign"


def load(root: Path, pkgs: list[Path]) -> tuple[list[File], dict[str, dict]]:
    """Every source file of ``pkgs`` parsed, and each package's lock, keyed by its path under ``root``."""
    files: list[File] = []
    locks: dict[str, dict] = {}
    for pkg in pkgs:
        pkg_rel = pkg.relative_to(root).as_posix()
        locks[pkg_rel] = json.loads((pkg / "lean-audit.json").read_text())
        for path in sources(pkg, pkgs):
            files.append(parse(pkg, pkg_rel, path))
    return files, locks


def overrides(root: Path, pkgs: list[Path], files: list[File]) -> list[dict]:
    """Each heartbeat override, as the Lean check lists it (`check.heartbeats`), with its package and tree."""
    tree = {f.rel: f.tree for f in files}
    out = []
    for pkg in pkgs:
        pkg_rel = pkg.relative_to(root).as_posix()
        mods = {f.module: f.rel.removeprefix(pkg_rel + "/") for f in files if f.package == pkg_rel}
        out += [h | {"package": pkg_rel, "tree": tree[f"{pkg_rel}/{h['file']}"]} for h in check.heartbeats(pkg, mods, {})]
    return out


def survey(root: Path, pkgs: list[Path], top: int) -> dict:
    files, locks = load(root, pkgs)
    by_full: dict[str, list[tuple[File, Decl]]] = defaultdict(list)
    for f in files:
        for d in f.decls:
            if d.full:
                by_full[d.full].append((f, d))
    trees = sorted({f.tree for f in files})
    out: dict = {"files": len(files), "packages": len(pkgs), "decls": sum(len(f.decls) for f in files)}

    # ---- naming: module path against namespace
    ns = {t: Counter() for t in trees}
    worst_ns: Counter = Counter()
    roots_by_area: dict[str, Counter] = defaultdict(Counter)
    foreign_roots: Counter = Counter()
    for f in files:
        for d in f.decls:
            if not d.name or d.kind in ("example", "instance"):
                continue
            c = namespace_class(f, d)
            ns[f.tree][c] += 1
            parts = d.full.split(".")
            if len(parts) > 1:
                roots_by_area[f"{f.tree}:{f.area}"][parts[0]] += 1
            if c in ("foreign", "none"):
                worst_ns[f.rel] += 1
                foreign_roots[parts[0] if c == "foreign" else "(root namespace)"] += 1
    out["namespace"] = {"by_tree": {t: dict(c) for t, c in ns.items()}, "worst_files": worst_ns.most_common(top),
                        "foreign_roots": foreign_roots.most_common(top),
                        "area_roots": {a: c.most_common(6) for a, c in sorted(roots_by_area.items())
                                       if len(c) > 1 or next(iter(c)) != a.split(":")[1]}}

    # ---- naming: casing and theorem forms
    case = Counter()
    case_ex: dict[str, list[str]] = defaultdict(list)
    forms = Counter()
    form_ex: dict[str, list[str]] = defaultdict(list)
    n_thm = 0

    def flag(key: str, f: File, d: Decl, into=case, ex=case_ex):
        into[key] += 1
        if len(ex[key]) < top:
            ex[key].append(f"{d.full} ({f.rel}:{d.line})")

    for f in files:
        for d in f.decls:
            last = d.full.split(".")[-1] if d.full else ""
            if not last or d.kind in ("example", "instance", "axiom"):
                continue
            upper = last[:1].isupper()
            if d.kind in TYPE_KINDS:
                case["type: total"] += 1
                if not upper:
                    flag("type: lowerCamel", f, d)
            elif d.kind in DEF_KINDS:
                case[f"def ({d.result}): total"] += 1
                if d.result in ("prop", "type") and not upper:
                    flag(f"def ({d.result}): lowerCamel", f, d)
                elif d.result == "data" and upper:
                    flag("def (data): UpperCamel", f, d)
                if not upper and "_" in last.strip("_"):
                    flag("def: snake_case", f, d)
            elif d.kind in THEOREM_KINDS:
                n_thm += 1
                parent = d.full.split(".")[-2] if "." in d.full else ""
                if upper:
                    if parent == "SecurityProofs":
                        flag("theorem: UpperCamel, SecurityProofs.G", f, d)
                    elif re.search(r"_(fails|meets)\b", last):
                        flag("theorem: UpperCamel, claim witness", f, d)
                    else:
                        flag("theorem: UpperCamel, other", f, d)
                elif "_" not in last and re.search(r"[a-z][A-Z]", last) and len(WORDS.findall(last)) >= 3:
                    flag("theorem: camelCase, no underscore", f, d)
                for key, pat in (("_of_", r"_of_"), ("_iff", r"_iff(_|$)"), ("_le/_lt/_ge/_eq/_ne", r"_(le|lt|ge|eq|ne)(_|'|$)"),
                                 ("_honest", r"honest"), ("_nontrivial", r"nontrivial"), ("_fails/_meets", r"_(fails|meets)"),
                                 ("_sound/_complete", r"_(sound|complete)"), ("primed (')", r"'$"),
                                 ("numbered (foo2)", r"[A-Za-z]\d+'*$"), ("uninformative (aux, key, step3)", UNINFORMATIVE.pattern)):
                    if re.search(pat, last):
                        flag(key, f, d, forms, form_ex)
    out["casing"] = {"counts": dict(case), "examples": dict(case_ex)}
    out["theorem_forms"] = {"theorems": n_thm, "counts": dict(forms), "examples": dict(form_ex)}

    # ---- naming: retired words, in identifiers and in prose
    retired_id: Counter = Counter()
    retired_id_ex: dict[str, list[str]] = defaultdict(list)
    retired_prose: Counter = Counter()
    retired_prose_files: dict[str, Counter] = defaultdict(Counter)
    block_prev: Counter = Counter()
    for f in files:
        names = [(d.full, d.line) for d in f.decls if d.full] + [(f.module, 0)]
        for full, line in names:
            w = words(full)
            hits = {key for key, ws in RETIRED_WORDS.items() if ws & set(w)}
            for i, x in enumerate(w):
                if x in ("block", "blocks"):
                    prev = w[i - 1] if i else ""
                    if prev == "physical":
                        hits.add("physical block")
                    elif prev not in BLOCK_PREFIXES:
                        hits.add("unprefixed block")
            for h in hits:
                retired_id[h] += 1
                if len(retired_id_ex[h]) < top:
                    retired_id_ex[h].append(f"{full} ({f.rel}{':' + str(line) if line else ''})")
        for key, pat in RETIRED_PROSE.items():
            n = len(pat.findall(f.prose))
            retired_prose[key] += n
            if n:
                retired_prose_files[key][f.rel] += n
        for m in PROSE_BLOCK.finditer(f.prose):
            prev = (m.group(1) or "").lower()
            if prev not in BLOCK_PREFIXES:
                retired_prose["unprefixed block"] += 1
                retired_prose_files["unprefixed block"][f.rel] += 1
                block_prev[prev or "(none)"] += 1
    out["retired"] = {"identifiers": dict(retired_id), "identifier_examples": dict(retired_id_ex),
                      "prose": dict(retired_prose),
                      "prose_files": {k: {"files": len(c), "worst": c.most_common(5)} for k, c in retired_prose_files.items()},
                      "prose_block_preceded_by": block_prev.most_common(15)}

    # ---- structure: sizes, imports, heartbeats, noncomputable, banned tokens
    sizes = {t: dist([f.lines for f in files if f.tree == t]) for t in trees}
    out["file_lines"] = {"by_tree": sizes, "over": {n: sum(f.lines > n for f in files) for n in (500, 1000, 2000, 5000)},
                         "over_by_tree": {t: {n: sum(f.lines > n for f in files if f.tree == t) for n in (500, 1000, 2000)}
                                          for t in trees},
                         "worst": [(f.rel, f.lines, len(f.decls)) for f in sorted(files, key=lambda f: -f.lines)[:top]]}
    out["decls_per_file"] = {"by_tree": {t: dist([len(f.decls) for f in files if f.tree == t]) for t in trees},
                             "over_300": sum(len(f.decls) > 300 for f in files)}
    out["imports"] = {"by_tree": {t: dist([len(f.imports) for f in files if f.tree == t]) for t in trees},
                      "wholesale_mathlib": {t: sum("Mathlib" in f.imports for f in files if f.tree == t) for t in trees},
                      "wholesale_mathlib_files": [f.rel for f in files if "Mathlib" in f.imports and f.tree in
                                                  ("definitions", "code")][:top],
                      "worst": [(f.rel, len(f.imports)) for f in sorted(files, key=lambda f: -len(f.imports))[:top]]}
    hb = overrides(root, pkgs, files)
    out["heartbeats"] = {
        "overrides": len(hb), "with_reason": sum(bool(h["reason"]) for h in hb), "unlimited (0)": sum(h["value"] == 0 for h in hb),
        "by_tree": {t: {"overrides": sum(h["tree"] == t for h in hb),
                        "no_reason": sum(h["tree"] == t and not h["reason"] for h in hb)} for t in trees},
        "files_with_override": len({(h["package"], h["file"]) for h in hb}),
        "max_value": max((h["value"] for h in hb), default=0),
        "worst_files_no_reason": Counter(f"{h['package']}/{h['file']}" for h in hb if not h["reason"]).most_common(top)}
    out["noncomputable"] = {t: {"sections": sum(f.noncomputable_sections for f in files if f.tree == t),
                                "files_with_section": sum(f.noncomputable_sections > 0 for f in files if f.tree == t),
                                "decls": sum(d.noncomputable for f in files if f.tree == t for d in f.decls)}
                            for t in trees}
    out["banned_tokens"] = [(f.rel, f.banned) for f in files if f.banned][:top]

    # ---- structure: duplicated statements across areas
    sigs: dict[str, list[tuple[File, Decl]]] = defaultdict(list)
    for f in files:
        for d in f.decls:
            if d.kind in THEOREM_KINDS and len(d.signature) >= 40:
                sigs[d.signature].append((f, d))
    dup = [(s, xs) for s, xs in sigs.items() if len({(x.package, x.area) for x, _ in xs}) > 1]
    dup.sort(key=lambda e: -len(e[1]))
    by_name: dict[str, list[tuple[File, Decl]]] = defaultdict(list)
    for f in files:
        for d in f.decls:
            if d.kind in THEOREM_KINDS and d.full:
                by_name[d.full.split(".")[-1]].append((f, d))
    dup_names = [(n, xs) for n, xs in by_name.items() if len({(x.package, x.area) for x, _ in xs}) > 1 and len(n) > 6]
    out["duplicates"] = {
        "statements_in_2+_areas": len(dup), "theorems_in_them": sum(len(xs) for _, xs in dup),
        "worst": [{"statement": s[:160], "at": [f"{d.full} ({f.rel}:{d.line})" for f, d in xs[:4]], "copies": len(xs)}
                  for s, xs in dup[:top]],
        "names_in_2+_areas": len(dup_names)}

    # ---- organization: areas per tree, theorems and docstrings in the definitions tree
    areas = {t: sorted({f.area for f in files if f.tree == t and f.area != "(root)"}) for t in trees if t != "code"}
    every = sorted(set().union(*areas.values())) if areas else []
    out["areas"] = {"by_tree": areas, "missing": {a: [t for t in areas if a not in areas[t]] for a in every
                                                  if any(a not in areas[t] for t in areas)}}
    defs_tree = [f for f in files if f.tree == "definitions"]
    thm_in_defs = Counter()
    for f in defs_tree:
        for d in f.decls:
            if d.kind in THEOREM_KINDS:
                thm_in_defs[f.rel] += 1
    reads = set()
    for lock in locks.values():
        for mod in (lock.get("reads") or {}).values():
            reads |= set(mod.get("definitions", {}))
    spec_defs = [(f, d) for f in defs_tree for d in f.decls if d.kind in DEF_KINDS | TYPE_KINDS and d.full]
    unread = [(f, d) for f, d in spec_defs if d.full not in reads]
    unread_by_file = Counter(f.rel for f, _ in unread)
    out["definitions_tree"] = {
        "files": len(defs_tree), "decls": sum(len(f.decls) for f in defs_tree),
        "theorems": sum(thm_in_defs.values()), "files_with_theorems": len(thm_in_defs),
        "theorems_worst": thm_in_defs.most_common(top),
        "securityproofs_decls": sum(".SecurityProofs." in d.full for f in defs_tree for d in f.decls),
        "spec_definitions": len(spec_defs), "read_by_no_guarantee": len(unread),
        "unread_worst_files": unread_by_file.most_common(top)}

    def docs(fs: list[File]) -> dict:
        ds = [d for f in fs for d in f.decls if d.kind in DEF_KINDS | TYPE_KINDS and d.name]
        return {"decls": len(ds), "no_docstring": sum(d.doc is None for d in ds)}

    is_g = lambda f: "Guarantees" in f.module.split(".")
    is_a = lambda f: "Assumptions" in f.module.split(".")
    nodoc = Counter(f.rel for f in defs_tree for d in f.decls if d.kind in DEF_KINDS | TYPE_KINDS and d.name and d.doc is None)
    out["docstrings"] = {
        "definitions_tree": docs(defs_tree), "guarantees_modules": docs([f for f in defs_tree if is_g(f)]),
        "assumptions_modules": docs([f for f in defs_tree if is_a(f)]),
        "vocabulary_modules": docs([f for f in defs_tree if not is_g(f) and not is_a(f)]),
        "properties_tree": docs([f for f in files if f.tree == "properties"]),
        "code": docs([f for f in files if f.tree == "code"]),
        "no_module_doc": {t: sum(not f.module_doc for f in files if f.tree == t) for t in trees},
        "worst_definitions_files": nodoc.most_common(top)}

    # ---- the locks: where guarantees and assumptions live, their names, honest/nontrivial lemmas
    theorems_lower = defaultdict(set)
    for f in files:
        for d in f.decls:
            if d.kind in THEOREM_KINDS and d.full:
                theorems_lower[d.full.split(".")[0]].add(d.full.lower())
    all_thm_lower = {d.full.lower() for f in files for d in f.decls if d.kind in THEOREM_KINDS and d.full}
    honest_names = [n for n in all_thm_lower if "honest" in n]
    nontrivial_names = [n for n in all_thm_lower if "nontrivial" in n]
    lk = {}
    for pkg_rel, lock in locks.items():
        gs = lock.get("guarantees") or {}
        if not gs and not lock.get("assumptions"):
            continue
        frozen = (lock.get("shape") or {}).get("frozen", {})
        g_form, g_tree, g_root = Counter(), Counter(), Counter()
        both = one = 0
        for g in gs:
            parts = g.split(".")
            g_root[parts[0]] += 1
            if "SecurityProofs" in parts[1:-1]:
                # `check.four_named`'s form, read from the text: `X.SecurityProofs.G` beside a definition `X.Guarantees.G`
                at = parts.index("SecurityProofs", 1)
                spec = ".".join([*parts[:at], "Guarantees", *parts[at + 1:]])
                g_form["four names" if spec in by_full else "X.SecurityProofs.G, no X.Guarantees.G"] += 1
            else:
                g_form["other"] += 1
            where = by_full.get(g)
            g_tree[where[0][0].tree if where else "not found in text"] += 1
            key = parts[-1].lower()
            h = any(key in n for n in honest_names)
            nt = any(key in n for n in nontrivial_names)
            both += h and nt
            one += h != nt
        asm = lock.get("assumptions") or {}
        a_form, a_mod, a_doc = Counter(), Counter(), Counter()
        for a in asm:
            a_form["<Area>.Assumptions.X" if ".Assumptions." in a else "other"] += 1
            where = by_full.get(a)
            if not where:
                a_mod["not found in text"] += 1
                continue
            f, d = where[0]
            a_mod["Assumptions module" if is_a(f) else f"{f.tree}, not an Assumptions module"] += 1
            a_doc["docstring" if d.doc else "no docstring"] += 1
        lk[pkg_rel] = {"guarantees": len(gs), "frozen": len(frozen),
                       "frozen_by_category": dict(Counter(v.split()[0].strip(":,;") for v in frozen.values())),
                       "forms": dict(g_form), "declared_in": dict(g_tree), "roots": dict(g_root.most_common()),
                       "honest_and_nontrivial": both, "only_one": one, "neither": len(gs) - both - one,
                       "four_names_roots": lock.get("four_names"), "assumptions": len(asm), "assumption_names": dict(a_form),
                       "assumption_modules": dict(a_mod), "assumption_docstrings": dict(a_doc)}
    out["locks"] = lk
    out["honest_nontrivial_theorems"] = {"honest": len(honest_names), "nontrivial": len(nontrivial_names)}
    return out


def render(r: dict, top: int) -> str:
    L = [f"{r['packages']} packages, {r['files']} files, {r['decls']} declarations (text)"]
    sec = lambda t: L.append(f"\n== {t}")
    sec("namespace against module path (declarations; path: one a prefix of the other, the tree root aside)")
    for t, c in r["namespace"]["by_tree"].items():
        n = sum(c.values())
        L.append(f"  {t:18} {n:6}  " + "  ".join(f"{k} {v} ({pct(v, n)})" for k, v in sorted(c.items(), key=lambda e: -e[1])))
    L.append("  foreign roots: " + ", ".join(f"{k} {v}" for k, v in r["namespace"]["foreign_roots"]))
    L.append("  areas whose declarations use another root: " + "; ".join(
        f"{a}: " + ", ".join(f"{k} {v}" for k, v in c) for a, c in list(r["namespace"]["area_roots"].items())[:30]))
    L.append("  worst files (foreign or root): " + ", ".join(f"{k} {v}" for k, v in r["namespace"]["worst_files"]))
    sec("casing")
    for k, v in sorted(r["casing"]["counts"].items()):
        L.append(f"  {k:42} {v}")
    for k, ex in r["casing"]["examples"].items():
        L.append(f"  e.g. {k}: " + "; ".join(ex[:5]))
    sec(f"theorem name forms ({r['theorem_forms']['theorems']} theorems)")
    for k, v in sorted(r["theorem_forms"]["counts"].items(), key=lambda e: -e[1]):
        L.append(f"  {k:34} {v:6} ({pct(v, r['theorem_forms']['theorems'])})")
    for k in ("uninformative (aux, key, step3)", "numbered (foo2)"):
        L.append(f"  e.g. {k}: " + "; ".join(r["theorem_forms"]["examples"].get(k, [])[:6]))
    sec("retired words")
    L.append("  identifiers (declaration names, module paths): " + json.dumps(r["retired"]["identifiers"]))
    for k, ex in r["retired"]["identifier_examples"].items():
        L.append(f"    {k}: " + "; ".join(ex[:6]))
    L.append("  prose (docstrings, comments): " + json.dumps(r["retired"]["prose"]))
    for k, v in r["retired"]["prose_files"].items():
        L.append(f"    {k}: {v['files']} files, worst " + ", ".join(f"{a} {b}" for a, b in v["worst"]))
    L.append("    'block' preceded by: " + ", ".join(f"{a} {b}" for a, b in r["retired"]["prose_block_preceded_by"]))
    sec("file lines")
    for t, d in r["file_lines"]["by_tree"].items():
        L.append(f"  {t:18} {json.dumps(d)}  over 500/1000/2000: {json.dumps(r['file_lines']['over_by_tree'][t])}")
    L.append(f"  over 500/1000/2000/5000 lines: {json.dumps(r['file_lines']['over'])}")
    L.append("  worst: " + ", ".join(f"{a} {b} lines/{c} decls" for a, b, c in r["file_lines"]["worst"]))
    L.append(f"  declarations per file: {json.dumps(r['decls_per_file'])}")
    sec("imports")
    for t, d in r["imports"]["by_tree"].items():
        L.append(f"  {t:18} {json.dumps(d)}  `import Mathlib`: {r['imports']['wholesale_mathlib'][t]}")
    L.append("  `import Mathlib` in definitions/code: " + ", ".join(r["imports"]["wholesale_mathlib_files"]))
    L.append("  most imports: " + ", ".join(f"{a} {b}" for a, b in r["imports"]["worst"]))
    sec("heartbeat overrides")
    h = r["heartbeats"]
    L.append(f"  {h['overrides']} overrides in {h['files_with_override']} files, {h['with_reason']} with a reason "
             f"({pct(h['with_reason'], h['overrides'])}), {h['unlimited (0)']} unlimited, max {h['max_value']}")
    L.append("  by tree: " + json.dumps(h["by_tree"]))
    L.append("  most without a reason: " + ", ".join(f"{a} {b}" for a, b in h["worst_files_no_reason"]))
    sec("noncomputable, banned tokens")
    L.append("  " + json.dumps(r["noncomputable"]))
    L.append("  sorry/native_decide/#exit in code: " + json.dumps(r["banned_tokens"]))
    sec("duplicated theorem statements across areas")
    d = r["duplicates"]
    L.append(f"  {d['statements_in_2+_areas']} statements in two or more areas ({d['theorems_in_them']} theorems); "
             f"{d['names_in_2+_areas']} theorem names in two or more areas")
    for x in d["worst"][:top]:
        L.append(f"    {x['copies']}x {x['statement'][:100]} | {'; '.join(x['at'][:3])}")
    sec("areas per tree")
    for t, a in r["areas"]["by_tree"].items():
        L.append(f"  {t:18} {', '.join(a)}")
    L.append("  missing: " + json.dumps(r["areas"]["missing"]))
    sec("the definitions tree (trusted)")
    dt = r["definitions_tree"]
    L.append(f"  {dt['files']} files, {dt['decls']} declarations; {dt['theorems']} theorems in {dt['files_with_theorems']} "
             f"files; {dt['securityproofs_decls']} SecurityProofs declarations; {dt['spec_definitions']} definitions, "
             f"{dt['read_by_no_guarantee']} read by no guarantee ({pct(dt['read_by_no_guarantee'], dt['spec_definitions'])})")
    L.append("  most theorems: " + ", ".join(f"{a} {b}" for a, b in dt["theorems_worst"]))
    L.append("  most unread definitions: " + ", ".join(f"{a} {b}" for a, b in dt["unread_worst_files"]))
    sec("docstrings on definitions (def, abbrev, structure, inductive, class)")
    for k in ("definitions_tree", "guarantees_modules", "assumptions_modules", "vocabulary_modules", "properties_tree", "code"):
        x = r["docstrings"][k]
        L.append(f"  {k:20} {x['decls']:6} decls, {x['no_docstring']:5} without ({pct(x['no_docstring'], x['decls'])})")
    L.append("  files without a module docstring: " + json.dumps(r["docstrings"]["no_module_doc"]))
    L.append("  worst definitions files: " + ", ".join(f"{a} {b}" for a, b in r["docstrings"]["worst_definitions_files"]))
    sec("locks: guarantees and assumptions")
    for p, x in r["locks"].items():
        L.append(f"  {p}: {x['guarantees']} guarantees ({x['frozen']} frozen: {json.dumps(x['frozen_by_category'])})")
        L.append(f"    forms {json.dumps(x['forms'])}; declared in {json.dumps(x['declared_in'])}")
        L.append(f"    roots {json.dumps(x['roots'])}")
        L.append(f"    honest and nontrivial lemmas: both {x['honest_and_nontrivial']}, one {x['only_one']}, "
                 f"neither {x['neither']}")
        L.append(f"    {x['assumptions']} assumptions: names {json.dumps(x['assumption_names'])}; in "
                 f"{json.dumps(x['assumption_modules'])}; {json.dumps(x['assumption_docstrings'])}")
    L.append(f"  theorems named honest/nontrivial anywhere: {json.dumps(r['honest_nontrivial_theorems'])}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("packages", nargs="*", type=Path, help="package directories (default: every one with a lean-audit.json)")
    ap.add_argument("--root", type=Path, default=REPO, help="the checkout to survey (default: this one), e.g. a worktree "
                    "of an older commit, to compare")
    ap.add_argument("--json", type=Path, help="also write the full survey here")
    ap.add_argument("--top", type=int, default=10, help="worst offenders listed per measure")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    pkgs = [p.resolve() for p in a.packages] or packages(root)
    r = survey(root, pkgs, a.top)
    print(render(r, a.top))
    if a.json:
        a.json.write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
