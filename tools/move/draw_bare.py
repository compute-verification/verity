#!/usr/bin/env python3
"""The bare words of the retired DRAW family in memory accounting's Lean and Python (`draw`, `drawn`, `draws`, `Draw`,
`Drawn`), which `draw_compound_map.toml` leaves to each owner, and their rename with `draw_bare_map.toml`.

    python3 tools/move/draw_bare.py census [--json]   every site by directory and kind, the `Draw` namespaces declared
                                                      and the paths that spell a word
    python3 tools/move/draw_bare.py plan [--write]    which words each file of ours may take by word (`[code.files]`)
                                                      and which files are renamed by hand (`[code.by_hand]`, each with
                                                      its reason); `--write` writes both into the map
    python3 tools/move/draw_bare.py code [--dry-run]  rename `[code.files]`' words in code and backticked names alone,
                                                      and `[code.tokens]`' names, the hand edits a build needs

The order: `rename.py --map tools/move/draw_bare_map.toml` (the namespaces, modules and paths), then `plan --write` on
the moved tree, then `code`, then the `by_hand` files; build the Lean after each, and read `resolve_check.py` first.

A site's kind is `rename.py`'s reading of where it stands (`regions`). In code: ours (a declaration or a module of ours,
or a local), others' (a name another owner declares, `[code] keep`), or by type (a field read `x.draw`, whose owner is
`x`'s structure). In a string literal: kept (the map's `[keep]`: a wire key, a reason prefix, a coin family) or output
(any other). In a comment or a docstring: prose, or a quoted name in backticks.

A file takes a word by `code` only when nothing else can own the word there: in Lean, no other owner's module that
declares it is reachable through the file's imports, and no file outside ours that reaches the file spells it; in
Python, the file imports no module outside ours that declares it, no file outside ours imports the file and spells it,
and the file serializes no record by its fields' names; and in either, the word's new word is no name in the file's code
yet. A word one of our files takes by hand is by hand in every file of ours linked to it by an import, either way, where
either file declares it. It reads source text and elaborates nothing, so the build is the authority. Run from the
repository's root.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rename  # noqa: E402

MAP = HERE / "draw_bare_map.toml"
SCOPE = [f"security/{t}/{a}{s}" for t in ("Definitions", "Proofs", "Properties", "Proofs/Properties")
         for a in ("Accounting", "Pous", "Poub") for s in ("/*.lean", ".lean")]
SCOPE += [f"security/{t}/Compose/{f}" for t in ("Definitions", "Proofs", "Properties", "Proofs/Properties")
          for f in ("Remote*", "Assumptions/Remote*", "Guarantees/Remote*", "Stacking*", "Guarantees/Stacking*")]
SCOPE += [f"verity/protocols/accounting/{p}/*.{e}" for p in ("pous", "poub") for e in ("lean", "py")]
#: PoUW's files in the accounting area, its lane's own
POUW = re.compile(r"^security/(?:Definitions|Properties|Proofs|Proofs/Properties)/Accounting/"
                  r"(?:(?:Assumptions|Guarantees|Lemmas)/)?(?:TS|Headline|Compose|HMu|Pouw)(?:[./]|[A-Z])")
WORDS = frozenset(("draw", "drawn", "draws", "Draw", "Drawn"))
WORD = re.compile(r"(?<![\w'])(draw|drawn|draws|Draw|Drawn)(?![\w'!?])")
TOKEN = re.compile(r"(?<![\w'.!?])((?:[A-Za-z_][\w'!?]*\.)*[A-Za-z_][\w'!?]*)")
FIELD = re.compile(r"\.(?:draw|drawn|draws)$")
NAMESPACE = re.compile(r"^\s*namespace\s+(\S+)", re.M)
IMPORT = re.compile(r"^(?:public\s+)?(?:meta\s+)?import\s+(?:all\s+)?(\S+)", re.M)
DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|public|noncomputable|partial|unsafe|nonrec|scoped|"
                  r"local|meta)\s+)*(?:def|theorem|lemma|abbrev|structure|inductive|class|instance|opaque|axiom)\s+"
                  r"([^\s(:{\[]+)", re.M)
LEAN_FIELD = re.compile(r"^\s+(\w+)\s*:", re.M)
HYPOTHESIS = re.compile(r"^\s*(?:from|import)\s+hypothesis\b", re.M)
SERIALIZES = re.compile(r"\basdict\(|\bfields\(|__dict__|\bvars\(")


def load() -> dict:
    return tomllib.loads(MAP.read_text())


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True, check=True).stdout.split("\0")
    return [f for f in out if f and Path(f).is_file()]


@cache
def excluded() -> tuple[str, ...]:
    return tuple(load()["rewrite"]["exclude"])


def ours(f: str) -> bool:
    return (any(fnmatch.fnmatchcase(f, g) for g in SCOPE) and not POUW.match(f)
            and not any(rename._match(f, g) for g in excluded()))


def keeps(data: dict) -> list[re.Pattern[str]]:
    out = [re.compile(p) for p in data["keep"].get("patterns", [])]
    return out + [re.compile(re.escape(s)) for s in data["keep"].get("strings", [])]


@cache
def text(f: str) -> str:
    return Path(f).read_text()


@cache
def spans(f: str) -> list[tuple[int, int, str]]:
    return rename.regions(f, text(f))


def is_code(f: str, i: int) -> bool:
    return rename.kind_at(spans(f), i) == rename.CODE


def quoted(f: str, i: int) -> bool:
    """Whether position `i` of `f` stands in a name quoted in backticks in prose."""
    sp = spans(f)
    if rename.kind_at(sp, i) != rename.LITERAL:
        return False
    a = next(a for a, b, _ in sp if a <= i < b)
    return text(f)[a] == "`"


def token_at(t: str, s: int) -> str:
    m = next((m for m in TOKEN.finditer(t, max(0, s - 200), s + 80) if m.start() <= s < m.end()), None)
    return m.group(1) if m else ""


def kind(f: str, s: int, e: int, keep: list[re.Pattern[str]], other: re.Pattern[str]) -> str:
    t = text(f)
    if is_code(f, s):
        tok = token_at(t, s)
        if other.search(tok):
            return "others'"
        if FIELD.search(tok) and not tok.startswith(("Accounting.", "Proofs.", "Definitions.", "Properties.")):
            return "by type"
        return "ours"
    if rename.kind_at(spans(f), s) == rename.PROSE:
        return "prose"
    if quoted(f, s):
        return "quoted name"
    a, b = next((a, b) for a, b, _ in spans(f) if a <= s < b)
    if any(m.start() <= s and e <= m.end() for p in keep for m in p.finditer(t, a, b)):
        return "kept"
    return "output"


def census() -> dict:
    data = load()
    keep, other = keeps(data), re.compile("|".join(data["code"]["keep"]))
    by_dir: dict[str, Counter] = defaultdict(Counter)
    namespaces, paths = [], []
    for f in tracked():
        if not (ours(f) and f.endswith((".lean", ".py"))):
            continue
        if WORD.fullmatch(Path(f).stem):
            paths.append(f)
        for m in WORD.finditer(text(f)):
            by_dir[str(Path(f).parent)][kind(f, m.start(), m.end(), keep, other)] += 1
        for m in NAMESPACE.finditer(text(f)):
            if WORD.search(m.group(1)) and is_code(f, m.start(1)):
                namespaces.append(f"{f}:{text(f).count(chr(10), 0, m.start()) + 1}: namespace {m.group(1)}")
    total: Counter = Counter()
    for c in by_dir.values():
        total += c
    return {"dirs": {d: dict(c) for d, c in sorted(by_dir.items())}, "total": dict(total), "namespaces": namespaces,
            "paths": paths}


def code_words(f: str) -> set[str]:
    """The bare words `f` spells in code or in a quoted name."""
    return {m.group(1) for m in WORD.finditer(text(f)) if is_code(f, m.start()) or quoted(f, m.start())}


def spells(f: str, w: str) -> bool:
    return any(is_code(f, m.start()) for m in re.finditer(rf"(?<![\w']){w}(?![\w'!?])", text(f)))


def lean_modules(files: list[str]) -> dict[str, str]:
    """Each Lean module's file: under `security/` from there, elsewhere from the nearest directory with a lakefile."""
    roots = sorted({str(Path(f).parent) for f in files if f.endswith("lakefile.toml") and "/tests/fixtures/" not in f},
                   key=len, reverse=True)
    out: dict[str, str] = {}
    for f in files:
        if not f.endswith(".lean"):
            continue
        root = "security" if f.startswith("security/") else next((r for r in roots if f.startswith(r + "/")), None)
        if root is not None:
            out.setdefault(f[len(root) + 1:-5].replace("/", "."), f)
    return out


def lean_declares(f: str, module: str = "") -> set[str]:
    """The bare words `f` declares in code: a declaration's last component, a structure's field, a component of a
    namespace it opens, or one of its module name `module`."""
    t = text(f)
    out = {m.group(1).split(".")[-1] for m in DECL.finditer(t) if is_code(f, m.start(1))}
    out |= {m.group(1) for m in LEAN_FIELD.finditer(t) if is_code(f, m.start(1))}
    out |= {c for m in NAMESPACE.finditer(t) if is_code(f, m.start(1)) for c in m.group(1).split(".")}
    return (out | set(module.split("."))) & WORDS


def py_tree(f: str) -> ast.AST:
    return ast.parse(text(f), filename=f)


def py_declares(f: str) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(py_tree(f)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
            out.add(n.attr)
        elif isinstance(n, ast.Assign):
            out |= {x.id for x in n.targets if isinstance(x, ast.Name)}
    return out & WORDS


def py_imports(f: str) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(py_tree(f)):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            out |= {n.module} | {f"{n.module}.{a.name}" for a in n.names}
    return out


def plan() -> tuple[dict[str, list[str]], dict[str, str]]:
    files = tracked()
    lean = [f for f in files if f.endswith(".lean")]
    mods = lean_modules(files)
    imports = {f: [mods[m] for m in IMPORT.findall(text(f)) if m in mods] for f in lean}

    @cache
    def reach(f: str) -> frozenset[str]:
        seen: set[str] = set()
        stack = list(imports.get(f, []))
        while stack:
            if (x := stack.pop()) not in seen:
                seen.add(x)
                stack += imports.get(x, [])
        return frozenset(seen)

    module_of = {f: m for m, f in mods.items()}
    declared = {f: d for f in lean if not ours(f) and (WORD.search(text(f)) or WORD.search(module_of.get(f, "")))
                and (d := lean_declares(f, module_of.get(f, "")))}
    reached_by: dict[str, set[str]] = defaultdict(set)
    for f in lean:
        if not ours(f) and WORD.search(text(f)):
            for g in reach(f):
                reached_by[g].add(f)

    py = [f for f in files if f.endswith(".py")]
    by_module: dict[str, str] = {}
    for f in py:
        parts = f[:-3].removesuffix("/__init__").split("/")
        for i in range(len(parts)):
            by_module.setdefault(".".join(parts[i:]), f)
    py_declared = {f: d for f in py if not ours(f) and WORD.search(text(f)) and (d := py_declares(f))}
    py_users: dict[str, set[str]] = defaultdict(set)
    for f in py:
        if not ours(f) and WORD.search(text(f)):
            for m in py_imports(f):
                if m in by_module:
                    py_users[by_module[m]].add(f)

    new = load()["code"]["words"]
    safe: dict[str, list[str]] = {}
    hand: dict[str, str] = {}
    whys: dict[str, tuple[set[str], dict[str, str]]] = {}
    for f in sorted(x for x in files if ours(x) and x.endswith((".lean", ".py"))):
        if not (words := code_words(f)):
            continue
        why: dict[str, str] = {}
        for w in sorted(words):
            if any(is_code(f, m.start()) for m in re.finditer(rf"(?<![\w'.]){new[w]}(?![\w'!?])", text(f))):
                why[w] = f"`{new[w]}` is already a name in its code"
        if f.endswith(".lean"):
            for g in sorted(reach(f)):
                for w in sorted(words & declared.get(g, set())):
                    why.setdefault(w, f"reaches {g}, which declares `{w}`")
            for g in sorted(reached_by.get(f, ())):
                for w in sorted(words - set(why)):
                    if spells(g, w):
                        why[w] = f"{g}, not ours, reaches it and spells `{w}`"
        else:
            for m in sorted(py_imports(f)):
                if (g := by_module.get(m)) and not ours(g):
                    for w in sorted(words & py_declared.get(g, set())):
                        why.setdefault(w, f"imports {g}, which declares `{w}`")
            for g in sorted(py_users.get(f, ())):
                for w in sorted(words - set(why)):
                    if spells(g, w):
                        why[w] = f"{g}, not ours, imports it and spells `{w}`"
            if "draw" in words and HYPOTHESIS.search(text(f)):
                why.setdefault("draw", "Hypothesis's `draw`, the library's name, kept")
            if SERIALIZES.search(text(f)):
                for w in sorted(words & py_declares(f)):
                    why.setdefault(w, f"serializes its records by field name and declares `{w}`")
        whys[f] = (words, why)

    def linked(f: str) -> set[str]:
        """Our files a word of `f`'s may name a declaration of, or be named from: those it reaches or imports, and
        those that reach or import it."""
        if f.endswith(".lean"):
            out = set(reach(f)) | {g for g in whys if g.endswith(".lean") and f in reach(g)}
        else:
            out = {by_module[m] for m in py_imports(f) if m in by_module}
            out |= {g for g in whys if g.endswith(".py") and any(by_module.get(m) == f for m in py_imports(g))}
        return {g for g in out if g in whys and g != f}
    links = {f: linked(f) for f in whys}
    by_last: dict[str, set[str]] = defaultdict(set)
    for g in whys:
        if g in module_of:
            by_last[module_of[g].rsplit(".", 1)[-1]].add(g)
    for f, (words, _) in whys.items():
        for q in re.findall(rf"\b([A-Z]\w*)\.(?:{'|'.join(map(re.escape, words))})\b", text(f)):
            links[f] |= by_last.get(q, set()) - {f}
    declares = {f: (lean_declares(f, module_of.get(f, "")) if f.endswith(".lean") else py_declares(f)) for f in whys}
    changed = True
    while changed:
        changed = False
        for f, (words, why) in whys.items():
            for g in sorted(links[f]):
                for w in sorted((words - set(why)) & set(whys[g][1])):
                    if w in declares[f] or w in declares[g]:
                        why[w], changed = f"{g}, ours, takes `{w}` by hand", True
    for f, (words, why) in whys.items():
        if ok := sorted(words - set(why)):
            safe[f] = ok
        if why:
            hand[f] = "; ".join(f"{w}: {r}" for w, r in sorted(why.items()))
    return safe, hand


def write_plan(safe: dict[str, list[str]], hand: dict[str, str]) -> None:
    head = re.split(r"\n\[code\.files\]\n", MAP.read_text())[0].rstrip("\n")
    body = "\n\n[code.files]\n" + "".join(f"{json.dumps(f)} = {json.dumps(w)}\n" for f, w in safe.items())
    body += "\n[code.by_hand]\n" + "".join(f"{json.dumps(f)} = {json.dumps(r)}\n" for f, r in hand.items())
    MAP.write_text(head + body)


def apply_code(dry: bool) -> int:
    data = load()
    keep = keeps(data) + [re.compile(p) for p in data["code"]["keep"]]
    new = data["code"]["words"]
    changed = sites = 0
    for f, words in data["code"].get("files", {}).items():
        t = text(f)
        out, last, n = [], 0, 0
        for m in WORD.finditer(t):
            if m.group(1) not in words or not (is_code(f, m.start()) or quoted(f, m.start())):
                continue
            lo, hi = max(0, m.start() - 80), m.end() + 80
            if any(x.start() <= m.start() and m.end() <= x.end() for p in keep for x in p.finditer(t, lo, hi)):
                continue
            out += [t[last:m.start()], new[m.group(1)]]
            last, n = m.end(), n + 1
        if n:
            changed, sites = changed + 1, sites + n
            print(f"{f}: {n}")
            if not dry:
                Path(f).write_text("".join(out) + t[last:])
    paths = tracked()
    for glob, toks in data["code"].get("tokens", {}).items():
        for f in (x for x in paths if fnmatch.fnmatchcase(x, glob) and ours(x)):
            t = text(f)
            out, last, n = [], 0, 0
            pat = re.compile("|".join(rf"(?<![\w'.]){re.escape(k)}(?![\w'!?])" for k in toks))
            for m in pat.finditer(t):
                if is_code(f, m.start()):
                    out += [t[last:m.start()], WORD.sub(lambda w: new[w.group(1)], m.group(0))]
                    last, n = m.end(), n + 1
            if n:
                changed, sites = changed + 1, sites + n
                print(f"{f}: {n} by hand")
                if not dry:
                    Path(f).write_text("".join(out) + t[last:])
    print(f"{sites} site(s) in {changed} file(s)" + (" (dry run)" if dry else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("census").add_argument("--json", action="store_true")
    sub.add_parser("plan").add_argument("--write", action="store_true")
    sub.add_parser("code").add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "code":
        return apply_code(a.dry_run)
    if a.cmd == "plan":
        safe, hand = plan()
        if a.write:
            write_plan(safe, hand)
        print(f"{len(safe)} file(s) take {sum(len(w) for w in safe.values())} (file, word) pairs by `code`; "
              f"{len(hand)} file(s) by hand")
        for f, r in hand.items():
            print(f"  {f}: {r}")
        return 0
    r = census()
    if a.json:
        print(json.dumps(r, indent=1))
        return 0
    kinds = ["ours", "others'", "by type", "quoted name", "prose", "kept", "output"]
    print("directory".ljust(66), *(k.rjust(11) for k in kinds))
    for d, c in r["dirs"].items():
        print(d.ljust(66), *(str(c.get(k, 0)).rjust(11) for k in kinds))
    print("total".ljust(66), *(str(r["total"].get(k, 0)).rjust(11) for k in kinds))
    for key in ("namespaces", "paths"):
        print(f"\n{key}:", *r[key], sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
