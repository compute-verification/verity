"""The modules of a Lake package that no listed guarantee uses, and their deletion, one area at a time (LEAN-73).

    python tools/verity/lean/unused.py --fetch-queued QUEUED.json
    python tools/verity/lean/unused.py [PKG] [--out DIR] [--used USED.json] [--queued QUEUED.json] [--frozen PREFIX ...]
    python tools/verity/lean/unused.py [PKG] --delete AREA --list DIR/unused.json --used DIR/used.json [--queued QUEUED.json]
                                       [--except MODULE ...]

PKG (default `Security/Proofs`) is a built Lake package with a `lean-audit.json`. `Used.lean`, compiled like `Facts.lean`,
reads its build as data (`--used`: a document it wrote before): from the names below, the closure of the constants their
types and proofs use, and each own module's constants, the ones that closure holds, and the modules its constants use. The
names are every guarantee of each lock whose statements PKG's build can hold (PKG's, each package PKG requires by path,
and each that names PKG under `proved_in`), and every other qualified name those locks hold (an assumption, a claim's
witness, a name in a reason), walked when it is a constant. A package PKG doesn't require imports nothing of PKG, so its
guarantees can't use a module of it.

A module of PKG is kept, with the first reason that holds:
- `used`: it holds a constant of the closure (`by` names the first guarantee whose closure reaches it);
- `root`, `runs`, `exempt`, `compile_time`: the lock lists it so (an exempt or `runs` module's imports are kept too);
- `syntax`: it declares a notation, macro, syntax, elaborator, simproc or attribute (comments and `local` ones aside) whose
  atom or name a module importing it uses, or one whose atoms can't be read off its source;
- `named`: a tracked file other than PKG's Lean sources names it, by its path (from the root, or from the source directory
  as `Proofs/A/B.lean`, quoted parts joined by `/` read as one; a path without `.lean` only where no directory has it: a
  directory's path names no module), its module name, or a constant's name one component under it
  (the move maps of `tools/move/`, `archive/` and the store's `moves.json` aside; a document's mention reads nothing, nor
  does a Lean file's other than its `import`s, and both are listed as `docs`). A module named so must still build, and
  what it needs for that is what its declarations use (`refs`, below), its imports being rewritten as `--delete` does; a
  reader that writes Lean against it, a difftest, a shell script or any file holding `import M` (a Lean file's imports, a
  test's Lean text), keeps what M imports too, transitively, since that Lean may use any of it;
- `queued`: a PR in the merge queue (`--queued`: what `--fetch-queued` wrote from `research queue status` and each PR's
  diff) edits, renames or deletes its file, or adds an `import` of it or (in another file: a lock, a test) its name; what
  it imports is kept too, unless the PRs only edit it and it declares nothing (an edit of an import-only module is its
  imports, which the diff shows);
- `checks`: it is a conformance check, which no guarantee reaches by design: it holds an `example`, or its name (or an
  enclosing module's) ends in `Checks` or `Vectors`. Its owner decides whether it goes;
- `refs`: a kept module's declarations use its constants. A module is deleted whole, so every declaration of a kept module
  must still build, used or not;
- `attribute`: it sets an attribute globally (`attribute [simp] f`), and a kept module imports it, whose elaboration may
  read it;
- `aggregator`: it declares nothing and imports a kept module.
Every other module is unused. The modules only `queued` or `checks` keep (with what those keep through `refs`,
`attribute` and `aggregator`) are listed apart (`held`), each with its reason. `--frozen PREFIX` lists the unused modules
under PREFIX apart, not for deletion (a freeze).
Each unused module carries its bytes, its area (`Proofs.Pouw.X`'s is `Pouw`), its owner (the owner most of the guarantees
declared under its nearest enclosing namespace of modules have, in the locks; else the lane most of the branches merged
into its directory's history name), and the files that mention it without reading it (`docs`: documents, and Lean
comments outside PKG), which its deletion edits.

`--out DIR` writes `used.json` (Used.lean's document), `unused.json` (the list, the kept modules' reasons and the totals)
and `unused.md` (the table, per area). `--delete AREA` removes the area's unused modules that `--list` holds (less those
`--except` names, those a PR of `--queued` touches or imports, and the frozen ones), and in every module left that imported
one, replaces that import with the kept
modules it reached through the deleted ones that the module's other imports don't already reach, so the rest builds as
before; it prints the documents and lock entries that still name what it deleted. Nothing here judges a proof: the
package's build and its records (`lean_changed.py --records PKG --update`) decide whether the deletion holds.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("lean_check", HERE / "check.py")
C = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("lean_check", C)
_spec.loader.exec_module(C)

#: an `import` line of a module's header, comments blanked: its keyword's prefix and the module
IMPORT = re.compile(r"(?m)^([ \t]*(?:(?:public|private)[ \t]+)?(?:meta[ \t]+)?import[ \t]+(?:all[ \t]+)?)([^\s]+)[ \t]*$")
#: a declaration that extends the syntax or the attributes other modules elaborate with, `local` ones aside
SYNTAX = re.compile(r"(?m)^[ \t]*(?:@\[[^\]\n]*\][ \t]*)*(?:(?:private|protected|noncomputable|scoped|partial|unsafe)[ \t]+)*"
                    r"(?:\([^)\n]*\)[ \t]*)?(notation\d*|infixl|infixr|infix|prefix|postfix|syntax|macro_rules|macro|elab_rules"
                    r"|elab|declare_syntax_cat|register_simp_attr|register_label_attr|simproc|dsimproc|simproc_decl"
                    r"|dsimproc_decl|initialize|builtin_initialize)\b([^\n]*)")
#: a global attribute command (`attribute [simp] f`), `local` and `scoped` ones aside
ATTRIBUTE = re.compile(r"(?m)^[ \t]*attribute[ \t]*\[(?![ \t]*(?:local|scoped)\b)")
#: a qualified name in a lock's text
QUALIFIED = re.compile(r"[^\W\d][\w'!?₀-₉]*(?:\.[^\W\d][\w'!?₀-₉]*)+")
STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
#: tracked files whose mentions of a module don't read it: the layout moves' maps, and what `archive/` froze
NOT_READERS = ("tools/move/", "archive/", "tools/research/src/research/store/moves.json")
#: a Lean `import` written in any file (a test's Lean text, a script's)
LEAN_IMPORT = re.compile(r"(?<![\w.])import[ \t]+(?:all[ \t]+)?([^\W\d][\w'.]*[\w'])")
#: a path joined from quoted parts (`ROOT / "Security" / "Proofs" / "A.lean"`), read as the one path it is
JOINED = re.compile(r"[\"'][ \t]*/[ \t]*[\"']")
#: a conformance check: a module named for one, or holding an `example` (which nothing can use)
CHECK_NAME = re.compile(r"(?:Checks|Vectors)(?:\.|$)")
EXAMPLE = re.compile(r"(?m)^[ \t]*example\b")
CODE = (".py", ".sh", ".toml", ".lean", ".json", ".yaml", ".yml", ".cfg", ".txt")
BRANCH_LANE = re.compile(r"cursor/([a-z][a-z0-9]*(?:-[a-z0-9]+)*?)-[a-z0-9]+-[0-9a-f]{4}\b|cursor/([a-z][a-z0-9]*)-")


def tracked() -> list[str]:
    return subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")[:-1]


def rel(pkg: Path) -> str:
    return os.path.relpath(pkg.resolve(), ROOT)


def lock_packages(pkg: str, files: list[str]) -> list[str]:
    """The packages whose locks' names PKG's build can hold: PKG, those it requires by path, and those naming it under
    `proved_in`."""
    la = _lean_audit()
    out = [pkg, *la.path_deps(pkg), *(p for p in la.packages(files) if pkg in la.provers(p))]
    return [p for p in dict.fromkeys(out) if (ROOT / p / C.POLICY).is_file()]


def _lean_audit():
    spec = importlib.util.spec_from_file_location("lean_audit_for_unused", HERE / "lean_audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def names_of(policy: dict, text: str) -> tuple[list[str], list[str]]:
    """(the lock's guarantees, every other qualified name in its keys and strings)."""
    guarantees = sorted(policy.get("guarantees", {}))
    extra = {q for s in STRING.findall(text) for q in QUALIFIED.findall(s)} - set(guarantees)
    return guarantees, sorted(extra)


def used_program(pkg: Path, work: Path) -> str:
    """`Used.lean` compiled with ``pkg``'s toolchain into ``work`` (as `check.py`'s `facts_program` compiles `Facts.lean`)."""
    src = HERE / "Used.lean"
    for argv in (["lake", "env", "lean", "-R", str(src.parent), "-c", str(work / "Used.c"), str(src)],
                 ["lake", "env", "leanc", "-O3", "-DNDEBUG", "-rdynamic", str(work / "Used.c"), "-o", str(work / "used")]):
        rc, text = C.run(argv, pkg, work / "compile.log")
        if rc:
            raise SystemExit(f"unused: `{' '.join(argv)}` failed (exit {rc}):\n{text[-3000:]}")
    return str(work / "used")


def run_used(pkg: Path, roots: list[str], pins: list[str], extra: list[str], out: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="lean-used-") as d:
        work = Path(d)
        config = work / "config.json"
        config.write_text(json.dumps({"roots": roots, "pins": pins, "extra": extra, "out": str(out)}))
        rc, text = C.run(["lake", "env", used_program(pkg, work), str(config)], pkg, work / "used.log")
        sys.stderr.write(text[-2000:])
        if rc or not out.is_file():
            raise SystemExit(f"unused: `Used.lean` failed (exit {rc}; a root that was not built?):\n{text[-3000:]}")
    return json.loads(out.read_text())


def source_imports(text: str) -> list[str]:
    return [m.group(2) for m in IMPORT.finditer(C.strip_comments(text))]


def closure(start: set[str], graph: dict[str, list[str]], within: set[str]) -> set[str]:
    """``start`` and every module of ``within`` it imports, transitively."""
    seen, stack = set(), list(start)
    while stack:
        if (m := stack.pop()) in seen:
            continue
        seen.add(m)
        stack += [i for i in graph.get(m, []) if i in within]
    return seen


def atoms(decl: str) -> set[str]:
    """What a module using a syntax declaration's rest (``decl``, after its keyword) writes: its string atoms, and for an
    attribute or simproc its name."""
    got = {a.strip() for a in STRING.findall(decl) if a.strip()}
    if not got and (m := re.match(r"[ \t]*([^\W\d][\w'.]*)", decl)):
        got.add(m.group(1).rsplit(".", 1)[-1])
    return got


def references(files: list[str], pkg: str, mods: dict[str, str]) -> dict[str, list[tuple[str, bool]]]:
    """{module of PKG: [(file, whether it reads what the module imports)]}, for each tracked file outside PKG's tree (and
    `NOT_READERS`) naming the module's file (from the root, or from the source directory, `Proofs/A/B.lean`; a path joined
    from quoted parts, `"Proofs" / "A"`, is read as one; a path without `.lean` names it only where no directory has that
    path), or its module name or a constant one component under it; a Lean file outside PKG names only what it imports.
    The mentions that read nothing (a document's, and a Lean comment's) are under the key `(docs, module)`: a deletion
    edits them."""
    by_path = {f"{pkg}/{r}": m for m, r in mods.items()}
    by_src = {m.replace(".", "/") + ".lean": m for m in mods}
    src = next(((ROOT / pkg / r).parents[m.count(".")] for m, r in mods.items()), ROOT / pkg)
    out: dict[str, list[tuple[str, bool]]] = collections.defaultdict(list)
    tops = sorted({m.split(".", 1)[0] for m in mods})
    either = "|".join(map(re.escape, tops))
    named = re.compile(r"(?<![\w.])(?:" + either + r")(?:\.[^\W\d][\w']*)+")
    paths = re.compile(re.escape(pkg) + r"/[\w./-]*\w")
    local = re.compile(r"(?<![\w./-])(?:" + either + r")/[\w./-]*\w")

    def mentioned(text: str) -> set[str]:
        hit: set[str] = set()
        for found, table, base in ((paths, by_path, ROOT), (local, by_src, src)):
            for p in found.finditer(text):
                x = p.group(0).rstrip(".")
                if x in table:
                    hit.add(table[x])
                elif x + ".lean" in table and text[p.end():p.end() + 1] != "/" and not (base / x).is_dir():
                    hit.add(table[x + ".lean"])
        for x in named.findall(text):
            if x in mods or (x.rsplit(".", 1)[1][:1].islower() and (x := x.rsplit(".", 1)[0]) in mods):
                hit.add(x)
        return hit

    for f in files:
        if (f.startswith(pkg + "/") and f.endswith(".lean")) or f.startswith(NOT_READERS) or not f.endswith(CODE + (".md",)):
            continue
        if f.endswith((C.POLICY, "lake-manifest.json")):
            continue
        try:
            text = JOINED.sub("/", (ROOT / f).read_text(errors="replace"))
        except OSError:
            continue
        if pkg not in text and not any(t + "." in text or t + "/" in text for t in tops):
            continue
        if f.endswith(".lean"):
            imported = {i for i in source_imports(text) if i in mods}
            for m in imported:
                out[m].append((f, True))
            for m in mentioned(text) - imported:
                out[("docs", m)].append((f, False))
            continue
        writes = {i for i in LEAN_IMPORT.findall(text) if i in mods}
        reads = "difftest" in f or f.endswith(".sh")
        for m in mentioned(text) | writes:
            out[("docs", m) if f.endswith(".md") else m].append((f, reads or m in writes))
    return out


#: a queued PR's added `import` line
ADDED_IMPORT = re.compile(r"^\+[ \t]*(?:(?:public|private)[ \t]+)?(?:meta[ \t]+)?import[ \t]+(?:all[ \t]+)?([^\s]+)[ \t]*$")


def fetch_queued(path: Path) -> dict:
    """Every PR `research queue status` lists, in any state, with the files its diff touches (old and new paths), the modules
    its added `import` lines name, and the qualified names its other files' added lines hold (a lock's entry, a test's
    module), written to ``path``: {PR: {"files": [...], "imports": [...], "names": [...]}}."""
    from concurrent.futures import ThreadPoolExecutor

    status = subprocess.run(["research", "queue", "status"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    prs = sorted({int(n) for n in re.findall(r"(?m)^\s*\S+\s+#(\d+)\s", status)})

    def one(n: int) -> tuple[str, dict]:
        r = subprocess.run(["gh", "api", "--paginate", f"repos/{{owner}}/{{repo}}/pulls/{n}/files?per_page=100", "--jq",
                            '.[] | [.filename, (.previous_filename // ""), (.patch // "")] | @json'],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"unused: the files of PR #{n} can't be read: {r.stderr.strip()[-300:]}")
        files, imports, names = set(), set(), set()
        for fn, prev, patch in (json.loads(line) for line in r.stdout.splitlines() if line.strip()):
            files |= {fn, prev} - {""}
            if fn.endswith(".lean"):
                imports |= {m.group(1) for line in patch.splitlines() if (m := ADDED_IMPORT.match(line))}
            else:
                names |= {q for line in patch.splitlines() if line.startswith("+") for q in QUALIFIED.findall(line)}
        return str(n), {"files": sorted(files), "imports": sorted(imports), "names": sorted(names)}

    with ThreadPoolExecutor(8) as ex:
        doc = dict(ex.map(one, prs))
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def queued_modules(queued: dict, pkg: str, mods: dict[str, str]) -> dict[str, tuple[list[str], bool]]:
    """{module of PKG: (the PRs of ``queued`` (`fetch_queued`'s document) whose diff touches its file, or adds an import or
    another mention of it; whether one of them imports or names it, rather than only touching its file)}."""
    by_path = {f"{pkg}/{r}": m for m, r in mods.items()}
    out: dict[str, set[str]] = collections.defaultdict(set)
    named: set[str] = set()
    for n, pr in queued.items():
        those = [i for i in [*pr.get("imports", []), *pr.get("names", [])] if i in mods]
        named.update(those)
        for m in [by_path[f] for f in pr.get("files", []) if f in by_path] + those:
            out[m].add(f"#{n}")
    return {m: (sorted(v, key=lambda s: int(s[1:])), m in named) for m, v in out.items()}


def owners(policies: dict[str, dict], pins: dict[str, str | None], files: list[str], pkg: str) -> dict[str, str]:
    """{module prefix: the owner most of the guarantees declared under it have}, for every prefix of a declaring module."""
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for policy in policies.values():
        for g, e in policy.get("guarantees", {}).items():
            m, who = pins.get(g), (e or {}).get("owner")
            if m and who:
                parts = m.split(".")
                for i in range(1, len(parts) + 1):
                    votes[".".join(parts[:i])][who] += 1
    return {p: c.most_common(1)[0][0] for p, c in votes.items()}


def lane_from_history(path: str) -> str | None:
    """The lane most of the branches merged into ``path``'s history name (`cursor/<lane>-…`), as `@lane`."""
    log = subprocess.run(["git", "log", "--format=%s", "-n", "400", "--", path], cwd=ROOT, capture_output=True, text=True).stdout
    c = collections.Counter(next(g for g in m.groups() if g) for m in BRANCH_LANE.finditer(log))
    return f"@{c.most_common(1)[0][0]}" if c else None


def area_of(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def decide(pkg: Path, policy: dict, used: dict, files: list[str], frozen: list[str], queued: dict | None = None,
           checks: bool = True) -> dict:
    """The kept modules of ``pkg`` with their reasons, and the unused ones (the module docstring's rules; ``checks``
    False drops the `checks` rule, as `held` is found)."""
    p = rel(pkg)
    mods = C.lean_modules(pkg)
    scope = set(mods)
    texts = {m: (pkg / r).read_text(errors="replace") for m, r in mods.items()}
    stripped = {m: C.strip_comments(t) for m, t in texts.items()}
    graph = {**used.get("imports", {}), **{m: [i for i in source_imports(t)] for m, t in texts.items()}}
    facts = used.get("modules", {})
    importers: dict[str, set[str]] = collections.defaultdict(set)
    for m in scope:
        for i in graph.get(m, []):
            importers[i].add(m)
    above: dict[str, set[str]] = {}

    def importing(m: str) -> set[str]:
        if m not in above:
            seen, stack = set(), list(importers.get(m, ()))
            while stack:
                if (x := stack.pop()) not in seen:
                    seen.add(x)
                    stack += list(importers.get(x, ()))
            above[m] = seen
        return above[m]

    kept: dict[str, str] = {}

    def keep(m: str, why: str) -> bool:
        if m in scope and m not in kept:
            kept[m] = why
            return True
        return False

    for m in sorted(scope):
        f = facts.get(m)
        if f and f.get("used"):
            keep(m, f"used: {f['used']} of its {f['constants']} constants are in the closure (first reached by {f['by']})")
    for m in policy.get("roots", []):
        keep(m, "root: a root of the lock")
    for m in policy.get("compile_time", {}):
        keep(m, "compile_time: listed under compile_time")
    for key, what in (("runs", "a `runs` test"), ("exempt", "exempt")):
        for m in policy.get(key, {}):
            reason = policy[key][m] if key == "exempt" else policy[key][m].get("why", "")
            keep(m, f"{key}: {what} ({str(reason)[:120]})")
            for i in sorted(closure({m}, graph, scope) - {m}):
                keep(i, f"{key}: imported by {m}, {what}")
    for m in sorted(scope):
        decls = [(k, rest) for k, rest in SYNTAX.findall(stripped[m])]
        if not decls:
            continue
        words = set().union(*(atoms(rest) for _, rest in decls))
        if not words:
            keep(m, f"syntax: declares {', '.join(sorted({k for k, _ in decls}))} whose atoms its source doesn't show")
            continue
        users = sorted(u for u in importing(m) if any(w in stripped[u] for w in words))
        if users:
            keep(m, f"syntax: declares {', '.join(sorted({k for k, _ in decls}))} ({', '.join(sorted(words)[:4])}), used by "
                    f"{users[0]}" + (f" and {len(users) - 1} more" if len(users) > 1 else ""))
    refs = references(files, p, mods)
    for m in sorted(k for k in refs if isinstance(k, str)):
        by = sorted(refs[m])
        keep(m, f"named: {by[0][0]}" + (f" and {len(by) - 1} more" if len(by) > 1 else ""))
        for f, reads in by:
            if reads:
                for i in sorted(closure({m}, graph, scope) - {m}):
                    keep(i, f"named: imported by {m}, which {f} reads")
    for m, (by, imported) in sorted(queued_modules(queued or {}, p, mods).items()):
        keep(m, f"queued: {', '.join(by[:4])}" + (f" and {len(by) - 4} more" if len(by) > 4 else "") + " touch or import it")
        if not imported and (facts.get(m) or {}).get("constants", 1) == 0:
            continue
        for i in sorted(closure({m}, graph, scope) - {m}):
            keep(i, f"queued: imported by {m}, which {by[0]} touches or imports")
    if checks:
        for m in sorted(scope):
            if CHECK_NAME.search(m) or EXAMPLE.search(stripped[m]):
                keep(m, "checks: a conformance check (it holds an `example`, or its name says Checks or Vectors)")
    changed = True
    while changed:
        changed = False
        for m in sorted(kept):
            for r in (facts.get(m) or {}).get("refs", []):
                if keep(r, f"refs: the declarations of {m} (kept) use its constants"):
                    changed = True
        for m in sorted(scope - set(kept)):
            if ATTRIBUTE.search(stripped[m]) and (users := sorted(importing(m) & set(kept))):
                keep(m, f"attribute: sets an attribute globally, and {users[0]} (kept) imports it")
                changed = True
        for m in sorted(scope - set(kept)):
            f = facts.get(m)
            if (f is None and not re.search(r"\S", re.sub(IMPORT, "", stripped[m]))) or (f is not None and f["constants"] == 0):
                if any(i in kept for i in graph.get(m, [])):
                    keep(m, "aggregator: declares nothing, and imports a kept module")
                    changed = True
    unbuilt = sorted(m for m in scope - set(kept) if m not in facts)
    for m in unbuilt:
        keep(m, "unbuilt: no .olean was read for it, so what it holds is not known")
    unused = sorted(scope - set(kept))
    docs = {m: sorted({f for f, _ in refs[("docs", m)]}) for m in scope if ("docs", m) in refs}
    return {"mods": mods, "kept": kept, "unused": unused, "docs": docs, "graph": graph, "frozen_under": list(frozen),
            "frozen": [m for m in unused if any(m == f or m.startswith(f + ".") for f in frozen)]}


def listing(pkg: Path, decided: dict, policies: dict[str, dict], used: dict, files: list[str],
            held: dict[str, str] | None = None) -> dict:
    """The list: ``decided``'s unused modules, each with its bytes, area, owner and docs, totals per area and in all; and
    ``held`` ({module: why}), the modules only a queued PR or the `checks` rule keeps, with the same columns."""
    p = rel(pkg)
    lead = owners(policies, used.get("pins", {}), files, p)
    history: dict[str, str | None] = {}

    def row(m: str) -> dict:
        parts = m.split(".")
        who = next((lead[".".join(parts[:i])] for i in range(len(parts) - 1, 1, -1) if ".".join(parts[:i]) in lead), None)
        how = "lock"
        if who is None:
            d = f"{p}/{os.path.dirname(decided['mods'][m]) or '.'}"
            if d not in history:
                history[d] = lane_from_history(d)
            who, how = history[d], "git log"
        f = pkg / decided["mods"][m]
        return {"module": m, "path": f"{p}/{decided['mods'][m]}", "bytes": f.stat().st_size, "area": area_of(m),
                "owner": who, "owner_from": how, "docs": decided["docs"].get(m, []),
                "frozen": any(m == f or m.startswith(f + ".") for f in decided.get("frozen_under", []))}

    rows = [row(m) for m in decided["unused"]]
    kept_rows = [{**row(m), "why": why} for m, why in sorted((held or {}).items())]
    areas: dict[str, dict] = {}
    for r in rows:
        a = areas.setdefault(r["area"], {"modules": 0, "bytes": 0, "frozen": 0, "frozen_bytes": 0, "owners": collections.Counter()})
        k = "frozen" if r["frozen"] else "modules"
        a[k] += 1
        a["frozen_bytes" if r["frozen"] else "bytes"] += r["bytes"]
        a["owners"][r["owner"] or "-"] += 1
    for a in areas.values():
        a["owners"] = dict(a["owners"].most_common())
    reasons = collections.Counter(why.split(":", 1)[0] for why in decided["kept"].values())
    total = {"modules": len(decided["mods"]), "kept": len(decided["kept"]), "unused": len(rows),
             "unused_bytes": sum(r["bytes"] for r in rows), "frozen": sum(r["frozen"] for r in rows),
             "held": len(kept_rows), "held_bytes": sum(r["bytes"] for r in kept_rows),
             "package_bytes": sum((pkg / r).stat().st_size for r in decided["mods"].values()), "kept_by": dict(reasons)}
    missing = sorted(g for g, m in used.get("pins", {}).items() if m is None)
    return {"package": p, "total": total, "areas": areas, "unused": rows, "held": kept_rows, "kept": decided["kept"],
            "missing_guarantees": missing}


def table(doc: dict) -> str:
    t = doc["total"]
    out = [f"# Unused modules of {doc['package']}", "",
           f"{t['unused']} of {t['modules']} modules, {t['unused_bytes']:,} of {t['package_bytes']:,} bytes "
           f"({t['frozen']} of them frozen); kept: " + ", ".join(f"{k} {v}" for k, v in sorted(t["kept_by"].items())), "",
           "| area | modules | bytes | frozen | frozen bytes | owners |", "|---|---|---|---|---|---|"]
    for a, r in sorted(doc["areas"].items(), key=lambda kv: -kv[1]["bytes"]):
        out.append(f"| {a} | {r['modules']} | {r['bytes']:,} | {r['frozen']} | {r['frozen_bytes']:,} | "
                   + ", ".join(f"{o} {n}" for o, n in r["owners"].items()) + " |")
    out += ["", "| module | bytes | area | owner | frozen | docs naming it |", "|---|---|---|---|---|---|"]
    for r in sorted(doc["unused"], key=lambda r: (r["area"], r["module"])):
        out.append(f"| {r['module']} | {r['bytes']:,} | {r['area']} | {r['owner'] or '-'} ({r['owner_from']}) | "
                   f"{'frozen' if r['frozen'] else ''} | {', '.join(r['docs'])} |")
    if doc.get("held"):
        out += ["", f"## Held: unused but for a queued PR or as a check ({t['held']} modules, {t['held_bytes']:,} bytes)",
                "", "| module | bytes | area | owner | frozen | why |", "|---|---|---|---|---|---|"]
        for r in sorted(doc["held"], key=lambda r: (r["area"], r["module"])):
            out.append(f"| {r['module']} | {r['bytes']:,} | {r['area']} | {r['owner'] or '-'} ({r['owner_from']}) | "
                       f"{'frozen' if r['frozen'] else ''} | {r['why']} |")
    return "\n".join(out) + "\n"


def rewrite_imports(text: str, deleted: set[str], replace: dict[str, list[str]], reach) -> str:
    """``text`` with each import of a deleted module replaced by ``replace[module]`` less what its other imports reach
    (``reach``: a module's import closure) and what it imports already, each written as the line it replaces was."""
    lines = text.split("\n")
    stripped = C.strip_comments(text).split("\n")
    kept_imports = [m.group(2) for s in stripped if (m := IMPORT.match(s)) and m.group(2) not in deleted]
    have = set(kept_imports).union(*(reach(i) for i in kept_imports)) if kept_imports else set()
    out = []
    for line, s in zip(lines, stripped):
        m = IMPORT.match(s)
        if not m or m.group(2) not in deleted:
            out.append(line)
            continue
        for r in replace[m.group(2)]:
            if r not in have:
                have |= {r} | reach(r)
                out.append(f"{m.group(1)}{r}")
    return "\n".join(out)


def delete(pkg: Path, area: str, doc: dict, used: dict, spare: set[str]) -> list[str]:
    """Delete ``area``'s unused modules of ``doc`` (less ``spare`` and the frozen ones) and rewrite the imports of the
    modules left; the modules deleted."""
    mods = C.lean_modules(pkg)
    gone = {r["module"] for r in doc["unused"] if r["area"] == area and not r["frozen"] and r["module"] not in spare}
    gone &= set(mods)
    graph = {**used.get("imports", {}), **{m: source_imports((pkg / r).read_text(errors="replace")) for m, r in mods.items()}}
    memo: dict[str, list[str]] = {}

    def expand(m: str) -> list[str]:
        if m not in memo:
            memo[m] = []
            memo[m] = list(dict.fromkeys(x for i in graph.get(m, []) for x in (expand(i) if i in gone else [i])))
        return memo[m]

    reach_memo: dict[str, set[str]] = {}

    def reach(m: str) -> set[str]:
        if m not in reach_memo:
            reach_memo[m] = closure({m}, graph, set(graph)) - {m}
        return reach_memo[m]

    replace = {m: expand(m) for m in gone}
    for m, r in mods.items():
        if m in gone:
            continue
        f = pkg / r
        text = f.read_text()
        if not any(i in gone for i in source_imports(text)):
            continue
        f.write_text(rewrite_imports(text, gone, replace, reach))
    for m in sorted(gone):
        (pkg / mods[m]).unlink()
    for m in sorted(gone):
        d = (pkg / mods[m]).parent
        while d != pkg and d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            d = d.parent
    return sorted(gone)


def stale_mentions(pkg: Path, gone: list[str], files: list[str], mods: dict[str, str]) -> list[str]:
    """The tracked files (documents and locks) still naming a deleted module's path (``mods``: where each was) or name."""
    p = rel(pkg)
    paths = {f"{p}/{mods[m]}" for m in gone}
    names = set(gone)
    out = []
    for f in files:
        if not f.endswith((".md", C.POLICY, "lakefile.toml")) or not (ROOT / f).is_file():
            continue
        text = (ROOT / f).read_text(errors="replace")
        hits = sorted({x for x in paths if x in text} | {n for n in names if f'"{n}"' in text})
        if hits:
            out.append(f"{f}: {', '.join(hits[:6])}" + (f" and {len(hits) - 6} more" if len(hits) > 6 else ""))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pkg", nargs="?", default="Security/Proofs", help="the built Lake package (default Security/Proofs)")
    ap.add_argument("--used", type=Path, help="a document Used.lean wrote for PKG, instead of running it")
    ap.add_argument("--out", type=Path, help="write used.json, unused.json and unused.md here")
    ap.add_argument("--frozen", nargs="*", default=[], metavar="PREFIX", help="list the unused modules under PREFIX apart")
    ap.add_argument("--delete", metavar="AREA", help="delete AREA's unused modules of --list")
    ap.add_argument("--list", type=Path, help="with --delete: an unused.json this tool wrote")
    ap.add_argument("--except", dest="spare", nargs="*", default=[], metavar="MODULE", help="with --delete: keep these")
    ap.add_argument("--queued", type=Path, help="keep what the queued PRs of this document (--fetch-queued's) touch or import")
    ap.add_argument("--fetch-queued", type=Path, metavar="QUEUED.json", help="write the merge queue's PRs' files and imports")
    a = ap.parse_args(argv)
    if a.fetch_queued:
        doc = fetch_queued(a.fetch_queued)
        print(f"unused: {len(doc)} PRs of the queue, {sum(len(v['files']) for v in doc.values())} files -> {a.fetch_queued}")
        return 0
    pkg = (ROOT / a.pkg) if not Path(a.pkg).is_absolute() else Path(a.pkg)
    files = tracked()
    queued = json.loads(a.queued.read_text()) if a.queued else None
    if a.delete:
        if not a.list or not a.used:
            ap.error("--delete needs --list (an unused.json) and --used (the used.json beside it)")
        doc, used = json.loads(a.list.read_text()), json.loads(a.used.read_text())
        mods = C.lean_modules(pkg)
        spare = set(a.spare) | set(queued_modules(queued or {}, rel(pkg), mods))
        gone = delete(pkg, a.delete, doc, used, spare)
        print(f"unused: deleted {len(gone)} modules of {a.delete}, "
              f"{sum(r['bytes'] for r in doc['unused'] if r['module'] in set(gone)):,} bytes")
        for line in stale_mentions(pkg, gone, tracked(), mods):
            print(f"  still named by {line}")
        return 0
    p = rel(pkg)
    policies = {q: C.load(ROOT / q) for q in lock_packages(p, files)}
    pins, extra = [], []
    for q in policies:
        g, x = names_of(policies[q], (ROOT / q / C.POLICY).read_text())
        pins += g
        extra += x
    policy = policies[p]
    if a.used:
        used = json.loads(a.used.read_text())
    else:
        out = (a.out or Path(tempfile.mkdtemp(prefix="lean-unused-")))
        out.mkdir(parents=True, exist_ok=True)
        roots = list(policy["roots"])
        used = run_used(pkg, roots, sorted(set(pins)), sorted(set(extra) - set(pins)), out / "used.json")
    decided = decide(pkg, policy, used, files, a.frozen, queued)
    bare = decide(pkg, policy, used, files, a.frozen, checks=False)["unused"]
    held = {m: decided["kept"][m] for m in bare if m in decided["kept"]}
    doc = listing(pkg, decided, policies, used, files, held)
    doc["locks"] = sorted(policies)
    doc["queued_prs"] = len(queued or {})
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        if a.used and a.used.resolve() != (a.out / "used.json").resolve():
            shutil.copy2(a.used, a.out / "used.json")
        (a.out / "unused.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        (a.out / "unused.md").write_text(table(doc))
    t = doc["total"]
    print(f"unused: {t['unused']} of {t['modules']} modules of {p} ({t['unused_bytes']:,} of {t['package_bytes']:,} bytes, "
          f"{t['frozen']} frozen; {t['held']} more, {t['held_bytes']:,} bytes, held by queued PRs or as checks); "
          f"locks read: {', '.join(doc['locks'])}")
    for g in doc["missing_guarantees"][:10]:
        print(f"  a guarantee no module of the build declares: {g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
