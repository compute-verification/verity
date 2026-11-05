"""`unread`: the Lean nothing reads, per module and per area, and what deleting it would do to the longest import chain.

    python tools/verity/lean/unread.py --out DIR [--root DIR] [--env PKG] [--setup] [--build] [--facts FILE] [--rev REV]

It reads every module of every Lake package under ``--root`` (default this checkout; `modules.packages`, the auditor's
fixtures aside) from the `.olean` files of one package's build (``--env``, default `Security/Proofs`, whose `lake env`
reaches the security analysis's three trees and the code packages they require): `Unread.lean`, compiled with that
package's toolchain, writes each constant with what its type and value mention, and this holds the policy. ``--setup``
first runs `setup.sh ENV`, and ``--build`` `lake build` there, with `@NAME` for each other analysis package
(`ANALYSIS`). ``--facts FILE`` reads facts written before instead.

**The closure.** Starting from the seeds, it follows what each reached constant's type and value mention
(`ConstantInfo.getUsedConstants`, an inductive type's constructors, `@[implemented_by]` and `@[csimp]`), and from any
constant to the declaration it belongs to. The seeds:

- every key and string value of each package's `lean-audit.json` (`reads` and `dependencies` aside) that names a
  constant: the guarantees, assumptions, witnesses, claims, `shape` and `verifier` entries, escapes;
- every declaration of the modules a lock names under `roots`, `runs`, `exempt` and `compile_time`;
- every constant with a `SecurityProofs` component, every declaration of `Proofs.Properties.*` and `Properties.*`;
- every constant of a code package (every package outside `ANALYSIS`: executable code), and every `main`;
- every constant Lean uses by name rather than through a term: syntax, macros, elaborators, tactics, delaborators,
  simprocs, attributes, options and `IO` (`meta`), unsafe and partial ones (compiled code), and `@[defeq]` theorems,
  which `dsimp` and `simp only [t]` use with no trace in the proof term;
- every declaration of the analysis trees that a file outside Lean names in full (`.py`, `.json`, `.toml`, `.rs`, `.sh`,
  ...; `CITED_BY`), so a test or tool that reads one by name keeps it.

The `@[defeq]` theorems, the lock-listed modules and the cited names are seeds to be safe (`CAUTIOUS`): the report counts
what each alone keeps (`kept_only_by`).

**Declarations.** A constant with a declaration range is a declaration, unless it belongs to a type (a constructor, a
recursor, a projection) or Lean generated it from another's command (the same first line as a declaration its name
extends: `@[ext]`'s lemmas); every other constant (`match_`, `proof_`, `eq_`, `_unfold`, `.eq_def`, `casesOn`,
`noConfusion`, `_private`'s aux) belongs to the nearest declaration its name extends. A declaration is read when the
closure reaches it or any constant of it other than a matcher or an abstracted proof, which Lean shares between the
declarations of a module and makes again for a reader whose owner is gone. A module is wholly unread when it declares
something and nothing of it is read or seeded, and it gives no instance or `simp` attribute to another module's
constant; a module that declares nothing (an aggregator) goes with the modules it imports when all of them go.

**The chain.** The longest import chain among the tree's modules, in modules and in lines (each module's file), before
and after the wholly unread modules go, each importer then importing what the deleted module imported.

It writes ``DIR/unread.json`` (every module's counts and unread declarations, by area and by owner, the chains, the
seeds) and ``DIR/unread.md`` (the summary). The owners of an area are the `owner` handles the locks give the guarantees
and assumptions its modules declare: the tree has no other map from Lean to lanes. Stdlib only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_unread", HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MODULES = _load("modules")
HEADERS = _load("headers")

#: the security analysis's packages: Definitions and Models, the proofs, the properties. Every other package is code.
ANALYSIS = ("Security", "Security/Proofs", "Security/Properties")
#: the lock sections that name modules, every declaration of which is a seed
MODULE_SECTIONS = ("roots", "runs", "exempt", "compile_time")
#: the lock sections that name no seed: what the guarantees read (reached from them anyway) and the pinned builds
SKIPPED_SECTIONS = ("reads", "dependencies")
#: module prefixes whose every declaration is a seed: the properties of the code and their proofs
SEEDED_PREFIXES = ("Properties", "Proofs.Properties")
#: the flags `Unread.lean` writes that make a constant a seed
SEED_FLAGS = ("meta", "unsafe", "partial", "defeq")
#: the seeds taken to be safe rather than because something reads them: each is reported with what it alone keeps
CAUTIOUS = ("defeq", "lock module", "cited")
CAUTIOUS_TEXT = {"defeq": "`@[defeq]` theorems", "lock module": "the modules locks list",
                 "cited": "names in files outside Lean"}
#: the files outside Lean whose full mention of a declaration keeps it (a test or tool reading it by name)
CITED_BY = (".py", ".json", ".toml", ".rs", ".sh", ".yaml", ".yml", ".txt", ".ts", ".js", ".cfg")
#: the files whose mention of a declaration is reported but keeps nothing
DOCS = (".md",)
#: a cited file larger than this is data, not code
CITE_MAX_BYTES = 4 << 20
#: a dotted identifier, as a file outside Lean would name a declaration
DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_'!?]*(?:\.[A-Za-z0-9_'!?]+)+")
#: the constants Lean shares between a module's declarations: a matcher, an abstracted proof
SHARED = re.compile(r"\.(?:match_\d+|_?proof_\d+)$")
SKIP_DIRS = {".lake", ".git", ".venv", "node_modules", "__pycache__", "target", ".mypy_cache", ".pytest_cache"}
#: the module the report gives the chain into, beside the longest overall
CHAIN_TO = "Proofs.Flock"


def tracked(root: Path) -> list[str]:
    """Every file under ``root``, relative to it: git's tracked files in a checkout, else a walk past build trees."""
    top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode == 0 and Path(top.stdout.strip()).resolve() == root.resolve():
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
        return sorted(p.decode() for p in out.split(b"\0") if p)
    files = []
    for d, dirs, names in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        rel = Path(d).relative_to(root)
        files += [(rel / n).as_posix() for n in names]
    return sorted(files)


class Tree:
    """The packages under ``root`` and each one's modules, with their files, kinds and locks."""

    def __init__(self, root: Path, files: list[str]):
        self.root, self.files = root, files
        self.packages = MODULES.packages(files)
        self.module_file: dict[str, str] = {}
        self.module_package: dict[str, str] = {}
        self.clashes: list[str] = []
        for pkg in self.packages:
            cfg = MODULES._lakefile(root / pkg)
            src = (root / pkg / cfg.get("srcDir", ".")).resolve()
            for f in MODULES.package_modules(pkg, files, root):
                mod = (root / f).resolve().relative_to(src).with_suffix("").as_posix().replace("/", ".")
                if mod in self.module_file:
                    self.clashes.append(mod)
                    continue
                self.module_file[mod] = f
                self.module_package[mod] = pkg
        self.locks = {pkg: json.loads((root / pkg / "lean-audit.json").read_text())
                      for pkg in self.packages if (root / pkg / "lean-audit.json").is_file()}
        self.names = {pkg: MODULES._lakefile(root / pkg).get("name") for pkg in self.packages}

    def analysis(self, mod: str) -> bool:
        return self.module_package.get(mod) in ANALYSIS

    def area(self, mod: str) -> str:
        """`Proofs/Flock`, `Definitions/Pouw`, `Models` for the analysis's modules (the path under `Security/`, two parts at
        most); a code package's path for its own."""
        pkg = self.module_package.get(mod, "?")
        if pkg not in ANALYSIS:
            return pkg
        rel = Path(self.module_file[mod]).relative_to("Security").with_suffix("").parts
        return "/".join(rel[:2])

    def lines(self, mod: str) -> int:
        try:
            return len((self.root / self.module_file[mod]).read_text(errors="replace").splitlines())
        except OSError:
            return 0

    def source_imports(self, mod: str) -> list[str]:
        try:
            return HEADERS.imports((self.root / self.module_file[mod]).read_text(errors="replace")) or []
        except OSError:
            return []


def log(msg: str) -> None:
    print(f"unread: {msg}", file=sys.stderr, flush=True)


def build(tree: Tree, env: str, setup: bool = True) -> None:
    """`setup.sh ENV` (``setup``), then `lake build` there of its default targets and every other analysis package's
    (`@NAME`)."""
    if setup:
        subprocess.run(["bash", str(HERE / "setup.sh"), str(tree.root / env)], check=True)
    targets = [f"@{tree.names[p]}" for p in ANALYSIS if p in tree.names and p != env]
    for t in [[], *([t] for t in targets)]:
        log(f"lake build {' '.join(t)} in {env}")
        subprocess.run(["lake", "build", *t], cwd=tree.root / env, check=True)


def program(env_dir: Path, work: Path) -> Path:
    """`Unread.lean` compiled with ENV's toolchain (`lean -c`, then `leanc`), as `check.py` compiles `Facts.lean`:
    interpreted, its walk over every constant would be most of the run."""
    c, exe = work / "Unread.c", work / "unread"
    for argv in (["lake", "env", "lean", "-R", str(HERE), "-c", str(c), str(HERE / "Unread.lean")],
                 ["lake", "env", "leanc", "-O3", "-DNDEBUG", "-rdynamic", str(c), "-o", str(exe)]):
        subprocess.run(argv, cwd=env_dir, check=True)
    return exe


def write_facts(tree: Tree, env: str, out: Path) -> Path:
    """`Unread.lean`'s facts of every module of the tree, read from ENV's `lake env`."""
    cfg = out / "unread-config.json"
    facts = out / "unread-facts.jsonl"
    cfg.write_text(json.dumps({"modules": sorted(tree.module_file), "out": str(facts.resolve())}))
    exe = program(tree.root / env, out)
    subprocess.run(["lake", "env", str(exe.resolve()), str(cfg.resolve())], cwd=tree.root / env, check=True)
    return facts


class Facts:
    """`Unread.lean`'s lines: the modules, then the constants, each by its index."""

    def __init__(self, path: Path):
        self.modules: list[dict] = []
        self.name: list[str] = []
        self.module: array = array("I")
        self.kind: list[str] = []
        self.flags: list[frozenset] = []
        self.range: dict[int, tuple[int, int]] = {}
        self.parent: dict[int, str] = {}
        self.used: list[array] = []
        flagsets: dict[tuple, frozenset] = {}
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if "module" in d:
                    self.modules.append(d)
                    continue
                self.name.append(d["n"])
                self.module.append(d["m"])
                self.kind.append(sys.intern(d["k"]))
                fl = tuple(d["f"])
                self.flags.append(flagsets.setdefault(fl, frozenset(fl)))
                i = len(self.name) - 1
                if "r" in d:
                    self.range[i] = (d["r"][0], d["r"][1])
                if "p" in d:
                    self.parent[i] = d["p"]
                self.used.append(array("I", d["u"]))
        self.index = {n: i for i, n in enumerate(self.name)}
        self.module_index = {m["module"]: i for i, m in enumerate(self.modules)}


def owners(facts: Facts) -> list[int]:
    """The declaration each constant belongs to (`decl`): itself, the type it belongs to, the declaration whose command
    generated it, or the nearest one its name extends."""
    n = len(facts.name)
    owner = list(range(n))
    for i in range(n):
        p = facts.parent.get(i)
        if p is not None and p in facts.index:
            owner[i] = facts.index[p]
            continue
        name = facts.name[i]
        prefix = name
        while "." in prefix:
            prefix = prefix.rsplit(".", 1)[0]
            j = facts.index.get(prefix)
            if j is None:
                continue
            if i not in facts.range:
                if j in facts.range or facts.kind[j] == "inductive":
                    owner[i] = j
                    break
            elif facts.range.get(j, (None,))[0] == facts.range[i][0] and facts.module[j] == facts.module[i]:
                owner[i] = j
                break
    for i in range(n):
        j = owner[i]
        seen = 0
        while owner[j] != j and seen < 64:
            j, seen = owner[j], seen + 1
        owner[i] = j
    return owner


def walk_strings(x, out: set[str]) -> None:
    if isinstance(x, str):
        out.add(x)
    elif isinstance(x, list):
        for v in x:
            walk_strings(v, out)
    elif isinstance(x, dict):
        for k, v in x.items():
            out.add(k)
            walk_strings(v, out)


def lock_seeds(tree: Tree) -> tuple[set[str], set[str]]:
    """The names and the modules the locks give."""
    names: set[str] = set()
    mods: set[str] = set()
    for lock in tree.locks.values():
        for k, v in lock.items():
            if k in MODULE_SECTIONS:
                mods.update(v if isinstance(v, list) else v.keys())
            if k not in SKIPPED_SECTIONS:
                out: set[str] = set()
                walk_strings(v, out)
                names |= out
    return names, mods


def citations(tree: Tree, candidates: set[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The candidate names files outside Lean mention in full: by code (`CITED_BY`), and by docs (`DOCS`), each with the
    files."""
    code: dict[str, list[str]] = defaultdict(list)
    docs: dict[str, list[str]] = defaultdict(list)
    for f in tree.files:
        suffix = Path(f).suffix
        if f.endswith(("lean-audit.json", "lake-manifest.json")) or suffix not in CITED_BY + DOCS:
            continue
        p = tree.root / f
        try:
            if p.stat().st_size > CITE_MAX_BYTES:
                continue
            text = p.read_text(errors="replace")
        except OSError:
            continue
        hits = {m.rstrip(".") for m in DOTTED.findall(text)} & candidates
        for h in hits:
            (docs if suffix in DOCS else code)[h].append(f)
    return code, docs


def closure(facts: Facts, owner: list[int], seeds: set[int]) -> bytearray:
    """The constants the seeds reach through what each mentions and the declaration each belongs to."""
    reached = bytearray(len(facts.name))
    stack = list(seeds)
    for i in stack:
        reached[i] = 1
    shared = [bool(SHARED.search(x)) for x in facts.name]
    while stack:
        i = stack.pop()
        nxt = list(facts.used[i])
        if not shared[i] and owner[i] != i:
            nxt.append(owner[i])
        for j in nxt:
            if not reached[j]:
                reached[j] = 1
                stack.append(j)
    return reached


def kind_of(facts: Facts, i: int) -> str:
    for k in ("instance", "class", "structure"):
        if k in facts.flags[i]:
            return k
    return facts.kind[i]


def longest(nodes: list[str], imports: dict[str, list[str]], weight: dict[str, int]) -> tuple[int, list[str]]:
    """The heaviest import chain: its weight and its modules, from the deepest import to the importer at its top."""
    best: dict[str, tuple[int, str | None]] = {}
    for start in nodes:
        stack = [(start, False)]
        while stack:
            m, done = stack.pop()
            if m in best:
                continue
            if done:
                w, via = 0, None
                for i in imports.get(m, ()):
                    if best[i][0] > w:
                        w, via = best[i][0], i
                best[m] = (w + weight[m], via)
                continue
            stack.append((m, True))
            stack += [(i, False) for i in imports.get(m, ()) if i not in best]
    if not best:
        return 0, []
    top = max(best, key=lambda m: (best[m][0], m))
    chain, m = [], top
    while m is not None:
        chain.append(m)
        m = best[m][1]
    return best[top][0], chain[::-1]


def contract(nodes: list[str], imports: dict[str, list[str]], gone: set[str]) -> dict[str, list[str]]:
    """Each kept module's imports once the modules in ``gone`` are deleted: what a deleted import imported, transitively."""
    memo: dict[str, frozenset[str]] = {}

    def through(m: str) -> frozenset[str]:
        stack = [(m, False)]
        while stack:
            x, done = stack.pop()
            if x in memo:
                continue
            if done:
                memo[x] = frozenset(j for i in imports.get(x, ()) for j in (memo[i] if i in gone else (i,)))
                continue
            stack.append((x, True))
            stack += [(i, False) for i in imports.get(x, ()) if i in gone and i not in memo]
        return memo[m]

    return {m: sorted({j for i in imports.get(m, ()) for j in (through(i) if i in gone else (i,))})
            for m in nodes if m not in gone}


def analyse(tree: Tree, facts: Facts) -> dict:
    owner = owners(facts)
    n = len(facts.name)
    mod_of = [facts.modules[facts.module[i]]["module"] for i in range(n)]
    lock_names, lock_mods = lock_seeds(tree)
    is_decl = [owner[i] == i and i in facts.range for i in range(n)]
    candidates = {facts.name[i] for i in range(n) if is_decl[i] and tree.analysis(mod_of[i])}
    cited_code, cited_docs = citations(tree, candidates)
    sources: dict[str, set[int]] = defaultdict(set)
    for i, name in enumerate(facts.name):
        mod = mod_of[i]
        if name in lock_names:
            sources["lock"].add(i)
        if mod in lock_mods:
            sources["lock module"].add(i)
        if ".SecurityProofs." in f".{name}.":
            sources["SecurityProofs"].add(i)
        if any(mod == p or mod.startswith(p + ".") for p in SEEDED_PREFIXES):
            sources["properties"].add(i)
        if not tree.analysis(mod):
            sources["code"].add(i)
        if name == "main" or name.endswith(".main"):
            sources["main"].add(i)
        for fl in SEED_FLAGS:
            if fl in facts.flags[i]:
                sources[fl].add(i)
        if name in cited_code:
            sources["cited"].add(i)
    seeds = set().union(*sources.values())
    shared = [bool(SHARED.search(x)) for x in facts.name]

    def reads(seeds: set[int]) -> tuple[bytearray, int]:
        reached = closure(facts, owner, seeds)
        read = bytearray(n)
        for i in range(n):
            if reached[i] and not shared[i]:
                read[owner[i]] = 1
        return read, sum(reached)

    read_decl, reached = reads(seeds)
    log(f"{len(seeds)} seeds reach {reached} of {n} constants")
    read_without = {c: reads(set().union(*(v for k, v in sources.items() if k != c)))[0] for c in CAUTIOUS}

    # what a module owns: a copy of another module's equation lemma, which the reader's module realizes again, is not its own
    by_module: dict[str, list[int]] = defaultdict(list)
    owns: Counter = Counter()
    module_read: dict[str, bool] = defaultdict(bool)
    for i in range(n):
        mod = mod_of[i]
        if mod_of[owner[i]] != mod:
            continue
        owns[mod] += 1
        if is_decl[i]:
            by_module[mod].append(i)
        if read_decl[owner[i]]:
            module_read[mod] = True
    found = {m["module"]: m for m in facts.modules}
    imports = {m: [i for i in (found[m].get("imports") if found.get(m, {}).get("found") else tree.source_imports(m))
                   if i in tree.module_file]
               for m in tree.module_file}
    lines = {m: tree.lines(m) for m in tree.module_file}

    status: dict[str, str] = {}
    order: list[str] = []
    seen: set[str] = set()
    for m in sorted(tree.module_file):
        stack = [(m, False)]
        while stack:
            x, done = stack.pop()
            if done:
                order.append(x)
                continue
            if x in seen:
                continue
            seen.add(x)
            stack.append((x, True))
            stack += [(i, False) for i in imports[x] if i not in seen]
    for m in order:
        info = found.get(m)
        seeded = m in lock_mods
        if not info or not info.get("found"):
            status[m] = "unbuilt"
        elif module_read[m] or seeded:
            status[m] = "read"
        elif info.get("foreign"):
            status[m] = "attributes"
        elif owns[m]:
            status[m] = "unread"
        elif all(status.get(i) in ("unread", "unread aggregator") for i in imports[m]):
            status[m] = "unread aggregator"
        else:
            status[m] = "aggregator"
    gone = {m for m, s in status.items() if s in ("unread", "unread aggregator")}

    guarantee_owner: dict[str, str] = {}
    for lock in tree.locks.values():
        for sec in ("guarantees", "assumptions"):
            for g, v in (lock.get(sec) or {}).items():
                if isinstance(v, dict) and v.get("owner"):
                    guarantee_owner[g] = v["owner"]

    modules: dict[str, dict] = {}
    areas: dict[str, dict] = {}
    totals = Counter()
    for m in sorted(tree.module_file):
        decls = by_module.get(m, [])
        unread = [i for i in decls if not read_decl[i]]
        kept_only = {c: sum(1 for i in decls if read_decl[i] and not read_without[c][i]) for c in CAUTIOUS}
        span = lambda i: facts.range[i][1] - facts.range[i][0] + 1  # noqa: E731
        kinds = Counter(kind_of(facts, i) for i in unread)
        rec = {"package": tree.module_package[m], "area": tree.area(m), "file": tree.module_file[m], "lines": lines[m],
               "status": status[m], "declarations": len(decls), "unread": len(unread),
               "unread_lines": sum(span(i) for i in unread), "unread_kinds": dict(sorted(kinds.items())),
               "kept_only_by": kept_only, "foreign_attributes": (found.get(m) or {}).get("foreign", []),
               "unread_declarations": [[facts.name[i], kind_of(facts, i), span(i)] for i in sorted(unread, key=lambda i: facts.range[i])],
               "unread_cited_in_docs": sorted(facts.name[i] for i in unread if facts.name[i] in cited_docs)}
        modules[m] = rec
        a = areas.setdefault(rec["area"], {"package": rec["package"], "modules": 0, "lines": 0, "declarations": 0,
                                           "unread": 0, "unread_lines": 0, "unread_kinds": Counter(),
                                           "wholly_unread_modules": 0, "wholly_unread_lines": 0, "unbuilt_modules": 0,
                                           "unbuilt_lines": 0, "kept_only_by": Counter(), "unread_cited_in_docs": 0,
                                           "owners": Counter()})
        for k in ("lines", "declarations", "unread", "unread_lines"):
            a[k] += rec[k]
        a["modules"] += 1
        a["unread_kinds"].update(kinds)
        a["kept_only_by"].update(kept_only)
        a["unread_cited_in_docs"] += len(rec["unread_cited_in_docs"])
        if m in gone:
            a["wholly_unread_modules"] += 1
            a["wholly_unread_lines"] += lines[m]
        if status[m] == "unbuilt":
            a["unbuilt_modules"] += 1
            a["unbuilt_lines"] += lines[m]
        for i in decls:
            if facts.name[i] in guarantee_owner:
                a["owners"][guarantee_owner[facts.name[i]]] += 1
    kinds_total, kept_total = Counter(), Counter()
    for a in areas.values():
        a["unread_kinds"] = dict(sorted(a["unread_kinds"].items()))
        a["kept_only_by"] = {c: a["kept_only_by"][c] for c in CAUTIOUS}
        a["owners"] = dict(a["owners"].most_common())
        if a["package"] in ANALYSIS:
            for k in ("modules", "lines", "declarations", "unread", "unread_lines", "wholly_unread_modules",
                      "wholly_unread_lines", "unbuilt_modules", "unbuilt_lines"):
                totals[k] += a[k]
            kinds_total.update(a["unread_kinds"])
            kept_total.update(a["kept_only_by"])
    totals["unread_kinds"] = dict(kinds_total.most_common())
    totals["kept_only_by"] = {c: kept_total[c] for c in CAUTIOUS}

    nodes = sorted(m for m in tree.module_file if status[m] != "unbuilt")
    after = contract(nodes, imports, gone)
    kept = [m for m in nodes if m not in gone]
    read_lines = {m: max(0, lines[m] - modules[m]["unread_lines"]) for m in nodes}

    def chain(weight, graph, ms, into=None):
        if into is not None:
            if into not in graph and into not in ms:
                return None
            sub, stack = set(), [into]
            while stack:
                x = stack.pop()
                if x not in sub:
                    sub.add(x)
                    stack += graph.get(x, [])
            ms = sorted(sub)
        w, c = longest(ms, graph, weight)
        return {"weight": w, "modules": len(c), "lines": sum(lines[x] for x in c),
                "chain": [[x, lines[x], modules[x]["status"]] for x in c]}

    one = {m: 1 for m in nodes}
    chains = {}
    for name, into in (("overall", None), (CHAIN_TO, CHAIN_TO)):
        if into is not None and into not in tree.module_file:
            continue
        chains[name] = {
            "before_by_modules": chain(one, imports, nodes, into),
            "before_by_lines": chain(lines, imports, nodes, into),
            "after_by_modules": chain(one, after, kept, into),
            "after_by_lines": chain(lines, after, kept, into),
            "after_by_read_lines": chain(read_lines, after, kept, into),
        }
    return {"schema": "lean-unread/v1", "packages": tree.packages, "analysis": list(ANALYSIS),
            "facts": {"modules": len(facts.modules), "found": sum(1 for m in facts.modules if m.get("found")),
                      "constants": n, "declarations": sum(is_decl)},
            "seeds": {k: len(v) for k, v in sorted(sources.items())}, "seeded": len(seeds),
            "reached": reached, "totals": dict(totals), "areas": dict(sorted(areas.items())),
            "modules": modules, "chains": chains, "clashes": tree.clashes,
            "cited_by_code": {k: v[:3] for k, v in sorted(cited_code.items())}}


def fmt(n: int) -> str:
    return f"{n:,}"


def markdown(rep: dict, rev: str | None) -> str:
    t = rep["totals"]
    out = [f"# The Lean nothing reads{f' at {rev}' if rev else ''}", "",
           f"The security analysis's three trees ({', '.join(rep['analysis'])}): {fmt(t.get('modules', 0))} modules, "
           f"{fmt(t.get('lines', 0))} lines, {fmt(t.get('declarations', 0))} declarations. Unread: "
           f"{fmt(t.get('unread', 0))} declarations ({fmt(t.get('unread_lines', 0))} lines); "
           f"{fmt(t.get('wholly_unread_modules', 0))} modules wholly unread ({fmt(t.get('wholly_unread_lines', 0))} lines), "
           f"deletable whole. {fmt(t.get('unbuilt_modules', 0))} modules ({fmt(t.get('unbuilt_lines', 0))} lines) have no "
           f"`.olean` (no package root imports them). {fmt(rep['facts']['constants'])} constants read from "
           f"{fmt(rep['facts']['found'])} of {fmt(rep['facts']['modules'])} modules' `.olean` files.", "",
           "Read only because a cautious seed keeps them (declarations): " +
           ", ".join(f"{CAUTIOUS_TEXT[c]} {fmt(v)}" for c, v in t.get("kept_only_by", {}).items()) + ".", "",
           "| area | modules | lines | declarations | unread | unread lines | wholly unread modules | their lines | owners (guarantees, assumptions) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    rows = sorted(((k, a) for k, a in rep["areas"].items() if a["package"] in rep["analysis"]),
                  key=lambda ka: (-ka[1]["unread_lines"] - ka[1]["wholly_unread_lines"], ka[0]))
    for k, a in rows:
        own = ", ".join(f"{o} {c}" for o, c in list(a["owners"].items())[:3])
        out.append(f"| {k} | {fmt(a['modules'])} | {fmt(a['lines'])} | {fmt(a['declarations'])} | {fmt(a['unread'])} | "
                   f"{fmt(a['unread_lines'])} | {fmt(a['wholly_unread_modules'])} | {fmt(a['wholly_unread_lines'])} | {own} |")
    out += ["", "Unread declarations by kind: " +
            ", ".join(f"{k} {fmt(v)}" for k, v in t.get("unread_kinds", {}).items()) + ".", ""]
    for name, c in rep["chains"].items():
        out += [f"## The longest import chain{'' if name == 'overall' else f' into {name}'}", "",
                "| | modules | lines |", "|---|---:|---:|"]
        for label, key in (("before, longest in modules", "before_by_modules"), ("before, longest in lines", "before_by_lines"),
                           ("after, longest in modules", "after_by_modules"), ("after, longest in lines", "after_by_lines")):
            ch = c[key]
            out.append(f"| {label} | {fmt(ch['modules'])} | {fmt(ch['lines'])} |" if ch else f"| {label} | gone | |")
        if ch := c["after_by_read_lines"]:
            out.append(f"| after, also without unread declarations' lines | {fmt(ch['modules'])} | {fmt(ch['weight'])} read lines |")
        for label, key in (("Before, longest in modules", "before_by_modules"), ("After, longest in modules", "after_by_modules"),
                           ("Before, longest in lines", "before_by_lines"), ("After, longest in lines", "after_by_lines")):
            if ch := c[key]:
                out += [f"{label}, from the deepest import: {' → '.join(x for x, _, _ in ch['chain'])}.", ""]
    out += ["## Seeds", "", ", ".join(f"{k} {fmt(v)}" for k, v in rep["seeds"].items()) +
            f": {fmt(rep['seeded'])} constants, reaching {fmt(rep['reached'])}.", "",
            "## What the closure can't see", "",
            "- A constant named only inside a syntax quotation or by `Name` (a macro's expansion, `open private`, `export`, "
            "an attribute naming one) is not a mention; a use of the expansion in a proof is.",
            "- A definition reached only through a definitional step a tactic took (`show`, `change`, `unfold` to a "
            "defeq target, a `decide` that reduced through it) leaves no constant in the term: deleting it breaks the "
            "proof, and the Lean check says so.",
            "- `@[defeq]` theorems are seeds (kept), since `dsimp` and `simp only [t]` use them with no trace, as are "
            "every declaration of a module a lock lists and each one a file outside Lean names; `kept_only_by` counts "
            "what each alone keeps.",
            "- A module with no read declaration that gives an instance or `simp` attribute to another module's constant is "
            "kept (`attributes`), and a module no package root imports is `unbuilt`, reported but not read.",
            "- Deleting a wholly unread module that a kept module imports means that importer imports what it imported.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--env", default="Security/Proofs", help="the package whose `lake env` reads the modules")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--setup", action="store_true", help="run setup.sh ENV first")
    ap.add_argument("--build", action="store_true", help="lake build ENV and the other analysis packages first")
    ap.add_argument("--facts", type=Path, help="read these facts instead of running Unread.lean")
    ap.add_argument("--rev", help="the commit the report names")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    tree = Tree(a.root, tracked(a.root))
    if a.build:
        build(tree, a.env, setup=a.setup)
    elif a.setup:
        subprocess.run(["bash", str(HERE / "setup.sh"), str(tree.root / a.env)], check=True)
    facts = Facts(a.facts or write_facts(tree, a.env, a.out))
    rep = analyse(tree, facts)
    rep["rev"] = a.rev
    (a.out / "unread.json").write_text(json.dumps(rep, indent=1) + "\n")
    (a.out / "unread.md").write_text(markdown(rep, a.rev))
    t = rep["totals"]
    print(f"unread: {t.get('unread', 0)} of {t.get('declarations', 0)} declarations unread ({t.get('unread_lines', 0)} lines), "
          f"{t.get('wholly_unread_modules', 0)} modules wholly unread ({t.get('wholly_unread_lines', 0)} lines); "
          f"{a.out / 'unread.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
