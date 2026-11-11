"""`unread`: the Lean nothing reads, per module and per area, and what deleting it would do to the longest import chain.

    python tools/verity/lean/unread.py --out DIR [--root DIR] [--env PKG] [--setup] [--build] [--facts FILE ...]
                                       [--caps FILE [--update-caps]] [--rev REV]

It reads every module of every Lake package under ``--root`` (default this checkout; `modules.packages`, the auditor's
fixtures aside) from the `.olean` files of one package's build (``--env``, default `security/Proofs`, whose `lake env`
reaches the security analysis's three trees and the code packages they require): `Unread.lean`, compiled with that
package's toolchain, writes each constant with what its type and value mention, and this holds the policy. ``--setup``
first runs `setup.sh ENV`, and ``--build`` `lake build` there, with `@NAME` for each other analysis package
(`ANALYSIS`). ``--facts FILE`` reads facts written before instead; given more than once, it joins them (`Facts`), as the
Lean check's audits of `security` and `security/Proofs` each write their own (`check.py`'s `unread_facts`).

**The closure.** Starting from the seeds, it follows what each reached constant's type and value mention
(`ConstantInfo.getUsedConstants`, an inductive type's constructors, `@[implemented_by]` and `@[csimp]`), and from any
constant to the declaration it belongs to. The seeds:

- every key and string value of each package's `lean-audit.json` (`reads` and `dependencies` aside) that names a
  constant: the guarantees, assumptions, witnesses, claims, `shape` and `verifier` entries, escapes;
- every declaration of the modules a lock names under `roots`, `runs`, `exempt` and `compile_time`;
- every declaration, and every declaration of each module, that a lock's texts (its strings with a space: a `why`, a
  reason) name, or that a document those texts name holds (a `PROTOCOL.md` a guarantee relies on; `lock_docs`);
- every constant with a `SecurityProofs` component, every declaration of `Proofs.Properties.*` and `Properties.*`;
- every constant of a code package (every package outside `ANALYSIS`: executable code), and every `main`;
- every constant Lean uses by name rather than through a term: syntax, macros, elaborators, tactics, delaborators,
  simprocs, attributes, options and `IO` (`meta`), unsafe and partial ones (compiled code), `@[defeq]` theorems,
  which `dsimp` and `simp only [t]` use with no trace in the proof term, and the instances of the coercion classes
  (`COERCION`), which elaboration unfolds where it inserts a coercion (`(w : Nat)` becomes `unwrap w`);
- every declaration of the analysis trees that a file outside Lean names in full (`.py`, `.json`, `.toml`, `.rs`, `.sh`,
  ...; `CITED_BY`), and every declaration of a module that code or a suite's `pyproject.toml` names (`MODULE_CITED_BY`:
  its dotted name, or its file's path from the root or from its package's source directory, or two or more of its
  path's last components, written out or built from string literals), so a test, a generator or a tool that reads one
  keeps it. This tool, its test and the Lean tools' fixtures read none (`NOT_READERS`).

The `@[defeq]` theorems, the coercion instances, the lock-listed modules, the names in the locks' texts and their
documents, and the cited names are seeds to be safe (`CAUTIOUS`): the report counts what each alone keeps
(`kept_only_by`).

**Declarations.** A constant with a declaration range is a declaration, unless it belongs to a type (a constructor, a
recursor, a projection) or Lean generated it from another's command (the same first line as a declaration its name
extends: `@[ext]`'s lemmas); every other constant (`match_`, `proof_`, `eq_`, `_unfold`, `.eq_def`, `casesOn`,
`noConfusion`, `_private`'s aux) belongs to the nearest declaration its name extends. A declaration is read when the
closure reaches it or any constant of it other than a matcher or an abstracted proof, which Lean shares between the
declarations of a module and makes again for a reader whose owner is gone. A module is wholly unread when it declares
something and nothing of it is read or seeded, it gives no instance or `simp` attribute to another module's constant,
and it isn't a module of checks (`CHECKS`: its name ends in `Checks` or `Vectors`) that mentions a read declaration of
another module, which tests a definition something reads; a module that declares nothing (an aggregator) goes with the
modules it imports when all of them go. What
mentions a wholly unread module's constants is unread too, but may sit in a module that stays: a wholly unread module is
deletable alone when nothing outside the modules deletable alone mentions it (`deletable`, which also reads the files'
`open` lines), and the rest go only with the unread declarations that mention them (`mentioned_by`).

**The chain.** The longest import chain among the tree's modules, in modules and in lines (each module's file), before
and after the wholly unread modules go, and after only those deletable alone go, each importer then importing what the
deleted module imported.

It writes ``DIR/unread.json`` (every module's counts and unread declarations, by area and by owner, the chains, the
seeds) and ``DIR/unread.md`` (the summary). The owners of an area are the `owner` handles the locks give the guarantees
and assumptions its modules declare: the tree has no other map from Lean to lanes.

**The ratchet.** Each area of the analysis's trees has a cap on its wholly unread modules and on their declarations
(`unread-caps.json`, `CAPS`), not on their lines, which a rewrap changes. A count may only come down: ``--caps FILE``
exits 1 when an area's count rose above its cap (an area the file doesn't list has a cap of nothing), and
``--update-caps`` lowers each cap to its count, never raising one. The Lean check runs it (`check`, from
`lean_audit.py`) on the facts its audits wrote, with no build; there the report is the short one (`analyse` without
``full``: no chains or `kept_only_by`). Neither this file nor the caps is in an audit's key (`lean_audit.NOT_READ`), so
changing them reruns no audit, and nor are the seeds outside Lean (`cited_kind`, `lock_docs`): drive's landing gate
(`unread_gate.py`) runs the ratchet on a change to them before it lands, from the facts a check of its Lean left
(`lean_audit.facts_file`). Those facts come from the builds of `security` and `security/Proofs` alone: a module of
`security/Properties` that `security/Proofs` doesn't import has no `.olean` there and is reported `unbuilt`, so a spec
definition that only such a module reads counts as unread. None does today (``--build`` builds all three trees). Stdlib
only.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
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

#: the security analysis's packages: the spec (Definitions), the proofs, the properties. Every other package is code.
ANALYSIS = ("security", "security/Proofs", "security/Properties")
#: the lock sections that name modules, every declaration of which is a seed
MODULE_SECTIONS = ("roots", "runs", "exempt", "compile_time")
#: the lock sections that name no seed: what the guarantees read (reached from them anyway) and the pinned builds
SKIPPED_SECTIONS = ("reads", "dependencies")
#: module prefixes whose every declaration is a seed: the properties of the code and their proofs
SEEDED_PREFIXES = ("Properties", "Proofs.Properties")
#: the flags `Unread.lean` writes that make a constant a seed
SEED_FLAGS = ("meta", "unsafe", "partial", "defeq")
#: the classes whose instances elaboration unfolds where it inserts a coercion, so no term that uses one mentions it
COERCION = frozenset({"Coe", "CoeTail", "CoeHead", "CoeOut", "CoeHTCT", "CoeHTC", "CoeOTC", "CoeTC", "CoeT", "CoeDep",
                      "CoeFun", "CoeSort"})
#: the seeds taken to be safe rather than because something reads them: each is reported with what it alone keeps
CAUTIOUS = ("defeq", "coercion", "lock module", "lock text", "lock doc", "cited")
CAUTIOUS_TEXT = {"defeq": "`@[defeq]` theorems", "coercion": "coercion instances", "lock module": "the modules locks list",
                 "lock text": "names in the locks' texts", "lock doc": "names in the docs the locks name",
                 "cited": "names in files outside Lean"}
#: a module of checks or vectors (its name's last component): theorems over concrete values that test the definitions
#: they mention, which nothing reads by design. It is kept when it mentions a declaration that is read (`checks`)
CHECKS = re.compile(r"(?:Checks|Vectors)$")
#: the ratchet's caps: each area's wholly unread modules and their declarations, which may only come down (`ratchet`)
CAPS = HERE / "unread-caps.json"
CAPS_SCHEMA = "lean-unread-caps/v1"
CAP_KEYS = ("modules", "declarations")
#: the files outside Lean whose full mention of a declaration keeps it (a test or tool reading it by name)
CITED_BY = (".py", ".json", ".toml", ".rs", ".sh", ".yaml", ".yml", ".txt", ".ts", ".js", ".cfg")
#: the files whose naming of a module keeps it: code, and a suite's declared inputs (`pyproject.toml`); data that only
#: records a module's name (a map of moves, an allowlist, a lakefile's roots) keeps nothing
MODULE_CITED_BY = (".py", ".sh", ".rs", ".ts", ".js")
#: the files whose mention of a declaration is reported but keeps nothing
DOCS = (".md",)
#: the files that name modules and declarations without reading them: this tool and its test (the Lean tools'
#: fixtures, `modules.FIXTURES`, neither)
NOT_READERS = ("tools/verity/lean/unread.py", "tools/verity/lean/Unread.lean", "tools/verity/lean/tests/test_unread.py")
#: a cited file larger than this is data, not code
CITE_MAX_BYTES = 4 << 20
#: a dotted identifier, as a file outside Lean would name a declaration: never from inside a word, and possessive, so a
#: long run of dots and letters is matched once (a quote a string closes with is stripped, `names`)
DOTTED = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_'!?]*+(?:\.[A-Za-z0-9_'!?]++)+")
#: a path to a Markdown file, as a lock's text names a document it relies on
DOC_PATH = re.compile(r"[A-Za-z0-9_./-]*[A-Za-z0-9_-]\.md\b")
#: a path to a Lean file, as a file outside Lean would name a module
LEAN_PATH = re.compile(r"[A-Za-z0-9_'!?./-]+\.lean\b")
#: string literals joined by `/` or `,`, the last a Lean file: a path code builds in pieces (`LEAN / "Prims" / "V.lean"`)
PIECES = re.compile(r"""(?:["'][^"'\n]*["']\s*[/,]\s*)+["'][^"'\n]*\.lean["']""")
LITERAL = re.compile(r"""["']([^"'\n]*)["']""")
#: what follows an `open` command's keyword, to the end of its line or a parenthesis
OPEN = re.compile(r"^[ \t]*open[ \t]+(?:scoped[ \t]+)?([^\n(]*)", re.M)
#: a private name's prefix, `_private.<module>.0.`, before the name it was declared with
PRIVATE = re.compile(r"^_private\..+?\.0\.")
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
        """`Proofs/Flock`, `Definitions/Pouw`, `Definitions` for the analysis's modules (the path under `security/`, two
        parts at most); a code package's path for its own."""
        pkg = self.module_package.get(mod, "?")
        if pkg not in ANALYSIS:
            return pkg
        rel = Path(self.module_file[mod]).relative_to("security").with_suffix("").parts
        return "/".join(rel[:2])

    def lines(self, mod: str) -> int:
        return len((self.root / self.module_file[mod]).read_text(errors="replace").splitlines())

    def source_imports(self, mod: str) -> list[str]:
        return HEADERS.imports((self.root / self.module_file[mod]).read_text(errors="replace")) or []


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
    with tempfile.TemporaryDirectory(prefix="lean-unread-") as d:
        exe = program(tree.root / env, Path(d))
        subprocess.run(["lake", "env", str(exe), str(cfg.resolve())], cwd=tree.root / env, check=True)
    return facts


def lines_of(path: Path):
    """A facts file's lines, gzipped (`.gz`, as an audit keeps them) or not."""
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as f:
        yield from f


class Facts:
    """`Unread.lean`'s lines: the modules, then the constants, each by its index. Several files (each package's audit writes
    its own) join by name: a module is the one a file found it in, its foreign attributes the union, and a constant two
    files hold mentions what either says it does (the same, since each reads the same `.olean`)."""

    def __init__(self, *paths: Path):
        self.modules: list[dict] = []
        self.name: list[str] = []
        self.module: array = array("I")
        self.kind: list[str] = []
        self.flags: list[frozenset] = []
        self.range: dict[int, tuple[int, int]] = {}
        self.parent: dict[int, str] = {}
        self.instance_of: dict[int, str] = {}
        self.used: list[array] = []
        self.index: dict[str, int] = {}
        self.module_index: dict[str, int] = {}
        flagsets: dict[tuple, frozenset] = {}
        for path in paths:
            local_module: list[int] = []
            rows: list[tuple[int, list[int]]] = []
            for line in lines_of(path):
                d = json.loads(line)
                if "module" in d:
                    local_module.append(self._module(d))
                    continue
                i = self.index.get(d["n"])
                if i is None:
                    i = self.index[d["n"]] = len(self.name)
                    self.name.append(d["n"])
                    self.module.append(local_module[d["m"]])
                    self.kind.append(sys.intern(d["k"]))
                    fl = tuple(d["f"])
                    self.flags.append(flagsets.setdefault(fl, frozenset(fl)))
                    if "r" in d:
                        self.range[i] = (d["r"][0], d["r"][1])
                    if "p" in d:
                        self.parent[i] = d["p"]
                    if "c" in d:
                        self.instance_of[i] = d["c"]
                    self.used.append(array("I"))
                rows.append((i, d["u"]))
            for i, used in rows:
                local = [rows[j][0] for j in used]
                self.used[i] = array("I", sorted(set(self.used[i]) | set(local))) if self.used[i] else array("I", local)

    def _module(self, d: dict) -> int:
        k = self.module_index.get(d["module"])
        if k is None:
            self.module_index[d["module"]] = len(self.modules)
            self.modules.append(d)
            return len(self.modules) - 1
        old = self.modules[k]
        if d.get("found") and not old.get("found"):
            self.modules[k] = d | {"foreign": sorted(set(d.get("foreign") or ()) | set(old.get("foreign") or ()))}
        elif d.get("found"):
            old["foreign"] = sorted(set(old.get("foreign") or ()) | set(d.get("foreign") or ()))
        return k


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


def lock_seeds(locks: dict[str, dict]) -> tuple[set[str], set[str], dict[str, list[str]]]:
    """The names and the modules the locks (by package, `Tree.locks`) give, and each lock's texts, by its package: the
    strings outside `SKIPPED_SECTIONS` with a space in them (a `why`, a reason). A string without one is a name, a prefix or
    a path, which seeds only a constant of its exact name."""
    seeds: set[str] = set()
    mods: set[str] = set()
    texts: dict[str, list[str]] = {}
    for pkg, lock in locks.items():
        out: set[str] = set()
        for k, v in lock.items():
            if k in MODULE_SECTIONS:
                mods.update(v if isinstance(v, list) else v.keys())
            if k not in SKIPPED_SECTIONS:
                walk_strings(v, out)
        seeds |= out
        texts[pkg] = sorted(s for s in out if " " in s)
    return seeds, mods, texts


def names(text: str) -> set[str]:
    """The dotted identifiers a text holds (`DOTTED`), each also without the quotes a string literal closes it with."""
    out = set()
    for m in DOTTED.findall(text):
        out.add(m)
        if m.endswith("'"):
            out.add(m.rstrip("'"))
    return out


def mentioned(text: str, known, modules: dict[str, str], named) -> tuple[set[str], set[str]]:
    """The names of ``known`` a text holds in full, and the modules it names: by dotted name or a file's path
    (``modules``), or by a path `path_namer` reads (``named``)."""
    dotted = names(text)
    mods = {modules[d] for d in dotted if d in modules}
    if ".lean" in text:
        for path in lean_paths(text):
            mods |= named(path)
    return {d for d in dotted if d in known}, mods


def lock_docs(tracked: list[str], texts: dict[str, list[str]]) -> dict[str, list[str]]:
    """{a document: the locks whose texts name it} (`DOC_PATH`): a path from the root or from the lock's package, else
    every tracked document whose path ends with it (a bare `PROTOCOL.md` is each one), the Lean tools' fixtures aside."""
    files = set(tracked)
    docs = [f for f in tracked if f.endswith(DOCS) and not f.startswith(MODULES.FIXTURES)]
    out: dict[str, set[str]] = defaultdict(set)
    for pkg, ts in texts.items():
        for t in ts:
            for p in DOC_PATH.findall(t):
                tail = "/".join(x for x in p.split("/") if x not in ("", ".", ".."))
                hits = [f for f in (os.path.normpath(p), os.path.normpath(f"{pkg}/{p}")) if f in files] or \
                    [f for f in docs if f == tail or f.endswith("/" + tail)]
                for h in hits:
                    out[h].add(f"{pkg}/lean-audit.json")
    return {d: sorted(v) for d, v in sorted(out.items())}


def lean_paths(text: str) -> list[str]:
    """The paths to Lean files a text names: written out, or built in pieces from string literals (`PIECES`)."""
    return LEAN_PATH.findall(text) + ["/".join(LITERAL.findall(m.group(0))) for m in PIECES.finditer(text)]


def path_namer(modules: dict[str, str]):
    """What a cited path names, given ``modules`` (each key a module's dotted name or one of its file's paths, to the
    module): the modules a tail of the path is a path of, and, for a path of two or more components, every module whose
    path ends with it (`"Prims/Vectors.lean"` names `Proofs.Core.Prims.Vectors`); a bare file name names no other."""
    by_file: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for k, m in modules.items():
        if k.endswith(".lean"):
            by_file[k.rsplit("/", 1)[-1]].append((k, m))

    def named(path: str) -> set[str]:
        parts = [p for p in path.split("/") if p not in ("", ".", "..")]
        out = {modules[s] for s in ("/".join(parts[k:]) for k in range(len(parts))) if s in modules}
        if len(parts) > 1:
            tail = "/" + "/".join(parts)
            out |= {m for k, m in by_file.get(parts[-1], ()) if ("/" + k).endswith(tail)}
        return out

    return named


def module_names(tree: Tree) -> dict[str, str]:
    """{each name a file outside Lean may give an analysis module by: its dotted name, its file's path from the root and
    from its package's source directory: the module}."""
    return {k: m for m, f in tree.module_file.items() if tree.analysis(m) for k in (m, f, m.replace(".", "/") + ".lean")}


def cited_kind(path: str) -> str | None:
    """What a file outside Lean keeps by naming it (`citations`): "module" for code and a suite's `pyproject.toml`
    (`MODULE_CITED_BY`), the declarations and the modules it names; "code" for the rest of `CITED_BY`, the declarations;
    "doc" for a document (`DOCS`), nothing unless a lock's text names it (`lock_docs`); None for a file that keeps nothing
    (a lock, a manifest, `NOT_READERS`, the Lean tools' fixtures, any other kind)."""
    suffix = Path(path).suffix
    if (path.endswith(("lean-audit.json", "lake-manifest.json")) or suffix not in CITED_BY + DOCS or path in NOT_READERS
            or path.startswith(MODULES.FIXTURES)):
        return None
    if suffix in DOCS:
        return "doc"
    return "module" if suffix in MODULE_CITED_BY or Path(path).name == "pyproject.toml" else "code"


def citations(tree: Tree, candidates: set[str], modules: dict[str, str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The candidate names files outside Lean mention in full, and the modules code names (`cited_kind`; ``modules``:
    `module_names`; a path as `path_namer` reads it): by code (`CITED_BY`), and by docs (`DOCS`), each with the files."""
    code: dict[str, list[str]] = defaultdict(list)
    docs: dict[str, list[str]] = defaultdict(list)
    named = path_namer(modules)
    for f in tree.files:
        if (kind := cited_kind(f)) is None:
            continue
        p = tree.root / f
        if not p.is_file() or p.stat().st_size > CITE_MAX_BYTES:
            continue
        text = p.read_text(errors="replace")
        dotted = names(text)
        hits = dotted & candidates
        if kind == "module":
            hits |= {modules[d] for d in dotted if d in modules}
            if ".lean" in text:
                for path in lean_paths(text):
                    hits |= named(path)
        for h in hits:
            (docs if kind == "doc" else code)[h].append(f)
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


def opened(text: str) -> list[str]:
    """The namespaces a Lean file's `open` commands name (`open scoped` too), up to `in`, `hiding`, `renaming` or a
    parenthesis."""
    out = []
    for rest in OPEN.findall(text):
        for word in rest.split("--", 1)[0].split():
            if word in ("in", "hiding", "renaming"):
                break
            out.append(word)
    return out


def deletable(facts: Facts, mod_of: list[str], shared: list[bool], gone: set[str],
              opens: dict[str, list[str]]) -> tuple[set[str], dict[str, set[str]]]:
    """The modules of ``gone`` deletable alone: the largest part of it whose constants nothing outside it mentions (a
    constant's type or value, a matcher or proof Lean shares aside; an instance or `simp` attribute another module gives;
    an `open` in another module, ``opens``, of a namespace only modules of ``gone`` hold, a name matching a namespace
    or its end). And each module of ``gone`` with the modules outside ``gone`` that mention it."""
    mentions: dict[str, set[str]] = defaultdict(set)
    for i, used in enumerate(facts.used):
        a = mod_of[i]
        for j in used:
            if mod_of[j] != a and mod_of[j] in gone and not shared[j]:
                mentions[a].add(mod_of[j])
    for m in facts.modules:
        for c in m.get("foreign") or ():
            j = facts.index.get(c)
            if j is not None and mod_of[j] in gone and mod_of[j] != m["module"]:
                mentions[m["module"]].add(mod_of[j])
    words = {w for ws in opens.values() for w in ws}
    ends = {w.rsplit(".", 1)[-1] for w in words}
    holders: dict[str, set[str]] = defaultdict(set)
    for i, name in enumerate(facts.name):
        parts = PRIVATE.sub("", name).split(".")[:-1]
        for k in range(len(parts)):
            if parts[k] in ends:
                holders[".".join(parts[:k + 1])].add(mod_of[i])
    by_end: dict[str, list[str]] = defaultdict(list)
    for ns in holders:
        by_end[ns.rsplit(".", 1)[-1]].append(ns)
    for a, ws in opens.items():
        for w in ws:
            held = set().union(*(holders[ns] for ns in by_end.get(w.rsplit(".", 1)[-1], ()) if ns == w or ns.endswith("." + w)))
            if held and held <= gone:
                mentions[a] |= held - {a}
    alone = set(gone)
    while bad := {b for a, bs in mentions.items() if a not in alone for b in bs if b in alone}:
        alone -= bad
    mentioned_by: dict[str, set[str]] = defaultdict(set)
    for a, bs in mentions.items():
        if a not in gone:
            for b in bs:
                mentioned_by[b].add(a)
    return alone, mentioned_by


def analyse(tree: Tree, facts: Facts, full: bool = True) -> dict:
    """The report (`unread.json`). Without ``full`` (the ratchet), no count of what each cautious seed alone keeps
    (`kept_only_by`) and no chains."""
    owner = owners(facts)
    n = len(facts.name)
    mod_of = [facts.modules[facts.module[i]]["module"] for i in range(n)]
    lock_names, lock_mods, lock_texts = lock_seeds(tree.locks)
    is_decl = [owner[i] == i and i in facts.range for i in range(n)]
    candidates = {facts.name[i] for i in range(n) if is_decl[i] and tree.analysis(mod_of[i])}
    named = module_names(tree)
    namer = path_namer(named)
    text_names, text_mods = set(), set()
    for ts in lock_texts.values():
        for t in ts:
            a, b = mentioned(t, facts.index, named, namer)
            text_names |= a
            text_mods |= b
    docs = lock_docs(tree.files, lock_texts)
    doc_names, doc_mods = set(), set()
    for d in docs:
        p = tree.root / d
        if p.is_file() and p.stat().st_size <= CITE_MAX_BYTES:
            a, b = mentioned(p.read_text(errors="replace"), facts.index, named, namer)
            doc_names |= a
            doc_mods |= b
    seeded_mods = lock_mods | text_mods | doc_mods
    cited_code, cited_docs = citations(tree, candidates, named)
    sources: dict[str, set[int]] = defaultdict(set)
    for i, name in enumerate(facts.name):
        mod = mod_of[i]
        if name in lock_names:
            sources["lock"].add(i)
        if mod in lock_mods:
            sources["lock module"].add(i)
        if name in text_names or mod in text_mods:
            sources["lock text"].add(i)
        if name in doc_names or mod in doc_mods:
            sources["lock doc"].add(i)
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
        if facts.instance_of.get(i) in COERCION:
            sources["coercion"].add(i)
        if name in cited_code or mod in cited_code:
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
    read_without = {c: reads(set().union(*(v for k, v in sources.items() if k != c)))[0] for c in CAUTIOUS} if full else {}
    checks = set()
    for i in range(n):
        m = mod_of[i]
        if m not in checks and mod_of[owner[i]] == m and CHECKS.search(m.rsplit(".", 1)[-1]) and \
                any(read_decl[owner[j]] and mod_of[j] != m for j in facts.used[i]):
            checks.add(m)

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
        seeded = m in seeded_mods
        if not info or not info.get("found"):
            status[m] = "unbuilt"
        elif module_read[m] or seeded:
            status[m] = "read"
        elif m in checks:
            status[m] = "checks"
        elif info.get("foreign"):
            status[m] = "attributes"
        elif owns[m]:
            status[m] = "unread"
        elif all(status.get(i) in ("unread", "unread aggregator") for i in imports[m]):
            status[m] = "unread aggregator"
        else:
            status[m] = "aggregator"
    gone = {m for m, s in status.items() if s in ("unread", "unread aggregator")}
    opens = {m: opened((tree.root / f).read_text(errors="replace")) for m, f in tree.module_file.items() if tree.analysis(m)}
    alone, mentioned_by = deletable(facts, mod_of, shared, gone, opens)

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
        kept_only = {c: sum(1 for i in decls if read_decl[i] and not without[i]) for c, without in read_without.items()}
        span = lambda i: facts.range[i][1] - facts.range[i][0] + 1  # noqa: E731
        kinds = Counter(kind_of(facts, i) for i in unread)
        rec = {"package": tree.module_package[m], "area": tree.area(m), "file": tree.module_file[m], "lines": lines[m],
               "status": status[m], "declarations": len(decls), "unread": len(unread),
               "unread_lines": sum(span(i) for i in unread), "unread_kinds": dict(sorted(kinds.items())),
               "kept_only_by": kept_only, "foreign_attributes": (found.get(m) or {}).get("foreign", []),
               "unread_declarations": [[facts.name[i], kind_of(facts, i), span(i)] for i in sorted(unread, key=lambda i: facts.range[i])],
               "unread_cited_in_docs": sorted(facts.name[i] for i in unread if facts.name[i] in cited_docs)}
        if m in gone:
            rec["deletable_alone"] = m in alone
            rec["mentioned_by"] = sorted(mentioned_by.get(m, ()))
        modules[m] = rec
        a = areas.setdefault(rec["area"], {"package": rec["package"], "modules": 0, "lines": 0, "declarations": 0,
                                           "unread": 0, "unread_lines": 0, "unread_kinds": Counter(),
                                           "wholly_unread_modules": 0, "wholly_unread_lines": 0, "alone_modules": 0,
                                           "alone_lines": 0, "unbuilt_modules": 0, "unbuilt_lines": 0,
                                           "kept_only_by": Counter(), "unread_cited_in_docs": 0, "owners": Counter()})
        for k in ("lines", "declarations", "unread", "unread_lines"):
            a[k] += rec[k]
        a["modules"] += 1
        a["unread_kinds"].update(kinds)
        a["kept_only_by"].update(kept_only)
        a["unread_cited_in_docs"] += len(rec["unread_cited_in_docs"])
        if m in gone:
            a["wholly_unread_modules"] += 1
            a["wholly_unread_lines"] += lines[m]
        if m in alone:
            a["alone_modules"] += 1
            a["alone_lines"] += lines[m]
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
                      "wholly_unread_lines", "alone_modules", "alone_lines", "unbuilt_modules", "unbuilt_lines"):
                totals[k] += a[k]
            kinds_total.update(a["unread_kinds"])
            kept_total.update(a["kept_only_by"])
    totals["unread_kinds"] = dict(kinds_total.most_common())
    totals["kept_only_by"] = {c: kept_total[c] for c in CAUTIOUS}

    nodes = sorted(m for m in tree.module_file if status[m] != "unbuilt")
    after = contract(nodes, imports, gone)
    kept = [m for m in nodes if m not in gone]
    after_alone = contract(nodes, imports, alone)
    kept_alone = [m for m in nodes if m not in alone]
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
    for name, into in (("overall", None), (CHAIN_TO, CHAIN_TO)) if full else ():
        if into is not None and into not in tree.module_file:
            continue
        chains[name] = {
            "before_by_modules": chain(one, imports, nodes, into),
            "before_by_lines": chain(lines, imports, nodes, into),
            "after_by_modules": chain(one, after, kept, into),
            "after_by_lines": chain(lines, after, kept, into),
            "after_by_read_lines": chain(read_lines, after, kept, into),
            "after_alone_by_modules": chain(one, after_alone, kept_alone, into),
            "after_alone_by_lines": chain(lines, after_alone, kept_alone, into),
        }
    return {"schema": "lean-unread/v1", "full": full, "packages": tree.packages, "analysis": list(ANALYSIS),
            "facts": {"modules": len(facts.modules), "found": sum(1 for m in facts.modules if m.get("found")),
                      "constants": n, "declarations": sum(is_decl)},
            "seeds": {k: len(v) for k, v in sorted(sources.items())}, "seeded": len(seeds),
            "reached": reached, "totals": dict(totals), "areas": dict(sorted(areas.items())),
            "modules": modules, "chains": chains, "clashes": tree.clashes,
            "cited_by_code": {k: v[:3] for k, v in sorted(cited_code.items())}, "lock_docs": docs}


def counts(rep: dict) -> dict[str, dict[str, int]]:
    """Each analysis area's wholly unread modules and their declarations, the areas with none aside: what the caps
    bound."""
    decls = Counter()
    for r in rep["modules"].values():
        if r["status"] in ("unread", "unread aggregator"):
            decls[r["area"]] += r["unread"]
    return {a: {"modules": v["wholly_unread_modules"], "declarations": decls[a]}
            for a, v in sorted(rep["areas"].items()) if v["package"] in rep["analysis"] and v["wholly_unread_modules"]}


def load_caps(path: Path) -> dict[str, dict[str, int]]:
    """The caps a file holds ({area: {"modules", "declarations"}}); ValueError when it isn't a caps file."""
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or doc.get("schema") != CAPS_SCHEMA or not isinstance(doc.get("areas"), dict):
        raise ValueError(f"{path} is not a {CAPS_SCHEMA} file")
    return doc["areas"]


def caps_text(areas: dict[str, dict[str, int]]) -> str:
    return json.dumps({"schema": CAPS_SCHEMA, "areas": dict(sorted(areas.items()))}, indent=1) + "\n"


def ratchet(rep: dict, caps: dict[str, dict[str, int]], where: str = str(CAPS.relative_to(ROOT))
            ) -> tuple[list[str], dict[str, dict[str, int]]]:
    """(a failure for each area whose wholly unread modules, or their declarations, rose above its cap in ``where``; the
    caps, each lowered to its count). The count only comes down: an area the caps don't list has a cap of nothing."""
    now, fails, zero = counts(rep), [], dict.fromkeys(CAP_KEYS, 0)
    for area in sorted(set(now) | set(caps)):
        got, cap = now.get(area, zero), zero | caps.get(area, {})
        if all(got[k] <= cap[k] for k in CAP_KEYS):
            continue
        mods = sorted((m for m, r in rep["modules"].items() if r["area"] == area and r["status"] in ("unread", "unread aggregator")),
                      key=lambda m: (-rep["modules"][m]["unread"], m))
        shown = ", ".join(f"{m} ({rep['modules'][m]['unread']} declarations)" for m in mods[:12])
        more = f" and {len(mods) - 12} more" if len(mods) > 12 else ""
        fails.append(f"unread: {area} has {got['modules']} wholly unread modules ({got['declarations']} declarations), above "
                     f"its cap of {cap['modules']} ({cap['declarations']} declarations) in {where}: {shown}{more}. Nothing "
                     f"reads such a module: delete it, or seed it: a lock/code reference (name the module, or a declaration "
                     f"of it, in a lock's text or in code that reads it). A module of checks (its name ending in Checks or "
                     f"Vectors) is kept once it mentions a declaration that is read, and a module moved to another area "
                     f"takes its cap with it")
    lowered = {a: {k: min(v.get(k, 0), now.get(a, zero)[k]) for k in CAP_KEYS} for a, v in caps.items()}
    return fails, {a: v for a, v in lowered.items() if v["modules"]}


def check(root: Path, files: list[str], facts: list[Path], out: Path, caps: Path = CAPS) -> dict:
    """The ratchet, as `lean_audit.py` runs it once the audits that write the facts have passed: the report of ``facts``
    joined (`Facts`, no chains or `kept_only_by`) in ``out`` (`unread.json`, `unread.md`), against ``caps``. Returns
    {"failures", "areas": each area's count and cap, "lower": whether a count is below its cap (``out``'s
    `unread-caps.json` then holds the caps at the counts), "seconds"}."""
    t = time.monotonic()
    rep = analyse(Tree(root, files), Facts(*facts), full=False)
    cap = load_caps(caps) if caps.is_file() else {}
    fails, lowered = ratchet(rep, cap, str(caps.relative_to(root)) if caps.is_relative_to(root) else str(caps))
    out.mkdir(parents=True, exist_ok=True)
    (out / "unread.json").write_text(json.dumps(rep, indent=1) + "\n")
    (out / "unread.md").write_text(markdown(rep, None))
    lower = not fails and lowered != cap
    if lower:
        (out / "unread-caps.json").write_text(caps_text(lowered))
    now = counts(rep)
    return {"failures": fails, "lower": lower, "seconds": round(time.monotonic() - t, 1),
            "areas": {a: {"count": now.get(a, dict.fromkeys(CAP_KEYS, 0)), "cap": cap.get(a)} for a in sorted(set(now) | set(cap))}}


def fmt(n: int) -> str:
    return f"{n:,}"


def markdown(rep: dict, rev: str | None) -> str:
    t = rep["totals"]
    out = [f"# The Lean nothing reads{f' at {rev}' if rev else ''}", "",
           f"The security analysis's three trees ({', '.join(rep['analysis'])}): {fmt(t.get('modules', 0))} modules, "
           f"{fmt(t.get('lines', 0))} lines, {fmt(t.get('declarations', 0))} declarations. Unread: "
           f"{fmt(t.get('unread', 0))} declarations ({fmt(t.get('unread_lines', 0))} lines); "
           f"{fmt(t.get('wholly_unread_modules', 0))} modules wholly unread ({fmt(t.get('wholly_unread_lines', 0))} lines), "
           f"{fmt(t.get('alone_modules', 0))} of them ({fmt(t.get('alone_lines', 0))} lines) deletable alone: nothing outside "
           f"them mentions them. Each of the rest goes only with the unread declarations of other modules that mention it "
           f"(`mentioned_by`). {fmt(t.get('unbuilt_modules', 0))} modules ({fmt(t.get('unbuilt_lines', 0))} lines) have no "
           f"`.olean` (no package root imports them). {fmt(rep['facts']['constants'])} constants read from "
           f"{fmt(rep['facts']['found'])} of {fmt(rep['facts']['modules'])} modules' `.olean` files.", "",
           *(["Read only because a cautious seed keeps them (declarations): " +
              ", ".join(f"{CAUTIOUS_TEXT[c]} {fmt(v)}" for c, v in t.get("kept_only_by", {}).items()) + ".", ""]
             if rep.get("full", True) else []),
           "| area | modules | lines | declarations | unread | unread lines | wholly unread modules | their lines | "
           "deletable alone (lines) | owners (guarantees, assumptions) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    rows = sorted(((k, a) for k, a in rep["areas"].items() if a["package"] in rep["analysis"]),
                  key=lambda ka: (-ka[1]["unread_lines"] - ka[1]["wholly_unread_lines"], ka[0]))
    for k, a in rows:
        own = ", ".join(f"{o} {c}" for o, c in list(a["owners"].items())[:3])
        out.append(f"| {k} | {fmt(a['modules'])} | {fmt(a['lines'])} | {fmt(a['declarations'])} | {fmt(a['unread'])} | "
                   f"{fmt(a['unread_lines'])} | {fmt(a['wholly_unread_modules'])} | {fmt(a['wholly_unread_lines'])} | "
                   f"{fmt(a['alone_modules'])} ({fmt(a['alone_lines'])}) | {own} |")
    out += ["", "Unread declarations by kind: " +
            ", ".join(f"{k} {fmt(v)}" for k, v in t.get("unread_kinds", {}).items()) + ".", ""]
    for name, c in rep["chains"].items():
        out += [f"## The longest import chain{'' if name == 'overall' else f' into {name}'}", "",
                "| | modules | lines |", "|---|---:|---:|"]
        for label, key in (("before, longest in modules", "before_by_modules"), ("before, longest in lines", "before_by_lines"),
                           ("after, longest in modules", "after_by_modules"), ("after, longest in lines", "after_by_lines"),
                           ("after only those deletable alone, in modules", "after_alone_by_modules"),
                           ("after only those deletable alone, in lines", "after_alone_by_lines")):
            ch = c[key]
            out.append(f"| {label} | {fmt(ch['modules'])} | {fmt(ch['lines'])} |" if ch else f"| {label} | gone | |")
        if ch := c["after_by_read_lines"]:
            out.append(f"| after, also without unread declarations' lines | {fmt(ch['modules'])} | {fmt(ch['weight'])} read lines |")
        out.append("")
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
            "coercion instances, which elaboration unfolds where it inserts the coercion, every declaration of a module "
            "a lock lists, each one (and each module) a lock's text or a document a lock names holds, and each one a "
            "file outside Lean names; `kept_only_by` counts what each alone keeps.",
            "- An `example`, `#guard` or `#eval` leaves nothing in the environment: a declaration only they use is "
            "reported unread, and a module only they use deletable alone; deleting it means deleting them.",
            "- A module of checks (theorems over concrete values or vectors, which the build checks) is read by nothing "
            "by design: it is kept when its name ends in `Checks` or `Vectors` and it mentions a declaration that is "
            "read (`checks`), or when a file outside Lean or a lock names it. A path code builds from string literals "
            "(`LEAN / \"Prims\" / \"Vectors.lean\"`) names it; one it computes otherwise (a glob, a bare file name "
            "under a variable) does not.",
            "- The ratchet reads the facts the audits of Security and security/Proofs write: there a module of "
            "security/Properties that security/Proofs doesn't import is `unbuilt`, so a spec definition only such a "
            "module reads counts as unread (none does today). `--build` builds all three trees.",
            "- A module with no read declaration that gives an instance or `simp` attribute to another module's constant is "
            "kept (`attributes`), and a module no package root imports is `unbuilt`, reported but not read.",
            "- Deleting a wholly unread module that a kept module imports means that importer imports what it imported, "
            "and an `open` of a namespace only deleted modules held goes too. `.olean` files don't record an `open`: "
            "the files' `open` lines are read, a name matching a namespace or its end and no namespace outside the tree "
            "known, so a module can count as mentioned that isn't, and deletable alone is a lower bound.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--env", default="security/Proofs", help="the package whose `lake env` reads the modules")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--setup", action="store_true", help="run setup.sh ENV first")
    ap.add_argument("--build", action="store_true", help="lake build ENV and the other analysis packages first")
    ap.add_argument("--facts", type=Path, action="append",
                    help="read these facts instead of running Unread.lean (an audit's `unread-facts.jsonl.gz`; repeat to join)")
    ap.add_argument("--caps", type=Path, help="check the counts against this caps file (`unread-caps.json`): exit 1 when an "
                                              "area's wholly unread modules or their declarations rose above its cap")
    ap.add_argument("--update-caps", action="store_true",
                    help="with --caps: lower each cap to its count (never raise one); a caps file not there yet gets the counts")
    ap.add_argument("--rev", help="the commit the report names")
    a = ap.parse_args(argv)
    if a.update_caps and not a.caps:
        ap.error("--update-caps needs --caps")
    a.out.mkdir(parents=True, exist_ok=True)
    tree = Tree(a.root, tracked(a.root))
    if a.build:
        build(tree, a.env, setup=a.setup)
    elif a.setup:
        subprocess.run(["bash", str(HERE / "setup.sh"), str(tree.root / a.env)], check=True)
    facts = Facts(*(a.facts or [write_facts(tree, a.env, a.out)]))
    rep = analyse(tree, facts)
    rep["rev"] = a.rev
    (a.out / "unread.json").write_text(json.dumps(rep, indent=1) + "\n")
    (a.out / "unread.md").write_text(markdown(rep, a.rev))
    t = rep["totals"]
    print(f"unread: {t.get('unread', 0)} of {t.get('declarations', 0)} declarations unread ({t.get('unread_lines', 0)} lines), "
          f"{t.get('wholly_unread_modules', 0)} modules wholly unread ({t.get('wholly_unread_lines', 0)} lines); "
          f"{a.out / 'unread.md'}")
    if not a.caps:
        return 0
    if a.update_caps and not a.caps.is_file():
        a.caps.write_text(caps_text(counts(rep)))
        print(f"unread: {a.caps} written with the counts")
        return 0
    fails, lowered = ratchet(rep, load_caps(a.caps) if a.caps.is_file() else {}, str(a.caps))
    for f in fails:
        print(f, file=sys.stderr)
    if a.update_caps:
        a.caps.write_text(caps_text(lowered))
        print(f"unread: {a.caps} lowered to the counts, none raised")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
