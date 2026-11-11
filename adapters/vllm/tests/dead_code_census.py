"""Dead-code census of `adapters/vllm`: static import reachability from the declared entry points (`census_roots.txt`), optionally
merged with a dynamic `sys.modules` record from the pod.  `tests/test_no_dead_modules.py` runs the static part on every test run against
the shrink-only keep-list `dead_code_keep.json`; this tool is the long form (classes, the deletion plan, artifacts) for a cleanup pass.
Its output is not committed: write it anywhere with --out.

    cd adapters/vllm
    python tests/dead_code_census.py [--imported imported.json] [--out /tmp/census.json] [--git-dates]
    python tests/dead_code_census.py --tests-for MODULE... [--out /tmp/tests_for.json]

ROOTS (`census_roots.txt`, reviewed by hand): shell entry points (paths or globs: every sweep runner and negatives runner by rule), Python
entry points, and directories of roots (the regression harness).  A root shell is parsed for `-m module`, `python <path>.py`,
`<name>.py` tokens and `from X import` / `import X` anywhere (heredocs, `-c` strings); every `.sh` a root shell names is a root too.
Shell scripts are entry points: the census never deletes one.  A module that only a non-root shell runs is non-live and must be kept
by name in the keep-list.

STATIC REACHABILITY: an ast import graph over every `.py` under the integration root but circuit-check's sidecars (`SIDECAR`, which
circuit-check loads by path).  Edges: `import a.b`, `from a.b import c`
(`c` as a submodule when it is one), relative imports, the parents of every imported dotted name (their `__init__`), string
literals that name an integration module (`importlib.import_module("verity_vllm.program.model")`, dotted registry paths), f-string
prefixes (`f"verity.definitions.registry.{m}"` + every short literal of the same file that completes to a module), path literals
(`verity_vllm/acquire/fa2_tap_src/build_ext.py`) and unique `<name>.py` literals joined at runtime.  Anything reachable from a
root is LIVE.  `verity_vllm/observe/fold/profiles/**` is resolved by name at runtime: the whole package is reachable when any consumer is.
`registry.catalog.load()` imports every module of `verity.definitions.registry` but `quarantine` (`pkgutil`): they are reachable from it.

CLASSES (per module of the census packages):
    LIVE       static_reachable or dynamically_imported
    TEST-ONLY  not LIVE, imported by at least one test file
    DEAD       not LIVE, imported by no test

DELETION PLAN: a non-LIVE module is kept when the keep-list names it (fixture generators and reference models that tests of LIVE
modules import, other lanes' protected paths, next-candidate clusters), otherwise deleted.  A test file goes when it imports a deleted
module, directly or through another deleted test (fixpoint); the harness dir never goes.  Artifacts: an unreferenced file (no tracked
file of the whole repo names it or an ancestor directory) is deleted only inside ARTIFACT_DELETE_SCOPES, and never a `.sh`.

TESTS FOR A SET (`--tests-for D`, the tests that go with a deletion): each test-tree file reaches the library modules it names
itself (an import, a string naming a module or its path, a `verity-vllm` command's name in a file that runs the CLI) and those
of the test-tree files it names (helpers, fixture modules), followed through them; never what a library module imports in
turn, so a test's targets are what it tests.  A test file whose every target is in D goes, one that reaches others too is
edited to drop D's cases, and the rest stay; a helper goes when every test reaching it goes.  After deleting, collection
passes and this census finds no new dead helper.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]  # adapters/vllm
REPO = ROOT.parents[1]
ROOTS_FILE = HERE / "census_roots.txt"
KEEP_FILE = HERE / "dead_code_keep.json"

PACKAGES = ("verity_vllm",)
CENSUS_PACKAGES = PACKAGES
#: the Definitions that moved from the package to `verity.definitions` (circuits', held to the keep-list as it stood): in the
#: graph and the census, each file keyed by its repository path
DEFINITIONS = REPO / "verity" / "definitions"
#: what of `DEFINITIONS` moved from the package: the modules at its top and these packages of it; any other package there is core's
MOVED_OUT = frozenset({"numerics", "registry", "tests"})
_TOPS = "|".join(PACKAGES + ("tests",))
_DEFINITIONS_MODULE = re.escape(".".join(DEFINITIONS.relative_to(REPO).parts))
TOP_MODULE_RE = re.compile(rf"^(?:{_TOPS}|{_DEFINITIONS_MODULE})(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
PATH_LITERAL_RE = re.compile(rf"^(?:{_TOPS}|verity/definitions)(?:/[A-Za-z0-9_.-]+)+\.py$")

#: Other lanes own these; the census reports them, the deletion plan never touches them (the keep-list names the non-live ones).
PROTECTED = (
    "verity_vllm/engine/profiles/",
    "verity_vllm/observe/fold/profiles/",
    "verity_vllm/query/",
    "verity_vllm/acquire/",
    "verity_vllm/check/match/global_match.py",
    "verity_vllm/correspondence/runtime.py",
)

#: row_pod.sh's Match runs run_config with `--skip census,...,golden,...`: the modules those stages would `-m` are statically reachable
#: (run_config's stage table names them) but never run by the live launcher.  Reported, not deleted.
RUN_CONFIG_SKIPPED_STAGE_MODULES = ("verity_vllm.properties.census", "verity_vllm.properties.noninterference", "verity_vllm.properties.golden")

#: a circuit-check sidecar (`circuit_check.sidecars.SUFFIX`, which this suite cannot import): its name is no module name, so nothing
#: imports it; circuit-check's discovery (`sidecars.discover`) loads it by path
SIDECAR = ".circuit-check.py"

ARTIFACT_DELETE_SCOPES = ("verity_vllm/ops/fixtures/",)
#: Unreferenced by name but read through a name pattern (file -> the reader).
ARTIFACT_KEEP: dict[str, str] = {
    **{f"tests/regression/expected/{r}.json": "harness contract read as expected/<row>.json" for r in (
        "olmoe-1b-7b__bf16__l40s__tp2__b8__i1024__o128__mixed__greedy__bi-eager", "qwen3-30b-a3b__bf16__l40s__tp2__b2__i1024__o128__mixed__greedy__bi-eager",
        "smollm2-135m__bf16__l40s__tp1__b16__i1024__o128__mixed__greedy__bi-eager")},
}

SHELL_MODULE_RE = re.compile(r"-m\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)")
SHELL_PY_RE = re.compile(r"(?<![\w/.-])((?:[A-Za-z0-9_./-]+/)?[A-Za-z0-9_]+\.py)(?![\w])")
SHELL_SH_RE = re.compile(r"(?<![\w.-])((?:[A-Za-z0-9_./$-]+/)?[A-Za-z0-9_]+\.sh)(?![\w])")
SHELL_IMPORT_RE = re.compile(rf"\b(?:from|import)\s+((?:{'|'.join(PACKAGES)}|{_DEFINITIONS_MODULE})(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")


# ---------------------------------------------------------------------------------------------------------------------------- files
def py_files() -> list[Path]:
    out: list[Path] = []
    for top in [ROOT / t for t in PACKAGES + ("tests",)] + [DEFINITIONS]:
        for p in sorted(top.rglob("*.py")):
            if "__pycache__" in p.parts or p.resolve() == Path(__file__).resolve():
                continue                                             # this tool names modules in string literals; it is not a consumer
            if p.name.endswith(SIDECAR) or p == DEFINITIONS / "__init__.py":
                continue                                             # verity.definitions' own __init__ is verity.ml's, not a Definition's
            if top == DEFINITIONS and len(parts := p.relative_to(top).parts) > 1 and parts[0] not in MOVED_OUT:
                continue                                             # core's package: neither in the graph nor in the census
            out.append(p)
    return out


def relpath(p: Path) -> str:
    """``p`` from ``adapters/vllm``, or from the repository outside it (the moved Definitions)."""
    return p.relative_to(ROOT if p.is_relative_to(ROOT) else REPO).as_posix()


def path_of(rel: str) -> Path:
    return REPO / rel if rel.startswith("verity/") else ROOT / rel


def in_census(rel: str) -> bool:
    return rel.split("/")[0] in CENSUS_PACKAGES or rel.startswith("verity/definitions/")


def modname(p: Path) -> str:
    parts = relpath(p).removesuffix(".py").split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_test_file(rel: str) -> bool:
    parts = rel.split("/")
    return parts[0] == "tests" or "tests" in parts[:-1] or parts[-1].startswith("test_")


def loc(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def protected(rel: str) -> str | None:
    return next((pre for pre in PROTECTED if rel == pre or rel.startswith(pre)), None)


def shell_files() -> list[str]:
    return sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.sh") if "__pycache__" not in p.parts)


def load_keep(path: Path = KEEP_FILE) -> dict[str, str]:
    """module -> why it is kept although no root reaches it."""
    with open(path) as f:
        return dict(json.load(f)["modules"])


# ---------------------------------------------------------------------------------------------------------------------------- graph
class Graph:
    def __init__(self) -> None:
        self.files = py_files()
        self.mod_of: dict[str, str] = {}            # rel path -> module
        self.file_of: dict[str, str] = {}           # module -> rel path
        for p in self.files:
            rel = relpath(p)
            m = modname(p)
            self.mod_of[rel] = m
            self.file_of[m] = rel
        self.by_basename: dict[str, list[str]] = defaultdict(list)   # "x.py" -> modules with that file name (non-__init__)
        for rel, m in self.mod_of.items():
            if Path(rel).name != "__init__.py":
                self.by_basename[Path(rel).name].append(m)
        self.edges: dict[str, set[str]] = defaultdict(set)      # module -> modules it imports (resolved, existing)
        for p in self.files:
            self._scan(p)

    def exists(self, m: str) -> bool:
        return m in self.file_of

    def add(self, src: str, dst: str) -> None:
        parts = dst.split(".")
        for m in [".".join(parts[:i]) for i in range(1, len(parts) + 1)]:
            if self.exists(m) and m != src:
                self.edges[src].add(m)

    @staticmethod
    def from_base(src_mod: str, src_is_pkg: bool, node: ast.ImportFrom) -> str:
        """The module a `from X import ...` reads ``X`` as, relative imports resolved."""
        if node.level == 0:
            return node.module or ""
        anchor = src_mod.split(".")
        if not src_is_pkg:
            anchor = anchor[:-1]
        anchor = anchor[: len(anchor) - (node.level - 1)] if node.level > 1 else anchor
        return ".".join(anchor + ([node.module] if node.module else []))

    def resolve_from(self, src_mod: str, src_is_pkg: bool, node: ast.ImportFrom) -> list[str]:
        base = self.from_base(src_mod, src_is_pkg, node)
        out = [base] if base else []
        for a in node.names:
            cand = f"{base}.{a.name}" if base else a.name
            if self.exists(cand):
                out.append(cand)
        return out

    def _scan(self, p: Path) -> None:
        rel = relpath(p)
        src_mod = self.mod_of[rel]
        try:
            tree = ast.parse(p.read_bytes(), filename=str(p))
        except SyntaxError as e:
            print(f"WARN syntax error {rel}: {e}", file=sys.stderr)
            return
        strings: list[str] = []
        fprefixes: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.add(src_mod, a.name)
            elif isinstance(node, ast.ImportFrom):
                for m in self.resolve_from(src_mod, p.name == "__init__.py", node):
                    self.add(src_mod, m)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                strings.append(node.value)
            elif isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant) \
                    and isinstance(node.values[0].value, str):
                fprefixes.append(node.values[0].value)
        for s in strings:
            if TOP_MODULE_RE.match(s) and self.exists(s):
                self.add(src_mod, s)
            elif PATH_LITERAL_RE.match(s) and s in self.mod_of:
                self.add(src_mod, self.mod_of[s])
            elif s.endswith(".py") and "/" not in s and len(self.by_basename.get(s, ())) == 1:
                self.add(src_mod, self.by_basename[s][0])            # os.path.join(REPO, "verity_vllm", "harness", "<name>.py")
        for pre in fprefixes:
            if not pre.endswith("."):
                continue
            pkg = pre[:-1]
            if not TOP_MODULE_RE.match(pkg) or not self.exists(pkg):
                continue
            self.add(src_mod, pkg)
            if pkg == "verity_vllm.observe.fold.profiles":                     # resolved by name from the workload's case at runtime
                for m in self.file_of:
                    if m.startswith(pkg + "."):
                        self.add(src_mod, m)
                continue
            if pkg == "verity.definitions.registry":                    # registry.catalog.load(): every module of it but quarantine
                for m in self.file_of:
                    if m.rsplit(".", 1)[0] == pkg and m != pkg + ".quarantine":
                        self.add(src_mod, m)
                continue
            for s in strings:
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s) and self.exists(f"{pkg}.{s}"):
                    self.add(src_mod, f"{pkg}.{s}")

    def closure(self, roots: set[str], skip: set[str] | None = None) -> set[str]:
        seen: set[str] = set()
        stack = [r for r in roots if self.exists(r)]
        while stack:
            m = stack.pop()
            if m in seen or (skip and m in skip):
                continue
            seen.add(m)
            stack.extend(self.edges.get(m, ()))
        return seen

    def why(self, roots: set[str], target: str) -> list[str] | None:
        """Shortest import path root -> ... -> target (BFS), or None."""
        from collections import deque
        prev: dict[str, str | None] = {r: None for r in roots if self.exists(r)}
        q = deque(prev)
        while q:
            m = q.popleft()
            if m == target:
                path = []
                while m is not None:
                    path.append(m)
                    m = prev[m]
                return path[::-1]
            for n in self.edges.get(m, ()):
                if n not in prev:
                    prev[n] = m
                    q.append(n)
        return None


# ---------------------------------------------------------------------------------------------------------------------------- roots
def read_roots_file(path: Path = ROOTS_FILE) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            out.append(entry)
    return out


def shells_named(text: str, shells: list[str]) -> set[str]:
    """Tracked shells a shell text names outside full-line comments (`# the sealed card comes from verify_lane.sh` runs nothing): by
    integration-relative path suffix, else by unique basename."""
    by_base: dict[str, list[str]] = defaultdict(list)
    for s in shells:
        by_base[os.path.basename(s)].append(s)
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    out: set[str] = set()
    for tok in SHELL_SH_RE.findall(code):
        tok = tok.lstrip("./")
        hits = [s for s in shells if tok == s or tok.endswith("/" + s) or s.endswith("/" + tok)]
        if not hits and len(by_base.get(os.path.basename(tok), ())) == 1:
            hits = by_base[os.path.basename(tok)]
        if len(hits) == 1:
            out.add(hits[0])
    return out


def shell_modules(g: Graph, text: str) -> set[str]:
    non_test_by_base: dict[str, list[str]] = defaultdict(list)
    for rel in g.mod_of:
        if not is_test_file(rel):
            non_test_by_base[os.path.basename(rel)].append(rel)
    mods = {m for m in SHELL_MODULE_RE.findall(text) if g.exists(m)}
    for m in SHELL_IMPORT_RE.findall(text):
        while m and not g.exists(m):                   # `from pkg.mod import name`: the longest existing prefix
            m = m.rpartition(".")[0]
        if m:
            mods.add(m)
    for tok in SHELL_PY_RE.findall(text):
        tok = tok.lstrip("./")
        if tok in g.mod_of:
            if not is_test_file(tok):                  # row_pod.sh names the tests that extract its shell blocks, in comments
                mods.add(g.mod_of[tok])
            continue
        cands = non_test_by_base.get(os.path.basename(tok), [])
        if len(cands) > 1:   # `$SRC/build_ext.py`: the dir the shell names (fa2_tap_src / norm_tap_src / router_tap_src) disambiguates
            cands = [c for c in cands if any(d in text for d in Path(c).parts[:-1] if d.endswith("_src"))]
        if len(cands) == 1:
            mods.add(g.mod_of[cands[0]])
    return mods


def resolve_roots(g: Graph, entries: list[str] | None = None) -> tuple[dict[str, set[str]], dict[str, set[str]], list[str]]:
    """(root module -> where it is declared, root shell -> modules it runs, entries that match nothing)."""
    entries = read_roots_file() if entries is None else entries
    shells = shell_files()
    by_mod: dict[str, set[str]] = defaultdict(set)
    root_shells: dict[str, str] = {}                   # shell -> the entry (or shell) that makes it a root
    unmatched: list[str] = []
    for e in entries:
        if e.endswith(".sh"):
            hits = [s for s in shells if fnmatch.fnmatchcase(s, e) or (e.startswith("**/") and fnmatch.fnmatchcase(s, e[3:]))]
            for s in hits:
                root_shells.setdefault(s, e)
        elif e.endswith("/"):
            hits = [rel for rel in g.mod_of if rel.startswith(e)]
            for rel in hits:
                by_mod[g.mod_of[rel]].add(e)
        else:
            hits = [e] if g.exists(e) else []
            if hits:
                by_mod[e].add(e)
        if not hits:
            unmatched.append(e)
    by_shell: dict[str, set[str]] = {}
    todo = list(root_shells)
    while todo:
        sh = todo.pop()
        if sh in by_shell:
            continue
        text = (ROOT / sh).read_text(errors="replace")
        by_shell[sh] = shell_modules(g, text)
        for m in by_shell[sh]:
            by_mod[m].add(sh)
        for s in shells_named(text, shells):
            if s not in root_shells:
                root_shells[s] = f"named by {sh}"
                todo.append(s)
    return by_mod, by_shell, unmatched


def static_live(g: Graph | None = None) -> tuple[Graph, set[str], set[str]]:
    """(graph, root modules, statically LIVE modules)."""
    g = g or Graph()
    by_mod, _, _ = resolve_roots(g)
    roots = set(by_mod)
    return g, roots, g.closure(roots)


def non_live_modules(g: Graph | None = None) -> dict[str, str]:
    """Production modules of the census packages no root reaches: module -> rel path (what the keep-list must cover)."""
    g, _, live = static_live(g)
    return {m: rel for rel, m in g.mod_of.items()
            if in_census(rel) and not is_test_file(rel) and m not in live}


def all_shell_refs(g: Graph) -> dict[str, set[str]]:
    """module -> every .sh under the integration that names it (root or not)."""
    out: dict[str, set[str]] = defaultdict(set)
    for rel in shell_files():
        for m in shell_modules(g, (ROOT / rel).read_text(errors="replace")):
            out[m].add(rel)
    return out


def git_dates(rels: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for rel in rels:
        r = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(path_of(rel))], cwd=ROOT, capture_output=True, text=True, check=False)
        out[rel] = r.stdout.strip() or None
    return out


# ---------------------------------------------------------------------------------------------------------------------------- artifacts
def artifact_refs() -> dict[str, dict]:
    """Every tracked non-.py file under the integration: which tracked files of the WHOLE repo name it -- by basename, or by one of its
    ancestor directories (data read by directory / glob)."""
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    self_files = {"adapters/vllm/tests/dead_code_census.py", "adapters/vllm/tests/dead_code_keep.json"}
    files = [f for f in files if f and not f.endswith(".py")]
    out: dict[str, dict] = {}

    def grep(needle: str) -> list[str]:
        r = subprocess.run(["git", "grep", "-l", "-I", "-F", "--", needle], cwd=REPO, capture_output=True, text=True, check=False)
        return [x for x in r.stdout.split("\n") if x and x not in self_files]

    dir_cache: dict[str, list[str]] = {}
    for f in files:
        self_rel = f"adapters/vllm/{f}"
        refs = {x for x in grep(os.path.basename(f)) if x != self_rel}
        dir_refs: dict[str, list[str]] = {}
        parts = f.split("/")[:-1]
        for i in range(len(parts), 0, -1):
            d = "/".join(parts[:i])
            if d in ("",) + PACKAGES + ("tests",):
                continue
            if any(p.suffix == ".py" for p in (ROOT / d).iterdir()):
                continue                                             # a code directory being named is not a reference to its data files
            if d not in dir_cache:
                dir_cache[d] = [x for x in grep(d + "/") if not x.startswith(f"adapters/vllm/{d}/")]
            if dir_cache[d]:
                dir_refs[d] = dir_cache[d]
        out[f] = {"bytes": os.path.getsize(ROOT / f), "referenced_by": sorted(refs), "directory_referenced_by": dir_refs}
    return out


def artifact_plan(arts: dict[str, dict], deleted_files: set[str]) -> dict:
    """Fixpoint: an artifact is unreferenced when every file naming it (or an ancestor directory) is itself deleted.  In scope -> delete;
    out of scope -> report.  Shell scripts are entry points: never in the plan."""
    gone = {f"adapters/vllm/{p}" for p in deleted_files}
    deleted: dict[str, str] = {}

    def unreferenced(e: dict) -> bool:
        return not [x for x in e["referenced_by"] if x not in gone] and \
            not any([x for x in xs if x not in gone] for xs in e["directory_referenced_by"].values())

    changed = True
    while changed:
        changed = False
        for f, e in arts.items():
            if f in deleted or f in ARTIFACT_KEEP or f.endswith(".sh") or not unreferenced(e):
                continue
            if f.startswith(ARTIFACT_DELETE_SCOPES):
                deleted[f] = "referenced by nothing" if not e["referenced_by"] else "referenced only by deleted files: " + ", ".join(e["referenced_by"])
                gone.add(f"adapters/vllm/{f}")
                changed = True
    out_of_scope = sorted(f for f, e in arts.items() if f not in deleted and f not in ARTIFACT_KEEP and not f.endswith(".sh") and unreferenced(e))
    return {"deleted": deleted, "deleted_bytes": sum(arts[f]["bytes"] for f in deleted),
            "unreferenced_out_of_scope": {f: arts[f]["bytes"] for f in out_of_scope},
            "unreferenced_shells": sorted(f for f, e in arts.items() if f.endswith(".sh") and unreferenced(e))}


# ---------------------------------------------------------------------------------------------------------------------------- tests of a set
CLI_MODULE = "verity_vllm.pipeline.cli"
#: a string that names a module, maybe with a name in it (`module.attr`, `module:attr`)
NAMED_RE = re.compile(rf"^((?:{_TOPS}|verity\.ml\.definitions)(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?::[A-Za-z_][A-Za-z0-9_.]*)?$")


def cli_commands(g: Graph) -> dict[str, str]:
    """`verity-vllm` command -> its module, from `cli.COMMANDS` as written (`{name: (module, help)}`), without importing it."""
    for node in ast.parse(path_of(g.file_of[CLI_MODULE]).read_text()).body:
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0] if isinstance(node, ast.Assign) else None
        if isinstance(target, ast.Name) and target.id == "COMMANDS" and isinstance(node.value, ast.Dict):
            return {k.value: v.elts[0].value for k, v in zip(node.value.keys, node.value.values)
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Tuple) and isinstance(v.elts[0], ast.Constant)}
    raise SystemExit(f"{g.file_of[CLI_MODULE]}: no `COMMANDS = {{name: (module, help)}}` literal to read the commands from")


def deepest(g: Graph, name: str) -> str | None:
    """The longest prefix of the dotted ``name`` that is a module of the graph."""
    parts = name.split(".")
    return next((m for m in (".".join(parts[:i]) for i in range(len(parts), 0, -1)) if g.exists(m)), None)


def named(g: Graph, rel: str, commands: dict[str, str]) -> set[str]:
    """The modules the file ``rel`` names itself, not the packages whose `__init__` an import runs on the way: what it imports,
    a string naming a module (`importlib.import_module`, a `monkeypatch.setattr` target, `"mod:fn"`) or its path, the
    modules other than packages the census graph resolves from it (a source file it reads by name), and, in a file that runs
    the CLI (it names `pipeline.cli`, or `verity-vllm` in a string), a command's name as its module."""
    src = g.mod_of[rel]
    out: set[str] = set()
    strings: set[str] = set()
    for node in ast.walk(ast.parse(path_of(rel).read_bytes(), filename=rel)):
        if isinstance(node, ast.Import):
            out |= {m for a in node.names if (m := deepest(g, a.name))}
        elif isinstance(node, ast.ImportFrom):
            base = g.from_base(src, Path(rel).name == "__init__.py", node)
            subs = {f"{base}.{a.name}" for a in node.names if g.exists(f"{base}.{a.name}")}
            if len(subs) < len(node.names) and (m := deepest(g, base)):
                out.add(m)                                   # a name the module defines, not a submodule
            out |= subs
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
    for s in strings:
        if (hit := NAMED_RE.match(s)) and (m := deepest(g, hit[1])) and "." in m:
            out.add(m)
        elif PATH_LITERAL_RE.match(s) and s in g.mod_of:
            out.add(g.mod_of[s])
    out |= {m for m in g.edges.get(src, ()) if Path(g.file_of[m]).name != "__init__.py"}
    if CLI_MODULE in out or "verity-vllm" in strings:
        out |= {commands[s] for s in strings if s in commands and g.exists(commands[s])}
    return out - {src}


def test_map(g: Graph) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(each test-tree file, packages' `__init__` aside -> the library modules it reaches; each -> the test-tree files it
    names).  A file reaches the library modules it names, and those of the test-tree files it names (a helper, a fixture
    module), followed through them; never what a library module imports in turn.  Only `verity_vllm` counts: a Definition
    of the registry a test reads is its input, not what it tests."""
    commands = cli_commands(g)
    files = sorted(rel for rel in g.mod_of if is_test_file(rel) and Path(rel).name != "__init__.py")
    direct = {rel: named(g, rel, commands) for rel in files}
    helpers = {rel: {g.file_of[m] for m in ms} & direct.keys() for rel, ms in direct.items()}
    targets: dict[str, set[str]] = {}
    for rel in files:
        seen = closure({rel}, helpers)
        targets[rel] = {m for f in seen for m in direct[f] if is_library(g, m)}
    return targets, helpers


def is_library(g: Graph, m: str) -> bool:
    return m.split(".", 1)[0] in PACKAGES and not is_test_file(g.file_of[m])


def closure(start: set[str], edges: dict[str, set[str]]) -> set[str]:
    seen, todo = set(start), list(start)
    while todo:
        for n in edges.get(todo.pop(), set()) - seen:
            seen.add(n)
            todo.append(n)
    return seen


def deleted_set(g: Graph, items: list[str]) -> set[str]:
    """The library modules ``items`` name: a module or a package (every module under it), by dotted name or by path from
    `adapters/vllm` (`verity_vllm/x/y.py`, a directory `verity_vllm/x/`)."""
    out: set[str] = set()
    for item in items:
        if item.endswith(".py"):
            mods = {g.mod_of[item]} if item in g.mod_of else set()
        else:
            pkg = item.rstrip("/").replace("/", ".")
            mods = {m for m in g.file_of if m == pkg or m.startswith(pkg + ".")}
        mods = {m for m in mods if is_library(g, m)}
        if not mods:
            raise SystemExit(f"--tests-for {item}: names no module of verity_vllm (a dotted module or package, or a "
                             f"path from adapters/vllm)")
        out |= mods
    return out


def tests_for(targets: dict[str, set[str]], helpers: dict[str, set[str]], deleted: set[str]) -> dict:
    """What deleting the modules ``deleted`` does to the tests (``targets`` and ``helpers`` from `test_map`): a test file whose
    every target is deleted goes with them, one that reaches others too is edited to drop the cases for the deleted ones,
    and the rest stay.  A helper goes when every test file reaching it goes, or when none does and its every target is
    deleted; one that stays and reaches a deleted module is edited."""
    named_by: dict[str, set[str]] = defaultdict(set)
    for rel, hs in helpers.items():
        for h in hs:
            named_by[h].add(rel)
    tests = {rel for rel in targets if Path(rel).name.startswith("test_")}
    goes = {rel for rel in tests if targets[rel] and targets[rel] <= deleted}
    helpers_go = set()
    for h in set(targets) - tests:
        users = closure({h}, named_by) & tests
        if users <= goes and (users or targets[h] and targets[h] <= deleted):
            helpers_go.add(h)
    edit = {rel: {"deleted": sorted(targets[rel] & deleted), "others": sorted(targets[rel] - deleted),
                  "named_for": named_for(rel, targets[rel] & deleted)}
            for rel in sorted(set(targets) - goes - helpers_go) if targets[rel] & deleted}
    return {"deleted": sorted(deleted), "goes": sorted(goes), "helpers_go": sorted(helpers_go), "edit": edit,
            "stay": len(tests - goes - edit.keys())}


def named_for(rel: str, modules: set[str]) -> str | None:
    """The module of ``modules`` a test file is named for (`test_<name>.py` or `test_<name>_<more>.py` for `....<name>`), if
    one: such a file is mostly about it, so it may go whole although it also reaches others."""
    stem = Path(rel).stem.removeprefix("test_")
    return next((m for m in sorted(modules, key=len, reverse=True)
                 if stem == m.rsplit(".", 1)[-1] or stem.startswith(m.rsplit(".", 1)[-1] + "_")), None)


def report_tests_for(items: list[str], out: str | None) -> int:
    g = Graph()
    plan = tests_for(*test_map(g), deleted_set(g, items))
    lines = {rel: loc(path_of(rel)) for rel in plan["goes"] + plan["helpers_go"]}
    edit = plan["edit"]
    print(f"deleting {len(plan['deleted'])} modules: {len(plan['goes'])} test files go ({sum(lines[r] for r in plan['goes'])} "
          f"lines), {len(plan['helpers_go'])} helpers go ({sum(lines[r] for r in plan['helpers_go'])} lines), {len(edit)} files "
          f"need edits ({sum(1 for e in edit.values() if e['named_for'])} of them named for a deleted module), {plan['stay']} "
          f"test files stay")
    for rel in plan["goes"]:
        print(f"  goes  {rel} ({lines[rel]} lines)")
    for rel in plan["helpers_go"]:
        print(f"  goes  {rel} ({lines[rel]} lines; a helper only the deleted tests reach)")
    for rel, e in edit.items():
        others = e["others"] if len(e["others"]) <= 3 else [*e["others"][:3], f"{len(e['others']) - 3} more"]
        print(f"  edit  {rel}: drop the cases for {', '.join(e['deleted'])}"
              + (f" (named for {e['named_for']})" if e["named_for"] else "") + f"; it also reaches {', '.join(others)}")
    if out:
        with open(out, "w") as f:
            json.dump(plan, f, indent=1)
            f.write("\n")
    return 0


# ---------------------------------------------------------------------------------------------------------------------------- census
def build(imported: set[str] | None, with_dates: bool, with_artifacts: bool) -> dict:
    g = Graph()
    roots_by_mod, roots_by_shell, unmatched = resolve_roots(g)
    roots = set(roots_by_mod)
    live = g.closure(roots)
    static = set(live)
    profiles_mods = {m for m in g.file_of if m.startswith("verity_vllm.observe.fold.profiles.")}
    live_sans_profiles = g.closure(roots, skip=profiles_mods)
    saved = set(g.edges["verity_vllm.pipeline.match"])
    g.edges["verity_vllm.pipeline.match"] -= set(RUN_CONFIG_SKIPPED_STAGE_MODULES)
    skipped_only = {m for m in live - g.closure(roots) if not is_test_file(g.file_of[m])}
    g.edges["verity_vllm.pipeline.match"] = saved
    if imported:
        live |= {m for m in imported if g.exists(m)}
    shell_refs = all_shell_refs(g)
    keep = load_keep()

    tests = {rel: m for rel, m in g.mod_of.items() if is_test_file(rel)}
    prod = {rel: m for rel, m in g.mod_of.items() if not is_test_file(rel) and in_census(rel)}
    imported_by_tests: dict[str, set[str]] = defaultdict(set)
    for rel, m in tests.items():
        for x in g.edges.get(m, ()):
            if not is_test_file(g.file_of[x]):
                imported_by_tests[x].add(rel)

    deleted_mods = {m for rel, m in prod.items() if m not in live and not protected(rel) and m not in keep}
    deleted_tests: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for rel, m in tests.items():
            if rel in deleted_tests or rel.startswith("tests/regression/") or Path(rel).name == "__init__.py":
                continue
            gone = sorted(x for x in g.edges.get(m, ()) if x in deleted_mods or g.file_of[x] in deleted_tests)
            if gone:
                deleted_tests[rel] = "imports deleted " + ", ".join(gone)
                changed = True
    broken = sorted((rel, x) for rel, m in g.mod_of.items() if m not in deleted_mods and rel not in deleted_tests
                    for x in g.edges.get(m, ()) if x in deleted_mods or g.file_of[x] in deleted_tests)
    aplan = artifact_plan(artifact_refs(), {g.file_of[m] for m in deleted_mods} | set(deleted_tests)) if with_artifacts else None

    dates = git_dates(sorted(prod)) if with_dates else {}
    modules: dict[str, dict] = {}
    for rel, m in sorted(prod.items()):
        c = "LIVE" if m in live else "TEST-ONLY" if imported_by_tests.get(m) else "DEAD"
        e = {"module": m, "package": rel.split("/")[0], "class": c, "static_reachable": m in static,
             "dynamically_imported": bool(imported and m in imported), "n_imported_by_tests": len(imported_by_tests.get(m, ())),
             "imported_by_tests": sorted(imported_by_tests.get(m, ()))[:8], "loc": loc(path_of(rel)), "last_commit_date": dates.get(rel),
             "referenced_by_shell": sorted(shell_refs.get(m, ())), "root_via": sorted(roots_by_mod.get(m, ())), "protected": protected(rel),
             "live_only_via_profiles": m in live and m not in live_sans_profiles and m not in profiles_mods,
             "why": g.why(roots, m) if m in static else None}
        if c != "LIVE":
            e["plan"] = "delete" if m in deleted_mods else "keep"
            e["kept_because"] = keep.get(m) or (f"protected: {e['protected']}" if e["protected"] else None)
        modules[rel] = e

    totals: dict[str, dict] = defaultdict(lambda: {"files": 0, "loc": 0})
    for e in modules.values():
        for k in (f"all::{e['class']}", f"{e['package']}::{e['class']}", "all::TOTAL", f"{e['package']}::TOTAL"):
            totals[k]["files"] += 1
            totals[k]["loc"] += e["loc"]
    plan_totals: dict[str, dict] = defaultdict(lambda: {"files": 0, "loc": 0})
    for e in modules.values():
        if e.get("plan") == "delete":
            for k in (e["package"], "all"):
                plan_totals[k]["files"] += 1
                plan_totals[k]["loc"] += e["loc"]
    for rel in deleted_tests:
        plan_totals["tests"]["files"] += 1
        plan_totals["tests"]["loc"] += loc(path_of(rel))

    return {
        "schema": "vllm-dead-code-census/v2",
        "roots": {"entries": read_roots_file(), "unmatched": unmatched, "shells": {s: sorted(ms) for s, ms in sorted(roots_by_shell.items())},
                  "non_root_shells": sorted(set(shell_files()) - set(roots_by_shell)), "n_roots": len(roots)},
        "dynamic_record": {"present": imported is not None, "n_modules": len(imported or ()),
                           "dynamic_not_static": sorted(m for m in (imported or ()) if g.exists(m) and m not in static and not is_test_file(g.file_of[m]))},
        "totals": dict(sorted(totals.items())),
        "plan_totals": dict(sorted(plan_totals.items())),
        "plan": {"modules": sorted(g.file_of[m] for m in deleted_mods), "tests": dict(sorted(deleted_tests.items())), "broken_imports": broken},
        "keep_stale": sorted(m for m in keep if not g.exists(m) or m in live),
        "live_only_via_run_config_skipped_stages": {g.file_of[m]: loc(path_of(g.file_of[m])) for m in sorted(skipped_only)},
        "live_only_via_profiles": sorted(rel for rel, e in modules.items() if e["live_only_via_profiles"]),
        "modules": modules,
        "artifacts": aplan,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--imported", default=None, help="the pod's sys.modules record (a list, or {'modules': list | {module: ...}})")
    ap.add_argument("--out", default=None, help="write the full census JSON here (not committed)")
    ap.add_argument("--git-dates", action="store_true", help="record last_commit_date per file (one git log per file)")
    ap.add_argument("--artifacts", action="store_true", help="index every non-.py artifact against the whole repo (git grep; needs a checkout)")
    ap.add_argument("--tests-for", nargs="+", metavar="MODULE", default=None,
                    help="instead of the census, what deleting these modules (dotted, or paths from adapters/vllm; a package "
                         "or directory is every module under it) does to the tests: the files that go, and those to edit "
                         "(--out: as JSON)")
    a = ap.parse_args(argv)
    if a.tests_for:
        return report_tests_for(a.tests_for, a.out)
    imported: set[str] | None = None
    if a.imported:
        with open(a.imported) as f:
            doc = json.load(f)
        mods = doc.get("modules", doc) if isinstance(doc, dict) else doc
        imported = set(mods if isinstance(mods, list) else mods.keys())
    census = build(imported, a.git_dates, a.artifacts)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(census, f, indent=1)
            f.write("\n")
    t = census["totals"]
    for pkg in ("all",) + CENSUS_PACKAGES:
        row = "  ".join(f"{c}: {t.get(f'{pkg}::{c}', {}).get('files', 0)} files / {t.get(f'{pkg}::{c}', {}).get('loc', 0)} loc"
                        for c in ("LIVE", "TEST-ONLY", "DEAD", "TOTAL"))
        print(f"{pkg:24s} {row}")
    print(f"plan: delete {dict(census['plan_totals'])}")
    for rel in census["plan"]["modules"]:
        print(f"  delete module {rel} ({census['modules'][rel]['class']}, {census['modules'][rel]['loc']} loc)")
    for rel, why in census["plan"]["tests"].items():
        print(f"  delete test   {rel}: {why}")
    r = census["roots"]
    print(f"roots: {r['n_roots']} modules from {len(r['shells'])} shells + entries; unmatched entries {r['unmatched']}; non-root shells {r['non_root_shells']}")
    print(f"keep-list stale: {census['keep_stale']}; broken imports after plan: {census['plan']['broken_imports']}; "
          f"dynamic-not-static: {census['dynamic_record']['dynamic_not_static']}")
    if census["artifacts"] is not None:
        ar = census["artifacts"]
        print(f"artifacts: delete {sorted(ar['deleted'])} ({ar['deleted_bytes']} bytes); unreferenced out of scope {len(ar['unreferenced_out_of_scope'])}; "
              f"unreferenced shells (entry points, kept) {ar['unreferenced_shells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
