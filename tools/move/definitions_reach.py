"""Which roots reach each file of the vLLM adapter's `definitions/`: a static import graph over every tracked `.py` and `.sh`.

    python3 tools/move/definitions_reach.py [--out FILE [--registered IDS.json]]

Edges: `import`/`from ... import` anywhere in a file (module level and in functions, relative ones too), the parents'
`__init__`, a string naming a module (`importlib.import_module("a.b")`, `"a.b:attr"`, a dotted literal), an f-string's
literal prefix naming a package (every module below it), a path literal naming a tracked file, a shell's `-m mod`,
`python path.py`, `from X import`/`import X`, and a `.sh` path; a sidecar `<m>.circuit-check.py` is reached with its
module `m` and its own imports are edges from `m`'s sidecar only. The catalog's lazy loads (`registry.catalog.load`'s f-string,
`definitions/__init__`'s `catalog_loader` of `registry.catalog` and `registry.boolean_gather`, and the makers of
`registry/__init__`'s lazy families) are the catalog edge, read once on and once off: each loads a module only when a
Definition id is resolved.

Roots, by class: `commands`, every console script outside `benchmarks/` and `tests/` (`[project.scripts]`), the `verity`
command, the applications' module strings, and the adapter's shells under `tools/served_run/` and `verity_vllm/ops/`;
`benchmarks`, every file under `benchmarks/`; `tests`, every file under `tests/`, every sidecar (circuit-check's targets) and
every other suite's file.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

REPO = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
DEFS = "adapters/vllm/verity_vllm/definitions/"
SIDECAR = ".circuit-check.py"
CATALOG = "adapters/vllm/verity_vllm/definitions/registry/catalog.py"
CLI = "adapters/vllm/verity_vllm/pipeline/cli.py"
#: what a command's or a benchmark's reach never enters: the check's tooling, the tests and the team's tooling
FENCES = {"commands": ("tests/", "benchmarks/", "infra/", "tools/", "adapters/vllm/tools/ratchet", "adapters/vllm/tools/migrate"),
          "tools": ("tests/", "benchmarks/"),
          "benchmarks": ("tests/", "infra/", "tools/check/", "tools/research/", "tools/verity/lean/", "adapters/vllm/tools/ratchet")}
OPEN = ("tools/verity/cli/", "tools/config/", "tools/difftest/")
DEFS_INIT = "adapters/vllm/verity_vllm/definitions/__init__.py"
#: its lazy families' makers load a module only when a Definition id is resolved, as the catalog does
REGISTRY_INIT = "adapters/vllm/verity_vllm/definitions/registry/__init__.py"
DOTTED = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
MODATTR = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):[A-Za-z_]\w*$")
SH_MOD = re.compile(r"-m\s+([A-Za-z_][\w.]*)")
SH_PY = re.compile(r"([\w./${}-]+\.(?:py|sh))\b")
SH_IMPORT = re.compile(r"(?:from\s+([A-Za-z_][\w.]*)\s+import|^\s*import\s+([A-Za-z_][\w.]*))", re.M)


def tracked() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).split("\n")


def module_index(files: list[str]) -> dict[str, list[str]]:
    """dotted name -> files, every top-level package (a directory with `__init__.py` whose parent has none) and every
    top-level module beside one."""
    have = set(files)
    index: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if not f.endswith(".py") or f.endswith(SIDECAR):
            continue
        parts = f[:-3].split("/")
        # climb while the directory is a package
        i = len(parts) - 1
        while i > 0 and "/".join(parts[:i]) + "/__init__.py" in have:
            i -= 1
        name = parts[i:]
        if name[-1] == "__init__":
            name = name[:-1]
        if name:
            index[".".join(name)].append(f)
    return index


def nearest(cands: list[str], frm: str) -> str:
    def common(a: str) -> int:
        n = 0
        for x, y in zip(a.split("/"), frm.split("/")):
            if x != y:
                break
            n += 1
        return n
    return max(cands, key=common)


class Graph:
    def __init__(self) -> None:
        self.files = [f for f in tracked() if f]
        self.have = set(self.files)
        self.index = module_index(self.files)
        self.by_prefix: dict[str, list[str]] = defaultdict(list)
        for m in self.index:
            parts = m.split(".")
            for k in range(1, len(parts)):
                self.by_prefix[".".join(parts[:k]) + "."].append(m)
        self.edges: dict[str, set[str]] = defaultdict(set)
        self.catalog_edges: dict[str, set[str]] = defaultdict(set)

    def resolve(self, mod: str, frm: str) -> list[str]:
        """the file of `mod` and its parents' `__init__`s"""
        out = []
        parts = mod.split(".")
        for k in range(1, len(parts) + 1):
            c = self.index.get(".".join(parts[:k]))
            if c:
                out.append(nearest(c, frm))
        return out

    def module_of(self, f: str) -> str | None:
        if not hasattr(self, "_rev"):
            self._rev = {x: m for m, fs in self.index.items() for x in fs}
        return self._rev.get(f)

    def add_py(self, f: str) -> None:
        try:
            tree = ast.parse((REPO / f).read_text(errors="replace"), filename=f)
        except SyntaxError:
            return
        src = f[:-len(SIDECAR)] + ".py" if f.endswith(SIDECAR) else f
        self.subcommands((REPO / f).read_text(errors="replace"), f, self.edges[f])
        pkg = self._pkg(src)
        edges = self.edges[f]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    edges.update(self.resolve(a.name, f))
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    if pkg is None:
                        continue
                    up = pkg.split(".")[: len(pkg.split(".")) - (node.level - 1)]
                    base = ".".join(up + ([base] if base else []))
                if not base:
                    continue
                edges.update(self.resolve(base, f))
                for a in node.names:
                    edges.update(self.index.get(f"{base}.{a.name}") and self.resolve(f"{base}.{a.name}", f) or [])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                self._string(node.value, f, edges)
            elif isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
                prefix = str(node.values[0].value)
                if prefix.endswith(".") and prefix in self.by_prefix:
                    target = self.catalog_edges[f] if f == CATALOG else edges
                    for m in self.by_prefix[prefix]:
                        target.update(self.resolve(m, f))
        if f == CLI:
            edges.intersection_update(self._cli_static)
        if f in (DEFS_INIT, REGISTRY_INIT):
            moved = {x for x in edges if "/definitions/registry/" in x and x != REGISTRY_INIT}
            self.catalog_edges[f] |= moved
            edges -= moved

    def _pkg(self, f: str) -> str | None:
        m = self.module_of(f)
        if m is None:
            return None
        return m if f.endswith("__init__.py") else m.rpartition(".")[0]

    def _string(self, s: str, f: str, edges: set[str]) -> None:
        s = s.strip()
        if len(s) > 300:
            return
        m = MODATTR.match(s)
        if m:
            edges.update(self.resolve(m[1], f))
        elif DOTTED.match(s) and s in self.index:
            edges.update(self.resolve(s, f))
        elif ("/" in s) and (s.endswith(".py") or s.endswith(".sh")):
            self._path(s, f, edges)

    def _path(self, p: str, f: str, edges: set[str]) -> None:
        p = p.lstrip("./")
        p = re.sub(r"^\$\{?\w+\}?/", "", p)
        if p in self.have:
            edges.add(p)
            return
        here = str(Path(f).parent / p)
        if here in self.have:
            edges.add(here)
            return
        if not hasattr(self, "_by_name"):
            self._by_name = defaultdict(list)
            for x in self.have:
                self._by_name[x.rsplit("/", 1)[-1]].append(x)
        hits = [x for x in self._by_name.get(p.rsplit("/", 1)[-1], ()) if x.endswith("/" + p)]
        if len(hits) == 1:
            edges.add(hits[0])

    def subcommands(self, text: str, f: str, edges: set[str]) -> None:
        for c in re.findall(r"(?:verity-vllm|verity_vllm\.pipeline\.cli)\s+([a-z0-9][a-z0-9-]*)", text):
            if c in self.commands:
                edges.update(self.resolve(self.commands[c], f))

    def add_sh(self, f: str) -> None:
        text = (REPO / f).read_text(errors="replace")
        edges = self.edges[f]
        self.subcommands(text, f, edges)
        for m in SH_MOD.findall(text):
            edges.update(self.resolve(m, f))
        for a, b in SH_IMPORT.findall(text):
            edges.update(self.resolve(a or b, f))
        for p in SH_PY.findall(text):
            self._path(p, f, edges)
        for s in re.findall(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", text):
            self._string(s, f, edges)

    def build(self) -> None:
        tree = ast.parse((REPO / CLI).read_text())
        self.commands = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Tuple) and isinstance(v.elts[0], ast.Constant):
                        self.commands[k.value] = v.elts[0].value
        self._cli_static = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    name = a.name if isinstance(node, ast.Import) else f"{node.module}.{a.name}"
                    self._cli_static.update(self.resolve(name, CLI))
                if isinstance(node, ast.ImportFrom) and node.module:
                    self._cli_static.update(self.resolve(node.module, CLI))
        for f in self.files:
            if f.endswith(".py"):
                self.add_py(f)
            elif f.endswith(".sh"):
                self.add_sh(f)
        # a module reaches its sidecar's loads only through circuit-check, which is a tests root
        for f in self.files:
            if f.endswith(SIDECAR):
                mod = f[:-len(SIDECAR)] + ".py"
                if mod in self.have:
                    self.edges[f].add(mod)

    def reach(self, roots: set[str], catalog: bool, fence: tuple[str, ...] = ()) -> set[str]:
        seen = set()
        todo = list(roots)
        while todo:
            x = todo.pop()
            if x in seen or (fence and x not in roots and x.startswith(fence) and not x.startswith(OPEN)):
                continue
            seen.add(x)
            todo.extend(self.edges.get(x, ()))
            if catalog:
                todo.extend(self.catalog_edges.get(x, ()))
        return seen


def roots(g: Graph) -> dict[str, set[str]]:
    commands: set[str] = set()
    tools: set[str] = set()
    for pp in [f for f in g.files if f.endswith("pyproject.toml")]:
        if pp.startswith(("benchmarks/", "tests/", "adapters/vllm/", "tools/verity/cli/")):
            continue
        scripts = tomllib.loads((REPO / pp).read_text()).get("project", {}).get("scripts", {})
        for target in scripts.values():
            tools.update(g.resolve(target.split(":")[0], pp))
    for f in g.files:
        if f.startswith("applications/") and f.endswith(".toml"):
            for s in re.findall(r'"([^"]+)"', (REPO / f).read_text()):
                g._string(s, f, commands)
        if (f.startswith("adapters/vllm/tools/served_run/") or f.startswith("adapters/vllm/verity_vllm/ops/")) and f.endswith(".sh"):
            commands.add(f)
        if f.startswith("tools/verity/cli/") and f.endswith(".py"):
            commands.add(f)
    bench = {f for f in g.files if f.startswith("benchmarks/") and f.endswith((".py", ".sh"))}
    tests = {f for f in g.files if f.endswith((".py", ".sh")) and (f.startswith("tests/") or f.endswith(SIDECAR))}
    tools |= {f for f in g.files if f.endswith((".py", ".sh")) and f.startswith(("tools/", "infra/", "adapters/vllm/tools/"))
              and not f.startswith("tools/verity/cli/")} - commands
    return {"commands": commands, "tools": tools, "benchmarks": bench, "tests": tests}


def lines(f: str) -> int:
    try:
        return len((REPO / f).read_text(errors="replace").splitlines())
    except (IsADirectoryError, FileNotFoundError):
        return 0


#: the registries' ids, as `--registered` reads them: run in an interpreter that ran nothing before, as
#: `tests/tools/verity/circuit_check/test_superseded_versions.py` reads them
REGISTERED = """
import json, sys
from circuit_check import sidecars, targets
from verity.circuits.defs import REGISTRY
targets.load_registries()
print(json.dumps({i: [sidecars.rel(sys.modules[sidecars.owner(d)].__file__), getattr(d, "word", None) is not None]
                  for i in targets.REGISTERED for d in [REGISTRY.defs[i]]}))
"""
ID = re.compile(r"(.+)_v(\d+)$")
KEPT = "tests/adapters/vllm/tests/program/test_superseded_kept.py"
SUPERSEDED = "tests/lint/allowlists/superseded_versions.json"
MODELS = "adapters/vllm/manifests/definitions.json"
CLASSES = ("commands: the verity command, the applications, the console scripts outside benchmarks/ and tests/, the adapter's "
           "served_run and ops shells and the verity-vllm subcommands they name; tools: the other console scripts and tools/; "
           "benchmarks: benchmarks/; tests: tests/ and circuit-check's sidecars. A command's reach never enters tests/, "
           "benchmarks/, infra/ or tools/ but the verity command and verity_config. catalog_on reads the by-id loads "
           "(registry.catalog.load, definitions/__init__'s catalog_loader, registry/__init__'s lazy families) as edges; "
           "catalog_off doesn't. Each row is the first class, in that order, that reaches the file; a sidecar and its pins "
           "are read with their module.")


def superseded(reg: dict[str, list]) -> dict[str, str]:
    """{each superseded id: the newest id of its name and kind}, as the superseded-versions test reads them."""
    fam: dict[tuple[str, bool], list[tuple[int, str]]] = defaultdict(list)
    for i, (_, boolean) in reg.items():
        if m := ID.fullmatch(i):
            fam[(m[1], boolean)].append((int(m[2]), i))
    out = {}
    for (name, _), vs in fam.items():
        top = max(v for v, _ in vs)
        out |= {i: f"{name}_v{top}" for v, i in vs if v < top}
    return out


def report(rows: dict[str, dict], reg: dict[str, list]) -> dict:
    py = {f: r for f, r in rows.items() if f.endswith(".py")}

    def listing(files) -> dict:
        files = sorted(files)
        return {"files": len(files), "lines": sum(rows[f]["lines"] for f in files), "list": files}

    models = {("adapters/vllm/" + k if k.startswith("verity_vllm/") else k): v
              for k, v in json.loads((REPO / MODELS).read_text())["models"].items()}
    models = {f: v for f, v in models.items() if f in rows}
    sup = superseded(reg)
    kept = set(re.findall(r'^    "(\w+_v\d+)": \(', (REPO / KEPT).read_text(), re.M))
    kept |= set(json.loads((REPO / SUPERSEDED).read_text())["held"])
    fam: dict[tuple[str, bool], list[tuple[int, str]]] = defaultdict(list)
    for i, (f, boolean) in reg.items():
        if m := ID.fullmatch(i):
            fam[(m[1], boolean)].append((int(m[2]), f))
    both = {f"{n} ({'Boolean' if b else 'word'})": [{"id": f"{n}_v{v}", "file": f, "superseded": f"{n}_v{v}" in sup}
                                                    for v, f in sorted(vs)]
            for (n, b), vs in sorted(fam.items()) if len({f.startswith("adapters/") for _, f in vs}) > 1}
    mine = sorted(i for i in sup if reg[i][0].startswith(DEFS))
    per = defaultdict(int)
    for i in reg:
        if reg[i][0].startswith(DEFS):
            per[reg[i][0]] += 1
    gone = defaultdict(int)
    for i in mine:
        gone[reg[i][0]] += 1
    return {
        "about": ("What of adapters/vllm/verity_vllm/definitions/ each class of root reaches (`reach`), and its superseded "
                  "Definition versions (`superseded`); written by tools/move/definitions_reach.py."),
        "classes": CLASSES,
        "python_files": len(py), "python_lines": sum(r["lines"] for r in py.values()),
        "reached_by_nothing": listing(f for f, r in py.items() if r["catalog_on"] == "none"),
        "commands_only_by_id": listing(f for f, r in py.items() if r["catalog_on"] == "commands" and r["catalog_off"] != "commands"),
        "tests_only": listing(f for f, r in py.items() if r["catalog_off"] == "tests"),
        "benchmarks_only": listing(f for f, r in py.items() if r["catalog_off"] == "benchmarks"),
        "tools_only": listing(f for f, r in py.items() if r["catalog_off"] == "tools"),
        "tests_only_with_the_by_id_loads": listing(f for f, r in py.items() if r["catalog_on"] == "tests"),
        "benchmarks_only_with_the_by_id_loads": listing(f for f, r in py.items() if r["catalog_on"] == "benchmarks"),
        "for_models_no_application_serves": {"source": MODELS, "models": sorted({m for v in models.values() for m in v}),
                                             **listing(models)},
        "families_on_both_sides": both,
        "superseded": {"registered_ids": len(reg), "superseded_ids": len(sup), "in_definitions": len(mine),
                       "kept_for_records": [i for i in mine if i in kept],
                       "modules_every_version_superseded": sorted(f for f in per if per[f] == gone[f]),
                       "to_delete": [{"id": i, "file": reg[i][0], "newest": sup[i]} for i in mine if i not in kept]},
        "reach": {f: rows[f] for f in sorted(rows)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="the data file to write (the summary alone is printed)")
    ap.add_argument("--registered", help="the registries' ids as JSON, the output of `uv run python -c \"$(python3 "
                                         "tools/move/definitions_reach.py --print-loader)\"`; without it no `superseded`")
    ap.add_argument("--print-loader", action="store_true", help="print the program that writes --registered's file")
    args = ap.parse_args()
    if args.print_loader:
        print(REGISTERED)
        return
    g = Graph()
    g.build()
    r = roots(g)
    reached = {(k, cat): g.reach(v, cat, FENCES.get(k, ())) for k, v in r.items() for cat in (True, False)}
    rows = {}
    for f in sorted(f for f in g.files if f.startswith(DEFS)):
        base = f[:-len(SIDECAR)] + ".py" if f.endswith(SIDECAR) else (f[:-len(".circuit-check.json")] + ".py" if f.endswith(".circuit-check.json") else f)
        key = base if base in g.have else f

        def first(cat: bool, key: str = key) -> str:
            return next((k for k in ("commands", "tools", "benchmarks", "tests") if key in reached[(k, cat)]), "none")
        rows[f] = {"lines": lines(f), "catalog_on": first(True), "catalog_off": first(False)}
    summary: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows.values():
        for view in ("catalog_on", "catalog_off"):
            summary[f"{view}:{row[view]}"][0] += 1
            summary[f"{view}:{row[view]}"][1] += row["lines"]
    if args.out:
        out = report(rows, json.loads(Path(args.registered).read_text())) if args.registered else {"reach": rows}
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    json.dump({k: {"files": v[0], "lines": v[1]} for k, v in sorted(summary.items())}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
