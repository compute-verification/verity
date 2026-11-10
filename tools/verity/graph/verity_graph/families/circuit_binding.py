"""circuit-check's catalog as tasks: one ``circuit-catalog`` task building every catalog binding's record (check.py's records
step, `circuit-check --all --records-only`) and cutting them into each binding's slice and the part they share (``--slices``,
`circuit_check.units.slices`: `slice_of`, `SHARED`), one ``circuit-binding`` task per catalog binding checking that binding's
targets from its slice and the shared part alone (`circuit-check --binding NAME --slice`, `circuit_check.shard.binding`'s
names), so one binding's edit changes only the slices that read what it changed, and writing their Definitions' failures
(`failures`), and one ``circuit-coverage`` task holding the bindings circuit-check makes, from every record, to the ones
listed here.

Bindings are read from the binding files' syntax alone (`ast`), as circuit-check names them: each sidecar's (`*.circuit-check.py`
under a regular package of `sidecars.PACKAGES`, named for its module as `sidecars.discover` names it) root functions
``_X_roots``, its ``_model_roots`` (every model one binding) and each ``TEMPLATE_PARAMS`` entry; `targets.py`'s own root
functions, the templates checked at their example (``TEMPLATE_PARAMS``) and the registered roots (``<registered>``).  Each task's
inputs are what any run of the catalog reads (`base`): every run loads every registry and binding file, so a binding's own
share of them narrows inside circuit-check, by its read-set cache, not here.  Its var ``lookups`` is what the tree answers the
lookups those runs make beyond those files' bytes (`lookups`), so a module file added where a run looks for one reruns them,
and an edit to a file none reads reruns none."""
from __future__ import annotations

import ast
import hashlib
import json
import tomllib
from pathlib import Path

from .. import contract
from ..graph import RUNNERS, GraphError

CHECKER = "tools/verity/circuit_check"
SOURCE = f"{CHECKER}/src/circuit_check"
TARGETS = f"{SOURCE}/targets.py"
SIDECARS = f"{SOURCE}/sidecars.py"
SUFFIX = ".circuit-check"
CATALOG = "circuit-catalog:records"
RECORDS = {"name": "records", "path": "out/circuit-records.tar.gz"}
#: the catalog task's slot, the large one (`check.tasks`' LEAN_SLOT, less its disk): every binding task waits on it, and its
#: builds run as many at once as the memory their heaviest hold leaves room for
CATALOG_SLOT = {"cores": 32, "mem_gb": 128, "disk_gb": 8}
#: the directory of the catalog task's slices, and the part every binding's run shares
SLICES = "out/slices"
SHARED = {"name": "shared", "path": f"{SLICES}/shared.json"}
#: the directory of the circuit-binding tasks' ``failures`` outputs
FAILURES = "out/failures"
#: the binding circuit-check gives the registered roots, and its task's entry name
REGISTERED = ("<registered>", "_registered")
#: the binding of the templates `targets.py` checks at their example, those no sidecar's ``TEMPLATE_PARAMS`` names
EXAMPLES = "TEMPLATE_PARAMS"
#: circuit-check's declaration of what its runs of the catalog read (`circuit_check.cli.READS`): its parts and their schema
READS = f"{CHECKER}/reads"
READS_PARTS = ("registries.json", "loaded.json", "bindings.json")
READS_SCHEMA = "circuit-check-reads/v1"
#: how a module file's name ends, on any platform (each of `importlib.machinery.all_suffixes()` ends so)
MODULE_SUFFIXES = (".py", ".pyw", ".pyc", ".pyd", ".so")


def base(root: Path, files: list[str]) -> list[str]:
    """The files any circuit-check run of the catalog may read: the checker's package but its tests, and what its suite declares
    it reads of the tree (``[tool.verity.tests] inputs``, which the suite's guard holds it to), prose aside."""
    try:
        declared = tomllib.loads((root / CHECKER / "pyproject.toml").read_text())["tool"]["verity"]["tests"]["inputs"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as e:
        raise GraphError(f"{CHECKER}/pyproject.toml: no [tool.verity.tests] inputs ({e})") from e
    dirs = tuple(p.rstrip("/") + "/" for p in (CHECKER, *declared))
    exact = set(declared)
    return [f for f in files if (f in exact or f.startswith(dirs)) and not f.startswith(f"{CHECKER}/tests/")
            and not f.endswith(".md")]


def _module_file(name: str) -> bool:
    return name.partition(".")[0].isidentifier() and name.endswith(MODULE_SUFFIXES)


def _at_root(files: list[str]) -> list[str]:
    """What a module lookup at the root can find: each module file there, each directory a package's name could be
    (``<name>/``) and the ``__init__`` file that makes one a regular package."""
    out = set()
    for f in files:
        head, slash, rest = f.partition("/")
        if not slash:
            if _module_file(head):
                out.add(head)
        elif head.isidentifier():
            out.add(f"{head}/")
            if "/" not in rest and rest.partition(".")[0] == "__init__" and _module_file(rest):
                out.add(f)
    return sorted(out)


def lookups(root: Path, files: list[str]) -> str:
    """The digest of what the tree answers each lookup a run of the catalog makes beyond its inputs' bytes: the kind of each path
    `READS` declares its runs probed and the names in each directory they listed (`circuit_check.cache.declared`'s ``probes``
    and ``listed``), and the module files and packages at the root, the first entry of the import path of a run from there
    (`runners.circuit_binding.checker`'s ``-m``), which `READS` needn't hold.  A file added or removed where a run looks
    changes it; an edit to a file doesn't."""
    have, got = set(files), {}
    for part in READS_PARTS:
        if (p := f"{READS}/{part}") in have:
            try:
                doc = json.loads((root / p).read_text())
            except (OSError, ValueError) as e:
                raise GraphError(f"{p}: {e}") from e
            if not isinstance(doc, dict) or doc.get("schema") != READS_SCHEMA:
                raise GraphError(f"{p} isn't {READS_SCHEMA}, the read sets this family keys its tasks' lookups on")
            got.update(doc)
    sets = [got.get("registries", {}), *got.get("groups", [])]
    dirs = {""}
    for f in files:
        i = f.rfind("/")
        while i > 0 and (d := f[:i]) not in dirs:
            dirs.add(d)
            i = f.rfind("/", 0, i)
    probes = {*got.get("loaded", ()), *(p for s in sets for p in s.get("probes", ()))}
    listed = sorted({d for s in sets for d in s.get("listed", ())})
    return contract.digest(contract.canonical({
        "probes": [[p, "file" if p in have else "dir"] for p in sorted(probes) if p in have or p in dirs],
        "listed": [[d, sorted({f[len(d) + 1:].partition("/")[0] for f in files if f.startswith(f"{d}/")})] for d in listed],
        "root": _at_root(files)}))


def _packages(root: Path) -> set[str]:
    """The top-level names of `sidecars.PACKAGES`, read from its source."""
    for node in ast.parse((root / SIDECARS).read_text()).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["PACKAGES"]:
            return {p.partition(".")[0] for p in ast.literal_eval(node.value)}
    raise GraphError(f"{SIDECARS}: no PACKAGES tuple to find the sidecars' packages by")


def module(path: str, have: set[str], tops: set[str]) -> str | None:
    """The module sidecar ``path`` binds, named from the top of the regular packages it is in (``__init__.py`` all the way
    down), as `sidecars.discover` names it; None outside one of ``tops``."""
    parts = path.split("/")
    i = len(parts) - 1
    while i > 0 and "/".join(parts[:i]) + "/__init__.py" in have:
        i -= 1
    pkg, stem = parts[i:-1], parts[-1][:-len(f"{SUFFIX}.py")]
    if not pkg or pkg[0] not in tops:
        return None
    return ".".join(pkg if stem == "__init__" else [*pkg, stem])


def _names(tree: ast.Module) -> dict[str, ast.stmt]:
    """The names a module binds at its top level, each with the statement that binds it."""
    out: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out[node.name] = node
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                out.update({n.id: node for n in ast.walk(t) if isinstance(n, ast.Name)})
        elif isinstance(node, ast.Import | ast.ImportFrom):
            out.update({(a.asname or a.name).partition(".")[0]: node for a in node.names})
    return out


def entries(path: str, source: str) -> list[str]:
    """The bindings binding file ``path`` (``source``) defines, by entry: each root function ``_X_roots`` (`targets._root_names`),
    ``_model_roots``, and ``TEMPLATE_PARAMS:<key>`` for each key of its ``TEMPLATE_PARAMS`` (``<template>`` or
    ``<template>@<name>``)."""
    names = _names(ast.parse(source, path))
    out = sorted(n for n in names if n.startswith("_") and n.endswith("_roots") and n != "_model_roots")
    out += ["_model_roots"] if "_model_roots" in names else []
    if (node := names.get("TEMPLATE_PARAMS")) is not None:
        value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
        if not isinstance(value, ast.Dict) or not all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in value.keys):
            raise GraphError(f"{path}: TEMPLATE_PARAMS is a dict literal with a string key per template, so that each entry "
                             "is a task of its own")
        out += [f"TEMPLATE_PARAMS:{k.value}" for k in value.keys]
    return out


def bindings(root: Path, files: list[str]) -> dict[str, tuple[str, str]]:
    """{binding name, as `circuit-check --binding` takes it: (its binding file, its entry)} for the tree at ``root``."""
    have, tops = set(files), _packages(root)
    out = {name: (TARGETS, entry) for name, entry in (REGISTERED, (EXAMPLES, EXAMPLES))}
    out.update({e: (TARGETS, e) for e in entries(TARGETS, (root / TARGETS).read_text()) if not e.startswith(f"{EXAMPLES}:")})
    for f in files:
        if f.endswith(f"{SUFFIX}.py") and (mod := module(f, have, tops)) is not None:
            out.update({f"{mod}{SUFFIX}#{e}": (f, e) for e in entries(f, (root / f).read_text())})
    return dict(sorted(out.items()))


def failures(name: str) -> dict:
    """Binding ``name``'s task's output: its Definition targets' failures, at a path of the binding's own under `FAILURES`."""
    return {"name": "failures", "path": f"{FAILURES}/{hashlib.sha256(name.encode()).hexdigest()[:16]}.json"}


def slice_of(name: str) -> dict:
    """The catalog task's output of binding ``name``'s slice, which that binding's task alone reads."""
    return {"name": f"slice {name}", "path": f"{SLICES}/{hashlib.sha256(name.encode()).hexdigest()[:16]}.json"}


def tasks(root: Path, files: list[str]) -> list[dict]:
    ins = base(root, files)
    found = bindings(root, files)
    if missing := sorted({f for f, _ in found.values()} - set(ins)):
        raise GraphError(f"binding files outside what circuit-check's suite declares it reads: {', '.join(missing[:5])} (add "
                         f"them to {CHECKER}/pyproject.toml's [tool.verity.tests] inputs)")
    reads = [{"task": CATALOG, "output": RECORDS["name"]}]
    shared = [*ins, f"{RUNNERS}/circuit_binding.py"]          # the catalog's and coverage's runners import the binding runner
    seen = {"lookups": lookups(root, files)}
    return [{"id": CATALOG, "kind": "circuit-catalog", "about": "build every circuit-check catalog binding's record and slice",
             "inputs": shared, "outputs": [RECORDS, SHARED, *(slice_of(name) for name in found)],
             "resources": CATALOG_SLOT, "vars": seen},
            {"id": "circuit-coverage:catalog", "kind": "circuit-coverage", "inputs": shared, "reads": reads,
             "resources": {"cores": 1, "mem_gb": 16, "disk_gb": 2},
             "about": "the circuit-check catalog's bindings are the circuit-binding tasks'",
             "vars": {"bindings": json.dumps(sorted(found), separators=(",", ":")), "records": RECORDS["path"], **seen}},
            *({"id": f"circuit-binding:{f}:{e}", "inputs": ins, "outputs": [failures(name)],
               "reads": [{"task": CATALOG, "output": SHARED["name"]}, {"task": CATALOG, "output": slice_of(name)["name"]}],
               "vars": {"binding": name, "slice": slice_of(name)["path"], "shared": SHARED["path"], **seen}}
              for name, (f, e) in found.items())]
