"""circuit-check's catalog as tasks: one ``circuit-catalog`` task building every catalog binding's record (check.py's records
step, `circuit-check --all --records-only`) and cutting them into each binding's slice and the part they share (``--slices``,
`circuit_check.units.slices`: `slice_of`, `SHARED`), one ``circuit-binding`` task per catalog binding checking that binding's
targets from its slice and the shared part alone (`circuit-check --binding NAME --slice`, `circuit_check.shard.binding`'s
names), so one binding's edit changes only the slices that read what it changed, and writing their Definitions' failures
(`failures`), and one ``circuit-coverage`` task holding the bindings circuit-check makes, from every record, to the ones
listed here.

Bindings are read from the binding files' syntax alone (`ast`), as circuit-check names them: each sidecar's (`*.circuit-check.py`
in a regular package under one of the directories `sidecars.discover` searches, `_packages`, named for its module as it names
it) root functions
``_X_roots``, its ``_model_roots`` (every model one binding) and each ``TEMPLATE_PARAMS`` entry; `targets.py`'s own root
functions, the templates checked at their example (``TEMPLATE_PARAMS``) and the registered roots (``<registered>``).  Each task's
inputs are what any run of the catalog reads (`base`): every run loads every registry and binding file, so a binding's own
share of them narrows inside circuit-check, by its read-set cache, not here.  Its var ``lookups`` holds the names, not the
bytes, of the paths where a run of the catalog can look beyond those files (`lookups`, at the var ``places``: `places`), so
a path added, removed or renamed there reruns them, and an edit to a file none reads reruns none; the binding runner refuses
a run whose report says it read or looked anywhere else (`runners.circuit_binding.undeclared`)."""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import tomllib
from pathlib import Path

from .. import contract
from ..graph import RUNNERS, GraphError
from ..runners import circuit_binding as RB

CHECKER = "tools/verity/circuit_check"
SOURCE = f"{CHECKER}/src/circuit_check"
TARGETS = f"{SOURCE}/targets.py"
SIDECARS = f"{SOURCE}/sidecars.py"
SUFFIX = ".circuit-check"
CATALOG = "circuit-catalog:records"
RECORDS = {"name": "records", "path": "out/circuit-records.tar.gz"}
#: the catalog task's slot: every binding task waits on it, so its builds run up to `cores` at once, as many as their recorded
#: peaks fit in `mem_gb` (its circuit-check's --jobs and --mem-gb), which bounds the run; within tasks.toml's `[slot]`
CATALOG_SLOT = {"cores": 32, "mem_gb": 80, "disk_gb": 8}
#: the directory of the catalog task's slices, and the part every binding's run shares
SLICES = "out/slices"
SHARED = {"name": "shared", "path": f"{SLICES}/shared.json"}
#: the directory of the circuit-binding tasks' ``failures`` outputs
FAILURES = "out/failures"
#: the binding circuit-check gives the registered roots, and its task's entry name
REGISTERED = ("<registered>", "_registered")
#: the binding of the templates `targets.py` checks at their example, those no sidecar's ``TEMPLATE_PARAMS`` names
EXAMPLES = "TEMPLATE_PARAMS"
#: where in a workspace member an editable install puts the member on the import path: its directory, ``src/`` or ``python/``
ENTRIES = ("", "/src", "/python")


def _declared(root: Path) -> list[str]:
    """What circuit-check's suite declares it reads of the tree (``[tool.verity.tests] inputs``, which the suite's guard holds it
    to)."""
    try:
        return tomllib.loads((root / CHECKER / "pyproject.toml").read_text())["tool"]["verity"]["tests"]["inputs"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as e:
        raise GraphError(f"{CHECKER}/pyproject.toml: no [tool.verity.tests] inputs ({e})") from e


def base(root: Path, files: list[str]) -> list[str]:
    """The files any circuit-check run of the catalog may read: the checker's package but its tests, and what its suite declares
    it reads of the tree (`_declared`), prose aside."""
    declared = _declared(root)
    dirs = tuple(p.rstrip("/") + "/" for p in (CHECKER, *declared))
    exact = set(declared)
    return [f for f in files if (f in exact or f.startswith(dirs)) and not f.startswith(f"{CHECKER}/tests/")
            and not f.endswith(".md")]


def places(root: Path, files: list[str]) -> dict:
    """Where a run of the catalog can look beyond its inputs' bytes, as `runners.circuit_binding.covering` reads it: ``under``,
    the checker and what its suite declares it reads (`_declared`; prose and tests too, which a probe sees), and ``at``, where a
    module lookup can find a module by its name: the root (the first entry of the import path of a run from there,
    `runners.circuit_binding.checker`'s ``-m``), each workspace member's places of `ENTRIES` that are directories, and each
    directory above a place of ``under``, where a package it is in, or a module beside it, is looked for."""
    under = sorted({p.rstrip("/") for p in (CHECKER, *_declared(root))})
    try:
        ws = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as e:
        raise GraphError(f"pyproject.toml: no [tool.uv.workspace] ({e})") from e
    dirs = {d for f in files for d in RB.above(f)}
    members = {d for m in ws.get("members", ()) for d in fnmatch.filter(dirs, m)}
    members -= {d for x in ws.get("exclude", ()) for d in fnmatch.filter(members, x)}
    at = {""} | {e for m in members for e in (m + s for s in ENTRIES) if e in dirs} | {d for u in under for d in RB.above(u)}
    return {"under": under, "at": sorted(at)}


def lookups(files: list[str], where: dict) -> str:
    """The digest of what the tree answers each lookup a run of the catalog can make beyond its inputs' bytes, at the places
    ``where`` (`places`): the name of every path under a place of ``under``; at each place of ``at``, each entry a lookup there
    can ask for by name (`runners.circuit_binding.named`), a directory's marked, and each package directory's ``__init__``;
    and the kind of each place of ``at`` and of each directory above a place.  A path added, removed or renamed where a run
    can look changes it; an edit to a file doesn't."""
    under, at = set(where["under"]), set(where["at"])
    prefixes = tuple(f"{u}/" for u in under)
    have, dirs = set(files), {d for f in files for d in RB.above(f)}
    names: dict[str, set[str]] = {e: set() for e in at}
    for f in files:
        for e in ("", *RB.above(f)):
            if e not in at:
                continue
            head, slash, tail = (f[len(e) + 1:] if e else f).partition("/")
            if not slash:
                if RB.named(head):
                    names[e].add(head)
            elif RB.named(head):
                names[e].add(f"{head}/")
                if head.isidentifier() and "/" not in tail and RB.package_init(tail):
                    names[e].add(f"{head}/{tail}")
    kinds = at | {d for p in (*under, *at) for d in RB.above(p)}
    return contract.digest(contract.canonical({
        "under": [f for f in files if f in under or f.startswith(prefixes)],
        "at": {e: sorted(n) for e, n in sorted(names.items())},
        "kinds": {p: "file" if p in have else "dir" for p in sorted(kinds) if p in have or p in dirs}}))


def _tuple(root: Path, name: str) -> tuple[str, ...]:
    """`sidecars`' tuple ``name``, read from its source."""
    for node in ast.parse((root / SIDECARS).read_text()).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [name]:
            return tuple(ast.literal_eval(node.value))
    raise GraphError(f"{SIDECARS}: no {name} tuple to find the sidecars' packages by")


def _packages(root: Path, have: set[str]) -> tuple[str, ...]:
    """The directories `sidecars.discover` searches (`sidecars._roots`), each with its slash: the top-level packages of
    `sidecars.PACKAGES`, at the root, and each package of `sidecars.TREES` (``<tree>/<stack>/<package>``) that registers
    itself by a sidecar of its own module."""
    own = (f"/__init__{SUFFIX}.py", f"/__init__{SUFFIX}.json")
    out = {p.partition(".")[0] + "/" for p in _tuple(root, "PACKAGES")}
    for tree in _tuple(root, "TREES"):
        out |= {f.rpartition("/")[0] + "/" for f in have if f.startswith(f"{tree}/") and f.count("/") == 3 and f.endswith(own)}
    return tuple(sorted(out))


def module(path: str, have: set[str], tops: tuple[str, ...]) -> str | None:
    """The module sidecar ``path`` binds, named from the top of the regular packages it is in (``__init__.py`` all the way
    down), as `sidecars.discover` names it; None in no regular package, or under none of the directories ``tops``."""
    parts = path.split("/")
    i = len(parts) - 1
    while i > 0 and "/".join(parts[:i]) + "/__init__.py" in have:
        i -= 1
    pkg, stem = parts[i:-1], parts[-1][:-len(f"{SUFFIX}.py")]
    if not pkg or not path.startswith(tops):
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
    have = set(files)
    tops = _packages(root, have)
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
    where = places(root, files)
    seen = {"lookups": lookups(files, where)}
    return [{"id": CATALOG, "kind": "circuit-catalog", "about": "build every circuit-check catalog binding's record and slice",
             "inputs": shared, "outputs": [RECORDS, SHARED, *(slice_of(name) for name in found)],
             "resources": CATALOG_SLOT, "vars": seen},
            {"id": "circuit-coverage:catalog", "kind": "circuit-coverage", "inputs": shared, "reads": reads,
             "resources": {"cores": 1, "mem_gb": 16, "disk_gb": 2},
             "about": "the circuit-check catalog's bindings are the circuit-binding tasks'",
             "vars": {"bindings": json.dumps(sorted(found), separators=(",", ":")), "records": RECORDS["path"], **seen}},
            *({"id": f"circuit-binding:{f}:{e}", "inputs": ins, "outputs": [failures(name)],
               "reads": [{"task": CATALOG, "output": SHARED["name"]}, {"task": CATALOG, "output": slice_of(name)["name"]}],
               "vars": {"binding": name, "slice": slice_of(name)["path"], "shared": SHARED["path"], **seen,
                        "places": json.dumps(where, separators=(",", ":"))}}
              for name, (f, e) in found.items())]
