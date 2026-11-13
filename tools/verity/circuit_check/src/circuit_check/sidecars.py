"""Each Definition module's circuit-check bindings and pins, in sidecar files beside it.

    fp8.py                    a Definition module (`verity.definitions.registry.fp8`)
    fp8.circuit-check.py      its bindings, in `targets.py`'s forms: root functions (``_boolean_roots``, any ``_X_roots``,
                              `targets.root_functions`), `_model_roots` entries and ``TEMPLATE_PARAMS`` entries; a Program
                              root too large to build whole goes in ``_by_calls_roots``, checked by its calls (`by_calls`);
                              a family registered over a range of statics, ``RANGES``: each rule and its spots (`ranges`);
                              a root function whose bindings may fail one by one, ``_X_bindings`` beside it: each as (label,
                              the bound Definition, or the error), so one that doesn't bind is named (`targets._root_function`)
    fp8.circuit-check.json    its pins (`checks`' lowering counts and circuit digest, ``type``, by kind: ``definitions``,
                              ``pieces``, ``units``; and the circuit digest of a gated Definition or of one bound at
                              served statics over `targets.MAX_GATES`, ``digests``, which keeps it in ``--all``; and a
                              family's struct budget, ``struct_max_gates``: {family: gates}, `checks.struct_budget`)

A module's bindings bind the Definitions it defines (a template module's, its templates), and its pins are the lowerings of its
Definitions (`owner`; a template's subcircuits for the template module), each the one its pin's key names alone (`pin_owner`),
so a pin's file is a function of the tree, never of what a process built first.  circuit-check finds every sidecar under the packages
that define Definitions or templates (`PACKAGES`, and each package of another distribution that registers itself, `TREES`:
none is named here) and loads a binding file by its path: its name is no module name, so no
registry enumeration imports it, and a Definition module imports nothing of circuit-check; the registries import every module
with a sidecar (`targets.load_registries`).  A sidecar whose module is gone is an error.  What circuit-check needs of another
distribution it reads the same way, from a sidecar that distribution owns, and never by importing it: the frontend vocabularies'
kinds (``_vocabulary_roots``, the word Call families) and the reader of a served workload's request Program
(``request_program(path)``, `targets.request_program`).  `targets.py` keeps what no one module
owns, and ``pins/`` the pins of Definitions no key names alone (made only inside another's body) or no module defines, one file
per pin (`entries`): ``pins/<kind>/<entries.name(key)>.json``, in a pins file's form (``{kind: {key: counts}}``).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

from . import cache as CC
from . import entries as E

REPO = Path(__file__).resolve().parents[5]
#: `verity`'s packages whose modules may have a sidecar: the ones that define Definitions (`targets._authored`)
PACKAGES = ("verity", "verity.circuits", "verity.definitions.kernels", "verity.core")
#: the repository's directories of other distributions' packages, one directory per serving stack holding the package that
#: attaches Verity to it.  A package there registers itself by a sidecar of its own module
#: (``adapters/<stack>/<package>/__init__.circuit-check.py``, or its ``.json``), and then every sidecar under it is found;
#: one inside a `PACKAGES` directory (the templates') is found there.  No distribution but `verity` is named here
TREES = ("adapters",)
SUFFIX = CC.SUFFIX
#: the pins of a Definition no module defines, one file per pin
SHARED_PINS = Path(__file__).with_name("pins")
PIN_KINDS = ("definitions", "pieces", "units", "digests")


def _locations(pkg: str) -> list[Path]:
    """Package ``pkg``'s directories as Python imports it; none for a name it imports no package under.  A loaded module
    answers for itself: `importlib.util.find_spec` refuses one that has no spec."""
    if (loaded := sys.modules.get(pkg)) is not None:
        return [Path(p).resolve() for p in getattr(loaded, "__path__", ())]
    spec = importlib.util.find_spec(pkg)
    return [Path(p).resolve() for p in (spec.submodule_search_locations or [])] if spec else []


def _roots() -> list[Path]:
    """The directories searched for sidecars: each of `PACKAGES`' own, and each package of `TREES` that registers itself."""
    out = [root for pkg in PACKAGES for root in _locations(pkg)]
    for tree in TREES:
        out += [d for d in sorted((REPO / tree).glob("*/*"))
                if any((d / f"__init__{SUFFIX}.{kind}").is_file() for kind in ("py", "json"))]
    return out


@cache
def discover() -> dict[str, dict[str, Path]]:
    """{module: {"py": its binding file, "json": its pins file}} for every sidecar under `_roots`, by module name: the name
    it has from the top of the regular packages it is in (``__init__.py`` all the way down), so a package inside another's
    tree names its own (`verity_flock`'s is inside `verity`'s).  One counts only where Python imports that top directory as
    the package of its name: a copy of a package behind a directory without ``__init__.py`` would name modules that another
    directory owns (``verity.protocols.flock.prover.python.verity_flock.rec_open``), and importing it registers their Definitions twice."""
    out: dict[str, dict[str, Path]] = {}
    imported: dict[Path, bool] = {}
    with CC.internal():
        for kind in ("py", "json"):
            for p in sorted({p for root in _roots() for p in root.rglob(f"*{SUFFIX}.{kind}")}):
                top = p.parent
                if "__pycache__" in p.parts or not (top / "__init__.py").is_file():
                    continue
                while (top.parent / "__init__.py").is_file():
                    top = top.parent
                if top not in imported:
                    imported[top] = top in _locations(top.name)
                if not imported[top]:
                    continue
                stem = p.name[:-len(f"{SUFFIX}.{kind}")]
                parts = [top.name, *p.parent.relative_to(top).parts]
                mod = ".".join(parts if stem == "__init__" else [*parts, stem])
                if not (p.with_name(f"{stem}.py").is_file() or (p.parent / stem / "__init__.py").is_file()):
                    raise FileNotFoundError(f"{p}: a circuit-check sidecar of {mod}, which has no module")
                out.setdefault(mod, {})[kind] = p
    return dict(sorted(out.items()))


def packages() -> set[str]:
    """The top-level packages whose modules may have a sidecar: `PACKAGES`' and each one that has one (`discover`)."""
    return {name.partition(".")[0] for name in (*PACKAGES, *discover())}


def path_of(module: str, kind: str) -> Path | None:
    """Where ``module``'s sidecar of ``kind`` (``py`` or ``json``) is or would be; None for a module outside `packages`."""
    if (have := discover().get(module, {}).get(kind)) is not None:
        return have
    head, _, rest = module.partition(".")
    for root in _locations(head) if head in packages() else []:
        base = root.joinpath(*rest.split(".")) if rest else root
        if base.with_suffix(".py").is_file():
            return base.parent / f"{base.name}{SUFFIX}.{kind}"
        if (base / "__init__.py").is_file():
            return base / f"__init__{SUFFIX}.{kind}"
    return None


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(REPO).as_posix()


_LOADED: dict[str, ModuleType] = {}


def binding_modules() -> list[tuple[str, ModuleType]]:
    """(module, its loaded binding file) for every sidecar that has one, by module name; each loaded once per process, after the
    registries (`targets.load_registries`)."""
    for mod, files in discover().items():
        if "py" in files and mod not in _LOADED:
            name = f"{mod}{SUFFIX}"
            spec = importlib.util.spec_from_file_location(name, files["py"])
            m = importlib.util.module_from_spec(spec)
            sys.modules[name] = m
            spec.loader.exec_module(m)
            _LOADED[mod] = m
    CC.unit_files(str(files["py"]) for files in discover().values() if "py" in files)
    return [(mod, _LOADED[mod]) for mod in discover() if mod in _LOADED]


# --- pins ---------------------------------------------------------------------------------------------------------------------

_SHARED: list[Path] = [SHARED_PINS]
_WHERE: dict[tuple[str, str], Path] = {}


def shared_pin(kind: str, key: str) -> Path:
    """Where pin ``key`` of ``kind`` is, or goes, in ``pins/``."""
    return E.path(_SHARED[0] / kind, key, ".json")


def _shared_files() -> list[Path]:
    """``pins/``'s files; one that is no one pin of its directory's kind, named by its key, is an error."""
    with CC.internal():
        found = sorted(_SHARED[0].rglob("*.json"))
    for p in found:
        kind, d = p.parent.name, _read(p)
        if p.parent.parent != _SHARED[0] or kind not in PIN_KINDS or set(d) != {kind} or len(d[kind]) != 1:
            raise ValueError(f"{p}: a file of {_SHARED[0]} is one pin, {{kind: {{key: counts}}}}, in the directory of its kind")
        E.check(p, next(iter(d[kind])))
    return found


def pin_files() -> list[Path]:
    """``pins/``'s files and every sidecar's pins file."""
    return [*_shared_files(), *(f["json"] for f in discover().values() if "json" in f)]


def _read(p: Path) -> dict:
    with CC.internal():
        return json.loads(p.read_text()) if p.exists() else {}


def load_pins(shared: Path = SHARED_PINS) -> dict:
    """Every pin, by kind and key: ``shared``'s (``pins/``) and each sidecar's.  A pin in two files is an error."""
    _SHARED[0] = Path(shared)
    out: dict[str, dict] = {k: {} for k in PIN_KINDS}
    where: dict[tuple[str, str], Path] = {}
    for p in pin_files():
        for kind, d in _read(p).items():
            for key, v in d.items():
                if (kind, key) in where:
                    raise ValueError(f"the pin {kind} {key} is in {where[kind, key]} and in {p}")
                where[kind, key] = p
                out.setdefault(kind, {})[key] = v
    _WHERE.clear()
    _WHERE.update(where)
    CC.pin_files([str(p) for p in pin_files()], pin_reads)
    return out


def pin_file(kind: str, key: str) -> Path:
    """The file that holds pin ``key`` of ``kind``, or would: the sidecar of the module that owns it (`pin_owner`), else its
    file in ``pins/`` (`shared_pin`)."""
    if (have := _WHERE.get((kind, key))) is not None:
        return have
    mod = pin_owner(kind, key)
    return (path_of(mod, "json") if mod else None) or shared_pin(kind, key)


def pin_reads(kind: str, key: str) -> list[Path]:
    """Where a check that looked up pin ``key`` of ``kind`` read it: the file that holds it; for one no file holds, the file a
    new one goes to (`pin_file`) and its file in ``pins/``, where it could be added by hand."""
    if (have := _WHERE.get((kind, key))) is not None:
        return [have]
    return list(dict.fromkeys([pin_file(kind, key), shared_pin(kind, key)]))


def write_pins(updates: dict[tuple[str, str], Any]) -> None:
    """Set each pin of ``updates`` ({(kind, key): counts}) in its file (`pin_file`), keys sorted."""
    by: dict[Path, dict] = {}
    for (kind, key), v in updates.items():
        by.setdefault(pin_file(kind, key), {}).setdefault(kind, {})[key] = v
    for p, kinds in by.items():
        d = _read(p)
        for kind, vals in kinds.items():
            d[kind] = dict(sorted({**d.get(kind, {}), **vals}.items()))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    discover.cache_clear()
    load_pins(_SHARED[0])


# --- owners -------------------------------------------------------------------------------------------------------------------

def pin_owner(kind: str, key: str) -> str | None:
    """The module that owns a pin: a template's subcircuit's, the module that defines the template (``units``); a Definition's
    or a piece's, the module that defines the Definition its key names (`named`, `owner`).  None when the key alone names none,
    whatever this process has built: then the pin is in ``pins/`` (`shared_pin`)."""
    if kind == "units":
        from verity_flock import subcircuit_templates as TM

        return TM.__name__ if key.partition("/")[0] in TM.TEMPLATES else None
    fid = key.partition("{")[0]
    if (have := HANDED.get(fid)) is not None:
        return have
    d = named(fid)
    return owner(d) if d is not None else None


_NAMED: dict[str, Any] = {}


def composite_lift(fid: str) -> Any | None:
    """`lifted.lift_composite` of the registered composite a ``Lifted[<composite>]_v<k>`` id wraps (the lift a body calling it
    makes, or its hand lift), whatever version the id writes; None when the id wraps no registered composite.  The on-demand
    ``Lifted[...]`` families lift primitives only."""
    from verity.circuits.defs import REGISTRY, CompositeDefinition

    m = re.fullmatch(r"Lifted\[(.+)\]_v\d+", fid)
    inner = REGISTRY.defs.get(m[1]) if m else None
    if not isinstance(inner, CompositeDefinition):
        return None
    from verity.definitions.registry import lifted as L

    return L.lift_composite(inner)


def named(fid: str) -> Any | None:
    """The Definition ``fid`` names, made from the id alone: one a loaded module binds at its top level (`_index`), else a
    registered composite's lift (`composite_lift`, when its id is ``fid``), else one an on-demand family makes from it
    (`verity.circuits.codec.lookup_primitive` with nothing registered, so the family's own factory whoever asked for it first).
    None for one made only inside another Definition's body (`boolean_gather`'s tree levels) or by nothing, built in this process
    or not."""
    from verity.circuits.codec import lookup_primitive
    from verity.circuits.defs import DefinitionTable

    if not _NAMED:
        _NAMED.update({v.id: v for v, _mod in _index().values()})
    if (d := _NAMED.get(fid)) is not None:
        return d
    with CC.internal():
        if (d := composite_lift(fid)) is not None:
            return d if d.id == fid else None
        try:
            return lookup_primitive(DefinitionTable(), fid)
        except (KeyError, ValueError):
            return None


def _top_level_names(path: str) -> set[str]:
    """The names a module's top level binds other than by import."""
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError, ValueError):
        return set()
    out: set[str] = set()

    def targets(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            out.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                targets(e)
        elif isinstance(t, ast.Starred):
            targets(t.value)

    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                targets(t)
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            targets(n.target)
    return out


_INDEX: dict[int, tuple[Any, str]] = {}
#: each Definition's owner by id as a process that has imported every module a full build imports before its checks has it
#: (`units.owners`), taken before this process's own (`owner`, `pin_owner`): set where the catalog comes from the bindings'
#: records, whose bodies no one process builds
HANDED: dict[str, str] = {}


def _index() -> dict[int, tuple[Any, str]]:
    """{id(Definition): (it, module)} for every Definition a loaded module of `packages` binds at its top level other than by
    import; where two do, the one its functions are defined in, else the first by name."""
    from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition

    from . import targets

    if _INDEX:
        return _INDEX
    targets.load_registries()
    tops = packages()
    with CC.internal():
        found: dict[int, tuple[Any, list[str]]] = {}
        for name, m in sorted(sys.modules.items()):
            f = getattr(m, "__file__", None)
            if name.partition(".")[0] not in tops or not f or not f.endswith(".py") or SUFFIX in name:
                continue
            for attr in _top_level_names(f):
                v = vars(m).get(attr)
                if isinstance(v, (CompositeDefinition, PrimitiveDefinition)):
                    found.setdefault(id(v), (v, []))[1].append(name)
    for k, (v, mods) in found.items():
        fm = CC._function_module(v)
        _INDEX[k] = (v, fm if fm in mods else mods[0])
    return _INDEX


def owner(d: Any) -> str | None:
    """The module that defines Definition ``d`` (a specialization's, its Definition's): the module whose top level binds it
    other than by import (`_index`; `HANDED`, where set, by its id), else, for one made on demand, the module of the function
    that builds or evaluates it.  Read once the registries are loaded."""
    d = getattr(d, "defn", d)
    if (have := HANDED.get(getattr(d, "id", None))) is not None:
        return have
    e = _index().get(id(d))
    return e[1] if e is not None and e[0] is d else CC._function_module(d)


__all__ = ["HANDED", "PACKAGES", "PIN_KINDS", "SHARED_PINS", "SUFFIX", "TREES", "binding_modules", "discover", "load_pins", "named",
           "owner", "packages", "path_of", "pin_file", "pin_files", "pin_owner", "pin_reads", "rel", "shared_pin", "write_pins"]
