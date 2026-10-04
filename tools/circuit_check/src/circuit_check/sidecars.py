"""Each Definition module's circuit-check bindings and pins, in sidecar files beside it.

    fp8.py                    a Definition module (`verity_vllm.program.registry.fp8`)
    fp8.circuit-check.py      its bindings, in `targets.py`'s forms: binding functions (``_boolean_roots`` and the other
                              names of `targets.ROOT_FUNCTIONS`), `_model_roots` entries and ``TEMPLATE_PARAMS`` entries
    fp8.circuit-check.json    its pins (`checks`' lowering counts by kind: ``definitions``, ``pieces``, ``units``)

A module's bindings bind the Definitions it defines (a template module's, its templates), and its pins are the counts of its
Definitions (`owner`; a template's subcircuits for the template module).  circuit-check finds every sidecar under the packages
that define Definitions or templates (`PACKAGES`) and loads a binding file by its path: its name is no module name, so no
registry enumeration imports it, and a Definition module imports nothing of circuit-check.  A sidecar whose module is gone is
an error.  `targets.py` keeps what no one module owns, and `pins.json` the pins of Definitions no module defines.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

from . import cache as CC

REPO = Path(__file__).resolve().parents[4]
#: the packages whose modules may have a sidecar: the ones that define Definitions (`targets._authored`) and the templates'
PACKAGES = ("verity", "verity_vllm", "verity_pouw", "verity_numerical")
SUFFIX = ".circuit-check"
#: the pins of a Definition no module defines
SHARED_PINS = Path(__file__).with_name("pins.json")
PIN_KINDS = ("definitions", "pieces", "units")


def _roots() -> list[tuple[str, Path]]:
    out = []
    for pkg in PACKAGES:
        spec = importlib.util.find_spec(pkg)
        out += [(pkg, Path(p).resolve()) for p in (spec.submodule_search_locations or [])] if spec else []
    return out


@cache
def discover() -> dict[str, dict[str, Path]]:
    """{module: {"py": its binding file, "json": its pins file}} for every sidecar under `PACKAGES`, by module name."""
    out: dict[str, dict[str, Path]] = {}
    with CC.internal():
        for pkg, root in _roots():
            for kind in ("py", "json"):
                for p in sorted(root.rglob(f"*{SUFFIX}.{kind}")):
                    if "__pycache__" in p.parts:
                        continue
                    stem = p.name[:-len(f"{SUFFIX}.{kind}")]
                    parts = [pkg, *p.relative_to(root).parent.parts]
                    mod = ".".join(parts if stem == "__init__" else [*parts, stem])
                    if not (p.with_name(f"{stem}.py").is_file() or (p.parent / stem / "__init__.py").is_file()):
                        raise FileNotFoundError(f"{p}: a circuit-check sidecar of {mod}, which has no module")
                    out.setdefault(mod, {})[kind] = p
    return dict(sorted(out.items()))


def path_of(module: str, kind: str) -> Path | None:
    """Where ``module``'s sidecar of ``kind`` (``py`` or ``json``) is or would be; None for a module outside `PACKAGES`."""
    if (have := discover().get(module, {}).get(kind)) is not None:
        return have
    head, _, rest = module.partition(".")
    for pkg, root in _roots():
        if pkg != head:
            continue
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


def pin_files() -> list[Path]:
    """`pins.json` and every sidecar's pins file."""
    return [_SHARED[0], *(f["json"] for f in discover().values() if "json" in f)]


def _read(p: Path) -> dict:
    with CC.internal():
        return json.loads(p.read_text()) if p.exists() else {}


def load_pins(shared: Path = SHARED_PINS) -> dict:
    """Every pin, by kind and key: ``shared``'s (`pins.json`) and each sidecar's.  A pin in two files is an error."""
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
    CC.pin_files([str(p) for p in pin_files()], pin_file)
    return out


def pin_file(kind: str, key: str) -> Path:
    """The file that holds pin ``key`` of ``kind``, or would: the sidecar of the module that owns it (`pin_owner`), else
    `pins.json`."""
    if (have := _WHERE.get((kind, key))) is not None:
        return have
    mod = pin_owner(kind, key)
    return (path_of(mod, "json") if mod else None) or _SHARED[0]


def write_pins(updates: dict[tuple[str, str], Any]) -> None:
    """Set each pin of ``updates`` ({(kind, key): counts}) in its file (`pin_file`), keys sorted."""
    by: dict[Path, dict] = {}
    for (kind, key), v in updates.items():
        by.setdefault(pin_file(kind, key), {}).setdefault(kind, {})[key] = v
    for p, kinds in by.items():
        d = _read(p)
        for kind, vals in kinds.items():
            d[kind] = dict(sorted({**d.get(kind, {}), **vals}.items()))
        p.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    discover.cache_clear()
    load_pins(_SHARED[0])


# --- owners -------------------------------------------------------------------------------------------------------------------

def pin_owner(kind: str, key: str) -> str | None:
    """The module that owns a pin: a template's subcircuit's, the module that defines the template (``units``); a Definition's
    or a piece's, the module that defines that Definition (`owner`).  None when nothing registered names it."""
    if kind == "units":
        from verity_numerical.bench import templates as TM

        return TM.__name__ if key.partition("/")[0] in TM.TEMPLATES else None
    from verity.ir.defs import REGISTRY

    d = REGISTRY.defs.get(key.partition("{")[0])
    return owner(d) if d is not None else None


def _function_module(d: Any) -> str | None:
    fields = getattr(d, "__dict__", {})
    for a in ("build", "body", "evaluate", "fn"):
        if callable(f := fields.get(a)) and getattr(f, "__module__", None):
            return f.__module__
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


def _index() -> dict[int, tuple[Any, str]]:
    """{id(Definition): (it, module)} for every Definition a loaded module of `PACKAGES` binds at its top level other than by
    import; where two do, the one its functions are defined in, else the first by name."""
    from verity.ir.defs import CompositeDefinition, PrimitiveDefinition

    from . import targets

    if _INDEX:
        return _INDEX
    targets.load_registries()
    with CC.internal():
        found: dict[int, tuple[Any, list[str]]] = {}
        for name, m in sorted(sys.modules.items()):
            f = getattr(m, "__file__", None)
            if name.partition(".")[0] not in PACKAGES or not f or not f.endswith(".py") or SUFFIX in name:
                continue
            for attr in _top_level_names(f):
                v = vars(m).get(attr)
                if isinstance(v, (CompositeDefinition, PrimitiveDefinition)):
                    found.setdefault(id(v), (v, []))[1].append(name)
    for k, (v, mods) in found.items():
        fm = _function_module(v)
        _INDEX[k] = (v, fm if fm in mods else mods[0])
    return _INDEX


def owner(d: Any) -> str | None:
    """The module that defines Definition ``d`` (a specialization's, its Definition's): the module whose top level binds it
    other than by import (`_index`), else, for one made on demand, the module of the function that builds or evaluates it.
    Read once the registries are loaded."""
    d = getattr(d, "defn", d)
    e = _index().get(id(d))
    return e[1] if e is not None and e[0] is d else _function_module(d)


__all__ = ["PACKAGES", "PIN_KINDS", "SHARED_PINS", "SUFFIX", "binding_modules", "discover", "load_pins", "owner", "path_of",
           "pin_file", "pin_files", "pin_owner", "rel", "write_pins"]
