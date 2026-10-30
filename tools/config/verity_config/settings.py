"""Settings registries: each distribution declares every environment variable it reads once, in its own registry, and reads
them only through it.  The vLLM integration's ``integrations/vllm/verity_vllm/settings.py`` is the template, and that
integration's registry; this is the same registry as a class, for every other distribution.

A distribution's registry is the ``settings.py`` beside its ``pyproject.toml``, which declares its variables and the prefixes
it owns::

    from verity_config.settings import VALUE, WHERE, Registry, Setting as S
    REGISTRY = Registry(__file__, ("PEARLC_",),
        S("PEARLC_SCHEDULE", str, "deferred", VALUE, "when the Pearl-C arm's hashes run: deferred (served) or serial"),
        ...)

and its code reads them by path, so two distributions' registries never shadow each other on ``sys.path`` (a ``research run``
puts every flat workspace member's directory there, so each ``settings.py`` would be a top-level ``settings``)::

    from verity_config.settings import of
    S = of(__file__)        # the registry of the distribution this file is in
    S.PEARLC_SCHEDULE       # parsed from the process environment at this read

A `Setting` has a name, a type (the parser of the variable's string), a default, a role and a doc.  The role says what it can
change:

* ``value`` (`VALUE`): what is computed or committed.  `Registry.record` carries every value setting that is set, with a
  SHA-512 digest, so it can go in the run's record; a credential's value never does (`redact`).
* ``where`` (`WHERE`): only where and how long things run: paths, devices, hosts, run ids and liveness bounds.  Its doc
  says why it changes nothing computed.
* ``debug`` (`DEBUG`): diagnostics that change nothing a run computes: dumps, traces, profiles, A/B probes of timing.  A run
  of record refuses them (`Registry.of_record`).

Read a setting as an attribute (``S.PEARLC_SCHEDULE``) or by name (`Registry.get`): its value is parsed at each read, never
cached, and is its default when the variable is unset or empty (`Setting.empty` keeps an empty string).  An undeclared name
is an AttributeError, a string that doesn't parse a ValueError naming the variable.  A flag (`flag`) reads
``1``/``true``/``yes``/``on`` and ``0``/``false``/``no``/``off`` in any case and refuses anything else; `one` is on exactly
at ``1``.  `Registry.put`, `setdefault` and `unset` write a declared variable for the processes this one starts, and
`environ` copies the whole environment for code that passes it on.

`SHARED` declares, once for every registry, the variables the tools around a run set: research's runner, gpu-lease, CUDA,
Hugging Face and Python.  Every `Registry` reads them too, and a distribution with no variables of its own reads them from
`SHARED` directly, with no ``settings.py``.

A registry's prefixes are its own: `Registry.validate_environ` refuses a variable with one of them that it neither declares
nor lists as read elsewhere, naming it, and `of` runs it when it first loads a registry, so a misspelt or retired knob fails
the run instead of being ignored.

A ship (files a build copies out of the tree to run on another machine) takes its distribution's ``settings.py`` with it,
and `of` finds it beside them.  A file that can't import the registry at all (stdlib only, or run under another tree's
PYTHONPATH) reads the environment itself; the registry still declares each variable it reads and names the file
(``standalone``), and `Registry.standalone_problems`, which the distribution's suite runs, holds each read to its
declaration.

Stdlib only.  Outside the vLLM integration the environment is read in the registries (a module named ``settings.py``) and in
the files ``tests/lint/allowlists/repo_env_reads.json`` still lists, which only shrinks.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALUE, WHERE, DEBUG = "value", "where", "debug"
ROLES = (VALUE, WHERE, DEBUG)
RECORD_SCHEMA = "verity-config/settings/v1"
#: a distribution's registry, beside its pyproject.toml
REGISTRY_FILE = "settings.py"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def flag(raw: str) -> bool:
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"not a flag: one of {', '.join(_TRUE)} or {', '.join(_FALSE)}")


def one(raw: str) -> bool:
    """On exactly when the string is ``1``."""
    return raw == "1"


def not_zero(raw: str) -> bool:
    """On unless the stripped string is ``0``."""
    return raw.strip() != "0"


def present(raw: str) -> bool:
    """On when set to anything non-empty, ``0`` included."""
    return True


def csv(raw: str) -> tuple[str, ...]:
    """A comma-separated list, in order; empty items are dropped."""
    return tuple(x for x in raw.split(",") if x)


def ints(raw: str) -> tuple[int, ...]:
    """Comma-separated integers, in order (``1,7,128``)."""
    return tuple(int(x) for x in raw.split(","))


def paths(raw: str) -> tuple[str, ...]:
    """A list of paths separated by ``os.pathsep``, as ``PATH`` is, in order; empty items are dropped."""
    return tuple(x for x in raw.split(os.pathsep) if x)


def words(raw: str) -> tuple[str, ...]:
    """A whitespace-separated list, in order (``SHAPES="8192 1024x1024x3072"``)."""
    return tuple(raw.split())


def stripped(raw: str) -> str:
    return raw.strip()


def choice(*options: str) -> Callable[[str], str]:
    """The string, which must be one of ``options``."""
    def parse(raw: str) -> str:
        if raw not in options:
            raise ValueError(f"one of {', '.join(options)}")
        return raw
    parse.__name__ = f"choice({', '.join(options)})"
    return parse


@dataclass(frozen=True)
class Setting:
    name: str
    type: Callable[[str], Any]
    default: Any
    role: str
    doc: str
    #: an empty string is a value (``CUDA_VISIBLE_DEVICES=""`` hides every device), not unset
    empty: bool = False
    #: a credential, whatever its name says: `Registry.record` names it and never holds its value
    secret: bool = False


#: A variable whose name holds one of these anywhere is a credential, declared or not (``HF_TOKEN``).
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
#: What a record holds in place of a credential's value.
REDACTED = "<redacted>"


def is_secret(name: str, setting: Setting | None = None) -> bool:
    return bool(setting and setting.secret) or any(w in name.upper() for w in SECRET_WORDS)


class UnknownSetting(ValueError):
    """A variable with a registry's prefix that nothing declares."""


class DebugOnRecord(ValueError):
    """A debug setting set on a run of record."""


class Registry:
    """One distribution's settings, read from the process environment (or a mapping given in its place)."""

    def __init__(self, path: str | os.PathLike, prefixes: Iterable[str], *settings: Setting,
                 families: Mapping[str, str] | None = None, elsewhere: Iterable[str] = (), standalone: Iterable[str] = (),
                 shared: bool = True) -> None:
        """``path`` is the registry's file, for messages.  ``prefixes`` are the ones it owns (`validate_environ`), ``families``
        the role of each prefix of other libraries' variables it reads or passes on as a whole (``{"VLLM_": VALUE}``), and
        ``elsewhere`` the owned-prefix names another distribution reads (a name ending ``_`` a family of them).
        ``standalone`` are the distribution's files, relative to the registry, that run outside the tree (stdlib only, shipped
        beside what they load) and so read the environment themselves: `standalone_problems` holds each read to its
        declaration here."""
        self.path = str(path)
        self.prefixes = tuple(prefixes)
        self.families = dict(families or {})
        self.elsewhere = frozenset(elsewhere)
        self.standalone = tuple(standalone)
        own: dict[str, Setting] = {}
        for s in settings:
            if s.name in own or (shared and s.name in SHARED.settings):
                raise ValueError(f"{s.name} is declared twice ({self.path}{', and in SHARED' if s.name not in own else ''})")
            if s.role not in ROLES:
                raise ValueError(f"{s.name}: role {s.role!r} is not one of {ROLES}")
            if not s.doc.strip():
                raise ValueError(f"{s.name}: a setting says what it does (doc)")
            own[s.name] = s
        for r, why in self.families.items():
            if why not in ROLES:
                raise ValueError(f"family {r}: role {why!r} is not one of {ROLES}")
        self.own = own
        self.settings = {**SHARED.settings, **own} if shared else own

    def __repr__(self) -> str:
        return f"Registry({self.path!r}, {len(self.own)} settings)"

    def setting(self, name: str) -> Setting:
        s = self.settings.get(name)
        if s is None:
            raise AttributeError(f"{name} is not a declared setting ({self.path})")
        return s

    def raw(self, name: str, environ: Mapping[str, str] | None = None) -> str | None:
        """A declared setting's string as the environment holds it, None when unset."""
        self.setting(name)
        return (os.environ if environ is None else environ).get(name)

    def is_set(self, name: str, environ: Mapping[str, str] | None = None) -> bool:
        """The variable is set to a value: anything but the empty string, unless the setting keeps it (`Setting.empty`)."""
        v = self.raw(name, environ)
        return v is not None and (v != "" or self.settings[name].empty)

    def get(self, name: str, environ: Mapping[str, str] | None = None) -> Any:
        """A declared setting's value in ``environ`` (default the process environment)."""
        s = self.setting(name)
        v = (os.environ if environ is None else environ).get(name)
        if v is None or (v == "" and not s.empty):
            return s.default
        try:
            return s.type(v)
        except ValueError as e:
            raise ValueError(f"{name}={v!r}: {e} ({self.path})") from None

    def need(self, name: str, environ: Mapping[str, str] | None = None) -> Any:
        """`get` for a setting the caller can't go without (``os.environ[NAME]``): unset, it raises KeyError naming it."""
        if not self.is_set(name, environ):
            raise KeyError(f"{name} is not set: {self.settings[name].doc} ({self.path})")
        return self.get(name, environ)

    def put(self, name: str, value: str) -> None:
        """Set a declared variable in this process's environment, which the processes it starts inherit."""
        self.setting(name)
        os.environ[name] = value

    def unset(self, name: str) -> None:
        self.setting(name)
        os.environ.pop(name, None)

    def setdefault(self, name: str, value: str) -> str:
        """Set a declared variable unless the environment holds it (empty included), as ``os.environ.setdefault``; its string."""
        self.setting(name)
        return os.environ.setdefault(name, value)

    @staticmethod
    def environ() -> dict[str, str]:
        """A copy of the whole process environment, to pass on to a child process.  Read one variable through its declaration."""
        return dict(os.environ)

    def __getattr__(self, name: str) -> Any:
        if name.isupper() and name in self.__dict__.get("settings", {}):
            return self.get(name)
        raise AttributeError(f"{type(self).__name__} {self.__dict__.get('path', '?')!r} has no attribute {name!r}")

    def __dir__(self) -> list[str]:
        return sorted([*super().__dir__(), *self.settings])

    # ------------------------------------------------------------------------------------------------------------ the record

    def role_of(self, name: str) -> str | None:
        """A variable's role: its declaration's, else its family's; None for a variable the registry doesn't know."""
        if name in self.settings:
            return self.settings[name].role
        return next((r for p, r in self.families.items() if name.startswith(p)), None)

    def redact(self, environ: Mapping[str, str]) -> dict[str, str]:
        """``environ`` as a record holds it: a credential (`is_secret`) by name, `REDACTED` in place of its value."""
        return {k: REDACTED if is_secret(k, self.settings.get(k)) else v for k, v in environ.items()}

    def record(self, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
        """The value settings in force: every ``value`` variable that is set, as the environment holds it (a credential
        redacted), with the digest: SHA-512 of the canonical JSON of ``values``."""
        env = os.environ if environ is None else environ
        values = self.redact({k: env[k] for k in sorted(env) if self.role_of(k) == VALUE})
        blob = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return {"schema": RECORD_SCHEMA, "values": values, "digest": hashlib.sha512(blob).hexdigest()}

    def debug_set(self, environ: Mapping[str, str] | None = None) -> list[str]:
        """The debug settings that are set."""
        return sorted(n for n, s in self.settings.items() if s.role == DEBUG and self.is_set(n, environ))

    def of_record(self, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
        """`record`, for a run of record, which refuses every debug setting that is set, naming each."""
        if bad := self.debug_set(environ):
            raise DebugOnRecord(f"{', '.join(bad)}: debug setting{'s' * (len(bad) > 1)}, refused on a run of record; unset "
                                f"{'them' if len(bad) > 1 else 'it'} ({self.path})")
        return self.record(environ)

    # ------------------------------------------------------------------------------------------------------------ unknown names

    def known(self, name: str) -> bool:
        return (name in self.settings or name in self.elsewhere
                or any(name.startswith(p) for p in self.elsewhere if p.endswith("_")))

    def unknown(self, environ: Iterable[str] | None = None) -> list[str]:
        return sorted(k for k in (os.environ if environ is None else environ)
                      if k.startswith(self.prefixes) and not self.known(k))

    def validate_environ(self, environ: Iterable[str] | None = None) -> None:
        """Refuse the variables with this registry's prefixes that nothing reads, naming each."""
        if bad := self.unknown(environ):
            them = "them" if len(bad) > 1 else "it"
            raise UnknownSetting(f"unknown setting{'s' * (len(bad) > 1)} {', '.join(bad)}: nothing reads {them}; unset {them}, "
                                 f"fix the name, or declare {them} in {self.path}")

    # ------------------------------------------------------------------------------------------------------------ standalone

    def standalone_problems(self) -> list[str]:
        """Each read in a `standalone` file that its declaration here doesn't cover: a name not declared, or a literal default
        (``os.environ.get("X", "2")``) that doesn't parse to the declared one.  A setting declared with no default (None)
        takes each reader's own."""
        out = []
        for rel in self.standalone:
            f = Path(self.path).parent / rel
            for line, name, raw in direct_reads(f.read_text(encoding="utf-8")):
                s = self.settings.get(name)
                if s is None:
                    out.append(f"{rel}:{line}: {name} is not declared in {self.path}")
                elif raw is not None and s.default is not None:
                    try:
                        parsed = s.type(raw)
                    except ValueError as e:
                        out.append(f"{rel}:{line}: {name}'s default {raw!r} here doesn't parse as declared: {e}")
                        continue
                    if parsed != s.default:
                        out.append(f"{rel}:{line}: {name}'s default {raw!r} here is not its declared {s.default!r}")
        return out


def direct_reads(source: str) -> list[tuple[int, str, str | None]]:
    """(line, name, literal default or None) of each read of a literal name in ``source``: ``os.environ.get``,
    ``os.getenv``, ``os.environ[...]`` and ``in os.environ``."""
    def environ(n: ast.AST) -> bool:
        return isinstance(n, ast.Attribute) and n.attr == "environ" and isinstance(n.value, ast.Name) and n.value.id == "os"

    def lit(n: ast.AST | None) -> str | None:
        return n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None

    out = []
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args and (
                n.func.attr == "get" and environ(n.func.value)
                or n.func.attr == "getenv" and isinstance(n.func.value, ast.Name) and n.func.value.id == "os"):
            if (name := lit(n.args[0])) is not None:
                out.append((n.lineno, name, lit(n.args[1]) if len(n.args) > 1 else None))
        elif isinstance(n, ast.Subscript) and environ(n.value) and isinstance(n.ctx, ast.Load):
            if (name := lit(n.slice)) is not None:
                out.append((n.lineno, name, None))
        elif isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops) \
                and any(environ(c) for c in n.comparators):
            if (name := lit(n.left)) is not None:
                out.append((n.lineno, name, None))
    return sorted(out)


SHARED = Registry(__file__, (), *(Setting(*a) for a in (
    ("RESEARCH_RUN_ID", str, None, WHERE, "the research run's id (research's runner sets it): it names a run's outputs"),
    ("RESEARCH_RUN_DIR", str, None, WHERE, "the research run's directory (research's runner sets it), where its result.json "
                                           "goes"),
    ("RESEARCH_SOURCE_SHA", str, None, WHERE, "the commit a research run's source tree is (research's runner sets it): "
                                              "provenance, which the run records"),
    ("RESEARCH_SOURCE_COMMIT", str, None, WHERE, "the commit a caller names for a run's provenance (research's runner sets "
                                                 "RESEARCH_SOURCE_SHA instead)"),
    ("RESEARCH_LANE", str, None, WHERE, "the lane a run is for (research's runner sets it)"),
    ("RESEARCH", str, "research", WHERE, "the research command a script runs"),
    ("OUT", str, None, WHERE, "the run's output directory (research's runner sets it: the writing-runs skill)"),
    ("GPU_LEASE_UUID", str, None, WHERE, "the UUID of the GPU gpu-lease leased the process (gpu-lease sets it)"),
    ("GPU_LEASE_WHO", str, None, WHERE, "who holds the process's GPU lease (gpu-lease sets it)"),
    ("CUDA_VISIBLE_DEVICES", str, None, WHERE, "the GPUs CUDA shows the process (gpu-lease sets it)", True),
    ("CUDA_DEVICE_ORDER", str, None, WHERE, "how CUDA numbers the GPUs (PCI_BUS_ID: as nvidia-smi does)"),
    ("CUDA_HOME", str, None, WHERE, "the CUDA toolkit's directory"),
    ("NVCC", str, None, WHERE, "the nvcc binary a build runs"),
    ("HF_HOME", str, None, WHERE, "Hugging Face's cache directory, where weights are read from"),
    ("PYTHONPATH", paths, (), WHERE, "Python's import roots (research's runner sets them from the workspace)"),
    ("OMP_NUM_THREADS", int, None, WHERE, "the threads of a CPU reference's BLAS: how fast it runs, not what it computes"),
    ("FILL_JOB", str, None, WHERE, "the fill job a process runs as (research's fill runner sets it): it names a run's outputs"),
    ("DITHER_NUMA_RECORD", json.loads, None, WHERE, "the NUMA binding benchmarks/hash/dither/numabind.py started the process "
                                                    "under, as JSON (it sets it): where the process ran, which a run records"),
)), shared=False)


def _loaded(root: Path) -> Registry:
    name = f"verity_config.settings:{root}"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, root / REGISTRY_FILE)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
            reg = getattr(mod, "REGISTRY", None)
            if not isinstance(reg, Registry):
                raise TypeError(f"{root / REGISTRY_FILE}: REGISTRY is the distribution's Registry")
            reg.validate_environ()
        except BaseException:
            del sys.modules[name]
            raise
    return mod.REGISTRY


def of(path: str | os.PathLike) -> Registry:
    """The registry of the distribution ``path`` is in: the ``settings.py`` beside the nearest ``pyproject.toml`` above it,
    or, in a ship (files copied out of the tree with their registry, no ``pyproject.toml`` above them), the ``settings.py``
    beside ``path``.  Loaded once a process (by its path, never by its module name) and checked then
    (`Registry.validate_environ`)."""
    here = Path(path).resolve()
    for root in here.parents:
        if (root / "pyproject.toml").is_file():
            if not (root / REGISTRY_FILE).is_file():
                raise LookupError(f"{here}: its distribution ({root}) has no settings registry: declare its variables in "
                                  f"{root / REGISTRY_FILE} (verity_config.settings), or read the shared ones from SHARED")
            return _loaded(root)
    if (here.parent / REGISTRY_FILE).is_file():
        return _loaded(here.parent)
    raise LookupError(f"{here}: no pyproject.toml above it and no {REGISTRY_FILE} beside it, so no settings registry: ship "
                      f"its distribution's {REGISTRY_FILE} with it")
