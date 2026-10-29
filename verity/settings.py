"""The settings machinery every registry in the tree is built on, and ``verity``'s own registry (`REGISTRY`): every
environment variable ``verity`` reads, declared once.

Each distribution declares every environment variable it reads once, in a `Registry` of its own, and reads them only through
it: ``verity``'s is `REGISTRY` here, the vLLM integration's ``integrations/vllm/verity_vllm/settings.py``, and every other
distribution's the ``settings.py`` beside its ``pyproject.toml`` (``tools/config/verity_config/settings.py``, which adds the
variables the tools around a run set, `SHARED` there, and finds a distribution's registry by path).  The machinery is here
because ``verity`` imports no other distribution.  The vLLM integration keeps its own copy of it, because a Build's top-level
process runs without ``verity`` on its path (``integrations/vllm/tests/pipeline/test_cli.py``'s dense-threads test).

A `Setting` has a name, a type (the parser of the variable's string), a default, a role and a doc.  The role says what the
setting can change:

* ``value`` (`VALUE`): what is computed, committed or decided.  `Registry.record` carries every value setting that is set,
  and every variable of a value family (``families``), with a digest, so a run's record says what it ran with.  A credential
  (declared ``secret``, or a name holding one of `SECRET_WORDS`) is recorded by its name alone, `REDACTED` in place of its
  value (`Registry.redact`).
* ``where`` (`WHERE`): only where and how long things run: paths, devices, hosts, run ids, liveness bounds, which tests run.
  Telemetry: its doc says why it changes nothing a run computes.
* ``debug`` (`DEBUG`): diagnostics (dumps, traces, profiles, injected delays), refused on a run of record
  (`Registry.of_record`).

Read a setting as an attribute (``REGISTRY.NAME``, or ``settings.NAME`` where a module forwards to its registry) or by name
(`Registry.get`): its value parsed from the environment at that read, never cached, or its default when the variable is unset
or empty (`Setting.empty` keeps an empty string).  An undeclared name is an AttributeError, and a string that doesn't parse
is a ValueError naming the variable.  A flag (`flag`) reads ``1``/``true``/``yes``/``on`` and ``0``/``false``/``no``/``off``,
in any case, and refuses anything else.  `Registry.put`, `setdefault` and `unset` write a declared variable, for the
processes this one starts and the libraries it loads; `Registry.environ` copies the whole environment, for code that passes
it on.

A registry's prefixes are its own: `Registry.validate_environ` refuses a variable under one that it neither declares nor
names as read elsewhere, naming it, so a misspelt or retired setting fails the run that carries it instead of being ignored.
``verity``'s registry owns ``VERITY_ML_``, and since a library has no startup to check at, every read of it here (`get`,
``settings.NAME``) and its `record` refuse such a variable.  An area adds its prefix once every variable under it is declared.

An area reads its variables through its registry, never ``os.environ`` itself: ``tests/lint/test_repo_env_reads.py`` holds
the direct reads not yet moved to one, by file, to a list that only shrinks.  Stdlib only and nothing from ``verity``, so
every unit may import it, those on the soundness path included (``tests/test_boundaries.py``).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALUE, WHERE, DEBUG = "value", "where", "debug"
ROLES = (VALUE, WHERE, DEBUG)

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


def yes(raw: str) -> bool | None:
    """On when the stripped string is ``1``, ``true`` or ``yes``, off for anything else (``on`` and ``TRUE`` included); blank,
    None (the reader's own default)."""
    v = raw.strip()
    return v in ("1", "true", "yes") if v else None


def not_zero(raw: str) -> bool:
    """On unless the stripped string is ``0`` (an A/B switch that reads ``false`` as on)."""
    return raw.strip() != "0"


def not_off(raw: str) -> bool:
    """On unless the stripped string is ``0``, ``false`` or ``no``."""
    return raw.strip() not in ("0", "false", "no")


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


def lowered(raw: str) -> str:
    return raw.strip().lower()


def number(raw: str) -> float | None:
    """A float; blank, None (the reader's own default)."""
    return float(raw) if raw.strip() else None


def int_or_zero(raw: str) -> int:
    """An integer; blank, 0."""
    return int(raw.strip() or 0)


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


#: A variable whose name holds one of these anywhere is a credential, declared ``secret`` or not (``VLLM_API_KEY``, ``HF_TOKEN``).
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
#: What a record holds in place of a credential's value.
REDACTED = "<redacted>"


class UnknownSetting(ValueError):
    """A variable under a registry's prefixes that nothing declares."""


class DebugOnRecord(ValueError):
    """A ``debug`` setting set on a run of record."""


class Registry:
    """One distribution's settings, read from the process environment (or a mapping given in its place)."""

    def __init__(self, path: str | os.PathLike, prefixes: Iterable[str], *settings: Setting, schema: str,
                 families: Mapping[str, str] | None = None, elsewhere: Iterable[str] = (), standalone: Iterable[str] = (),
                 shared: Registry | None = None) -> None:
        """``path`` is the registry's file, for messages, and ``schema`` names its records (`record`).  ``prefixes`` are the
        ones it owns (`validate_environ`), ``families`` the role of each prefix of other libraries' variables it reads or
        passes on as a whole (``{"VLLM_": VALUE}``), and ``elsewhere`` the owned-prefix names another part of the tree reads (a
        name ending ``_`` a family of them).  ``standalone`` are the distribution's files, relative to the registry, that run
        outside the tree (stdlib only, shipped beside what they load) and so read the environment themselves:
        `standalone_problems` holds each read to its declaration here.  ``shared`` is a registry whose settings this one reads
        too, and may not declare again."""
        self.path = str(path)
        self.schema = schema
        self.prefixes = tuple(prefixes)
        self.families = dict(families or {})
        self.elsewhere = frozenset(elsewhere)
        self.standalone = tuple(standalone)
        common = shared.settings if shared is not None else {}
        own: dict[str, Setting] = {}
        for s in settings:
            if s.name in own or s.name in common:
                where = f", and in {shared.path}" if s.name not in own and shared is not None else ""
                raise ValueError(f"{s.name} is declared twice ({self.path}{where})")
            if s.role not in ROLES:
                raise ValueError(f"{s.name}: role {s.role!r} is not one of {ROLES}")
            if not s.doc.strip():
                raise ValueError(f"{s.name}: a setting says what it does (doc)")
            own[s.name] = s
        for r, why in self.families.items():
            if why not in ROLES:
                raise ValueError(f"family {r}: role {why!r} is not one of {ROLES}")
        self.own = own
        self.settings = {**common, **own}

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
        """A copy of the whole process environment, to pass on: a child process's environment, or an identity that keys on it.
        Read one variable through its declaration instead."""
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

    def is_secret(self, name: str) -> bool:
        """A credential: declared ``secret``, or its name holds one of `SECRET_WORDS`, declared or not."""
        s = self.settings.get(name)
        return bool(s and s.secret) or any(w in name.upper() for w in SECRET_WORDS)

    def redact(self, environ: Mapping[str, str]) -> dict[str, str]:
        """``environ`` as a record holds it: each variable in its order, a credential (`is_secret`) by name, `REDACTED` in place
        of its value.  `record` holds its values so, as must every record that copies a family of the environment."""
        return {k: REDACTED if self.is_secret(k) else v for k, v in environ.items()}

    def record(self, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
        """The value settings in force: every ``value`` variable that is set, as the environment holds it (a credential
        redacted), with the digest: SHA-512 of the canonical JSON of ``values``."""
        env = os.environ if environ is None else environ
        values = self.redact({k: env[k] for k in sorted(env) if self.role_of(k) == VALUE})
        blob = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return {"schema": self.schema, "values": values, "digest": hashlib.sha512(blob).hexdigest()}

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
        """Refuse the variables under this registry's prefixes that nothing in the tree reads, naming each."""
        if bad := self.unknown(environ):
            them = "them" if len(bad) > 1 else "it"
            raise UnknownSetting(f"unknown setting{'s' * (len(bad) > 1)} {', '.join(bad)}: nothing in the tree reads {them}; "
                                 f"unset {them}, fix the name, or declare {them} in {self.path} (as read elsewhere, when "
                                 f"another part of the tree reads {them})")

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


# ---------------------------------------------------------------------------------------------------------------- verity's own

RECORD_SCHEMA = "verity/settings/v1"

REGISTRY = Registry(
    __file__, ("VERITY_ML_",),
    # ---- verity.ml.circuits, its suites included
    Setting("VERITY_ML_BOOLEAN_EXHAUSTIVE", flag, False, WHERE, "the Boolean lowerings' exhaustive tests run, on every f32 word "
            "and every pair of bf16 words (2^32 instances each, minutes): which tests run, never what a circuit computes"),
    schema=RECORD_SCHEMA)


def get(name: str, environ: Mapping[str, str] | None = None) -> Any:
    """A declared setting's value in ``environ`` (default the process environment), which may hold no unknown ``VERITY_ML_*``."""
    REGISTRY.validate_environ(environ)
    return REGISTRY.get(name, environ)


def record(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """`Registry.record` of ``verity``'s value settings, in an environment that holds no unknown ``VERITY_ML_*``."""
    REGISTRY.validate_environ(environ)
    return REGISTRY.record(environ)


def __getattr__(name: str) -> Any:
    if name in REGISTRY.settings:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *REGISTRY.settings])
