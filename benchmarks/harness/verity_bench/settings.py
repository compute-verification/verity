"""Every environment variable verity-bench reads, declared once, in the vLLM integration's shape (``verity_vllm/settings.py``).

A `Setting` has a name, a type (the parser of the variable's string), a default, a role and a doc.  The role says what the
setting can change:

* ``value`` (`VALUE`): what is computed or measured.  `Registry.record` carries every value setting in force, with a SHA-512
  digest, and a point writes it beside its result (``out/settings.json``).
* ``where`` (`WHERE`): only where and how long things run: paths, interpreters, run ids.  Its doc says why it can't change a
  result.
* ``debug`` (`DEBUG`): diagnostics.  `Registry.validate` refuses one that is set on a run of record.

Read a setting as an attribute, ``settings.VERITY_BENCH_VENV`` (or `get` by name): its value parsed from the process environment
at that read, or its default when the variable is unset or empty.  An undeclared name is an AttributeError, and a string that
doesn't parse a ValueError naming the variable.  `put`, `setdefault` and `unset` write a declared variable; `environ` is a copy
of the whole environment, for code that passes it on (a spec's ``[env]`` and ``"$NAME"`` values name variables the arms read).
A variable the vLLM integration declares (``verity_vllm/settings.py``) is never declared here too: the harness writes one through
the integration's writer, ``verity_vllm.engine.env.apply_target``.

`validate` refuses a ``VERITY_BENCH_*`` variable that nothing reads (the registry's ``elsewhere`` names the ones other parts of
the tree read), naming it: ``verity-bench`` runs it at startup, so a misspelt or retired setting fails the run instead of being
ignored.

The harness reads and writes ``os.environ`` here and nowhere else, but for the uses ``tests/test_settings.py``'s shrink-only
``BASELINE`` holds (`direct_uses` counts them).  `Registry` is the shape itself, which ``benchmarks/recursion_e2e/settings.py`` declares
its own variables in: stdlib only, and nothing from ``verity_bench``, since the recursion job runs it with the pod's python3.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

VALUE, WHERE, DEBUG = "value", "where", "debug"
ROLES = (VALUE, WHERE, DEBUG)
#: A variable whose name holds one of these is a credential: a record names it and never holds its value.
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
REDACTED = "<redacted>"


@dataclass(frozen=True)
class Setting:
    name: str
    type: Callable[[str], Any]
    default: Any
    role: str
    doc: str


class UnknownSetting(ValueError):
    """A prefixed variable nothing in the tree reads, or a debug setting on a run of record."""


def csv(raw: str) -> tuple[str, ...]:
    """A comma-separated list, in order; empty items are dropped."""
    return tuple(x for x in raw.split(",") if x)


def is_secret(name: str) -> bool:
    return any(w in name.upper() for w in SECRET_WORDS)


class Registry:
    """One distribution's settings: `settings` declared once each, the `prefixes` it owns (a variable under one that is
    neither declared nor in `elsewhere` is refused), and `elsewhere`, the prefixed names other parts of the tree read."""

    def __init__(self, owner: str, schema: str, settings: Iterable[Setting], *, prefixes: tuple[str, ...],
                 elsewhere: Iterable[str] = ()) -> None:
        self.owner, self.schema, self.prefixes = owner, schema, prefixes
        self.settings: dict[str, Setting] = {}
        for s in settings:
            if s.name in self.settings:
                raise ValueError(f"{s.name} is declared twice")
            if s.role not in ROLES:
                raise ValueError(f"{s.name}: role {s.role!r} is not one of {ROLES}")
            self.settings[s.name] = s
        self.elsewhere = frozenset(elsewhere)
        if both := sorted(self.elsewhere & set(self.settings)):
            raise ValueError(f"{', '.join(both)}: declared here and named as read elsewhere")

    def setting(self, name: str) -> Setting:
        s = self.settings.get(name)
        if s is None:
            raise AttributeError(f"{name} is not a declared setting ({self.owner})")
        return s

    def raw(self, name: str, environ: Mapping[str, str] | None = None) -> str | None:
        """A declared setting's string as the environment holds it, None when unset."""
        self.setting(name)
        return (os.environ if environ is None else environ).get(name)

    def is_set(self, name: str, environ: Mapping[str, str] | None = None) -> bool:
        return self.raw(name, environ) is not None

    def get(self, name: str, environ: Mapping[str, str] | None = None) -> Any:
        """A declared setting's value in ``environ`` (default the process environment)."""
        s = self.setting(name)
        v = (os.environ if environ is None else environ).get(name)
        if not v:
            return s.default
        try:
            return s.type(v)
        except ValueError as e:
            raise ValueError(f"{name}={v!r}: {e}") from None

    def put(self, name: str, value: str) -> None:
        """Set a declared variable in this process's environment, which the processes it starts inherit."""
        self.setting(name)
        os.environ[name] = value

    def unset(self, name: str) -> None:
        self.setting(name)
        os.environ.pop(name, None)

    def setdefault(self, name: str, value: str) -> str:
        self.setting(name)
        return os.environ.setdefault(name, value)

    @staticmethod
    def environ() -> dict[str, str]:
        """A copy of the whole process environment, to pass on.  Read one variable through its declaration instead."""
        return dict(os.environ)

    def record(self, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
        """The value settings in force: every declared ``value`` variable that is set, as the environment holds it (a
        credential by name, `REDACTED` in place of its value), with the digest (SHA-512 of the canonical JSON of ``values``)."""
        env = os.environ if environ is None else environ
        values = {k: REDACTED if is_secret(k) else env[k] for k in sorted(env)
                  if k in self.settings and self.settings[k].role == VALUE}
        blob = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return {"schema": self.schema, "values": values, "digest": hashlib.sha512(blob).hexdigest()}

    def known(self, name: str) -> bool:
        return name in self.settings or name in self.elsewhere

    def unknown(self, environ: Iterable[str] | None = None) -> list[str]:
        return sorted(k for k in (os.environ if environ is None else environ)
                      if k.startswith(self.prefixes) and not self.known(k))

    def validate(self, environ: Mapping[str, str] | None = None, *, of_record: bool) -> None:
        """Refuse a prefixed variable nothing reads, a declared value that doesn't parse, and, on a run of record, a debug
        setting that is set; each by name."""
        env = os.environ if environ is None else environ
        if bad := self.unknown(env):
            them = "them" if len(bad) > 1 else "it"
            raise UnknownSetting(f"unknown setting{'s' * (len(bad) > 1)} {', '.join(bad)}: nothing in the tree reads {them}; "
                                 f"unset {them}, fix the name, or declare {them} in {self.owner}")
        for name in self.settings:
            self.get(name, env)
        if of_record and (debug := sorted(k for k, s in self.settings.items() if s.role == DEBUG and env.get(k))):
            raise UnknownSetting(f"{', '.join(debug)} (debug) set on a run of record: unset it, or run it outside a record")


ENV_NAMES = frozenset({"os.environ", "os.environb", "os.getenv", "os.getenvb", "os.putenv", "os.unsetenv"})


def direct_uses(src: str) -> list[tuple[int, str]]:
    """Each use of the process environment in a Python source, through ``os`` or a name imported from it, as (line, name):
    reads and writes alike, which only a registry makes."""
    tree = ast.parse(src)
    alias: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            alias |= {a.asname or "os": "os" for a in n.names if a.name == "os"}
        elif isinstance(n, ast.ImportFrom) and n.module == "os":
            alias |= {a.asname or a.name: f"os.{a.name}" for a in n.names}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and alias.get(n.value.id) == "os" \
                and f"os.{n.attr}" in ENV_NAMES:
            out.append((n.lineno, f"os.{n.attr}"))
        elif isinstance(n, ast.Name) and alias.get(n.id) in ENV_NAMES:
            out.append((n.lineno, alias[n.id]))
    return sorted(out)


S = Setting
REGISTRY = Registry("benchmarks/harness/verity_bench/settings.py", "verity-bench/settings/v1", (
    # ---- the dispatcher and a point's job (jobs.py, run_point.sh, env.sh, setup/)
    S("VERITY_BENCH_VENV", str, None, WHERE, "the node's pinned-vLLM Python a point runs under (env.sh), which the dispatcher "
      "forwards when set: an interpreter, whose vLLM and torch versions the result's identity records"),
    S("VERITY_BENCH_SETUP", str, None, VALUE, "the spec's setup script, which run_point.sh sources before the point's Python "
      "(the dispatcher sets it from the spec; a point refuses any other)"),
    S("VERITY_BENCH_LIB_CACHE", str, None, WHERE, "the node's cache of the libraries setup/vllm_encoded.sh builds, each under its "
      "recipe's and sources' digest; unset, /workspace/research/cache/verity-bench-libs"),
    # ---- research run's (the point's run directory and record)
    S("RESEARCH_RUN_DIR", str, ".", WHERE, "the run directory a point writes result.json in"),
    S("OUT", str, None, WHERE, "the directory a point writes its outputs in; unset, RESEARCH_RUN_DIR/out"),
    S("RESEARCH_RUN_ID", str, "local", WHERE, "the research run a point is, which its result names"),
    S("RESEARCH_SOURCE_SHA", str, None, VALUE, "the commit research run shipped, which a point's identity hashes; unset, the "
      "checkout's HEAD"),
), prefixes=("VERITY_BENCH_",), elsewhere=(
    "VERITY_BENCH_FULL",       # benchmarks/numerical/tests/bench/test_instances.py's opt-in
    "VERITY_BENCH_INSTANCES",  # archive/sp1's instance directory
))
del S

SETTINGS = REGISTRY.settings
setting, raw, is_set, get, put, unset, setdefault = (REGISTRY.setting, REGISTRY.raw, REGISTRY.is_set, REGISTRY.get,
                                                     REGISTRY.put, REGISTRY.unset, REGISTRY.setdefault)
environ, record, unknown, validate = REGISTRY.environ, REGISTRY.record, REGISTRY.unknown, REGISTRY.validate


def __getattr__(name: str) -> Any:
    if name in SETTINGS:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *SETTINGS])
