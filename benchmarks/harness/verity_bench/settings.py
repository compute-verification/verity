"""Every environment variable verity-bench reads, declared once in a `verity_config.settings.Registry`, the settings
machinery every distribution's registry is built on: what a `Setting` and its roles (`VALUE`, `WHERE`, `DEBUG`) are, how a
read parses and what a record holds are said there.

The harness declares its own run variables (``RESEARCH_RUN_ID`` defaults to ``local`` here, and ``RESEARCH_SOURCE_SHA`` is a
value, which a point's record carries) rather than reading `verity_config.settings.SHARED`'s.  A point writes the record of the
value settings in force (`record`) beside its result (``out/settings.json``).

Read a setting as an attribute, ``settings.VERITY_BENCH_VENV`` (or `get` by name).  `put`, `setdefault` and `unset` write a
declared variable; `environ` is a copy of the whole environment, for code that passes it on (a spec's ``[env]`` and
``"$NAME"`` values name variables the arms read).  A variable the vLLM integration declares (``verity_vllm/settings.py``) is
never declared here too: the harness writes one through the integration's writer, ``verity_vllm.engine.env.apply_target``.

`validate` (`check` of this registry) refuses a ``VERITY_BENCH_*`` variable that nothing reads (the registry's ``elsewhere``
names the ones other parts of the tree read), a declared value that doesn't parse, and on a run of record a debug setting that
is set, each by name: ``verity-bench`` runs it at startup, so a misspelt or retired setting fails the run instead of being
ignored.  ``benchmarks/recursion_e2e/settings.py`` checks its own registry with `check`.

The harness reads and writes ``os.environ`` here and nowhere else, but for the uses ``tests/test_settings.py``'s shrink-only
``BASELINE`` holds (`direct_uses` counts them).
"""
from __future__ import annotations

import ast
import functools
from collections.abc import Mapping
from typing import Any

from verity_config.settings import VALUE, WHERE, Registry, Setting

def check(registry: Registry, environ: Mapping[str, str] | None = None, *, of_record: bool) -> None:
    """Refuse a variable under ``registry``'s prefixes that nothing reads, a declared value that doesn't parse, and, on a run
    of record, a debug setting that is set; each by name."""
    registry.validate_environ(environ)
    for name in registry.settings:
        registry.get(name, environ)
    if of_record:
        registry.of_record(environ)


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
REGISTRY = Registry(__file__, ("VERITY_BENCH_",),
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
    elsewhere=(
        "VERITY_BENCH_FULL",       # benchmarks/numerical/tests/bench/test_instances.py's opt-in
        "VERITY_BENCH_INSTANCES",  # archive/sp1's instance directory
    ), shared=False)
del S

SETTINGS = REGISTRY.settings
setting, raw, is_set, get, put, unset, setdefault = (REGISTRY.setting, REGISTRY.raw, REGISTRY.is_set, REGISTRY.get,
                                                     REGISTRY.put, REGISTRY.unset, REGISTRY.setdefault)
environ, record, unknown = REGISTRY.environ, REGISTRY.record, REGISTRY.unknown


validate = functools.partial(check, REGISTRY)


def __getattr__(name: str) -> Any:
    if name in SETTINGS:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *SETTINGS])
