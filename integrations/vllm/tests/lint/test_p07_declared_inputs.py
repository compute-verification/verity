"""P7: results depend only on declared inputs.

Checked today, against ``allowlists/p07_declared_inputs.json``:

* ``environ``: any use of ``os.environ``/``os.environb``/``os.getenv``/``os.putenv``/``os.unsetenv`` (keyed by the
  variable when it is a literal).
* ``cwd``: ``os.getcwd()``, ``os.getcwdb()``, ``Path.cwd()``.
* ``machine-path``: a string literal naming a machine path (``/workspace``, ``/root``, ``/home``, ``/Users``,
  ``/tmp``, ``/mnt``, ``/opt``, ``/usr/local``, ``/data``, ``/scratch``, ``/dev/shm``).
* ``broad-except``: an ``except Exception`` / ``except BaseException`` / bare ``except`` handler that neither
  re-raises nor records (records = uses the bound exception, or calls a logger or ``warnings.warn``).
* ``optional-import``: an ``ImportError``/``ModuleNotFoundError`` handler that does not raise: an optional import that
  fails quietly.
* ``seed-default``: a parameter named ``seed``/``*_seed``/``seed_*`` with a default value.

``SETTINGS_OWNER``, ``verity_vllm/settings.py``, is where the library reads and writes the environment: every variable is declared
there, with the machine path a ``where`` setting defaults to, and every other ``environ`` use is a violation.  ``ENV_OWNERS`` are the two modules that did it before the registry, until
their uses move there: they may read the environment and name machine paths (``pipeline/cli.py`` turns both into declared inputs,
option defaults and ``config.RunEnv``; ``engine/env.py`` writes vLLM's pins), and they may only leave the set.
The rule's torch-stub test (the plan digest is the same with torch present and stubbed out) is not a lint.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache

import pytest

from . import _ratchet as R

NAME = "p07_declared_inputs"
RULE = "P7 (results depend only on declared inputs)"
SETTINGS_OWNER = "verity_vllm/settings.py"
ENV_OWNERS: frozenset[str] = frozenset({"verity_vllm/engine/env.py", "verity_vllm/pipeline/cli.py"})
ENV_NAMES = frozenset({"os.environ", "os.environb", "os.getenv", "os.putenv", "os.unsetenv", "os.getenvb"})
CWD_CALLS = frozenset({"os.getcwd", "os.getcwdb", "pathlib.Path.cwd", "Path.cwd"})
MACHINE_PATH = re.compile(r"^(/workspace|/root|/home|/Users|/tmp|/mnt|/opt|/usr/local|/data|/scratch|/dev/shm)(/|$)")
SEED_NAME = re.compile(r"^(seed|seeds|\w+_seed|seed_\w+)$")
LOGGISH = re.compile(r"(^|\.)(log|logger|logging|_log|_logger|LOG|LOGGER)\.|(^|\.)(warn|warning|exception|error)$|^warnings\.warn$")


def _env_detail(node: ast.AST, parents: dict[int, ast.AST]) -> str:
    """The variable name for ``os.environ["X"]``, ``os.environ.get("X")``, ``os.getenv("X")``, ``"X" in os.environ``,
    else the access form (``os.environ.copy``, ``os.environ``)."""
    parent = parents.get(id(node))
    if isinstance(parent, ast.Subscript) and parent.value is node and R.is_str(parent.slice):
        return parent.slice.value
    if isinstance(parent, ast.Attribute):
        call = parents.get(id(parent))
        if isinstance(call, ast.Call) and call.func is parent and call.args and R.is_str(call.args[0]):
            return call.args[0].value
        return f"os.environ.{parent.attr}"
    if isinstance(parent, ast.Call) and parent.func is node and parent.args and R.is_str(parent.args[0]):
        return parent.args[0].value
    if isinstance(parent, ast.Compare) and R.is_str(parent.left):
        return parent.left.value
    return R.dotted(node) or "os.environ"


def _handler_records(h: ast.ExceptHandler) -> bool:
    for sub in ast.walk(ast.Module(body=h.body, type_ignores=[])):
        if isinstance(sub, ast.Raise):
            return True
        if h.name and isinstance(sub, ast.Name) and sub.id == h.name:
            return True
        if isinstance(sub, ast.Call):
            fn = R.dotted(sub.func) or ""
            if LOGGISH.search(fn) or fn in ("traceback.print_exc", "traceback.format_exc", "sys.exc_info"):
                return True
    return False


def _names(t: ast.AST | None) -> set[str]:
    if t is None:
        return {"<bare>"}
    elts = t.elts if isinstance(t, ast.Tuple) else [t]
    return {R.dotted(e) or "?" for e in elts}


def file_violations(src: str, rel: str) -> list[R.Violation]:
    tree = R.parse_src(src)
    aliases = R.aliases(tree)
    docs = R.docstring_ids(tree)
    out: list[R.Violation] = []
    envy = "environ" in src or "getenv" in src or "putenv" in src
    parents: dict[int, ast.AST] = {}
    for node in (ast.walk(tree) if envy else ()):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    def emit(kind: str, node: ast.AST, symbol: str, detail: str) -> None:
        out.append(R.Violation(rel, kind, symbol, detail, getattr(node, "lineno", 0)))

    for node, symbol in R.walk(tree):
        if envy and isinstance(node, (ast.Name, ast.Attribute)) and rel not in ENV_OWNERS and rel != SETTINGS_OWNER:
            origin = aliases.resolve(node)
            parent = parents.get(id(node))
            if origin in ENV_NAMES and not (isinstance(parent, ast.Attribute) and aliases.resolve(parent) in ENV_NAMES):
                emit("environ", node, symbol, _env_detail(node, parents))
        if isinstance(node, ast.Call):
            origin = aliases.resolve(node.func) or R.dotted(node.func)
            if origin in CWD_CALLS:
                emit("cwd", node, symbol, origin)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs \
                and rel not in ENV_OWNERS and rel != SETTINGS_OWNER:
            m = MACHINE_PATH.match(node.value)
            if m:
                emit("machine-path", node, symbol, m.group(1))
        if isinstance(node, ast.ExceptHandler):
            names = _names(node.type)
            if names & {"Exception", "BaseException", "<bare>"} and not _handler_records(node):
                emit("broad-except", node, symbol, "except " + "|".join(sorted(names)))
            elif names & {"ImportError", "ModuleNotFoundError"} \
                    and not any(isinstance(s, ast.Raise) for s in ast.walk(ast.Module(body=node.body, type_ignores=[]))):
                emit("optional-import", node, symbol, "except " + "|".join(sorted(names)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            pos = a.posonlyargs + a.args
            defaults = dict(zip([x.arg for x in pos][len(pos) - len(a.defaults):], a.defaults))
            defaults.update({x.arg: d for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None})
            fn = getattr(node, "name", "<lambda>")
            inner = fn if symbol == "<module>" else f"{symbol}.{fn}"
            for arg in defaults:
                if SEED_NAME.match(arg):
                    emit("seed-default", node, inner, arg)
    return out


PIN_OWNER = "verity_vllm/engine/env.py"
PIN_PREFIXES = ("VLLM_", "HF_", "TORCHINDUCTOR_", "TOKENIZERS_")


def pin_writes(src: str, rel: str) -> list[str]:
    """Writes of a vLLM / Hugging Face / Inductor switch (literal key) and ``os.environ.update(<x>.env())`` outside
    ``PIN_OWNER``: ``engine/env.py`` is the one writer of vLLM's pins (``pipeline/cli.py`` only reads the environment)."""
    if rel == PIN_OWNER or "environ" not in src:
        return []
    tree = R.parse_src(src)
    aliases = R.aliases(tree)
    out = []
    for node, symbol in R.walk(tree):
        keys: list[ast.AST] = []
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(t, ast.Subscript) and aliases.resolve(t.value) == "os.environ":
                    keys.append(t.slice)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and aliases.resolve(node.func.value) == "os.environ" and node.args:
            if node.func.attr == "setdefault":
                keys.append(node.args[0])
            elif node.func.attr == "update" and isinstance(node.args[0], ast.Call) \
                    and isinstance(node.args[0].func, ast.Attribute) and node.args[0].func.attr == "env":
                out.append(f"{rel}:{node.lineno} {symbol} os.environ.update(....env())")
        for k in keys:
            if R.is_str(k) and k.value.startswith(PIN_PREFIXES):
                out.append(f"{rel}:{node.lineno} {symbol} {k.value}")
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Take configuration as arguments (the CLI reads the environment once), fail loudly on a missing "
                    "optional dependency, re-raise or record broad exceptions, and make seeds required.")


@pytest.mark.slow
def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_only_engine_env_writes_vllm_pins() -> None:
    out = [w for p in R.library_files() for w in pin_writes(R.source(p), R.rel(p))]
    assert not out, f"write vLLM's switches through verity_vllm.engine.env: {out}"
    bad = (
        "import os\n"
        "os.environ.setdefault('VLLM_BATCH_INVARIANT', '1')\n"
        "def build(target):\n"
        "    os.environ['HF_HUB_OFFLINE'] = '1'\n"
        "    os.environ.update(target.env())\n"
        "    os.environ['VERITY_RETAIN'] = 'x'\n"
        "    os.environ.update(saved)\n"
    )
    assert sorted(w.split(" ", 1)[1] for w in pin_writes(bad, "verity_vllm/x.py")) == [
        "<module> VLLM_BATCH_INVARIANT", "build HF_HUB_OFFLINE", "build os.environ.update(....env())"]
    assert pin_writes(bad, "verity_vllm/engine/env.py") == []


def test_env_owners_only_shrink() -> None:
    assert ENV_OWNERS <= {"verity_vllm/engine/env.py", "verity_vllm/pipeline/cli.py"}, \
        f"use the environment through {SETTINGS_OWNER} (verity_vllm.settings), not a new owner: {sorted(ENV_OWNERS)}"


def test_detectors_on_examples() -> None:
    bad = (
        "import os\n"
        "from pathlib import Path\n"
        "try:\n"
        "    import torch\n"
        "except ImportError:\n"
        "    torch = None\n"
        "def run(x, seed=0, *, layout_seed=None):\n"
        "    layout = os.environ.get('VERITY_LAYOUT')\n"
        "    home = os.getenv('HF_HOME', '/workspace/hf')\n"
        "    here = Path.cwd()\n"
        "    try:\n"
        "        return x()\n"
        "    except Exception:\n"
        "        return None\n"
    )
    got = sorted((v.kind, v.detail) for v in file_violations(bad, "verity_vllm/x.py"))
    assert got == [("broad-except", "except Exception"), ("cwd", "pathlib.Path.cwd"), ("environ", "HF_HOME"),
                   ("environ", "VERITY_LAYOUT"), ("machine-path", "/workspace"),
                   ("optional-import", "except ImportError"), ("seed-default", "layout_seed"),
                   ("seed-default", "seed")], got
    good = (
        "import logging\n"
        "log = logging.getLogger(__name__)\n"
        "def run(x, seed):\n"
        "    try:\n"
        "        return x()\n"
        "    except Exception as e:\n"
        "        log.warning('x failed: %s', e)\n"
        "        raise\n"
    )
    assert file_violations(good, "verity_vllm/x.py") == []
