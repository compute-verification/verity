"""A record that copies the environment by family (``VLLM_*``, ``NCCL_*``, ...) names a credential and never holds its value.

Such a copy is a dict comprehension over the environment that selects names by prefix (``startswith``).  Each one in `verity_vllm`
goes through the registry's rule (`settings.is_secret`): wrapped in `settings.redact`, which names a credential with
`settings.REDACTED` in place of its value, or filtered by ``not settings.is_secret(k)``, which drops it.  The scan finds every copy;
then each copy's own expression runs here, in its module, under dummy credentials."""
from __future__ import annotations

import ast
import builtins
import functools
import importlib
import json
import os
from pathlib import Path

import pytest

from verity_vllm import settings as S

PACKAGE = Path(__file__).resolve().parents[1] / "verity_vllm"
#: Every copy, by its file and the function holding it: the record it fills.
SITES = {
    ("pipeline/tp/commit.py", "main"): "the TP Commit record (versions.json's env)",
    ("pipeline/tp/match.py", "_engine_main"): "the TP Match record (topology.json's env)",
    ("pipeline/tp/rank_worker.py", "_tp_facts"): "each rank's nccl_env, in both TP records' ranks",
    ("pipeline/hot.py", "env_identity"): "the hot worker's key (KEY.json, and what `hot key` prints)",
    ("pipeline/row.py", "effective_config"): "the row's effective_config.json (engine_env), which drops a credential",
}
KEY, TOKEN = "dummy-not-a-real-key", "dummy-not-a-real-token"
ENVIRON = {"VLLM_API_KEY": KEY, "NCCL_DUMMY_TOKEN": TOKEN, "HF_TOKEN": TOKEN, "VLLM_BATCH_INVARIANT": "1", "NCCL_ALGO": "Ring"}


def _environ(node: ast.expr) -> bool:
    """``node`` is the whole environment (``os.environ``, ``settings.environ()``, an ``environ`` handed on), sorted or its items."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sorted" and node.args:
        return _environ(node.args[0])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("items", "keys"):
        return _environ(node.func.value)
    return ast.unparse(node).endswith(("environ", "environ()"))


def _calls(cond: ast.expr, name: str, negated: bool) -> bool:
    """``cond`` calls ``name`` (a method name, or a dotted callee), under ``not`` or not."""
    nots = {id(n.operand) for n in ast.walk(cond) if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)}
    return any(isinstance(n, ast.Call) and name in (ast.unparse(n.func), getattr(n.func, "attr", None)) and (id(n) in nots) == negated
               for n in ast.walk(cond))


@functools.cache
def _copies() -> dict[tuple[str, str], list[tuple[ast.DictComp, ast.expr | None]]]:
    """Each copy by site, with the `settings.redact` call that wraps it (None when none does)."""
    out: dict[tuple[str, str], list[tuple[ast.DictComp, ast.expr | None]]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.DictComp) and any(_environ(g.iter) for g in node.generators)
                    and any(_calls(c, "startswith", negated=False) for g in node.generators for c in g.ifs)):
                continue
            up, fn = parents.get(node), "<module>"
            wrap = up if isinstance(up, ast.Call) and ast.unparse(up.func) == "settings.redact" else None
            while up is not None and fn == "<module>":
                fn = up.name if isinstance(up, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
                up = parents.get(up)
            out.setdefault((path.relative_to(PACKAGE).as_posix(), fn), []).append((node, wrap))
    return out


def _drops(node: ast.DictComp) -> bool:
    return any(_calls(c, "settings.is_secret", negated=True) for g in node.generators for c in g.ifs)


def test_every_copy_of_the_environment_by_family_goes_through_the_credential_rule() -> None:
    copies = _copies()
    bare = sorted(f"{p}:{n.lineno} ({fn})" for (p, fn), ns in copies.items() for n, wrap in ns if wrap is None and not _drops(n))
    assert not bare, (f"{bare} copy the environment by family without the credential rule, so a record could hold a credential's "
                      "value: wrap the copy in settings.redact(...) (the credential named, its value <redacted>), or filter it with "
                      "`not settings.is_secret(k)`")
    assert set(copies) == set(SITES), (f"new copies {sorted(set(copies) - set(SITES))}, gone {sorted(set(SITES) - set(copies))}: "
                                       "name each new one in SITES with the record it fills (its test runs it under dummy "
                                       "credentials) and remove the gone ones")


@pytest.mark.parametrize("site", sorted(SITES), ids=lambda s: f"{s[0]}:{s[1]}")
def test_a_record_names_a_credential_and_never_holds_its_value(site: tuple[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """The site's own expression, in its module, with ``VLLM_API_KEY``, an ``NCCL_*_TOKEN`` and ``HF_TOKEN`` set to dummies: each
    credential it selects is named with `S.REDACTED` (or dropped, where the copy filters credentials out), and no dummy value is
    held; a plain variable it selects is held as set.  A name the expression reads that its module doesn't define is the
    environment it was handed (`row.effective_config`'s ``environ``)."""
    path = site[0]
    module = importlib.import_module("verity_vllm." + path.removesuffix(".py").replace("/", "."))
    for k in list(os.environ):
        monkeypatch.delenv(k)
    for k, v in ENVIRON.items():
        monkeypatch.setenv(k, v)
    for node, wrap in _copies()[site]:
        expr = wrap or node
        names = [n for n in ast.walk(expr) if isinstance(n, ast.Name)]
        free = {n.id for n in names if isinstance(n.ctx, ast.Load)} - {n.id for n in names if isinstance(n.ctx, ast.Store)}
        handed = {n: dict(os.environ) for n in free if n not in vars(module) and not hasattr(builtins, n)}
        got = eval(compile(ast.Expression(expr), path, "eval"), {**vars(module), **handed})  # noqa: S307 (evaluates the record expression the repository's own source builds)
        held = json.dumps(got)
        assert KEY not in held and TOKEN not in held, f"{path}:{node.lineno} holds a credential's value"
        named = {k for k in got if S.is_secret(k)}
        assert all(got[k] == S.REDACTED for k in named) and (bool(named) == (wrap is not None)), (path, node.lineno, sorted(named))
        assert any(got.get(k) == ENVIRON[k] for k in ("VLLM_BATCH_INVARIANT", "NCCL_ALGO")), (path, node.lineno, sorted(got))
