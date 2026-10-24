"""P14: one way to build a vLLM engine, and one module that knows vLLM's object graph.

Checked against ``allowlists/p14_one_way.json``, keyed ``(file, kind, symbol, detail)``:

* ``vllm-LLM``: a call of ``vllm.LLM`` (however it was imported) outside `BUILD`, the one place that builds a vLLM
  engine; detail ``vllm.LLM``.
* ``vllm-internals``: a read or write of vLLM's engine internals, the attributes in `INTERNALS` (as ``x.llm_engine`` or as
  the name a ``getattr``/``hasattr``/``setattr``/``delattr`` call is given), outside `HANDLE`, the one module that walks
  from an ``LLM`` to its scheduler, model runner, config and workers; detail the attribute.
"""
from __future__ import annotations

import ast
from functools import lru_cache

from . import _ratchet as R

NAME = "p14_one_way"
RULE = "P14 (one way to build and reach the vLLM engine)"
BUILD = "verity_vllm/engine/build.py"
HANDLE = "verity_vllm/engine/handle.py"
VLLM_LLM = frozenset({"vllm.LLM", "vllm.entrypoints.llm.LLM"})
INTERNALS = frozenset({"llm_engine", "engine_core", "model_executor"})
_ATTR_FUNCS = frozenset({"getattr", "hasattr", "setattr", "delattr"})


def file_violations(src: str, rel: str) -> list[R.Violation]:
    tree = R.parse_src(src)
    names = R.aliases(tree)
    out: list[R.Violation] = []
    for node, symbol in R.walk(tree):
        if rel != BUILD and isinstance(node, ast.Call) and names.resolve(node.func) in VLLM_LLM:
            out.append(R.Violation(rel, "vllm-LLM", symbol, "vllm.LLM", node.lineno))
        if rel == HANDLE:
            continue
        if isinstance(node, ast.Attribute) and node.attr in INTERNALS:
            out.append(R.Violation(rel, "vllm-internals", symbol, node.attr, node.lineno))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ATTR_FUNCS \
                and len(node.args) >= 2 and R.is_str(node.args[1]) and node.args[1].value in INTERNALS:
            out.append(R.Violation(rel, "vllm-internals", symbol, node.args[1].value, node.lineno))
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    f"Build a vLLM engine only in {BUILD} (`build_engine`, or `verity_vllm.LLM` on top of it), and reach "
                    f"its scheduler, model runner, config or workers only through {HANDLE}'s accessors: add the accessor "
                    "there if it is missing, rather than walking `llm_engine`/`engine_core`/`model_executor` here.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    bad = (
        "import vllm\n"
        "from vllm import LLM\n"
        "def build(kw):\n"
        "    a = LLM(**kw)\n"
        "    b = vllm.LLM(model='m')\n"
        "    core = getattr(a.llm_engine, 'engine_core', None)\n"
        "    return hasattr(core, 'model_executor'), getattr(b, 'other', None), {'engine_core': 1}\n"
    )
    got = sorted((v.kind, v.symbol, v.detail) for v in file_violations(bad, "verity_vllm/pipeline/x.py"))
    assert got == [("vllm-LLM", "build", "vllm.LLM"), ("vllm-LLM", "build", "vllm.LLM"),
                   ("vllm-internals", "build", "engine_core"), ("vllm-internals", "build", "llm_engine"),
                   ("vllm-internals", "build", "model_executor")], got
    assert [v.kind for v in file_violations(bad, BUILD)] == ["vllm-internals"] * 3
    assert [v.kind for v in file_violations(bad, HANDLE)] == ["vllm-LLM"] * 2
    good = "from verity_vllm.llm import LLM\ndef f(m):\n    return LLM(m).generate(['x'])\n"
    assert file_violations(good, "verity_vllm/pipeline/x.py") == []
