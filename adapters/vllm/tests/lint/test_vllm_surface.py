"""vLLM's surface: the closed list (`vllm_surface.py`) of vLLM's modules and names that the library code imports."""
from __future__ import annotations

import ast
import importlib
import warnings

import pytest

from verity_vllm.engine import stock

from . import _ratchet as R
from .vllm_surface import SURFACE


def imported(text: str) -> set[str]:
    """The vLLM modules a source imports whole, and the ``module:name`` pairs it imports from one."""
    out: set[str] = set()
    for n in ast.walk(R.parse_src(text)):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names if a.name.split(".")[0] == "vllm"}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and n.module.split(".")[0] == "vllm":
            out |= {f"{n.module}:{a.name}" for a in n.names}
    return out


def test_it_reads_both_kinds_of_import_at_any_depth():
    text = "import vllm.envs as e\nimport vllmx\nfrom . import vllm\ndef f():\n    from vllm.config import a, b as c\n"
    assert imported(text) == {"vllm.envs", "vllm.config:a", "vllm.config:b"}


def test_every_vllm_import_of_the_library_is_listed():
    uses: dict[str, set[str]] = {}
    for p in R.library_files(root=R.PKG):
        for key in imported(R.source(p)):
            uses.setdefault(key, set()).add(R.rel(p))
    assert len(uses) > 50
    missing = {k: sorted(v) for k, v in uses.items() if k not in SURFACE}
    assert not missing, ("vLLM imports that tests/lint/vllm_surface.py doesn't list: add each with why the integration "
                         "needs it, or reach it through a module that already imports it:\n"
                         + "\n".join(f"  {k}: {', '.join(v)}" for k, v in sorted(missing.items())))
    stale = sorted(set(SURFACE) - set(uses))
    if stale:
        warnings.warn(f"entries of tests/lint/vllm_surface.py that nothing imports: delete each once no change in flight "
                      f"adds its import: {stale}", stacklevel=1)


def test_every_entry_resolves_at_the_installed_vllm():
    vllm = pytest.importorskip("vllm")
    assert stock.build() in stock.BUILDS, (f"vLLM {vllm.__version__} ({stock.build()}) is installed, no build of "
                                           f"engine.stock.BUILDS: {', '.join(stock.BUILDS)}")
    unresolved, unchecked = [], []
    for key in SURFACE:
        module, _, name = key.partition(":")
        try:
            m = importlib.import_module(module)
            if name and not hasattr(m, name):
                importlib.import_module(f"{module}.{name}")
        except ModuleNotFoundError as e:
            (unresolved if (e.name or "").split(".")[0] == "vllm" else unchecked).append(f"{key}: {e}")
    assert not unresolved, "entries the installed vLLM doesn't have:\n" + "\n".join(unresolved)
    if unchecked:
        warnings.warn("entries not checked, for a dependency this host lacks:\n" + "\n".join(unchecked), stacklevel=1)
