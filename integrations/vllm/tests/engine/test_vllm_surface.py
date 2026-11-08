"""`engine.vllm_surface`: the closed list of vLLM's modules and names that the library code imports."""
from __future__ import annotations

import ast
import importlib
import warnings
from pathlib import Path

import pytest

from verity_vllm import config
from verity_vllm.engine.vllm_surface import SURFACE

PACKAGE = Path(__file__).resolve().parents[2] / "verity_vllm"


def imported(text: str) -> set[str]:
    """The vLLM modules a source imports whole, and the ``module:name`` pairs it imports from one."""
    out: set[str] = set()
    for n in ast.walk(ast.parse(text)):
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
    for p in sorted(PACKAGE.rglob("*.py")):
        for key in imported(p.read_text()):
            uses.setdefault(key, set()).add(str(p.relative_to(PACKAGE)))
    assert len(uses) > 50
    missing = {k: sorted(v) for k, v in uses.items() if k not in SURFACE}
    assert not missing, ("vLLM imports that engine.vllm_surface.SURFACE doesn't list: add each with why the integration "
                         "needs it, or reach it through a module that already imports it:\n"
                         + "\n".join(f"  {k}: {', '.join(v)}" for k, v in sorted(missing.items())))
    stale = sorted(set(SURFACE) - set(uses))
    if stale:
        warnings.warn(f"entries of engine.vllm_surface.SURFACE that nothing imports: delete each once no change in flight "
                      f"adds its import: {stale}", stacklevel=1)


def test_every_entry_resolves_at_the_installed_vllm():
    vllm = pytest.importorskip("vllm")
    assert config.VLLM in vllm.__version__, f"vLLM {vllm.__version__} is installed, not the pin {config.VLLM}"
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
