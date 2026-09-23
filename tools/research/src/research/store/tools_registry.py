"""name -> `dotted.module:NAME` for `research run --tool NAME` (store README §6.1).  Keep it tiny; declarations live next to
the tool they describe.  `load` also accepts the spec form directly and, when the module is not importable (a shipped `git
archive` cwd, or a console-script sys.path without the repo root), loads `<source_root>/<module path>.py` by file."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path

from .tool import Tool

REGISTRY: dict[str, str] = {
    "tc_probe": "tools.tc_probe.tool:TC_PROBE",
    "tc_probe_fp4": "tools.tc_probe_fp4.tool:TC_PROBE_FP4",
    "bench_vu": "backends.direct.ligero.tool:BENCH_VU",
    "a_gpu_prove": "backends.gkr.gpu.tool:A_GPU_PROVE",
    "bench_vu_fp8": "backends.direct.ligero.fp8.tool:BENCH_VU_FP8",
    # vLLM integration sweep stages (integration-owned declarations).
    "vllm.build": "integrations.vllm.verity_vllm.research_tools:VLLM_BUILD",
    "vllm.match": "integrations.vllm.verity_vllm.research_tools:VLLM_MATCH",
    "vllm.commit": "integrations.vllm.verity_vllm.research_tools:VLLM_COMMIT",
}


def _from_file(module: str, attr: str, source_root: Path):
    """Execute `<source_root>/<module path>.py` under a name unique to that file: no sys.path edits, no cross-root module caching."""
    rel = Path(*module.split("."))
    for cand in (source_root / rel.with_suffix(".py"), source_root / rel / "__init__.py"):
        if cand.is_file():
            name = f"_research_tool_{hashlib.sha256(str(cand.resolve()).encode()).hexdigest()[:16]}"
            mod = sys.modules.get(name)
            if mod is None:
                spec = importlib.util.spec_from_file_location(name, cand)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return getattr(mod, attr)
    raise ImportError(f"no {rel}.py under {source_root}")


def load(spec_or_name: str, source_root: Path | str | None = None) -> Tool:
    spec = REGISTRY.get(spec_or_name, spec_or_name)
    if ":" not in spec:
        raise LookupError(f"unknown tool {spec_or_name!r}; registered: {', '.join(sorted(REGISTRY))} (or give dotted.module:NAME)")
    module, attr = spec.rsplit(":", 1)
    root = Path(source_root or os.getcwd())
    try:
        obj = getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as e1:
        try:
            obj = _from_file(module, attr, root)
        except (ImportError, AttributeError) as e2:
            raise ImportError(f"{spec}: {e1}; {e2}") from e2
    if not isinstance(obj, Tool):
        raise TypeError(f"{spec} is {type(obj).__name__}, not a research.store.tool.Tool")
    if obj.spec is None:
        obj.spec = spec
    return obj
