"""In this benchmark, every file imports the vLLM adapter through its public interface only: a `verity_vllm` module that
``tests/test_vllm_public_api.py`` lists in ``PUBLIC``, and from it only the names its ``__all__`` lists. An import is what
the harness's scan (`tests/benchmarks/harness/test_adapter.py`) counts, each form it states."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = (Path(__file__).resolve().parents[3] / "benchmarks" / "inference_only")
ROOT = HERE.parents[1]
SCAN = ROOT / "tests/benchmarks/harness/test_adapter.py"
#: files whose `verity_vllm` strings are data, not imports, each with why
NOT_SCANNED = {"tests/benchmarks/inference_only/test_adapter.py": "this file"}


def _scan():
    spec = importlib.util.spec_from_file_location("harness_adapter_scan", SCAN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S = _scan()


def sites() -> list:
    files = [p for p in sorted(HERE.rglob("*")) if p.suffix in (".py", ".sh") and p.is_file()
             and str(p.relative_to(ROOT)) not in NOT_SCANNED
             and not any(s.startswith(".") or s == "__pycache__" for s in p.relative_to(HERE).parts)]
    return [s for p in files for s in S.sites_in(str(p.relative_to(ROOT)), p.read_text(encoding="utf-8"))]


def test_no_file_reaches_past_the_public_interface():
    found = sites()
    bad = [f"{s.file}:{s.line}: imports {s.module}{'' if s.name is None else '.' + s.name}, which is not the vLLM adapter's "
           f"public interface (PUBLIC in {S.PUBLIC_API}, and each module's __all__): import a name it offers, or ask the "
           "adapter to make the name public"
           for s in found if not S.is_public(s)]
    assert found and not bad, "\n".join(bad) or "the scan found no import of the adapter here: check SCAN and HERE"
