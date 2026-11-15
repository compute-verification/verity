"""In the grader's tests, only `adapter.py` imports the vLLM adapter past its public interface (a `verity_vllm` module that
``tests/test_vllm_public_api.py`` lists in ``PUBLIC``, and from it only the names its ``__all__`` lists), and it imports only
what that interface doesn't offer. An import is what the harness's scan (`benchmarks/harness/tests/test_adapter.py`)
counts, each form it states."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
SCAN = ROOT / "benchmarks/harness/tests/test_adapter.py"
ADAPTER = "benchmarks/training_slowdown/grader/tests/adapter.py"
#: files whose `verity_vllm` strings are data, not imports, each with why
NOT_SCANNED = {"benchmarks/training_slowdown/grader/tests/test_adapter.py": "this file"}


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


def test_only_the_adapter_module_reaches_past_the_public_interface():
    bad = [f"{s.file}:{s.line}: imports {s.module}{'' if s.name is None else '.' + s.name}, which is not the vLLM adapter's "
           f"public interface (PUBLIC in {S.PUBLIC_API}, and each module's __all__): import it in {ADAPTER} once, with "
           "the name asked of the adapter in that module's table, and take it from `adapter` here"
           for s in sites() if s.file != ADAPTER and not S.is_public(s)]
    assert not bad, "\n".join(bad)


def test_the_adapter_module_imports_only_what_has_no_public_home():
    reached = [s for s in sites() if s.file == ADAPTER]
    assert reached, f"{ADAPTER} imports nothing of {S.PACKAGE}: delete it and this test, now the interface offers every name"
    stale = [f"{s.module}{'' if s.name is None else '.' + s.name}" for s in reached if S.is_public(s)]
    assert not stale, (f"{ADAPTER} imports names the public interface now offers: import them from there in their users, "
                       f"and delete their lines and table rows in {ADAPTER}: {stale}")
