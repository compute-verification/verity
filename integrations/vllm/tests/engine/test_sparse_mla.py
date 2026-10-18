"""`engine.sparse_mla`: the serving switch is off by default (vLLM's heuristic); ``"1"`` pins FA3's sparse MLA backend to one
split, through that module's own global, armed before the module is imported or installed on it; any other value is refused."""
from __future__ import annotations

import sys
import types

import pytest

from verity_vllm.engine import sparse_mla


def _fa3(**kwargs):
    return kwargs.get("num_splits", 0)


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv(sparse_mla.ENV, raising=False)
    sparse_mla.uninstall()
    yield monkeypatch
    sparse_mla.uninstall()


@pytest.fixture
def on(clean):
    clean.setenv(sparse_mla.ENV, "1")
    return clean


def _backend() -> types.ModuleType:
    m = types.ModuleType("fake_flashattn_mla_sparse")
    m.flash_attn_varlen_func = _fa3
    return m


@pytest.mark.parametrize("value", [None, "0"], ids=["unset", "zero"])
def test_off_leaves_the_heuristic_and_arms_nothing(clean, value) -> None:
    if value is not None:
        clean.setenv(sparse_mla.ENV, value)
    m = _backend()
    assert sparse_mla.wanted() is None
    assert sparse_mla.install(m) is None and m.flash_attn_varlen_func is _fa3 and sparse_mla.num_splits() is None
    sparse_mla.arm("fake_never_imported_site")
    assert not any(isinstance(f, sparse_mla._OnImport) for f in sys.meta_path)


def test_any_other_value_is_refused(clean) -> None:
    clean.setenv(sparse_mla.ENV, "2")
    with pytest.raises(ValueError, match=sparse_mla.ENV):
        sparse_mla.install(_backend())


def test_the_pin_passes_one_split_and_uninstall_restores_the_module(on) -> None:
    m = _backend()
    p = sparse_mla.install(m)
    assert m.flash_attn_varlen_func(q=None) == 1 and m.flash_attn_varlen_func(q=None, num_splits=0) == 1
    assert p.original is _fa3 and sparse_mla.num_splits() == 1
    assert sparse_mla.install(m) is p
    sparse_mla.uninstall()
    assert m.flash_attn_varlen_func is _fa3 and sparse_mla.num_splits() is None


def test_arm_installs_when_the_backend_is_imported_and_imports_nothing_itself(on, tmp_path) -> None:
    pkg = tmp_path / "fake_vllm_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "sparse.py").write_text("def flash_attn_varlen_func(**kw):\n    return kw.get('num_splits', 0)\n")
    on.syspath_prepend(str(tmp_path))
    site = "fake_vllm_pkg.sparse"
    try:
        sparse_mla.arm(site)
        sparse_mla.arm(site)
        assert site not in sys.modules and sparse_mla.num_splits() is None
        assert sum(isinstance(f, sparse_mla._OnImport) and f.site == site for f in sys.meta_path) == 1
        import fake_vllm_pkg.sparse as backend

        assert backend.flash_attn_varlen_func(q=None) == 1 and sparse_mla.num_splits() == 1
        sparse_mla.uninstall()
        assert backend.flash_attn_varlen_func(q=None) == 0
        sparse_mla.arm(site)
        assert backend.flash_attn_varlen_func(q=None) == 1
    finally:
        sys.modules.pop(site, None)
        sys.modules.pop("fake_vllm_pkg", None)


def test_a_rank_arms_the_switch_at_import_and_the_driver_before_it_builds() -> None:
    import ast
    import inspect

    from verity_vllm.engine import build, rank_worker

    top = ast.parse(inspect.getsource(sparse_mla)).body
    assert any(isinstance(n, ast.Expr) and ast.unparse(n) == "arm()" for n in top)
    assert any(isinstance(n, ast.ImportFrom) and n.module == "verity_vllm.engine" and "sparse_mla" in {a.name for a in n.names}
               for n in ast.parse(inspect.getsource(rank_worker)).body)
    src = inspect.getsource(build.build_engine)
    assert src.index("sparse_mla.arm()") < src.index("llm = LLM(**kw)")
