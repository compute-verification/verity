"""`engine.topp_pin`: every served `_apply_topp_split` launch runs at the registered split count, whatever its sampler batch, through
the module's own global, armed before the module is imported or installed on it."""
from __future__ import annotations

import sys
import types

import pytest

from verity_vllm.engine import topp_pin
from verity.ml.definitions.registry.topp_geometry import SPLIT_MAX_BATCH, registered_splits, splits_for

#: the RTX PRO 6000, the L40S, the H100, and two SM counts below 32
SM_COUNTS = (188, 142, 132, 16, 6)

_SPLIT_LOOP = """
def _apply_topp_split(logits, k, p, mask_value, num_sm):
    batch_size = logits.shape[0]
    splits = 1
    while splits < 32 and splits * 2 * batch_size <= num_sm:
        splits *= 2
    LAUNCHES.append(splits)
"""


class _Logits:
    def __init__(self, rows: int) -> None:
        self.shape = (rows, 4096)


def _vllm_module(name: str = "fake_topk_topp_triton") -> types.ModuleType:
    """vLLM d9105ea80's `_apply_topp_split` as far as S goes: its doubling loop, recording the split count it launches with."""
    m = types.ModuleType(name)
    m.LAUNCHES = []
    exec(_SPLIT_LOOP, m.__dict__)
    return m


@pytest.fixture
def clean():
    topp_pin.uninstall()
    yield
    topp_pin.uninstall()


def test_unpinned_vllm_picks_s_from_the_batch(clean) -> None:
    m = _vllm_module()
    for B in (1, 5, 6, 12, 24, 48, 64):
        m._apply_topp_split(_Logits(B), None, None, float("-inf"), 188)
    assert m.LAUNCHES == [splits_for(B, 188) for B in (1, 5, 6, 12, 24, 48, 64)] == [32, 32, 16, 8, 4, 2, 2]


@pytest.mark.parametrize("num_sm", SM_COUNTS)
def test_the_pin_launches_the_registered_s_at_every_batch(clean, num_sm) -> None:
    m = _vllm_module()
    p = topp_pin.install(m)
    for B in range(1, SPLIT_MAX_BATCH + 1):
        m._apply_topp_split(_Logits(B), None, None, float("-inf"), num_sm)
    assert set(m.LAUNCHES) == {registered_splits(num_sm)} and len(m.LAUNCHES) == SPLIT_MAX_BATCH
    assert topp_pin.pinned() and topp_pin.install(m) is p


def test_uninstall_restores_vllms_split_count(clean) -> None:
    m = _vllm_module()
    original = m._apply_topp_split
    topp_pin.install(m)
    topp_pin.uninstall()
    assert m._apply_topp_split is original and not topp_pin.pinned()
    m._apply_topp_split(_Logits(64), None, None, float("-inf"), 188)
    assert m.LAUNCHES == [2]


def test_arm_installs_when_vllms_module_is_imported_and_imports_nothing_itself(clean, tmp_path, monkeypatch) -> None:
    pkg = tmp_path / "fake_vllm_sampler"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "topk_topp_triton.py").write_text("LAUNCHES = []\n" + _SPLIT_LOOP)
    monkeypatch.syspath_prepend(str(tmp_path))
    site = "fake_vllm_sampler.topk_topp_triton"
    try:
        topp_pin.arm(site)
        topp_pin.arm(site)
        assert site not in sys.modules and not topp_pin.pinned()
        assert sum(isinstance(f, topp_pin._OnImport) and f.site == site for f in sys.meta_path) == 1
        import fake_vllm_sampler.topk_topp_triton as mod

        mod._apply_topp_split(_Logits(24), None, None, float("-inf"), 188)
        assert mod.LAUNCHES == [32] and topp_pin.pinned()
    finally:
        sys.modules.pop(site, None)
        sys.modules.pop("fake_vllm_sampler", None)


def test_the_pin_is_armed_at_import_by_every_rank_and_by_the_driver_before_it_builds() -> None:
    import ast
    import inspect

    from verity_vllm.engine import build
    from verity_vllm.pipeline.tp import rank_worker

    top = ast.parse(inspect.getsource(topp_pin)).body
    assert any(isinstance(n, ast.Expr) and ast.unparse(n) == "arm()" for n in top)
    assert any(isinstance(n, ast.ImportFrom) and n.module == "verity_vllm.engine" and "topp_pin" in {a.name for a in n.names}
               for n in ast.parse(inspect.getsource(rank_worker)).body)
    src = inspect.getsource(build.build_engine)
    assert src.index("topp_pin.arm()") < src.index("llm = LLM(**kw)")
