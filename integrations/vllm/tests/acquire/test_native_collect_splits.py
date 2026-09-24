"""[R16 stoch REQ-03 / reqval M-0305] the collector binding of `runner.sampler/splits` (manifest family `sampler_splits`): the top-p split
count S is a HOST constant of vLLM `_apply_topp_split` (constexpr kwarg `S` of the three `_topp_sb_*_kernel` launches, grid = rows * S),
never on the device.  The collector installs launch proxies on the module's kernel globals, records (kernel, grid, S) per launch and binds
the step's S once per sampled row -- it never recomputes S from the batch.  Disagreeing launches bind nothing (-> coverage fails closed)."""
from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("torch")

from verity_vllm.acquire import native_collect as NC  # noqa: E402


class _FakeKernel:
    """Stands in for a Triton JITFunction: `kernel[grid](*args, **kw)`; records what it was launched with."""

    def __init__(self, name: str) -> None:
        self.name, self.launches = name, []

    def __getitem__(self, grid):
        def run(*args, **kw):
            self.launches.append((grid, args, kw))
            return f"{self.name}-ran"
        return run


@pytest.fixture
def fake_vllm_module(monkeypatch):
    mod = types.ModuleType(NC.TOPP_SPLIT_MODULE)
    for nm in NC.TOPP_SPLIT_KERNELS:
        setattr(mod, nm, _FakeKernel(nm))
    monkeypatch.setitem(sys.modules, NC.TOPP_SPLIT_MODULE, mod)
    return mod


def test_launch_proxy_records_the_kernels_constexpr_s_and_launches_the_original(fake_vllm_module):
    mod = fake_vllm_module
    originals = {nm: getattr(mod, nm) for nm in NC.TOPP_SPLIT_KERNELS}
    sink: list = []
    orig = NC.install_split_kernel_proxies(sink)
    assert orig == originals
    assert all(isinstance(getattr(mod, nm), NC._SplitKernelProxy) for nm in NC.TOPP_SPLIT_KERNELS)
    # idempotent: a second install wraps nothing twice
    assert NC.install_split_kernel_proxies(sink) == {}
    B, S = 3, 16
    out = mod._topp_sb_stats_kernel[(B * S,)]("logits", 128, "stats", "k", "p", HAS_K=False, VOCAB_SIZE=128256, S=S, BLOCK=8192, num_warps=8)
    assert out == "_topp_sb_stats_kernel-ran"
    for round_i in range(4):
        mod._topp_sb_step_kernel[(B * S,)]("logits", 128, "stats", "parts", "k", "p", round_i, HAS_K=False, S=S, F=4, NUM_ROUNDS=4, VOCAB_SIZE=128256, BLOCK=2048, num_warps=8)
    mod._topp_sb_mask_kernel[(B * S,)]("logits", 128, "stats", "parts", "k", "p", HAS_K=False, MASK_VALUE=float("-inf"), S=S, F=4, NUM_ROUNDS=4, VOCAB_SIZE=128256, BLOCK=8192, num_warps=8)
    assert [n for n, _g, _s in sink] == ["_topp_sb_stats_kernel"] + ["_topp_sb_step_kernel"] * 4 + ["_topp_sb_mask_kernel"]
    assert {g for _n, g, _s in sink} == {B * S} and {s for _n, _g, s in sink} == {S}
    # the ORIGINAL kernels ran with their arguments unchanged
    assert len(originals["_topp_sb_stats_kernel"].launches) == 1 and originals["_topp_sb_stats_kernel"].launches[0][2]["S"] == S
    assert len(originals["_topp_sb_step_kernel"].launches) == 4 and len(originals["_topp_sb_mask_kernel"].launches) == 1
    assert originals["_topp_sb_mask_kernel"].launches[0][0] == (B * S,)
    # attribute fall-through, then uninstall restores the module namespace exactly
    assert mod._topp_sb_mask_kernel.name == "_topp_sb_mask_kernel"
    NC.uninstall_split_kernel_proxies(orig)
    assert {nm: getattr(mod, nm) for nm in NC.TOPP_SPLIT_KERNELS} == originals


def test_install_without_the_split_pipeline_binds_nothing(monkeypatch):
    """An older vLLM without `_apply_topp_split`: no proxies, no S -> the manifest's `sampler_splits` rows fail coverage closed (visibly)."""
    monkeypatch.setitem(sys.modules, NC.TOPP_SPLIT_MODULE, types.ModuleType(NC.TOPP_SPLIT_MODULE))
    assert NC.install_split_kernel_proxies([]) == {}
    NC.uninstall_split_kernel_proxies({})                                  # no-op


def _step(B: int, S: int, *, mask: bool = True, s_mask: int | None = None, grid_mask: int | None = None) -> list[tuple]:
    L = [("_topp_sb_stats_kernel", B * S, S)] + [("_topp_sb_step_kernel", B * S, S)] * 4
    if mask:
        L.append(("_topp_sb_mask_kernel", grid_mask if grid_mask is not None else B * S, s_mask if s_mask is not None else S))
    return L


def test_split_count_of_one_step_is_the_agreed_kernel_s_or_nothing():
    for B, S in ((1, 16), (8, 8), (2, 32), (64, 1), (5, 4)):
        assert NC.split_count_of_launches(_step(B, S), B) == (S, None)
    # F-stoch-01's shape: a kernel S that is not the rows' expected geometry is still THE value of record (the collector does not judge it)
    assert NC.split_count_of_launches(_step(8, 16), 8) == (16, None)
    # nothing launched (a greedy / dense-path step): bind nothing, say so
    S, why = NC.split_count_of_launches([], 1)
    assert S is None and "no split-kernel launch" in why
    # disagreement within one step -> nothing (never the last launch's S)
    S, why = NC.split_count_of_launches(_step(1, 16, s_mask=8, grid_mask=8), 1)
    assert S is None and "disagree" in why
    # a launch without a readable S (callable grid / missing kwarg) -> nothing
    S, why = NC.split_count_of_launches([("_topp_sb_mask_kernel", None, None)], 1)
    assert S is None and "disagree" in why
    # the grid must be rows * S: a launch over other rows than the sampled ones is not this step's S
    S, why = NC.split_count_of_launches(_step(2, 16), 1)
    assert S is None and "grid" in why
    # stats/step launched, the mask kernel did not: the mask is what consumed S -> nothing
    S, why = NC.split_count_of_launches(_step(1, 16, mask=False), 1)
    assert S is None and "mask kernel" in why
    # a non-power-of-two S is not a split geometry the Program can take
    S, why = NC.split_count_of_launches([("_topp_sb_mask_kernel", 3, 3), ("_topp_sb_stats_kernel", 3, 3)], 1)
    assert S is None and "power of two" in why
