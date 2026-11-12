"""The split-kernel launch proxies `pipeline.topp_pin_check` reads: the top-p split count S is a HOST constant of vLLM
`_apply_topp_split` (constexpr kwarg `S` of the three `_topp_sb_*_kernel` launches, grid = rows * S), never on the device.  The proxies
sit on the module's kernel globals, record (kernel, grid, S) per launch and launch the original unchanged."""
from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("torch")

from verity_vllm.commit.committer import topp_splits as NC


class _FakeKernel:
    """Stands in for a Triton JITFunction: `kernel[grid](*args, **kw)`; records what it was launched with."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.launches: list[tuple] = []

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
    assert {nm: p.original for nm, p in orig.items()} == originals
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
    """An older vLLM without `_apply_topp_split`: no proxies, so no launch S is read."""
    monkeypatch.setitem(sys.modules, NC.TOPP_SPLIT_MODULE, types.ModuleType(NC.TOPP_SPLIT_MODULE))
    assert NC.install_split_kernel_proxies([]) == {}
    NC.uninstall_split_kernel_proxies({})                                  # no-op
