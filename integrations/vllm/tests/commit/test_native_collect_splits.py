"""The collector binding of `runner.sampler/splits` (manifest family `sampler_splits`): the top-p split
count S is a HOST constant of vLLM `_apply_topp_split` (constexpr kwarg `S` of the three `_topp_sb_*_kernel` launches, grid = rows * S),
never on the device.  The collector installs launch proxies on the module's kernel globals, records (kernel, grid, S) per launch and binds
the step's S once per sampled row -- it never recomputes S from the batch.  Disagreeing launches bind nothing (-> coverage fails closed)."""
from __future__ import annotations

import json
import sys
import types

import pytest

pytest.importorskip("torch")

from verity_vllm.commit.committer import native_collect as NC


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
    # a kernel S that is not the rows' expected geometry is still THE value of record (the collector does not judge it)
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



def test_a_step_vllm_skipped_top_p_on_binds_the_prescribed_s():
    """TinyLlama Gumbel (top_p 1) B8 on sm_120: the derived-S Program requires `runner.sampler/splits` per
    request per step, but vLLM runs no split kernel when every row has top_p 1 (`get_top_k_top_p` -> top_p None), so 118/118 were unbound.
    The prescribed S = SplitsFor_v1(rows, num_SMs) is bound instead -- only for a derived-S Program and only when EVERY row has top_p 1."""
    np = pytest.importorskip("numpy")
    from verity_vllm.commit.committer import topp_splits as TS
    from verity.ml.definitions.registry.topp_geometry import splits_for_v1
    ones, sm = (lambda: np.ones(8, dtype=np.float32)), (lambda: 188)
    assert TS.step_splits([], 8, splits_for_v1, ones, sm) == (16, None, "splits_prescribed_steps")
    assert TS.step_splits([], 1, splits_for_v1, lambda: [1.0], sm) == (32, None, "splits_prescribed_steps")
    # not a derived-S Program (B1 constant S, greedy): nothing bound, nothing said -- the pre-fix behaviour
    assert TS.step_splits([], 8, None, ones, sm) == (None, None, "")
    # a launch wins over the prescription: the kernel's S is the value of record
    assert TS.step_splits(_step(8, 8), 8, splits_for_v1, ones, sm) == (8, None, "splits_bound_steps")
    S, why, _ = TS.step_splits([], 8, splits_for_v1, lambda: np.array([1.0] * 7 + [0.95], dtype=np.float32), sm)
    assert S is None and "top_p < 1" in why
    S, why, _ = TS.step_splits([], 8, splits_for_v1, lambda: None, sm)
    assert S is None and "no per-row top_p" in why
    S, why, _ = TS.step_splits([], 8, splits_for_v1, lambda: [1.0] * 7, sm)
    assert S is None and "rows" in why


def test_every_tp_rank_arms_the_prescribed_s_as_the_single_rank_commit_does(tmp_path):
    """phi4-14b TP2 B8 top_p 1 on sm_120: each rank left its 121 `runner.sampler/splits` unbound.
    Only the single-rank `taps.attach` armed the prescribed S; a TP rank attaches through `taps.for_ranks` (None with no tap on) and
    `attach_rank`, which did not.  Every rank arms it now, with or without taps, for a derived-S manifest only."""
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    from verity_vllm.acquire.sources import taps as T
    from verity_vllm.commit.committer import topp_splits as TS
    from verity.ml.definitions.registry.topp_geometry import splits_for_v1
    rows = [{"family": "sampler_splits", "op_path": "runner.sampler", "request_id": "r0", "step": 0, "rows": 1, "rank": r} for r in (0, 1)]
    derived = {"populations": {"by_family": {"sampler_splits": 2}}, "identities": rows}
    (tmp_path / "derived.json").write_text(json.dumps(derived))
    (tmp_path / "constant.json").write_text(json.dumps({"populations": {"by_family": {"sampled_token_ids": 2}}, "identities": []}))
    ones, sm = (lambda: np.ones(8, dtype=np.float32)), (lambda: 188)
    for r in (0, 1):
        com = SimpleNamespace(prescribe_idle_splits=None, name="rec")
        assert T.for_ranks(T.TapSources())(com, None, str(tmp_path / "derived.json"), r) == {"prescribe_idle_splits": "SplitsFor_v1"}
        assert com.prescribe_idle_splits is splits_for_v1
        assert TS.step_splits([], 8, com.prescribe_idle_splits, ones, sm) == (16, None, "splits_prescribed_steps")
        # greedy, or B1 whose S is a constant of the Program: nothing armed, nothing bound on an idle step
        com = SimpleNamespace(prescribe_idle_splits=None, name="rec")
        assert T.for_ranks(T.TapSources())(com, None, str(tmp_path / "constant.json"), r) == {}
        assert com.prescribe_idle_splits is None and TS.step_splits([], 8, None, ones, sm) == (None, None, "")
    assert T.for_ranks(T.TapSources())(SimpleNamespace(name="rec"), None, str(tmp_path / "derived.json"), 0) == {}, "a committer without it"
    com = SimpleNamespace(prescribe_idle_splits=None, name="rec")
    assert T.attach(com, derived, (), T.TapSources()) == () and com.prescribe_idle_splits is splits_for_v1


def test_the_served_batch_top_p_is_read_from_the_sampling_states():
    np = pytest.importorskip("numpy")
    from types import SimpleNamespace as NS
    from verity_vllm.commit.committer import topp_splits as TS
    runner = NS(sampler=NS(sampling_states=NS(top_p=NS(np=np.array([0.5, 1.0, 1.0, 0.9], dtype=np.float32)))))
    ib = NS(idx_mapping_np=np.array([2, 1]))
    assert list(TS.batch_top_p(runner, ("logits", ib), {})) == [1.0, 1.0]
    assert list(TS.batch_top_p(runner, (), {"input_batch": ib})) == [1.0, 1.0]
    assert TS.batch_top_p(NS(), ("logits", ib), {}) is None


def test_the_select_ignores_s_at_top_p_one():
    """What makes the prescribed S exact: at p >= 1 the S-split top-p mask returns early and keeps every lane, for every S -- the served
    no-top-p logits row."""
    np = pytest.importorskip("numpy")
    from verity.ml.definitions.registry.topp_split import topp_split_row
    x = np.random.default_rng(0).standard_normal(32000).astype(np.float32) * 4
    for S in (1, 2, 4, 8, 16, 32):
        out, _stats, _parts, trace = topp_split_row(x, 1.0, S)
        assert trace["keep"].all() and np.array_equal(out, x)
