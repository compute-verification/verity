"""The norm-scale source (`acquire.sources.norm_scale_source`) on CPU tensors with a recording committer: the module scope, the Triton
stand-in and the rsqrt site acquire `<module>/norm_scale` only inside one of its norm modules during a committed step, a site of another
kind is a collector error, and install/uninstall restore `torch.rsqrt`."""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import norm_scale_source as S  # noqa: E402
from verity_vllm.query import norm_scales as NS  # noqa: E402


class Recorder:
    def __init__(self) -> None:
        self._cur: types.SimpleNamespace | None = None
        self.errors: list[str] = []
        self.acquired: list[tuple[str, tuple]] = []
        self.sources: list = []
        self.name = "rec"
        self.model = None

    def _acquire(self, name, t, job):
        self.acquired.append((name, tuple(t.shape)))

    def attach_source(self, src):
        self.sources.append(src)


class Kernel:
    """A stand-in for a jitted kernel: `k[grid](*args, **kw)` records the launch."""

    def __init__(self) -> None:
        self.launches: list[tuple] = []
        self.src = "kernel source"

    def __getitem__(self, grid):
        def launch(*a, **kw):
            self.launches.append((grid, a, kw))
            return "launched"

        return launch


def model():
    m = torch.nn.Module()
    m.tri = torch.nn.Identity()
    m.chain = torch.nn.Identity()
    return m


def source(com=None, mods=None):
    com = com or Recorder()
    mods = mods or {"tri": "RMSNormTriton_v1", "chain": "RsqrtF32_v1"}
    return S.NormScaleSource(com, model(), mods), com


def enter(src, op, x):
    src._enter(op)(None, (x,))


def test_where_needs_a_committed_step_and_a_norm_module_of_its_kind():
    src, com = source()
    x = torch.zeros(3, 8)
    assert src.where("triton") is None
    com._cur = types.SimpleNamespace(step=4)
    assert src.where("triton") is None, "outside every norm module"
    enter(src, "tri", x)
    job, op, lead = src.where("triton")
    assert (job.step, op, lead) == (4, "tri", (3,))
    assert src.where("aten", strict=False) is None and com.errors == []
    assert src.where("cuda") is None and "a cuda norm kernel ran inside tri" in com.errors[0]
    src._leave("tri")(None, (), None)
    assert src.where("triton") is None and src._scope == []


def test_triton_stand_in_launches_the_copy_with_a_scale_inside_the_module_only():
    src, com = source()
    installed, tapped = Kernel(), Kernel()
    stand_in = S.TritonTap(src, installed, tapped)
    x = torch.zeros(6, 8)
    args = (x, x, torch.empty_like(x), 8, 8, 8, 1e-5)
    stand_in[(6,)](*args, BLOCK_SIZE=1024, HAS_WEIGHT=True)
    assert len(installed.launches) == 1 and not tapped.launches, "no committed step: the installed kernel"
    com._cur = types.SimpleNamespace(step=0)
    enter(src, "tri", x.view(2, 3, 8))
    assert stand_in[(6,)](*args, BLOCK_SIZE=1024, HAS_WEIGHT=True) == "launched"
    grid, a, kw = tapped.launches[0]
    assert grid == (6,) and a[:3] == args[:3] and a[4:] == args[3:] and kw == {"BLOCK_SIZE": 1024, "HAS_WEIGHT": True}
    assert a[3].dtype == torch.float32 and tuple(a[3].shape) == (6,)
    assert com.acquired == [(f"tri/{NS.MEMBER}", (2, 3))], "the scale takes the norm input's leading shape (tokens x heads)"
    stand_in[(6,)](*args[:5], BLOCK_SIZE=1024)
    assert len(installed.launches) == 2 and "7" in com.errors[-1], "another launch shape falls back, named"
    assert stand_in.src == "kernel source", "every other attribute is the installed kernel's"


def test_rsqrt_site_takes_the_chain_norms_scale():
    src, com = source()
    rsqrt = src._rsqrt(torch.rsqrt)
    v = torch.full((4, 1), 4.0)
    assert torch.equal(rsqrt(v), torch.full((4, 1), 0.5)) and com.acquired == []
    com._cur = types.SimpleNamespace(step=1)
    enter(src, "tri", torch.zeros(4, 8))
    rsqrt(v)
    assert com.acquired == [] and com.errors == [], "an rsqrt inside a kernel norm is not its scale site"
    src._leave("tri")(None, (), None)
    enter(src, "chain", torch.zeros(4, 8))
    out = rsqrt(v)
    assert com.acquired == [(f"chain/{NS.MEMBER}", (4,))] and torch.equal(out, torch.full((4, 1), 0.5))
    assert src.stats["tensors"] == 1 and src.stats["by_kind"]["aten"] == 1 and src.stats["bytes"] == 16


def test_install_uninstall_restores_torch_rsqrt_and_the_module_hooks():
    com = Recorder()
    src = S.NormScaleSource(com, model(), {"chain": "RsqrtF32_v1"})
    orig = torch.rsqrt
    src.install()
    try:
        assert torch.rsqrt is not orig and src.active
        com._cur = types.SimpleNamespace(step=0)
        src.model.chain(torch.zeros(2, 8))
        assert src._scope == [], "the forward hook unmarks the module"
    finally:
        src.uninstall()
    assert torch.rsqrt is orig and not src.active and not src.model.chain._forward_pre_hooks and not src.model.chain._forward_hooks


def test_a_fused_norm_needs_the_tap_build():
    with pytest.raises(RuntimeError, match="no tap build"):
        S.NormScaleSource(Recorder(), model(), {"tri": "RMSNormFusedCuda_v2"})
    with pytest.raises(ValueError, match="no norm family"):
        S.NormScaleSource(Recorder(), model(), {"tri": "Gemm_v1"})
