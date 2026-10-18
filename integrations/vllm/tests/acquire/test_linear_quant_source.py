"""The block-FP8 linears' input quantization source: inside a linear's call during a committed step, the `_C` group-quant launch's
outputs are bound as `<linear>/x_q` (as written) and `<linear>/x_s` (the column-major scales' row-major copy) and copied at once;
nothing outside a linear or a step; a second launch in one call, or scales of another shape, is a named collector error."""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import linear_quant_source as LQ  # noqa: E402

K, GROUP = 256, 128
OPS = types.SimpleNamespace()


def _quant(x, x_q, x_s, group_size, *a, **k):
    x_q.copy_(x.to(x_q.dtype))
    x_s.copy_(torch.arange(x_s.numel(), dtype=torch.float32).reshape(x_s.shape) + 1)


class BlockFp8(torch.nn.Module):
    """vLLM's DeepGEMM branch as it reaches the op: column-major scales, one launch per call (`launches` to break that)."""

    def __init__(self, launches: int = 1, scale_cols: int = K // GROUP) -> None:
        super().__init__()
        self.weight_scale_inv = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
        self.quant_method, self.input_size_per_partition = object(), K
        self.launches, self.scale_cols = launches, scale_cols

    def forward(self, x):
        for _ in range(self.launches):
            x_q = torch.empty(x.shape, dtype=torch.float8_e4m3fn)
            x_s = torch.empty(self.scale_cols, x.shape[0], dtype=torch.float32).permute(1, 0)
            OPS.per_token_group_fp8_quant(x, x_q, x_s, GROUP, 1e-10, -448.0, 448.0, True)
        return x


class Model(torch.nn.Module):
    def __init__(self, **kw) -> None:
        super().__init__()
        self.q_a_proj, self.norm = BlockFp8(**kw), torch.nn.LayerNorm(K)

    def forward(self, x):
        OPS.per_token_group_fp8_quant(x, torch.empty(x.shape, dtype=torch.float8_e4m3fn), torch.empty(x.shape[0], K // GROUP), GROUP)
        return self.q_a_proj(self.norm(x))


class Committer:
    gather = True

    def __init__(self) -> None:
        self.log, self.acquired, self.errors, self._cur = [], {}, [], None
        self._nc = types.SimpleNamespace(flush=lambda: self.log.append(("flush",)))

    def _acquire(self, name, t, job):
        self.log.append(("acquire", name))
        self.acquired[name] = t


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(OPS, "per_token_group_fp8_quant", _quant, raising=False)
    c, model = Committer(), Model()
    src = LQ.LinearQuantSource(c, model)
    src.torch = types.SimpleNamespace(ops=types.SimpleNamespace(_C=OPS))
    src.install()
    yield c, model, src
    src.uninstall()


def test_inside_a_linear_during_a_step_it_binds_what_deepgemm_reads(run) -> None:
    c, model, src = run
    assert src.sites == ["q_a_proj"]
    x = torch.randn(5, K)
    model(x)
    assert c.log == []                                          # no step: warm-up, profiling
    c._cur = types.SimpleNamespace(step=0)
    model(x)
    assert c.log == [("acquire", "q_a_proj/x_q"), ("acquire", "q_a_proj/x_s"), ("flush",)]    # the model's own launch is not a linear's
    q, s = c.acquired["q_a_proj/x_q"], c.acquired["q_a_proj/x_s"]
    want = model.norm(x).detach().to(torch.float8_e4m3fn)                        # the linear's own input, quantized
    assert q.dtype == torch.float8_e4m3fn and tuple(q.shape) == (5, K) and torch.equal(q.detach().view(torch.uint8), want.view(torch.uint8))
    assert s.is_contiguous() and tuple(s.shape) == (5, K // GROUP) and torch.equal(s, torch.arange(10.0).reshape(5, 2) + 1)
    assert src.stats["calls"] == 1 and src.stats["rows"] == 5 and c.errors == []


def test_a_second_launch_or_other_scales_is_a_named_error(monkeypatch) -> None:
    monkeypatch.setattr(OPS, "per_token_group_fp8_quant", _quant, raising=False)
    for kw, msg in (({"launches": 2}, "launched per_token_group_fp8_quant 2 times"), ({"scale_cols": 1}, "into scales (5, 1)")):
        c = Committer()
        src = LQ.LinearQuantSource(c, Model(**kw))
        src.torch = types.SimpleNamespace(ops=types.SimpleNamespace(_C=OPS))
        src.install()
        c._cur = types.SimpleNamespace(step=0)
        try:
            src.model(torch.randn(5, K))
        finally:
            src.uninstall()
        assert len(c.errors) == 1 and msg in c.errors[0] and src.stats["errors"] == 1
    assert OPS.per_token_group_fp8_quant is _quant


def test_it_attaches_only_where_the_manifest_binds_the_definition() -> None:
    rows = {"identities": [{"spec": "DeepGemmFp8Linear_v1{M=5,K=256}"}]}
    assert LQ.manifest_names_it(rows) and not LQ.manifest_names_it({"identities": [{"spec": "Linear_v1{K=2}"}]}) and not LQ.manifest_names_it(None)
    assert LQ.attach(types.SimpleNamespace(), Model(), {"identities": []}) is None
    with pytest.raises(ValueError, match="no block-FP8 linear"):
        LQ.LinearQuantSource(Committer(), torch.nn.Linear(2, 2))
    assert LQ.LinearQuantSource(Committer(), Model()).extra_step_bytes({}, Model()) == {"per_token_bytes": K + K * 4 // GROUP, "per_seq_bytes": 0,
                                                                                        "tensors_per_step": 2}
