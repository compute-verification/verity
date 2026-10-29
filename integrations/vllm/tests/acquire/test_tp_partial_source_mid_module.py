"""The TP partial source at a MID-module collective site -- a model class (OlmoeAttention) that calls
`tensor_model_parallel_all_gather` from its own module, twice per forward (q, then k).  The source finds the site from the class source
(`verity_vllm.engine.collectives.collective_sites`), hooks the module, binds each call's rank-local INPUT as `<module>/shard_rank<r>` and RECEIVED output
as `<module>/out` (occurrence k = the manifest's invocation k), declares 2 partials per step for identity_check, and pairs the Match oracle
by (step, module, occurrence).  Torch-free (fake tensors / modules / committer), as test_tp_partial_source_moe."""
from __future__ import annotations

import types

import pytest

pytest.importorskip("verity_vllm.acquire.sources.partial_source")
from verity_vllm.acquire.sources import partial_source as tps
from tests.acquire.test_tp_partial_source_moe import _Committer, _Mod, _T, _make, _model
from verity_vllm.engine.collectives import collective_sites as CS

THIS = __name__


def tensor_model_parallel_all_gather(x, dim=-1):                 # the name a model module binds by `from vllm.distributed import ...`
    return x


class OlmoeAttentionLike(_Mod):
    """vLLM olmoe.py's shape: two gathers in a helper the forward calls, under `if self.tp_size > 1`."""

    tp_size = 2
    hidden_size = 2048

    def _apply_qk_norm(self, q, k):
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q)
            k = tensor_model_parallel_all_gather(k)
        return q, k


class PlainMlp(_Mod):
    """No collective call of its own (the fused MoE's all-reduce is the runner's)."""

    def forward(self, x):
        return x


class Qwen3MoeSparseMoeBlockLike(_Mod):
    """vLLM qwen3_moe.py's shape: the block all-gathers its output only under sequence-parallel MoE, which plain TP leaves off."""

    tp_size = 2

    def __init__(self, is_sequence_parallel: bool) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel

    def forward(self, x):
        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x, 0)
        return x


class FlagBranches(_Mod):
    tp_size = 2

    def __init__(self, flag) -> None:
        super().__init__()
        self.flag = flag

    def forward(self, x, already=False):
        if not self.flag:
            x = tensor_model_parallel_all_gather(x)
        else:
            x = tensor_model_parallel_all_gather(x)
            x = tensor_model_parallel_all_gather(x)
        if self.flag and not already:                    # vLLM deepseek_v2.py / qwen3_next.py's MoE blocks
            x = tensor_model_parallel_all_gather(x)
        else:
            x = tensor_model_parallel_all_gather(x)
        return x


def test_class_collective_calls_reads_the_class_source_under_the_model_prefix():
    assert CS.class_collective_calls(OlmoeAttentionLike, model_prefix=THIS) == {"all_gather": 2}
    assert CS.class_collective_calls(PlainMlp, model_prefix=THIS) == {}
    assert CS.class_collective_calls(OlmoeAttentionLike) == {}                   # not a vllm.model_executor.models.* class: not a site
    assert CS.class_collective_calls(type("Dyn", (), {}), model_prefix="") == {}  # no source: not a site (fail closed by omission)


def test_mid_module_sites_and_patch_targets():
    attn, mlp = OlmoeAttentionLike(), PlainMlp()
    m = _model({"model.layers.0.self_attn": attn, "model.layers.0.mlp": mlp})
    sites = CS.mid_module_sites(m, 2, model_prefix=THIS)
    assert [(s["name"], s["cls"], s["calls"], s["per_step"]) for s in sites] == [("model.layers.0.self_attn", "OlmoeAttentionLike", {"all_gather": 2}, 2)]
    assert CS.mid_module_sites(m, 1, model_prefix=THIS) == []                    # TP1: no collective
    tgts = CS.patch_targets(sites)
    assert [(mod.__name__, attr, kind) for mod, attr, kind in tgts] == [(THIS, "tensor_model_parallel_all_gather", "all_gather")]


def test_a_call_under_an_instance_flag_counts_only_when_the_flag_allows_it():
    """A Qwen3-30B-A3B TP2 Commit: the finder counted Qwen3MoeSparseMoeBlock's sequence-parallel all-gather at all
    48 `model.layers.N.mlp` blocks, the run issued none, and identity_check failed every step on both ranks (`tp_sites_bad` 48)."""
    off, on = Qwen3MoeSparseMoeBlockLike(False), Qwen3MoeSparseMoeBlockLike(True)
    assert CS.class_collective_calls(Qwen3MoeSparseMoeBlockLike, model_prefix=THIS) == {"all_gather": 1}
    assert CS.class_collective_calls(Qwen3MoeSparseMoeBlockLike, model_prefix=THIS, instance=off) == {}
    assert CS.class_collective_calls(Qwen3MoeSparseMoeBlockLike, model_prefix=THIS, instance=on) == {"all_gather": 1}
    sites = CS.mid_module_sites(_model({"model.layers.0.mlp": off, "model.layers.1.mlp": on}), 2, model_prefix=THIS)
    assert [(s["name"], s["per_step"]) for s in sites] == [("model.layers.1.mlp", 1)]
    # `not self.flag` and its `else`, and a flag conjunct of an `and` (its `else` concludes nothing); a flag of another type, or one the
    # instance lacks, rules nothing out
    assert CS.class_collective_calls(FlagBranches, model_prefix=THIS) == {"all_gather": 5}
    assert CS.class_collective_calls(FlagBranches, model_prefix=THIS, instance=FlagBranches(False)) == {"all_gather": 2}
    assert CS.class_collective_calls(FlagBranches, model_prefix=THIS, instance=FlagBranches(None)) == {"all_gather": 2}
    assert CS.class_collective_calls(FlagBranches, model_prefix=THIS, instance=FlagBranches(True)) == {"all_gather": 4}
    assert CS.class_collective_calls(FlagBranches, model_prefix=THIS, instance=FlagBranches("x")) == {"all_gather": 5}
    assert CS.class_collective_calls(FlagBranches, model_prefix=THIS, instance=types.SimpleNamespace()) == {"all_gather": 5}
    # a condition other than a bare flag is not evaluated: OLMoE's `if self.tp_size > 1:` keeps both gathers, whatever tp_size holds
    attn = OlmoeAttentionLike()
    attn.tp_size = 1
    assert CS.class_collective_calls(OlmoeAttentionLike, model_prefix=THIS, instance=attn) == {"all_gather": 2}


def test_source_on_a_plain_tp_qwen3_moe_layer_declares_only_the_runner_site(monkeypatch):
    monkeypatch.setattr(CS, "MODEL_MODULE_PREFIX", THIS)
    src = tps.TPPartialSource(_Committer(), _model({"model.layers.0.mlp": Qwen3MoeSparseMoeBlockLike(False),
                                                    "model.layers.0.mlp.experts": _make("MoERunner")}), rank=0, world=2)
    assert src.site_names() == ["model.layers.0.mlp.experts"]
    assert src.received_site_names() == ["model.layers.0.mlp.experts"]
    assert src.site_counts() == {"model.layers.0.mlp.experts": 1}


def test_source_binds_shard_and_received_output_per_call_and_declares_two_per_step(monkeypatch):
    monkeypatch.setattr(CS, "MODEL_MODULE_PREFIX", THIS)
    c = _Committer()
    attn, o_proj = OlmoeAttentionLike(), _make("RowParallelLinear")
    src = tps.TPPartialSource(c, _model({"model.layers.0.self_attn": attn, "model.layers.0.self_attn.o_proj": o_proj}), rank=0, world=2)
    assert src.site_names() == ["model.layers.0.self_attn.o_proj", "model.layers.0.self_attn"]
    assert src.received_site_names() == ["model.layers.0.self_attn"]           # this source binds the gathered rows: no layer hook does
    assert src.site_counts() == {"model.layers.0.self_attn.o_proj": 1, "model.layers.0.self_attn": 2}
    src.install()
    # the model module's OWN name is patched (the vllm layer modules are not importable here and are skipped)
    assert any(p.owner.__name__ == THIS and p.name == "tensor_model_parallel_all_gather" for p in src._hooks.entries)
    q, k, gq, gk = _T(4 * 1024), _T(4 * 1024), _T(4 * 2048), _T(4 * 2048)
    outs = iter([gq, gk])
    # drive a forward: self_attn.pre -> gather(q) -> gather(k) -> o_proj.pre -> all_reduce -> o_proj.post -> self_attn.post
    attn._pre[0](attn, (q,))
    gather = src._wrap("all_gather", lambda t, *a, **kw: next(outs))
    assert gather(q) is gq and gather(k) is gk                                   # the computation is unchanged
    o_proj._pre[0](o_proj, (q,))
    src._wrap("all_reduce", lambda t, *a, **kw: _T(4 * 2048))(_T(4 * 2048))
    o_proj._post[0](o_proj, (q,), q)
    attn._post[0](attn, (q,), q)
    src.uninstall()
    assert c.errors == []
    assert [m for m, _ in c.acquired] == ["model.layers.0.self_attn/shard_rank0", "model.layers.0.self_attn/out",
                                          "model.layers.0.self_attn/shard_rank0", "model.layers.0.self_attn/out",
                                          "model.layers.0.self_attn.o_proj/part_rank0"]
    assert c.acquired[0][1] is q and c.acquired[1][1] is gq and c.acquired[2][1] is k and c.acquired[3][1] is gk
    # each received row is a collection point (flushed right away); the occurrence counter names the k gather as the second commit
    assert src.stats["received_flushes"] == 2 and src.stats["received_bound"] == 2
    assert src._occ[(0, "model.layers.0.self_attn/shard_rank0")] == 2 and src._occ[(0, "model.layers.0.self_attn/out")] == 2
    # the declaration names the site
    d = src.declaration_classes()[0]
    assert d["mid_module_sites"] == {"model.layers.0.self_attn": {"cls": "OlmoeAttentionLike", "calls": {"all_gather": 2}}}
    # arena bound: 2 calls x (shard + gathered row) <= 2 x 2H per token, 4 tensors for the site + 1 for the linear
    b = src.extra_step_bytes({}, None)
    assert b["tensors_per_step"] == 5 and b["per_token_bytes"] == 2 * 2 * 2048 * 2 + 2048 * 2


def test_occurrence_counter_restarts_every_run(monkeypatch):
    """Steps restart at 0 every run of the process (warm-up, timed pairs, then the value-check run): the value-check run's first commit
    of a member at step 0 is occurrence 0, whatever the earlier runs bound -- else every reference tap misses its committed meta."""
    monkeypatch.setattr(CS, "MODEL_MODULE_PREFIX", THIS)
    monkeypatch.setattr(tps, "_bytes", lambda t: b"\0" * (t.numel() * t.element_size()))
    c = _Committer()
    attn, o_proj = OlmoeAttentionLike(), _make("RowParallelLinear")
    src = tps.TPPartialSource(c, _model({"model.layers.0.self_attn": attn, "model.layers.0.self_attn.o_proj": o_proj}), rank=0, world=2)

    def one_step():
        attn._pre[-1](attn, (_T(8),))
        gather = src._wrap("all_gather", lambda t, *a, **kw: _T(16))
        gather(_T(8))
        gather(_T(8))
        o_proj._pre[-1](o_proj, (_T(8),))
        src._wrap("all_reduce", lambda t, *a, **kw: _T(16))(_T(16))
        o_proj._post[-1](o_proj, (_T(8),), None)
        attn._post[-1](attn, (_T(8),), None)

    for _run in range(3):                                          # warm-up + timed pairs, no reference tap
        src.install()
        one_step()
        src.uninstall()
    fn = getattr(src, "begin_run", None)                           # native_collect.reset(): begin_run() on every source that has one
    if fn is not None:
        fn()
    src.install()
    src.reference, src.ref_reexec_steps = True, 0
    one_step()
    src.uninstall()
    assert sorted(src.ref[0]) == ["model.layers.0.self_attn.o_proj/part_rank0", "model.layers.0.self_attn/out", "model.layers.0.self_attn/out#1",
                                  "model.layers.0.self_attn/shard_rank0", "model.layers.0.self_attn/shard_rank0#1"]


def test_reference_keys_and_nth_meta_pair_the_second_call_with_the_second_commit():
    assert tps._ref_key("m/shard_rank0", 0) == "m/shard_rank0" and tps._ref_key("m/shard_rank0", 1) == "m/shard_rank0#1"
    assert tps._split_ref_key("m/shard_rank0#1") == ("m/shard_rank0", 1) and tps._split_ref_key("m/out") == ("m/out", 0)
    a, b, c_ = (types.SimpleNamespace(name="m/out", i=i) for i in range(3))
    by = tps._metas_by_name([a, types.SimpleNamespace(name="x/0", i=9), b, c_])
    assert tps._nth(by, "m/out", 1) is b and tps._nth(by, "m/out", 3) is None and tps._nth(by, "nope", 0) is None


def test_identity_check_requires_the_declared_count_per_site():
    """native_host.identity_check: a mid-module site with 2 declared partials per step is fine at 2 and named at 1 or 3."""
    nh = pytest.importorskip("verity_vllm.commit.committer.native_host")
    src = types.SimpleNamespace(short_name="tp_partial", rank=0, site_names=lambda: ["model.layers.0.self_attn", "model.layers.0.self_attn.o_proj"],
                                received_site_names=lambda: ["model.layers.0.self_attn"], site_counts=lambda: {"model.layers.0.self_attn": 2, "model.layers.0.self_attn.o_proj": 1})
    self = types.SimpleNamespace(_modules=[("model.layers.0.self_attn.o_proj", _make("RowParallelLinear"))], runner=None, sources=[src])
    man = nh.NativeHostCommitter.expected_manifest(self)
    assert man["tp_site_counts"] == {"model.layers.0.self_attn": 2, "model.layers.0.self_attn.o_proj": 1}
    assert "model.layers.0.self_attn" in man["modules"]
