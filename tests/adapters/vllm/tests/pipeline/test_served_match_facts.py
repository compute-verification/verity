"""What `pipeline.served_match` reads off a serving engine and a committer, on a CPU.

A served Match needs a GPU and vLLM.  Here the engine, its layers and the committer are stand-ins: vLLM's classes are
`tests.fake_vllm`'s, each set to a small class of this file, and the committed bytes are a block this file fills.  So
these cases show what the module reads and under which names: a tensor's words, the served weights, each fused-MoE
layer's facts and the engine's dispatch facts, the workload's schedule, and the check's walk over the committed steps.
That vLLM serves so is what the Match says on a GPU.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from tests import fake_vllm
from verity_vllm.acquire import served_moe
from verity_vllm.acquire.inductor_templates import Unmodelled
from verity_vllm.check.replay import served_check
from verity_vllm.pipeline import served_match as sm

MOE = "vllm.model_executor.layers.fused_moe"
BF16, F16, F32 = torch.bfloat16, torch.float16, torch.float32
VALUES = [[1.0, -2.0, 0.5], [0.0, 3.0, -0.0]]
_ABSENT = object()


def _same(got: Any, want: Any) -> bool:
    """Equal, and the same JSON: a flag that is 0 or 1 is not one that is false or true."""
    return bool(got == want) and json.dumps(got) == json.dumps(want)


@pytest.mark.parametrize(("dtype", "words", "bits"), [
    (BF16, np.uint16, [[0x3F80, 0xC000, 0x3F00], [0x0000, 0x4040, 0x8000]]),
    (F16, np.uint16, [[0x3C00, 0xC000, 0x3800], [0x0000, 0x4200, 0x8000]]),
    (F32, np.uint32, [[0x3F800000, 0xC0000000, 0x3F000000], [0x00000000, 0x40400000, 0x80000000]]),
])
def test_a_float_tensors_words_are_its_bits_as_unsigned_integers(dtype, words, bits):
    t = torch.nn.Parameter(torch.tensor(VALUES, dtype=dtype))
    for got, want in ((sm._words(t), bits), (sm._words(t.t()), np.array(bits).T)):
        assert got.dtype == words and got.tolist() == np.array(want).tolist()


def test_a_tensor_of_another_type_is_its_own_array():
    got = sm._words(torch.tensor([[3, -1], [7, 2]], dtype=torch.int64).t())
    assert got.dtype == np.int64 and got.tolist() == [[3, 7], [-1, 2]]


class _Served(torch.nn.Module):
    """An embedding tied to the head, a buffer under two names, and one stack of two experts."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Parameter(torch.tensor(VALUES, dtype=BF16))
        self.head = self.embed
        self.register_buffer("scale", torch.tensor([1.0, 0.5], dtype=F32))
        self.register_buffer("scale_again", self.scale)
        self.stack = torch.nn.Parameter(torch.arange(24, dtype=F32).reshape(2, 3, 4).to(BF16))


def test_a_served_weight_is_read_by_any_of_its_names_once_and_a_stack_of_experts_one_expert_at_a_time():
    model = _Served()
    weight = sm.served_weights(model)
    assert weight("embed").tolist() == [[0x3F80, 0xC000, 0x3F00], [0x0000, 0x4040, 0x8000]] == weight("head").tolist()
    assert weight("embed") is weight("embed") and weight("scale").tolist() == [0x3F800000, 0x3F000000] == weight("scale_again").tolist()
    stack = weight("stack")
    assert stack.shape == (2, 3, 4) and weight("stack") is not stack
    assert stack[1].dtype == np.uint16 and stack[1].tolist() == sm._words(model.stack[1]).tolist() != stack[0].tolist()
    assert stack[np.int64(0)].shape == (3, 4)
    with pytest.raises(KeyError):
        weight("no.such.weight")


class _Runner(torch.nn.Module):
    """vLLM's MoE runner as far as `moe_facts` reads it."""


class _Router:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _OtherRouter(_Router):
    pass


class _Triton:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _OtherExperts(_Triton):
    pass


class _Activation:
    SILU = "silu"
    GELU = "gelu"


def _vllm(monkeypatch) -> fake_vllm.FakeVllm:
    fake = fake_vllm.install(monkeypatch)
    fake.set(f"{MOE}.runner.moe_runner:MoERunner", _Runner)
    fake.set(f"{MOE}.router.fused_topk_router:FusedTopKRouter", _Router)
    fake.set(f"{MOE}.experts.triton_moe:TritonExperts", _Triton)
    fake.set(f"{MOE}.activation:MoEActivation", _Activation)
    return fake


def _experts(E: int, dtype: torch.dtype = BF16, *, kernel: Any = "triton") -> torch.nn.Module:
    """The routed experts of one layer: E experts of 2 x 6 and 3 x 2 weights under a quant method."""
    ex = torch.nn.Module()
    ex.w13_weight = torch.nn.Parameter(torch.zeros(E, 6, 2, dtype=dtype))
    ex.w2_weight = torch.nn.Parameter(torch.zeros(E, 2, 3, dtype=dtype))
    if kernel == "triton":
        kernel = _Triton(w1_bias=None, w2_bias=None, activation_config=SimpleNamespace(clamp_limit=None))
    ex.quant_method = SimpleNamespace(is_monolithic=False, moe_kernel=SimpleNamespace(fused_experts=kernel))
    return ex


def _block(E: int = 4, *, name: str = "layers.0.mlp", top_k: int = 2, gate: bool = True, experts: torch.nn.Module | None = None,
           router: Any = None, **runner: Any) -> torch.nn.Module:
    """A MoE block as the expansion requires it, but for what the keywords say: the block registers its gate first and its
    runner holds the same gate again."""
    block = torch.nn.Module()
    if gate:
        block.gate = torch.nn.Linear(2, E, bias=False, dtype=BF16)
    r = _Runner()
    r.layer_name = name
    r.routed_experts = _experts(E) if experts is None else experts
    r.router = _Router(top_k=top_k, renormalize=True, scoring_func="softmax") if router is None else router
    if gate:
        r.gate = block.gate
    for k, v in {"_fse_fuse_gate": False, "shared_experts": None, "routed_scaling_factor": 1.0, "activation": _Activation.SILU, **runner}.items():
        if v is not _ABSENT:
            setattr(r, k, v)
    block.runner = r
    return block


def _model(*blocks: torch.nn.Module) -> torch.nn.Module:
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList(blocks)
    return model


def _layer(i: int, E: int, k: int, vpt: int, **flags: Any) -> dict[str, Any]:
    """The facts of block ``i`` as `_block` makes it with ``E`` experts and a top-k of ``k``, but for ``flags``."""
    return {"gate": {"weight": f"blocks.{i}.gate.weight", "shape": [E, 2]},
            "w13": {"weight": f"blocks.{i}.runner.routed_experts.w13_weight", "shape": [E, 6, 2]},
            "w2": {"weight": f"blocks.{i}.runner.routed_experts.w2_weight", "shape": [E, 2, 3]},
            "E": E, "top_k": k, "renormalize": True, "VPT": vpt, "gate_held": True, "fused_gate": False, "shared_experts": False,
            "routed_scaling": False, "softmax_router": True, "triton_experts": True, "bias": False, "clamp": False, "silu": True,
            "bf16": True, **flags}


def test_a_model_with_no_moe_runner_has_no_moe_facts(monkeypatch):
    _vllm(monkeypatch)
    assert sm.moe_facts(_Served()) == {}


def test_each_moe_layer_is_named_by_its_runner_with_its_weights_under_their_first_names_and_its_own_flags(monkeypatch):
    _vllm(monkeypatch)
    model = _model(_block(4, name="layers.0.mlp", top_k=2),
                   _block(64, name="layers.1.mlp", top_k=8, router=_Router(top_k=8, renormalize=False, scoring_func="softmax"),
                          routed_scaling_factor=2.5))
    facts = sm.moe_facts(model)
    assert _same(facts, {"layers.0.mlp": _layer(0, 4, 2, 4), "layers.1.mlp": _layer(1, 64, 8, 8, renormalize=False, routed_scaling=True)})
    # the expansion's own reader takes the first layer's facts, (E, top-k, H, I), and refuses the second's for its scaling
    assert served_moe._block("layers.0.mlp", facts["layers.0.mlp"], {"width": 2}) == (4, 2, 2, 3)
    with pytest.raises(Unmodelled, match="routed_scaling"):
        served_moe._block("layers.1.mlp", facts["layers.1.mlp"], {"width": 2})


@pytest.mark.parametrize(("block", "flags"), [
    ({"_fse_fuse_gate": True}, {"fused_gate": True}),
    ({"shared_experts": torch.nn.Linear(2, 2)}, {"shared_experts": True}),
    ({"routed_scaling_factor": 0.5}, {"routed_scaling": True}),
    ({"activation": _Activation.GELU}, {"silu": False}),
    ({"router": _OtherRouter(top_k=2, renormalize=True, scoring_func="softmax")}, {"softmax_router": False}),
    ({"router": _Router(top_k=2, renormalize=True, scoring_func="sigmoid")}, {"softmax_router": False}),
    ({"router": _Router(top_k=3, renormalize=0, scoring_func="softmax")}, {"top_k": 3, "renormalize": False}),
])
def test_a_layer_that_is_not_as_the_expansion_requires_says_so_in_its_flag(monkeypatch, block, flags):
    _vllm(monkeypatch)
    assert _same(sm.moe_facts(_model(_block(**block))), {"layers.0.mlp": _layer(0, 4, 2, 4, **flags)})


@pytest.mark.parametrize(("kernel", "monolithic", "flags"), [
    (_OtherExperts(w1_bias=None, w2_bias=None, activation_config=SimpleNamespace(clamp_limit=None)), False, {"triton_experts": False}),
    (_Triton(w1_bias=None, w2_bias=None, activation_config=SimpleNamespace(clamp_limit=None)), True, {"triton_experts": False}),
    (_Triton(w1_bias=torch.zeros(1), w2_bias=None, activation_config=SimpleNamespace(clamp_limit=None)), False, {"bias": True}),
    (_Triton(w1_bias=None, w2_bias=torch.zeros(1), activation_config=SimpleNamespace(clamp_limit=None)), False, {"bias": True}),
    (_Triton(w1_bias=None, w2_bias=None, activation_config=SimpleNamespace(clamp_limit=7.0)), False, {"clamp": True}),
])
def test_the_experts_kernel_its_bias_and_its_clamp_each_have_a_flag(monkeypatch, kernel, monolithic, flags):
    _vllm(monkeypatch)
    ex = _experts(4, kernel=kernel)
    ex.quant_method.is_monolithic = monolithic
    assert _same(sm.moe_facts(_model(_block(experts=ex))), {"layers.0.mlp": _layer(0, 4, 2, 4, **flags)})


@pytest.mark.parametrize(("block", "flags"), [
    ({"_fse_fuse_gate": _ABSENT}, {"fused_gate": None}),
    ({"shared_experts": _ABSENT}, {"shared_experts": None}),
    ({"routed_scaling_factor": _ABSENT}, {"routed_scaling": None}),
    ({"activation": _ABSENT}, {"silu": None}),
    ({"router": _Router(top_k=2, scoring_func="softmax")}, {"renormalize": None}),
    ({"router": _Router(top_k=2, renormalize=True)}, {"softmax_router": None}),
    ({"router": _OtherRouter(top_k=2, renormalize=True)}, {"softmax_router": False}),
])
def test_a_flag_the_layer_does_not_let_it_read_is_none_and_the_expansion_refuses_the_layer(monkeypatch, block, flags):
    _vllm(monkeypatch)
    facts = sm.moe_facts(_model(_block(**block)))
    assert _same(facts, {"layers.0.mlp": _layer(0, 4, 2, 4, **flags)})
    with pytest.raises(Unmodelled, match=next(iter(flags))):
        served_moe._block("layers.0.mlp", facts["layers.0.mlp"], {"width": 2})


def test_experts_with_no_kernel_to_read_are_not_tritons_and_their_bias_and_clamp_are_none(monkeypatch):
    _vllm(monkeypatch)
    for quant_method in (SimpleNamespace(is_monolithic=False), SimpleNamespace(is_monolithic=False, moe_kernel=SimpleNamespace())):
        ex = _experts(4)
        ex.quant_method = quant_method
        assert _same(sm.moe_facts(_model(_block(experts=ex))), {"layers.0.mlp": _layer(0, 4, 2, 4, triton_experts=False, bias=None, clamp=None)})
    ex = _experts(4, kernel=_Triton(w1_bias=None, w2_bias=None))
    assert _same(sm.moe_facts(_model(_block(experts=ex))), {"layers.0.mlp": _layer(0, 4, 2, 4, clamp=None)})
    ex = _experts(4)
    ex.quant_method = SimpleNamespace(moe_kernel=ex.quant_method.moe_kernel)
    assert _same(sm.moe_facts(_model(_block(experts=ex))), {"layers.0.mlp": _layer(0, 4, 2, 4, triton_experts=None)})


def test_a_runner_that_holds_no_gate_and_weights_that_are_not_bf16_are_said(monkeypatch):
    _vllm(monkeypatch)
    assert _same(sm.moe_facts(_model(_block(gate=False))), {"layers.0.mlp": _layer(0, 4, 2, 4, gate=None, gate_held=False, bf16=False)})
    assert _same(sm.moe_facts(_model(_block(experts=_experts(4, F16)))), {"layers.0.mlp": _layer(0, 4, 2, 4, bf16=False)})


class _Attention(torch.nn.Module):
    """vLLM's attention layer as far as `facts_of` reads it: its backend's name and, where the kernel has one, its version."""

    def __init__(self, backend: str, version: int | None = None) -> None:
        super().__init__()
        self._backend = backend
        self.impl = SimpleNamespace() if version is None else SimpleNamespace(vllm_flash_attn_version=version)

    def get_attn_backend(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: self._backend)


def _engine(model: torch.nn.Module, capture_sizes: list[int] | None) -> SimpleNamespace:
    config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=capture_sizes, cache_dir="/a/cache"),
                             scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=2048),
                             model_config=SimpleNamespace(max_model_len=4096))
    return SimpleNamespace(case="A_CASE", effective={"seed": 0}, model=model, llm=SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=config)))


def _device(monkeypatch, fake: fake_vllm.FakeVllm) -> None:
    def properties(device: int) -> SimpleNamespace:
        assert device == 0
        return SimpleNamespace(multi_processor_count=132)

    fake.set("vllm.model_executor.layers.attention.attention:Attention", _Attention)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA H100 80GB HBM3")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)


def test_the_engines_facts_are_its_config_its_device_its_attention_kernels_and_its_moe_layers(monkeypatch):
    _device(monkeypatch, _vllm(monkeypatch))
    model = _model(_block(4))
    model.attention = torch.nn.ModuleList([_Attention("FLASH_ATTN", 3), _Attention("FLASHINFER"), _Attention("FLASH_ATTN", 3),
                                           _Attention("FLASH_ATTN", 2)])
    assert sm.facts_of(_engine(model, [8, 4, 2, 1])) == {
        "case": "A_CASE", "effective": {"seed": 0}, "cudagraph_capture_sizes": [8, 4, 2, 1], "max_num_seqs": 8,
        "max_num_batched_tokens": 2048, "max_model_len": 4096, "cache_dir": "/a/cache", "device": "NVIDIA H100 80GB HBM3",
        "cc": [9, 0], "sms": 132,
        "attention": [{"backend": "FLASHINFER", "fa_version": None}, {"backend": "FLASH_ATTN", "fa_version": 2},
                      {"backend": "FLASH_ATTN", "fa_version": 3}],
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda}, "moe": {"layers.0.mlp": _layer(0, 4, 2, 4)}}


def test_an_engine_with_no_capture_sizes_no_attention_layer_and_no_moe_layer_has_each_empty(monkeypatch):
    _device(monkeypatch, _vllm(monkeypatch))
    facts = sm.facts_of(_engine(_Served(), None))
    assert (facts["cudagraph_capture_sizes"], facts["attention"], facts["moe"]) == ([], [], {})


def test_the_schedule_is_the_workloads_under_the_idle_gap_rule_and_spans_to_its_last_token():
    requests = [{"prompt_token_ids": [5, 6, 7], "max_tokens": 3, "arrive_step": 0}, {"prompt_token_ids": [8], "max_tokens": 6, "arrive_step": 1},
                {"prompt_token_ids": [9, 9], "max_tokens": 2, "arrive_step": 20}]
    assert sm.schedule_of({"requests": requests}) == ([
        {"request_id": "r0", "arrive": 0, "prompt_tokens": 3, "max_tokens": 3}, {"request_id": "r1", "arrive": 1, "prompt_tokens": 1, "max_tokens": 6},
        {"request_id": "r2", "arrive": 7, "prompt_tokens": 2, "max_tokens": 2}], 9)
    assert sm.schedule_of({"requests": requests[:2]})[1] == 7


class _Check:
    """`ServedCheck` as `check_committed` drives it: what it was made with, and each step in the order it came."""

    made: list[_Check] = []

    def __init__(self, build: dict[str, Any], weight: Any, **kw: Any) -> None:
        self.build, self.weight, self.kw = build, weight, kw
        self.steps: list[served_check.Step] = []
        _Check.made.append(self)

    def step(self, st: served_check.Step) -> None:
        self.steps.append(st)

    def report(self) -> dict[str, Any]:
        return {"ok": True, "steps_checked": [st.step for st in self.steps]}


def _committer(steps: dict[int, dict[str, torch.Tensor]], rows: dict[int, Any]) -> SimpleNamespace:
    """A committer that retained ``steps``' values, each step's in one block of bytes at the offsets its layout names, and
    recorded the row order of the steps in ``rows``."""
    layouts, blocks = {}, {}
    for s, values in steps.items():
        metas, raw, off = [], [], 16
        for name, t in values.items():
            b = t.contiguous().view(torch.uint8).reshape(-1)
            metas.append(SimpleNamespace(name=name, dev_off=-1, stream_off=off, nbytes=b.numel(), dtype=str(t.dtype), shape=tuple(t.shape)))
            raw.append(b)
            off += b.numel()
        blocks[s] = (torch.cat([torch.zeros(16, dtype=torch.uint8), *raw]), off)
        layouts[s] = metas
    return SimpleNamespace(_layouts=layouts, _gpu_blocks=blocks, step_rows=rows)


@pytest.mark.parametrize(("given", "used"), [({}, {"rows": served_check.ROWS, "log": None}), ({"rows": 3, "log": print}, {"rows": 3, "log": print})])
def test_the_check_walks_the_committed_steps_in_order_each_with_its_values_words_and_its_requests_by_their_logical_ids(monkeypatch, given, used):
    monkeypatch.setattr(_Check, "made", [])
    monkeypatch.setattr(served_check, "ServedCheck", _Check)
    hidden, ids = torch.tensor(VALUES, dtype=BF16), torch.tensor([4, 9], dtype=torch.int64)
    com = _committer({2: {"h": hidden * 2, "ids": ids + 1}, 0: {"ids": ids, "h": hidden}, 1: {"h": hidden}},
                     {0: [("req-b-0123abcd", 0, 1), ("req-a", 1, 2)], 2: [("a stranger", 0, 2)]})
    build, weight, prompts = {"program_digest": "d"}, (lambda name: name), {"r0": [1, 2]}
    report = sm.check_committed(build, com, weight, prompts, {"req-a": "r0", "req-b": "r1"}, seed=11, **given)
    check, = _Check.made
    assert report == {"ok": True, "steps_checked": [0, 1, 2]}
    assert check.build is build and check.weight is weight
    assert check.kw == {"prompts": prompts, "seed": 11, **used} and check.kw["prompts"] is prompts
    assert [st.rows for st in check.steps] == [[("r1", 0, 1), ("r0", 1, 2)], [], [("a stranger", 0, 2)]]
    assert [list(st.values) for st in check.steps] == [["ids", "h"], ["h"], ["h", "ids"]]
    assert check.steps[0].values["h"].dtype == np.uint16 and check.steps[0].values["h"].tolist() == [[0x3F80, 0xC000, 0x3F00], [0x0000, 0x4040, 0x8000]]
    assert check.steps[2].values["h"].tolist() == [[0x4000, 0xC080, 0x3F80], [0x0000, 0x40C0, 0x8000]]
    assert check.steps[0].values["ids"].tolist() == [4, 9] and check.steps[2].values["ids"].tolist() == [5, 10]


@pytest.mark.parametrize("rows", [{}, {0: None}, {0: []}, {0: ()}])
def test_a_step_the_committer_recorded_no_row_order_for_is_checked_with_no_request_rows(monkeypatch, rows):
    monkeypatch.setattr(_Check, "made", [])
    monkeypatch.setattr(served_check, "ServedCheck", _Check)
    com = _committer({0: {"ids": torch.tensor([4, 9], dtype=torch.int64)}}, rows)
    sm.check_committed({"program_digest": "d"}, com, None, {}, {"req-a": "r0"}, seed=None)
    check, = _Check.made
    assert [st.rows for st in check.steps] == [[]] and type(check.steps[0].rows) is list
