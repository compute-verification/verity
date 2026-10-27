"""The transformers 5.x *plumbing* spellings found on the real classes,
each reproduced here as a plain torch module (no `transformers` import) with the op pattern the exported graph carries, checked
for acceptance (and evaluation against torch CPU where the pattern computes something) AND a neighbouring negative that must
stay refused. Op forms verified against `torch.export` of `transformers 5.16.1` `LlamaForCausalLM` / `GPT2LMHeadModel` (sdpa, f32)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.derive import Unsupported, derive, derive_report
from verity.ml.definitions.numerics.serve3_reference import materialize
from verity.ml.circuits.types import Array, Value

REF = dict(mode="reference")


# ---------------------------------------------------------------------------------------------------------
# evaluation against torch CPU (same host): program inputs = the example tensors' words, then the weights record
# ---------------------------------------------------------------------------------------------------------


def _words(t: torch.Tensor) -> list[int]:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return (t.view(torch.int16).numpy().astype(np.int64) & 0xFFFF).ravel().tolist()
    if t.dtype == torch.float32:
        return (t.view(torch.int32).numpy().astype(np.int64) & 0xFFFFFFFF).ravel().tolist()
    return [int(x) & 0xFFFFFFFFFFFFFFFF for x in t.to(torch.int64).ravel().tolist()]


def _leaf_width(T) -> int:
    while isinstance(T, Array):
        T = T.elem
    assert isinstance(T, Value), T
    return T.w


def _evaluate_words(rep, module: nn.Module, inputs: tuple, extra: dict | None = None) -> tuple[list[int], int]:
    """The program's root output words (and their width), materialised gate by gate on this CPU."""
    P = rep.program
    named = {**dict(module.named_parameters(remove_duplicate=False)), **dict(module.named_buffers(remove_duplicate=False)), **(extra or {})}
    words = [_words(x) for x in inputs]
    if rep.provenance["weights"]:  # the weights record parameter exists only when the module has leaves / derived tables
        wwords: list[int] = []
        for w in rep.provenance["weights"]:
            wwords += _words(named[w["name"]])
        words.append(wwords)
    assert sum(len(x) for x in words) == P.input_gates, (sum(len(x) for x in words), P.input_gates)
    tab = materialize(P, words)
    outs = P.circuit.root_outputs()
    return [int(tab[g]) for g in outs], P.circuit.gate(outs[0]).prim.ret.w


def _evaluate(rep, module: nn.Module, inputs: tuple, extra: dict | None = None) -> torch.Tensor:
    got, w = _evaluate_words(rep, module, inputs, extra)
    bits = np.array(got, dtype=np.uint32)
    if w == 32:
        return torch.from_numpy(bits.astype(np.int32)).view(torch.float32)
    return torch.from_numpy(bits.astype(np.uint16).view(np.int16)).view(torch.bfloat16).float()


def _close(rep, module, inputs, extra=None, rtol=1e-5, atol=1e-5):
    got = _evaluate(rep, module, inputs, extra)
    with torch.no_grad():
        want = module(*inputs).reshape(-1)
    assert torch.allclose(got, want.float(), rtol=rtol, atol=atol), (got - want.float()).abs().max()


def _refused(module, inputs, *reasons: str, contains: str | None = None) -> Unsupported:
    r = derive(module, inputs, **REF)
    assert isinstance(r, Unsupported), f"expected a refusal, got a program {r}"
    if reasons:
        assert r.reason in reasons, f"expected {reasons}, got {r.reason}: {r.hint}"
    if contains is not None:
        assert contains in r.hint, r.hint
    return r


def _rand(*shape, dt=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 0.5).to(dt)


# ---------------------------------------------------------------------------------------------------------
# G-RH-1 — `masking_utils.find_packed_sequence_indices` + the mask functions on static positions (diff / ne / cumsum /
# new_ones / __and__ / index / to.dtype_layout): static structure, folded at derivation time by `StaticIndex_v2`
# ---------------------------------------------------------------------------------------------------------


def _hf5_causal_mask(T: int, device) -> torch.Tensor:
    """transformers 5.16 `create_causal_mask` under `torch.export`, op for op: `cache_position = arange(T) + 0`,
    `find_packed_sequence_indices` (diff -> ne -> cumsum), `and_masks(causal_mask_function, packed_sequence_mask_function)`
    vmapped over the index grids (new_ones accumulator, `le`, `index`, `eq`, `.to(dtype=bool, layout=strided, device=…)`),
    `expand`. Result: bool [1, 1, T, T], allowed iff kv <= q (one unpadded sequence)."""
    pos = (torch.arange(T) + 0).unsqueeze(0)
    first = pos[:, :1] - 1
    packed = (torch.diff(pos, prepend=first, dim=-1) != 1).cumsum(-1)
    b = torch.arange(1)[:, None, None, None]
    q = (torch.arange(T) + 0)[None, None, :, None]
    kv = (torch.arange(T) + 0)[None, None, None, :]
    # `mask(...).to(result.device)` (masking_utils.and_masks) — export records it as `aten.to.dtype_layout(dtype=bool, layout=strided, device=…)`
    m = q.new_ones((), dtype=torch.bool) & (kv <= q).to(device)
    m = m & (packed[b, kv] == packed[b, q]).to(device)
    return m.expand(1, -1, T, T)


class HF5SdpaAttention(nn.Module):
    """`sdpa_attention_forward` as transformers 5.16 exports it for one unpadded prompt: the static bool causal mask
    (G-RH-8) built by the masking utilities (G-RH-1), `scale=head_dim**-0.5` passed explicitly (G-RH-9), `repeat_kv`
    (mask present => `use_gqa_in_sdpa` is False)."""

    def __init__(self, NH: int, KVH: int, gqa: str = "repeat_kv") -> None:
        super().__init__()
        self.NH, self.KVH, self.gqa = NH, KVH, gqa

    def forward(self, q, k, v):  # q [1, NH, T, D]; k, v [1, KVH, T, D]
        T, D = q.shape[-2], q.shape[-1]
        mask = _hf5_causal_mask(T, q.device)
        kw = {}
        if self.KVH != self.NH:
            if self.gqa == "repeat_kv":
                n_rep = self.NH // self.KVH
                k = k[:, :, None, :, :].expand(1, self.KVH, n_rep, T, D).reshape(1, self.NH, T, D)
                v = v[:, :, None, :, :].expand(1, self.KVH, n_rep, T, D).reshape(1, self.NH, T, D)
            else:
                kw = {"enable_gqa": True}
        return F.scaled_dot_product_attention(q, k, v, mask, scale=D ** -0.5, **kw)


def test_grh1_static_mask_plumbing_is_folded_and_the_sdpa_binds():
    m = HF5SdpaAttention(NH=4, KVH=2)
    q, k, v = _rand(1, 4, 5, 8, seed=1), _rand(1, 2, 5, 8, seed=2), _rand(1, 2, 5, 8, seed=3)
    rep = derive_report(m, (q, k, v), **REF, name="HF5Sdpa")
    assert rep.ok, rep.unsupported
    ops = {a["op"] for a in rep.provenance["applications"] if a["rule"] == "StaticIndex_v2"}
    assert {"aten.diff.default", "aten.cumsum.default", "aten.ne.Scalar", "aten.new_ones.default", "aten.__and__.Tensor",
            "aten.index.Tensor", "aten.to.dtype_layout"} <= ops, sorted(ops)
    sd = [a for a in rep.provenance["applications"] if a["rule"] == "Sdpa[reference]_v2"]
    assert len(sd) == 1
    pc = sd[0]["preconditions_checked"]
    assert pc["causal"].startswith("static bool attn_mask verified") and pc["scale"].endswith("== D^-0.5 as f32") and pc["KVH"] == 4
    # the mask never became gates: the only gate-bearing rule is the attention
    assert set(rep.provenance["rules_applied"]) <= {"StaticIndex_v2", "Sdpa[reference]_v2", "Reshape_v1", "Expand_v1", "IdentityView_v1",
                                                    "SliceSelect_v1", "Permute_v1"}, rep.provenance["rules_applied"]
    _close(rep, m, (q, k, v))


def test_grh1_negative_diff_or_new_ones_on_a_value_is_not_static():
    class DiffValue(nn.Module):
        def forward(self, x):
            return torch.diff(x, dim=-1)

    class NewOnesValue(nn.Module):
        def forward(self, x):
            return x * x.new_ones((), dtype=x.dtype)

    class StaticIntoValue(nn.Module):  # a folded static index read as a value
        def forward(self, x):
            return x + torch.diff(torch.arange(x.shape[-1] + 1), dim=-1).to(x.dtype)

    r = _refused(DiffValue(), (torch.empty(2, 6),), "decomposition", "numerics_unregistered")
    assert "StaticIndex_v2: result dtype torch.float32 is not integral" in r.hint, r.hint
    _refused(NewOnesValue(), (torch.empty(2, 6),), "decomposition", "numerics_unregistered")
    _refused(StaticIntoValue(), (torch.empty(2, 6),), "numerics_unregistered", "decomposition", "side_input")


# ---------------------------------------------------------------------------------------------------------
# G-RH-2 — `.to(device=…, dtype=…)` (`aten.to.device`): the device is not semantics; the dtype is the whole meaning
# ---------------------------------------------------------------------------------------------------------


class ScaleByBuffer(nn.Module):
    def __init__(self, dt_buf: torch.dtype, dt_to: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("w", torch.linspace(0.5, 1.5, 8).to(dt_buf), persistent=False)
        self.dt_to = dt_to

    def forward(self, x):
        # (explicit expand: broadcasting is not a bound rule; the `to.device` is the op under test)
        return x * self.w[None, :].to(device=x.device, dtype=self.dt_to).expand(x.shape[0], -1)


def test_grh2_to_device_same_dtype_is_a_view():
    m = ScaleByBuffer(torch.float32, torch.float32)
    x = _rand(3, 8)
    rep = derive_report(m, (x,), **REF, name="ToDeviceView")
    assert rep.ok, rep.unsupported
    apps = {a["op"]: a["rule"] for a in rep.provenance["applications"]}
    assert "aten.to.device" not in apps or apps["aten.to.device"] == "ViewDtype_v2"  # a view records no application
    assert set(rep.provenance["rules_applied"]) == {"Elementwise[reference]_v3"}, rep.provenance["rules_applied"]
    _close(rep, m, (x,))


def test_grh2_to_device_with_a_dtype_change_is_the_cast_kind():
    m = ScaleByBuffer(torch.bfloat16, torch.float32)
    x = _rand(3, 8)
    rep = derive_report(m, (x,), **REF, name="ToDeviceCast")
    assert rep.ok, rep.unsupported
    casts = [a for a in rep.provenance["applications"] if a["rule"] == "CastF32Bf16_v2"]
    assert len(casts) == 1 and casts[0]["op"] == "aten.to.device" and casts[0]["preconditions_checked"]["cast"] == "bf16_to_f32"
    _close(rep, m, (x,))


def test_grh2_negative_to_device_to_an_unregistered_dtype_is_refused():
    class ToF64(nn.Module):
        def forward(self, x):
            return (x.to(device=x.device, dtype=torch.float64) * 2.0).to(torch.float32)

    r = _refused(ToF64(), (torch.empty(2, 4),), "numerics_unregistered")
    assert "to.device" in str(r.node), r


def test_grh2_static_mask_to_device_stays_static():
    """`_preprocess_mask_arguments`: `attention_mask.to(device=…, dtype=torch.bool)` on a mask built in forward."""
    class MaskToDevice(nn.Module):
        def forward(self, q, k, v):
            T = q.shape[-2]
            am = torch.ones(1, T, dtype=torch.int64).to(device=q.device, dtype=torch.bool)  # 2-D padding mask, all ones
            causal = torch.ones(T, T, dtype=torch.bool).tril()[None, None] & am[:, None, None, :]
            return F.scaled_dot_product_attention(q, k, v, causal)

    q, k, v = _rand(1, 2, 4, 8, seed=4), _rand(1, 2, 4, 8, seed=5), _rand(1, 2, 4, 8, seed=6)
    m = MaskToDevice()
    rep = derive_report(m, (q, k, v), **REF, name="MaskToDevice")
    assert rep.ok, rep.unsupported
    assert "aten.to.device" in {a["op"] for a in rep.provenance["applications"] if a["rule"] == "StaticIndex_v2"}
    _close(rep, m, (q, k, v))


# ---------------------------------------------------------------------------------------------------------
# G-RH-4 — `nn.Dropout` in eval mode: `aten.dropout(x, p, train=False)` is the identity
# ---------------------------------------------------------------------------------------------------------


class DropMLP(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return self.drop(F.relu(self.fc(self.drop(x))))


def test_grh4_dropout_in_eval_mode_is_the_identity():
    m = DropMLP(0.1).eval()
    x = _rand(3, 8)
    rep = derive_report(m, (x,), **REF, name="DropEval")
    assert rep.ok, rep.unsupported
    d = [a for a in rep.provenance["applications"] if a["rule"] == "DropoutEval_v1"]
    assert len(d) == 2 and all(a["preconditions_checked"] == {"train": False, "p": 0.1} for a in d)
    _close(rep, m, (x,))


def test_grh4_negative_dropout_in_training_mode_is_a_side_input():
    m = DropMLP(0.0).train()  # even p = 0: the module is in training mode, `train=True` is what the graph carries
    _refused(m, (torch.empty(3, 8),), "side_input")


# ---------------------------------------------------------------------------------------------------------
# G-RH-6 — rotary positions `arange(T) + past_seen_tokens` (a scalar add on the static positions): the table's pre-scan
# looks through `+`/`-` of a Python int; the positions bound are the folded StaticIndex values
# ---------------------------------------------------------------------------------------------------------


class RopeHF5(nn.Module):
    def __init__(self, D: int, offset: int = 0, spelling: str = "add") -> None:
        super().__init__()
        self.rotary_emb = ex.HFRotaryEmbedding(D)
        self.offset, self.spelling = offset, spelling

    def forward(self, q):  # [1, NH, T, D]
        T = q.shape[2]
        if self.spelling == "add":
            pos = (torch.arange(T) + self.offset).unsqueeze(0)  # transformers 5.x: `cache_position = arange(...) + past_seen_tokens`
        elif self.spelling == "sub":
            pos = (torch.arange(self.offset, T + self.offset) - self.offset).unsqueeze(0)
        else:
            pos = (torch.arange(T) * 2).unsqueeze(0)  # static too, but not the add/sub pre-scan form
        cos, sin = self.rotary_emb(q, pos)
        qe, _ = ex.hf_apply_rotary_pos_emb(q, q, cos, sin)
        return qe


def _rope_tables(m: nn.Module, rep, dt: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """The derived tables as the recorded derivation says: the module's own rotary on the recorded positions."""
    app = next(a for a in rep.provenance["applications"] if a["rule"] == "HFRotaryTable[reference]_v4")
    pos = torch.tensor(app["preconditions_checked"]["positions"], dtype=torch.int64).unsqueeze(0)
    with torch.no_grad():
        cos, sin = m.rotary_emb(torch.zeros(1, dtype=dt), pos)
    return {"rotary_emb.cos_table": cos[0], "rotary_emb.sin_table": sin[0]}


def _derived_tables(m: nn.Module, rep) -> dict[str, torch.Tensor]:
    """Every declared derived table (any number of rotary modules / applications): the values a verifier supplies = the source's own
    rotary module — with its own, possibly bf16, `inv_freq` buffer — on the recorded positions, in the recorded table dtype. This is
    exactly the recorded derivation (`source_buffer`, `source_dtype`, `source_cast`, `positions`, `dtype`)."""
    mods = dict(m.named_modules())
    apps = [a for a in rep.provenance["applications"] if a["rule"].startswith("HFRotaryTable")]
    out: dict[str, torch.Tensor] = {}
    for w in rep.provenance["weights"]:
        if w["kind"] != "derived_table" or w["derivation"]["table"] != "cos":
            continue
        cn = w["name"]
        app = next(a for a in apps if a["preconditions_checked"]["fields"][0] == cn)
        der = w["derivation"]
        rot = mods[der["source_buffer"].rsplit(".", 1)[0]]
        assert str(rot.inv_freq.dtype).removeprefix("torch.").replace("bfloat16", "bf16").replace("float32", "f32") == der["source_dtype"]
        pos = torch.tensor(app["preconditions_checked"]["positions"], dtype=torch.int64).unsqueeze(0)
        tdt = torch.float32 if der["dtype"] == "f32" else torch.bfloat16
        with torch.no_grad():
            cos, sin = rot(torch.zeros(1, dtype=tdt), pos)
        out[cn], out[app["preconditions_checked"]["fields"][1]] = cos[0], sin[0]
    return out


@pytest.mark.parametrize("offset,spelling", [(0, "add"), (3, "add"), (2, "sub")])
def test_grh6_positions_through_a_scalar_add_bind_the_rotary_table(offset, spelling):
    m = RopeHF5(8, offset, spelling)
    q = _rand(1, 2, 5, 8, seed=7)
    rep = derive_report(m, (q,), **REF, name="RopeHF5")
    assert rep.ok, rep.unsupported
    tbl = [a for a in rep.provenance["applications"] if a["rule"] == "HFRotaryTable[reference]_v4"]
    want_pos = [i + (offset if spelling == "add" else 0) for i in range(5)]
    assert len(tbl) == 1 and tbl[0]["preconditions_checked"]["positions"] == want_pos
    ws = {w["name"]: w for w in rep.provenance["weights"]}
    assert ws["rotary_emb.cos_table"]["kind"] == "derived_table" and ws["rotary_emb.cos_table"]["shape"] == [5, 8]
    got = _evaluate(rep, m, (q,), _rope_tables(m, rep))
    with torch.no_grad():
        want = m(q).reshape(-1)
    assert torch.equal(got, want)  # RoPE is bit-exact vs torch CPU (as before)


def test_grh6_negative_positions_through_a_multiply_are_not_the_pattern():
    r = _refused(RopeHF5(8, 0, "mul"), (torch.empty(1, 2, 5, 8),), "decomposition", "numerics_unregistered")
    assert "HFRotaryTable[reference]_v4: not the HF rotary table pattern" in r.hint, r.hint


def test_grh6_negative_positions_that_are_a_value_stay_refused():
    class RopeInputPos(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = ex.HFRotaryEmbedding(8)

        def forward(self, q, pos):
            cos, sin = self.rotary_emb(q, pos.unsqueeze(0))
            return ex.hf_apply_rotary_pos_emb(q, q, cos, sin)[0]

    # positions that are an input value never reach the table (the source scan finds a placeholder, not an arange)
    r = _refused(RopeInputPos(), (torch.empty(1, 2, 5, 8), torch.empty(5, dtype=torch.int64)), "numerics_unregistered")
    assert "positions are a value" in r.hint, r.hint


# ---------------------------------------------------------------------------------------------------------
# G-RH-8 — SDPA with an explicit static bool causal mask (is_causal=False) is the causal attention; anything else is not
# ---------------------------------------------------------------------------------------------------------


class SdpaMask(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, q, k, v):
        T = q.shape[-2]
        if self.kind == "causal":
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.kind == "tril":
            m = torch.ones(T, T, dtype=torch.bool).tril()
        elif self.kind == "tril4d":
            m = torch.ones(T, T, dtype=torch.bool).tril()[None, None].expand(1, 1, T, T)
        elif self.kind == "triu":
            m = torch.ones(T, T, dtype=torch.bool).triu()
        elif self.kind == "all":
            m = torch.ones(T, T, dtype=torch.bool)
        elif self.kind == "window":
            m = torch.ones(T, T, dtype=torch.bool).tril() & ~torch.ones(T, T, dtype=torch.bool).tril(-2)
        return F.scaled_dot_product_attention(q, k, v, m)


@pytest.mark.parametrize("kind", ["tril", "tril4d"])
@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_grh8_static_causal_bool_mask_binds_and_equals_is_causal(kind, dt):
    q, k, v = _rand(1, 2, 5, 8, dt=dt, seed=8), _rand(1, 2, 5, 8, dt=dt, seed=9), _rand(1, 2, 5, 8, dt=dt, seed=10)
    m = SdpaMask(kind)
    rep = derive_report(m, (q, k, v), **REF, name="SdpaMask")
    assert rep.ok, rep.unsupported
    sd = [a for a in rep.provenance["applications"] if a["rule"] == "Sdpa[reference]_v2"]
    assert len(sd) == 1 and sd[0]["preconditions_checked"]["causal"].startswith("static bool attn_mask verified")
    # the same program as the is_causal=True spelling of the same class: identical digest (the mask is not in the program)
    ref = derive_report(SdpaMask("causal"), (q, k, v), **REF, name="SdpaMask")
    assert ref.ok and ref.flat_digest == rep.flat_digest and ref.digest == rep.digest
    tol = dict(rtol=1e-5, atol=1e-5) if dt == torch.float32 else dict(rtol=1 / 64, atol=1 / 32)
    _close(rep, m, (q, k, v), **tol)


@pytest.mark.parametrize("kind,needle", [("triu", "not the causal pattern"), ("all", "not the causal pattern"), ("window", "not the causal pattern")])
def test_grh8_negative_other_masks_are_refused(kind, needle):
    e = (torch.empty(1, 2, 5, 8),) * 3
    r = _refused(SdpaMask(kind), e, "decomposition", contains=needle)
    assert "scaled_dot_product_attention" in str(r.node), r


def test_grh8_negative_mask_from_an_input_value_is_refused():
    class ValueMask(nn.Module):
        def forward(self, q, k, v, m):
            return F.scaled_dot_product_attention(q, k, v, m)

    e = (torch.empty(1, 2, 5, 8),) * 3 + (torch.empty(5, 5, dtype=torch.bool),)
    _refused(ValueMask(), e, "decomposition", contains="attn_mask that is a value")


# ---------------------------------------------------------------------------------------------------------
# G-RH-9 — an explicit `scale` equal (as f32) to D^-0.5 is the default; any other scale is a different function
# ---------------------------------------------------------------------------------------------------------


class SdpaScale(nn.Module):
    def __init__(self, scale) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)


@pytest.mark.parametrize("D,scale", [(8, 8 ** -0.5), (8, 1 / math.sqrt(8)), (16, 0.25), (16, 16 ** -0.5), (12, 12 ** -0.5)])
def test_grh9_explicit_default_scale_binds_and_is_the_same_program(D, scale):
    q, k, v = _rand(1, 2, 4, D, seed=11), _rand(1, 2, 4, D, seed=12), _rand(1, 2, 4, D, seed=13)
    m = SdpaScale(scale)
    rep = derive_report(m, (q, k, v), **REF, name="SdpaScale")
    assert rep.ok, rep.unsupported
    ref = derive_report(SdpaScale(None), (q, k, v), **REF, name="SdpaScale")
    assert ref.ok and ref.flat_digest == rep.flat_digest  # the scale did not change the program
    sd = [a for a in rep.provenance["applications"] if a["rule"] == "Sdpa[reference]_v2"]
    assert sd[0]["preconditions_checked"]["scale"] == f"{scale} == D^-0.5 as f32"
    _close(rep, m, (q, k, v))


@pytest.mark.parametrize("D,scale", [(8, 0.5), (8, 8 ** -0.5 * (1 + 1e-6)), (16, 0.2500001), (16, 1.0), (8, 0)])
def test_grh9_negative_other_scales_are_refused(D, scale):
    e = (torch.empty(1, 2, 4, D),) * 3
    r = _refused(SdpaScale(scale), e, "decomposition", contains="is not D^-0.5")
    assert f"scale={scale}" in r.hint, r.hint


# ---------------------------------------------------------------------------------------------------------
# G-RH-10 — `enable_gqa=True`: torch repeats each KV head NH/KVH times (repeat_interleave over heads: query head h reads KV head
# h // (NH/KVH)); `EagerAttentionRef{NH, KVH}` has exactly that head mapping, so the kind is called with the [T][KVH][D] caches
# ---------------------------------------------------------------------------------------------------------


class SdpaGqa(nn.Module):
    def __init__(self, enable: bool) -> None:
        super().__init__()
        self.enable = enable

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.enable)


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_grh10_enable_gqa_binds_with_kvh_and_equals_repeat_kv(dt):
    NH, KVH, T, D = 4, 2, 5, 8
    q, k, v = _rand(1, NH, T, D, dt=dt, seed=14), _rand(1, KVH, T, D, dt=dt, seed=15), _rand(1, KVH, T, D, dt=dt, seed=16)
    m = SdpaGqa(True)
    rep = derive_report(m, (q, k, v), **REF, name="SdpaGqa")
    assert rep.ok, rep.unsupported
    sd = [a for a in rep.provenance["applications"] if a["rule"] == "Sdpa[reference]_v2"]
    assert len(sd) == 1 and sd[0]["preconditions_checked"]["KVH"] == KVH and sd[0]["preconditions_checked"]["enable_gqa"] is True
    assert sd[0]["emitted"].startswith(f"EagerAttentionRef_v1{{T=1..{T},NH={NH},KVH={KVH}")
    tol = dict(rtol=1e-5, atol=1e-5) if dt == torch.float32 else dict(rtol=1 / 64, atol=1 / 32)
    got = _evaluate(rep, m, (q, k, v))
    with torch.no_grad():
        want = m(q, k, v).reshape(-1).float()
        # torch's own definition of enable_gqa, spelled out: repeat_interleave over the head axis
        want2 = F.scaled_dot_product_attention(q, k.repeat_interleave(NH // KVH, dim=1), v.repeat_interleave(NH // KVH, dim=1),
                                               is_causal=True).reshape(-1).float()
    if not torch.allclose(want, want2, **tol):  # torch: enable_gqa is repeat_interleave of the caches (kernel choice may differ)
        pytest.xfail(f"torch's enable_gqa and repeat_interleave spellings differ by {(want - want2).abs().max():.3g} in {dt}"
                     " (seen once in a full run on xdist workers, never alone)")
    assert torch.allclose(got, want, **tol), (got - want).abs().max()
    # and the program's value equals the repeat_kv-spelled program's value (same function, KVH=NH program with repeated caches)
    rep2 = derive_report(HF5SdpaAttention(NH, KVH, "repeat_kv"), (q, k, v), **REF, name="SdpaRepeatKv")
    assert rep2.ok, rep2.unsupported
    got2 = _evaluate(rep2, HF5SdpaAttention(NH, KVH, "repeat_kv"), (q, k, v))
    assert torch.equal(got, got2)


def test_grh10_negative_kvh_without_enable_gqa_and_non_dividing_heads():
    q, k, v = torch.empty(1, 4, 5, 8), torch.empty(1, 2, 5, 8), torch.empty(1, 2, 5, 8)
    # KVH=2 != NH=4 without enable_gqa: torch itself rejects the call at export (the rule's own check is the defensive twin)
    r = derive(SdpaGqa(False), (q, k, v), **REF)
    assert isinstance(r, Unsupported), r
    assert r.reason == "export" or (r.reason == "decomposition" and "KVH=2 != NH=4 without enable_gqa" in r.hint), r
    # NH=4, KVH=3: torch itself rejects the call at export
    r = derive(SdpaGqa(True), (torch.empty(1, 4, 5, 8), torch.empty(1, 3, 5, 8), torch.empty(1, 3, 5, 8)), **REF)
    assert isinstance(r, Unsupported), r


def test_hf5_all_three_spellings_together_match_torch():
    """The exact `sdpa_attention_forward` call of transformers 5.16 for a GQA model when the mask is None at runtime
    (`enable_gqa=True`, `scale=` explicit, `is_causal=True`) and with the mask (repeat_kv + static bool mask + scale)."""
    NH, KVH, T, D = 4, 2, 6, 16
    q, k, v = _rand(1, NH, T, D, seed=17), _rand(1, KVH, T, D, seed=18), _rand(1, KVH, T, D, seed=19)

    class NoMask(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, None, is_causal=True, scale=D ** -0.5, enable_gqa=True)

    for m in (NoMask(), HF5SdpaAttention(NH, KVH, "repeat_kv")):
        rep = derive_report(m, (q, k, v), **REF, name=type(m).__name__)
        assert rep.ok, rep.unsupported
        _close(rep, m, (q, k, v))


# ---------------------------------------------------------------------------------------------------------
# DERIVE-3 — G-RH-14: `nn.Embedding(V, H, padding_idx=k)` exports as `aten.embedding(w, ids, k)` (Olmo: pad 1, Gemma: pad 0 — any
# config with `pad_token_id`). ATen's forward is `weight.index_select(0, ids)` for every padding_idx; only `embedding_backward`
# reads it (zero gradient for that row). So the forward IS the plain gather: `Embedding[reference]_v2` binds it, records the value.
# ---------------------------------------------------------------------------------------------------------


class PadEmbed(nn.Module):
    def __init__(self, V: int, H: int, padding_idx, dt: torch.dtype = torch.float32, sparse: bool = False) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(V, H, padding_idx=padding_idx, sparse=sparse, dtype=dt)
        with torch.no_grad():  # every row non-zero, the padding row INCLUDED: the forward reads whatever is stored there
            self.embed_tokens.weight.copy_(_rand(V, H, dt=dt, seed=20))

    def forward(self, ids):
        return self.embed_tokens(ids)


@pytest.mark.parametrize("pad", [0, 1, 5, None])
@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_grh14_embedding_with_padding_idx_is_the_plain_gather(pad, dt):
    V, H = 16, 8
    m = PadEmbed(V, H, pad, dt)
    p = pad if pad is not None else 0
    ids = torch.tensor([3, p, 15, p, 7], dtype=torch.int64)  # the padding id appears twice
    rep = derive_report(m, (ids,), **REF, name="PadEmbed")
    assert rep.ok, rep.unsupported
    emb = [a for a in rep.provenance["applications"] if a["rule"] == "Embedding[reference]_v2"]
    assert len(emb) == 1 and emb[0]["preconditions_checked"]["padding_idx"] == pad and emb[0]["emitted"].startswith("EmbeddingRef_v1{")
    # the same program as padding_idx=None (the argument leaves no trace: same kind, same wiring, same digest)
    ref = derive_report(PadEmbed(V, H, None, dt), (ids,), **REF, name="PadEmbed")
    assert ref.ok and ref.digest == rep.digest and ref.flat_digest == rep.flat_digest
    got = _evaluate(rep, m, (ids,))
    with torch.no_grad():
        want = m(ids).reshape(-1).float()
    assert torch.equal(got, want)  # a gather: bit-exact vs torch CPU, the padding rows included (they are NOT zeroed in forward)


def test_grh14_negative_sparse_embedding_stays_refused():
    r = _refused(PadEmbed(16, 8, 1, sparse=True), (torch.empty(4, dtype=torch.int64),), "decomposition", contains="scale_grad_by_freq / sparse are set")
    assert "embedding" in str(r.node), r


# ---------------------------------------------------------------------------------------------------------
# DERIVE-3 — G-RH-7: `model.to(torch.bfloat16)` casts the `inv_freq` BUFFER with the parameters, so on a real bf16 Llama / Qwen2 / Qwen3
# the rotary's `inv_freq[None, :, None].float()` is a real bf16 -> f32 cast. `HFRotaryTable[reference]_v4` accepts exactly one such cast
# on the buffer path; the derived-Input declaration records the buffer dtype and the cast; the table is then computed from the
# bf16-ROUNDED inv_freq — the computation HF runs for that model (the fixed computation preserved), not a deviation. The cast is
# consumed with the chain (no `CastF32Bf16` gates), and the rule fires at the buffer's cast when export places it first.
# ---------------------------------------------------------------------------------------------------------

_C_BF16 = dict(H=16, NH=2, KVH=1, D=8, I=32, EPS=1e-5, BIAS=False, V=32, NL=2)


def _bf16_model(cls, *args) -> nn.Module:
    torch.manual_seed(0)
    m = cls(*args).to(torch.bfloat16)  # as HF: parameters AND buffers (inv_freq) become bf16
    with torch.no_grad():
        for p in m.parameters():
            p.copy_((torch.randn_like(p, dtype=torch.float32) * 0.3).to(torch.bfloat16))
    return m


def test_grh7_bf16_inv_freq_buffer_binds_the_rotary_table_through_its_cast():
    m = RopeHF5(8, 0, "add").to(torch.bfloat16)
    assert m.rotary_emb.inv_freq.dtype == torch.bfloat16
    q = _rand(1, 2, 5, 8, dt=torch.bfloat16, seed=21)
    rep = derive_report(m, (q,), **REF, name="RopeBf16")
    assert rep.ok, rep.unsupported
    tbl = [a for a in rep.provenance["applications"] if a["rule"] == "HFRotaryTable[reference]_v4"]
    assert len(tbl) == 1
    pc = tbl[0]["preconditions_checked"]
    assert pc["source_dtype"] == "bf16" and "bf16 -> f32" in pc["source_cast"] and "bf16-rounded" in pc["source_cast"]
    assert pc["positions"] == [0, 1, 2, 3, 4] and pc["fired_at"] == "buffer cast"  # export places the buffer's .float() before the positions'
    ws = {w["name"]: w for w in rep.provenance["weights"]}
    assert ws["rotary_emb.inv_freq"]["dtype"] == "bfloat16" and ws["rotary_emb.inv_freq"]["kind"] == "buffer"
    der = ws["rotary_emb.cos_table"]["derivation"]
    assert der["source_dtype"] == "bf16" and der["source_cast"] == pc["source_cast"] and der["dtype"] == "bf16"
    # the buffer's cast is inside the consumed chain: no cast kind fired, the only gate-bearing rule is the RoPE
    assert set(rep.provenance["rules_applied"]) == {"HFRoPE[reference]_v1", "HFRotaryTable[reference]_v4", "StaticIndex_v2"}, rep.provenance["rules_applied"]
    got = _evaluate(rep, m, (q,), _rope_tables(m, rep, torch.bfloat16))
    with torch.no_grad():
        want = m(q).reshape(-1).float()
    assert torch.equal(got, want)  # RoPE is bit-exact vs torch CPU given the tables the source computes from ITS (bf16) buffer
    # what "bf16-rounded inv_freq" means: the buffer is not the f32 one (1/10000^(2i/D) rounded to 8 bits of mantissa)
    assert not torch.equal(m.rotary_emb.inv_freq.float(), RopeHF5(8, 0, "add").rotary_emb.inv_freq)


def test_grh7_f32_inv_freq_records_no_cast_and_the_program_is_unchanged():
    m = RopeHF5(8, 0, "add")
    q = _rand(1, 2, 5, 8, seed=21)
    rep = derive_report(m, (q,), **REF, name="RopeF32")
    assert rep.ok, rep.unsupported
    pc = next(a for a in rep.provenance["applications"] if a["rule"] == "HFRotaryTable[reference]_v4")["preconditions_checked"]
    assert pc["source_dtype"] == "f32" and pc["source_cast"] is None and pc["fired_at"] == "positions cast"
    ws = {w["name"]: w for w in rep.provenance["weights"]}
    assert ws["rotary_emb.cos_table"]["derivation"]["source_cast"] is None
    got = _evaluate(rep, m, (q,), _rope_tables(m, rep))
    with torch.no_grad():
        want = m(q).reshape(-1)
    assert torch.equal(got, want)


@pytest.mark.slow
def test_grh7_bf16_tiny_llama_forward_greedy_token_matches_torch():
    """`HFTinyLM` (HF `LlamaForCausalLM` at the op interface) cast to bf16 as HF does: bf16 weights AND bf16 `inv_freq`."""
    m = _bf16_model(ex.HFTinyLM, _C_BF16, torch.bfloat16)
    assert m.rotary_emb.inv_freq.dtype == torch.bfloat16
    rep = derive_report(m, (torch.empty(4, dtype=torch.int64),), **REF, name="HFTinyLMbf16")
    assert rep.ok, rep.unsupported
    tbl = [a for a in rep.provenance["applications"] if a["rule"] == "HFRotaryTable[reference]_v4"]
    assert len(tbl) == 1 and tbl[0]["preconditions_checked"]["source_dtype"] == "bf16"
    assert not any(a["rule"] == "CastF32Bf16_v2" and "rotary_emb" in str(a["module_path"]) for a in rep.provenance["applications"])
    ws = {w["name"]: w for w in rep.provenance["weights"]}
    assert ws["rotary_emb.cos_table"]["derivation"]["source_dtype"] == "bf16"
    for ids in (torch.tensor([3, 7, 11, 2]), torch.tensor([0, 31, 5, 5]), torch.tensor([9, 9, 9, 9])):
        with torch.no_grad():
            want = m(ids)
        got, w = _evaluate_words(rep, m, (ids,), _derived_tables(m, rep))
        assert w == 32 and got == want.tolist(), (ids, got, want)  # greedy token bit-equal to torch CPU bf16


@pytest.mark.slow
def test_grh7_bf16_tiny_llama_greedy_request_tokens_match_torch():
    m = _bf16_model(ex.HFGreedy, _C_BF16, 2, torch.bfloat16)
    assert m.lm.rotary_emb.inv_freq.dtype == torch.bfloat16
    rep = derive_report(m, (torch.empty(3, dtype=torch.int64),), **REF, name="HFGreedyBf16")
    assert rep.ok, rep.unsupported
    tbl = [a for a in rep.provenance["applications"] if a["rule"] == "HFRotaryTable[reference]_v4"]
    assert len(tbl) == 3 and all(a["preconditions_checked"]["source_dtype"] == "bf16" for a in tbl)
    assert sorted(a["preconditions_checked"]["positions"] for a in tbl) == [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
    for prompt in (torch.tensor([3, 7, 11]), torch.tensor([31, 0, 4])):
        with torch.no_grad():
            want = m(prompt)
        got, _ = _evaluate_words(rep, m, (prompt,), _derived_tables(m, rep))
        assert got == want.tolist(), (prompt, got, want)  # STEPS+1 tokens bit-equal to torch CPU bf16


def test_grh7_negative_f16_inv_freq_is_not_the_pattern():
    """Only the bf16 -> f32 widening (a registered exact kind) is accepted on the buffer path; an f16 buffer is refused at its cast."""
    m = RopeHF5(8, 0, "add").to(torch.float16)
    r = _refused(m, (torch.empty(1, 2, 5, 8, dtype=torch.float16),), "numerics_unregistered", "decomposition")
    assert "HFRotaryTable[reference]_v4: not the HF rotary table pattern" in r.hint, r.hint
    assert "rotary_emb" in str(r.module_path), r


def test_grh7_negative_bf16_buffer_with_positions_that_are_a_value_stays_refused():
    class RopeInputPos(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = ex.HFRotaryEmbedding(8)

        def forward(self, q, pos):
            cos, sin = self.rotary_emb(q, pos.unsqueeze(0))
            return ex.hf_apply_rotary_pos_emb(q, q, cos, sin)[0]

    m = RopeInputPos().to(torch.bfloat16)
    r = _refused(m, (torch.empty(1, 2, 5, 8, dtype=torch.bfloat16), torch.empty(5, dtype=torch.int64)), "numerics_unregistered")
    assert "positions are a value" in r.hint, r.hint


# ---------------------------------------------------------------------------------------------------------
# DERIVE-4 — G-RH-16: `OlmoRotaryEmbedding.forward` returns the f32 `cos` / `sin` WITHOUT the trailing `.to(x.dtype)`. `HFRotaryTable
# [reference]_v4` accepts both terminals — a cast to the model dtype (the cast is the table) or none (the f32 cos|sin is the table) —
# and records which (`terminal_cast`). Structural: the no-cast terminal must be f32; both tables must agree.
# ---------------------------------------------------------------------------------------------------------


class OlmoStyleRotaryEmbedding(ex.HFRotaryEmbedding):
    """transformers `OlmoRotaryEmbedding` at the op interface: same chain, no terminal cast (the layers read the f32 tables)."""

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos, sin


class RopeNoCast(nn.Module):
    def __init__(self, D: int, layers: int = 2, scaling: float = 1.0) -> None:
        super().__init__()
        self.rotary_emb = OlmoStyleRotaryEmbedding(D, attention_scaling=scaling)
        self.layers = layers

    def forward(self, q):  # [1, NH, T, D]; the tables are read by `layers` consumers, as a multi-layer model reads them
        pos = (torch.arange(q.shape[2]) + 0).unsqueeze(0)
        cos, sin = self.rotary_emb(q, pos)
        y = q
        for _ in range(self.layers):
            y = ex.hf_apply_rotary_pos_emb(y, y, cos, sin)[0]
        return y


@pytest.mark.parametrize("layers", [1, 2])
def test_grh16_rotary_table_without_terminal_cast_binds_and_records_it(layers):
    m = RopeNoCast(8, layers)
    q = _rand(1, 2, 5, 8, seed=31)
    rep = derive_report(m, (q,), **REF, name=f"RopeNoCast{layers}")
    assert rep.ok, rep.unsupported
    tbl = [a for a in rep.provenance["applications"] if a["rule"] == "HFRotaryTable[reference]_v4"]
    assert len(tbl) == 1
    pc = tbl[0]["preconditions_checked"]
    assert pc["terminal_cast"].startswith("none") and pc["dtype"] == "f32" and pc["positions"] == [0, 1, 2, 3, 4]
    ws = {w["name"]: w for w in rep.provenance["weights"]}
    assert ws["rotary_emb.cos_table"]["derivation"]["terminal_cast"] == pc["terminal_cast"]
    assert ws["rotary_emb.cos_table"]["derivation"]["dtype"] == "f32"
    # cos / sin are the table Inputs themselves (not consumed): no cast kind, only the RoPE emits gates
    assert set(rep.provenance["rules_applied"]) == {"HFRoPE[reference]_v1", "HFRotaryTable[reference]_v4", "StaticIndex_v2"}, rep.provenance["rules_applied"]
    assert sum(1 for a in rep.provenance["applications"] if a["rule"] == "HFRoPE[reference]_v1") == 2 * layers
    got = _evaluate(rep, m, (q,), _rope_tables(m, rep))
    with torch.no_grad():
        want = m(q).reshape(-1)
    assert torch.equal(got, want)


def test_grh16_terminal_cast_is_still_recorded_as_such_and_the_program_is_unchanged():
    """The Llama spelling (terminal `.to(x.dtype)`) records `cast to model dtype`; its digest is DERIVE-3's under the same name."""
    m = RopeHF5(8, 0, "add")
    rep = derive_report(m, (_rand(1, 2, 5, 8, seed=21),), **REF, name="RopeF32")
    assert rep.ok, rep.unsupported
    pc = next(a for a in rep.provenance["applications"] if a["rule"] == "HFRotaryTable[reference]_v4")["preconditions_checked"]
    assert pc["terminal_cast"] == "cast to model dtype"
    rep_nc = derive_report(RopeNoCast(8, 1), (_rand(1, 2, 5, 8, seed=21),), **REF, name="RopeF32")
    assert rep_nc.ok, rep_nc.unsupported
    # the same gates and Inputs (the terminal cast was never a gate: f32 -> f32); the digests differ only by the root id's source class
    assert (rep_nc.gates, rep_nc.program.input_gates) == (rep.gates, rep.program.input_gates)
    assert rep_nc.digest != rep.digest and rep_nc.root != rep.root


def test_grh16_negative_no_cast_with_scaled_tables_that_disagree_is_not_the_pattern():
    class Disagree(RopeNoCast):
        def forward(self, q):
            pos = (torch.arange(q.shape[2]) + 0).unsqueeze(0)
            cos, sin = self.rotary_emb(q, pos)
            return ex.hf_apply_rotary_pos_emb(q, q, cos, sin.to(torch.bfloat16).to(torch.float32))[0]  # sin cast away and back: not one terminal

    r = _refused(Disagree(8, 1), (torch.empty(1, 2, 5, 8),), "numerics_unregistered", "decomposition")
    assert "HFRotaryTable[reference]_v4: not the HF rotary table pattern" in r.hint, r.hint


# ---------------------------------------------------------------------------------------------------------
# DERIVE-4 — G-RH-17: `GemmaTextScaledWordEmbedding`: `embeds * embed_scale` with `embed_scale` a 0-d BUFFER (`sqrt(H)`).
# `Elementwise[reference]_v3` reads the buffer's one Input leaf N times (zero-stride refs; no gates) and records the operand's
# provenance (field, kind, dtype, value). Not a silent broadcast: a 0-d COMPUTED value, a [1] operand, a Python scalar still refuse.
# ---------------------------------------------------------------------------------------------------------


class ScaledEmbed(nn.Module):
    def __init__(self, V: int, H: int, scale: float, dt: torch.dtype = torch.float32, as_parameter: bool = False) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(V, H, padding_idx=0, dtype=dt)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(_rand(V, H, dt=dt, seed=40))
        s = torch.tensor(scale, dtype=dt)
        if as_parameter:
            self.embed_scale = nn.Parameter(s)
        else:
            self.register_buffer("embed_scale", s, persistent=True)

    def forward(self, ids):
        return self.embed_tokens(ids) * self.embed_scale


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("as_parameter", [False, True])
def test_grh17_zero_d_weight_operand_binds_with_recorded_provenance(dt, as_parameter):
    m = ScaledEmbed(16, 8, 8.0, dt, as_parameter)
    ids = torch.tensor([3, 0, 7, 0, 15, 1])
    rep = derive_report(m, (torch.empty(6, dtype=torch.int64),), **REF, name="ScaledEmbed")
    assert rep.ok, rep.unsupported
    app = next(a for a in rep.provenance["applications"] if a["rule"] == "Elementwise[reference]_v3")
    sc = app["preconditions_checked"]["scalar_operand"]
    assert sc["arg"] == 1 and sc["field"] == "embed_scale" and sc["shape"] == [] and sc["value"] == 8.0
    assert sc["weight_kind"] == ("parameter" if as_parameter else "buffer") and sc["dtype"] == str(dt).removeprefix("torch.")
    assert app["preconditions_checked"]["N"] == 48
    ws = {w["name"]: w for w in rep.provenance["weights"]}
    assert ws["embed_scale"]["shape"] == [] and ws["embed_scale"]["kind"] == sc["weight_kind"]
    assert rep.provenance["weight_roles"]["embed_scale"]["role"] == "mul.scalar"
    got = _evaluate(rep, m, (ids,))
    with torch.no_grad():
        want = m(ids).reshape(-1).float()
    assert torch.equal(got, want)  # the kind's f32/bf16 multiply against the buffer value: bit-exact vs torch CPU


def test_grh17_negative_zero_d_computed_value_stays_refused():
    class ComputedScale(nn.Module):
        def forward(self, x):
            return x * x[0, 0]  # a 0-d VIEW of a value: not a declared weight leaf

    r = _refused(ComputedScale(), (torch.empty(4, 8),), "decomposition", "numerics_unregistered")
    assert "0-d COMPUTED value" in r.hint, r.hint


def test_grh17_negative_shape_1_buffer_and_python_scalar_stay_refused():
    class Shape1(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("s", torch.tensor([8.0]))

        def forward(self, x):
            return x * self.s

    r = _refused(Shape1(), (torch.empty(4, 8),), "decomposition", "numerics_unregistered")
    assert "operands differ in shape (4, 8) vs (1,) (broadcast is not a bound rule)" in r.hint, r.hint

    class PyScalar(nn.Module):
        def forward(self, x):
            return x * 8.0

    r = _refused(PyScalar(), (torch.empty(4, 8),), "numerics_unregistered", "decomposition")
    assert "a Python scalar operand needs a constant kind" in r.hint, r.hint
