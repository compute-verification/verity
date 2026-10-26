"""DeepSeek-V2's decoder layers (`deepseek_v2`; DeepSeek-V2-Lite), op for op as HF transformers' `DeepseekV2DecoderLayer`
computes them with SDPA attention and eager experts (`check_reference` compares them bit for bit on the check window).

Attention is MLA as HF expands it: q_proj (no q LoRA: Lite's), kv_a_proj_with_mqa to the 512-wide latent and one 64-wide
RoPE key every head shares, kv_a_layernorm, kv_b_proj to each head's 128-wide key part and 128-wide value. So the
attention is 16 KV heads of 192-wide keys and 128-wide values. Its RoPE is YaRN's, on interleaved pairs as complex
numbers (`rotary`, `rope`), and the softmax scale carries YaRN's mscale squared.

The router (FP32 softmax, greedy top-k) is never an arm's. An arm's linear steps reach q_proj, kv_a_proj_with_mqa,
kv_b_proj, o_proj, layer 0's dense MLP, the shared experts' gate/up/down and each routed expert's gate_up and down (named
`mlp.experts.E.gate_up_proj` and `mlp.experts.E.down_proj`), each expert on the rows routed to it, as HF calls it. Its
attention steps see Q and K zero-padded to a power of 2 wide (`pad_width`: `rotate`'s Hadamard needs one; S is unchanged) and
V at its 128. SmoothQuant's groups are the layer's (`groups`, `first`): one m per input that linears share, so 1 / m
folds into the norm or the rows before it, and each routed expert's down its own. The LUT softmax is refused: it reads
d^-0.5, not this scale."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from matmul import MM, Run, run_chain
from model import Layer, rms_norm

#: RMSNorm's default eps, which HF's kv_a_layernorm takes (only the layer norms read rms_norm_eps)
LORA_EPS = 1e-6


def pad_width(w: int) -> int:
    """Q's and K's width in an arm's attention: w zero-padded to a power of 2 (V2-Lite's 192 to 256)."""
    return 1 << (w - 1).bit_length()


def get_mscale(scale: float, mscale: float = 1) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def inv_freq(cfg) -> tuple[torch.Tensor, float]:
    """HF's inverse frequencies and attention factor, default or YaRN (`_compute_yarn_parameters`), FP32 on the CPU."""
    r = cfg.rope
    base, dim = r["rope_theta"], cfg.head_dim
    if r["rope_type"] == "default":
        return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)), 1.0
    if r["rope_type"] != "yarn":
        raise ValueError(f"DeepSeek-V2: rope {r['rope_type']}")
    factor, omax = r["factor"], r["original_max_position_embeddings"]
    mscale, mscale_all = r.get("mscale"), r.get("mscale_all_dim")
    att = r.get("attention_factor")
    if att is None:
        att = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all)) if mscale and mscale_all \
            else get_mscale(factor)
    fast, slow = r.get("beta_fast") or 32, r.get("beta_slow") or 1

    def corr(n):
        return (dim * math.log(omax / (n * 2 * math.pi))) / (2 * math.log(base))

    low, high = corr(fast), corr(slow)
    if r.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    pos = base ** (torch.arange(0, dim, 2).to(dtype=torch.float) / dim)
    extra = 1 - ramp
    return (1.0 / (factor * pos)) * (1 - extra) + (1.0 / pos) * extra, att


def rotary(cfg, T: int, device) -> tuple[torch.Tensor, None]:
    """HF's `DeepseekV2RotaryEmbedding` table for positions 0..T-1: e^(i pos f) times the attention factor, complex64
    [1, T, head_dim / 2]. It stands in the engine's (cos, sin) as (table, None)."""
    inv, att = inv_freq(cfg)
    pos = torch.arange(T, device=device)[None]
    freqs = pos[..., None].float() * inv.to(device=device, dtype=torch.float)
    return torch.polar(torch.ones_like(freqs), freqs) * att, None


def rope(x: torch.Tensor, cis: torch.Tensor) -> torch.Tensor:
    """HF's `apply_rotary_emb` on one tensor ([B, H, T, D]; cis [B, T, D / 2])."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * cis.unsqueeze(1).to(xc.device)).flatten(3).type_as(x)


def scaling(cfg) -> float:
    """HF's softmax scale: qk_head_dim^-0.5, times YaRN's mscale(factor, mscale_all_dim)^2 (`yarn_apply_mscale`)."""
    d = cfg.dsv2
    s = (d["qk_nope_head_dim"] + d["qk_rope_head_dim"]) ** (-0.5)
    r = cfg.rope
    if r.get("rope_type", "default") != "default" and r.get("mscale_all_dim", 0):
        m = get_mscale(r["factor"], r["mscale_all_dim"])
        s = s * m * m
    return s


class Dsv2Layer(Layer):
    def __init__(self, cfg, li: int, tensors: dict, device):
        self.cfg, self.li, self.g = cfg, li, cfg.dsv2
        g = self.g
        if g["q_lora_rank"] is not None:
            raise ValueError("DeepSeek-V2: the bench runs q_lora_rank = null (V2-Lite)")
        if g["topk_method"] != "greedy" or g["scoring_func"] != "softmax" or (g["moe_layer_freq"] or 1) != 1:
            raise ValueError("DeepSeek-V2: the bench runs greedy softmax routing in every layer past the dense ones")
        p = f"model.layers.{li}."
        t = {n[len(p):]: x.to(device=device, dtype=torch.bfloat16) for n, x in tensors.items()}
        self.sparse = li >= g["first_k_dense_replace"]
        E = g["n_routed_experts"]
        if self.sparse:
            for e in range(E):
                q = f"mlp.experts.{e}."
                t[q + "gate_up_proj.weight"] = torch.cat([t.pop(q + "gate_proj.weight"), t.pop(q + "up_proj.weight")])
        self.t = t
        self.scaling = scaling(cfg)
        self.groups = {"self_attn.q_proj": "qkv", "self_attn.kv_a_proj_with_mqa": "qkv", "self_attn.kv_b_proj": "kvb",
                       "self_attn.o_proj": "o"}
        if self.sparse:
            self.groups.update({"mlp.shared_experts.gate_proj": "moe", "mlp.shared_experts.up_proj": "moe",
                                "mlp.shared_experts.down_proj": "shared_down"})
            for e in range(E):
                self.groups[f"mlp.experts.{e}.gate_up_proj"] = "moe"
                self.groups[f"mlp.experts.{e}.down_proj"] = f"down.{e}"
        else:
            self.groups.update({"mlp.gate_proj": "gu", "mlp.up_proj": "gu", "mlp.down_proj": "down"})
        #: the linear whose input stands for its group's (calibration reads it): the shared expert's gate sees every row
        self.first: dict = {}
        for n, grp in self.groups.items():
            self.first.setdefault(grp, n)

    def linear(self, name: str, x: torch.Tensor, run: Run | None, collect=None) -> torch.Tensor:
        w = self.t[f"{name}.weight"]
        if collect is not None:
            collect(name, x.reshape(-1, x.shape[-1]).float())
        steps = run.steps_for("linear", name) if run is not None else []
        if not steps:
            return F.linear(x, w)
        mm = MM("linear", name, run, x.reshape(-1, x.shape[-1]).float(), w, maps=(lambda t: t.float(),),
                extra={"weight64": w.double})
        y = run_chain(steps, mm)
        return y.to(x.dtype).reshape(*x.shape[:-1], w.shape[0])

    def forward(self, h: torch.Tensor, cos, sin, run: Run | None = None, collect=None) -> torch.Tensor:
        c, g = self.cfg, self.g
        nope, rp, vd, lora = g["qk_nope_head_dim"], g["qk_rope_head_dim"], g["v_head_dim"], g["kv_lora_rank"]
        edited = run is not None and bool(run.steps_for("qk", "qk") or run.steps_for("pv", "pv"))
        if edited and any(s.kw.get("softmax") == "lut" for s in run.arm.attention):
            raise ValueError("DeepSeek-V2: the LUT softmax reads d^-0.5; this attention's scale is YaRN's")
        B, T, _ = h.shape
        a = "self_attn."
        res = h
        x = rms_norm(h, self.t["input_layernorm.weight"], c.eps)
        q = self.linear(a + "q_proj", x, run, collect).view(B, T, -1, nope + rp).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [nope, rp], dim=-1)
        kv_nope, k_pe = torch.split(self.linear(a + "kv_a_proj_with_mqa", x, run, collect), [lora, rp], dim=-1)
        k_nope = rms_norm(kv_nope, self.t[a + "kv_a_layernorm.weight"], LORA_EPS).view(B, 1, T, lora)
        k_pe = k_pe.view(B, 1, T, rp)
        q_pe, k_pe = rope(q_pe, cos), rope(k_pe, cos)
        query = torch.cat((q_nope, q_pe), dim=-1)
        kv = self.linear(a + "kv_b_proj", k_nope, run, collect).view(B, T, -1, nope + vd).transpose(1, 2)
        k_nope, value = torch.split(kv, [nope, vd], dim=-1)
        key = kv.new_empty(*kv.shape[:-1], nope + rp)
        key[..., :nope].copy_(k_nope)
        key[..., nope:].copy_(k_pe.expand(-1, k_nope.shape[1], -1, -1))
        pad = (0, pad_width(nope + rp) - (nope + rp))
        if collect is not None:
            collect("keys", F.pad(key, pad))
        if edited:
            o = self.edited_attention(F.pad(query, pad), F.pad(key, pad), value, run)
        else:
            o = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, scale=self.scaling,
                                               is_causal=True).transpose(1, 2).contiguous()
        o = o.reshape(B, T, -1).contiguous()
        h = res + self.linear(a + "o_proj", o, run, collect)
        res = h
        x = rms_norm(h, self.t["post_attention_layernorm.weight"], c.eps)
        return res + (self.moe(x, run, collect) if self.sparse else self.mlp("mlp.", x, run, collect))

    def mlp(self, p: str, x: torch.Tensor, run, collect) -> torch.Tensor:
        m = F.silu(self.linear(p + "gate_proj", x, run, collect)) * self.linear(p + "up_proj", x, run, collect)
        return self.linear(p + "down_proj", m, run, collect)

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """HF's `DeepseekV2TopkRouter` (greedy): the top-k experts of each row and their weights."""
        g = self.g
        logits = F.linear(x.type(torch.float32), self.t["mlp.gate.weight"].type(torch.float32))
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        w, idx = torch.topk(scores, k=g["num_experts_per_tok"], dim=-1, sorted=False)
        return idx, w * g["routed_scaling_factor"]

    def moe(self, x: torch.Tensor, run, collect) -> torch.Tensor:
        """HF's `DeepseekV2Moe` with eager experts: the routed experts on their rows, then the shared experts."""
        shape = x.shape
        flat = x.view(-1, shape[-1])
        idx, w = self.route(flat)
        E = self.g["n_routed_experts"]
        out = torch.zeros_like(flat)
        mask = F.one_hot(idx, num_classes=E + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = int(e[0])
            if e == E:
                continue
            pos, tok = torch.where(mask[e])
            gate, up = self.linear(f"mlp.experts.{e}.gate_up_proj", flat[tok], run, collect).chunk(2, dim=-1)
            y = self.linear(f"mlp.experts.{e}.down_proj", F.silu(gate) * up, run, collect)
            out.index_add_(0, tok, (y * w[tok, pos, None]).to(out.dtype))
        return out.view(*shape) + self.mlp("mlp.shared_experts.", x, run, collect)


def hf_state(layer: Dsv2Layer) -> dict:
    """The layer's weights as HF's decoder layer holds them: the routed experts stacked."""
    sd = {n: t for n, t in layer.t.items() if ".experts." not in n}
    if layer.sparse:
        E = layer.g["n_routed_experts"]
        sd["mlp.experts.gate_up_proj"] = torch.stack([layer.t[f"mlp.experts.{e}.gate_up_proj.weight"] for e in range(E)])
        sd["mlp.experts.down_proj"] = torch.stack([layer.t[f"mlp.experts.{e}.down_proj.weight"] for e in range(E)])
    return sd


def check_reference(gates, cfg, snapshot, layer: Dsv2Layer, h: torch.Tensor, cos, sin, out: torch.Tensor) -> None:
    """The engine's bf16 layer output equals HF's `DeepseekV2DecoderLayer` on the same input and weights, bit for bit,
    with HF's RoPE table and its causal SDPA (no mask, as `create_causal_mask` leaves SDPA)."""
    try:
        import transformers
        from transformers.models.deepseek_v2 import modeling_deepseek_v2 as hf
    except ImportError:
        gates.skip("hf_reference", "transformers with deepseek_v2 is not importable")
        return
    hc = transformers.AutoConfig.from_pretrained(str(snapshot))
    hc._attn_implementation = "sdpa"
    hc._experts_implementation = "eager"
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            hl = hf.DeepseekV2DecoderLayer(hc, layer.li)
    finally:
        torch.set_default_dtype(prev)
    hl.load_state_dict(hf_state(layer), strict=True, assign=True)
    hl.eval()
    dev = h.device
    rot = hf.DeepseekV2RotaryEmbedding(hc).to(dev)
    B, T, _ = h.shape
    pos = torch.arange(T, device=dev)[None].expand(B, -1)
    with torch.no_grad():
        cis = rot(h, pos)
        want = hl(h, attention_mask=None, position_ids=pos, position_embeddings=cis)
    want = want[0] if isinstance(want, tuple) else want
    ok = bool(torch.equal(cis, cos.expand(B, -1, -1)) and torch.equal(want, out))
    gates.record("hf_reference", ok, f"layer {layer.li}", None if ok else float((want.float() - out.float()).abs().max()))
    del hl
