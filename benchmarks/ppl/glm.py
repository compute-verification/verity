"""GLM-5.3's decoder layers (`glm_moe_dsa`), from its block-FP8 checkpoint, op for op as HF transformers'
`GlmMoeDsaDecoderLayer` computes them with SDPA attention and eager experts (`check_reference` compares them bit for bit
on the check window).

The weights are the checkpoint's: each FP8 linear's E4M3 codes times its weight_scale_inv blocks (128 x 128), in FP32
(`dequant`); the bf16 reference rounds that to bf16, as HF's FP8 dequantization does. A step reads the same product in
float64, exactly, as `weight64` (what the int8 recipe `verity-int8-channel127/v3` requantizes: `int8.exact`). Everything else is cast to bf16 but the router's e_score_correction_bias (FP32,
HF's `_keep_in_fp32_modules_strict`).

Attention is MLA with DSA. At T <= index_topk every query's top-k is every key it may see, so the indexer selects all of
them and attention is dense causal: the layer refuses longer windows rather than run the indexer. The router (FP32
sigmoid scores, noaux_tc) is never an arm's: an arm's linear steps reach q_a, q_b, kv_a, kv_b, o, the dense MLP's,
the shared expert's and each routed expert's gate_up and down (named `mlp.experts.E.gate_up_proj`), each expert on the
rows routed to it, as HF calls it. Attention steps and calibration are refused."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from matmul import MM, Run, run_chain

BLOCK = 128
FP32_KEEP = ("mlp.gate.e_score_correction_bias",)


def dequant(codes: torch.Tensor, s: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """E4M3 codes (their bytes, n x k) times their per-(128 x 128)-block FP32 scales, in `dtype`: FP32 rounds the
    product once (RN32), float64 holds it exactly (4 + 24 significant bits)."""
    n, k = codes.shape
    blocks = s.to(dtype).repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :k]
    return codes.view(torch.float8_e4m3fn).to(dtype) * blocks


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(dtype)


def rope_interleave(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF's `apply_rotary_pos_emb_interleave` on one tensor ([B, H, T, D], cos/sin [B, T, D])."""
    cos = cos[..., : cos.shape[-1] // 2].unsqueeze(1)
    sin = sin[..., : sin.shape[-1] // 2].unsqueeze(1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GlmLayer:
    #: RMSNorm's default eps, which HF's MLA norms take (only the layer norms read rms_norm_eps)
    LORA_EPS = 1e-6

    def __init__(self, cfg, li: int, tensors: dict, device):
        self.cfg, self.li, self.g = cfg, li, cfg.glm
        p = f"model.layers.{li}."
        self.codes: dict = {}
        self.t: dict = {}
        for n, t in tensors.items():
            n = n[len(p):]
            if n.endswith(".weight_scale_inv"):
                self.t[n] = t.to(device=device, dtype=torch.float32)
            elif t.dtype == torch.uint8:
                self.codes[n] = t.to(device=device)
            else:
                self.t[n] = t.to(device=device, dtype=torch.float32 if n in FP32_KEEP else torch.bfloat16)
        self.sparse = self.g["mlp_layer_types"][li] == "sparse"
        self.scaling = (self.g["qk_nope_head_dim"] + self.g["qk_rope_head_dim"]) ** -0.5

    # ------------------------------------------------------------ weights

    def weight32(self, name: str, dtype=torch.float32) -> torch.Tensor:
        """A linear's weight in FP32 (`dtype` float64: exact): an expert's gate_up is its gate's rows, then its up's."""
        if name.startswith("mlp.experts.") and name.endswith(".gate_up_proj"):
            e = name[:-len(".gate_up_proj")]
            return torch.cat([self.weight32(f"{e}.gate_proj", dtype), self.weight32(f"{e}.up_proj", dtype)])
        w = f"{name}.weight"
        if w in self.codes:
            return dequant(self.codes[w], self.t[f"{name}.weight_scale_inv"], dtype)
        return self.t[w].to(dtype)

    def weight(self, name: str) -> torch.Tensor:
        return self.weight32(name).to(torch.bfloat16)

    def linear(self, name: str, x: torch.Tensor, run: Run | None, collect=None) -> torch.Tensor:
        if collect is not None:
            raise ValueError("GLM-5.3: no arm reads calibration statistics")
        steps = run.steps_for("linear", name) if run is not None else []
        if not steps:
            return F.linear(x, self.weight(name))
        mm = MM("linear", name, run, x.reshape(-1, x.shape[-1]).float(), name, maps=(self.weight32,),
                extra={"weight64": lambda: self.weight32(name, torch.float64)})
        y = run_chain(steps, mm)
        return y.to(x.dtype).reshape(*x.shape[:-1], y.shape[-1])

    # ------------------------------------------------------------ the layer

    def forward(self, h: torch.Tensor, cos, sin, run: Run | None = None, collect=None) -> torch.Tensor:
        c, g = self.cfg, self.g
        if run is not None and (run.steps_for("qk", "qk") or run.steps_for("pv", "pv")):
            raise ValueError("GLM-5.3: attention runs in bf16 (MLA); an arm's steps reach its linears")
        B, T, _ = h.shape
        if T > g["index_topk"]:
            raise ValueError(f"GLM-5.3: T = {T} > index_topk = {g['index_topk']}, where DSA is sparse")
        res = h
        x = rms_norm(h, self.t["input_layernorm.weight"], c.eps)
        nope, rope, vd = g["qk_nope_head_dim"], g["qk_rope_head_dim"], g["v_head_dim"]
        a = "self_attn."
        q_resid = rms_norm(self.linear(a + "q_a_proj", x, run, collect), self.t[a + "q_a_layernorm.weight"],
                           self.LORA_EPS)
        q = self.linear(a + "q_b_proj", q_resid, run, collect).view(B, T, -1, nope + rope).transpose(1, 2)
        q_pass, q_rot = torch.split(q, [nope, rope], dim=-1)
        kv_pass, k_rot = torch.split(self.linear(a + "kv_a_proj_with_mqa", x, run, collect),
                                     [g["kv_lora_rank"], rope], dim=-1)
        k_pass = rms_norm(kv_pass, self.t[a + "kv_a_layernorm.weight"], self.LORA_EPS).view(B, 1, T, g["kv_lora_rank"])
        k_rot = k_rot.view(B, 1, T, rope)
        q_rot, k_rot = rope_interleave(q_rot, cos, sin), rope_interleave(k_rot, cos, sin)
        query = torch.cat((q_pass, q_rot), dim=-1)
        kv = self.linear(a + "kv_b_proj", k_pass, run, collect).view(B, T, -1, nope + vd).transpose(1, 2)
        k_nope, value = torch.split(kv, [nope, vd], dim=-1)
        key = kv.new_empty(*kv.shape[:-1], nope + rope)
        key[..., :nope].copy_(k_nope)
        key[..., nope:].copy_(k_rot.expand(-1, k_nope.shape[1], -1, -1))
        mask = torch.ones(B, 1, T, T, dtype=torch.bool, device=h.device).tril()
        o = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0, scale=self.scaling,
                                           is_causal=False)
        o = o.transpose(1, 2).contiguous().reshape(B, T, -1).contiguous()
        h = res + self.linear(a + "o_proj", o, run, collect)
        res = h
        x = rms_norm(h, self.t["post_attention_layernorm.weight"], c.eps)
        return res + (self.moe(x, run, collect) if self.sparse else self.mlp("mlp.", x, run, collect))

    def mlp(self, p: str, x: torch.Tensor, run, collect) -> torch.Tensor:
        m = F.silu(self.linear(p + "gate_proj", x, run, collect)) * self.linear(p + "up_proj", x, run, collect)
        return self.linear(p + "down_proj", m, run, collect)

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """HF's `GlmMoeDsaTopkRouter`: the top-k experts of each row and their weights."""
        g = self.g
        E, k = g["n_routed_experts"], g["num_experts_per_tok"]
        logits = F.linear(x.type(torch.float32), self.t["mlp.gate.weight"].type(torch.float32))
        scores = logits.sigmoid()
        choice = scores + self.t["mlp.gate.e_score_correction_bias"]
        ng = g["n_group"]
        group_scores = choice.view(-1, ng, E // ng).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=g["topk_group"], dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, ng, E // ng).reshape(-1, E)
        choice = choice.masked_fill(~score_mask.bool(), float("-inf"))
        idx = torch.topk(choice, k=k, dim=-1, sorted=False)[1]
        w = scores.gather(1, idx)
        if g["norm_topk_prob"]:
            w /= w.sum(dim=-1, keepdim=True) + 1e-20
        return idx, w * g["routed_scaling_factor"]

    def moe(self, x: torch.Tensor, run, collect) -> torch.Tensor:
        shape = x.shape
        flat = x.view(-1, shape[-1])
        idx, w = self.route(flat)
        E = self.g["n_routed_experts"]
        out = torch.zeros_like(flat)
        mask = F.one_hot(idx, num_classes=E).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = int(e[0])
            pos, tok = torch.where(mask[e])
            gate, up = self.linear(f"mlp.experts.{e}.gate_up_proj", flat[tok], run, collect).chunk(2, dim=-1)
            y = self.linear(f"mlp.experts.{e}.down_proj", F.silu(gate) * up, run, collect)
            out.index_add_(0, tok, (y * w[tok, pos, None]).to(out.dtype))
        return out.view(*shape) + self.mlp("mlp.shared_experts.", x, run, collect)


def hf_state(layer: GlmLayer) -> dict:
    """The layer's weights as HF's decoder layer holds them: bf16 linears, the experts stacked."""
    sd = {}
    names = {n[:-len(".weight")] for n in layer.codes} | {n[:-len(".weight")] for n in layer.t if n.endswith(".weight")}
    for n in sorted(names):
        if ".experts." not in n:
            sd[f"{n}.weight"] = layer.weight(n)
    for n, t in layer.t.items():
        if not n.endswith((".weight", ".weight_scale_inv")):
            sd[n] = t
    if layer.sparse:
        E = layer.g["n_routed_experts"]
        sd["mlp.experts.gate_up_proj"] = torch.stack([layer.weight(f"mlp.experts.{e}.gate_up_proj") for e in range(E)])
        sd["mlp.experts.down_proj"] = torch.stack([layer.weight(f"mlp.experts.{e}.down_proj") for e in range(E)])
    return sd


def check_reference(gates, cfg, snapshot, layer: GlmLayer, h: torch.Tensor, cos, sin, out: torch.Tensor) -> None:
    """The engine's bf16 layer output equals HF's `GlmMoeDsaDecoderLayer` on the same input and weights, bit for bit,
    with HF's own causal mask, indexer (so its top-k covers every key) and RoPE tables."""
    try:
        import transformers
        from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as hf
    except ImportError:
        gates.skip("hf_reference", "transformers with glm_moe_dsa is not importable")
        return
    hc = transformers.AutoConfig.from_pretrained(str(snapshot))
    hc._attn_implementation = "sdpa"
    hc._experts_implementation = "eager"
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            hl = hf.GlmMoeDsaDecoderLayer(hc, layer.li)
    finally:
        torch.set_default_dtype(prev)
    hl.load_state_dict(hf_state(layer), strict=True, assign=True)
    hl.eval()
    dev = h.device
    rot = hf.GlmMoeDsaRotaryEmbedding(hc).to(dev)
    B, T, _ = h.shape
    pos = torch.arange(T, device=dev)[None].expand(B, -1)
    mask = torch.ones(T, T, dtype=torch.bool, device=dev).tril().view(1, 1, T, T).expand(B, 1, T, T)
    prev_topk = torch.arange(T, device=dev, dtype=torch.int32).view(1, 1, T).expand(B, T, T)
    with torch.no_grad():
        hcos, hsin = rot(h, pos)
        want, _ = hl(h, attention_mask=mask, position_ids=pos, position_embeddings=(hcos, hsin),
                     prev_topk_indices=prev_topk)
    ok = bool(torch.equal(hcos, cos.expand(B, -1, -1)) and torch.equal(hsin, sin.expand(B, -1, -1))
              and torch.equal(want, out))
    gates.record("hf_reference", ok, f"layer {layer.li}", None if ok else float((want.float() - out.float()).abs().max()))
    del hl
