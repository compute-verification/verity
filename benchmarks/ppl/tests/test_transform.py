"""transform.py: the H1 offline weight transform is the weight side of steps.py's H1 linear arm (smooth fold, rotation,
±110 floor quant), so a transformed linear reconstructs the original map within the weight quant error, its weights are
int8 in [-110, 110] with one scale a row, and its quant is bit-for-bit steps.py's."""
import pytest
import torch

import steps
import transform as tf
from model import Config


def tiny_cfg() -> Config:
    return Config(arch="llama", hidden=16, layers=1, heads=4, kv_heads=2, head_dim=4, intermediate=28, eps=1e-5,
                  vocab=32, tie=False, qkv_bias=False, rope={"rope_type": "default", "rope_theta": 1e4})


def tiny_sd(cfg: Config, gen: torch.Generator) -> dict:
    H, I, hd = cfg.hidden, cfg.intermediate, cfg.head_dim
    q, kv = cfg.heads * hd, cfg.kv_heads * hd
    rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32)
    sd = {}
    for li in range(cfg.layers):
        p = f"model.layers.{li}."
        for n, (o, i) in {"self_attn.q_proj": (q, H), "self_attn.k_proj": (kv, H), "self_attn.v_proj": (kv, H),
                          "self_attn.o_proj": (H, q), "mlp.gate_proj": (I, H), "mlp.up_proj": (I, H),
                          "mlp.down_proj": (H, I)}.items():
            sd[p + n + ".weight"] = rn(o, i)
        sd[p + "input_layernorm.weight"] = rn(H).abs() + 0.5
        sd[p + "post_attention_layernorm.weight"] = rn(H).abs() + 0.5
    sd["lm_head.weight"] = rn(cfg.vocab, H)
    sd["model.embed_tokens.weight"] = rn(cfg.vocab, H)
    return sd


def tiny_calib(cfg: Config, gen: torch.Generator) -> dict:
    widths = {"qkv": cfg.hidden, "gu": cfg.hidden, "down": cfg.intermediate, "o": cfg.heads * cfg.head_dim}
    return {(0, g): torch.rand(w, generator=gen).float() + 0.1 for g, w in widths.items()}


def test_quant_weight_matches_steps():
    """quant_weight is steps.int8_linear's dithered weights without the per-call bit (co32c-u16-cohalf: weight_top 111,
    the 1-bit dither's m = 0, so t = 110 and W stays in [-110, 110])."""
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(40, 96, generator=gen)
    q, s = tf.quant_weight(w)
    m = steps.dither_shift(1)
    t = tf.WEIGHT_TOP - 1 - m
    s_ref = w.abs().amax(dim=1).clamp(min=1e-30) / t
    q_ref = (torch.floor((w / s_ref[:, None]).clamp(-t, t)) - m).to(torch.int8)
    assert torch.equal(q, q_ref)
    assert torch.equal(s, s_ref.float())
    assert int(q.min()) >= -t and int(q.max()) <= t


def test_ranges_and_scale_shape():
    cfg = tiny_cfg()
    gen = torch.Generator().manual_seed(2)
    sd = tiny_sd(cfg, gen)
    out = tf.transform_checkpoint(sd, cfg, tiny_calib(cfg, gen))
    top = tf.WEIGHT_TOP - 1
    for n in tf.LINEARS + ["lm_head"]:
        key = f"model.layers.0.{n}.weight" if n in tf.LINEARS else "lm_head.weight"
        w, s = out[key], out[(key[:-7] if n == "lm_head" else f"model.layers.0.{n}") + ".scale"]
        assert w.dtype == torch.int8
        assert int(w.min()) >= -top and int(w.max()) <= top
        assert s.dtype == torch.float32 and s.shape == (w.shape[0],) and bool((s > 0).all())


@pytest.mark.parametrize("name", ["self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj", "self_attn.o_proj"])
def test_reconstructs_linear_map(name):
    """smooth fold + rotation + weight dequant reconstruct the original linear map within the quant error."""
    cfg = tiny_cfg()
    gen = torch.Generator().manual_seed(3)
    sd = tiny_sd(cfg, gen)
    calib = tiny_calib(cfg, gen)
    out = tf.transform_checkpoint(sd, cfg, calib)
    p = "model.layers.0."
    group = steps.GROUP[name]
    w0 = sd[p + name + ".weight"].float()
    k = w0.shape[1]
    # the smooth factor the transform used, and the online rotation Q for this input width
    gw = [p + x + ".weight" for x in steps.GROUP if steps.GROUP[x] == group]
    aw = torch.stack([sd[n].float().abs().amax(dim=0) for n in gw]).amax(dim=0)
    m = tf.smooth_m(calib[(0, group)].float(), aw, 0.5, True, group, cfg.rep, cfg.head_dim)
    Q = steps.Rotation(k, "q", w0.device).dense()
    x = torch.randn(8, k, generator=gen)
    y0 = x @ w0.T                                      # the true map on the raw (pre-smooth) input
    w_eff = out[p + name + ".weight"].float() * out[p + name + ".scale"][:, None]
    y1 = ((x / m[None, :]) @ Q) @ w_eff.T             # engine: folded norm (x / m), rotate (x Q), int8 GEMM, dequant
    rel = (y1 - y0).norm() / y0.norm().clamp(min=1e-9)
    assert rel < 0.05, rel


def test_norm_fold_preserves_input_path():
    """1/m folds into the RMSNorm so the folded norm times the raw input equals the smoothed input."""
    cfg = tiny_cfg()
    gen = torch.Generator().manual_seed(4)
    sd = tiny_sd(cfg, gen)
    calib = tiny_calib(cfg, gen)
    out = tf.transform_checkpoint(sd, cfg, calib)
    p = "model.layers.0."
    gw = [p + x + ".weight" for x in steps.GROUP if steps.GROUP[x] == "qkv"]
    aw = torch.stack([sd[n].float().abs().amax(dim=0) for n in gw]).amax(dim=0)
    m = tf.smooth_m(calib[(0, "qkv")].float(), aw, 0.5, True, "qkv", cfg.rep, cfg.head_dim)
    folded = out[p + "input_layernorm.weight"].float()
    orig = sd[p + "input_layernorm.weight"].float()
    assert torch.allclose(folded * m, orig, atol=1e-5)


def test_identity_skips_smoothing():
    cfg = tiny_cfg()
    gen = torch.Generator().manual_seed(5)
    sd = tiny_sd(cfg, gen)
    out = tf.transform_checkpoint(sd, cfg, None)
    p = "model.layers.0."
    assert torch.equal(out[p + "input_layernorm.weight"], sd[p + "input_layernorm.weight"])
    w, s = tf.quant_weight(steps.Rotation(cfg.hidden, "q", sd[p + "self_attn.q_proj.weight"].device)
                           .apply(sd[p + "self_attn.q_proj.weight"].float()))
    assert torch.equal(out[p + "self_attn.q_proj.weight"], w)
