"""H1's offline weight transform (`specs/co32c-u16-cohalf.toml`): an HF checkpoint to the int8 weights the
served co32c linears take (cgemm.cu's `cwedit_h1`: W in [-110, 110], plus one fp32 scale a row).

It is the weight side of `steps.py`'s H1 linear arm, folded once so serving does no smoothing: SmoothQuant
(`smooth`), the rotation (`rotate_linear`, Q seeded by the input width), and the ±110 floor quantization
(`int8_linear`'s weights). The three steps are, per linear:

  smooth   W <- W * m (input columns), and 1 / m folds into the activation path: the RMSNorm before q/k/v and gate/up,
           up_proj's rows (before down_proj), and v_proj's rows (o_proj's m, shared per KV group: `kv_shared`). lm_head
           is not smoothed.
  rotate   W <- W Q, Q = `steps.Rotation(k, "q")` for the linear's input width k (shared by every hidden-width linear
           and lm_head; down_proj has its own on the intermediate width). The engine applies x Q online.
  int8     s_w = amax_row / 110, W = floor(clamp(W / s_w, -110, 110)) in int8. cwedit adds the per-call bit c and U on
           ±16; serving dequantizes y = (f32(Y) * s_a[token]) * s_w[row].

The served activation quantizer is cquant_h1 (per-token s_a = amax / 126, X = floor(clamp(x / s_a, -126, 126)) + b), on
the device; it needs no offline step. Calibration (SmoothQuant's per-input-channel amax of the bf16 activations) is read
from a file; `benchmarks/ppl/run.py` already produces those maxima (`calib["amax"]`), or pass `--identity` for no
smoothing. Weights, scales and folded norms are written as one safetensors checkpoint.

    uv run --extra torch-cpu python benchmarks/ppl/transform.py SNAPSHOT OUT --calib CALIB.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Config  # noqa: E402
from steps import GROUP, Rotation, dither_shift  # noqa: E402

#: `steps.int8_linear`'s `weight_top` for the H1 arm (`specs/co32c-u16-cohalf.toml`). The 1-bit dither (m = 0) clamps at
#: t = weight_top - 1 = 110, so the quantized W is in [-110, 110] and the served W + c (cwedit's bit) in [-110, 111].
WEIGHT_TOP = 111
#: a group's first linear's input stands for the group (calibration reads it), and 1 / m folds into this tensor's rows.
FOLD = {"qkv": "input_layernorm", "gu": "post_attention_layernorm", "down": "mlp.up_proj", "o": "self_attn.v_proj"}
LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def smooth_m(ax: torch.Tensor, aw: torch.Tensor, alpha: float, kv_shared: bool, group: str,
             rep: int, head_dim: int) -> torch.Tensor:
    """`steps.smooth_factors`: m = amax_x^alpha / amax_W^(1 - alpha) per input channel, shared per (KV head, channel)
    for o_proj when `kv_shared`."""
    if kv_shared and group == "o":
        ax, aw = (a.view(-1, rep, head_dim).amax(1, keepdim=True).expand(-1, rep, head_dim).reshape(-1) for a in (ax, aw))
    return (ax.clamp(min=1e-5) ** alpha / aw.clamp(min=1e-5) ** (1 - alpha)).clamp(min=1e-5)


def quant_weight(w: torch.Tensor, weight_top: int = WEIGHT_TOP, dither_bits: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """`steps.int8_linear`'s dithered weights, without the per-call bit: m = `dither_shift`(bits), t = weight_top - 1 - m,
    s_w = amax_row / t, W = floor(clamp(w / s_w, -t, t)) - m in int8, within [-t - m, t - m]. The H1 arm takes the 1-bit
    dither (m = 0), so W is in [-110, 110]."""
    m = dither_shift(dither_bits)
    t = weight_top - 1 - m
    s = w.abs().amax(dim=1).clamp(min=1e-30) / t
    q = (torch.floor((w / s[:, None]).clamp(-t, t)) - m).to(torch.int8)
    return q, s.to(torch.float32)


def _group_weights(sd: dict, p: str, group: str) -> list[str]:
    return [p + n + ".weight" for n in GROUP if GROUP[n] == group]


def transform_checkpoint(sd: dict[str, torch.Tensor], cfg: Config, calib: dict | None,
                         *, alpha: float = 0.5, kv_shared: bool = True) -> dict[str, torch.Tensor]:
    """Return a new state dict: `<linear>.weight` int8 in [-110, 110], `<linear>.scale` fp32 per output row, and the
    folded RMSNorm / up_proj / v_proj. `calib` is {(layer, group): amax_x}; None (or `--identity`) skips smoothing."""
    out = {k: v.clone() for k, v in sd.items()}
    rep, hd = cfg.rep, cfg.head_dim
    rots: dict[int, Rotation] = {}

    def rot(w: torch.Tensor) -> torch.Tensor:
        k = w.shape[1]
        if k not in rots:
            rots[k] = Rotation(k, "q", w.device)
        return rots[k].apply(w.float())

    for li in range(cfg.layers):
        p = f"model.layers.{li}."
        for group in ("qkv", "gu", "down", "o"):
            if calib is None or (li, group) not in calib:
                continue
            gw = _group_weights(out, p, group)
            aw = torch.stack([out[n].float().abs().amax(dim=0) for n in gw]).amax(dim=0)
            m = smooth_m(calib[(li, group)].float(), aw, alpha, kv_shared, group, rep, hd)
            for n in gw:
                out[n] = out[n].float() * m[None, :]
            fold = FOLD[group]
            key = p + fold + ".weight"
            if fold.endswith("layernorm"):
                out[key] = out[key].float() / m
            elif group == "o":  # o_proj's m is shared per (KV head, channel); fold one share into v_proj's rows
                if not kv_shared:
                    raise ValueError("o_proj's smoothing folds into v_proj only under kv_shared")
                rows = m.view(cfg.kv_heads, rep, hd)[:, 0, :].reshape(-1)
                out[key] = out[key].float() / rows[:, None]
            else:
                out[key] = out[key].float() / m[:, None]   # up_proj's rows

    heads = [f"model.layers.{li}.{n}" for li in range(cfg.layers) for n in LINEARS]
    if "lm_head.weight" in out:
        heads.append("lm_head")
    elif not cfg.tie:
        raise ValueError("no lm_head.weight and not tied")
    for n in heads:
        key = n + ".weight"
        q, s = quant_weight(rot(out[key]))
        out[key], out[n + ".scale"] = q, s
    return out


def load_calib(path: Path) -> dict:
    """The calibration file (one `torch.save`d dict): a tensor per `"<layer>.<group>"` (group one of qkv, gu, down, o),
    the per-input-channel amax of the group's calibration activations -- the same `amax_x` `steps.smooth_factors` reads
    as `calib["amax"][group]`. Returns {(layer, group): amax_x}."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    calib = {}
    for k, v in raw.items():
        li, group = k.rsplit(".", 1)
        calib[(int(li), group)] = v
    return calib


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--calib", type=Path, help="SmoothQuant amax per (layer, group); omit with --identity")
    ap.add_argument("--identity", action="store_true", help="no smoothing (rotate and quantize only)")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    if not args.identity and args.calib is None:
        ap.error("pass --calib FILE or --identity")
    from safetensors.torch import load_file, save_file
    cfg = Config.read(args.snapshot)
    sd = {}
    for f in sorted(args.snapshot.glob("*.safetensors")):
        sd.update(load_file(f))
    calib = None if args.identity else load_calib(args.calib)
    out = transform_checkpoint(sd, cfg, calib, alpha=args.alpha)
    args.out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(args.out / "model.int8.safetensors"))
    print(f"wrote {args.out / 'model.int8.safetensors'} ({len(out)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
