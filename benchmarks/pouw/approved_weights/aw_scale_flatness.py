"""NVFP4 block-scale flatness of registered weights, as registered and after rung 3's keyed Haar rotation (RMSNorm scale
folded, input side of q/k/v/gate/up): per row, the share of 16-blocks at the row's modal UE4M3 scale byte; and the share of
2^L blocks drawn at random from one matrix (any rows) that share one (row scale, block scale) pair, the condition for
integer pre-adds (fp4-int8-route.md §1)."""
import glob, json, os, sys
import torch
from safetensors import safe_open

def e4m3(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

def scales(w):
    n, k = w.shape
    gs = w.abs().amax(1, keepdim=True).clamp_min(1e-30) / (6.0 * 448.0)
    v = (w / gs).reshape(n, k // 16, 16)
    s = e4m3((v.abs().amax(2) / 6.0).clamp_min(2.0 ** -9)).view(torch.uint8)
    return gs.squeeze(1), s

def haar(d, seed=20260930):
    g = torch.Generator().manual_seed(seed + d)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q * torch.sign(torch.diagonal(r))

def stats(gs, s, g):
    n, b = s.shape
    modal = torch.tensor([torch.bincount(r.long(), minlength=256).max().item() / b for r in s]).mean().item()
    out = {"modal_share_per_row": round(modal, 4)}
    for L in (1, 2, 3, 4):
        m = 2 ** L
        ri = torch.randint(0, n, (4000, m), generator=g); bi = torch.randint(0, b, (4000, m), generator=g)
        sc, gg = s[ri, bi], gs[ri]
        same_block = (sc == sc[:, :1]).all(1).float().mean().item()
        same_both = ((sc == sc[:, :1]).all(1) & (gg == gg[:, :1]).all(1)).float().mean().item()
        out[f"L{L}_block_scale_equal"] = round(same_block, 4); out[f"L{L}_row_and_block_equal"] = round(same_both, 4)
        ri2 = torch.randint(0, n, (4000, 1), generator=g).expand(4000, m); bi2 = torch.randint(0, b, (4000, m), generator=g)
        out[f"L{L}_one_row_block_scale_equal"] = round((s[ri2, bi2] == s[ri2[:, :1], bi2[:, :1]]).all(1).float().mean().item(), 4)
    return out

res = {}
hub = os.path.expanduser("~/.cache/huggingface/hub")
for model in sys.argv[1:]:
    snap = glob.glob(f"{hub}/models--{model.replace('/', '--')}/snapshots/*")[0]
    f = safe_open(glob.glob(f"{snap}/*.safetensors")[0], "pt")
    keys = [k for k in f.keys() if k.endswith(("q_proj.weight", "gate_proj.weight", "o_proj.weight", "down_proj.weight"))]
    layers = sorted({int(k.split(".")[2]) for k in keys}); pick = {layers[0], layers[len(layers) // 2], layers[-1]}
    g = torch.Generator().manual_seed(1)
    for k in keys:
        if int(k.split(".")[2]) not in pick:
            continue
        w = f.get_tensor(k).float()
        layer = k.rsplit(".", 3)[0]
        if k.endswith(("q_proj.weight", "gate_proj.weight")):
            nm = layer + (".input_layernorm.weight" if "q_proj" in k else ".post_attention_layernorm.weight")
            wr = (w * f.get_tensor(nm).float()[None, :]) @ haar(w.shape[1])
        else:
            wr = haar(w.shape[0]).T @ w
        res[f"{model}:{k}"] = {"plain": stats(*scales(w), g), "rot": stats(*scales(wr), g)}
        print(k, json.dumps(res[f"{model}:{k}"]), flush=True)
json.dump(res, open("/tmp/aw/flat/scale_flatness.json", "w"), indent=1)
