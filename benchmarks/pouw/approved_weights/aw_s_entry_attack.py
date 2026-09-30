"""Can a registrant flatten NVFP4 block scales of rung-3 weights through the unmixed channels? Rows: a Haar-rotated part
(Gaussian, as any master's rotated row is) plus one unmixed entry at M x the rotated part's RMS. Per M: the share of a row's
16-blocks at its modal UE4M3 byte, the share at the minimum byte, and, for 2^L blocks drawn from the matrix, how often they
share one byte; with the side product the unmixed entry leaves the chain and the row is quantized without it."""
import json, torch
torch.manual_seed(0)
def e4m3(x): return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
def scales(w):
    n, k = w.shape
    gs = w.abs().amax(1, keepdim=True).clamp_min(1e-30) / (6.0 * 448.0)
    v = (w / gs).reshape(n, k // 16, 16)
    return e4m3((v.abs().amax(2) / 6.0).clamp_min(2.0 ** -9)).view(torch.uint8)
def stats(s, g):
    n, b = s.shape
    modal = torch.tensor([torch.bincount(r.long(), minlength=256).max().item() / b for r in s]).mean().item()
    mn = (s == e4m3(torch.tensor(2.0 ** -9)).view(torch.uint8)).float().mean().item()
    out = {"modal": round(modal, 4), "at_min_byte": round(mn, 4)}
    for L in (1, 2, 3, 4):
        ri = torch.randint(0, n, (4000, 2 ** L), generator=g); bi = torch.randint(0, b, (4000, 2 ** L), generator=g)
        sc = s[ri, bi]; out[f"L{L}"] = round((sc == sc[:, :1]).all(1).float().mean().item(), 4)
    return out
res = {}
for k in (3584, 8192):
    g = torch.Generator().manual_seed(k)
    w = torch.randn(512, k)
    pos = 1 + 8 * torch.randint(0, k // 8, (1,)).item()          # an unmixed position, off every 8th
    res[f"k{k}/gaussian"] = stats(scales(w), g)
    for M in (16, 100, 1e3, 1e4, 1e5, 3e5, 1e6, 1e7):
        w2 = w.clone(); w2[:, pos] = M
        res[f"k{k}/unmixed entry {M:g}x RMS, in the chain"] = stats(scales(w2), g)
        w3 = w2.clone(); w3[:, pos] = 0.0                           # the side product: S leaves the chain
        res[f"k{k}/unmixed entry {M:g}x RMS, side product"] = stats(scales(w3), g)
for kk, v in res.items(): print(kk, v)
json.dump(res, open("/tmp/aw/flat/s_entry_attack.json", "w"), indent=1)
