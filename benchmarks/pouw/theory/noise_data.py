"""The data-dependent terms of noise_sweep.py's designs, on Llama-3.2-1B's real int8 operands (WikiText-2).

noise_sweep.py prices nsf, disp2s and perm as "0 + dup": their words repeat wherever a k32 block's dot product is zero,
and perm's repeat across identical rows. per2's α depends on its limb count, max |fold₃₂ B| / 127. This measures those
on the int8 operands the stock recipe serves (int8-channel127-v1: activations one scale a token, weights one an output
row, both rounded to ±127). It hooks each fused linear's input (qkv, o, gate_up, down) over `--windows` windows of
`--ctx` tokens, each led by BOS as a request is.

Per linear and layer, at m = 64 (the first rows of window 0, BOS included), m = ctx (window 0) and m = windows·ctx:

- zero_blocks: the share of (row, k32 block) pairs whose 32 entries are all 0, and rows with most blocks zero;
- zero_dot: the share of (row, column, block) words whose block dot product is 0, over `--cols` sampled columns. Such a
  word equals the one before it, so it is a duplicate for nsf, disp2s and perm;
- identical_rows and zero_rows: rows equal to an earlier row of the same call, and all-zero rows;
- zero_regs: the share of 4-entry registers that are all 0 (os2 and per2 form A + E there, a duplicate of E).

Per weight: per2's limbs, the share of zero k32 blocks, and identical output rows.

Run (CPU, a few minutes): python noise_data.py --model DIR --text wikitext-2-raw-v1-test.json --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys

import torch

LINEARS = {"qkv": ("self_attn", ("q_proj", "k_proj", "v_proj")), "o": ("self_attn", ("o_proj",)),
           "gate_up": ("mlp", ("gate_proj", "up_proj")), "down": ("mlp", ("down_proj",))}


def quant_rows(x: torch.Tensor, top: int = 127) -> torch.Tensor:
    s = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-30) / top
    return torch.round(x / s).clamp(-top, top)


def row_hashes(q: torch.Tensor) -> list[bytes]:
    b = q.to(torch.int8).numpy()
    return [hashlib.blake2b(r.tobytes(), digest_size=16).digest() for r in b]


class Acc:
    def __init__(self) -> None:
        self.d: dict[str, float] = {}

    def add(self, key: str, v: float) -> None:
        self.d[key] = self.d.get(key, 0.0) + v


def act_stats(q: torch.Tensor, wcols: torch.Tensor, acc: Acc, seen: set | None) -> None:
    m, k = q.shape
    t = k // 32
    blk = q.view(m, t, 32)
    zb = (blk == 0).all(-1)
    acc.add("rows", m)
    acc.add("blocks", m * t)
    acc.add("zero_blocks", float(zb.sum()))
    acc.add("massive_rows", float((zb.float().mean(1) > 0.5).sum()))
    acc.add("zero_rows", float((q == 0).all(1).sum()))
    acc.add("regs", m * k // 4)
    acc.add("zero_regs", float((q.view(m, k // 4, 4) == 0).all(-1).sum()))
    hs = row_hashes(q)
    local = set() if seen is None else seen
    dup = 0
    for h in hs:
        if h in local:
            dup += 1
        local.add(h)
    acc.add("identical_rows", dup)
    c = wcols.shape[0]
    wb = wcols.view(c, t, 32).permute(1, 2, 0)
    zd = 0.0
    for r0 in range(0, m, 512):
        a = blk[r0:r0 + 512].permute(1, 0, 2)
        dots = torch.bmm(a, wb)
        zd += float((dots == 0).sum())
    acc.add("words", m * c * t)
    acc.add("zero_dot", zd)


def weight_stats(wq: torch.Tensor) -> dict:
    n, k = wq.shape
    t = k // 32
    fold = wq.view(n, t, 32).sum(1)
    hs = row_hashes(wq)
    return {"n": n, "K": k, "fold_max": float(fold.abs().max()), "limbs": math.ceil(float(fold.abs().max()) / 127),
            "fold_p999": float(fold.abs().flatten().kthvalue(int(0.999 * fold.numel())).values),
            "zero_wblocks": float((wq.view(n, t, 32) == 0).all(-1).float().mean()),
            "identical_wrows": n - len(set(hs))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the model's snapshot directory")
    ap.add_argument("--text", required=True, help="WikiText-2 test split as a JSON list of lines")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=128)
    ap.add_argument("--small", type=int, default=64)
    args = ap.parse_args(argv)
    import transformers

    torch.manual_seed(0)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    lines = json.load(open(args.text))
    ids = tok("\n\n".join(lines), add_special_tokens=False)["input_ids"]
    bos = tok.bos_token_id
    layers = model.model.layers
    weights, wcols, wstats = {}, {}, {}
    for li, layer in enumerate(layers):
        for name, (part, mods) in LINEARS.items():
            w = torch.cat([getattr(getattr(layer, part), mo).weight.detach().float() for mo in mods], 0)
            wq = quant_rows(w)
            wstats[f"{li}.{name}"] = weight_stats(wq)
            g = torch.Generator().manual_seed(li * 7 + len(name))
            idx = torch.randperm(wq.shape[0], generator=g)[:args.cols]
            wcols[(li, name)] = wq[idx].contiguous()
    accs = {(li, name, scope): Acc() for li in range(len(layers)) for name in LINEARS for scope in ("m64", "ctx", "all")}
    seen_all = {(li, name): set() for li in range(len(layers)) for name in LINEARS}
    state = {"w": 0}

    def hook(li: int, name: str):
        def f(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            q = quant_rows(x)
            wc = wcols[(li, name)]
            act_stats(q, wc, accs[(li, name, "all")], seen_all[(li, name)])
            if state["w"] == 0:
                act_stats(q, wc, accs[(li, name, "ctx")], None)
                act_stats(q[:args.small], wc, accs[(li, name, "m64")], None)
        return f

    handles = []
    for li, layer in enumerate(layers):
        for name, (part, mods) in LINEARS.items():
            handles.append(getattr(getattr(layer, part), mods[0]).register_forward_pre_hook(hook(li, name)))
    step = args.ctx - 1
    with torch.no_grad():
        for w in range(args.windows):
            state["w"] = w
            chunk = [bos] + ids[w * step:(w + 1) * step]
            model(torch.tensor([chunk]))
    for h in handles:
        h.remove()
    per = {f"{li}.{name}.{scope}": a.d for (li, name, scope), a in accs.items()}
    summary = {}
    for scope in ("m64", "ctx", "all"):
        tot = {"macs": 0.0, "zero_dot_w": 0.0, "ident_w": 0.0, "zero_block_w": 0.0, "zero_reg_w": 0.0, "massive": 0.0,
               "rows": 0.0}
        for li in range(len(layers)):
            for name in LINEARS:
                a = accs[(li, name, scope)].d
                ws = wstats[f"{li}.{name}"]
                macs = a["rows"] * ws["n"] * ws["K"]
                tot["macs"] += macs
                tot["zero_dot_w"] += macs * a["zero_dot"] / a["words"]
                tot["ident_w"] += macs * a["identical_rows"] / a["rows"]
                tot["zero_block_w"] += macs * a["zero_blocks"] / a["blocks"]
                tot["zero_reg_w"] += macs * a["zero_regs"] / a["regs"]
                tot["massive"] += a["massive_rows"]
                tot["rows"] += a["rows"]
        summary[scope] = {"zero_dot_share": tot["zero_dot_w"] / tot["macs"],
                          "identical_row_share": tot["ident_w"] / tot["macs"],
                          "zero_block_share": tot["zero_block_w"] / tot["macs"],
                          "zero_reg_share": tot["zero_reg_w"] / tot["macs"],
                          "massive_row_share": tot["massive"] / tot["rows"],
                          "perm_dup_share": (tot["zero_dot_w"] + tot["ident_w"]) / tot["macs"]}
    limbs = {name: max(wstats[f"{li}.{name}"]["limbs"] for li in range(len(layers))) for name in LINEARS}
    macs_tok = {name: wstats[f"0.{name}"]["n"] * wstats[f"0.{name}"]["K"] for name in LINEARS}
    alpha_per2 = sum(macs_tok[nm] * (1 + 32 * limbs[nm] / wstats[f"0.{nm}"]["K"]) for nm in LINEARS) / sum(macs_tok.values())
    out = {"args": vars(args), "summary": summary, "per2_limbs_max": limbs, "per2_alpha_llama": alpha_per2,
           "weights": wstats, "per_linear": per}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"summary": summary, "per2_limbs_max": limbs, "per2_alpha_llama": alpha_per2}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
