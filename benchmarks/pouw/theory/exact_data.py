"""The data-dependent terms of exact_sweep.py's designs, on Llama-3.2-1B's real int8 operands (WikiText-2).

exact_sweep.py's exact designs remove their noise in the stock epilogue, so three things depend on the data:

- the magic conversion's range: `IADD3(D, 0x4B400000 − r_i, −c_j)` is a float equal to 1.5·2²³ + A·Bᵀ only while
  |A·Bᵀ| < 2²², so this measures max |P|, the share of outputs at or past 2²², and the share of rows whose
  Cauchy–Schwarz bound ‖A_i‖·max_j ‖B_j‖ reaches 2²² (the rows an honest epilogue routes to its fallback);
- the words a single-axis design repeats: with row-common noise only, a final word repeats wherever two outputs of a row
  are equal (`final_row_dup`, the share of final words that equal an earlier one in their row, n − distinct a row), and
  every k32 partial sum repeats wherever two of a row's partial sums agree at the same step (`rowonly_dup`); the
  column analogues are `final_col_dup` and `colonly_dup`;
- whether the folded scale is exact: FFMA(v, s_j, −1.5·2²³·s_j) equals stock's FMUL(s_j, I2F(P)) bit for bit when
  1.5·2²³·s_j is exact in fp32, which holds for a share `scale_fold_exact` of the per-channel scales.

The operands are the stock recipe's (int8-channel127-v1: activations one scale a token, weights one an output row, both
rounded to ±127), hooked at each fused linear's input (qkv, o, gate_up, down) on one window of `--ctx` tokens led by
BOS, at m = ctx and on its first `--small` rows. Products are exact (float64 holds every int8 dot product below 2⁵³).

Run (CPU, ~10 min): python exact_data.py --model DIR --text wikitext-2-raw-v1-test.json --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

LINEARS = {"qkv": ("self_attn", ("q_proj", "k_proj", "v_proj")), "o": ("self_attn", ("o_proj",)),
           "gate_up": ("mlp", ("gate_proj", "up_proj")), "down": ("mlp", ("down_proj",))}
MAGIC_RANGE = 2 ** 22
MAGIC = 1.5 * 2 ** 23


def quant_rows(x: torch.Tensor, top: int = 127) -> tuple[torch.Tensor, torch.Tensor]:
    s = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-30) / top
    return torch.round(x / s).clamp(-top, top), s


def dup_count(v: torch.Tensor, dim: int) -> int:
    """Entries equal to an earlier entry along `dim` (size − distinct, summed over the other axis)."""
    s, _ = torch.sort(v, dim=dim)
    return int((s.narrow(dim, 1, s.shape[dim] - 1) == s.narrow(dim, 0, s.shape[dim] - 1)).sum())


def scale_fold_exact(s: torch.Tensor) -> float:
    """The share of fp32 scales s for which MAGIC·s is exact in fp32."""
    x = s.double() * MAGIC
    return float((x.float().double() == x).double().mean())


def final_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    p = a @ b.T
    m, n = p.shape
    bound = a.norm(dim=1) * b.norm(dim=1).max()
    return {"rows": m, "outputs": m * n, "max_abs": float(p.abs().max()),
            "ge_range": float((p.abs() >= MAGIC_RANGE).sum()), "cs_unsafe_rows": float((bound >= MAGIC_RANGE).sum()),
            "final_row_dup": dup_count(p, 1), "final_col_dup": dup_count(p, 0)}


def partial_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    m, k = a.shape
    n = b.shape[0]
    p = torch.zeros(m, n, dtype=torch.float64)
    row = col = 0
    for t in range(k // 32):
        p += a[:, 32 * t:32 * t + 32] @ b[:, 32 * t:32 * t + 32].T
        row += dup_count(p, 1)
        col += dup_count(p, 0)
    return {"words": m * n * (k // 32), "rowonly_dup": row, "colonly_dup": col}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the model's snapshot directory")
    ap.add_argument("--text", required=True, help="WikiText-2 test split as a JSON list of lines")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--small", type=int, default=64)
    args = ap.parse_args(argv)
    import transformers

    torch.manual_seed(0)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    lines = json.load(open(args.text))
    ids = tok("\n\n".join(lines), add_special_tokens=False)["input_ids"]
    layers = model.model.layers
    weights, wstats = {}, {}
    for li, layer in enumerate(layers):
        for name, (part, mods) in LINEARS.items():
            w = torch.cat([getattr(getattr(layer, part), mo).weight.detach().float() for mo in mods], 0)
            wq, sw = quant_rows(w)
            weights[(li, name)] = wq.double()
            wstats[f"{li}.{name}"] = {"n": wq.shape[0], "K": wq.shape[1], "scale_fold_exact": scale_fold_exact(sw)}
    per = {}

    def hook(li: int, name: str):
        def f(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            q, _ = quant_rows(x)
            a, b = q.double(), weights[(li, name)]
            per[f"{li}.{name}.ctx"] = final_stats(a, b)
            per[f"{li}.{name}.m64"] = final_stats(a[:args.small], b) | partial_stats(a[:args.small], b)
            print(li, name, per[f"{li}.{name}.m64"], flush=True)
        return f

    handles = [getattr(getattr(layer, part), mods[0]).register_forward_pre_hook(hook(li, name))
               for li, layer in enumerate(layers) for name, (part, mods) in LINEARS.items()]
    with torch.no_grad():
        model(torch.tensor([[tok.bos_token_id] + ids[:args.ctx - 1]]))
    for h in handles:
        h.remove()
    shares = (("ge_range", "outputs"), ("final_row_dup", "outputs"), ("final_col_dup", "outputs"),
              ("cs_unsafe_rows", "rows"), ("rowonly_dup", "words"), ("colonly_dup", "words"))
    summary = {}
    for scope in ("m64", "ctx"):
        macs, acc, top = 0.0, {}, 0.0
        for key, st in per.items():
            if not key.endswith(scope):
                continue
            w = st["rows"] * wstats[key.rsplit(".", 1)[0]]["n"] * wstats[key.rsplit(".", 1)[0]]["K"]
            macs += w
            for f, base in shares:
                if f in st:
                    acc[f] = acc.get(f, 0.0) + w * st[f] / st[base]
            top = max(top, st["max_abs"])
        summary[scope] = {f"{f}_share": v / macs for f, v in acc.items()} | {"max_abs": top}
    w_macs = {k: v["n"] * v["K"] for k, v in wstats.items()}
    summary["scale_fold_exact_share"] = sum(w_macs[k] * v["scale_fold_exact"] for k, v in wstats.items()) / sum(w_macs.values())
    out = {"args": vars(args), "summary": summary, "weights": wstats, "per_linear": per}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
