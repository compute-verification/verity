#!/usr/bin/env python3
"""Pearl-C at the headline k on a real model: Qwen2.5-7B's down_proj (k = 18,944, n = 3,584), weights from HF, activations
from a CPU forward streamed one decoder layer at a time (the model does not fit in RAM).

Per layer, on the real down_proj input X (the window's tokens) and weight W, under Pi's rules with §5's statistic (ρ over
every 8th position, σ = δ·ρ, α = 448/(s + 4σ); the noise floor σ·α ≥ 1 and liveness, at most k/64 entries at |x| ≥ 8σ):
- the share of rows that fail, and whether the whole unit at the real shape (every token row and every one of the 3,584
  weight rows) is admissible, which `pearlCDomainAt` needs;
- the channel split: the smallest set S of input channels, chosen greedily, whose move to a side GEMM (in BF16, outside the
  Pearl-C unit) leaves every row of X and W admissible;
- γ at the split unit's shape under the cap, twice: with the split adopted as the model's own quantizer boundary (the side
  GEMM is then glue, outside W_ref), and with the side GEMM as uncredited protocol work (2 units per BF16 MAC, plus one
  FP32 add per output word to merge it);
- on sampled 64×8 tiles of the split unit (padded to a multiple of 128 with zero channels), the replayed debit (option (A),
  `pearlc_census.debit`), the undebited residual and the cross-group greedy set.

    python benchmarks/pouw/pearlc_headline.py --tokens 512 --layers all --tile-layers 0,4,8,12,16,20,24,27
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pearlc_census as C  # noqa: E402
import run_result  # noqa: E402

GAMMA0 = 1 / 400


def rules(X: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per row: floor ok, live ok, σ, and s/ρ, on float32 rows."""
    k = X.shape[1]
    s = np.abs(X).max(1)
    sig = delta * np.sqrt((X[:, ::8].astype(np.float64) ** 2).mean(1))
    alpha = np.where(s + 4 * sig > 0, 448.0 / np.maximum(s + 4 * sig, 1e-300), 0.0)
    floor = sig * alpha >= 1.0
    live = (np.abs(X) >= 8 * sig[:, None]).sum(1) * 64 <= k
    return floor, live, sig, s / np.maximum(sig, 1e-300)


def split_channels(X: np.ndarray, W: np.ndarray, delta: float, budget: int, batch: int = 4) -> tuple[list[int], list[dict]]:
    """Greedy: move the input channels that offend the most failing rows (entries at or above 8σ of a row failing liveness,
    or above 444ρ of a row failing the floor) until every row passes or the budget is spent. This is the one pinned rule
    for S: ties in the offence count go to the lower channel index, and S never exceeds the budget (⌊k/100⌋ as deployed)."""
    k = X.shape[1]
    keep = np.ones(k, dtype=bool)
    S: list[int] = []
    trace = []
    while True:
        idx = np.flatnonzero(keep)
        Xk, Wk = X[:, idx], W[:, idx]
        off = np.zeros(len(idx))
        fails = 0
        for M in (Xk, Wk):
            fl, lv, sig, _ = rules(M, delta)
            bad = ~(fl & lv)
            fails += int(bad.sum())
            if bad.any():
                Mb, sb = np.abs(M[bad]), sig[bad][:, None]
                off += ((~fl[bad][:, None] & (Mb > 444 * sb)) | (~lv[bad][:, None] & (Mb >= 8 * sb))).sum(0)
        trace.append({"moved": len(S), "failing_rows": fails})
        if fails == 0 or len(S) >= budget:
            return S, trace
        top = idx[np.argsort(-off, kind="stable")[:batch]]
        top = [int(c) for c in top if off[np.searchsorted(idx, c)] > 0] or [int(idx[np.argmax(off)])]
        for c in top[:budget - len(S)]:
            keep[c] = False
            S.append(c)


def gamma(n: int, k: int, rho: float, side_mac_units: float = 0.0, merge: float = 0.0, G: int = 4, r: int = 32,
          fs: int = 64, qa: int = 4) -> tuple[float, float]:
    """γ per m·k at the cap ρ (forming credited, chain-only), with optional uncredited side work per m·k."""
    chain = n + n / G
    credit = chain + r + 4 * r * n / k + fs
    wref = credit + qa + side_mac_units + merge
    g = 1 - (1 - GAMMA0) * (1 - rho) * credit / wref
    gco = 1 - (1 - GAMMA0) * (credit - fs - rho * credit) / wref
    return g, gco


def main() -> None:
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2 import modeling_qwen2 as Q

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--line-norm", type=float, default=16.0)
    ap.add_argument("--cap", type=float, default=1 / 400)
    ap.add_argument("--budget", type=int, default=512, help="most input channels the split may move")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--tile-layers", default="0,4,8,12,16,20,24,27")
    ap.add_argument("--tiles", type=int, default=2)
    ap.add_argument("--cross-words", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="pearlc_headline.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    path = Path(snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"]))
    cfg = AutoConfig.from_pretrained(path)
    cfg._attn_implementation = "sdpa"
    wmap = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name: str) -> torch.Tensor:
        with safe_open(path / wmap[name], framework="pt") as f:
            return f.get_tensor(name)

    tok = AutoTokenizer.from_pretrained(path)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0][args.window * args.tokens:(args.window + 1) * args.tokens][None]
    h = tensor("model.embed_tokens.weight")[ids[0]].to(torch.float32)[None]
    pos = torch.arange(h.shape[1])[None]
    cos, sin = Q.Qwen2RotaryEmbedding(config=cfg)(h, pos)
    L = cfg.num_hidden_layers
    want = set(range(L)) if args.layers == "all" else {int(x) for x in args.layers.split(",")}
    tile_layers = {int(x) for x in args.tile_layers.split(",") if x != ""}
    out_layers, t0 = [], time.time()
    for i in range(max(want) + 1):
        layer = Q.Qwen2DecoderLayer(cfg, i)
        pre = f"model.layers.{i}."
        layer.load_state_dict({kk[len(pre):]: tensor(kk).to(torch.float32) for kk in wmap if kk.startswith(pre)})
        layer.eval()
        cap: dict = {}
        hk = layer.mlp.down_proj.register_forward_pre_hook(lambda m, inp: cap.setdefault("x", inp[0].detach()))
        with torch.no_grad():
            h = layer(h, position_ids=pos, position_embeddings=(cos, sin))
        hk.remove()
        if i in want:
            X = cap["x"].reshape(-1, cap["x"].shape[-1]).numpy().astype(np.float32)
            Wt = layer.mlp.down_proj.weight.detach().numpy().astype(np.float32)
            out_layers.append(measure(i, X, Wt, args, rng, i in tile_layers))
            print(json.dumps({kk: v for kk, v in out_layers[-1].items() if kk not in ("tiles", "split_trace")}), flush=True)
        del layer
    doc = summarize(out_layers, args, time.time() - t0)
    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    s = doc["summary"]
    run_result.write(out, s["residual_share_max"] <= GAMMA0, "pearlc-headline",
                     f"units admissible unsplit {s['units_admissible_unsplit']}/{s['layers']}; split channels max {s['split_channels_max']}; "
                     f"debit max {s['debit_share_max']:.5f}; gamma at cap (split adopted) {s['gamma_cap_split_adopted_max']:.5f}",
                     [("units_admissible_unsplit", s["units_admissible_unsplit"], "count"),
                      ("split_channels_max", s["split_channels_max"], "count"),
                      ("debit_share_max", s["debit_share_max"], "fraction"),
                      ("residual_share_max", s["residual_share_max"], "fraction"),
                      ("gamma_cap_split_adopted_max", s["gamma_cap_split_adopted_max"], "fraction"),
                      ("gamma_cap_split_uncredited_max", s["gamma_cap_split_uncredited_max"], "fraction")])
    print(json.dumps(s))


def measure(i: int, X: np.ndarray, W: np.ndarray, args, rng, tiles: bool) -> dict:
    m, k = X.shape
    n = W.shape[0]
    fA, lA, _, rA = rules(X, args.delta)
    fB, lB, _, rB = rules(W, args.delta)
    rec = {"layer": i, "m": m, "n": n, "k": k,
           "A_floor_fail": int((~fA).sum()), "A_live_fail": int((~lA).sum()), "A_rows_fail": int((~(fA & lA)).sum()),
           "A_rows_fail_excl_first_token": int((~(fA & lA))[1:].sum()),
           "B_floor_fail": int((~fB).sum()), "B_live_fail": int((~lB).sum()), "B_rows_fail": int((~(fB & lB)).sum()),
           "A_ratio_max": round(float(rA.max()), 1), "A_ratio_p99": round(float(np.percentile(rA, 99)), 1),
           "B_ratio_max": round(float(rB.max()), 1), "B_ratio_p99": round(float(np.percentile(rB, 99)), 1),
           "unit_admissible_unsplit": bool((fA & lA).all() and (fB & lB).all())}
    S, trace = split_channels(X, W, args.delta, args.budget)
    keep = np.setdiff1d(np.arange(k), S)
    fA2, lA2, _, _ = rules(X[:, keep], args.delta)
    fB2, lB2, _, _ = rules(W[:, keep], args.delta)
    k2 = len(keep)
    kpad = -(-k2 // 128) * 128
    rec.update({"split_channels": len(S), "split_channel_share": round(len(S) / k, 6), "split_trace": trace,
                "unit_admissible_split": bool((fA2 & lA2).all() and (fB2 & lB2).all()),
                "A_rows_fail_split": int((~(fA2 & lA2)).sum()), "B_rows_fail_split": int((~(fB2 & lB2)).sum()),
                "k_split": k2, "k_padded": kpad})
    side = 2.0 * n * len(S) / kpad
    merge = 32.0 * n / kpad if S else 0.0
    rec["gamma_cap_split_adopted"] = [round(x, 6) for x in gamma(n, kpad, args.cap)]
    rec["gamma_cap_split_uncredited"] = [round(x, 6) for x in gamma(n, kpad, args.cap, side, merge)]
    rec["gamma_nodebit_split_adopted"] = [round(x, 6) for x in gamma(n, kpad, 0.0)]
    rec["tiles"] = []
    if tiles and rec["unit_admissible_split"] or (tiles and (fA2 & lA2).sum() >= 64 and (fB2 & lB2).sum() >= 8):
        ia, ib = np.flatnonzero(fA2 & lA2), np.flatnonzero(fB2 & lB2)
        FcA = C.enc(C.unit_lines(kpad, rng) * (args.line_norm / np.sqrt(32)))
        FcB = C.enc(C.unit_lines(kpad, rng) * (args.line_norm / np.sqrt(32)))
        for _ in range(args.tiles):
            XA = np.zeros((64, kpad))
            XB = np.zeros((8, kpad))
            XA[:, :k2] = X[rng.choice(ia, 64, replace=False)][:, keep]
            XB[:, :k2] = W[rng.choice(ib, 8, replace=False)][:, keep]
            a, _, _, flA, lvA = C.form_s5(XA, FcA, args.delta, args.line_norm, rng)
            b, _, _, flB, lvB = C.form_s5(XB, FcB, args.delta, args.line_norm, rng)
            ii, jj = np.meshgrid(np.arange(64), np.arange(8), indexing="ij")
            A, B = a[ii.ravel()], b[jj.ravel()]
            T = kpad // 32
            tid, tex = C.chain(A, B, 4)[:2]
            flags, eF = C.debit(A, B, 4)
            size, beyond = C.cross_group(A, B, flags, np.unique(np.linspace(0, len(A) - 1, args.cross_words).astype(int)), 4)
            rec["tiles"].append({"rules_ok_padded": bool(flA and lvA and flB and lvB),
                                 "debit_share": float(flags.sum(1).mean()) / T, "debit_max_word_share": float(flags.sum(1).max()) / T,
                                 "tile_identity_share": float(tid.sum()) / T, "residual_share": C.exact_run_atoms(tex) / T,
                                 "cross_group_share_mean": float(size.mean()) / T, "cross_group_beyond_debit_mean": float(beyond.mean()) / T})
        rec["debit_share_max"] = round(max(t["debit_share"] for t in rec["tiles"]), 6)
        rec["residual_share_max"] = round(max(t["residual_share"] for t in rec["tiles"]), 6)
        rec["cross_group_share_mean"] = round(float(np.mean([t["cross_group_share_mean"] for t in rec["tiles"]])), 6)
        d = rec["debit_share_max"]
        chain = n + n / 4
        credit = chain + 32 + 128 * n / kpad + 64
        rho = (chain * d + 1.0) / credit
        rec["rho_measured"] = round(rho, 6)
        rec["gamma_measured_split_adopted"] = [round(x, 6) for x in gamma(n, kpad, rho)]
    return rec


def summarize(layers: list[dict], args, seconds: float) -> dict:
    tiled = [r for r in layers if r["tiles"]]
    return {"tool": "pouw_pearlc_headline", "model": args.model, "tokens": args.tokens, "window": args.window, "delta": args.delta,
            "line_norm": args.line_norm, "cap": args.cap, "layers": layers, "seconds": round(seconds, 1),
            "summary": {"layers": len(layers),
                        "units_admissible_unsplit": sum(r["unit_admissible_unsplit"] for r in layers),
                        "units_admissible_split": sum(r["unit_admissible_split"] for r in layers),
                        "A_rows_fail_max": max(r["A_rows_fail"] for r in layers),
                        "B_rows_fail_max": max(r["B_rows_fail"] for r in layers),
                        "split_channels_max": max(r["split_channels"] for r in layers),
                        "split_channels_mean": float(np.mean([r["split_channels"] for r in layers])),
                        "debit_share_max": max((r["debit_share_max"] for r in tiled), default=0.0),
                        "residual_share_max": max((r["residual_share_max"] for r in tiled), default=0.0),
                        "cross_group_share_mean": float(np.mean([r["cross_group_share_mean"] for r in tiled])) if tiled else 0.0,
                        "gamma_cap_split_adopted_max": max(r["gamma_cap_split_adopted"][0] for r in layers),
                        "gamma_cap_split_uncredited_max": max(r["gamma_cap_split_uncredited"][0] for r in layers)}}


if __name__ == "__main__":
    main()
