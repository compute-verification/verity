#!/usr/bin/env python3
"""Pearl-C's domain rules and replayed debit on real activations: every linear layer of Qwen2.5-0.5B on WikiText-2 test.

Each linear layer's first input X (the window's token rows) and its weight W are read in FP32. For both, per row and under
§5's statistic (ρ over every 8th position, σ = δ·ρ, α = 448/(s + 4σ)): the noise floor σ·α ≥ 1 and liveness (at most k/64
entries with |x| ≥ 8σ), the rules `OperandOK` applies. A unit is admissible only if all its rows are, so the fraction of
64-row token tiles and 8-row weight tiles with every row admissible is reported too. On sampled tiles of admissible rows,
64 tokens × 8 output channels, §5's forming (bit-exact atom-mix, E4M3 noise lines of norm --line-norm, one cast) and the
G4 chain give each word's replayed debit (`pearlc_census.debit`: eF and the skip-one count) and the tile's undebited
residual (exact runs across the tile).

    python benchmarks/pouw/pearlc_activations.py --tiles 2 --out pearlc_activations.json

Under `research run` it writes result.json: passed when every sampled tile's residual is at most 1/400.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pearlc_census as C  # noqa: E402
import run_result  # noqa: E402


def row_rules(X: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = X.shape[1]
    s = np.abs(X).max(1)
    sig = delta * np.sqrt((X[:, ::8] ** 2).mean(1))
    alpha = np.where(s + 4 * sig > 0, C.Q_MAX / np.maximum(s + 4 * sig, 1e-300), 0.0)
    floor = sig * alpha >= 1.0
    live = (np.abs(X) >= 8 * sig[:, None]).sum(1) * 64 <= k
    return floor, live, s / np.maximum(sig, 1e-300)


def tiles_ok(ok: np.ndarray, size: int) -> float:
    n = len(ok) // size
    return float(ok[: n * size].reshape(n, size).all(1).mean()) if n else 0.0


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--line-norm", type=float, default=16.0)
    ap.add_argument("--tiles", type=int, default=2, help="sampled 64x8 tiles of admissible rows per layer")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default="pearlc_activations.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunk = ids[args.window * args.seqlen:(args.window + 1) * args.seqlen].unsqueeze(0)
    rng = np.random.default_rng(args.seed)
    layers, t0 = [], time.time()

    def measure(name: str, X: np.ndarray, W: np.ndarray) -> dict:
        k = X.shape[1]
        fA, lA, rA = row_rules(X, args.delta)
        fB, lB, rB = row_rules(W, args.delta)
        okA, okB = fA & lA, fB & lB
        rec = {"layer": name, "k": k, "rows_A": len(X), "rows_B": len(W), "tiled": k % 128 == 0,
               "A_floor_fail": round(1 - float(fA.mean()), 6), "A_live_fail": round(1 - float(lA.mean()), 6),
               "B_floor_fail": round(1 - float(fB.mean()), 6), "B_live_fail": round(1 - float(lB.mean()), 6),
               "A_ratio_p50": round(float(np.median(rA)), 2), "A_ratio_max": round(float(rA.max()), 2),
               "B_ratio_max": round(float(rB.max()), 2),
               "A_tiles64_admissible": round(tiles_ok(okA, 64), 6), "B_tiles8_admissible": round(tiles_ok(okB, 8), 6)}
        ia, ib = np.flatnonzero(okA), np.flatnonzero(okB)
        tiles = []
        if k % 128 == 0 and len(ia) >= 64 and len(ib) >= 8:
            FcA = C.enc(C.unit_lines(k, rng) * (args.line_norm / np.sqrt(32)))
            FcB = C.enc(C.unit_lines(k, rng) * (args.line_norm / np.sqrt(32)))
            for _ in range(args.tiles):
                XA, XB = X[rng.choice(ia, 64, replace=False)], W[rng.choice(ib, 8, replace=False)]
                a = C.form_s5(XA, FcA, args.delta, args.line_norm, rng)[0]
                b = C.form_s5(XB, FcB, args.delta, args.line_norm, rng)[0]
                ii, jj = np.meshgrid(np.arange(64), np.arange(8), indexing="ij")
                A, B = a[ii.ravel()], b[jj.ravel()]
                T = k // 32
                tid, tex, _, _ = C.chain(A, B, 4)
                eF, ex = C.debit(A, B, 4)
                tiles.append({"debit_share": float((eF + ex).mean()) / T, "debit_eF_share": float(eF.mean()) / T,
                              "debit_extra_share": float(ex.mean()) / T, "debit_max_word_share": float((eF + ex).max()) / T,
                              "tile_identity_share": float(tid.sum()) / T, "residual_share": C.exact_run_atoms(tex) / T})
        rec["tiles"] = tiles
        if tiles:
            rec["debit_share_mean"] = round(float(np.mean([t["debit_share"] for t in tiles])), 6)
            rec["debit_share_max"] = round(float(max(t["debit_share"] for t in tiles)), 6)
            rec["residual_share_max"] = round(float(max(t["residual_share"] for t in tiles)), 6)
        print(json.dumps({k2: v for k2, v in rec.items() if k2 != "tiles"}), flush=True)
        return rec

    def hook(mod, inputs, name=None):
        if getattr(mod, "_pc_done", False):
            return
        mod._pc_done = True
        X = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float32).numpy().astype(np.float64)
        W = mod.weight.detach().to(torch.float32).numpy().astype(np.float64)
        layers.append(measure(name, X, W))

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.register_forward_pre_hook(lambda m, i, name=name: hook(m, i, name))
    with torch.no_grad():
        model(chunk)
    measured = [r for r in layers if r.get("tiles")]
    worst_resid = max((r["residual_share_max"] for r in measured), default=0.0)
    doc = {"tool": "pouw_pearlc_activations", "model": args.model, "dataset": "wikitext-2-raw-v1:test", "window": args.window,
           "seqlen": args.seqlen, "delta": args.delta, "line_norm": args.line_norm, "layers": layers,
           "summary": {"layers": len(layers), "layers_measured": len(measured),
                       "A_rows_inadmissible_max": max(max(r["A_floor_fail"], r["A_live_fail"]) for r in layers),
                       "A_tiles64_admissible_min": min(r["A_tiles64_admissible"] for r in layers),
                       "B_tiles8_admissible_min": min(r["B_tiles8_admissible"] for r in layers),
                       "debit_share_max": max((r["debit_share_max"] for r in measured), default=0.0),
                       "debit_share_mean": float(np.mean([r["debit_share_mean"] for r in measured])) if measured else 0.0,
                       "residual_share_max": worst_resid},
           "seconds": round(time.time() - t0, 1)}
    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    s = doc["summary"]
    run_result.write(out, worst_resid <= 1 / 400, "pearlc-activations",
                     f"debit share max {s['debit_share_max']:.5f}, mean {s['debit_share_mean']:.5f}; residual max {worst_resid:.5f}; "
                     f"admissible 64-row A tiles min {s['A_tiles64_admissible_min']:.3f}",
                     [("debit_share_max", s["debit_share_max"], "fraction"), ("debit_share_mean", s["debit_share_mean"], "fraction"),
                      ("residual_share_max", worst_resid, "fraction"),
                      ("A_tiles64_admissible_min", s["A_tiles64_admissible_min"], "fraction")])
    print(json.dumps(doc["summary"]))


if __name__ == "__main__":
    main()
