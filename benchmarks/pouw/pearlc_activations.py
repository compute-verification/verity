#!/usr/bin/env python3
"""Pearl-C's domain rules and replayed debit on real activations: every linear layer of Qwen2.5-0.5B on WikiText-2 test.

Each linear layer's first input X (the window's token rows) and its weight W are read in FP32. For both, per row and under
§5's statistic (ρ over every 8th position, σ = δ·ρ, α = 448/(s + 4σ)): the noise floor σ·α ≥ 1 and liveness (at most k/64
entries with |x| ≥ 8σ), the rules `OperandOK` applies. A unit is admissible only if all its rows are, so the fraction of
64-row token tiles and 8-row weight tiles with every row admissible is reported too. On sampled tiles of admissible rows,
64 tokens × 8 output channels, §5's forming (bit-exact atom-mix, E4M3 noise lines of norm --line-norm, one cast) and the
G4 chain give each word's replayed debit (`pearlc_census.debit`, P1's option (A): every atom of a within-group joint skip
that leaves the ticket word unchanged), P3's not-needed share, the tile's undebited residual (exact runs across it), and
P1's greedy cross-group joint set on --cross-words words per tile (`pearlc_census.cross_group`), which no debit covers.

    python benchmarks/pouw/pearlc_activations.py --tiles 2 --out pearlc_activations.json
    python benchmarks/pouw/pearlc_activations.py --atom sm120-e4m3-k32 --tiles 2 --out pearlc_activations_sm120.json

A larger model runs as parallel shards, one process per --layers regex over the module names (each runs the whole forward
pass and measures only its layers), e.g. Qwen2.5-7B's down_proj four layers at a time:

    python benchmarks/pouw/pearlc_activations.py --model Qwen/Qwen2.5-7B --atom sm120 --G 0 \
        --layers 'layers\\.([0-3])\\.mlp\\.down_proj$' --out down-0-3.json

--replay DIR reads X from a `pearlc_capture.py` directory (its manifest's sha256s checked) and W from the model's safetensors
in the manifest's snapshot, without torch or a forward pass: a model too large for one process (Llama-3.1-70B) is captured
once and replayed per layer. --split adds the channel split to admission (`pearlc_headline.split_channels`, the pinned greedy
rule, |S| ≤ ⌊k/100⌋): the unit's rows are judged again without S, and the tiles are drawn from the split unit's admissible
rows, zero-padded to a multiple of 128 channels as `pearlc_headline.py` pads them.

Under `research run` it writes result.json: passed when every sampled tile's residual is at most 1/400.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pearlc_census as C  # noqa: E402
import pearlc_headline as H  # noqa: E402
import run_result  # noqa: E402

ST_DTYPES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}


def safetensor(path: Path, name: str) -> np.ndarray:
    """One tensor of a .safetensors file as float32 (from BF16, F16 or F32), read without torch."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        meta = json.loads(f.read(n))[name]
        lo, hi = meta["data_offsets"]
        f.seek(8 + n + lo)
        raw = np.frombuffer(f.read(hi - lo), dtype=ST_DTYPES[meta["dtype"]])
    if meta["dtype"] == "BF16":
        raw = (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32).reshape(meta["shape"])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def replay(rd: Path, regex: str):
    """(module name, X, W) for each captured linear whose name matches `regex`, in the manifest's order."""
    man = json.loads((rd / "manifest.json").read_text())
    snap = Path(man["snapshot"])
    wmap = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    for name, ent in man["inputs"].items():
        if not re.search(regex, name):
            continue
        f = rd / ent["file"]
        if sha256(f) != ent["sha256"]:
            raise SystemExit(f"{f}: sha256 differs from {rd / 'manifest.json'}")
        wname = name + ".weight"
        yield name, np.load(f).astype(np.float64), safetensor(snap / wmap[wname], wname).astype(np.float64)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--line-norm", type=float, default=16.0)
    ap.add_argument("--tiles", type=int, default=2, help="sampled 64x8 tiles of admissible rows per layer")
    ap.add_argument("--cross-words", type=int, default=4, help="words per tile for P1's greedy cross-group search")
    ap.add_argument("--transfer", action="store_true", help="the reviewer's transfer test on the cross-group sample words")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--atom", default="hopper-e4m3-k32", help="the FP8 atom (pearlc_census --atom: a name, an alias or groups:width:zero)")
    ap.add_argument("--G", type=int, default=4, help="promotion period in atoms; 0: no promotion (sm_120 v2)")
    ap.add_argument("--layers", default="", help="measure only the linear layers whose module name this regex matches (a shard)")
    ap.add_argument("--replay", default="", help="a pearlc_capture.py directory: X from its manifest, W from the model's safetensors")
    ap.add_argument("--split", action="store_true", help="admission with the channel split, |S| <= k/100; tiles from the split unit")
    ap.add_argument("--out", default="pearlc_activations.json")
    args = ap.parse_args()
    C.set_atom(args.atom)
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
               "A_tiles64_admissible": round(tiles_ok(okA, 64), 6), "B_tiles8_admissible": round(tiles_ok(okB, 8), 6),
               "A_rows_fail": int((~okA).sum()), "A_row0_fail": bool(not okA[0]), "B_rows_fail": int((~okB).sum()),
               "unit_admissible_unsplit": bool(okA.all() and okB.all())}
        kpad = k
        if args.split:
            S, trace = H.split_channels(X.astype(np.float32), W.astype(np.float32), args.delta, k // 100)
            keep = np.setdiff1d(np.arange(k), S)
            X, W = X[:, keep], W[:, keep]
            fA, lA, _ = row_rules(X, args.delta)
            fB, lB, _ = row_rules(W, args.delta)
            okA, okB = fA & lA, fB & lB
            kpad = -(-len(keep) // 128) * 128
            rec.update({"split_budget": k // 100, "split_channels": len(S), "split_rounds": len(trace), "k_split": len(keep),
                        "k_padded": kpad, "A_rows_fail_split": int((~okA).sum()), "A_row0_fail_split": bool(not okA[0]),
                        "B_rows_fail_split": int((~okB).sum()), "unit_admissible_split": bool(okA.all() and okB.all()),
                        "A_tiles64_admissible_split": round(tiles_ok(okA, 64), 6),
                        "B_tiles8_admissible_split": round(tiles_ok(okB, 8), 6)})
        ia, ib = np.flatnonzero(okA), np.flatnonzero(okB)
        tiles = []
        if kpad % 128 == 0 and len(ia) >= 64 and len(ib) >= 8:
            FcA = C.enc(C.unit_lines(kpad, rng) * (args.line_norm / np.sqrt(32)))
            FcB = C.enc(C.unit_lines(kpad, rng) * (args.line_norm / np.sqrt(32)))
            for _ in range(args.tiles):
                XA, XB = np.zeros((64, kpad)), np.zeros((8, kpad))
                XA[:, :X.shape[1]] = X[rng.choice(ia, 64, replace=False)]
                XB[:, :W.shape[1]] = W[rng.choice(ib, 8, replace=False)]
                a, _, _, flA, lvA = C.form_s5(XA, FcA, args.delta, args.line_norm, rng)
                b, _, _, flB, lvB = C.form_s5(XB, FcB, args.delta, args.line_norm, rng)
                ii, jj = np.meshgrid(np.arange(64), np.arange(8), indexing="ij")
                A, B = a[ii.ravel()], b[jj.ravel()]
                T = kpad // 32
                tid, tex = C.chain(A, B, args.G)[:2]
                sample = np.unique(np.linspace(0, len(A) - 1, args.cross_words).astype(int))
                if args.G:
                    flags, eF = C.debit(A, B, args.G)
                    size, beyond = C.cross_group(A, B, flags, sample, args.G)
                else:
                    flags = C.debit_pure(A, B)
                    eF = flags.sum(1)
                    sets = [C.greedy_pure(A[p:p + 1], B[p:p + 1]) for p in sample]
                    size = np.array([len(S) for S in sets])
                    beyond = np.array([sum(1 for t in S if not flags[p, t]) for S, p in zip(sets, sample)])
                per_word = flags.sum(1)
                tr = {}
                if args.transfer and args.G:
                    a_sim = C.form_s5(XA, FcA, args.delta, args.line_norm, rng)[0]
                    b_sim = C.form_s5(XB, FcB, args.delta, args.line_norm, rng)[0]
                    tr = C.transfer(a_sim[ii.ravel()], b_sim[jj.ravel()], A, B, flags, sample, 4)
                tiles.append({"debit_share": float(per_word.mean()) / T, "debit_eF_share": float(eF.mean()) / T,
                              "debit_max_word_share": float(per_word.max()) / T, "not_needed_share": C.not_needed_share(flags, 64, 8),
                              "tile_identity_share": float(tid.sum()) / T, "residual_share": C.exact_run_atoms(tex) / T,
                              "cross_group_share_mean": float(size.mean()) / T, "cross_group_share_max": float(size.max()) / T,
                              "cross_group_beyond_debit_mean": float(beyond.mean()) / T, **tr,
                              **({"rules_ok_padded": bool(flA and lvA and flB and lvB)} if args.split else {})})
        rec["tiles"] = tiles
        if tiles:
            rec["debit_share_mean"] = round(float(np.mean([t["debit_share"] for t in tiles])), 6)
            rec["debit_share_max"] = round(float(max(t["debit_share"] for t in tiles)), 6)
            rec["residual_share_max"] = round(float(max(t["residual_share"] for t in tiles)), 6)
            rec["cross_group_share_mean"] = round(float(np.mean([t["cross_group_share_mean"] for t in tiles])), 6)
            rec["cross_group_share_max"] = round(float(max(t["cross_group_share_max"] for t in tiles)), 6)
        print(json.dumps({k2: v for k2, v in rec.items() if k2 != "tiles"}), flush=True)
        return rec

    source = {}
    if args.replay:
        rd = Path(args.replay)
        man = json.loads((rd / "manifest.json").read_text())
        args.model, args.window, args.seqlen = man["model"], man["window"], man["tokens"]
        source = {"replay": str(rd), "manifest_sha256": sha256(rd / "manifest.json"), "revision": man["revision"]}
        for name, X, W in replay(rd, args.layers):
            layers.append(measure(name, X, W))
    else:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(args.threads)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        model.eval()
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        ids = tok(text, return_tensors="pt").input_ids[0]
        chunk = ids[args.window * args.seqlen:(args.window + 1) * args.seqlen].unsqueeze(0)

        def hook(mod, inputs, name=None):
            if getattr(mod, "_pc_done", False):
                return
            mod._pc_done = True
            X = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float32).numpy().astype(np.float64)
            W = mod.weight.detach().to(torch.float32).numpy().astype(np.float64)
            layers.append(measure(name, X, W))

        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and re.search(args.layers, name):
                mod.register_forward_pre_hook(lambda m, i, name=name: hook(m, i, name))
        with torch.no_grad():
            model(chunk)
    measured = [r for r in layers if r.get("tiles")]
    worst_resid = max((r["residual_share_max"] for r in measured), default=0.0)
    split = {}
    if args.split:
        split = {"units_admissible_split": sum(r["unit_admissible_split"] for r in layers),
                 "split_channels_max": max(r["split_channels"] for r in layers),
                 "split_at_budget": sum(r["split_channels"] == r["split_budget"] for r in layers),
                 "A_rows_fail_split_max": max(r["A_rows_fail_split"] for r in layers),
                 "B_rows_fail_split_max": max(r["B_rows_fail_split"] for r in layers)}
    doc = {"tool": "pouw_pearlc_activations", "atom": args.atom, "G": args.G, "layers_regex": args.layers, "model": args.model, "dataset": "wikitext-2-raw-v1:test", "window": args.window,
           "seqlen": args.seqlen, "delta": args.delta, "line_norm": args.line_norm, "split": args.split, **source, "layers": layers,
           "summary": {"layers": len(layers), "layers_measured": len(measured),
                       "A_rows_inadmissible_max": max(max(r["A_floor_fail"], r["A_live_fail"]) for r in layers),
                       "A_tiles64_admissible_min": min(r["A_tiles64_admissible"] for r in layers),
                       "B_tiles8_admissible_min": min(r["B_tiles8_admissible"] for r in layers),
                       "units_admissible_unsplit": sum(r["unit_admissible_unsplit"] for r in layers),
                       "A_rows_fail_max": max(r["A_rows_fail"] for r in layers),
                       "A_row0_fail": sum(r["A_row0_fail"] for r in layers), **split,
                       "debit_share_max": max((r["debit_share_max"] for r in measured), default=0.0),
                       "debit_share_mean": float(np.mean([r["debit_share_mean"] for r in measured])) if measured else 0.0,
                       "residual_share_max": worst_resid,
                       "cross_group_share_mean": float(np.mean([r["cross_group_share_mean"] for r in measured])) if measured else 0.0,
                       "cross_group_share_max": max((r["cross_group_share_max"] for r in measured), default=0.0)},
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
