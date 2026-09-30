#!/usr/bin/env python3
"""The fork's debit census on real activations with keyed-rotated weights (CPU, bit-exact sm_120 chain).

GPU 3's pearlc_activations.py (the shipped tree 5f1cb93e, benchmarks/pouw) measures Pearl-C's per-tile debit on Qwen2.5-7B's
real activations with noise on both operands. This is the fork's version: the activations are formed as today (§5's forming,
delta 1, norm-16 lines), and the weights carry no noise: they are the deployed codes, per-row E4M3 of either the checkpoint
as registered ('plain') or the keyed rotation ('rot': y = (x R)(W R)^T for q/k/v/gate/up, whose activations are rotated
too, and y = x (R^T W)^T for o/down). Per sampled 64 x 8 tile: v1's debit (P1's jointFlags under G = 4) with cross-group
sets and the transfer test, and v2's lone-skip flags (debit_pure, G = 0), against the caps 1/400 and 1/1,000.

Chunks: layer regexes, one JSON each (the checkpoint). Exit 0 when all are done, 99 while any remain, 143 when preempted;
$RESEARCH_RUN_DIR/result.json in research/result/v0.1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

PRE = {"flag": False}
CHUNKS = [("attn-0-13", r"layers\.([0-9]|1[0-3])\.self_attn\."), ("attn-14-27", r"layers\.(1[4-9]|2[0-7])\.self_attn\."),
          ("mlp-0-6", r"layers\.[0-6]\.mlp\."), ("mlp-7-13", r"layers\.([7-9]|1[0-3])\.mlp\."),
          ("mlp-14-20", r"layers\.(1[4-9]|20)\.mlp\."), ("mlp-21-27", r"layers\.2[1-7]\.mlp\.")]


def on_term(signum, frame):
    PRE["flag"] = True


def rot_np(d, seed):
    g = torch.Generator().manual_seed(seed * 1000003 + d)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))).numpy()


def main():
    signal.signal(signal.SIGTERM, on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("--census-src", default="/workspace/research/src/5f1cb93eba16e9117b238650cf2c0d71c2757501/benchmarks/pouw")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiles", type=int, default=1)
    ap.add_argument("--cross-words", type=int, default=4)
    ap.add_argument("--layer-stride", type=int, default=2, help="measure every n-th matching linear")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-seconds", type=float, default=1200)
    args = ap.parse_args()
    sys.path.insert(0, args.census_src)
    import pearlc_census as C
    from pearlc_activations import row_rules
    C.set_atom("sm120")
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    names = [c for c, _ in CHUNKS]
    todo = [(c, rx) for c, rx in CHUNKS if not (out / f"{c}.json").exists()]
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if todo:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(args.threads)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        model.eval()
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        chunk_ids = tok(text, return_tensors="pt").input_ids[0][:2048].unsqueeze(0)
        R = rot_np(model.config.hidden_size, args.seed)
    for cname, rx in todo:
        if PRE["flag"] or time.time() - t0 > args.max_seconds:
            break
        rng = np.random.default_rng(args.seed + names.index(cname))
        recs, seen = [], {"n": 0}

        def measure(name, X, W):
            k = X.shape[1]
            rec = {"layer": name, "k": k}
            out_rot = name.endswith(("o_proj", "down_proj"))
            for mode in ("plain", "rot"):
                if mode == "rot" and out_rot:
                    Xm, Wm = X, R.T @ W
                elif mode == "rot":
                    Xm, Wm = X @ R, W @ R
                else:
                    Xm, Wm = X, W
                fA, lA, _ = row_rules(Xm, 1.0)
                ia = np.flatnonzero(fA & lA)
                if k % 128 or len(ia) < 64:
                    rec[mode] = {"skipped": "fewer than 64 admissible rows"}
                    continue
                FcA = C.enc(C.unit_lines(k, rng) * (16.0 / np.sqrt(32)))
                tiles = []
                for _ in range(args.tiles):
                    XA = Xm[rng.choice(ia, 64, replace=False)]
                    WB = Wm[rng.choice(len(Wm), 8, replace=False)]
                    a = C.form_s5(XA, FcA, 1.0, 16.0, rng)[0]
                    b = C.enc(WB * (C.Q_MAX / np.abs(WB).max(1, keepdims=True)))
                    ii, jj = np.meshgrid(np.arange(64), np.arange(8), indexing="ij")
                    A, B = a[ii.ravel()], b[jj.ravel()]
                    T = k // 32
                    sample = np.unique(np.linspace(0, len(A) - 1, args.cross_words).astype(int))
                    flags, eF = C.debit(A, B, 4)
                    size, beyond = C.cross_group(A, B, flags, sample, 4)
                    a_sim = C.form_s5(XA, FcA, 1.0, 16.0, rng)[0]
                    tr = C.transfer(a_sim[ii.ravel()], B, A, B, flags, sample, 4)
                    fpure = C.debit_pure(A, B)
                    tiles.append({"v1_debit_share": float(flags.sum(1).mean()) / T,
                                  "v1_debit_max_word": float(flags.sum(1).max()) / T,
                                  "v1_cross_group_mean": float(size.mean()) / T, "v1_cross_group_max": float(size.max()) / T,
                                  "v2_debit_share": float(fpure.sum(1).mean()) / T,
                                  "v2_debit_max_word": float(fpure.sum(1).max()) / T, **{f"transfer_{k2}": v for k2, v in tr.items()}})
                rec[mode] = {"tiles": tiles, "B_ratio_max": float((np.abs(Wm).max(1) / np.sqrt((Wm[:, ::8] ** 2).mean(1))).max())}
            print(json.dumps({"layer": name, **{m: {k2: round(v, 6) for k2, v in rec[m]["tiles"][0].items() if isinstance(v, float)}
                                                for m in ("plain", "rot") if "tiles" in rec.get(m, {})}}), flush=True)
            return rec

        def hook(mod, inputs, name=None):
            if getattr(mod, "_aw_done", False) or PRE["flag"]:
                return
            mod._aw_done = True
            seen["n"] += 1
            if (seen["n"] - 1) % args.layer_stride:
                return
            X = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().numpy().astype(np.float64)
            W = mod.weight.detach().float().numpy().astype(np.float64)
            recs.append(measure(name, X, W))

        handles = [mod.register_forward_pre_hook(lambda m, i, name=name: hook(m, i, name))
                   for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and re.search(rx, name)]
        for _, mod in model.named_modules():
            mod._aw_done = False
        with torch.no_grad():
            model(chunk_ids)
        for h in handles:
            h.remove()
        if PRE["flag"]:
            break
        doc = {"chunk": cname, "layers_regex": rx, "model": args.model, "atom": "sm120", "census_src": args.census_src,
               "records": recs, "seconds": round(time.time() - t0, 1)}
        (out / f"{cname}.json").write_text(json.dumps(doc, indent=1))
    done = all((out / f"{c}.json").exists() for c in names)
    if rd:
        write_result(Path(rd), out, names)
    if PRE["flag"] and not done:
        return 143
    return 0 if done else 99


def write_result(run, out, names):
    meas, files, bad, agg = [], [], [], {}
    for c in names:
        f = out / f"{c}.json"
        if not f.exists():
            bad.append(f"{c}: missing"); continue
        try:
            files.append(str(f.resolve().relative_to(run.resolve())))
        except ValueError:
            pass
        for r in json.loads(f.read_text())["records"]:
            for mode in ("plain", "rot"):
                for t in r.get(mode, {}).get("tiles", []):
                    for key in ("v1_debit_share", "v1_debit_max_word", "v1_cross_group_mean", "v2_debit_share"):
                        agg.setdefault((mode, key), []).append(t[key])
    for (mode, key), vals in sorted(agg.items()):
        meas += [{"name": f"{mode}/{key}/mean", "value": float(np.mean(vals)), "unit": "fraction of atoms"},
                 {"name": f"{mode}/{key}/max", "value": float(np.max(vals)), "unit": "fraction of atoms"}]
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"workload": "approved-weights/debit/v1", "source_sha": os.environ.get("RESEARCH_SOURCE_SHA", ""),
                                    "chunks": names},
           "validation": {"status": "failed" if bad else "passed", "checker": "approved-weights/debit-completeness/v1",
                          "detail": "; ".join(bad) if bad else f"all {len(names)} chunks present", "evidence": files},
           "measurements": meas, "measurement_files": files}
    tmp = run / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run / "result.json")


if __name__ == "__main__":
    sys.exit(main())
