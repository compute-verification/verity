#!/usr/bin/env python3
"""Pearl-C's FP8 quality on a Llama model too large for RAM (Llama-3.1-70B): `pearlc_split7b.py`'s four variants of every
transformer linear, in FP32 on the CPU, streamed one decoder layer at a time from the safetensors (`pearlc_capture.Stream`).

The variants are BF16 (the reference), plain FP8 (per-row α = 448/max), FP8 with the headroom α (α = 448/(s + 4ρ), ρ over
every 8th position: `Costs.adopted`'s quantizer), and FP8 with α plus the fixed channel split (S in BF16, the rest in FP8 with
α on the kept channels), with `pearlc_split7b.py`'s quantizers. S per linear comes from `pearlc_headline.split_channels` (the
pinned greedy rule, budget ⌊k/100⌋) on a fixed calibration slice of WikiText-2 train (the first --calib-tokens tokens of its
first 4,000 lines) run through the BF16 model, before any evaluation token is seen. Evaluation is --eval-windows windows of
--seqlen WikiText-2 test tokens; the final norm and lm_head stay BF16 in every variant. It reports each variant's perplexity,
its change against BF16 with a one-sided 95% bound (windows as blocks), and every linear's |S|, as `pearlc_split7b.py` does.

Restartable: the four hidden states are checkpointed after every decoder layer, and past --budget-min it exits 99. The
self-test, run first, checks the streamed BF16 variant's per-token NLL against the full model's FP32 forward on a tiny random
Llama.

    python benchmarks/pouw/pearlc_quality_llama.py --budget-min 20 --out llama70b-quality
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pearlc_capture as P  # noqa: E402
import pearlc_headline as H  # noqa: E402
from pearlc_split7b import bf16, q_headroom, q_plain  # noqa: E402

LINEARS = [m for mods in P.ROLES.values() for m in mods]
VARIANTS = ("bf16", "fp8", "fp8a", "fp8as")


class Layer:
    """Decoder layer i in FP32 with its linears replaced by the variant `mode` selects (BF16 captures the inputs)."""

    def __init__(self, S: P.Stream, i: int, delta: float):
        self.m = S.load(i).to(torch.float32)
        self.delta, self.mode, self.capture, self.cache = delta, "bf16", None, {}
        self.split: dict[str, torch.Tensor] = {n: torch.zeros(0, dtype=torch.long) for n in LINEARS}
        self.keep: dict[str, torch.Tensor] = {}
        for n in LINEARS:
            mod = self.m.get_submodule(n)
            assert mod.bias is None, f"{n} has a bias"
            mod.forward = lambda x, n=n, W=mod.weight: self.linear(n, W, x)

    def weights(self, n: str, W: torch.Tensor):
        key = (n, self.mode)
        if key not in self.cache:
            S = self.split[n]
            if self.mode == "fp8":
                self.cache[key] = (q_plain(W), None)
            elif self.mode == "fp8a" or not len(S):
                self.cache[key] = (q_headroom(W, self.delta), None)
            else:
                self.cache[key] = (q_headroom(W[:, self.keep[n]], self.delta), W[:, S])
        return self.cache[key]

    def linear(self, n: str, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.capture is not None:
            self.capture[n] = x2.detach().clone()
        S = self.split[n]
        if self.mode == "bf16":
            y = x2 @ W.T
        elif self.mode == "fp8":
            y = q_plain(x2) @ self.weights(n, W)[0].T
        elif self.mode == "fp8a" or not len(S):
            y = q_headroom(x2, self.delta) @ self.weights(n, W)[0].T
        else:
            wq, ws = self.weights(n, W)
            y = q_headroom(x2[:, self.keep[n]], self.delta) @ wq.T + bf16(x2[:, S]) @ ws.T
        return y.reshape(*shape[:-1], -1)

    def run(self, mode: str, h, pos, rope, mask, capture: bool = False):
        self.mode, self.capture = mode, ({} if capture else None)
        with torch.no_grad():
            out = self.m(h, attention_mask=mask, position_ids=pos, position_embeddings=rope)
        self.cache = {k: v for k, v in self.cache.items() if k[1] != mode}
        return out[0] if isinstance(out, tuple) else out

    def calibrate(self, X: dict[str, torch.Tensor]) -> dict:
        rec = {}
        for n in LINEARS:
            W = self.m.get_submodule(n).weight.detach().numpy()
            Xc = X[n].numpy()
            k = W.shape[1]
            S, _ = H.split_channels(Xc, W, self.delta, k // 100)
            fl, lv, _, _ = H.rules(Xc, self.delta)
            self.split[n] = torch.tensor(sorted(S), dtype=torch.long)
            self.keep[n] = torch.tensor(np.setdiff1d(np.arange(k), S), dtype=torch.long)
            rec[n] = {"k": k, "split_channels": len(S), "budget": k // 100, "calib_rows_fail": int((~(fl & lv)).sum())}
        return rec


def head_nll(S: P.Stream, cfg, h: torch.Tensor, ev: torch.Tensor) -> np.ndarray:
    norm = transformers.models.llama.modeling_llama.LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight = torch.nn.Parameter(S.tensor("model.norm.weight").to(torch.float32))
    key = "lm_head.weight" if "lm_head.weight" in S.wmap else "model.embed_tokens.weight"
    head = S.tensor(key).to(torch.float32)
    with torch.no_grad():
        hn = norm(h)
        per = [F.cross_entropy((hn[w] @ head.T)[:-1], ev[w, 1:], reduction="none") for w in range(len(ev))]
    return torch.cat(per).numpy().astype(np.float64)


def quality(nll: dict, windows: int) -> dict:
    res = {}
    for v in VARIANTS:
        d = nll[v] - nll["bf16"]
        blocks = d.reshape(windows, -1).mean(1)
        se = float(blocks.std(ddof=1) / math.sqrt(windows)) if windows > 1 else float("nan")
        res[v] = {"ppl": float(math.exp(nll[v].mean())), "vs_bf16_pct": float(100 * (math.exp(d.mean()) - 1)),
                  "vs_bf16_pct_upper95": float(100 * (math.exp(d.mean() + 1.645 * se) - 1)), "tokens": len(d)}
    for a, b in (("fp8a", "fp8"), ("fp8as", "fp8a"), ("fp8as", "fp8")):
        d = nll[a] - nll[b]
        blocks = d.reshape(windows, -1).mean(1)
        res[f"{a}_vs_{b}"] = {"pct": float(100 * (math.exp(d.mean()) - 1)),
                              "pct_se": float(100 * blocks.std(ddof=1) / math.sqrt(windows)) if windows > 1 else float("nan")}
    return res


def selftest(seed: int = 0) -> dict:
    """The streamed BF16 variant's per-token NLL against the full model's FP32 forward on a tiny random Llama."""
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=352, num_hidden_layers=3,
                                   num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024,
                                   rope_theta=500000.0, tie_word_embeddings=False,
                                   rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                                 "high_freq_factor": 4.0, "original_max_position_embeddings": 64})
    with tempfile.TemporaryDirectory() as d:
        transformers.LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d)
        full = transformers.LlamaForCausalLM.from_pretrained(d, dtype=torch.float32, attn_implementation="sdpa").eval()
        ev = torch.randint(0, cfg.vocab_size, (2, 96))
        with torch.no_grad():
            logits = full(ev, use_cache=False).logits
        ref = torch.cat([F.cross_entropy(logits[w, :-1], ev[w, 1:], reduction="none") for w in range(2)]).numpy()
        scfg = transformers.AutoConfig.from_pretrained(d)
        scfg._attn_implementation = "sdpa"
        S = P.Stream(Path(d), scfg)
        h, pos, rope, mask = S.start(ev, torch.float32)
        hq = h
        for i in range(cfg.num_hidden_layers):
            lay = Layer(S, i, 1.0)
            h = lay.run("bf16", h, pos, rope, mask)
            hq = lay.run("fp8a", hq, pos, rope, mask)
        got, got_q = head_nll(S, scfg, h, ev), head_nll(S, scfg, hq, ev)
    err = float(np.abs(got - ref).max())
    return {"passed": err <= 1e-4 and float(np.abs(got_q - got).max()) > 0, "nll_max_abs_diff": err,
            "fp8a_differs": float(np.abs(got_q - got).max()), "tokens": len(ref),
            "config": "Llama, hidden 128, intermediate 352, 3 layers, 8 heads (2 kv), llama3 rope scaling, FP32 on BF16 weights"}


def main() -> int:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-70B", help="a Hub repository, or a local model directory")
    ap.add_argument("--revision", default="1b7306651142d0cc65d993076a250a6a82cf046c")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--eval-windows", type=int, default=4)
    ap.add_argument("--calib-tokens", type=int, default=512)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--budget-min", type=float, default=0.0, help="exit 99 past this many minutes while layers remain (0: no limit)")
    ap.add_argument("--out", default="pearlc_quality_llama")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    st = selftest()
    print(json.dumps({"selftest": st}), flush=True)
    if not st["passed"]:
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(args.model) if Path(args.model).is_dir() else \
        Path(snapshot_download(args.model, revision=args.revision, allow_patterns=["*.json", "tokenizer*"]))
    cfg = transformers.AutoConfig.from_pretrained(path)
    cfg._attn_implementation = "sdpa"
    S = P.Stream(path, cfg)
    ids_file = out / "ids.npz"
    if not ids_file.exists():
        tok = transformers.AutoTokenizer.from_pretrained(path)
        test = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]),
                   return_tensors="np").input_ids[0]
        train = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"][:4000]),
                    return_tensors="np").input_ids[0]
        T = args.seqlen
        np.savez(ids_file, ev=np.stack([test[w * T:(w + 1) * T] for w in range(args.eval_windows)]).astype(np.int64),
                 ca=train[:args.calib_tokens][None].astype(np.int64))
    z = np.load(ids_file)
    ev, ca = torch.from_numpy(z["ev"]), torch.from_numpy(z["ca"])
    he, pos_e, rope_e, mask_e = S.start(ev, torch.float32)
    hc, pos_c, rope_c, mask_c = S.start(ca, torch.float32)
    hs = {v: he for v in VARIANTS}
    state = out / "state.pt"
    nxt, spent, linears = 0, 0.0, {}
    if state.exists():
        ck = torch.load(state)
        hs, hc, nxt, spent, linears = ck["hs"], ck["hc"], ck["next"], ck["seconds"], ck["linears"]
    for i in range(nxt, cfg.num_hidden_layers):
        lay = Layer(S, i, args.delta)
        hc = lay.run("bf16", hc, pos_c, rope_c, mask_c, capture=True)
        rec = lay.calibrate(lay.capture)
        for v in VARIANTS:
            hs[v] = lay.run(v, hs[v], pos_e, rope_e, mask_e)
        linears.update({f"model.layers.{i}.{n}": r for n, r in rec.items()})
        del lay
        torch.save({"hs": hs, "hc": hc, "next": i + 1, "seconds": spent + time.time() - t0, "linears": linears}, out / "state.tmp")
        (out / "state.tmp").replace(state)
        print(json.dumps({"layer": i, "split": {n: r["split_channels"] for n, r in rec.items()},
                          "seconds": round(time.time() - t0, 1)}), flush=True)
        if args.budget_min and time.time() - t0 > 60 * args.budget_min and i < cfg.num_hidden_layers - 1:
            return P.MORE
    nll = {v: head_nll(S, cfg, hs[v], ev) for v in VARIANTS}
    res = quality(nll, args.eval_windows)
    ls = list(linears.values())
    doc = {"tool": "pouw_pearlc_quality_llama", "model": args.model, "revision": args.revision, "snapshot": str(path),
           "seqlen": args.seqlen, "eval_windows": args.eval_windows, "eval": "wikitext-2-raw-v1:test, windows from token 0",
           "calib": f"wikitext-2-raw-v1:train, first {args.calib_tokens} tokens of the first 4000 lines", "delta": args.delta,
           "budget": "floor(k/100)", "lm_head": "BF16 in every variant", "arithmetic": "FP32 on the CPU, BF16 weights",
           "torch": torch.__version__, "transformers": transformers.__version__, "threads": args.threads, "selftest": st,
           "quality": res, "linears": linears,
           "summary": {"linears": len(ls), "linears_with_split": sum(r["split_channels"] > 0 for r in ls),
                       "split_channels_max": max(r["split_channels"] for r in ls),
                       "split_at_budget": sum(r["split_channels"] == r["budget"] for r in ls)},
           "seconds": round(spent + time.time() - t0, 1)}
    (out / "quality.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"quality": res, "summary": doc["summary"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
