"""N-RED's N1 on real data: what an adaptive prover saves on ssn_1l with Llama-3.2-1B's int8 activations (WikiText-2).

`nred_attacks.py` prices A1 (identical rows), A1b (shared prefixes) and A2 (zero tails) on chosen inputs. This measures
the same prover on served traffic. Each fused linear's input is hooked as in `noise_data.py` (the same quantization,
windows and scopes: m = 64, one window of ctx rows, all windows as one call). For every row and ssn_1l step t the key
is (the data prefix through step t, with all-zero blocks carried over from the step before, a₁, c_t), from coin words
drawn and read by the served extraction (`ssn_1l_bits`). Equal keys mean equal step-t vectors (`nred_attacks.adversary`
checks this word for word), so the prover writes n words for each distinct key and copies the rest for one pointer a
row a step. The saving is 32·(m·S − keys)·(n − 1) priced units against the call's stock m·n·K, over `--seeds` draws.

Run (CPU, a few minutes): python nred_data.py --model DIR --text wikitext-2-raw-v1-test.json --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_data as nd  # noqa: E402
import noise_sweep as ns  # noqa: E402

N_OUT = {name: n for name, (n, _) in ns.LLAMA.items()}


def prefix_ids(q: np.ndarray, s: int) -> np.ndarray:
    """cid[i, t]: equal iff rows' data prefixes through step t agree once all-zero blocks are dropped (with positions)."""
    m = q.shape[0]
    width = ns.DATA_PER_STEP + 31 * (s - 1)
    x = np.zeros((m, width), dtype=np.int8)
    x[:, :q.shape[1]] = q
    bounds = [(0, ns.DATA_PER_STEP)] + [(ns.DATA_PER_STEP + 31 * (t - 1), ns.DATA_PER_STEP + 31 * t) for t in range(1, s)]
    cid = np.zeros((m, s), dtype=np.int64)
    prev = np.zeros(m, dtype=np.int64)
    nxt = 1
    for t, (lo, hi) in enumerate(bounds):
        blk = np.ascontiguousarray(x[:, lo:hi])
        nz = blk.any(axis=1)
        cur = prev.copy()
        if nz.any():
            b = blk[nz].view(np.dtype((np.void, hi - lo))).ravel()
            _, bid = np.unique(b, return_inverse=True)
            pairs = np.stack([prev[nz], bid.ravel()], axis=1)
            _, inv = np.unique(pairs, axis=0, return_inverse=True)
            inv = inv.ravel()
            cur[nz] = nxt + inv
            nxt += int(inv.max()) + 1
        cid[:, t] = cur
        prev = cur
    return cid


def keys_for(cid: np.ndarray, h: np.ndarray, s: int) -> np.ndarray:
    a1, al = ns.ssn_1l_bits(h, s)
    cs = np.concatenate([np.zeros((h.shape[0], 1), dtype=np.int64), np.cumsum(al, axis=1)], axis=1)
    p = a1 @ (1 << np.arange(4))
    return (cid * 16 + p[:, None]) * 1024 + cs


def self_check(seed: int = 3) -> dict:
    """Equal keys give equal honest words (nred_attacks.honest) on rows mixing repeats, shared prefixes, zero tails
    and zero middle blocks; and the keys do merge (the check isn't vacuous)."""
    import nred_attacks as na
    rng = np.random.default_rng(seed)
    m, n, s = 256, 16, 6
    k = na.data_k(s, False)
    a = np.clip(np.rint(rng.normal(0, 30, (m, k))), -127, 127).astype(np.int64)
    a[64:128] = a[64]
    a[128:192, :k - 31] = a[128, :k - 31]
    a[192:224, 59:] = 0
    a[224:, 28:59] = 0
    a[224:240] = a[224]
    b = np.clip(np.rint(rng.normal(0, 25, (k, n))), -127, 127).astype(np.int64)
    h = rng.integers(0, 2 ** 32, (m, ns.ssn_1l_words(s)), dtype=np.int64)
    w = na.honest(a, b, h, s)
    key = keys_for(prefix_ids(a.astype(np.int8), s), h, s).ravel()
    vec = w.transpose(0, 2, 1).reshape(m * s, n)
    order = np.argsort(key, kind="stable")
    ks, vs = key[order], vec[order]
    same = ks[1:] == ks[:-1]
    ok = bool((vs[1:][same] == vs[:-1][same]).all())
    return {"pass": ok and int(same.sum()) > 0, "merged_pairs": int(same.sum()), "keys": int(len(np.unique(key))),
            "row_steps": m * s}


def call_saving(q: np.ndarray, n: int, k: int, seeds: int, rng: np.random.Generator) -> dict:
    m = q.shape[0]
    s = ns.steps1l(k)
    cid = prefix_ids(q, s)
    out = []
    for _ in range(seeds):
        h = rng.integers(0, 2 ** 32, (m, ns.ssn_1l_words(s)), dtype=np.int64)
        keys = len(np.unique(keys_for(cid, h, s)))
        out.append(ns.H32 * (m * s - keys) * (n - 1))
    return {"m": m, "S": s, "stock": m * n * k, "saved_units_mean": float(np.mean(out)),
            "saved_units_min": float(np.min(out)), "saved_units_max": float(np.max(out)),
            "identical_rows": int(m - len(np.unique(np.ascontiguousarray(q).view(np.dtype((np.void, q.shape[1]))))))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--small", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--self-check-only", action="store_true")
    args = ap.parse_args(argv)
    sc = self_check()
    print("self-check", json.dumps(sc))
    if not sc["pass"] or args.self_check_only:
        return 0 if sc["pass"] else 1
    import transformers

    torch.manual_seed(0)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    ids = tok("\n\n".join(json.load(open(args.text))), add_special_tokens=False)["input_ids"]
    layers = model.model.layers
    store: dict[tuple[int, str], list[np.ndarray]] = {(li, nm): [] for li in range(len(layers)) for nm in nd.LINEARS}

    def hook(li: int, name: str):
        def f(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            store[(li, name)].append(nd.quant_rows(x).to(torch.int8).numpy())
        return f

    handles = [getattr(getattr(layer, part), mods[0]).register_forward_pre_hook(hook(li, name))
               for li, layer in enumerate(layers) for name, (part, mods) in nd.LINEARS.items()]
    step = args.ctx - 1
    with torch.no_grad():
        for w in range(args.windows):
            model(torch.tensor([[tok.bos_token_id] + ids[w * step:(w + 1) * step]]))
    for hd in handles:
        hd.remove()
    rng = np.random.default_rng(17)
    per, tot = {}, {}
    for (li, name), chunks in store.items():
        n, k = N_OUT[name], chunks[0].shape[1]
        assert (n, k) == ns.LLAMA[name], (name, n, k)
        calls = {"m64": chunks[0][:args.small], "ctx": chunks[0], "all": np.concatenate(chunks, axis=0)}
        for scope, q in calls.items():
            r = call_saving(q, n, k, args.seeds, rng)
            per[f"{li}.{name}.{scope}"] = r
            t = tot.setdefault(scope, {"stock": 0.0, "saved_mean": 0.0, "saved_min": 0.0, "saved_max": 0.0,
                                       "identical_rows": 0, "rows": 0})
            t["stock"] += r["stock"]
            t["saved_mean"] += r["saved_units_mean"]
            t["saved_min"] += r["saved_units_min"]
            t["saved_max"] += r["saved_units_max"]
            t["identical_rows"] += r["identical_rows"]
            t["rows"] += r["m"]
        store[(li, name)] = []
    summary = {scope: {"gap_share_of_stock_mean": t["saved_mean"] / t["stock"],
                       "gap_share_of_stock_min": t["saved_min"] / t["stock"],
                       "gap_share_of_stock_max": t["saved_max"] / t["stock"],
                       "identical_row_fraction": t["identical_rows"] / t["rows"]} for scope, t in tot.items()}
    with open(args.out, "w") as fh:
        json.dump({"args": vars(args), "self_check": sc, "summary": summary, "per_call": per}, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
