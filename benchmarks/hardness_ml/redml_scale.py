"""HT-REDML: what design A hardens at real scale, from public model configs (arithmetic, no measurement).

Per model and context length: the share of forward MACs in the large linears (hardenable) against attention, the LM head
and small per-head matmuls left uncredited; HBM held by int8 weights against an int8 KV cache on one 8×H200 node; decode
bytes by weights against KV; live-coin bandwidth for one fresh coin bit per element of every hardened linear's distinct
input (linears that read the same input share its coins); and the adapter rank at which design A's adapters alone cost 1%
of the R-products, r·(1/n + 1/m) = 1%, which also bounds what they can cancel of R (T1).

Prices: commit cost 3,520 W1 units per committed 32-bit output (HASH.md: about 110 register writes a word at 32 units),
so h_commit = 3,520/K for an output of a K-long dot product. H200: 141 GB HBM, 4.8 TB/s, int8 dense 9.89e14 MAC/s (spec)
and 3.96e14 MAC/s (the rate D-OUT priced with).
"""

import argparse
import json

H200_HBM, H200_BW = 141e9, 4.8e12
MAC_RATES = {"spec": 9.89e14, "dout": 3.96e14}
COMMIT_UNITS = 3520.0


def dense_model(name, d, L, H, kv, hd, ffn, vocab):
    q, k = H * hd, kv * hd
    lin = [("qkv", "x1", d, q + 2 * k), ("o", "attn", q, d), ("gate_up", "x2", d, 2 * ffn), ("down", "h", ffn, d)]
    layer = {"linears": lin, "small": [], "attn_per_key": 2 * H * hd, "attn_per_key_prefill": 2 * H * hd,
             "kv_bytes": 2 * k}
    return {"name": name, "layers": [layer] * L, "head": (d, vocab)}


def deepseek_v3():
    """MLA with weight absorption at decode (scores over the 512 + 64 latent, values over the 512 latent, per head) and
    the expanded form at prefill (192-dim scores, 128-dim values); 8 routed + 1 shared expert active."""
    d, H, nope, rope, vh, qr, kvr = 7168, 128, 128, 64, 128, 1536, 512
    dense_ffn, exp_ffn, n_act = 18432, 2048, 9
    layers = []
    for li in range(61):
        lin = [("q_a", "x1", d, qr), ("kv_a", "x1", d, kvr + rope), ("q_b", "q_lat", qr, H * (nope + rope)),
               ("o", "attn", H * vh, d)]
        if li < 3:
            lin += [("gate_up", "x2", d, 2 * dense_ffn), ("down", "h", dense_ffn, d)]
        else:
            for e in range(n_act):
                lin += [("gate_up", "x2", d, 2 * exp_ffn), ("down", f"h{e}", exp_ffn, d)]
        small = [("q_absorb", nope, kvr)] * H + [("v_up", kvr, vh)] * H + [("router", d, 256)]
        layers.append({"linears": lin, "small": small, "attn_per_key": H * ((kvr + rope) + kvr),
                       "attn_per_key_prefill": H * ((nope + rope) + vh), "kv_bytes": kvr + rope})
    return {"name": "DeepSeek-V3", "layers": layers, "head": (d, 129280), "weights_bytes": 671e9}


MODELS = [
    dense_model("Llama-3.1-8B", 4096, 32, 32, 8, 128, 14336, 128256),
    dense_model("Llama-3.1-70B", 8192, 80, 64, 8, 128, 28672, 128256),
    dense_model("Llama-3.1-405B", 16384, 126, 128, 8, 128, 53248, 128256),
    deepseek_v3(),
]


def per_token(model, pos, phase="decode"):
    lin = small = attn = coins = 0
    commit_units = 0.0
    caps = {}
    for layer in model["layers"]:
        groups = {}
        for nm, grp, n, m in layer["linears"]:
            lin += n * m
            groups[grp] = n
            commit_units += m * COMMIT_UNITS
            caps.setdefault(nm, 0.01 / (1 / n + 1 / m))
        coins += sum(groups.values())
        small += sum(n * m for _, n, m in layer["small"])
        attn += (layer["attn_per_key_prefill"] if phase == "prefill" else layer["attn_per_key"]) * pos
    hn, hm = model["head"]
    return {"linear": lin, "small": small, "attn": attn, "head": hn * hm, "coin_bits": coins,
            "h_commit": commit_units / lin, "rank_cap": caps}


def weights_bytes(model):
    if "weights_bytes" in model:
        return model["weights_bytes"]
    hn, hm = model["head"]
    return sum(n * m for layer in model["layers"] for _, _, n, m in layer["linears"]) + 2 * hn * hm


def report(contexts, gpus, batch):
    out = []
    for model in MODELS:
        w = weights_bytes(model)
        kvb = sum(layer["kv_bytes"] for layer in model["layers"])
        base = per_token(model, 0)
        row = {"model": model["name"], "weights_GB_int8": w / 1e9, "kv_bytes_per_token_int8": kvb,
               "linear_MACs_per_token": base["linear"], "head_MACs_per_token": base["head"],
               "small_MACs_per_token": base["small"], "coin_KB_per_token_1bit": base["coin_bits"] / 8 / 1e3,
               "coins_per_MAC": base["coin_bits"] / base["linear"], "h_commit": base["h_commit"],
               "rank_cap_at_1pct": {k: round(v, 1) for k, v in base["rank_cap"].items()}, "contexts": []}
        for rate_name, rate in MAC_RATES.items():
            row[f"coin_Gbps_per_GPU_at_{rate_name}"] = rate * base["coin_bits"] / base["linear"] / 1e9
        hbm, bw = gpus * H200_HBM, gpus * H200_BW
        for L in contexts:
            for phase, pos in (("prefill", L / 2), ("decode", L)):
                t = per_token(model, pos, phase)
                tot = t["linear"] + t["small"] + t["attn"] + t["head"]
                c = {"ctx": L, "phase": phase, "linear_share": t["linear"] / tot, "attn_share": t["attn"] / tot,
                     "head_share": t["head"] / tot, "small_share": t["small"] / tot}
                if phase == "decode":
                    seqs_fit = max(0.0, (hbm - w) / (L * kvb))
                    b = min(batch, seqs_fit)
                    kv_total = b * L * kvb
                    c.update({"seqs_fit_on_node": seqs_fit, "batch": b, "hbm_share_kv_at_batch": kv_total / hbm,
                              "hbm_share_weights": w / hbm,
                              "decode_bytes_share_kv": kv_total / (kv_total + w) if b else 0.0,
                              "decode_step_ms_bw_bound": (kv_total + w) / bw * 1e3})
                row["contexts"].append(c)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="*", default=[2048, 8192, 32768, 131072])
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = report(a.contexts, a.gpus, a.batch)
    for row in res:
        print(f"\n{row['model']}: weights {row['weights_GB_int8']:.0f} GB int8, KV {row['kv_bytes_per_token_int8'] / 1e3:.1f}"
              f" KB/token int8, linears {row['linear_MACs_per_token'] / 1e9:.2f} G MAC/token, head "
              f"{row['head_MACs_per_token'] / 1e9:.2f} G, small {row['small_MACs_per_token'] / 1e9:.2f} G, "
              f"h_commit {row['h_commit']:.3f}")
        print(f"  coins at 1 bit/element: {row['coin_KB_per_token_1bit']:.0f} KB/token (1 per "
              f"{1 / row['coins_per_MAC']:,.0f} MACs); per GPU {row['coin_Gbps_per_GPU_at_spec']:.0f} Gb/s at spec rate, "
              f"{row['coin_Gbps_per_GPU_at_dout']:.0f} Gb/s at D-OUT's rate")
        print(f"  rank cap r with r(1/n+1/m) = 1%: {row['rank_cap_at_1pct']}")
        for c in row["contexts"]:
            s = (f"  ctx {c['ctx']:>6} {c['phase']:<7} linear {c['linear_share']:.1%} attn {c['attn_share']:.1%} "
                 f"head {c['head_share']:.1%} small {c['small_share']:.1%}")
            if c["phase"] == "decode":
                s += (f" | node fits {c['seqs_fit_on_node']:.0f} seqs; at batch {c['batch']:.0f}: KV "
                      f"{c['hbm_share_kv_at_batch']:.0%} of HBM, weights {c['hbm_share_weights']:.0%}; KV "
                      f"{c['decode_bytes_share_kv']:.0%} of decode bytes, {c['decode_step_ms_bw_bound']:.1f} ms/step")
            print(s)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
