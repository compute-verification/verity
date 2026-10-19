"""kappa for frontier pretraining: twice the counted multiplications over the reference training circuit's computed
gates, per token at a context length and over a model's pretraining schedule.

``Nci.SecurityProofs.ThreadedTraining`` (``Security/Models/Nci/Guarantees/Threads.lean``) takes kappa through its
hypothesis ``kappa * c.D.size c.D.computed <= 2 * T.M.card``: every computed gate of the reference circuit ``c``
against twice its counted multiplications ``M``, the gates that multiply an updated weight by an activation or an
activation gradient (``Nci.Threading``). A gate is one scalar operation and an addition reads two values, so a matmul's
gates are its flops, 2 per multiply-add, and a nonlinear function, comparison or select is one gate. The circuit is the
minimal one: causal attention computes only the pairs it uses, and nothing is recomputed. Per token, with ``N`` the
active weights outside the embedding and the LM head:

* counted, ``2 |M|``: the forward ``2N``, the backward to the activations ``2N``, and the LM head's backward. The LM
  head's forward products stay out of ``M``: one final hidden value multiplies every vocabulary row, and ``Threaded``
  at an ``e`` below the vocabulary needs them out;
* the rest of the denominator: the weight gradients (an activation gradient times an activation), the LM head's
  forward, attention's score and value products and their backward (activations times activations), softmax, norms,
  SwiGLU, routing and the combine, residual adds, RoPE, the loss, the embedding (a lookup and a scatter-add), and the
  optimizer step spread over the batch's tokens.

Over a schedule kappa is a ratio of sums, the token-weighted harmonic mean of its phases' kappa.

    python benchmarks/nci/kappa.py       # the breakdown per model and phase, and each schedule's kappa
"""

from __future__ import annotations

import argparse

#: gates per query-key pair per head besides its products: forward max, subtract, exp, sum and normalize; backward
#: P (dP - D) and the scale
SOFTMAX = 9
#: gates per entry, forward and backward: an RMSNorm, a SwiGLU hidden unit, a rotated RoPE entry, a vocabulary entry
#: of the loss, an active expert's combine per hidden entry
NORM, SWIGLU, ROPE, LOSS, COMBINE = 12, 15, 6, 8, 5
#: routing gates per expert per token: score, bias, selection
ROUTE = 5
#: optimizer gates per weight: AdamW's moments, correction and update; Muon's momentum, scale, decay and update
ADAMW, MUON = 12, 10
#: Newton-Schulz iterations per Muon step, each 4 m^2 n + 2 m^3 flops on an m x n matrix (m <= n)
NS_STEPS = 5

#: the shapes (config.json) and pretraining schedules (tokens, sequence length) of the reports
MODELS = {
    "kimi-k2": {
        "source": "moonshotai/Kimi-K2-Base config.json; Kimi K2 report (arXiv 2507.20534) 2.5: 15.5T tokens at 4,096, "
                  "then an annealing phase of 400B tokens at 4k and 60B at 32k, at 67M tokens per batch; MuonClip",
        "d": 7168, "layers": 61, "dense_layers": 1, "ffn": 18432, "vocab": 163840,
        "attention": {"kind": "mla", "heads": 64, "q_rank": 1536, "kv_rank": 512, "nope": 128, "rope": 64, "v": 128},
        "experts": 384, "active": 8, "shared": 1, "expert_ffn": 2048, "mtp": 0,
        "optimizer": "muon", "batch_tokens": 67e6,
        "schedule": [(15.5e12, 4096), (400e9, 4096), (60e9, 32768)],
    },
    "deepseek-v3": {
        "source": "deepseek-ai/DeepSeek-V3-Base config.json; DeepSeek-V3 report (arXiv 2412.19437) 4.2-4.3: 14.8T tokens "
                  "at 4K, then 1000 steps of 1920 x 32K and 1000 of 480 x 128K, at 15360 x 4K per batch; AdamW; one MTP "
                  "module",
        "d": 7168, "layers": 61, "dense_layers": 3, "ffn": 18432, "vocab": 129280,
        "attention": {"kind": "mla", "heads": 128, "q_rank": 1536, "kv_rank": 512, "nope": 128, "rope": 64, "v": 128},
        "experts": 256, "active": 8, "shared": 1, "expert_ffn": 2048, "mtp": 1,
        "optimizer": "adamw", "batch_tokens": 15360 * 4096,
        "schedule": [(14.8e12, 4096), (1000 * 1920 * 32768, 32768), (1000 * 480 * 131072, 131072)],
    },
    "llama-3-405b": {
        "source": "Llama 3.1 405B config.json; Llama 3 report (arXiv 2407.21783) 3.4: 15.6T tokens at 8K after 252M at "
                  "4K, of them about 800B in six stages from 8K to 128K, taken here as equal tokens at 8K x 16^(k/6); "
                  "16M tokens per batch; AdamW",
        "d": 16384, "layers": 126, "dense_layers": 126, "ffn": 53248, "vocab": 128256,
        "attention": {"kind": "gqa", "heads": 128, "kv_heads": 8, "head_dim": 128},
        "experts": 0, "active": 0, "shared": 0, "expert_ffn": 0, "mtp": 0,
        "optimizer": "adamw", "batch_tokens": 16e6,
        "schedule": [(252e6, 4096), (15.6e12 - 800e9 - 252e6, 8192)]
                    + [(800e9 / 6, round(8192 * 16 ** (k / 6))) for k in range(1, 7)],
    },
    "qwen3-235b-a22b": {
        "source": "Qwen/Qwen3-235B-A22B config.json; Qwen3 report (arXiv 2505.09388) 3.2: over 30T tokens at 4,096, "
                  "about 5T at 4,096, then hundreds of billions at 32,768, taken here as 0.3T; batch not stated "
                  "(AdamW's step is negligible at any frontier batch)",
        "d": 4096, "layers": 94, "dense_layers": 0, "ffn": 0, "vocab": 151936,
        "attention": {"kind": "gqa", "heads": 64, "kv_heads": 4, "head_dim": 128, "qk_norm": True},
        "experts": 128, "active": 8, "shared": 0, "expert_ffn": 1536, "mtp": 0,
        "optimizer": "adamw", "batch_tokens": 16e6,
        "schedule": [(30.7e12, 4096), (5e12, 4096), (0.3e12, 32768)],
    },
}

#: the products of the Kimi K2 rows' chain (``rows.json``): per layer q_a, q_b, o and the shared expert's up and down,
#: over 61 layers
KIMI_CHAIN_PRODUCTS = 7.2e9


def attention_dims(m: dict) -> tuple[int, int, int]:
    """Heads and the per-head width of a query-key product and of a value."""
    a = m["attention"]
    if a["kind"] == "mla":
        return a["heads"], a["nope"] + a["rope"], a["v"]
    return a["heads"], a["head_dim"], a["head_dim"]


def matrices(m: dict) -> list[tuple[int, int, int, int]]:
    """The weight matrices outside the embedding and the LM head: ``(rows, cols, copies, active copies per token)``,
    over every layer and the multi-token-prediction modules (each a projection and one MoE layer)."""
    d, a = m["d"], m["attention"]
    if a["kind"] == "mla":
        h = a["heads"]
        att = [(a["q_rank"], d), (h * (a["nope"] + a["rope"]), a["q_rank"]), (a["kv_rank"] + a["rope"], d),
               (h * (a["nope"] + a["v"]), a["kv_rank"]), (d, h * a["v"])]
    else:
        hd = a["head_dim"]
        att = [(a["heads"] * hd, d), (a["kv_heads"] * hd, d), (a["kv_heads"] * hd, d), (d, a["heads"] * hd)]
    dense = [(m["ffn"], d), (m["ffn"], d), (d, m["ffn"])]
    expert = [(m["expert_ffn"], d), (m["expert_ffn"], d), (d, m["expert_ffn"])]
    moe_layers = m["layers"] - m["dense_layers"] + m["mtp"]
    out = [(r, c, m["layers"] + m["mtp"], m["layers"] + m["mtp"]) for r, c in att]
    out += [(r, c, m["dense_layers"], m["dense_layers"]) for r, c in dense]
    if moe_layers:
        out.append((m["experts"], d, moe_layers, moe_layers))
        out += [(r, c, (m["experts"] + m["shared"]) * moe_layers, (m["active"] + m["shared"]) * moe_layers)
                for r, c in expert]
    out += [(d, 2 * d, m["mtp"], m["mtp"])]
    return out


def active_weights(m: dict) -> float:
    """``N``: the active weights per token outside the embedding and the LM head."""
    return sum(r * c * act for r, c, _, act in matrices(m))


def elementwise(m: dict, seq: int) -> float:
    """Gates per token outside the matmuls: softmax, norms, SwiGLU, routing and the combine, residual adds, RoPE, the
    loss and the embedding, forward and backward."""
    d, a = m["d"], m["attention"]
    h, _, _ = attention_dims(m)
    att_layers = m["layers"] + m["mtp"]
    moe_layers = m["layers"] - m["dense_layers"] + m["mtp"]
    if a["kind"] == "mla":
        latents = a["q_rank"] + a["kv_rank"]
    else:
        latents = (a["heads"] + a["kv_heads"]) * a["head_dim"] if a.get("qk_norm") else 0
    rotated = h * a["rope"] + a["rope"] if a["kind"] == "mla" else (a["heads"] + a["kv_heads"]) * a["head_dim"]
    units = m["dense_layers"] * m["ffn"] + moe_layers * (m["active"] + m["shared"]) * m["expert_ffn"]
    heads = 1 + m["mtp"]
    return (SOFTMAX * h * (seq + 1) / 2 * att_layers
            + NORM * (att_layers * (2 * d + latents) + heads * d + m["mtp"] * 2 * d)
            + SWIGLU * units
            + moe_layers * (ROUTE * m["experts"] + COMBINE * m["active"] * d)
            + 4 * d * att_layers + ROPE * rotated * att_layers
            + LOSS * m["vocab"] * heads + 2 * d * heads)


def optimizer_step(m: dict) -> float:
    """Gates per step: Muon on every weight matrix (AdamW on the embedding, the LM head and the norms), or AdamW on
    everything."""
    vector = 2 * m["vocab"] * m["d"] + 2 * m["d"] * (m["layers"] + m["mtp"] + 1)
    if m["optimizer"] == "adamw":
        return ADAMW * (sum(r * c * n for r, c, n, _ in matrices(m)) + vector)
    ns = sum(n * (NS_STEPS * (4 * min(r, c) ** 2 * max(r, c) + 2 * min(r, c) ** 3) + MUON * r * c)
             for r, c, n, _ in matrices(m))
    return ns + ADAMW * vector


def per_token(m: dict, seq: int) -> dict[str, float]:
    """The reference circuit's gates per token at sequence length ``seq``, by part; the first two parts are ``2 |M|``."""
    N, head, heads = active_weights(m), m["vocab"] * m["d"], 1 + m["mtp"]
    h, dqk, dv = attention_dims(m)
    return {
        "weights forward (counted)": 2 * N,
        "weights backward to activations (counted)": 2 * N + 2 * head * heads,
        "LM head forward": 2 * head * heads,
        "weight gradients": 2 * (N + head * heads),
        "attention scores and values": 6 * h * (dqk + dv) * (seq + 1) / 2 * (m["layers"] + m["mtp"]),
        "softmax, norms, SwiGLU, routing, loss": elementwise(m, seq),
        "optimizer step": optimizer_step(m) / m["batch_tokens"],
    }


COUNTED = ("weights forward (counted)", "weights backward to activations (counted)")


def counted(m: dict) -> float:
    """``2 |M|`` per token, the same at every context length."""
    p = per_token(m, 1)
    return sum(p[k] for k in COUNTED)


def kappa(m: dict, seq: int) -> float:
    return counted(m) / sum(per_token(m, seq).values())


def schedule(m: dict) -> tuple[float, float]:
    """kappa over the pretraining schedule and the mean gates per token: ratios of sums over the phases' tokens."""
    tokens = sum(t for t, _ in m["schedule"])
    gates = sum(t * sum(per_token(m, s).values()) for t, s in m["schedule"]) / tokens
    return counted(m) / gates, gates


def matmul_flops(m: dict, seq: int) -> float:
    """Model flops per token as the reports count them: the weight matmuls and attention's products."""
    p = per_token(m, seq)
    return sum(v for k, v in p.items() if k not in ("softmax, norms, SwiGLU, routing, loss", "optimizer step"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), action="append", help="default: every model")
    args = ap.parse_args(argv)
    for name in args.model or MODELS:
        m = MODELS[name]
        seqs = sorted({s for _, s in m["schedule"]})
        head = m["vocab"] * m["d"]
        print(f"{name}: {m['source']}")
        print(f"  N = {active_weights(m):.4g} active weights outside the embedding and LM head ({head:.3g} each); "
              f"2|M| = {counted(m) / 1e9:.1f} GFLOP per token")
        print(f"  {'GFLOP per token at sequence length':44}" + "".join(f"{s:>10,}" for s in seqs))
        rows = {s: per_token(m, s) for s in seqs}
        for part in rows[seqs[0]]:
            print(f"  {part:44}" + "".join(f"{rows[s][part] / 1e9:10.4g}" for s in seqs))
        print(f"  {'total':44}" + "".join(f"{sum(rows[s].values()) / 1e9:10.4g}" for s in seqs))
        print(f"  {'kappa':44}" + "".join(f"{kappa(m, s):10.4f}" for s in seqs))
        k, gates = schedule(m)
        tokens = sum(t for t, _ in m["schedule"])
        print(f"  over {tokens / 1e12:.2f}T tokens: kappa {k:.4f}, {gates / 1e9:.1f} GFLOP per token\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
