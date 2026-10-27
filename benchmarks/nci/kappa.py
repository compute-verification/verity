"""kappa for frontier training: twice the counted multiplications over the reference training circuit's computed gates,
per token at a context length, over a model's pretraining schedule, and for the board's other rows (``ROWS``).

``Nci.SecurityProofs.ThreadedTraining`` (``Security/Definitions/Nci/Guarantees/Threads.lean``) takes kappa through its
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

A frozen weight is committed in theta, where ``WeightsIncompressible`` fails, so its products are in the denominator
only, and it takes no weight gradient or optimizer step. A circuit may run several forwards per token (a zeroth-order
step's two) and no backward (evolution strategies). Over a schedule kappa is a ratio of sums, the token-weighted harmonic
mean of its phases' kappa.

    python benchmarks/nci/kappa.py       # the breakdown per model and phase, each schedule's kappa, and ROWS'
"""

from __future__ import annotations

import argparse

#: gates per query-key pair per head besides its products, (forward, backward): max, subtract, exp, sum and normalize;
#: P (dP - D) and the scale
SOFTMAX = (5, 4)
#: gates per entry, (forward, backward): an RMSNorm, a SwiGLU hidden unit, a tanh-GELU hidden unit, a rotated RoPE
#: entry, a vocabulary entry of the loss (sampling a token reads the same softmax), an active expert's combine per
#: hidden entry, a layer's two residual adds, the embedding
NORM, SWIGLU, GELU, ROPE, LOSS, COMBINE = (4, 8), (6, 9), (13, 11), (3, 3), (4, 4), (2, 3)
RESIDUAL, EMBED = (2, 2), (1, 1)
#: routing gates per expert per token, (forward, backward): score, bias, selection; the selected scores' gradient
ROUTE = (4, 1)
#: optimizer gates per weight: AdamW's moments, correction and update; Muon's momentum, scale, decay and update
ADAMW, MUON = 12, 10
#: Newton-Schulz iterations per Muon step, each 4 m^2 n + 2 m^3 flops on an m x n matrix (m <= n)
NS_STEPS = 5
#: gates per weight to regenerate one seeded perturbation and add it (the rows' decode cost)
SEED = 10

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
    "dense-100b": {
        "source": "the thread draft's dense example (models.json dense-100b): d_model 12288 with a 4x GELU MLP and 96 "
                  "heads of 128 (12 d^2 a layer), 55 layers for 1e11 weights, Llama 3's vocabulary; assumed to train on "
                  "Llama 3 405B's schedule (arXiv 2407.21783 3.4.1-3.4.2), the one dense frontier report: 252M tokens "
                  "at 4K, then 8K, and about 800B in six stages from 8K to 128K, at 16M tokens per batch; AdamW",
        "d": 12288, "layers": 55, "dense_layers": 55, "ffn": 4 * 12288, "mlp": "gelu", "vocab": 128256,
        "attention": {"kind": "gqa", "heads": 96, "kv_heads": 96, "head_dim": 128},
        "experts": 0, "active": 0, "shared": 0, "expert_ffn": 0, "mtp": 0,
        "optimizer": "adamw", "batch_tokens": 16e6,
        "schedule": [(252e6, 4096), (15.6e12 - 800e9 - 252e6, 8192)]
                    + [(800e9 / 6, round(8192 * 16 ** (k / 6))) for k in range(1, 7)],
    },
}

#: the products of the Kimi K2 rows' chain (``rows.json``): per layer q_a, q_b, o and the shared expert's up and down,
#: over 61 layers
KIMI_CHAIN_PRODUCTS = 7.2e9
#: the dense-100b row's chain: q, o, up and down, 10 d^2 of a layer's 12 d^2, over its layers
DENSE_CHAIN_PRODUCTS = 10 * 12288 ** 2 * 55

#: RL's rollouts: R1-Zero's 512 sampled sequences a step at the 32,768-token cap it kept for its first 8.2k steps
#: (DeepSeek-R1, arXiv 2501.12948, 2.2)
RL_SEQUENCES, ROLLOUT = 512, 32768
#: R1-Zero's forwards per sampled token: the rollout's, the reference model's for the KL term (the latest policy every
#: 400 steps, so not committed in theta), and a fresh one for 15 of each rollout's 16 minibatches, trained after the
#: weights moved
R1_FORWARDS = 2 + 15 / 16

#: ESFT (arXiv 2407.01906) on Kimi K2: 10% of the routed experts updated, carrying ESFT-Token's p = 0.2 of a token's
#: routed slots; attention, the router, the shared expert, the embedding and the LM head frozen; AdamW
#: (configs/base.yaml) at 32 sequences of 4096 tokens a step
ESFT = {"updated": {"experts": 0.1, "slots": 0.2}, "optimizer": "adamw", "batch_tokens": 32 * 4096}
#: ES at Scale (arXiv 2509.24372) on Kimi K2: per iteration each of 30 members perturbs every weight and restores it,
#: then the update adds the 30 perturbations back, 30 x 2 + 30 regenerations; each member rolls out its 200 training
#: samples
ES_MEMBERS, ES_SAMPLES = 30, 200
ES = {"optimizer": "seeds", "seed_gates": SEED * 3 * ES_MEMBERS, "batch_tokens": ES_MEMBERS * ES_SAMPLES * ROLLOUT}

#: the board's rows besides those priced over a pretraining schedule (Kimi K2's and dense-100b's, ``rows.json``): the
#: model, the sequence length, and each token's forwards and backwards. RL is the on-policy step, whose rollout forward
#: is the training forward: the fewest gates and the lowest kappa of the policy-gradient recipes (``R1_FORWARDS``)
ROWS = {
    "moe-expert-subset-sft": (MODELS["kimi-k2"] | ESFT, 4096, 1, 1),
    "rl-full-weights": (MODELS["kimi-k2"] | {"batch_tokens": RL_SEQUENCES * ROLLOUT}, ROLLOUT, 1, 1),
    "es-population-30": (MODELS["kimi-k2"] | ES, ROLLOUT, 1, 0),
}


def attention_dims(m: dict) -> tuple[int, int, int]:
    """Heads and the per-head width of a query-key product and of a value."""
    a = m["attention"]
    if a["kind"] == "mla":
        return a["heads"], a["nope"] + a["rope"], a["v"]
    return a["heads"], a["head_dim"], a["head_dim"]


def expert_shapes(m: dict) -> list[tuple[int, int]]:
    return [(m["expert_ffn"], m["d"]), (m["expert_ffn"], m["d"]), (m["d"], m["expert_ffn"])]


def moe_layers(m: dict) -> int:
    return m["layers"] - m["dense_layers"] + m["mtp"]


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
    dense = [(m["ffn"], d), (d, m["ffn"])] if m.get("mlp") == "gelu" else [(m["ffn"], d), (m["ffn"], d), (d, m["ffn"])]
    moe = moe_layers(m)
    out = [(r, c, m["layers"] + m["mtp"], m["layers"] + m["mtp"]) for r, c in att]
    out += [(r, c, m["dense_layers"], m["dense_layers"]) for r, c in dense]
    if moe:
        out.append((m["experts"], d, moe, moe))
        out += [(r, c, (m["experts"] + m["shared"]) * moe, (m["active"] + m["shared"]) * moe) for r, c in expert_shapes(m)]
    out += [(d, 2 * d, m["mtp"], m["mtp"])]
    return out


def active_weights(m: dict) -> float:
    """``N``: the active weights per token outside the embedding and the LM head."""
    return sum(r * c * act for r, c, _, act in matrices(m))


def updated_weights(m: dict) -> float:
    """The updated weights among ``N``: all of them, or under ``updated`` the routed experts' share of a token's slots."""
    if "updated" not in m:
        return active_weights(m)
    return m["updated"]["slots"] * m["active"] * moe_layers(m) * sum(r * c for r, c in expert_shapes(m))


def elementwise(m: dict, seq: int, forwards: float = 1, backward: float = 1) -> float:
    """Gates per token outside the matmuls: softmax, norms, SwiGLU or GELU, routing and the combine, residual adds, RoPE,
    the loss and the embedding, in ``forwards`` forward passes and ``backward`` backward ones."""
    d, a = m["d"], m["attention"]
    h, _, _ = attention_dims(m)
    att_layers = m["layers"] + m["mtp"]
    moe = moe_layers(m)
    if a["kind"] == "mla":
        latents = a["q_rank"] + a["kv_rank"]
    else:
        latents = (a["heads"] + a["kv_heads"]) * a["head_dim"] if a.get("qk_norm") else 0
    rotated = h * a["rope"] + a["rope"] if a["kind"] == "mla" else (a["heads"] + a["kv_heads"]) * a["head_dim"]
    dense_units = m["dense_layers"] * m["ffn"]
    expert_units = moe * (m["active"] + m["shared"]) * m["expert_ffn"]
    heads = 1 + m["mtp"]

    def g(cost: tuple[int, int]) -> float:
        return forwards * cost[0] + backward * cost[1]

    return (g(SOFTMAX) * h * (seq + 1) / 2 * att_layers
            + g(NORM) * (att_layers * (2 * d + latents) + heads * d + m["mtp"] * 2 * d)
            + g(GELU if m.get("mlp") == "gelu" else SWIGLU) * dense_units + g(SWIGLU) * expert_units
            + moe * (g(ROUTE) * m["experts"] + g(COMBINE) * m["active"] * d)
            + g(RESIDUAL) * d * att_layers + g(ROPE) * rotated * att_layers
            + g(LOSS) * m["vocab"] * heads + g(EMBED) * d * heads)


def updated_matrices(m: dict) -> tuple[list[tuple[int, int, float]], float]:
    """The updated weight matrices ``(rows, cols, copies)`` and the other updated weights (the embedding, the LM head and
    the norms): every weight, or under ``updated`` its share of the routed experts' matrices alone."""
    if "updated" in m:
        return [(r, c, m["updated"]["experts"] * m["experts"] * moe_layers(m)) for r, c in expert_shapes(m)], 0.0
    vector = 2 * m["vocab"] * m["d"] + 2 * m["d"] * (m["layers"] + m["mtp"] + 1)
    return [(r, c, n) for r, c, n, _ in matrices(m)], vector


def optimizer_step(m: dict) -> float:
    """Gates per step: Muon on every updated matrix (AdamW on the embedding, the LM head and the norms), AdamW on
    everything updated, or ``seeds``, ``seed_gates`` per weight to perturb and update from seeds."""
    mats, vector = updated_matrices(m)
    weights = sum(r * c * n for r, c, n in mats) + vector
    if m["optimizer"] == "adamw":
        return ADAMW * weights
    if m["optimizer"] == "seeds":
        return m["seed_gates"] * weights
    ns = sum(n * (NS_STEPS * (4 * min(r, c) ** 2 * max(r, c) + 2 * min(r, c) ** 3) + MUON * r * c) for r, c, n in mats)
    return ns + ADAMW * vector


def per_token(m: dict, seq: int, forwards: float = 1, backward: float = 1) -> dict[str, float]:
    """The reference circuit's gates per token at sequence length ``seq``, by part, in ``forwards`` forward passes and
    ``backward`` backward ones; the parts marked counted are ``2 |M|``. Parts a circuit lacks are left out."""
    N, upd, head, heads = active_weights(m), updated_weights(m), m["vocab"] * m["d"], 1 + m["mtp"]
    head_updated = "updated" not in m
    h, dqk, dv = attention_dims(m)
    parts = {
        "weights forward (counted)": 2 * upd * forwards,
        "weights backward to activations (counted)": backward * (2 * upd + 2 * head * heads * head_updated),
        "frozen weights forward": 2 * (N - upd) * forwards,
        "frozen weights backward to activations": backward * (2 * (N - upd) + 2 * head * heads * (not head_updated)),
        "LM head forward": 2 * head * heads * forwards,
        "weight gradients": backward * 2 * (upd + head * heads * head_updated),
        "attention scores and values": (2 * forwards + 4 * backward) * h * (dqk + dv) * (seq + 1) / 2
                                       * (m["layers"] + m["mtp"]),
        "softmax, norms, SwiGLU, routing, loss": elementwise(m, seq, forwards, backward),
        "optimizer step": optimizer_step(m) / m["batch_tokens"],
    }
    return {k: v for k, v in parts.items() if v}


def counted(m: dict, forwards: float = 1, backward: float = 1) -> float:
    """``2 |M|`` per token, the same at every context length."""
    return sum(v for k, v in per_token(m, 1, forwards, backward).items() if k.endswith("(counted)"))


def kappa(m: dict, seq: int, forwards: float = 1, backward: float = 1) -> float:
    return counted(m, forwards, backward) / sum(per_token(m, seq, forwards, backward).values())


def schedule(m: dict) -> tuple[float, float]:
    """kappa over the pretraining schedule and the mean gates per token: ratios of sums over the phases' tokens."""
    tokens = sum(t for t, _ in m["schedule"])
    gates = sum(t * sum(per_token(m, s).values()) for t, s in m["schedule"]) / tokens
    return counted(m) / gates, gates


def row(rid: str) -> tuple[float, float]:
    """A ``ROWS`` row's kappa and gates per token."""
    m, seq, forwards, backward = ROWS[rid]
    return kappa(m, seq, forwards, backward), sum(per_token(m, seq, forwards, backward).values())


def matmul_flops(m: dict, seq: int) -> float:
    """Model flops per token as the reports count them: the weight matmuls and attention's products."""
    p = per_token(m, seq)
    return sum(v for k, v in p.items() if k not in ("softmax, norms, SwiGLU, routing, loss", "optimizer step"))


def table(m: dict, seqs: list[int], forwards: float = 1, backward: float = 1) -> None:
    print(f"  {'GFLOP per token at sequence length':44}" + "".join(f"{s:>10,}" for s in seqs))
    rows = {s: per_token(m, s, forwards, backward) for s in seqs}
    for part in rows[seqs[0]]:
        print(f"  {part:44}" + "".join(f"{rows[s][part] / 1e9:10.4g}" for s in seqs))
    print(f"  {'total':44}" + "".join(f"{sum(rows[s].values()) / 1e9:10.4g}" for s in seqs))
    print(f"  {'kappa':44}" + "".join(f"{kappa(m, s, forwards, backward):10.4f}" for s in seqs))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), action="append", help="default: every model")
    args = ap.parse_args(argv)
    for name in args.model or MODELS:
        m = MODELS[name]
        head = m["vocab"] * m["d"]
        print(f"{name}: {m['source']}")
        print(f"  N = {active_weights(m):.4g} active weights outside the embedding and LM head ({head:.3g} each); "
              f"2|M| = {counted(m) / 1e9:.1f} GFLOP per token")
        table(m, sorted({s for _, s in m["schedule"]}))
        k, gates = schedule(m)
        tokens = sum(t for t, _ in m["schedule"])
        print(f"  over {tokens / 1e12:.2f}T tokens: kappa {k:.4f}, {gates / 1e9:.1f} GFLOP per token\n")
    if args.model:
        return 0
    for rid, (m, seq, forwards, backward) in ROWS.items():
        print(f"{rid}: {forwards:g} forward(s) and {backward:g} backward(s) per token at {seq:,}")
        table(m, sorted({4096, 8192, 32768, 65536, seq}), forwards, backward)
        k, gates = row(rid)
        print(f"  the row: kappa {k:.4f}, {gates / 1e9:.1f} GFLOP per token\n")
    m, _, _, _ = ROWS["rl-full-weights"]
    print(f"rl-full-weights under R1-Zero's recipe ({R1_FORWARDS:g} forwards a sampled token)")
    table(m, [8192, 32768, 65536], R1_FORWARDS, 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
