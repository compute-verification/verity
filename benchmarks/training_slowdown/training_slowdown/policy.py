"""The policy plane: the (c, k) values the formalism scores every record at, and a start's reference shapes
(c in Boolean gates, k in bits).

A start's shape (d_model, n_layer, vocab, ctx, parameters, the tokens it was trained on, its forward MACs per token as
the harness counts them) gives the two reference shapes and the reference policy:

- **inference block** (one token): c = one token's forward at full context, w_fwd = forward MACs per token at the
  bf16 GEMM price (training_slowdown.gates); k = what that block reads beyond θ: its token id, ⌈log2 V⌉ bits, and its
  context's keys and values, (ctx − 1)·2·n_layer·d_model bf16 values;
- **reference** policy: c = 4096·w_fwd, one 4,096-token session per block, and k = 10 kB, 80,000 bits, where honest
  serving still passes Fit (Δ = 0 and s·τ = 4096·⌈log2 V⌉ ≤ k);
- **server block**: the start's whole training in one block, c = C0 (its training's gates: `c0_gates` when its shape
  has the count, else 3 forward MACs per token per training token at the bf16 GEMM price, labelled), k = its training
  state beyond θ, P·(16 + 2·32) bits (new bf16 weights and AdamW's two f32 moments).

The grid is every decade of c and of k from the least to the greatest of those shapes' (`grid`), keyed
`c1e<i>/k1e<j>` (`key`). A start is scored at the grid's cells and at the three shapes exactly, keyed by name
(`cells`).

The console's policy (`CONSOLE`: one full session of the reference model a block, attention included) is
`session_gates` at the 45M reference (`REFERENCE_PARAMS`) with 4,096 input and 1,024 output tokens (`CONSOLE_SESSION`),
and k the session's input ids, 4,096·16 = 65,536 bits. `session` is the one count every producer reads: a session's
in + out tokens each run the reference model's forward at the shape `SHAPES` states for its size (the lf family's dense
decoder: d_model, layers, heads, d_ff, vocab), and it counts every product of two operands:

- each layer's matrices, attention's four (4·d²) and the SwiGLU MLP's three (3·d·d_ff), at every token;
- the LM head over the tied embedding (d·V), at every token: the model's forward computes logits at each position;
- attention's score and value products (QKᵀ and AV, d each per position attended, every layer): a prompt token
  attends over the whole prompt, as training_slowdown.flops counts the causal SDPA (the math kernel's two products
  over the full score matrix, which every counted |C| uses), and the j-th output token over the prompt and the output
  tokens up to itself, its KV cache;

each MAC at the served precision's GEMM price (`SERVED_PRECISION`, bf16: training_slowdown.gates' 1,480.125 gates). The
elementwise ops (RoPE, the norms' gains, SwiGLU's product, attention's scalings) are left out: under 0.1% of a
token's counted MACs, and their count moves with training_slowdown.flops' version.

The console's dials move the reference model over `REFERENCE_SIZES` (20M to 1B at vocab 50,304, each a shape `SHAPES`
states) and k over `K_GRID`, a session's input ids at each of `DIAL_IN_TOKENS` input tokens (16,384 to 524,288 bits).

A config names its policy, and `named` is the one place a name becomes (c, k): `console` is the rule relative to the
start (`own_session`: one console session of the reference model of the start's own size, k the console's), and any
other name is a fixed policy of the slowdown calculator's registry (training_slowdown.calculator), which holds its
numbers. No other module holds a policy's (c, k).
"""
from __future__ import annotations

import math

from .gates import per_mac

ACT_BITS = 16
#: bits per parameter of the training state beyond θ: the new weights in bf16 and AdamW's two f32 moments
STATE_BITS = 16 + 2 * 32
#: the reference policy's session, in tokens, and its k
SESSION_TOKENS = 4096
REFERENCE_K_BITS = 8 * 10_000
MARKS = ("inference", "reference", "server")
#: the reference model's shape at each size the starts have and at each of the console's reference sizes
#: (`REFERENCE_SIZES`), by parameter count: the lf family's s, m, l16 and xl at the GPT-2 vocabulary padded to 50,304
#: or bpe32k's 32,000, its l, and three lf shapes in the family's form
SHAPES = {
    45_029_888: {"d_model": 512, "n_layer": 6, "n_head": 8, "d_ff": 1408, "vocab": 50_304},
    95_269_632: {"d_model": 768, "n_layer": 8, "n_head": 12, "d_ff": 2048, "vocab": 50_304},
    35_658_240: {"d_model": 512, "n_layer": 6, "n_head": 8, "d_ff": 1408, "vocab": 32_000},
    81_212_160: {"d_model": 768, "n_layer": 8, "n_head": 12, "d_ff": 2048, "vocab": 32_000},
    238_322_688: {"d_model": 1024, "n_layer": 16, "n_head": 16, "d_ff": 2816, "vocab": 32_000},
    656_146_944: {"d_model": 1536, "n_layer": 21, "n_head": 24, "d_ff": 4224, "vocab": 32_000},
    21_179_200: {"d_model": 320, "n_layer": 4, "n_head": 5, "d_ff": 896, "vocab": 50_304},
    205_677_568: {"d_model": 1024, "n_layer": 12, "n_head": 16, "d_ff": 2816, "vocab": 50_304},
    400_532_160: {"d_model": 1344, "n_layer": 15, "n_head": 21, "d_ff": 3712, "vocab": 50_304},
    999_397_440: {"d_model": 1728, "n_layer": 25, "n_head": 27, "d_ff": 4736, "vocab": 50_304},
}
#: the console's reference sizes, its dials' model sizes at vocab 50,304, each by the parameters of the shape `SHAPES`
#: states for it: the lf family's s, m and l where the family has a shape
#: within 5% of the size, else the lf shape nearest it in the family's form: heads of 64, the SwiGLU width the multiple
#: of 64 nearest 2.75 d (halves up), and d/n_layer from l16's 64 to m's 96
REFERENCE_SIZES = {"20M": 21_179_200, "45M": 45_029_888, "95M": 95_269_632, "200M": 205_677_568,
                   "400M": 400_532_160, "1B": 999_397_440}
#: the console's reference model: the lf family's s at vocab 50,304
REFERENCE_PARAMS = 45_029_888
#: the console's session, (input tokens, output tokens)
CONSOLE_SESSION = (4096, 1024)
SERVED_PRECISION = "bf16"


def id_bits(vocab: int) -> int:
    return math.ceil(math.log2(vocab))


def parameters(shape: dict) -> int:
    """An lf model's parameters: each layer's matrices and two norms, the tied embedding and the final norm."""
    d, n, f, v = shape["d_model"], shape["n_layer"], shape["d_ff"], shape["vocab"]
    return n * (4 * d * d + 3 * d * f + 2 * d) + (v + 1) * d


def session(params: int, in_tokens: int, out_tokens: int, shape: dict | None = None) -> dict:
    """One session of the reference model of `params` parameters, `in_tokens` in and `out_tokens` out (module
    docstring): its shape (`SHAPES`', unless `shape` gives one, whose parameters must be `params`), the tokens that run
    a forward, the positions their attention covers, the MACs of each product, the gates a MAC, the gates (c) and
    the session's input ids in bits (k)."""
    shape = SHAPES.get(params) if shape is None else shape
    if shape is None:
        raise ValueError(f"training_slowdown.policy: no stated shape for {params:,} parameters (SHAPES: "
                         f"{', '.join(f'{p:,}' for p in SHAPES)}); pass its shape")
    if parameters(shape) != params:
        raise ValueError(f"training_slowdown.policy: the shape {shape} has {parameters(shape):,} parameters, not {params:,}")
    if in_tokens < 1 or out_tokens < 0:
        raise ValueError(f"training_slowdown.policy: a session has at least one input token, not {in_tokens} in, {out_tokens} out")
    d, n, f, v = shape["d_model"], shape["n_layer"], shape["d_ff"], shape["vocab"]
    tokens = in_tokens + out_tokens
    attended = in_tokens * in_tokens + out_tokens * in_tokens + out_tokens * (out_tokens + 1) // 2
    macs = {"matrices": tokens * n * (4 * d * d + 3 * d * f), "lm_head": tokens * d * v, "attention": 2 * n * d * attended}
    per = per_mac(f"gemm/{SERVED_PRECISION}")
    return {"params": params, "in_tokens": in_tokens, "out_tokens": out_tokens, "shape": dict(shape),
            "precision": SERVED_PRECISION, "tokens": tokens, "attended": attended, "macs": macs,
            "gates_per_mac": per, "gates": sum(macs.values()) * per, "k_bits": in_tokens * id_bits(v)}


def session_gates(params: int, in_tokens: int, out_tokens: int, shape: dict | None = None) -> float:
    """c of one session of the reference model (`session`)."""
    return session(params, in_tokens, out_tokens, shape)["gates"]


#: the console's policy, (c gates, k bits): one session of the 45M reference, 4,096 tokens in and 1,024 out
CONSOLE = (session_gates(REFERENCE_PARAMS, *CONSOLE_SESSION),
           float(CONSOLE_SESSION[0] * id_bits(SHAPES[REFERENCE_PARAMS]["vocab"])))
#: the console's k dial: a session of each of these input tokens, its ids at the reference
#: vocabulary's 16 bits, so k from 16,384 to 524,288 bits, the console's 65,536 among them (`K_GRID`)
DIAL_IN_TOKENS = (1024, 2048, 4096, 8192, 16_384, 32_768)
K_GRID = tuple(n * id_bits(SHAPES[REFERENCE_PARAMS]["vocab"]) for n in DIAL_IN_TOKENS)


def own_session(shape: dict) -> tuple[float, float]:
    """The console's rule at a start, (c gates, k bits): c one full session (`CONSOLE_SESSION`) of the reference model
    of the start's own size, at the shape `SHAPES` states for it, else the start's own; k the console's, the
    session's input ids at 16 bits each."""
    p = shape["parameters"]
    return session_gates(p, *CONSOLE_SESSION, None if p in SHAPES or "d_ff" not in shape else shape), CONSOLE[1]


#: what a named policy is (`named`): a rule this module states relative to the start, or a fixed policy of the
#: slowdown calculator's registry
RULE, FIXED = "rule relative to the start", "fixed policy in the calculator's registry"


def named(name: str, shape: dict) -> dict:
    """The policy `name` at a start of `shape`, the one place a name becomes (c, k): its `kind` (`RULE` or `FIXED`),
    `c_gates`, `k_bits` and `why`. `console` is this module's rule relative to the start (`own_session`). Any other
    name is a fixed policy of the slowdown calculator's registry (training_slowdown.calculator), its Γ in FLOP as
    Γ/2 MACs at the served precision's GEMM price and its K in bits; a name that is neither is refused."""
    if name == "console":
        c, k = own_session(shape)
        tokens = " in, ".join(f"{n:,}" for n in CONSOLE_SESSION)
        return {"name": name, "kind": RULE, "c_gates": c, "k_bits": k,
                "why": f"c one session ({tokens} out) of the reference model of the start's {shape['parameters']:,} "
                       f"parameters; k the session's input ids, {id_bits(SHAPES[REFERENCE_PARAMS]['vocab'])} bits each"}
    from . import calculator

    fixed = calculator.policy(name)
    if fixed is None:
        raise ValueError(f"training_slowdown.policy: no policy named {name!r}: a config's policy is `console`, the "
                         f"{RULE}, or a {FIXED} ({', '.join(calculator.policies())})")
    return {"name": name, "kind": FIXED, "c_gates": fixed["gamma_flop"] / 2 * per_mac(f"gemm/{SERVED_PRECISION}"),
            "k_bits": fixed["k_bits"], "gamma_flop": fixed["gamma_flop"],
            "why": f"the registry's Γ, {fixed['gamma_flop']:.4g} FLOP, as Γ/2 MACs at the {SERVED_PRECISION} GEMM "
                   f"price; its K in bits"}


def c0(shape: dict) -> tuple[float, str]:
    """The start's training compute in gates, and where it comes from."""
    if shape.get("c0_gates"):
        return float(shape["c0_gates"]), "counted"
    if shape.get("c0_macs"):
        return shape["c0_macs"] * per_mac("gemm/bf16"), "the start's counted MACs, every MAC at the bf16 GEMM price"
    macs = 3 * shape["forward_macs_per_token"] * shape["tokens"]
    return macs * per_mac("gemm/bf16"), "stub: 3 x forward MACs per token x training tokens, bf16 GEMM"


def w_fwd(shape: dict) -> float:
    """One token's forward in gates: forward MACs per token at the bf16 GEMM price."""
    return shape["forward_macs_per_token"] * per_mac("gemm/bf16")


def shapes(shape: dict) -> dict[str, dict]:
    """The start's reference shapes (module docstring), each {c, k, why}."""
    w = w_fwd(shape)
    ids, ctx = id_bits(shape["vocab"]), shape["ctx"]
    kv = (ctx - 1) * 2 * shape["n_layer"] * shape["d_model"] * ACT_BITS
    c_s, how = c0(shape)
    return {
        "inference": {"c": w, "k": ids + kv, "why": "one token's forward at full context: its id and its context's K, V"},
        "reference": {"c": SESSION_TOKENS * w, "k": REFERENCE_K_BITS,
                      "why": "the reference policy: one 4,096-token session per block, k = 10 kB"},
        "server": {"c": c_s, "k": shape["parameters"] * STATE_BITS,
                   "why": f"the start's training in one block (C0 {how}) and its state beyond theta"},
    }


def decades(lo: float, hi: float) -> list[float]:
    return [10.0 ** i for i in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)]


def grid(shape: dict) -> list[tuple[float, float]]:
    """Every decade of c and k spanning the start's reference shapes."""
    ref = shapes(shape).values()
    cs = decades(min(r["c"] for r in ref), max(r["c"] for r in ref))
    ks = decades(min(r["k"] for r in ref), max(r["k"] for r in ref))
    return [(c, k) for c in cs for k in ks]


def cells(shape: dict) -> dict[str, tuple[float, float]]:
    """Every (c, k) a start is scored at: the grid's, keyed by decade (`key`), and its reference shapes', exactly, keyed
    by name (`MARKS`)."""
    return {key(c, k): (c, k) for c, k in grid(shape)} | {m: (s["c"], s["k"]) for m, s in shapes(shape).items()}


def key(c: float, k: float) -> str:
    return f"c1e{round(math.log10(c))}/k1e{round(math.log10(k))}"
