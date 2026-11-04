"""The policy plane: the (c, k) values the formalism scores every record at, and a start's reference shapes
(c in Boolean gates, k in bits).

    python -m te_core.policy [START ...]    # each start's reference shapes and grid

A start's shape (`starts_store.json`'s `shape`: d_model, n_layer, vocab, ctx, parameters, the tokens it was trained
on, its forward MACs per token as the harness counts them) gives the two reference shapes and the reference policy:

- **inference block** (one token): c = one token's forward at full context, w_fwd = forward MACs per token at the
  bf16 GEMM price (te_core.gates); k = what that block reads beyond θ: its token id, ⌈log2 V⌉ bits, and its context's
  keys and values, (ctx − 1)·2·n_layer·d_model bf16 values;
- **reference** policy (the experiments' E0):
  c = 4096·w_fwd, one 4,096-token session per block, and k = 10 kB, 80,000 bits, where honest serving still passes Fit
  (Δ = 0 and s·τ = 4096·⌈log2 V⌉ ≤ k);
- **server block**: the start's whole training in one block, c = C0 (its training's gates: `c0_gates` when its shape
  has the count, else 3 forward MACs per token per training token at the bf16 GEMM price, labelled), k = its training
  state beyond θ, P·(16 + 2·32) bits (new bf16 weights and AdamW's two f32 moments).

The grid is every decade of c and of k from the least to the greatest of those shapes' (`grid`), keyed
`c1e<i>/k1e<j>` (`key`). A start is scored at the grid's cells and at the three shapes exactly, keyed by name
(`cells`). P1, P2 and P4 read the reference policy (`REFERENCE`); P3 marks all three.

The console's policy (`CONSOLE`, the dashboard's ruling: one full session of the reference model a block, attention
included) is `session_gates` at the 45M reference (`REFERENCE_PARAMS`) with 4,096 input and 1,024 output tokens
(`CONSOLE_SESSION`), and k the session's input ids, 4,096·16 = 65,536 bits. `session` is the one count every producer
reads: a session's in + out tokens each run the reference model's forward at the shape `SHAPES` states for its size
(the lf family, model.py's dense decoder: d_model, layers, heads, d_ff, vocab), and it counts every product of two
operands:

- each layer's matrices, attention's four (4·d²) and the SwiGLU MLP's three (3·d·d_ff), at every token;
- the LM head over the tied embedding (d·V), at every token: the model's forward computes logits at each position;
- attention's score and value products (QKᵀ and AV, d each per position attended, every layer): a prompt token
  attends over the whole prompt, as te_core.flops counts the causal SDPA (torch's formula, the full score matrix,
  which every counted |C| uses), and the j-th output token over the prompt and the output tokens up to itself, its
  KV cache;

each MAC at the served precision's GEMM price (`SERVED_PRECISION`, bf16: te_core.gates' 1,480.125 gates). The
elementwise ops (RoPE, the norms' gains, SwiGLU's product) are left out: under 0.1% of a token's counted MACs, and
their count moves with te_core.flops' version (the starts' `forward_macs_per_token` exceed the products by 27,392
at s50 and 40,192 at s-t20).

The console's dials move the reference model over `REFERENCE_SIZES` (20M to 1B at vocab 50,304, each a shape `SHAPES`
states) and k over `K_GRID`, a session's input ids at each of `DIAL_IN_TOKENS` input tokens (16,384 to 524,288 bits).

The scoring cap (`eligible`; coordinator 08:09Z): a checkpoint is scored only when its own forward gates per token are
at most the greater of twice its start's (`CAP_FALLBACK`) and the compute-optimal model's at the loss under the ladder's
fit (`ladder_fit`), and twice its start's where the fit gives no value there. Nothing caps what the harness
evaluates; scoring reads the count the evaluation job records.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .gates import per_mac

ACT_BITS = 16
#: bits per parameter of the training state beyond θ: the new weights in bf16 and AdamW's two f32 moments
STATE_BITS = 16 + 2 * 32
#: the reference policy's session, in tokens, and its k
SESSION_TOKENS = 4096
REFERENCE_K_BITS = 8 * 10_000
MARKS = ("inference", "reference", "server")
REFERENCE = "reference"
MANIFEST = Path(__file__).resolve().parents[1] / "starts_store.json"
#: the scoring cap where the ladder's fit gives no value: this many times the start's forward gates per token
CAP_FALLBACK = 2.0
#: data v3's tokenizer (covert_witness's data_v3.TOKENIZER), which a start on v3's data has in its shape
V3_TOKENIZER = "bpe32k"
#: the reference model's shape at each size the starts have and at each of the console's reference sizes
#: (`REFERENCE_SIZES`), by parameter count: the lf family's s, m, l16 and xl (model.py, islands/ladder-v3's `SIZES`) at
#: the GPT-2 vocabulary padded to 50,304 or bpe32k's 32,000, model.py's l, and three lf shapes in the family's form
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
#: the console's reference sizes, its dials' model sizes at vocab 50,304 (the dashboard's 19:01Z), each by the
#: parameters of the shape `SHAPES` states for it: the lf family's s, m and l (model.py) where the family has a shape
#: within 5% of the size, else the lf shape nearest it in the family's form: heads of 64, the SwiGLU width the multiple
#: of 64 nearest 2.75 d (halves up), and d/n_layer from l16's 64 to m's 96
REFERENCE_SIZES = {"20M": 21_179_200, "45M": 45_029_888, "95M": 95_269_632, "200M": 205_677_568,
                   "400M": 400_532_160, "1B": 999_397_440}
#: the console's reference model: the 45M dense starts (s50 and s100)
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
        raise ValueError(f"te_core.policy: no stated shape for {params:,} parameters (SHAPES: "
                         f"{', '.join(f'{p:,}' for p in SHAPES)}); pass its shape")
    if parameters(shape) != params:
        raise ValueError(f"te_core.policy: the shape {shape} has {parameters(shape):,} parameters, not {params:,}")
    if in_tokens < 1 or out_tokens < 0:
        raise ValueError(f"te_core.policy: a session has at least one input token, not {in_tokens} in, {out_tokens} out")
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
#: the console's k dial (the dashboard's 19:01Z): a session of each of these input tokens, its ids at the reference
#: vocabulary's 16 bits, so k from 16,384 to 524,288 bits, the console's 65,536 among them (`K_GRID`)
DIAL_IN_TOKENS = (1024, 2048, 4096, 8192, 16_384, 32_768)
K_GRID = tuple(n * id_bits(SHAPES[REFERENCE_PARAMS]["vocab"]) for n in DIAL_IN_TOKENS)


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


def eligible(w: float | None, start_w: float, loss: float | None, ladder=None) -> bool:
    """The scoring cap: whether a checkpoint whose model's forward is `w` gates per token is scored, `w` at most the
    greater of `CAP_FALLBACK` times the start's `start_w` and the compute-optimal model's at `loss` under the ladder's
    fit (`ladder(loss)`, `ladder_fit`'s), and `CAP_FALLBACK` times the start's where there is no fit or it has no point
    at that loss (coordinator 08:09Z: for every attempt, honest and covert). A checkpoint whose forward isn't counted
    (`w` None) has the start's."""
    at = ladder(loss) if ladder and loss is not None else None
    return w is None or w <= max(CAP_FALLBACK * start_w, at or 0.0)


def ladder_fit(path: Path):
    """The ladder's fit as the cap reads it: the compute-optimal model's forward gates per token at a held-out loss,
    from `path`'s points (`{"points": [{"loss": L, "forward_gates_per_token": W}, ...]}`, at least two), log-linear in
    loss between neighbours and None outside their span."""
    pts = sorted((p["loss"], math.log(p["forward_gates_per_token"])) for p in json.loads(path.read_text())["points"])
    if len(pts) < 2:
        raise SystemExit(f"te_core.policy: {path} has {len(pts)} ladder points, not at least two")

    def fit(loss: float) -> float | None:
        for (l1, g1), (l2, g2) in zip(pts, pts[1:]):
            if l1 <= loss <= l2:
                return math.exp(g1 + (g2 - g1) * (loss - l1) / (l2 - l1)) if l2 > l1 else math.exp(max(g1, g2))
        return None
    return fit


def shapes(shape: dict) -> dict[str, dict]:
    """The start's reference shapes (module docstring), each {c, k, why}."""
    w = w_fwd(shape)
    ids, ctx = id_bits(shape["vocab"]), shape["ctx"]
    kv = (ctx - 1) * 2 * shape["n_layer"] * shape["d_model"] * ACT_BITS
    c_s, how = c0(shape)
    return {
        "inference": {"c": w, "k": ids + kv, "why": "one token's forward at full context: its id and its context's K, V"},
        "reference": {"c": SESSION_TOKENS * w, "k": REFERENCE_K_BITS,
                      "why": "the draft's reference policy: one 4,096-token session per block, k = 10 kB"},
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


def of_key(name: str) -> tuple[float, float]:
    c, k = name.split("/")
    return 10.0 ** int(c[3:]), 10.0 ** int(k[3:])


def starts(manifest: Path = MANIFEST) -> dict[str, dict]:
    """Each start's shape, from the starts manifest."""
    return {n: s["shape"] for n, s in json.loads(manifest.read_text())["starts"].items() if s.get("shape")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    known = starts()
    for name in a.names or sorted(known):
        g = grid(known[name])
        cs, ks = sorted({c for c, _ in g}), sorted({k for _, k in g})
        print(f"{name}: c 1e{math.log10(cs[0]):.0f}..1e{math.log10(cs[-1]):.0f} gates, k 1e{math.log10(ks[0]):.0f}.."
              f"1e{math.log10(ks[-1]):.0f} bits ({len(g)} cells)")
        for m, s in shapes(known[name]).items():
            print(f"  {m:9} c {s['c']:.3g} gates, k {s['k']:.3g} bits: {s['why']}" + (" (reference)" if m == REFERENCE else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
