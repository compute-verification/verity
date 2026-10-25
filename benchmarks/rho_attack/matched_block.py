"""A falsification test of Proposition 3's activation premise on one matched computation, the cost of a training step
built from such blocks, and the closure check for Kimi K2 and DeepSeek-V4-Pro.

    python3 matched_block.py --extract B1_BUNDLE_DIR A2_RUNS_DIR --out inputs/a2_full_blocks.json
    python3 matched_block.py --inputs inputs/a2_full_blocks.json --out OUT

The premise: every block's counted activations a_{V_I} have LDL_{t,R}(a_{V_I}) >= beta_x |V_I| - lambda_x. A block that
rebuilds its weights from l_w bits, reads l_in other bits and generates n_I counted activations within the time bound
gives beta_x <= (l_w + l_in + a_dec + lambda_x) / n_I. Every l_w and l_in here is a code length B1's compressors achieved
(the minimum over them, basis included), so it bounds LDL from above, which is what a violation needs.

The matched computation is one GRPO step of A2's full-parameter arm on Qwen2.5-1.5B-Instruct (32 prompts x 8 rollouts,
the step's kept rollouts trained on prompt + completion), with theta' = the step's checkpoint and theta = the base model.
Inputs per entry: B1's per-boundary F (forward, given theta's boundary output) and G (backward), B1's per-map weight
update code lengths on its sampled layers (0, 9, 18, 27) and the embedding table's, A2's logged step (kept rollouts,
completion tokens) and the measured prompt length. A layer B1 did not sample is priced at the largest sampled layer's
code (an estimate, flagged).

A block is either
- boundary entry: theta's forward to boundary b (token ids, 17 bits each), theta''s boundary output from F_b bits per
  token, then held layers b+1 .. b+m; or
- embedding entry: theta''s embedding table from its code, then held layers 0 .. m-1 (token ids only).
It holds whole rollouts, so every attention context is inside it. Counted activations per token per layer are the maps'
inputs: attention input d, o input (heads x head_dim), MLP input d and down input (ffn).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import configs  # noqa: E402
import rho_attack as ra  # noqa: E402

H_TOK = 17.0
ADV_BITS = 16.0  # one GRPO advantage per rollout
SEED_BITS = 64.0  # one sampling seed per generated rollout (rho_attack's seed_bits)
GRAD_BITS = 16.0  # a raw bf16 partial weight-gradient entry
PRECISIONS = {"fp8": "fp8t", "nvfp4": "nvfp4"}
#: lzma of B1's counted.py (the coders) and transformers' modeling_qwen2.py (the forward): 21,484 bytes; the Python and
#: PyTorch runtime is the reference machine R
A_DEC = 171_872.0
K = 2e7
C = 1.34e17


# ------------------------------------------------------------------------------------------------------------ inputs

def extract(bundle: Path, a2_runs: Path, prompt_mean: float) -> dict:
    """The A2 full-arm entries from B1's per-entry counted JSONs and A2's step logs."""
    entries = {}
    for f in sorted(glob.glob(str(bundle / "r2026*" / "counted_A2_*-full-*.json"))):
        r = json.load(open(f))
        name, meta = r["name"], r["meta"]
        if name in entries:
            continue
        log = next(iter(glob.glob(str(a2_runs / "*" / f"{meta['task']}-full-seed{meta['seed']}.jsonl"))), None)
        if log is None:
            continue
        step = next(e for e in map(json.loads, open(log)) if e.get("event") == "step" and e["step"] == meta["step"])
        w = r["weights"]
        per = {}
        for p, fmt in PRECISIONS.items():
            s = r["summary"][fmt]
            per[p] = {"F": s["F_fwd_per_layer"], "G": s["F_bwd_per_layer"], "beta_w": s["beta_w"],
                      "embed_bits": w["embed"][fmt]["min"],
                      "layer_bits": {str(L): {"attn": sum(w[f"L{L}.self_attn.{m}_proj"][fmt]["min"] for m in "qkvo"),
                                              "mlp": sum(w[f"L{L}.mlp.{m}_proj"][fmt]["min"] for m in ("gate", "up", "down"))}
                                     for L in r["layers_sampled"]}}
        entries[name] = {"task": meta["task"], "seed": meta["seed"], "step": meta["step"], "eval_tokens": r["tokens"],
                         "embed_params": w["embed"]["n"], "rollouts": meta["prompts_per_step"] * meta["group"],
                         "prompts": meta["prompts_per_step"], "n_seq": step["n_seq"], "n_tok": step["n_tok"],
                         "clen": step["clen"], "prompt_mean": prompt_mean if meta["task"] == "gsm8k" else None, **per}
    return {"source": "B1's per-entry counted JSONs (art:608df1e131dd8edd1c6eb01ec3527210e631faa1370d835e789b47322520893b, "
                      "run r20261024-221611-4e17) and A2's step logs "
                      "(art:d8feafe2a5fcc095b4814c5f00509da7505515ea0245573169347671abc79c20)",
            "prompt_mean_note": "GSM8K train prompts under A2's chat template, measured; MATH entries have none and "
                                "are skipped", "entries": entries}


# ------------------------------------------------------------------------------------------------------------ the step

class Step:
    """One A2 full-arm entry at one precision: per-layer prices and the step's tokens."""

    SMALL = 5_120 * 16.0  # a layer's two RMSNorm weights and q/k/v biases, raw bf16
    FINAL_NORM = 1_536 * 16.0

    def __init__(self, entry: dict, precision: str, cfg: configs.Config):
        p = entry[precision]
        self.cfg, self.entry, self.precision = cfg, entry, precision
        self.L = cfg.layers
        self.F, self.G, self.beta_w = p["F"], p["G"], p["beta_w"]
        sampled = {int(k): v["attn"] + v["mlp"] for k, v in p["layer_bits"].items()}
        top = max(sampled.values())
        self.w = [sampled.get(j, top) + self.SMALL for j in range(self.L)]
        self.measured = set(sampled)
        self.emb = p["embed_bits"]
        self.beta_e = self.emb / entry["embed_params"]
        self.n_seq, self.rollouts, self.prompts = entry["n_seq"], entry["rollouts"], entry["prompts"]
        self.T = entry["n_tok"] + entry["n_seq"] * entry["prompt_mean"]
        self.len = self.T / self.n_seq
        self.gen_tokens = entry["prompts"] * entry["prompt_mean"] + entry["rollouts"] * entry["clen"]
        self.acts = sum(m.d_in for m in cfg.maps if not m.resid_in) + 2 * cfg.d  # counted activations per token-layer
        self.prods = sum(m.d_in * m.d_out for m in cfg.maps)  # products per token-layer
        self.e = max(sum(m.d_out for m in cfg.maps if m.resid_in and m.mlp),
                     sum(m.d_out for m in cfg.maps if m.resid_in and not m.mlp))

    def flops(self, ctx: float) -> float:
        """Forward FLOPs per token at ``ctx`` attended positions."""
        return 2 * self.prods * self.L + self.cfg.f_head + 4 * self.cfg.attn_width * ctx * self.L

    @property
    def T0(self) -> float:
        """Honest step FLOPs: rollout generation (prefill once per prompt, then every rollout's decode) and the training
        forward and backward of the kept rollouts."""
        pm = self.entry["prompt_mean"]
        gen = self.prompts * pm * self.flops(pm / 2) + self.rollouts * self.entry["clen"] * self.flops(pm + self.entry["clen"] / 2)
        return gen + 3 * self.T * self.flops(self.len / 2)

    @property
    def M(self) -> float:
        """Products of the layers' maps in the step's training forward (the head left out, as N is)."""
        return self.prods * self.L * self.T

    def beta_rep(self, reading: str) -> float:
        F = min(self.F) if reading == "Ff_min" else statistics.fmean(self.F)
        return (H_TOK + F) / self.cfg.d

    # -------------------------------------------------------------------------------------------- task 1: one block

    def block(self, start: int, m: int, k: float = K, c: float = C) -> dict | None:
        """The block holding layers start .. start+m-1, entering at the embedding (start 0) or at boundary start-1."""
        held = range(start, start + m)
        l_w = sum(self.w[j] for j in held) + (self.emb if start == 0 else 0.0)
        per_tok = H_TOK + (self.F[start - 1] if start > 0 else 0.0)
        room = k - l_w - A_DEC
        n = min(self.n_seq, math.floor(room / (self.len * per_tok))) if room > 0 else 0
        if n < 1:
            return None
        Q = n * self.len
        l_in = Q * per_tok
        n_I = Q * m * self.acts
        work = Q * 2 * self.prods * (start + m) + Q * 4 * self.cfg.attn_width * (self.len / 2) * (start + m)
        if work > c:
            return None
        bits = l_w + l_in + A_DEC
        products = Q * m * self.prods
        readings = ("Ff_min", "mean")
        return {"entry": "embedding" if start == 0 else f"boundary {start - 1}", "layers": f"{start}-{start + m - 1}",
                "m": m, "rollouts": n, "Q": Q, "l_w": l_w, "l_in": l_in, "a_dec": A_DEC, "n_I": n_I,
                "beta_x_bound": bits / n_I, "decode_flops": work,
                "weights_measured": all(j in self.measured for j in held),
                "products": products,
                "products_per_entering_coordinate": m * self.prods / self.cfg.d if start > 0 else None,
                "products_per_input_bit": products / bits,
                "violation": {r: self.beta_rep(r) * n_I / bits for r in readings},
                "R": {r: self.beta_rep(r) * products / (self.e * bits) for r in readings},
                "lambda_x_to_rescue": {r: self.beta_rep(r) * n_I - bits for r in readings}}

    def best_block(self, measured_only: bool = False, entry: str | None = None) -> dict | None:
        starts = {None: range(self.L), "embedding": range(1), "boundary": range(1, self.L)}[entry]
        found = [b for s in starts for m in range(1, self.L - s + 1)
                 if (b := self.block(s, m)) is not None and (b["weights_measured"] or not measured_only)]
        return min(found, key=lambda b: b["beta_x_bound"]) if found else None


    # -------------------------------------------------------------------------------------------- task 2: the step

    def _tiles(self, state: float, acc: float, fwd_roll: float, bwd_roll: float, train: bool, acc_mode: str,
               params: float, k: float) -> float:
        """Blocks for one pass of a unit holding ``state`` over the pass's rollouts: whole rollouts per block, or, when
        one doesn't fit, a rollout cut into s token chunks, each chunk reading the forward input of its whole prefix (to
        rebuild keys and values) and the backward input of its own tokens. A training pass split over several blocks
        carries the running gradient (``acc`` bits, chain) or emits raw partial gradients that reducing blocks read
        (reduce)."""
        rollouts = self.n_seq if train else self.rollouts

        def blocks(s: float) -> float:
            room = k - s - A_DEC
            if room <= 0:
                return math.inf
            n = min(rollouts, math.floor(room / (fwd_roll + bwd_roll)))
            if n >= 1:
                return math.ceil(rollouts / n)
            if not train or room <= fwd_roll:
                return math.inf
            return rollouts * math.ceil(bwd_roll / (room - fwd_roll))

        t = blocks(state)
        if not train or t <= 1 or not math.isfinite(t):
            return t
        if acc_mode == "chain":
            return blocks(state + acc)
        return t + math.ceil(t * params * GRAD_BITS / k)

    def _segment(self, a: int, b: int, train: bool, acc_mode: str, k: float, c: float) -> float:
        """Blocks for layers a .. b-1 over the step's tokens: entry at boundary a-1 (F) or, for a = 0, the embedding
        (the whole table's code, or one row per token at the table's mean code per entry, as rho_attack prices a
        first-stage token), exit gradient G at boundary b-1 when training."""
        layers = range(a, b)
        l_w = sum(self.w[j] for j in layers)
        params = len(layers) * self.cfg.layer_params
        length = self.len if train else self.gen_tokens / self.rollouts
        bwd = length * self.G[b - 1] if train else 0.0
        options = [(l_w, H_TOK + self.F[a - 1])] if a > 0 else \
            [(l_w + self.emb, H_TOK), (l_w, H_TOK + self.cfg.d * self.beta_e)]
        return min(self._tiles(s, self.beta_w * params, length * t, bwd, train, acc_mode, params, k) for s, t in options)

    def whole_blocks(self, train: bool, acc_mode: str = "chain", k: float = K) -> float:
        """Every layer and the tied table in one block: token ids in, the advantage (training) or seed (generation) per
        rollout."""
        V = self.entry["embed_params"]
        length = self.len if train else self.gen_tokens / self.rollouts
        state = sum(self.w) + self.emb + self.FINAL_NORM
        acc = self.beta_w * self.L * self.cfg.layer_params + self.beta_e * V
        return self._tiles(state, acc, length * H_TOK + (ADV_BITS if train else SEED_BITS), 0.0, train, acc_mode,
                           self.L * self.cfg.layer_params + V, k)

    def head_blocks(self, train: bool, acc_mode: str = "chain", k: float = K) -> tuple[float, int]:
        """The tied embedding table as the LM head, split into r vocabulary-row groups, each over every token: theta''s
        final hidden state (F at the last boundary) and token id per token, a logsumexp per token when r > 1, the
        advantage (training) or sampling seed (generation) per rollout. Training also takes the gradient at the
        embedding output for the group's tokens (G at boundary 0 standing in for it) and accumulates the group's rows."""
        V = self.entry["embed_params"]
        length = self.len if train else self.gen_tokens / self.rollouts
        best = (math.inf, 0)
        for r in range(1, 129):
            per_tok = H_TOK + self.F[-1] + (ra.LSE_BITS if r > 1 else 0.0) + (self.G[0] / r if train else 0.0)
            bits_roll = length * per_tok + (ADV_BITS if train else SEED_BITS)
            t = self._tiles((self.emb + self.FINAL_NORM) / r, self.beta_e * V / r, bits_roll, 0.0, train, acc_mode,
                            V / r, k)
            if r * t < best[0]:
                best = (r * t, r)
        return best

    def forward_loss(self, k: float = K) -> dict:
        """The corrected count of one forward over the layers, per token and with tokens unbounded (contiguous layer
        segments, each entering at its own boundary's measured F, or at the embedding), against the fan-out count
        L N beta_x^rep / (e k) per token: the step-level loss factor of the fan-out allowance at each reading."""
        best = [0.0] + [math.inf] * self.L
        cut = [0] * (self.L + 1)
        for b in range(1, self.L + 1):
            for a in range(b):
                l_w = sum(self.w[a:b])
                opts = [(l_w, H_TOK + self.F[a - 1])] if a > 0 else \
                    [(l_w + self.emb, H_TOK), (l_w, H_TOK + self.cfg.d * self.beta_e)]
                v = best[a] + min((t / (k - s - A_DEC) if s + A_DEC < k else math.inf) for s, t in opts)
                if v < best[b]:
                    best[b], cut[b] = v, a
        segs, b = [], self.L
        while b > 0:
            segs.append((cut[b], b))
            b = cut[b]
        fan = {r: self.L * self.prods * self.beta_rep(r) / (self.e * k) for r in READINGS}
        return {"blocks_per_token": best[-1], "segments": segs[::-1], "fanout_blocks_per_token": fan,
                "loss": {r: fan[r] / best[-1] for r in READINGS}}

    def step_blocks(self, train: bool, acc_mode: str = "chain", k: float = K, c: float = C) -> tuple[float, list]:
        """The fewest blocks for one pass over the step's layers, segments contiguous."""
        U = self.L
        best = [0.0] + [math.inf] * U
        cut = [0] * (U + 1)
        for b in range(1, U + 1):
            for a in range(b):
                v = best[a] + self._segment(a, b, train, acc_mode, k, c)
                if v < best[b]:
                    best[b], cut[b] = v, a
        segs, b = [], U
        while b > 0:
            segs.append((cut[b], b))
            b = cut[b]
        return best[U], segs[::-1]

    def step(self, acc_mode: str, k: float = K, c: float = C) -> dict:
        tb, ts = self.step_blocks(True, acc_mode, k, c)
        gb, gs = self.step_blocks(False, acc_mode, k, c)
        th, tr = self.head_blocks(True, acc_mode, k)
        gh, gr = self.head_blocks(False, acc_mode, k)
        wt, wg = self.whole_blocks(True, acc_mode, k), self.whole_blocks(False, acc_mode, k)
        if wt <= tb + th:
            tb, ts, th, tr = wt, [(0, self.L)], 0.0, "in the block"
        if wg <= gb + gh:
            gb, gs, gh, gr = wg, [(0, self.L)], 0.0, "in the block"
        blocks = tb + gb + th + gh
        kappa = self.T0 / self.M
        kappa_all = self.T0 / (self.M + self.prods * self.L * self.gen_tokens)
        return {"acc_mode": acc_mode, "train_blocks": tb, "train_segments": ts, "train_head_blocks": th,
                "train_head_groups": tr, "gen_blocks": gb, "gen_segments": gs, "gen_head_blocks": gh,
                "gen_head_groups": gr, "blocks": blocks, "T0": self.T0, "M": self.M, "kappa": kappa,
                "kappa_with_generation_products": kappa_all, "rho": c * blocks / self.T0, "rho_one_block": c / self.T0,
                "fanout_prediction": {r: c * self.beta_rep(r) / (kappa * self.e * k) for r in ("Ff_min", "mean")},
                "fanout_prediction_with_generation_products":
                    {r: c * self.beta_rep(r) / (kappa_all * self.e * k) for r in ("Ff_min", "mean")}}


def median_block(cfg: configs.Config, beta_w: float, F: float, m: int, beta_rep: dict[str, float],
                 k: float = K) -> dict:
    """The coordinator's first instantiation: m layers at the model-wide median beta_w, entering at a boundary of the
    median F, tokens filling k (no a_dec, no whole-rollout rounding, no cap at the step's tokens)."""
    prods = sum(mp.d_in * mp.d_out for mp in cfg.maps)
    acts = sum(mp.d_in for mp in cfg.maps if not mp.resid_in) + 2 * cfg.d
    e = max(sum(mp.d_out for mp in cfg.maps if mp.resid_in and mp.mlp),
            sum(mp.d_out for mp in cfg.maps if mp.resid_in and not mp.mlp))
    l_w = m * beta_w * cfg.layer_params
    Q = (k - l_w) / (H_TOK + F)
    n_I = Q * m * acts
    return {"entry": f"boundary at F = {F:,.0f}", "layers": f"{m} layers", "m": m, "Q": Q, "l_w": l_w,
            "l_in": Q * (H_TOK + F), "a_dec": 0.0, "n_I": n_I, "beta_x_bound": k / n_I,
            "products": Q * m * prods, "products_per_entering_coordinate": m * prods / cfg.d,
            "violation": {r: beta_rep[r] * n_I / k for r in READINGS},
            "R": {r: beta_rep[r] * Q * m * prods / (e * k) for r in READINGS}}


# ------------------------------------------------------------------------------------------------------------ task 3

READINGS = ("Ff_min", "mean")


def corrected_R(p: float, l_w: float, beta_in: float, beta_rep: float, e: float, k: float = K) -> float:
    """The corrected per-block allowance A = (k - l_w - a_dec) p / beta_in (entering coordinates at beta_in bits each,
    p products per entering coordinate through the held weights) over the fan-out allowance e k / beta_x^rep."""
    room = k - l_w - A_DEC
    return room * p * beta_rep / (beta_in * e * k) if room > 0 else 0.0


def closure(cfg: configs.Config, beta_w: float, beta_rep: dict[str, float], k: float = K) -> list[dict]:
    """Held sets of an MoE config and their allowance ratio R at both residual readings. A boundary-entry block enters at
    the cheapest boundary (beta_in = beta_x^rep at Ff_min, whatever reading the fan-out allowance uses); an embedding-entry
    block reads a token id and the token's embedding row (d beta_w bits, the table's mean code per entry: an estimate),
    so per d-wide coordinate beta_in = (17 + d beta_w) / d and p = products per token / d. A block spanning layers needs
    every routed expert between them, so no held set crosses a layer's MoE."""
    d = cfg.d
    attn_p = sum(m.d_in * m.d_out for m in cfg.maps if not m.mlp)
    e_ffn = next(m.d_out for m in cfg.maps if m.name == "e_gate")
    expert_p = 3 * d * e_ffn
    router_p = d * cfg.n_experts
    share = cfg.n_active / cfg.n_experts
    e = max(sum(m.d_out * m.active for m in cfg.maps if m.resid_in and m.mlp),
            sum(m.d_out * m.active for m in cfg.maps if m.resid_in and not m.mlp))
    rows = []

    def add(name: str, params: float, prods_per_tok: float, note: str = "", embedding: bool = False):
        l_w = params * beta_w
        p = prods_per_tok / d
        beta_in = (H_TOK + d * beta_w) / d if embedding else beta_rep["Ff_min"]
        rows.append({"held": name, "entry": "embedding" if embedding else "cheapest boundary", "l_w": l_w, "p": p,
                     "p_over_e": p / e, "beta_in": beta_in,
                     "R": {r: corrected_R(p, l_w, beta_in, beta_rep[r], e, k) for r in READINGS}, "note": note})

    add("attention, 1 layer", attn_p, attn_p, "attention of n layers is n such blocks: the MoE between isn't held")
    add("shared expert + router", expert_p + router_p, expert_p + router_p)
    add("1 routed expert, its tokens only", expert_p + router_p, expert_p + router_p)
    for h in range(2, cfg.n_active + 1):
        add(f"{h} routed + shared + router", (h + 1) * expert_p + router_p, (h + 1) * expert_p + router_p,
            f"best case: only tokens routing all {cfg.n_active} of theirs into the held {h}" if h == cfg.n_active else
            f"best case: tokens routing {h} of their {cfg.n_active} into it, the rest re-entering elsewhere")
    add("attention + shared + router", attn_p + expert_p + router_p, attn_p + expert_p + router_p,
        "whole rollouts: tokens can't be chosen by routing")
    for h in (1, 4, 16):
        add(f"attention + shared + router + {h} routed", attn_p + (h + 1) * expert_p + router_p,
            attn_p + expert_p + router_p + h * share * expert_p, "routed experts see A/E of the tokens on average")
    add("embedding rows + layer-0 attention", attn_p, attn_p, "token ids in", embedding=True)
    add("embedding rows + layer-0 attention + shared + router", attn_p + expert_p + router_p,
        attn_p + expert_p + router_p, "token ids in", embedding=True)
    return rows


def dense_loss(cfg: configs.Config, beta_w: float, beta_rep: dict[str, float], k: float = K) -> list[dict]:
    """Whole consecutive layers of a dense config, entering at the cheapest boundary or at the embedding (rows at
    d beta_w bits): R at the best m, beside the continuous optimum k / (4 beta_w e d) of boundary entry at one reading
    (a_dec left out)."""
    w = beta_w * cfg.layer_params
    N = sum(m.d_in * m.d_out for m in cfg.maps)
    e = max(sum(m.d_out for m in cfg.maps if m.resid_in and m.mlp), sum(m.d_out for m in cfg.maps if m.resid_in and not m.mlp))
    rows = []
    for entry, beta_in in (("cheapest boundary", beta_rep["Ff_min"]), ("embedding", (H_TOK + cfg.d * beta_w) / cfg.d)):
        R = {}
        for r in READINGS:
            R[r] = max((corrected_R(m * N / cfg.d, m * w, beta_in, beta_rep[r], e, k), m)
                       for m in range(1, cfg.layers + 1))
        rows.append({"held": "m whole layers", "entry": entry, "beta_in": beta_in, "m": R["Ff_min"][1],
                     "l_w": R["Ff_min"][1] * w, "p": R["Ff_min"][1] * N / cfg.d, "p_over_e": R["Ff_min"][1] * N / cfg.d / e,
                     "R": {r: R[r][0] for r in READINGS},
                     "R_continuous": {r: k * beta_rep[r] / (4 * w / N * e * cfg.d * beta_in) for r in READINGS},
                     "note": "continuous optimum k beta_x^rep / (4 beta_w e d beta_in): k / (4 beta_w e d) at "
                             "beta_in = beta_x^rep"})
    return rows


# ------------------------------------------------------------------------------------------------------------ report

def sig(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3g}" if 1e-3 <= abs(x) < 1e5 else f"{x:.2e}"


MATCHED = "A2_gsm8k-full-seed1_step150"
#: beta_w per precision: A2's full arm (median over its four GSM8K and two MATH entries) and every A2 arm (B1's median)
BETA_W = {"fp8": {"full arm": 0.108, "all A2 arms": 0.103}, "nvfp4": {"full arm": 0.0096, "all A2 arms": 0.036}}


def medians(measured_doc: dict, cfg: configs.Config) -> dict[str, dict[str, float]]:
    """beta_x^rep at both readings from B1's model-wide medians (F scaled by d / 1536, as rho_attack scales it)."""
    inp = ra.Inputs.from_json(measured_doc)
    out = {}
    for p in PRECISIONS:
        F_min = inp.F_for("full", p, cfg)
        F_mean = measured_doc["F_layer_mean"]["full"][p] * cfg.d / inp.d_model
        out[p] = {"F_min": F_min, "F_mean": F_mean, "Ff_min": (H_TOK + F_min) / cfg.d, "mean": (H_TOK + F_mean) / cfg.d}
    return out


def _block_row(label: str, inputs: str, b: dict, st: Step | None, brep: tuple[float, float]) -> str:
    ppe = b["products_per_entering_coordinate"]
    q = f"{b['rollouts']} x {st.len:.0f} = {b['Q']:,.0f}" if st else f"{b['Q']:,.0f} (fills k)"
    return (f"| {label} | {inputs} | {b['entry']}, layers {b['layers']} | {q} | "
            f"{sig(b['l_w'])} | {sig(b['l_in'])} | {sig(b['a_dec'])} | {sig(b['n_I'])} | "
            f"{sig(b['beta_x_bound'])} | {sig(brep[0])}, {sig(brep[1])} | "
            f"{sig(b['violation']['Ff_min'])}, {sig(b['violation']['mean'])} | "
            f"{'token ids only' if ppe is None else sig(ppe)} | {sig(b['R']['Ff_min'])}, {sig(b['R']['mean'])} | "
            f"{'n/a' if 'decode_flops' not in b else sig(b['decode_flops'] / C)} |")


def report(doc: dict, measured_doc: dict, existing: dict) -> tuple[str, dict]:
    qwen = configs.get("qwen2.5-1.5b")
    med = medians(measured_doc, qwen)
    res = {"blocks": {}, "coordinator_instantiation": {}, "steps": {}, "closure": {}, "dense": {}, "loss_factors": {},
           "medians": med, "k": K, "c": C, "a_dec": A_DEC}
    ent = doc["entries"][MATCHED]
    e = Step(ent, "fp8", qwen).e
    out = [f"## 1. Matched block: one GRPO step of A2's full-parameter arm (Qwen2.5-1.5B-Instruct, {MATCHED})", "",
           f"k = {K:.3g} bits (2.5 MB), c = {C:.3g} FLOPs, a_dec = {A_DEC:,.0f} bits (lzma of B1's coders and "
           "transformers' Qwen2 forward). The step trains "
           f"{ent['n_seq']} kept rollouts of {ent['rollouts']} generated, {Step(ent, 'fp8', qwen).len:.0f} tokens each "
           f"(prompt {ent['prompt_mean']} measured + completion from A2's log). e = {e:,}; counted activations per "
           f"token-layer {Step(ent, 'fp8', qwen).acts:,}; products per token-layer {Step(ent, 'fp8', qwen).prods:,}. "
           "beta_x^rep = (17 + F)/d at the cheapest boundary (Ff_min) and at the mean over boundaries; violation = "
           "beta_x^rep / bound; R = beta_x^rep products / (e (l_w + l_in + a_dec)), the block's products over the "
           "fan-out allowance at its own input size.", "",
           "| block | inputs | entry, held layers | tokens (rollouts x length) | l_w | l_in | a_dec | n_I | beta_x <= | "
           "beta_x^rep (Ff_min, mean) | violation (Ff_min, mean) | products / entering coordinate (e = 17,920) | "
           "R (Ff_min, mean) | decode FLOPs / c |", "|" + "---|" * 14]
    for p in PRECISIONS:
        bw = BETA_W[p]["all A2 arms"]
        for reading, F in (("Ff_min", med[p]["F_min"]), ("mean", med[p]["F_mean"])):
            mb = min((median_block(qwen, bw, F, m, med[p]) for m in range(1, qwen.layers + 1)
                      if m * bw * qwen.layer_params < K), key=lambda x: x["beta_x_bound"])
            res["coordinator_instantiation"][f"{p}/{reading}"] = mb
            out.append(_block_row(f"coordinator's instantiation, {p}, entering at F_{reading}",
                                  f"medians (beta_w {bw}, F {F:,.0f})", mb, None, (med[p]["Ff_min"], med[p]["mean"])))
        st = Step(ent, p, qwen)
        brep = (st.beta_rep("Ff_min"), st.beta_rep("mean"))
        for tag, b in (("best boundary entry, measured weights", st.best_block(True, "boundary")),
                       ("best boundary entry", st.best_block(False, "boundary")),
                       ("best embedding entry, measured weights", st.best_block(True, "embedding")),
                       ("best embedding entry", st.best_block(False, "embedding"))):
            if b is None:
                continue
            res["blocks"][f"{MATCHED}/{p}/{tag}"] = b
            src = "per-layer F, per-map weight codes" + ("" if b["weights_measured"] else
                                                          " (unsampled layers at the largest sampled layer's code)")
            out.append(_block_row(f"{tag}, {p}", src, b, st, brep))
    out += ["", "### The other A2 full-arm GSM8K entries (per-entry, per-layer inputs)", "",
            "| entry | prec | block | tokens | l_w | l_in | n_I | beta_x <= | violation (Ff_min, mean) | R (Ff_min, mean) |",
            "|" + "---|" * 10]
    for name, en in doc["entries"].items():
        if en["prompt_mean"] is None or name == MATCHED:
            continue
        for p in PRECISIONS:
            st = Step(en, p, qwen)
            for tag, b in (("boundary, measured", st.best_block(True, "boundary")),
                           ("embedding, measured", st.best_block(True, "embedding")),
                           ("best", st.best_block())):
                if b is None:
                    continue
                res["blocks"][f"{name}/{p}/{tag}"] = b
                out.append(f"| {name} | {p} | {tag}: {b['entry']}, layers {b['layers']} | {b['Q']:,.0f} | "
                           f"{sig(b['l_w'])} | {sig(b['l_in'])} | {sig(b['n_I'])} | {sig(b['beta_x_bound'])} | "
                           f"{sig(b['violation']['Ff_min'])}, {sig(b['violation']['mean'])} | "
                           f"{sig(b['R']['Ff_min'])}, {sig(b['R']['mean'])} |")

    out += ["", "## 2. The step from such blocks (forward, backward, accumulation, rollout generation)", "",
            "Layers: contiguous segments (DP), forward entering at F, backward exiting with G, whole rollouts per block. "
            "Accumulator: chain carries the running gradient at beta_w per weight; reduce emits raw 16-bit partial "
            "gradients that reducing blocks read. Head: the tied table in r vocabulary-row groups. Generation: "
            "the same segments, forward only, the token-by-token dependency ignored (rho_attack's convention). "
            "kappa = T_0 / M with M the training forward's layer products (fanout.py's N per trained token); "
            "kappa' adds generation's products.", "",
            "| entry | prec | accumulator | train: layer blocks (segments) + head blocks (groups) | generation: layer "
            "blocks + head blocks | blocks | T_0 (FLOPs) | kappa (kappa') | rho = c blocks / T_0 (one-block floor "
            "c / T_0) | fan-out prediction at kappa (Ff_min, mean) | at kappa' (Ff_min, mean) |",
            "|" + "---|" * 11]
    for name, en in doc["entries"].items():
        if en["prompt_mean"] is None:
            continue
        for p in PRECISIONS:
            st = Step(en, p, qwen)
            for acc in ("chain", "reduce"):
                s = st.step(acc)
                res["steps"][f"{name}/{p}/{acc}"] = s
                segs = " ".join(f"{a}-{b - 1}" for a, b in s["train_segments"]) if math.isfinite(s["train_blocks"]) else ""
                fp, fg = s["fanout_prediction"], s["fanout_prediction_with_generation_products"]
                out.append(f"| {name} | {p} | {acc} | {sig(s['train_blocks'])} ({segs}) + {sig(s['train_head_blocks'])} "
                           f"({s['train_head_groups']}) | {sig(s['gen_blocks'])} + {sig(s['gen_head_blocks'])} | "
                           f"{sig(s['blocks'])} | {sig(s['T0'])} | {sig(s['kappa'])} ({sig(s['kappa_with_generation_products'])}) | "
                           f"{sig(s['rho'])} ({sig(s['rho_one_block'])}) | {sig(fp['Ff_min'])}, {sig(fp['mean'])} | "
                           f"{sig(fg['Ff_min'])}, {sig(fg['mean'])} |")
    out += ["", "rho_attack's own Qwen2.5-1.5B full-training GRPO rows at the same k and c (model-wide medians, "
            "2^20-token steps, every generated token trained, rho against 4f per token):", "",
            "| prec | best exact attack | best attack |", "|---|---|---|"]
    for p in PRECISIONS:
        out.append(f"| {p} | {existing[p]['exact']} | {existing[p]['best']} |")
    res["existing_rows"] = {p: {k_: v for k_, v in existing[p].items() if k_ != 'row'} for p in PRECISIONS}

    out += ["", "## 3. Closure check, Kimi K2 and DeepSeek-V4-Pro at k = 2.5 MB", "",
            "Inputs are B1's medians (beta_w per column; F_min and F_mean of the full arm scaled by d/1536). "
            "R = (k - l_w - a_dec) p beta_x^rep / (beta_in e k): the corrected allowance over the fan-out allowance. "
            "Boundary entry pays beta_in = beta_x^rep(Ff_min); embedding entry pays (17 + d beta_w)/d per coordinate "
            "with p = products per token / d. R > 1 beats the fan-out allowance e k / beta_x^rep.", "",
            "| config | prec | held set | entry | p | p / e | l_w (full arm, all arms) | R full arm (Ff_min, mean) | "
            "R all arms (Ff_min, mean) | note |", "|" + "---|" * 10]
    for cname in ("kimi-k2", "deepseek-v4-pro"):
        cfg = configs.get(cname)
        mc = medians(measured_doc, cfg)
        for p in PRECISIONS:
            tables = {which: closure(cfg, b, mc[p]) for which, b in BETA_W[p].items()}
            for which, rows in tables.items():
                res["closure"][f"{cname}/{p}/{which}"] = rows
            for rf, ra_ in zip(tables["full arm"], tables["all A2 arms"]):
                out.append(f"| {cname} | {p} | {rf['held']} | {rf['entry']} | {sig(rf['p'])} | {sig(rf['p_over_e'])} | "
                           f"{sig(rf['l_w'])}, {sig(ra_['l_w'])} | {sig(rf['R']['Ff_min'])}, {sig(rf['R']['mean'])} | "
                           f"{sig(ra_['R']['Ff_min'])}, {sig(ra_['R']['mean'])} | {rf['note']} |")

    out += ["", "## 4. Loss factor of the fan-out allowance: max(1, max R over blocks)", "",
            "Per config, precision, beta_w reading and residual reading. 'routing' rows need tokens chosen by their "
            "routing; 'embedding' rows price an embedding row at d beta_w (an estimate).", "",
            "| config | prec | beta_w | reading | best boundary block (R) | best embedding block (R) | best routing-chosen "
            "block (R) | loss factor, whole-rollout blocks | loss factor, any block | step-level loss factor (share of "
            "products the embedding block can cover) |", "|" + "---|" * 10]
    for cname in ("qwen2.5-1.5b", "kimi-k2", "deepseek-v4-pro"):
        cfg = configs.get(cname)
        mc = medians(measured_doc, cfg)
        for p in PRECISIONS:
            for which, b in BETA_W[p].items():
                rows = dense_loss(cfg, b, mc[p]) if cfg.n_experts == 0 else closure(cfg, b, mc[p])
                if cfg.n_experts == 0:
                    res["dense"][f"{cname}/{p}/{which}"] = rows
                for r in READINGS:
                    bb = _best(rows, r, "boundary")
                    be = _best(rows, r, "embedding")
                    br = _best(rows, r, "routing")
                    whole = max(x["R"][r] for x in (bb, be) if x)
                    anyb = max([whole] + ([br["R"][r]] if br else []))
                    s = embedding_share(cfg, be)
                    rest = max(1.0, max(x["R"][r] for x in (bb, br) if x))
                    agg = 1.0 / (s / max(1.0, be["R"][r]) + (1 - s) / rest)
                    res["loss_factors"][f"{cname}/{p}/{which}/{r}"] = {
                        "boundary": bb and (bb["held"], bb["R"][r]), "embedding": be and (be["held"], be["R"][r]),
                        "routing": br and (br["held"], br["R"][r]), "whole_rollouts": max(1.0, whole),
                        "any": max(1.0, anyb), "embedding_share_of_products": s, "aggregate": agg}
                    out.append(f"| {cname} | {p} | {b} ({which}) | {r} | {_show(bb, r)} | {_show(be, r)} | "
                               f"{_show(br, r)} | {sig(max(1.0, whole))} | {sig(max(1.0, anyb))} | {sig(agg)} ({s:.1%}) |")
    out += ["", "The step-level column gives every product outside the embedding block the best other block's R, so it "
            "bounds the step-level loss from above. With B1's per-boundary F (the matched step, Qwen2.5-1.5B), one "
            "forward over the layers in contiguous segments, each entering at its own boundary or the embedding, "
            "tokens unbounded:", "",
            "| entry | prec | segments | corrected blocks per token | fan-out blocks per token (Ff_min, mean) | "
            "step-level loss factor (Ff_min, mean) |", "|---|---|---|---|---|---|"]
    for name, en in doc["entries"].items():
        if en["prompt_mean"] is None:
            continue
        for p in PRECISIONS:
            fl = Step(en, p, qwen).forward_loss()
            res["forward_loss"] = res.get("forward_loss", {}) | {f"{name}/{p}": fl}
            segs = " ".join(f"{a}-{b - 1}" for a, b in fl["segments"])
            fb = fl["fanout_blocks_per_token"]
            out.append(f"| {name} | {p} | {segs} | {sig(fl['blocks_per_token'])} | {sig(fb['Ff_min'])}, "
                       f"{sig(fb['mean'])} | {sig(fl['loss']['Ff_min'])}, {sig(fl['loss']['mean'])} |")
    return "\n".join(out) + "\n", res


def _kind(row: dict) -> str:
    if row["entry"] == "embedding":
        return "embedding"
    return "routing" if "routed" in row["held"] and "attention" not in row["held"] else "boundary"


def _best(rows: list[dict], reading: str, kind: str) -> dict | None:
    cand = [x for x in rows if _kind(x) == kind]
    return max(cand, key=lambda x: x["R"][reading]) if cand else None


def _show(row: dict | None, reading: str) -> str:
    if row is None:
        return "n/a"
    held = row["held"] + (f", m = {row['m']}" if "m" in row else "")
    return f"{held} ({sig(row['R'][reading])})"


def embedding_share(cfg: configs.Config, row: dict) -> float:
    """Share of a forward's layer products an embedding-entry block can reach: its held set, once (layer 0 only)."""
    total = cfg.layers * sum(m.d_in * m.d_out * m.active for m in cfg.maps)
    return row["p"] * cfg.d / total


def existing_rows(measured_doc: dict) -> dict:
    """rho_attack's own Qwen2.5-1.5B full-training GRPO rows at the same k and c (model-wide medians)."""
    inp = ra.Inputs.from_json(measured_doc)
    cfg = configs.get("qwen2.5-1.5b")
    out = {}
    for p in PRECISIONS:
        row = ra.evaluate(cfg, inp, ra.Knobs(lora_bits=16), ["full"], K, C, p, "grpo")[0]

        def show(a):
            return "none" if not a else f"rho {sig(a['rho'])} ({a['strategy']})"
        out[p] = {"row": row, "exact": show(row.get("attack_exact")), "best": show(row.get("attack"))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--extract", nargs=2, metavar=("B1_BUNDLE", "A2_RUNS"))
    ap.add_argument("--prompt-mean", type=float, default=103.6)
    ap.add_argument("--inputs", default=str(Path(__file__).resolve().parent / "inputs" / "a2_full_blocks.json"))
    ap.add_argument("--measured", default=str(Path(__file__).resolve().parent / "inputs" / "measured_b1.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.extract:
        doc = extract(Path(args.extract[0]), Path(args.extract[1]), args.prompt_mean)
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{len(doc['entries'])} entries -> {args.out}")
        return 0
    doc = json.loads(Path(args.inputs).read_text())
    measured = json.loads(Path(args.measured).read_text())
    text, res = report(doc, measured, existing_rows(measured))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_block.md").write_text(text)
    (out / "matched_block.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
