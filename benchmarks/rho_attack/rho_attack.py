"""rho_attack(S, k, c): the cheapest layout of a training run as inference-like blocks, against the bound's rho(S).

A block does at most ``c`` FLOPs and reads at most ``k`` unregistered input bits (theta is registered and free). The
attack's slowdown is ``rho = c * blocks / honest FLOPs``: blocks used, over the blocks an unconstrained run of the same
honest work would need. Definitions, F, beta and the bound's formulas are note:console/20261024T1810Z-finding-production-
bound-red-team and note:console/20261024T1900Z-draft-training-vs-inference-program.

Pricing, shared by every strategy:

- a block may hold token ids (``h`` bits each) and recompute theta's own activations and gradients for them; the FLOPs of
  that recompute count against ``c`` but are not useful work;
- whatever crosses into a block costs its fresh part given what the block can recompute: a residual-stream crossing ``F``
  bits per token forward and ``G = grad_factor * F`` backward; an interior activation ``beta_x`` per coordinate; an
  interior output gradient ``beta_g`` per value; a counted value assembled from another block ``beta_v``;
- the state a block holds costs its bits: a share of ``S`` (spread over parameters), or ``beta_w`` per weight of a row,
  plus ``b_acc`` per weight for the running gradient accumulator a token tile passes to the next one;
- blocks are a partition of the training circuit, so a block may feed one that feeds it back (forward and backward of one
  stage share a block).

The strategies are cost models with free parameters (stages, rank, rows, tokens, rollouts per prompt); the optimizer is a
search over those parameters, and ``rho_attack`` is the minimum over strategies. Each strategy that goes below the bound
says which assumption of the finding's section 4 it breaks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import configs  # noqa: E402
from configs import Config, Map  # noqa: E402

KB = 8_000  # bits in a (decimal) kB, as the notes use it
C_FIXED = 1.4e17
K_GRID = {"10kB": 10 * KB, "100kB": 100 * KB, "1MB": 1000 * KB, "2.5MB": 2500 * KB}  # 2.5 MB: the deck's k


@dataclass(frozen=True)
class Beta:
    """Fresh bits per counted value (``beta_v``), per input coordinate (``beta_x``) and per weight of the update
    (``beta_w``) at one precision, with the derived defaults the measurement doesn't give."""

    beta_v: float
    beta_x: float
    beta_w: float
    beta_g: float | None = None  # per interior output-gradient value; default beta_v
    beta_gx: float | None = None  # per counted value given theta's weights on theta's input; default beta_v
    raw_bits: float | None = None  # per coordinate sent without a theta baseline; default 2 * beta_x
    b_acc: float | None = None  # per weight of a gradient accumulator passed between token tiles; default beta_w

    @property
    def g(self) -> float:
        return self.beta_v if self.beta_g is None else self.beta_g

    @property
    def gx(self) -> float:
        """A counted value (or its gradient) sent to a block that recomputes theta's: the correction's fresh bits."""
        return self.beta_v if self.beta_gx is None else self.beta_gx

    @property
    def raw(self) -> float:
        return 2 * self.beta_x if self.raw_bits is None else self.raw_bits

    @property
    def acc(self) -> float:
        return self.beta_w if self.b_acc is None else self.b_acc


@dataclass(frozen=True)
class Knobs:
    grad_factor: float = 1.0  # G / F: the backward crossing's fresh bits against the forward's
    ranks: tuple[int, ...] = (64,)  # lossy low-rank corrections tried (the program's "rank 64")
    b_coef: float = 4.0  # bits per low-rank coefficient per token
    b_basis: float = 4.0  # bits per basis coordinate (d_cross x r per block per crossing)
    lora_bits: float = 2.0  # bits per LoRA parameter, for "rankR" labels
    prompt: int = 256  # GRPO prompt tokens
    completion: int = 768  # GRPO completion tokens
    group: int = 8  # GRPO rollouts per prompt
    seed_bits: int = 64  # sampling seed per rollout generated inside a block
    phi0: float = 0.5  # hybrid: share of honest compute done while the state still fits in k
    traffic: float = 1.0  # phi = traffic * f / h
    expert_out_factor: float = 1.0  # one routed expert's output correction, in units of F
    max_stages: int | None = None  # default 2 * layers (cuts after attention and after the MLP)
    step_tokens: float = 2.0**20  # tokens per optimizer step: a block holding fewer carries the step's accumulator


@dataclass
class Attack:
    strategy: str
    rho: float
    params: dict
    breaks: str | None = None  # the finding-§4 assumption a sub-bound result of this strategy breaks

    def to_json(self) -> dict:
        return {"strategy": self.strategy, "rho": self.rho, "params": self.params, "breaks": self.breaks}


@dataclass(frozen=True)
class Inputs:
    """Measured (or in-hand) beta per precision and F per S label, in the format B1 writes."""

    precisions: dict[str, Beta]
    F: dict[str, float | dict[str, float]]
    d_model: int
    F_scaling: str = "per_dim"  # per_dim: F / d is what transfers to another model; absolute: F does
    S_bits: dict[str, float] = field(default_factory=dict)
    beta_by_S: dict[str, dict[str, Beta]] = field(default_factory=dict)
    source: str = ""
    G: dict[str, float | dict[str, float]] = field(default_factory=dict)  # backward crossing per S label; default F

    @classmethod
    def load(cls, path: str | Path) -> "Inputs":
        doc = json.loads(Path(path).read_text())
        return cls.from_json(doc)

    @classmethod
    def from_json(cls, doc: dict) -> "Inputs":
        missing = [k for k in ("precisions", "F") if k not in doc]
        if missing:
            raise ValueError(f"rho_attack inputs need {missing}: {{'precisions': {{P: {{beta_v, beta_x, beta_w}}}}, "
                             f"'F': {{S label: bits per token}}, 'measured_on': {{'d_model': D}}}}")
        betas = {}
        for p, b in doc["precisions"].items():
            keys = {k: b[k] for k in ("beta_v", "beta_x", "beta_w", "beta_g", "beta_gx", "raw_bits", "b_acc") if k in b}
            for need in ("beta_v", "beta_x", "beta_w"):
                if need not in keys:
                    raise ValueError(f"precision {p!r} lacks {need}")
            betas[p] = Beta(**keys)
        by_s = {lab: {p: replace(betas[p], **v) for p, v in per.items()} for lab, per in doc.get("beta_by_S", {}).items()}
        d_model = int(doc.get("measured_on", {}).get("d_model", doc.get("d_model", 0)))
        scaling = doc.get("F_scaling", "per_dim")
        if scaling == "per_dim" and d_model <= 0:
            raise ValueError("F_scaling 'per_dim' needs measured_on.d_model, the width F was measured at")
        return cls(betas, doc["F"], d_model, scaling, dict(doc.get("S_bits", {})), by_s, doc.get("source", ""),
                   dict(doc.get("G", {})))

    def F_for(self, label: str, precision: str, cfg: Config) -> float:
        v = self.F[label]
        if isinstance(v, dict):
            v = v[precision]
        return float(v) * (cfg.d / self.d_model if self.F_scaling == "per_dim" else 1.0)

    def grad_factor(self, label: str, precision: str, kn: "Knobs") -> float:
        """G / F for a label: measured when the inputs give G, else the knob."""
        if label not in self.G:
            return kn.grad_factor
        g, f = self.G[label], self.F[label]
        g = g[precision] if isinstance(g, dict) else g
        f = f[precision] if isinstance(f, dict) else f
        return float(g) / float(f)

    def beta_for(self, label: str, precision: str) -> Beta:
        return self.beta_by_S.get(label, {}).get(precision, self.precisions[precision])


_SIZE = re.compile(r"^([\d.]+)\s*(bit|kbit|mbit|b|kb|mb|gb)$", re.IGNORECASE)
_UNITS = {"bit": 1, "kbit": 1e3, "mbit": 1e6, "b": 8, "kb": 8e3, "mb": 8e6, "gb": 8e9}


def lora_params(cfg: Config, r: int) -> float:
    return cfg.layers * sum(r * (m.d_in + m.d_out) * m.count for m in cfg.maps)


def resolve_S(label: str, cfg: Config, beta: Beta, kn: Knobs, explicit: dict[str, float] | None = None) -> float:
    """Bits of unregistered state for an S label on ``cfg``: a size ("10kB", "10kbit"), "tinylora" (26 bytes), "rankR"
    or "loraR" (a rank-R LoRA on every map at ``lora_bits`` per parameter) or "full" (every weight at beta_w)."""
    low = label.lower().replace("_", "").replace("-", "")
    if low == "full":
        return cfg.params * beta.beta_w
    if low == "tinylora":
        return 26 * 8
    if m := re.match(r"^(?:rank|lora)(\d+)$", low):
        # the factors at lora_bits, or the dense delta they make at beta_w per weight, whichever is smaller
        return min(lora_params(cfg, int(m.group(1))) * kn.lora_bits, cfg.layers * cfg.layer_params * beta.beta_w)
    if explicit and label in explicit:
        return float(explicit[label])
    if m := _SIZE.match(low):
        return float(m.group(1)) * _UNITS[m.group(2).lower()]
    raise ValueError(f"S label {label!r}: give a size (10kB, 10kbit), tinylora, rankR, full, or an S_bits entry")


def tokens_per_block(k: float, c: float, state: float, bits_tok, flops_tok):
    """Tokens a block holds beside ``state`` bits: the input bound and the compute bound, whichever binds."""
    room = k - state
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.minimum(np.where(bits_tok > 0, room / np.maximum(bits_tok, 1e-300), np.inf), c / flops_tok)
    return np.where(room > 0, q, 0.0)


def tokens_with_acc(k: float, c: float, state, acc, bits_tok, flops_tok, kn: Knobs, share: float = 1.0):
    """Tokens per block under the fixed recipe: a block that holds a whole step's tokens (``share`` of the step's
    tokens reach this block's map, e.g. A / E for a routed expert) applies its own update; a smaller one carries the
    step's running gradient (``acc`` bits) to the next block beside its state."""
    q0 = tokens_per_block(k, c, state, bits_tok, flops_tok)
    q1 = tokens_per_block(k, c, state + acc, bits_tok, flops_tok)
    return np.where(q0 >= kn.step_tokens * share, q0, q1)


# --------------------------------------------------------------------------------------------------------- strategies


def whole_state(cfg: Config, S: float, k: float, c: float, beta: Beta, kn: Knobs, workload: str) -> Attack | None:
    """The whole state rides in every block beside a batch, so every step is an inference-like session (S <= k).

    SFT pays ``h`` per token of data. GRPO generates its rollouts inside the block from a prompt and a seed, and shares
    the prompt across the rollouts the block holds (duplicate sharing): it pays ``p * h`` per prompt."""
    f = cfg.f
    acc = acc_bits(beta, S / cfg.params) * cfg.params
    if workload == "sft":
        q = float(tokens_with_acc(k, c, S, acc, cfg.h, 3 * f, kn))
        if q < 1:
            return None
        return Attack("whole_state", c / (q * 3 * f), {"tokens": q, "input_bound": q < c / (3 * f)})
    p, g, G = kn.prompt, kn.completion, kn.group
    best = None
    for n in range(1, G + 1):
        bits, flops = p * cfg.h + n * kn.seed_bits, 4 * f * (p + n * g)
        m0 = float(tokens_per_block(k, c, S, bits, flops))
        m = m0 if m0 * n * (p + g) >= kn.step_tokens else float(tokens_per_block(k, c, S + acc, bits, flops))
        if m < 1:
            continue
        useful = m * n * 4 * f * (g + p / G)
        a = Attack("whole_state", c / useful, {"prompts": m, "rollouts_per_prompt": n})
        best = a if best is None or a.rho < best.rho else best
    return best


def half_layers(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(parameters, forward FLOPs per token) of the pipeline's units in order: each layer's attention half and its MLP
    (or MoE) half, the residual points between them being the only cuts a crossing costs F at. The head is not a unit:
    the pipeline tiles it (``cfg.head``), and a first-stage token brings its own embedding row."""
    params, flops = [], []
    attn = [m for m in cfg.maps if not m.mlp]
    attn_p = sum(m.params for m in attn)
    attn_f = 2 * sum(m.d_in * m.d_out * m.active for m in attn) + 4 * cfg.attn_width * cfg.ctx_eff
    for _ in range(cfg.layers):
        params += [attn_p, cfg.layer_params - attn_p]
        flops += [attn_f, cfg.f_layer - attn_f]
    return np.asarray(params, dtype=float), np.asarray(flops, dtype=float)


def stages(cfg: Config, P: int) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous groups of the half-layer units into P stages balanced by parameters: (parameters, FLOPs) each."""
    params, flops = half_layers(cfg)
    cum = np.concatenate([[0.0], np.cumsum(params)])
    cuts = np.searchsorted(cum, cum[-1] * np.arange(1, P) / P)
    edges = np.unique(np.concatenate([[0], np.clip(cuts, 1, len(params) - 1), [len(params)]]))
    if len(edges) - 1 != P:
        return np.empty(0), np.empty(0)
    cf = np.concatenate([[0.0], np.cumsum(flops)])
    return np.diff(cum[edges]), np.diff(cf[edges])


def _stage_costs(cfg: Config, f_stage: np.ndarray, F: float, G: float, beta: Beta, grad_mode: str,
                 rank: int | None, kn: Knobs, train: bool, emb_bits: float,
                 head_merged: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fixed bits, bits per token, FLOPs per token) of each stage of a layer pipeline. Unless the head is merged into
    the last stage, every stage takes a gradient from above (the head's tiles sit past the last stage)."""
    f = cfg.f
    P = len(f_stage)
    s = np.arange(P)
    f_s = f_stage
    f_pre = np.concatenate([[0.0], np.cumsum(f_stage)[:-1]])
    f_suf = f - f_pre - f_stage
    has_in, has_grad = s > 0, np.full(P, train) & ~((s == P - 1) & head_merged)
    cross = rank * kn.b_coef if rank else min(F, cfg.d_cross * beta.raw)
    basis = cfg.d_cross * rank * kn.b_basis if rank else 0.0
    bits = cfg.h + has_in * cross + (s == 0) * emb_bits
    fixed = has_in * basis
    rec = has_in * f_pre  # theta's activations at entry, to decode the forward correction
    if grad_mode == "corr":
        bits = bits + has_grad * (cross if rank else G)
        fixed = fixed + has_grad * basis
        rec = np.where(has_grad, f + 2 * f_suf, rec)  # theta's forward to the loss and backward to the exit
    else:
        bits = bits + has_grad * cfg.d_cross * beta.raw
    return fixed, bits, (3 if train else 1) * f_s + rec


def pipeline(cfg: Config, S: float, F: float, k: float, c: float, beta: Beta, kn: Knobs, workload: str,
             rank: int | None = None) -> Attack | None:
    """Layer pipelining: the state split by layers into P stages (cuts at residual points), each stage holding its share
    whole; every token crosses each cut forward (F) and backward (G). With ``rank``, each crossing is a rank-r correction
    instead: r * b_coef per token plus the d_cross x r basis once per block."""
    G = kn.grad_factor * F
    u = (4 if workload == "grpo" else 3) * cfg.f
    s_pp = S / cfg.params
    passes = (True, False) if workload == "grpo" else (True,)
    cross = rank * kn.b_coef if rank else F
    head_tiled = sum(_map_blocks(cfg, cfg.head, cfg.layers - 1, s_pp, cross, k, c, beta, kn, "exact", t, False)[0]
                     for t in passes)
    head_params = cfg.vocab * cfg.d
    best = None
    for P in range(1, (kn.max_stages or 2 * cfg.layers) + 1):
        if S / P >= k:
            continue
        params, f_stage = stages(cfg, P)
        if len(params) == 0:
            continue
        for merged in (False, True):
            p_st, f_st = params.copy(), f_stage.copy()
            if merged:
                p_st[-1] += head_params
                f_st[-1] += cfg.f_head
            elif not math.isfinite(head_tiled):
                continue
            if s_pp * p_st.max() >= k:
                continue
            for grad_mode in ("corr", "raw"):
                blocks = 0.0 if merged else head_tiled
                for train in passes:
                    fixed, bits, flops = _stage_costs(cfg, f_st, F, G, beta, grad_mode, rank, kn, train,
                                                      cfg.d * s_pp, merged)
                    acc = acc_bits(beta, s_pp) * p_st if train else 0.0 * p_st
                    q = tokens_with_acc(k, c, s_pp * p_st + fixed, acc, bits, flops, kn)
                    if np.any(q < 1):
                        blocks = math.inf
                        break
                    blocks += float(np.sum(1 / q))
                if not math.isfinite(blocks):
                    continue
                a = Attack("lowrank_pipeline" if rank else "layer_pipeline", c * blocks / u,
                           {"stages": P, "grad": grad_mode, "head": "merged" if merged else "tiled",
                            **({"rank": rank} if rank else {})})
                best = a if best is None or a.rho < best.rho else best
    return best


def lowrank_pipeline(cfg: Config, S: float, F: float, k: float, c: float, beta: Beta, kn: Knobs,
                     workload: str) -> Attack | None:
    """Recompute theta's activations from tokens and send rank-r corrections at each crossing. Exact only when the
    fresh part has rank <= r; when r * b_coef < F (forward) or < G (backward), the attack approximates (E) or beats the
    measurement's compressor (D)."""
    best = None
    for r in kn.ranks:
        a = pipeline(cfg, S, F, k, c, beta, kn, workload, rank=r)
        if a is not None and (best is None or a.rho < best.rho):
            best = a
    if best is not None:
        r_bits = best.params["rank"] * kn.b_coef
        if r_bits < F:
            best.breaks = "E or D: a rank-r forward correction cheaper than the measured fresh part F"
        elif r_bits < kn.grad_factor * F:
            best.breaks = "E or D: a rank-r gradient correction cheaper than the measured fresh part G"
    return best


_R_GRID = 96


def acc_bits(beta: Beta, S_per_param: float) -> float:
    """Accumulator bits per weight: the state's own parameterization, so b_acc at full-training density and
    proportionally less for a smaller state."""
    return beta.acc * min(1.0, S_per_param / beta.beta_w) if beta.beta_w > 0 else beta.acc

LSE_BITS = 16.0  # a per-token logsumexp (or one tile's partial sum of it)


def _map_blocks(cfg: Config, m: Map, layer, S_per_param: float, F: float, k: float, c: float, beta: Beta,
                kn: Knobs, mode: str, train: bool, cheap_in: bool) -> tuple[float, dict]:
    """Blocks per token for one map tiled into R output rows x Q tokens, minimized over R in each of ``layer`` (an int
    or an array of layer indices) and summed over them.

    ``mode``: "exact" prices a tile as locality allows (a residual-fed map reads F per token but needs its output
    gradient per value; a residual-writing map reads its interior input per coordinate and its gradient as G, and its
    outputs are assembled at beta_v per counted value); "linearized" prices training linearized at theta, where every
    map's input is theta's own activation (token ids only), its gradient is theta's backward Jacobian applied to one
    output-error correction G per token, and each tile's tangent reaches the loss as one correction F per token.
    ``cheap_in``: the map's input is theta's own activation (first trained layer after a frozen prefix)."""
    f, L = cfg.f, cfg.layers
    lay = np.atleast_1d(np.asarray(layer, dtype=float))[:, None]
    f_pre = f * lay / L
    f_suf = f * (L - 1 - lay) / L
    G = kn.grad_factor * F
    R = np.unique(np.concatenate([np.geomspace(1, m.d_out, _R_GRID).round(), [m.d_out]]))[None, :]
    a = m.d_in * S_per_param
    a_acc = m.d_in * acc_bits(beta, S_per_param) if train else 0.0
    routed = m.expert
    # theta's baseline for a routed token needs its whole context: n_experts / n_active sequence tokens per routed one
    ctx = (cfg.n_experts / cfg.n_active) if routed else 1.0
    b_tok = cfg.h * ctx + (math.log2(cfg.n_experts / cfg.n_active) if routed else 0.0)
    options = []  # (bits per token as a function of R, recompute FLOPs per token, assembly bits per token per tile)
    if mode == "linearized":
        options.append((b_tok + (G if train else 0.0) + 0 * R, (f + 2 * f_suf) * ctx if train else f_pre * ctx, F + 0 * R))
    else:
        if m.resid_in and m.softmax:
            # softmax - onehot from the tile's own logits and a per-token logsumexp, whose per-tile partial sums the
            # tiles send to the block forming it
            c_in = min(m.d_in * beta.beta_x, F)
            options.append((b_tok + c_in + LSE_BITS + 0 * R, f_pre, LSE_BITS + 0 * R))
        elif m.resid_in:
            c_in = 0.0 if cheap_in else min(m.d_in * beta.beta_x, F)
            c_g = R * beta.g if train else 0 * R
            options.append((b_tok + c_in + c_g, f_pre * ctx, 0 * R))
            if train:  # recompute theta's backward too, and take the output gradient as a correction
                options.append((b_tok + c_in + R * beta.gx, (f + 2 * f_suf) * ctx, 0 * R))
        else:
            c_in = m.d_in * beta.beta_x
            c_g = np.minimum(R * beta.gx, G) if train else 0 * R
            rec = ((f + 2 * f_suf) if train else f_pre) * ctx
            options.append((b_tok + c_in + c_g, rec, 0 * R))
        # without a theta baseline: the raw input, and the raw output gradient per value
        options.append((m.d_in * beta.raw + (R * beta.raw if train else 0 * R), 0.0, 0 * R))
    per_layer = np.full(lay.shape[0], np.inf)
    rows = np.zeros(lay.shape[0])
    toks = np.zeros(lay.shape[0])
    for bits, rec, assemble in options:
        flops = (6 if train else 2) * m.d_in * R + rec
        q = tokens_with_acc(k, c, R * a, R * a_acc, np.broadcast_to(bits, flops.shape), flops, kn,
                            m.active / m.count if routed else 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            bpt = np.where(q >= 1, (m.d_out / R) / q + (m.d_out / R) * assemble / k, np.inf) * m.active
        i = np.argmin(bpt, axis=1)
        got = bpt[np.arange(len(i)), i]
        better = got < per_layer
        per_layer = np.where(better, got, per_layer)
        rows = np.where(better, R[0, i], rows)
        toks = np.where(better, q[np.arange(len(i)), i], toks)
    total = float(np.sum(per_layer))
    if mode == "exact" and m.resid_out and train:
        # the counted outputs, assembled where the next map's F is formed
        total += lay.shape[0] * m.active * min(m.d_in, m.d_out) * beta.gx / k
    return total, {"rows": int(rows[-1]), "tokens": float(toks[-1])}


def tiling(cfg: Config, S: float, F: float, k: float, c: float, beta: Beta, kn: Knobs, workload: str,
           mode: str = "exact", trained: range | None = None, cheap_first: bool = False) -> Attack | None:
    """Rows of each map's update against token tiles (the Loomis-Whitney / k^2 count), one map per block, the state
    spread over the trained parameters at S / params bits each. ``trained`` limits training to some layers (a frozen
    prefix); ``cheap_first`` gives the first trained layer theta's own activations as input."""
    layers = trained if trained is not None else range(cfg.layers)
    s_pp = S / (cfg.params if trained is None else len(layers) * cfg.layer_params)
    head = cfg.head
    total, worst = 0.0, {}
    passes = (True, False) if workload == "grpo" else (True,)
    lay = np.asarray(list(layers))
    for train in passes:
        for m in cfg.maps:
            groups = [(lay[:1], True), (lay[1:], False)] if cheap_first and m.resid_in else [(lay, False)]
            for idx, cheap in groups:
                if len(idx) == 0:
                    continue
                bpt, par = _map_blocks(cfg, m, idx, s_pp, F, k, c, beta, kn, mode, train, cheap)
                if not math.isfinite(bpt):
                    return None
                total += bpt
                worst[m.name] = par
        if trained is None:
            bpt, par = _map_blocks(cfg, head, cfg.layers - 1, s_pp, F, k, c, beta, kn, mode, train, False)
            if not math.isfinite(bpt):
                return None
            total += bpt
    if trained is None:
        u = (4 if workload == "grpo" else 3) * cfg.f
    else:
        f_top = cfg.f_layer * len(layers) + cfg.f_head
        u = cfg.f + 2 * f_top + (cfg.f if workload == "grpo" else 0.0)
    name = {"exact": "tiling", "linearized": "linearized_tiling"}[mode]
    if trained is not None:
        name = "frozen_prefix_tiling"
    return Attack(name, c * total / u, {"last_layer_tiles": worst})


def expert_local(cfg: Config, S: float, F: float, k: float, c: float, beta: Beta, kn: Knobs,
                 workload: str) -> Attack | None:
    """MoE: a block holds one routed expert's whole update and the tokens routed to it; each layer's attention, router
    and shared expert form one stage block (or are tiled, when their share of S doesn't fit). A routed token costs its
    context (theta's baseline) plus F in and G back, and its expert output reaches the next stage as one correction."""
    if not cfg.n_experts or workload != "sft":
        return None
    f, L, A, E = cfg.f, cfg.layers, cfg.n_active, cfg.n_experts
    s_pp = S / cfg.params
    G = kn.grad_factor * F
    experts = [m for m in cfg.maps if m.expert]
    dense_maps = [m for m in cfg.maps if not m.expert]
    e_params = sum(m.d_in * m.d_out for m in experts)
    d_params = sum(m.params for m in dense_maps)
    e_flops = 2 * e_params
    f_dense = cfg.f_layer - A * e_flops
    Fe = kn.expert_out_factor * F
    acc = acc_bits(beta, s_pp)
    total = 0.0
    tiled_dense = False
    for layer in range(L):
        f_suf = f * (L - 1 - layer) / L
        # the dense stage: residual in and gradient out, plus each routed expert's output and input-gradient corrections
        bits = cfg.h + (F if layer else 0) + G + A * (Fe + G)
        q = float(tokens_with_acc(k, c, s_pp * d_params, acc * d_params, bits, 3 * f_dense + f + 2 * f_suf, kn))
        if q >= 1:
            total += 1 / q
        else:
            tiled_dense = True
            for m in dense_maps:
                bpt, _ = _map_blocks(cfg, m, layer, s_pp, F, k, c, beta, kn, "exact", True, False)
                total += bpt
            total += A * Fe / k  # expert outputs, assembled with the tiled residual
        ctx = E / A
        rbits = min(cfg.h * ctx + math.log2(E / A) + F, cfg.d * beta.raw) + G
        q = float(tokens_with_acc(k, c, s_pp * e_params, acc * e_params, rbits, 3 * e_flops + (f + 2 * f_suf) * ctx,
                                  kn, A / E))
        if q < 1:
            return None
        total += A / q
    hp = cfg.vocab * cfg.d
    q = float(tokens_with_acc(k, c, s_pp * hp, acc * hp, cfg.h + F, 3 * cfg.f_head + f, kn))
    if q >= 1:
        total += 1 / q
    else:
        bpt, _ = _map_blocks(cfg, cfg.head, L - 1, s_pp, F, k, c, beta, kn, "exact", True, False)
        total += bpt
    if not math.isfinite(total):
        return None
    return Attack("expert_local", c * total / (3 * f), {"expert_state_bits": s_pp * e_params, "dense_tiled": tiled_dense})


def hybrid(split: Attack | None, whole_small: Attack | None, kn: Knobs) -> Attack | None:
    """A state of S <= k carried whole for the first phi0 of the honest compute, then the split state for the rest."""
    if split is None or whole_small is None:
        return None
    rho = kn.phi0 * whole_small.rho + (1 - kn.phi0) * split.rho
    return Attack("hybrid", rho, {"phi0": kn.phi0, "then": split.strategy})


# -------------------------------------------------------------------------------------------------------------- bound


def bound(cfg: Config, S: float, F: float, k: float, c: float, beta: Beta, kn: Knobs, full: bool,
          workload: str = "sft") -> dict:
    """The bound's rho(S) as the program states it, against the workload's honest FLOPs per trained token ``u`` (3f for
    SFT, 4f for GRPO, whose rollouts are generated): O(1) for S < k, but never below the data floor (every token id, or
    for GRPO every prompt and seed, enters some block: ``c * bits / (k u)``, eta / 3 for SFT); at the cliff
    (2/3) * crossings * F / h with two crossings, times eta = c / (k phi) and 3f / u; for full training the covering
    theorem |T| >= w(V) / k, i.e. ``c * beta_v * V / (k u)`` (eta * phi * beta_v / (6 d_eff) for SFT). ``rho`` is the
    bound with that count at beta_v; ``rho_gx`` the same with the count at beta_gx (they differ only at full training)."""
    phi = cfg.phi(kn.traffic)
    eta = c / (k * phi)
    f = cfg.f
    if workload == "grpo":
        p, g, G = kn.prompt, kn.completion, kn.group
        u = 4 * f
        floor = c * (p * cfg.h + G * kn.seed_bits) / (k * 4 * f * (p + G * g))
    else:
        u = 3 * f
        floor = c * cfg.h / (k * u)
    cliff = eta * (3 * f / u) * (2 / 3) * 2 * F / cfg.h
    full_rho = c * beta.beta_v * cfg.values_per_token / (k * u)
    full_gx = c * beta.gx * cfg.values_per_token / (k * u)
    if S < k:
        rho = rho_gx = max(1.0, floor)
    elif full:
        rho, rho_gx = max(1.0, floor, full_rho), max(1.0, floor, full_gx)
    else:
        rho = rho_gx = max(1.0, floor, cliff)
    return {"rho": rho, "rho_gx": rho_gx, "eta": eta, "cliff": cliff, "full": full_rho, "full_gx": full_gx,
            "floor": floor, "regime": "S<k" if S < k else ("full" if full else "cliff")}


# ------------------------------------------------------------------------------------------------------------ optimizer

#: Layouts of the fixed recipe that respect uniformity, locality and exactness.
EXACT = ("whole_state", "layer_pipeline", "tiling", "expert_local")
#: Layouts that change the recipe (the capability version of the bound) or approximate it.
RECIPE = ("lowrank_pipeline", "linearized_tiling", "frozen_prefix_tiling")
FROZEN_TOP = (1, 4)  # trained top layers tried for the frozen-prefix recipe


def rho_attack(cfg: Config, S: float, F: float, k: float, c: float, beta: Beta, kn: Knobs = Knobs(),
               workload: str = "sft") -> dict:
    """Every strategy's best layout for state S; the cheapest exact one, the cheapest of all, and the hybrid."""
    found: dict[str, Attack | None] = {}
    found["whole_state"] = whole_state(cfg, S, k, c, beta, kn, workload) if S < k else None
    found["layer_pipeline"] = pipeline(cfg, S, F, k, c, beta, kn, workload)
    found["tiling"] = tiling(cfg, S, F, k, c, beta, kn, workload, "exact")
    found["expert_local"] = expert_local(cfg, S, F, k, c, beta, kn, workload)
    found["lowrank_pipeline"] = lowrank_pipeline(cfg, S, F, k, c, beta, kn, workload)
    lin = tiling(cfg, S, F, k, c, beta, kn, workload, "linearized")
    if lin is not None:
        lin.breaks = "C (cap): every input is theta's own activation (cheap inputs)"
    found["linearized_tiling"] = lin
    frozen = [tiling(cfg, S, F, k, c, beta, kn, workload, "exact", range(cfg.layers - m, cfg.layers), True)
              for m in FROZEN_TOP]
    frozen = [(m, a) for m, a in zip(FROZEN_TOP, frozen) if a is not None]
    if frozen:
        m, a = min(frozen, key=lambda ma: ma[1].rho)
        a.params["trained_top_layers"] = m
        a.breaks = "C (cap): the first trained layer reads theta's own activations (cheap inputs)"
        found["frozen_prefix_tiling"] = a
    live = {n: a for n, a in found.items() if a is not None and math.isfinite(a.rho)}
    exact = [a for n, a in live.items() if n in EXACT]
    best_exact = min(exact, key=lambda a: a.rho) if exact else None
    best_any = min(live.values(), key=lambda a: a.rho) if live else None
    hyb = None
    if S >= k and best_exact is not None:
        hyb = hybrid(best_exact, whole_state(cfg, k / 4, k, c, beta, kn, workload), kn)
    return {"best": best_any, "best_exact": best_exact, "all": live, "hybrid": hyb}


BELOW = 1 - 1e-6  # an attack counts as below the bound under this ratio, past rounding


def _breaks(best: Attack, b: dict, full: bool) -> str | None:
    if best.rho >= BELOW * b["rho"]:
        return None
    if best.breaks:
        return best.breaks
    if best.strategy in ("tiling", "expert_local"):
        if best.rho < BELOW * b["rho"] * b["full_gx"] / b["full"] and full:
            return "C (cap): a tile's inputs (F per token, or cheap) carry fewer fresh bits than its counted outputs"
        return ("D (decoding): counted values reach blocks that recompute theta's at beta_gx; the bound charges "
                "beta_v" if full else "C (cap): a tile's inputs carry fewer fresh bits than its counted outputs")
    if best.strategy == "layer_pipeline":
        if full:
            return ("C (cap, subnetwork clause): a stage's update fits in k and its counted outputs carry more fresh bits "
                    "per token than its F + G inputs")
        return "none (formula slack: the cliff formula's crossings and state share against the pipeline's optimum)"
    if best.strategy == "whole_state":
        return "none (S < k)"
    return "unclassified"


def evaluate(cfg: Config, inputs: Inputs, kn: Knobs, labels: list[str], k: float, c: float, precision: str,
             workload: str) -> list[dict]:
    rows = []
    for lab in labels:
        beta = inputs.beta_for(lab, precision)
        S = resolve_S(lab, cfg, beta, kn, inputs.S_bits)
        F = inputs.F_for(lab, precision, cfg)
        full = lab.lower() == "full"
        kl = replace(kn, grad_factor=inputs.grad_factor(lab, precision, kn))
        res = rho_attack(cfg, S, F, k, c, beta, kl, workload)
        b = bound(cfg, S, F, k, c, beta, kl, full, workload)
        best, ex, hyb = res["best"], res["best_exact"], res["hybrid"]
        rows.append({
            "config": cfg.name, "S_label": lab, "S_bits": S, "F": F, "G": F * kl.grad_factor, "beta": asdict(beta),
            "k_bits": k, "c": c, "precision": precision,
            "workload": workload, "bound": b,
            "attack": best.to_json() if best else None,
            "attack_exact": ex.to_json() if ex else None,
            "strategies": {n: a.to_json() for n, a in res["all"].items()},
            "ratio": (best.rho / b["rho"]) if best else None,
            "ratio_exact": (ex.rho / b["rho"]) if ex else None,
            "ratio_gx": (best.rho / b["rho_gx"]) if best else None,
            "breaks": _breaks(best, b, full) if best else None,
            "breaks_exact": _breaks(ex, b, full) if ex else None,
            "hybrid": hyb.to_json() | {"bound_same_phases": kn.phi0 + (1 - kn.phi0) * b["rho"]} if hyb else None,
        })
    return rows


_INTERP = ("beta_v", "beta_x", "beta_w", "beta_gx")


def F_curve(inputs: Inputs, cfg: Config, precision: str, kn: Knobs) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """F, G / F and the betas against S on cfg at the labeled points, sorted by S, for interpolation in log S
    (constant past the ends)."""
    pts = []
    for lab in inputs.F:
        b = inputs.beta_for(lab, precision)
        pts.append((resolve_S(lab, cfg, b, kn, inputs.S_bits),
                    {"F": inputs.F_for(lab, precision, cfg), "gf": inputs.grad_factor(lab, precision, kn),
                     **{f: getattr(b, f) if f != "beta_gx" else b.gx for f in _INTERP}}))
    pts.sort(key=lambda p: p[0])
    return np.array([p[0] for p in pts]), {key: np.array([p[1][key] for p in pts]) for key in pts[0][1]}


def curve(cfg: Config, inputs: Inputs, kn: Knobs, k: float, c: float, precision: str, workload: str,
          n: int = 36, s_min: float = 1e2) -> dict:
    """rho_attack and the bound's rho(S) at n sizes up to full training, F, G / F and the betas interpolated in log S
    between the labels; ``cliff_2k`` the same at S = 2k, just above the cliff."""
    base = inputs.beta_for("full", precision) if "full" in inputs.F else inputs.precisions[precision]
    s_full = cfg.params * base.beta_w
    sx, ys = F_curve(inputs, cfg, precision, kn)
    at = lambda key, S: float(np.interp(np.log(S), np.log(sx), ys[key]))  # noqa: E731

    def point(S: float) -> tuple[dict, dict]:
        beta = replace(base, **{f: at(f, S) for f in _INTERP})
        kl = replace(kn, grad_factor=at("gf", S))
        return rho_attack(cfg, S, at("F", S), k, c, beta, kl, workload), bound(cfg, S, at("F", S), k, c, beta, kl,
                                                                               False, workload)

    out = {"S": [], "attack": [], "strategy": [], "attack_exact": [], "strategy_exact": [], "bound": [], "cliff": [],
           "full": None}
    for S in np.geomspace(s_min, s_full, n):
        res, b = point(float(S))
        out["S"].append(float(S))
        for key, a in (("", res["best"]), ("_exact", res["best_exact"])):
            out["attack" + key].append(a.rho if a else None)
            out["strategy" + key].append(a.strategy if a else None)
        out["bound"].append(b["rho"])
        out["cliff"].append(b["cliff"])
    res, b = point(2 * k)
    out["cliff_2k"] = {"S": 2 * k, "bound": b["rho"],
                       **{"attack" + key: (a.rho if a else None) for key, a in (("", res["best"]), ("_exact", res["best_exact"]))},
                       **{"strategy" + key: (a.strategy if a else None)
                          for key, a in (("", res["best"]), ("_exact", res["best_exact"]))}}
    full = bound(cfg, s_full, float(ys["F"][-1]), k, c, base, kn, True, workload)
    out["full"], out["full_gx"] = full["rho"], full["rho_gx"]
    return out


# ------------------------------------------------------------------------------------------------------------------ CLI


def c_modes(cfg: Config, k: float, kn: Knobs) -> dict[str, float]:
    return {"eta1": k * cfg.phi(kn.traffic), "c1.4e17": C_FIXED}


def plot(curves: list[dict], path: Path, precision: str) -> None:
    """One panel per config (rows) and c mode (columns), for one precision and the first workload: the exact attack
    (thick), the attack with recipe changes (dashed), the bound's rho(S) (thin), the full-training bound (star) and k
    (dotted)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wl = curves[0].get("workload") if curves else None
    curves = [cv for cv in curves if cv["precision"] == precision and cv.get("workload") == wl]
    cfgs = list(dict.fromkeys(cv["config"] for cv in curves))
    cmodes = sorted({cv["c_mode"] for cv in curves}, reverse=True)
    fig, axes = plt.subplots(len(cfgs), len(cmodes), figsize=(6.5 * len(cmodes), 3.6 * len(cfgs)), squeeze=False)
    colors = {"10kB": "tab:blue", "100kB": "tab:orange", "1MB": "tab:green"}
    nan = lambda xs: np.array([x if x is not None else np.nan for x in xs], dtype=float)  # noqa: E731
    for i, name in enumerate(cfgs):
        for j, cm in enumerate(cmodes):
            ax = axes[i][j]
            for cv in curves:
                if cv["config"] != name or cv["c_mode"] != cm:
                    continue
                col = colors.get(cv["k_label"], "k")
                S = np.array(cv["S"])
                ax.plot(S, nan(cv["attack_exact"]), "-", color=col, lw=1.8, label=f"exact attack, k={cv['k_label']}")
                ax.plot(S, nan(cv["attack"]), "--", color=col, lw=1.0, label=f"with recipe changes, k={cv['k_label']}")
                ax.plot(S, cv["bound"], "-", color=col, lw=0.7, alpha=0.6)
                ax.scatter([S[-1]], [cv["full"]], marker="*", color=col, s=50, zorder=5)
                ax.axvline(cv["k"], color=col, lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{name}, c {cm}, {precision}, {wl or 'sft'}", fontsize=9)
            ax.set_xlabel("S (bits of unregistered state)")
            ax.set_ylabel("rho")
            ax.grid(alpha=0.3, which="both", lw=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, ncol=1)
    fig.suptitle("rho_attack against the bound's rho(S) (thin); star: the full-training bound; dotted: k", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt(x: float | None) -> str:
    return "-" if x is None else (f"{x:.3g}" if x < 1e4 else f"{x:.2e}")


def summarize(rows: list[dict]) -> str:
    lines = ["| config | c | k | precision | workload | S | bound (beta_v count) (regime) | bound (beta_gx count) | "
             "exact attack | exact / bound (beta_v count) | any attack | any / bound (beta_v count) | "
             "any / bound (beta_gx count) | breaks |", "|" + "---|" * 14]
    for r in rows:
        a, e = r["attack"], r["attack_exact"]
        lines.append(f"| {r['config']} | {r['c_mode']} | {r['k_label']} | {r['precision']} | {r['workload']} | "
                     f"{r['S_label']} ({fmt(r['S_bits'])} b) | {fmt(r['bound']['rho'])} ({r['bound']['regime']}) | "
                     f"{fmt(r['bound']['rho_gx'])} | "
                     f"{fmt(e['rho']) + ' ' + e['strategy'] if e else 'infeasible'} | {fmt(r['ratio_exact'])} | "
                     f"{fmt(a['rho']) + ' ' + a['strategy'] if a else 'infeasible'} | {fmt(r['ratio'])} | "
                     f"{fmt(r['ratio_gx'])} | {r['breaks'] or ''} |")
    return "\n".join(lines) + "\n"


def merge(paths: list[str], out: Path) -> None:
    """One table and one figure per precision from the JSON of several runs of one input file (one per config, say)."""
    docs = [json.loads(Path(p).read_text()) for p in paths]
    tag = Path(docs[0]["inputs"]).stem
    rows = [r for d in docs for r in d["rows"]]
    curves = [cv for d in docs for cv in d["curves"]]
    doc = {**docs[0], "rows": rows, "curves": curves, "merged_from": paths}
    (out / f"rho_attack_{tag}.json").write_text(json.dumps(doc, indent=1, default=str))
    (out / f"rho_attack_{tag}.md").write_text(summarize(rows))
    for prec in dict.fromkeys(cv["precision"] for cv in curves):
        plot(curves, out / f"rho_attack_{tag}_{prec}.png", prec)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", nargs="+", help="input JSON files (beta per precision, F per S label)")
    ap.add_argument("--merge", nargs="+", help="instead of a search: merge these output JSONs of one input file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="*", default=list(configs.CONFIGS))
    ap.add_argument("--workloads", nargs="*", default=["sft", "grpo"])
    ap.add_argument("--curve-points", type=int, default=36)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--knobs", default="{}", help="JSON overrides of Knobs fields")
    args = ap.parse_args(argv)
    kn = Knobs(**json.loads(args.knobs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args.merge, out)
        return 0
    if not args.inputs:
        ap.error("give --inputs (or --merge)")
    for path in args.inputs:
        inputs = Inputs.load(path)
        tag = Path(path).stem
        rows, curves = [], []
        for name in args.configs:
            cfg = configs.get(name)
            labels = list(inputs.F)
            for k_label, k in K_GRID.items():
                for c_mode, c in c_modes(cfg, k, kn).items():
                    for prec in inputs.precisions:
                        for wl in args.workloads:
                            for r in evaluate(cfg, inputs, kn, labels, k, c, prec, wl):
                                r.update(c_mode=c_mode, k_label=k_label)
                                rows.append(r)
                            cv = curve(cfg, inputs, kn, k, c, prec, wl, args.curve_points)
                            cv.update(config=name, c_mode=c_mode, k_label=k_label, k=k, precision=prec, workload=wl)
                            curves.append(cv)
        doc = {"inputs": path, "source": inputs.source, "knobs": asdict(kn), "rows": rows, "curves": curves}
        (out / f"rho_attack_{tag}.json").write_text(json.dumps(doc, indent=1, default=str))
        (out / f"rho_attack_{tag}.md").write_text(summarize(rows))
        if not args.no_plots:
            for prec in inputs.precisions:
                plot(curves, out / f"rho_attack_{tag}_{prec}.png", prec)
        n_below = sum(1 for r in rows if r["ratio"] is not None and r["ratio"] < BELOW)
        print(f"{tag}: {len(rows)} rows, attack below the bound in {n_below}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
