"""Covert workloads under the policy, each a model: from a workload's dials to ρ's inputs, and ρ by te_core.formalism.at
at a policy (c, k), for the console's *Standard training under the policy* plot (TRN-41).

    python -m te_core.workloads [--json OUT]    # every workload's reference points at the three policies

A workload (`WORKLOADS`) is a function from its dials (`Dial`: name, unit, default, range, why) to a `Model`: ρ's inputs
as a te_core.formalism `Measured` (|C|, w, σ, τ, a, the profiles u and u_in, Δ, d, Δ_grad and Δ_t, each labelled with
where it comes from) and w_fwd. `columns` gives them as the console's columns, te_core.algorithm_policy's `INPUTS` with
`w_fit_gates`, `alpha_ratio` and `C_gates`, on which `algorithm_policy.page_rho` gives `formalism.at`'s ρ. `reference`
evaluates each workload's reference points (`Workload.points`) at the three `POLICIES`, and its `Check`s, pilot records
beside the model; `workloads.json` is its output, which the console's tests pin. The spec, each mapping with its
reasons, is note:training-20261103T0600Z-draft-workload-models.

The conventions every workload shares:

- **Policies** (`session`): a block is one served session of the console's reference model, the 45M dense start: c is
  the session's tokens at N MACs a token (2N FLOPs, attention left out, as the console counts it) at the bf16 GEMM
  price, and k its input ids at ⌈log2 V⌉ bits. `default` is 4,096 tokens in and 1,024 out, `tighter` 2,048 in and 3,072
  out (k halved at the same c) and `looser` 16,384 in and 1,024 out. Serving the registered model fits a block by
  this count (Δ = 0, s·τ = k); with its attention at te_core.flops' count the default session is 1.57 c.
- **Shapes** (`dense`): a dense decoder of N parameters with the reference model's d/L and SwiGLU width, d solved from N.
- **Work**: a token's MACs at te_core.gates' prices, its weight GEMMs at the workload's precision and its attention in
  bf16 at te_core.flops' count (the whole span, so a training token at context T has 2·L·d·T), with `ELEMENTWISE` bf16
  elementwise MACs per MAC. a is the bf16 residual stream, 16·d.
- **A1's reading** (the dial `t`): u is 0 up to the free share t and x·u(1) beyond it, u_in x·u(1) (`profile`:
  `formalism.with_tail` of `formalism.even`); t = 0 is the evenly spread reading. Dense training's default t is
  `T_DENSE`, te_core's literal t measured on the 45M pilot's trace; elsewhere t is the modelled share of a token's
  affected work that reads no update value (`Token.t`).
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import VERSION, algorithm_policy, formalism, policy
from .gates import per_mac
from .opnames import FUSED_OPTIMIZER_MACS

NOTE = "note:training-20261103T0600Z-draft-workload-models"
REFERENCE = Path(__file__).with_name("workloads.json")
VOCAB = 50_304
#: the console's served model: the 45M dense starts' parameters (starts_store.json's s50 and s100)
REFERENCE_PARAMS = 45_029_888
TAU_BITS = policy.id_bits(VOCAB)
ACT_BITS = policy.ACT_BITS
#: the reference model's d_model / n_layer (512 / 6) and SwiGLU hidden width over d_model (1,408 / 512)
ASPECT, MLP = 512 / 6, 1408 / 512
#: bf16 elementwise MACs per GEMM or attention MAC: 0.01 puts a 45M training token at 3.08 forwards, 2.3396e11 gates,
#: within 1% of te_core's counts on the pilot's full-training records (2.319e11 to 2.341e11)
ELEMENTWISE = 0.01
#: a weight as a GEMM of each precision reads it: NVFP4 is 4 bits and an e4m3 scale per 16 values
WEIGHT_BITS = {"bf16": 16.0, "e4m3": 8.0, "fp4": 4.5}
#: te_core 4.1's literal t on the 45M pilot's traced full-training step (every ref-optim record's): the share of a
#: training token's affected work in ops that read no update value
T_DENSE = 0.692646
NEWTON_SCHULZ_STEPS = 5
#: optimizer state bits per trained value (Muon's is its momentum on matrices, AdamW's on everything else)
STATE_BITS = {"adamw": 64, "sgd": 0, "muon": 32}
#: (d_in + d_out) per rank per layer of each LoRA target set, in d_models
TARGETS = {"q": 2.0, "qv": 4.0, "attn": 8.0, "mlp": 3 * (1 + MLP), "all": 8.0 + 3 * (1 + MLP)}


def session(inputs: int, outputs: int, params: float = REFERENCE_PARAMS) -> tuple[float, float]:
    """(c, k) of one served session a block: its tokens at `params` MACs each at the bf16 GEMM price, and its input ids."""
    return (inputs + outputs) * params * per_mac("gemm/bf16"), float(inputs * TAU_BITS)


SESSIONS = {"default": (4096, 1024), "tighter": (2048, 3072), "looser": (16384, 1024)}
POLICIES = {name: session(*s) for name, s in SESSIONS.items()}


# ---- shapes and work ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    params: float
    d_model: float
    n_layer: float

    @property
    def layer_params(self) -> float:
        """One layer's matrices: attention's four, d², and the SwiGLU MLP's three, MLP·d²."""
        return (4 + 3 * MLP) * self.d_model ** 2

    @property
    def matrix_params(self) -> float:
        return self.n_layer * self.layer_params

    def attention(self, span: float) -> float:
        """QK and PV MACs of one token attending over `span` positions, every layer."""
        return 2 * self.n_layer * self.d_model * span


def dense(params: float, aspect: float = ASPECT) -> Shape:
    """The dense decoder of `params` parameters at d/L = `aspect`: N = L·((4 + 3·MLP)·d² + 2d) + (V + 1)·d (each layer's
    matrices and two norms, a tied embedding and the final norm), with L = d/aspect a cubic in d, solved by Cardano's
    formula (one real root). 45,029,888 gives the reference model, d = 512 and L = 6."""
    a, b, c = (4 + 3 * MLP) / aspect, 2 / aspect, VOCAB + 1.0
    p = (3 * a * c - b * b) / (3 * a * a)
    q = (2 * b ** 3 - 9 * a * b * c - 27 * a * a * params) / (27 * a ** 3)
    r = math.cbrt(-q / 2 + math.sqrt(q * q / 4 + p ** 3 / 27))
    d = r - p / (3 * r) - b / (3 * a)
    return Shape(float(params), d, d / aspect)


def _add(*macs: dict) -> dict:
    out: dict[str, float] = {}
    for m in macs:
        for kind, n in m.items():
            out[kind] = out.get(kind, 0.0) + n
    return out


def priced(macs: dict, elementwise: bool = True) -> float:
    """Gates of MACs by kind (te_core.gates' kinds), with `ELEMENTWISE` bf16 elementwise MACs per MAC if `elementwise`."""
    extra = ELEMENTWISE * sum(macs.values()) * per_mac("elementwise/bf16") if elementwise else 0.0
    return sum(n * per_mac(kind) for kind, n in macs.items()) + extra


@dataclass(frozen=True)
class Token:
    """One kind of token block: its MACs by kind, those of its affected work (the work that depends on θ'), and those
    of its affected work that reads update values."""
    macs: dict
    affected: dict
    reading: dict

    @property
    def w(self) -> float:
        return priced(self.macs)

    @property
    def alpha(self) -> float:
        return priced(self.affected)

    @property
    def t(self) -> float:
        """The share of its affected work in ops that read no update value (te_core's literal t, by this count)."""
        return 1 - priced(self.reading, elementwise=False) / self.alpha


def training(sh: Shape, precision: str, span: float) -> Token:
    """A dense training token: every weight's forward, input gradient and weight gradient at `precision`, attention's
    forward and its backward (twice it) in bf16; all affected, and the forward and input gradient read the weights."""
    gemm = f"gemm/{precision}"
    macs = _add({gemm: 3 * sh.params}, {"gemm/bf16": 3 * sh.attention(span)})
    return Token(macs, macs, {gemm: 2 * sh.params})


def forward(sh: Shape, precision: str, span: float) -> Token:
    """An inference token: every weight's forward at `precision` and attention over `span` in bf16, all affected."""
    gemm = f"gemm/{precision}"
    macs = _add({gemm: sh.params}, {"gemm/bf16": sh.attention(span)})
    return Token(macs, macs, {gemm: sh.params})


def w_fwd(sh: Shape, precision: str, span: float) -> float:
    """A token's forward in gates, with no elementwise share (te_core.policy's w_fwd's form)."""
    return sh.params * per_mac(f"gemm/{precision}") + sh.attention(span) * per_mac("gemm/bf16")


def decode_span(prompt: float, generated: float) -> float:
    """The mean span a token attends over in one served sequence, as te_core.flops counts it: the prompt's prefill at
    the whole prompt, then each generated token over the prompt and the tokens before it."""
    return (prompt * prompt + generated * prompt + generated * generated / 2) / (prompt + generated)


def optimizer_gates(sh: Shape, optimizer: str) -> float:
    """One step's optimizer gates: AdamW (and SGD) at te_core's fused-op count, 8 (2) f32 MACs a parameter; Muon's
    Newton-Schulz in bf16 on every matrix (per iteration 2a²b + a³ MACs on an a x b matrix, a <= b: 3d³ for attention's,
    (2·MLP + 1)·d³ for the MLP's) and AdamW on the rest."""
    f32 = per_mac("elementwise/f32")
    if optimizer != "muon":
        return sh.params * FUSED_OPTIMIZER_MACS[f"_fused_{optimizer}_"] * f32
    ns = NEWTON_SCHULZ_STEPS * sh.n_layer * sh.d_model ** 3 * (4 * 3 + 3 * (2 * MLP + 1))
    return ns * per_mac("gemm/bf16") + (sh.params - sh.matrix_params) * FUSED_OPTIMIZER_MACS["_fused_adamw_"] * f32


def delta_grad(sh: Shape, optimizer: str) -> float:
    """Δ_grad of full training: the gradient at 32 bits a value and the optimizer's state (Muon's momentum on the
    matrices, AdamW's two moments on the rest). The pilot's traces measure exactly these: 96·N for AdamW
    (ref-optim/adamw-rewarm) and 3,706,306,560 for Muon (ref-optim/cooldown-forks), at the 45M start."""
    if optimizer == "muon":
        return 32 * sh.params + 32 * sh.matrix_params + 64 * (sh.params - sh.matrix_params)
    return (32 + STATE_BITS[optimizer]) * sh.params


@dataclass(frozen=True)
class Adapter:
    """A LoRA on `targets` (`TARGETS`) of the top `layers` layers at `rank`: its trained values, and its MACs a token
    (one pass: x·A and (x·A)·B per adapted matrix)."""
    values: float
    layers: float

    @classmethod
    def of(cls, sh: Shape, rank: float, targets: str, layers: float | None) -> Adapter:
        n = sh.n_layer if layers is None else min(float(layers), sh.n_layer)
        return cls(rank * TARGETS[targets] * sh.d_model * n, n)

    @property
    def macs(self) -> float:
        return self.values


def lora_training(sh: Shape, ad: Adapter, span: float) -> Token:
    """A LoRA training token in bf16: every layer's forward; the input gradient through the head and the adapted layers
    (the frozen weights get no weight gradient); the adapter's forward, input gradient and weight gradient. The layers
    below the lowest adapted one are unaffected; the adapter's forward and input gradient read its values."""
    head, frac = VOCAB * sh.d_model, ad.layers / sh.n_layer
    fwd = {"gemm/bf16": sh.params + sh.attention(span) + ad.macs}
    bwd = {"gemm/bf16": head + ad.layers * sh.layer_params + 2 * frac * sh.attention(span) + 2 * ad.macs}
    below = {"gemm/bf16": (sh.n_layer - ad.layers) * sh.layer_params + (1 - frac) * sh.attention(span)}
    macs = _add(fwd, bwd)
    return Token(macs, _add(macs, {k: -v for k, v in below.items()}), {"gemm/bf16": 2 * ad.macs})


def lora_forward(sh: Shape, ad: Adapter, precision: str, span: float) -> Token:
    """An inference token through a LoRA kept apart from its base: the base at `precision`, attention and the adapter in
    bf16; the layers below the lowest adapted one unaffected, and the adapter's ops read its values."""
    frac = ad.layers / sh.n_layer
    macs = _add({f"gemm/{precision}": sh.params}, {"gemm/bf16": sh.attention(span) + ad.macs})
    below = _add({f"gemm/{precision}": (sh.n_layer - ad.layers) * sh.layer_params},
                 {"gemm/bf16": (1 - frac) * sh.attention(span)})
    return Token(macs, _add(macs, {k: -v for k, v in below.items()}), {"gemm/bf16": ad.macs})


# ---- the model a workload gives --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    """A workload at its dials: ρ's inputs (`measured`), the trained or served model's forward per token (`w_fwd`), and
    the quantities they came from (`derived`)."""
    workload: str
    dials: dict
    measured: formalism.Measured
    w_fwd: float
    derived: dict = field(default_factory=dict)


def profile(u1: float, t: float) -> tuple[tuple, tuple]:
    """(u, u_in) at the free share t (A1 and A1′, module docstring): u_in evenly spread, u 0 up to t."""
    u_in = formalism.even(u1)
    return formalism.with_tail(u_in, t), u_in


def _measured(parts, tau: float, a: float, u1: float, t: float, delta: float, labels: dict,
              decoder: float = 0.0, delta_t: float | None = None) -> formalism.Measured:
    u, u_in = profile(u1, t)
    return formalism.Measured(tuple(parts), tau, a, u, u_in, delta, decoder, labels, t if 0 < t < 1 else 0.0, None,
                              delta_t)


def _labels(**extra) -> dict:
    """The labels every model has (te_core.formalism's kinds: "model: ..." is an estimate), and `extra`'s."""
    return {"gates": "model: the tokens' work and the optimizer's (te_core.workloads)",
            "tokens": "model: the workload's tokens (a dial)",
            "token_gates": "model: each token's MACs at te_core.gates' prices (te_core.workloads.Token)",
            "tau_bits": "model: the input ids a token block reads, ceil(log2 V) bits each",
            "a_bits": "upper bound (witness): d_model x 16, the bf16 residual stream (measured on both pilot starts)",
            "u": "assumption (A1): 0 up to the free share t, x u(1) beyond it",
            "u_in": "assumption (A1'): evenly spread, x u(1)",
            "decoder_gates": "model: none, Delta is theta' itself"} | extra


def _dense_t(t: float | None) -> float:
    return T_DENSE if t is None else float(t)


def standard(n_params: float = REFERENCE_PARAMS, precision: str = "bf16", seq_len: float = 1024,
             batch_tokens: float = 65_536, optimizer: str = "adamw", t: float | None = None,
             tokens_per_param: float = 20.0) -> Model:
    """Standard dense pretraining: every weight trained, each token's forward and backward in token blocks, the
    optimizer in none. u(1) is every weight as the GEMMs read it, Δ θ' at bf16 (fp32 latents for Δ_t where the GEMMs
    are narrower), no decoder."""
    sh = dense(n_params)
    tok = training(sh, precision, seq_len)
    tokens = tokens_per_param * n_params
    steps = tokens / batch_tokens
    opt = optimizer_gates(sh, optimizer)
    tf = _dense_t(t)
    part = formalism.Part(tokens * tok.w + steps * opt, tokens, tokens * tok.w, tokens * tok.alpha, max(1, round(steps)),
                          delta_grad(sh, optimizer))
    latents = precision != "bf16"
    labels = _labels(u1_bits=f"model: every weight as its GEMMs read it, N x {WEIGHT_BITS[precision]:g} bits",
                     delta_bits="model: theta' at bf16, 16 N",
                     delta_grad_bits=f"model: the gradient at 32 bits a value and {optimizer}'s state")
    if latents:
        labels["delta_t_bits"] = "model: fp32 latents, 32 N (a straight-through run)"
    m = _measured([part], TAU_BITS, ACT_BITS * sh.d_model, n_params * WEIGHT_BITS[precision], tf, 16 * n_params, labels,
                  delta_t=32 * n_params if latents else None)
    return Model("standard", _dials(standard, locals()), m, w_fwd(sh, precision, seq_len),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "w_gates": tok.w, "t": tf, "t_counted": tok.t,
                  "optimizer_gates_per_step": opt})


def diloco(n_params: float = REFERENCE_PARAMS, precision: str = "bf16", seq_len: float = 1024,
           batch_tokens: float = 65_536, optimizer: str = "adamw", t: float | None = None,
           inner_steps: float = 500, syncs: float = 28, sync_bits: float | None = None) -> Model:
    """DiLoCo-style pretraining, the record's prefix S syncs: each a phase of H inner steps of B tokens (all workers'),
    standard training on each worker, then an outer step from the workers' pseudo-gradients, each `sync_bits` b once
    compressed (None: 32 N, uncompressed). Its token blocks are standard training's, and Slowdown reads no Δ_grad, so
    its bound is standard training's. Only where the syncs and a phase's tokens describe every step's weights in fewer
    bits than 16 N does Δ change: S·b + H·B·τ, decoded by replaying the phase from the synced weights (d = H·B·w), which
    needs the inner optimizer to restart at each sync. That decoder counts only where it fits a block
    (te_core.formalism.at's d < c), and then a phase's inner gradients never leave its block, which passes on only its
    compressed delta (Δ_grad = b, an outer step without momentum: Nesterov's would carry 32 N)."""
    sh = dense(n_params)
    tok = training(sh, precision, seq_len)
    phase = inner_steps * batch_tokens
    tokens = syncs * phase
    steps = syncs * inner_steps
    opt = optimizer_gates(sh, optimizer)
    b = 32 * n_params if sync_bits is None else float(sync_bits)
    code = syncs * b + phase * TAU_BITS
    compressed = code < 16 * n_params
    dg = b if compressed else delta_grad(sh, optimizer)
    part = formalism.Part(tokens * tok.w + steps * opt, tokens, tokens * tok.w, tokens * tok.alpha, max(1, round(steps)),
                          dg)
    latents = precision != "bf16" and not compressed
    labels = _labels(u1_bits=f"model: every weight as its GEMMs read it, N x {WEIGHT_BITS[precision]:g} bits",
                     delta_bits=("model: S syncs of b bits and the phase's token ids, S b + H B tau" if compressed
                                 else "model: theta' at bf16, 16 N (shorter than the syncs)"),
                     delta_grad_bits=("model: the phase's compressed delta, b (its inner gradients stay in its block "
                                      "wherever the decoder fits one)" if compressed else
                                      f"model: the gradient at 32 bits a value and {optimizer}'s state"))
    if compressed:
        labels["decoder_gates"] = "model: replaying the phase from the synced weights, H B w"
    if latents:
        labels["delta_t_bits"] = "model: fp32 latents, 32 N (a straight-through run)"
    m = _measured([part], TAU_BITS, ACT_BITS * sh.d_model, n_params * WEIGHT_BITS[precision], _dense_t(t),
                  code if compressed else 16 * n_params, labels, decoder=phase * tok.w if compressed else 0.0,
                  delta_t=32 * n_params if latents else None)
    return Model("diloco", _dials(diloco, locals()), m, w_fwd(sh, precision, seq_len),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "w_gates": tok.w, "tokens": tokens,
                  "code_bits": code, "compressed": compressed, "phase_tokens": phase, "phase_gates": phase * tok.w})


def rl(n_params: float = REFERENCE_PARAMS, update: str = "full", rank: float = 16, targets: str = "all",
       layers: float | None = None, rollout_precision: str = "fp4", prompt_tokens: float = 256,
       generated_tokens: float = 768, samples_per_trained: float = 1.0, optimizer: str = "adamw",
       batch_tokens: float = 65_536, t: float | None = None, tokens: float = 1e8) -> Model:
    """RL on the policy being trained: rollouts (prefill and generation) of the current policy at `rollout_precision`,
    `samples_per_trained` of them for each sequence a learner trains on in bf16 (every position of each, prompt and
    generated). Every token is affected (the generated ids are sampled from θ'), and only the prompts' ids are inputs.
    `update` "full" trains every weight (u(1) the least of the two kinds', the rollouts' narrower weights); "lora" a
    LoRA kept apart from the base (u(1) = Δ, its values). One profile serves both kinds, at the larger free share and
    the smaller u(1), as te_core's single A1 needs; `sensitivity` gives the bounds that count each kind's tokens
    apart."""
    sh = dense(n_params)
    seq = prompt_tokens + generated_tokens
    span = decode_span(prompt_tokens, generated_tokens)
    if update == "full":
        roll, learn = forward(sh, rollout_precision, span), training(sh, "bf16", seq)
        u1s = (n_params * WEIGHT_BITS[rollout_precision], 16 * n_params)
        ts = (roll.t, T_DENSE)
        delta, dg, opt = 16 * n_params, delta_grad(sh, optimizer), optimizer_gates(sh, optimizer)
    else:
        ad = Adapter.of(sh, rank, targets, layers)
        roll, learn = lora_forward(sh, ad, rollout_precision, span), lora_training(sh, ad, seq)
        delta = 16 * ad.values
        u1s = (delta, delta)
        ts = (roll.t, learn.t)
        state = "sgd" if optimizer == "sgd" else "adamw"
        dg = (32 + STATE_BITS[state]) * ad.values
        opt = ad.values * FUSED_OPTIMIZER_MACS[f"_fused_{state}_"] * per_mac("elementwise/f32")
    if t is not None:
        ts = (float(t), float(t))
    tf = max(ts)
    n_l = tokens
    n_r = samples_per_trained * n_l
    steps = n_l / batch_tokens
    parts = [formalism.Part(n_r * roll.w, n_r, n_r * roll.w, n_r * roll.alpha, 1, 0.0),
             formalism.Part(n_l * learn.w + steps * opt, n_l, n_l * learn.w, n_l * learn.alpha, max(1, round(steps)), dg)]
    tau = TAU_BITS * prompt_tokens / seq
    labels = _labels(tau_bits="model: the prompts' ids, ceil(log2 V) bits each over the sequence's tokens (generated ids "
                              "are computed)",
                     u1_bits=("model: the least of the two kinds' weights as their GEMMs read them" if update == "full"
                              else "model: the adapter's values at 16 bits, Delta"),
                     delta_bits="model: theta' at bf16, 16 N" if update == "full" else "model: the adapter at 16 bits",
                     delta_grad_bits="model: the gradient at 32 bits a value and the optimizer's state")
    m = _measured(parts, tau, ACT_BITS * sh.d_model, min(u1s), tf, delta, labels)
    return Model("rl", _dials(rl, locals()), m, w_fwd(sh, rollout_precision, span),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "rollout_w": roll.w, "learner_w": learn.w,
                  "rollout_tokens": n_r, "learner_tokens": n_l,
                  "rollout_share": n_r * roll.w / (n_r * roll.w + n_l * learn.w), "t": tf,
                  "kinds": [{"kind": kind, "tokens": n, "u1_bits": u1, "t": tk}
                            for kind, n, u1, tk in zip(("rollout", "learner"), (n_r, n_l), u1s, ts)]})


def inference(n_params: float = REFERENCE_PARAMS, precision: str = "fp4", prompt_tokens: float = 4096,
              generated_tokens: float = 1024, t: float | None = None, tokens: float = 1e9) -> Model:
    """Serving an unregistered model: its weights are no part of θ, so every one is update bits, as if θ' (Δ = its
    weights, d = 0, no training: Δ_grad = 0, so Fit is running θ', Δ + s·τ <= k). σ = 1; t is the share of a token's
    work in ops that read no weight, attention and the elementwise ops."""
    sh = dense(n_params)
    span = decode_span(prompt_tokens, generated_tokens)
    tok = forward(sh, precision, span)
    tf = tok.t if t is None else float(t)
    u1 = n_params * WEIGHT_BITS[precision]
    part = formalism.Part(tokens * tok.w, tokens, tokens * tok.w, tokens * tok.alpha, 1, 0.0)
    labels = _labels(tau_bits="model: the prompt's ids, ceil(log2 V) bits each over the session's tokens",
                     u1_bits=f"model: every weight as its GEMMs read it, N x {WEIGHT_BITS[precision]:g} bits",
                     delta_bits=f"model: the served weights, N x {WEIGHT_BITS[precision]:g} bits",
                     delta_grad_bits="model: none (no training)")
    m = _measured([part], TAU_BITS * prompt_tokens / (prompt_tokens + generated_tokens), ACT_BITS * sh.d_model, u1, tf,
                  u1, labels)
    return Model("inference", _dials(inference, locals()), m, w_fwd(sh, precision, span),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "span": span, "t_counted": tok.t})


def lora(n_params: float = REFERENCE_PARAMS, rank: float = 8, targets: str = "all", layers: float | None = None,
         adapter_bits: float = 16, optimizer: str = "adamw", seq_len: float = 1024, batch_tokens: float = 65_536,
         t: float | None = None, tokens: float = 1e8) -> Model:
    """Fine-tuning a registered model with a LoRA on `targets` of its top `layers` layers (None: all): Δ the adapter's
    values at `adapter_bits` (fp32 latents for Δ_t below 16), Δ_grad their gradient at 32 bits and the optimizer's
    state, u(1) = Δ, and t the share of the affected work that reads no adapter value."""
    sh = dense(n_params)
    ad = Adapter.of(sh, rank, targets, layers)
    tok = lora_training(sh, ad, seq_len)
    tf = tok.t if t is None else float(t)
    steps = tokens / batch_tokens
    opt = ad.values * FUSED_OPTIMIZER_MACS[f"_fused_{optimizer}_"] * per_mac("elementwise/f32")
    delta = ad.values * adapter_bits
    dg = (32 + STATE_BITS[optimizer]) * ad.values
    part = formalism.Part(tokens * tok.w + steps * opt, tokens, tokens * tok.w, tokens * tok.alpha, max(1, round(steps)),
                          dg)
    labels = _labels(u1_bits="model: the adapter's values, Delta",
                     delta_bits=f"model: the adapter's values at {adapter_bits:g} bits",
                     delta_grad_bits=f"model: their gradient at 32 bits and {optimizer}'s state")
    latents = adapter_bits < 16
    if latents:
        labels["delta_t_bits"] = "model: fp32 latents, 32 bits a value (a straight-through run)"
    m = _measured([part], TAU_BITS, ACT_BITS * sh.d_model, delta, tf, delta, labels,
                  delta_t=32 * ad.values if latents else None)
    return Model("lora", _dials(lora, locals()), m, w_fwd(sh, "bf16", seq_len),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "adapter_values": ad.values, "w_gates": tok.w,
                  "t_counted": tok.t})


def distill(student_params: float = REFERENCE_PARAMS, teacher_params: float = 10 * REFERENCE_PARAMS,
            teacher_share: float = 1.0, teacher_precision: str = "bf16", seq_len: float = 1024,
            batch_tokens: float = 65_536, optimizer: str = "adamw", t: float | None = None,
            tokens_per_param: float = 20.0) -> Model:
    """Distillation from a registered teacher: the student's dense training, and the teacher's forward on
    `teacher_share` of its tokens in the same token blocks. The teacher reads θ and the tokens alone, so its work is
    outside A: it raises w and leaves the affected count, u, t and a the student's."""
    sh, te = dense(student_params), dense(teacher_params)
    stu, teach = training(sh, "bf16", seq_len), forward(te, teacher_precision, seq_len)
    tok = Token(_add(stu.macs, {k: teacher_share * v for k, v in teach.macs.items()}), stu.affected, stu.reading)
    tokens = tokens_per_param * student_params
    steps = tokens / batch_tokens
    opt = optimizer_gates(sh, optimizer)
    part = formalism.Part(tokens * tok.w + steps * opt, tokens, tokens * tok.w, tokens * tok.alpha, max(1, round(steps)),
                          delta_grad(sh, optimizer))
    labels = _labels(u1_bits="model: the student's weights at bf16, 16 N_s", delta_bits="model: the student at bf16, 16 N_s",
                     delta_grad_bits=f"model: the student's gradient at 32 bits and {optimizer}'s state")
    m = _measured([part], TAU_BITS, ACT_BITS * sh.d_model, 16 * student_params, _dense_t(t), 16 * student_params, labels)
    return Model("distill", _dials(distill, locals()), m, w_fwd(sh, "bf16", seq_len),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "student_w": stu.w, "teacher_w": teach.w,
                  "student_share": stu.w / tok.w})


def _dials(fn, scope: dict) -> dict:
    return {name: scope[name] for name in inspect.signature(fn).parameters}


# ---- ρ and the console's columns ---------------------------------------------------------------------------------------

def columns(m: Model) -> dict:
    """The console's columns for the model: `C_gates`, te_core.algorithm_policy's `INPUTS` (u and u_in capped at Δ, as
    formalism.at reads them where d < c), `w_fit_gates` and `alpha_ratio`."""
    me = m.measured
    return ({"C_gates": me.gates} | dict(zip(algorithm_policy.INPUTS, algorithm_policy.inputs(me, m.w_fwd)))
            | {"w_fit_gates": me.w_fit, "alpha_ratio": me.alpha_ratio})


def rho(m: Model, c: float, k: float) -> dict:
    """ρ at (c, k) by te_core.formalism.at (None: infinite), its basis (fit: c/(c − d); target: Slowdown's bound),
    Slowdown's terms, and Fit's thresholds there: the most Δ its compute block admits, k − (c/w_fit)·τ, and the most
    Δ_grad its reduction block admits, k/2 (its update block also needs Δ_t + Δ_grad <= k)."""
    me = m.measured
    v = formalism.at(me, c, k)
    return {"rho": None if math.isinf(v["rho"]) else v["rho"], "basis": "fit" if v["fit"] else "target",
            "s": v["s"], "x0": v["x0"], "s_star": v["s_star"],
            "fit_delta_bits": k - c / me.w_fit * me.tau_bits, "fit_delta_grad_bits": k / 2}


def sensitivity(m: Model, c: float, k: float) -> dict:
    """Bounds beside te_core's for a record whose token blocks differ (RL's two kinds), from the counting of tokens'
    worth that s* bounds: every token with affected work leaves (1 − x0) of a token's worth to blocks that do at most s*
    each, so ρ >= (c/|C|)·n_A(1 − x0)/s* (`tokens_worth`, te_core's x0 and s*: its form without ᾱ/α_max), and with
    each kind's own profile ρ >= (c/|C|)·Σ_j n_j(1 − x0_j)/s*_j (`per_kind`). Each is max(1, ·) where Fit fails."""
    v = formalism.at(m.measured, c, k)
    if v["fit"]:
        return {"tokens_worth": v["rho"], "per_kind": v["rho"]}
    me = m.measured
    n_a = sum(p.tokens for p in me.parts if p.affected_gates)
    out = {"tokens_worth": c / me.gates * n_a * (1 - v["x0"]) / v["s_star"] if v["s_star"] else math.inf}
    kinds = m.derived.get("kinds")
    if kinds:
        total = 0.0
        for kd in kinds:
            u, u_in = profile(kd["u1_bits"], kd["t"])
            if v["delta_counted"]:
                u, u_in = formalism.capped(u, me.delta_bits), formalism.capped(u_in, me.delta_bits)
            s_star = formalism.room(u, k) / me.a_bits
            x0 = formalism.bottom_share(u_in, k)
            total += kd["tokens"] * (1 - x0) / s_star if s_star else math.inf
        out["per_kind"] = c / me.gates * total
    return {key: None if math.isinf(b) else max(1.0, b) for key, b in out.items()}


# ---- the workloads and their reference points ------------------------------------------------------------------------

@dataclass(frozen=True)
class Dial:
    name: str
    unit: str
    default: object
    range: tuple
    why: str


@dataclass(frozen=True)
class Check:
    """A pilot record beside the model at `dials`: the record's console columns (te_core 4.1's measured quantities,
    t literal; `RECORD_SOURCE`), on which page_rho gives the record's ρ at each policy."""
    name: str
    dials: dict
    record: str
    columns: dict
    differs: str


@dataclass(frozen=True)
class Workload:
    name: str
    title: str
    what: str
    fn: object
    dials: tuple[Dial, ...]
    #: (point name, dial settings beside the defaults)
    points: tuple[tuple[str, dict], ...]
    checks: tuple[Check, ...] = ()


RECORD_SOURCE = ("te_core 4.1's console columns for the record (algorithm_policy, quantities measured, t literal); "
                 "delta_grad_bits its trace's gradient cut and state; u_in_points the trace's u_in at the policies' k")
#: the console's u and u_in of a 45M full-training record: 0 up to the literal t, then the whole traced u(1) (the fp32
#: embedding and bf16 matrices); u_in jumps there at x = 0, the trace's cut through the embedding table
FULL_U = {"u_points": f"0:0;{T_DENSE}:0;{T_DENSE}:1.13268e9;1:1.13268e9",
          "u_in_points": "0:0;0:80000;0:145568;0:800000;0:8e6;0:8e7;0:8e8;1:1.13268e9"}
FULL_RECORD = {"tau_bits": 16.0156, "a_bits": 8192, "delta_bits": 1440996608, "delta_t_bits": 1.441e9, "d_gates": 0}


N_RANGE = (1e7, 1e11)
COMMON = {
    "n_params": Dial("n_params", "parameters", REFERENCE_PARAMS, N_RANGE,
                     "the trained model; d_model and n_layer follow at the reference model's d/L (te_core.workloads.dense)"),
    "precision": Dial("precision", "GEMM format", "bf16", ("bf16", "e4m3", "fp4"),
                      "the weight GEMMs' format, forward and backward; attention stays bf16"),
    "seq_len": Dial("seq_len", "tokens", 1024, (256, 32768), "the training context: each token's attention span"),
    "batch_tokens": Dial("batch_tokens", "tokens a step", 65_536, (1e4, 1e7),
                         "sets the optimizer's share, 1 - sigma; the pilot's 64 x 1,024"),
    "optimizer": Dial("optimizer", "rule", "adamw", ("adamw", "muon", "sgd"),
                      "sets Delta_grad (the state) and the optimizer's gates"),
    "t_dense": Dial("t", "share of a token's affected work", None, (0.0, 0.9),
                    f"A1's free share: None is te_core's literal t on the 45M trace, {T_DENSE}; 0 is evenly spread"),
    "tokens_per_param": Dial("tokens_per_param", "tokens / N", 20.0, (1.0, 2000.0),
                             "sets |C| only (Chinchilla's 20 by default)"),
}

WORKLOADS = {w.name: w for w in (
    Workload("standard", "Standard dense pretraining",
             "Every weight of a dense decoder trained by backpropagation, its optimizer outside the token blocks.",
             standard,
             (COMMON["n_params"], COMMON["precision"], COMMON["seq_len"], COMMON["batch_tokens"], COMMON["optimizer"],
              COMMON["t_dense"], COMMON["tokens_per_param"]),
             (("default", {}), ("evenly spread", {"t": 0.0}), ("7B", {"n_params": 7e9})),
             (Check("45M ref-optim/adamw-rewarm-mild", {}, "r20261102-140347-a1c2@946",
                    FULL_RECORD | FULL_U | {"w_gates": 2.31908e11, "sigma": 0.999823, "w_fit_gates": 1.5441e11,
                                            "alpha_ratio": 0.995234, "delta_grad_bits": 4322869248},
                    "w 0.9% below the model's; u(1) the fp32 embedding with the bf16 matrices, 25 bits a parameter; "
                    "u_in all of it at x = 0; Delta the submitted code, 32 N"),
              Check("45M ref-optim/cooldown-forks", {"optimizer": "muon"}, "r20261102-134356-ee8f@694",
                    FULL_RECORD | FULL_U | {"w_gates": 2.32623e11, "sigma": 0.993392, "w_fit_gates": 1.03235e11,
                                            "alpha_ratio": 0.992069, "delta_grad_bits": 3706306560},
                    "as adamw-rewarm-mild; sigma the trace's 0.99339 against the model's 0.98774, five bf16 "
                    "Newton-Schulz iterations on every matrix"))),
    Workload("diloco", "DiLoCo-style pretraining",
             "Standard training on workers that sync every H inner steps by an outer step on their pseudo-gradients, "
             "optionally compressed.",
             diloco,
             (Dial("n_params", "parameters", REFERENCE_PARAMS, N_RANGE,
                   "the inner model, which every worker trains; d_model and n_layer follow (te_core.workloads.dense)"),
              COMMON["precision"], COMMON["seq_len"],
              Dial("batch_tokens", "tokens an inner step", 65_536, (256, 1e7), "B, an inner step's tokens, all workers"),
              COMMON["optimizer"], COMMON["t_dense"],
              Dial("inner_steps", "steps", 500, (1, 10_000), "H, the inner steps between syncs"),
              Dial("syncs", "syncs", 28, (1, 1e5),
                   "S, the record's outer steps so far; its tokens are S H B (28: 20 a parameter at 45M)"),
              Dial("sync_bits", "bits a sync", None, (1e2, 1.5e9),
                   "b, one compressed outer delta; None is 32 N, uncompressed")),
             (("default", {}), ("1-bit syncs", {"sync_bits": REFERENCE_PARAMS}),
              ("Fit's corner", {"inner_steps": 2, "batch_tokens": 256, "syncs": 32, "sync_bits": 1024}))),
    Workload("rl", "RL rollouts on the policy being trained",
             "Rollouts of the current policy in FP4 and a learner training on them, every token affected.",
             rl,
             (COMMON["n_params"],
              Dial("update", "what trains", "full", ("full", "lora"), "every weight, or a LoRA kept apart from the base"),
              Dial("rank", "rank", 16, (1, 256), "the LoRA's rank (update lora)"),
              Dial("targets", "matrices", "all", tuple(TARGETS), "the LoRA's matrices in each adapted layer"),
              Dial("layers", "layers", None, (1, 128), "the LoRA's adapted layers, from the top; None is all"),
              Dial("rollout_precision", "GEMM format", "fp4", ("bf16", "e4m3", "fp4"), "the rollouts' weight GEMMs"),
              Dial("prompt_tokens", "tokens", 256, (16, 32768), "a rollout's prompt, the only inputs"),
              Dial("generated_tokens", "tokens", 768, (16, 32768), "a rollout's generated tokens"),
              Dial("samples_per_trained", "sequences", 1.0, (1.0, 16.0),
                   "rollouts generated for each one trained on (1: each once; 4: one in four kept)"),
              COMMON["optimizer"],
              Dial("batch_tokens", "learner tokens a step", 65_536, (1e4, 1e7), "sets the optimizer's share, 1 - sigma"),
              Dial("t", "share of a token's affected work", None, (0.0, 0.999),
                   "A1's free share: None is the larger of the two kinds' (the learner's te_core literal t, or a "
                   "LoRA's modelled one); 0 is evenly spread"),
              Dial("tokens", "learner tokens", 1e8, (1e6, 1e11), "sets |C| only")),
             (("default", {}), ("LoRA rank 16", {"update": "lora"}),
              ("LoRA in one layer", {"update": "lora", "rank": 1, "targets": "q", "layers": 1}))),
    Workload("inference", "Inference on an unregistered model",
             "Serving a model whose weights are not registered, so all of them are update bits.",
             inference,
             (Dial("n_params", "parameters", REFERENCE_PARAMS, N_RANGE, "the served model"),
              Dial("precision", "GEMM format", "fp4", ("bf16", "e4m3", "fp4"), "the served weights' format"),
              Dial("prompt_tokens", "tokens", 4096, (16, 131072), "a session's prompt, the only inputs"),
              Dial("generated_tokens", "tokens", 1024, (1, 32768), "a session's generated tokens"),
              Dial("t", "share of a token's work", None, (0.0, 0.95),
                   "A1's free share: None is attention's modelled share; 0 is evenly spread"),
              Dial("tokens", "tokens served", 1e9, (1e6, 1e13), "sets |C| only")),
             (("default", {}), ("evenly spread", {"t": 0.0}), ("7B", {"n_params": 7e9}))),
    Workload("lora", "Fine-tuning a registered model with a LoRA",
             "A small adapter trained on a registered model; the base is frozen.",
             lora,
             (COMMON["n_params"],
              Dial("rank", "rank", 8, (1, 256), "the adapter's rank"),
              Dial("targets", "matrices", "all", tuple(TARGETS), "the adapted matrices in each adapted layer"),
              Dial("layers", "layers", None, (1, 128), "the adapted layers, from the top; None is all"),
              Dial("adapter_bits", "bits a value", 16, (1, 32), "the adapter's values as submitted"),
              Dial("optimizer", "rule", "adamw", ("adamw", "sgd"), "sets Delta_grad: 64 or 0 state bits a value"),
              COMMON["seq_len"], COMMON["batch_tokens"],
              Dial("t", "share of a token's affected work", None, (0.0, 0.9999),
                   "A1's free share: None is the modelled share reading no adapter value; 0 is evenly spread"),
              Dial("tokens", "tokens", 1e8, (1e6, 1e11), "sets |C| only")),
             (("default", {}), ("rank 1 on W_q", {"rank": 1, "targets": "q"}),
              ("rank 1 on W_q, one layer", {"rank": 1, "targets": "q", "layers": 1})),
             (Check("45M sparse/lora-alloc", {"rank": 4, "targets": "qv"}, "r20261102-132959-286b@1720",
                    {"w_gates": 1.66764e11, "sigma": 0.999972, "tau_bits": 16.0156, "a_bits": 8192,
                     "delta_bits": 788712, "delta_t_bits": 788712, "d_gates": 0, "delta_grad_bits": 4657152,
                     "w_fit_gates": 1.6639e11, "alpha_ratio": 0.998823,
                     "u_points": "0:0;0.999235:0;0.999235:788712;1:788712",
                     "u_in_points": "0:0;0.00105837:32768;0.00315535:65536;0.100031:262144;1:788712"},
                    "48,512 trained values against the model's 49,152; u_in front-loaded (a token's forward reads every "
                    "adapter value in dependency order before most of its work), where the model's A1' is even"),)),
    Workload("distill", "Distillation from a registered teacher",
             "A student trained on a registered teacher's outputs, the teacher's forward in each token block.",
             distill,
             (Dial("student_params", "parameters", REFERENCE_PARAMS, N_RANGE, "the trained student"),
              Dial("teacher_params", "parameters", 10 * REFERENCE_PARAMS, (1e7, 1e12), "the registered teacher"),
              Dial("teacher_share", "share of tokens", 1.0, (0.0, 1.0), "the student tokens the teacher runs on"),
              Dial("teacher_precision", "GEMM format", "bf16", ("bf16", "e4m3", "fp4"), "the teacher's weight GEMMs"),
              COMMON["seq_len"], COMMON["batch_tokens"], COMMON["optimizer"], COMMON["t_dense"],
              COMMON["tokens_per_param"]),
             (("default", {}), ("no teacher", {"teacher_share": 0.0}), ("100x teacher", {"teacher_params": 100 * 45_029_888}))),
)}


def model(name: str, **dials) -> Model:
    return WORKLOADS[name].fn(**dials)


def point(m: Model) -> dict:
    """A model's reference point: its dials, its columns and what they're estimated from, ρ at each of `POLICIES`, the
    sensitivity bounds where its token blocks differ, and what it derived."""
    out = {"dials": m.dials, "inputs": columns(m), "estimated": algorithm_policy.estimated(m.measured),
           "rho": {name: rho(m, c, k) for name, (c, k) in POLICIES.items()}, "derived": m.derived}
    if m.derived.get("kinds"):
        out["sensitivity"] = {name: sensitivity(m, c, k) for name, (c, k) in POLICIES.items()}
    return out


def _finite(v: float) -> float | None:
    return None if math.isinf(v) else v


def _check(ck: Check, workload: str) -> dict:
    """A pilot record beside its model (`Check`): ρ by page_rho on the record's columns at each policy, and the
    model's at the check's dials."""
    m = model(workload, **ck.dials)
    return {"record": ck.record, "source": RECORD_SOURCE, "dials": m.dials, "differs": ck.differs,
            "record_inputs": ck.columns, "model_inputs": columns(m),
            "rho": {name: {"record": _finite(algorithm_policy.page_rho(ck.columns, c, k)[0]),
                           "model": rho(m, c, k)["rho"]} for name, (c, k) in POLICIES.items()}}


def reference() -> dict:
    """Every workload's reference points and checks at `POLICIES` (module docstring): `workloads.json`."""
    out = {"id": "te_core.workloads", "te_core": VERSION, "spec": NOTE,
           "policies": {name: {"c": c, "k": k, "session": dict(zip(("inputs", "outputs"), SESSIONS[name]))}
                        for name, (c, k) in POLICIES.items()},
           "gates_per_mac": {kind: per_mac(kind) for kind in ("gemm/bf16", "gemm/e4m3", "gemm/fp4", "gemm/f32",
                                                              "elementwise/bf16", "elementwise/f32")},
           "workloads": {}}
    for w in WORKLOADS.values():
        out["workloads"][w.name] = {
            "title": w.title, "what": w.what,
            "dials": [{"name": d.name, "unit": d.unit, "default": d.default, "range": list(d.range), "why": d.why}
                      for d in w.dials],
            "points": {name: point(model(w.name, **settings)) for name, settings in w.points},
            "checks": {ck.name: _check(ck, w.name) for ck in w.checks}}
    return out


def _fmt(v) -> str:
    return "inf" if v is None else f"{v:.4g}"


def table(ref: dict) -> str:
    lines = ["| workload | point | " + " | ".join(f"rho {p}" for p in POLICIES) + " |",
             "|---|---|" + "---|" * len(POLICIES)]
    for name, w in ref["workloads"].items():
        for pname, p in w["points"].items():
            cells = [f"{_fmt(p['rho'][q]['rho'])} ({p['rho'][q]['basis']})" for q in POLICIES]
            lines.append(f"| {name} | {pname} | " + " | ".join(cells) + " |")
        for cname, ck in w["checks"].items():
            cells = [f"{_fmt(ck['rho'][q]['record'])} record, {_fmt(ck['rho'][q]['model'])} model" for q in POLICIES]
            lines.append(f"| {name} | check: {cname} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path, help=f"write the reference points here ({REFERENCE.name} is the pinned copy)")
    a = ap.parse_args(argv)
    ref = reference()
    if a.json:
        a.json.write_text(json.dumps(ref, indent=1, allow_nan=False) + "\n")
    print(table(ref))
    return 0


if __name__ == "__main__":
    sys.exit(main())
