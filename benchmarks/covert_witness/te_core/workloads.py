"""Covert workloads under the policy, each a model: from a workload's dials to the inputs of the one bound's kinds, and
ρ at a policy (c, k), for the console's *Standard training under the policy* plot.

    python -m te_core.workloads [--json OUT]    # every workload's reference points at the three policies

A workload (`WORKLOADS`) is a function from its dials (`Dial`: name, unit, default, range, why) to a `Model`: the token
kind's inputs as a te_core.formalism `Measured` (|C|, w, σ, τ, a, the profiles u and u_in, Δ, d, Δ_grad and Δ_t, each
labelled with where it comes from), its products of two fresh operands (`Fresh`), its optimizer steps (`Steps`) and
w_fwd. `rho` gives every kind's term at a policy (c, k), their max and the binding kind; `columns` gives the token
kind's inputs as the console's columns (te_core.algorithm_policy's `INPUTS`, with `w_fit_gates`, `alpha_ratio` and
`C_gates`), on which `algorithm_policy.page_rho` gives te_core.formalism.at's ρ, and `menu` the other kinds' inputs.
`reference` evaluates each workload's reference points (`Workload.points`) at the three `POLICIES`, and its `Check`s,
pilot records beside the model; `workloads.json` is its output, which the console's tests pin. The spec, each mapping
with its reasons, is the note `NOTE` names.

**The one bound** (the coordinator's 06:51Z ruling; Lean's `Counting` and `Slowdown`): ρ ≥ the max over a fixed menu
of kinds K of `kindBound`, (|B_K|/|C|)·c/cap_K(k), cap_K(k) the most of K's work one block does reading at most k bits
beyond θ. Each term is a lower bound, so the max is one. Where Fit holds, its construction gives c/(c − d)
(te_core.formalism), which `rho` reports with the menu beside it. The kinds (`KINDS`, each with what it rests on):

- **fresh products** (`Fresh`): products whose two operands both depend on θ'. A product C = A·B by the standard
  algorithm is a triple, and the amortized Loomis-Whitney cap (the Lean lane's `sum_card_le_cap`, which charges only
  reads) bounds the blocks of a partition: Σ|T| ≤ B·cap + μ*·E, cap = k^1.5/(4·sqrt(b_A b_B b_C)), μ* = cap/k and E
  the b_C bits of each output entry, so B ≥ MACs/cap − b_C·entries/k (`fresh_blocks`, at the widths as passed,
  `WIDTHS`; the io-bound module's term replaces it). The board's (k/3)^1.5 form charges a block for every entry it
  touches, so its cap is 4/(3·sqrt 3) of this one and its term 1.30 times this one's. The entry term leaves no bound on
  products summing fewer than μ*·b_C terms an entry (22.6 at the console default's bf16): a LoRA's (x·A)·B, over its
  rank. A block's normalized reads are convex, so products of several widths share its k and their blocks add. A
  product with an operand in θ (a frozen weight, a registered teacher's) isn't fresh, and a block computes a product's
  operands itself where it can read the update up to the product's layer and the input ids up to its token (a
  forward) or of its whole sequence (a backward, whose gradient reads every later token): those products escape
  (`Fresh.counted`).
- **token**: te_core.formalism.at's token form, the affected work no block reaches from a token's inputs, at reading B
  (a token's work lumped, under A1's profiles) until te_core implements reading A's entry-point cut.
- **reduction** (`Steps`): (c/|C|)·Σ steps·(b − 1)·Δ_g/k, b a step's token blocks (its tokens over s) or partials and
  Δ_g one partial gradient at its passed width: max(1, (1 − 1/b)·Δ_g/k) for one kind of step at σ = 1 (the 07:48Z
  correction: compute blocks with spare reads can each add their partial to a running sum, so the term isn't 1 plus).
- **update**: (c/|C|)·Σ steps·(Δ_t + Δ_g)/k, the weights and the gradient each step's update reads (05:18Z).
- **decoding**: c/(c − d) where a block computing tokens from their inputs reads the code (Fit's compute clause holds)
  and the code is shorter than the update it describes (a LoRA's E is its adapter, which a block runs as it is).

The reduction and update terms count only where Fit fails (05:02Z): where Fit holds, a step's reduction and update
application each fit a block.

The conventions every workload shares:

- **Policies** (`POLICIES`): `default` is the console default as the console defines it (the dashboard's 05:34Z
  contract): one served session of the 45M reference, 4,096 tokens in and 1,024 out, c = 5,120 x 2 x 45e6 x 740 gates
  (2N FLOPs a token, attention left out) and k = 65,536 bits. `default_full` is its c corrected to the session's full
  cost, attention at te_core.flops' count (`ATTENTION`, 1.57 c). `served` is the registered serving policy, the served
  demo's measured block (te_core.panels.SERVED; its c is a lower bound, the linear layers still to be counted).
- **Shapes** (`dense`): a dense decoder of N parameters with the reference model's d/L and SwiGLU width, d solved from N
  and the vocabulary (`vocab`: the 45M reference's 50,304, s-t20's 32,000), τ its ⌈log2 V⌉ bits an id.
- **Work**: a token's MACs at te_core.gates' prices, its weight GEMMs at the workload's precision and its attention in
  bf16 at te_core.flops' count (the whole span, so a training token at context T has 2·L·d·T), with `ELEMENTWISE` bf16
  elementwise MACs per MAC. Attention's QK^T and dO·V^T sum over a head's `HEAD_DIM` values, its other products over
  the span. a is the bf16 residual stream, 16·d. Δ_g is a partial gradient in fp32 (`GRAD_BITS`).
- **The decoder** d, by te_core 4.2's rule: 0 where E stores θ' itself (every full-weight workload), the harness's
  decoder for an E holding a LoRA (`Adapter.decoder_gates`), and compressed DiLoCo's replay of a phase.
- **A1's reading** (the dial `t`, the token kind's only): u is 0 up to the free share t and x·u(1) beyond it, u_in
  x·u(1) (`profile`: `formalism.with_tail` of `formalism.even`). Dense training's default t is `T_DENSE`, te_core's
  literal t measured on the 45M pilot's trace; elsewhere t is the modelled share of a token's affected work that reads
  no update value (`Token.t`). t = 0, the evenly spread reading, overcharges a block (the coordinator, 07:05Z).
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import VERSION, algorithm_policy, formalism, panels, policy
from .gates import gates, per_mac
from .opnames import FUSED_OPTIMIZER_MACS

NOTE = "note:training-20261103T0600Z-draft-workload-models"
REFERENCE = Path(__file__).with_name("workloads.json")
VOCAB = 50_304
#: the console's served model: the 45M dense starts' parameters (starts_store.json's s50 and s100)
REFERENCE_PARAMS = 45_029_888
ACT_BITS = policy.ACT_BITS
#: the reference model's d_model / n_layer (512 / 6) and SwiGLU hidden width over d_model (1,408 / 512)
ASPECT, MLP = 512 / 6, 1408 / 512
#: bf16 elementwise MACs per GEMM or attention MAC: 0.01 puts a 45M training token at 3.08 forwards, 2.3396e11 gates,
#: within 1% of te_core's counts on the pilot's full-training records (2.319e11 to 2.341e11)
ELEMENTWISE = 0.01
#: a weight as a GEMM of each precision reads it: NVFP4 is 4 bits and an e4m3 scale per 16 values
WEIGHT_BITS = {"bf16": 16.0, "e4m3": 8.0, "fp4": 4.5}
#: the widths (b_A, b_B, b_C) a product of each GEMM kind passes: both operands as the GEMM reads them (W4A4 under
#: NVFP4: each activation row coded too) and its partial sums in fp32
WIDTHS = {f"gemm/{p}": (bits, bits, 32.0) for p, bits in WEIGHT_BITS.items()}
#: a partial gradient's bits a trained value, as passed
GRAD_BITS = 32
#: te_core 4.1's literal t on the 45M pilot's traced full-training step (every ref-optim record's): the share of a
#: training token's affected work in ops that read no update value
T_DENSE = 0.692646
NEWTON_SCHULZ_STEPS = 5
#: optimizer state bits per trained value (Muon's is its momentum on matrices, AdamW's on everything else)
STATE_BITS = {"adamw": 64, "sgd": 0, "muon": 32}
#: the lf family's attention head, d_model / n_head at every size (s 512/8, m 768/12, l 1024/16)
HEAD_DIM = 64
#: each LoRA target set's matrices in a layer: (Σ d_in, Σ d_out) in d_models, and how many
TARGET_DIMS = {"q": (1.0, 1.0, 1), "qv": (2.0, 2.0, 2), "attn": (4.0, 4.0, 4), "mlp": (2 + MLP, 1 + 2 * MLP, 3),
               "all": (6 + MLP, 5 + 2 * MLP, 7)}
#: (d_in + d_out) per rank per layer of each LoRA target set, in d_models
TARGETS = {name: d_in + d_out for name, (d_in, d_out, _) in TARGET_DIMS.items()}
#: the entries of the matrices each LoRA target set adapts, per layer, in d_model²
TARGET_ENTRIES = {"q": 1.0, "qv": 2.0, "attn": 4.0, "mlp": 3 * MLP, "all": 4.0 + 3 * MLP}
#: the share of a LoRA's x·A MACs (by d_in) at its lowest adapted layer whose input is fresh, by targets: W_o's after
#: adapted attention, the MLP's after it and the down projection's after adapted gate and up (W_q, W_k and W_v read
#: the layer's registered input); its (x·A)·B and dy·B^T always have two fresh operands
LOWEST_INPUT = {"q": 0.0, "qv": 0.0, "attn": 0.25, "mlp": MLP / (2 + MLP), "all": (3 + MLP) / (6 + MLP)}
#: the lowest adapted layer's attention products with two fresh operands, by targets, in d·span MACs a token: forward
#: (QK^T, PV) and backward (dO·V^T, then dV, dQ and dK together). QK^T needs q and k adapted, PV v; the backward's
#: products follow theirs and compute only the gradients the adapted matrices need; the MLP comes after attention
LOWEST_ATTENTION = {"q": ((0, 0), (0, 0)), "qv": ((0, 1), (1, 1)), "attn": ((1, 1), (1, 3)), "mlp": ((0, 0), (0, 0)),
                    "all": ((1, 1), (1, 3))}
#: the one bound's menu (module docstring), each kind with what its term rests on
KINDS = {
    "fresh_products": "no A1: the amortized Loomis-Whitney cap for products computed by the standard algorithm, at the "
                      "widths as passed (the Lean lane's sum_card_le_cap at training-lw-lean 0aa80a2a75, word-level; "
                      "the 06:57Z obligations of a bit-level cap and recomputed operands are open)",
    "token": "A1 at reading B: te_core.formalism's token form, a token's work lumped (reading A's entry-point cut "
             "replaces it)",
    "reduction": "the counting lemma: every partial gradient a step sums but one is read by some block, at its passed "
                 "width",
    "update": "the counting lemma: each step's update reads Δ_t + Δ_g, the weights and the gradient (05:18Z; a block "
              "keeping a coordinate across steps reads its weight once, so this may overstate)",
    "decoding": "Fit's construction: a block computing tokens from their inputs reads the code and runs its decoder",
}
FIT = "Fit's construction (te_core.formalism): every block reads Δ and its tokens' ids, and runs the decoder"


def fresh_cap(k: float, widths: tuple[float, float, float]) -> float:
    """The amortized cap on products of two fresh operands, k^1.5/(4·sqrt(b_A b_B b_C)) a block (`sum_card_le_cap`):
    MACs ≤ sqrt(|A||B||C|) (Loomis-Whitney), the operands A and B read and each output entry of C paid for once by
    reads across the partition."""
    a, b, partial = widths
    return k ** 1.5 / (4 * math.sqrt(a * b * partial))


def fresh_blocks(macs: float, entries: float, k: float, widths: tuple[float, float, float]) -> float:
    """The blocks a set of products summing into `entries` output entries needs: Σ|T| ≤ B·cap + μ*·b_C·entries with
    μ* = cap/k, so B ≥ MACs/cap − b_C·entries/k, and none where that's negative (products summing fewer than μ*·b_C
    terms an entry)."""
    return max(0.0, macs / fresh_cap(k, widths) - widths[2] * entries / k)


# ---- shapes and work ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    params: float
    d_model: float
    n_layer: float
    vocab: float = VOCAB

    @property
    def tau(self) -> float:
        """A token's input id, ⌈log2 V⌉ bits."""
        return float(policy.id_bits(self.vocab))

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


def dense(params: float, aspect: float = ASPECT, vocab: float = VOCAB) -> Shape:
    """The dense decoder of `params` parameters at d/L = `aspect`: N = L·((4 + 3·MLP)·d² + 2d) + (V + 1)·d (each layer's
    matrices and two norms, a tied embedding and the final norm), with L = d/aspect a cubic in d, solved by Cardano's
    formula (one real root). 45,029,888 gives the reference model, d = 512 and L = 6, as 35,658,240 does at V = 32,000
    (s-t20)."""
    a, b, c = (4 + 3 * MLP) / aspect, 2 / aspect, vocab + 1.0
    p = (3 * a * c - b * b) / (3 * a * a)
    q = (2 * b ** 3 - 9 * a * b * c - 27 * a * a * params) / (27 * a ** 3)
    r = math.cbrt(-q / 2 + math.sqrt(q * q / 4 + p ** 3 / 27))
    d = r - p / (3 * r) - b / (3 * a)
    return Shape(float(params), d, d / aspect, float(vocab))


def decode_span(prompt: float, generated: float) -> float:
    """The mean span a token attends over in one served sequence, as te_core.flops counts it: the prompt's prefill at
    the whole prompt, then each generated token over the prompt and the tokens before it."""
    return (prompt * prompt + generated * prompt + generated * generated / 2) / (prompt + generated)


#: the console's c over the session's full cost: a session token's attention at te_core.flops' count beside its N MACs
ATTENTION = 1 + dense(REFERENCE_PARAMS).attention(decode_span(4096, 1024)) / REFERENCE_PARAMS
#: the console default as the console defines it (the dashboard's 05:34Z contract), (c gates, k bits)
CONSOLE = (5120 * 2 * 45e6 * 740, 65_536.0)
POLICIES = {"default": CONSOLE, "default_full": (CONSOLE[0] * ATTENTION, CONSOLE[1]),
            "served": (float(panels.SERVED[0]), float(panels.SERVED[1]))}
POLICY_NOTES = {
    "default": "the console default as the console defines it (dashboard, 05:34Z): one 4,096-in, 1,024-out session of "
               "the 45M reference a block, c = 5,120 x 2 x 45e6 x 740 gates (attention left out), k = 65,536 bits",
    "default_full": f"the console default with c the session's full cost, attention at te_core.flops' count "
                    f"({ATTENTION:.4f} c)",
    "served": "the registered serving policy: the served demo's measured block (te_core.panels.SERVED, one slot's 35-step "
              "session of Llama-3.2-1B int8 under vLLM): c a lower bound, its linear layers still to be counted and "
              "its 133 BitAtx32_v1 activations a block unpriced; k its fresh bits",
}


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


def newton_schulz_macs(sh: Shape) -> float:
    """One step's Newton-Schulz MACs on every matrix: per iteration 2a²b + a³ on an a x b matrix, a ≤ b (3d³ for
    attention's, (2·MLP + 1)·d³ for the MLP's)."""
    return NEWTON_SCHULZ_STEPS * sh.n_layer * sh.d_model ** 3 * (4 * 3 + 3 * (2 * MLP + 1))


def optimizer_gates(sh: Shape, optimizer: str) -> float:
    """One step's optimizer gates: AdamW (and SGD) at te_core's fused-op count, 8 (2) f32 MACs a parameter; Muon's
    Newton-Schulz in bf16 on every matrix and AdamW on the rest."""
    f32 = per_mac("elementwise/f32")
    if optimizer != "muon":
        return sh.params * FUSED_OPTIMIZER_MACS[f"_fused_{optimizer}_"] * f32
    return (newton_schulz_macs(sh) * per_mac("gemm/bf16")
            + (sh.params - sh.matrix_params) * FUSED_OPTIMIZER_MACS["_fused_adamw_"] * f32)


def delta_grad(sh: Shape, optimizer: str) -> float:
    """Δ_grad of full training: the gradient at 32 bits a value and the optimizer's state (Muon's momentum on the
    matrices, AdamW's two moments on the rest). The pilot's traces measure exactly these: 96·N for AdamW
    (ref-optim/adamw-rewarm) and 3,706,306,560 for Muon (ref-optim/cooldown-forks), at the 45M start."""
    if optimizer == "muon":
        return GRAD_BITS * sh.params + 32 * sh.matrix_params + 64 * (sh.params - sh.matrix_params)
    return (GRAD_BITS + STATE_BITS[optimizer]) * sh.params


@dataclass(frozen=True)
class Adapter:
    """A LoRA on `targets` (`TARGETS`) of the top `layers` layers at `rank`: its trained values, the entries of the
    matrices it adapts, and its MACs a token (one pass: x·A and (x·A)·B per adapted matrix)."""
    values: float
    layers: float
    rank: float
    entries: float
    targets: str = "all"

    @classmethod
    def of(cls, sh: Shape, rank: float, targets: str, layers: float | None) -> Adapter:
        n = sh.n_layer if layers is None else min(float(layers), sh.n_layer)
        return cls(rank * TARGETS[targets] * sh.d_model * n, n, float(rank),
                   TARGET_ENTRIES[targets] * sh.d_model ** 2 * n, targets)

    @property
    def macs(self) -> float:
        return self.values

    def decoder_gates(self, quantized: bool = False) -> float:
        """d for an E holding this LoRA, te_core 4.2's witness: the harness's decoder (`prepare.rebuild_macs`), r + 2
        fp32 multiply-adds an adapted entry (B @ A, the scale, the add) and, for factors below 16 bits (its lora2),
        r·(d_in + d_out) more to dequantize them, at `formalism.DECODER_DEFINITION`'s gates each."""
        macs = (self.rank + 2) * self.entries + (self.values if quantized else 0.0)
        return macs * gates(formalism.DECODER_DEFINITION)


def lora_training(sh: Shape, ad: Adapter, span: float) -> Token:
    """A LoRA training token in bf16: every layer's forward; the input gradient through the head and the adapted layers
    (the frozen weights get no weight gradient); the adapter's forward, input gradient and weight gradient. The layers
    below the lowest adapted one are unaffected; the adapter's forward and input gradient read its values."""
    head, frac = sh.vocab * sh.d_model, ad.layers / sh.n_layer
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


# ---- the other kinds' inputs: fresh products and optimizer steps -------------------------------------------------------

def _layers(n: float) -> list[tuple[int, float]]:
    """Each layer of a stack of n from the bottom, with its weight (the top one's n's fraction where n isn't whole)."""
    return [(j, min(1.0, n - j)) for j in range(math.ceil(n - 1e-9))]


@dataclass(frozen=True)
class Fresh:
    """Products of two fresh operands: `count` tokens (or steps) of `macs` MACs of GEMM kind `kind`, summing into
    `entries` output entries, at each of `layers` layers of the update (the lowest's share `lowest` of them; the share
    of the update up to layer j is (j + 1)/layers), in sequences of `positions` tokens whose first `inputs` read an
    input id of `tau` bits each; `sequence` where a block computing their operands needs the whole sequence and update
    (a backward: the gradient reads every later token), not only the update up to the layer and the ids up to the
    token (a forward)."""
    kind: str
    macs: float
    entries: float
    layers: float
    count: float
    positions: float
    inputs: float
    tau: float
    lowest: float = 1.0
    sequence: bool = False

    def escaped(self, update_bits: float, k: float) -> float:
        """The share of a layer's products that a block computes from θ, `update_bits` of the update's share up to
        it and its tokens' ids within k bits: every token where the whole sequence's ids fit beside the update,
        else (a forward) the first ⌊(k − update)/τ⌋ positions, each reading the ids up to its own."""
        left = k - update_bits
        if left >= self.tau * self.inputs:
            return 1.0
        if self.sequence or left < self.tau:
            return 0.0
        return math.floor(left / self.tau) / self.positions

    def _layers(self) -> list[tuple[float, float]]:
        """(the update's share a block reads for the layer, the layer's weight) for each layer."""
        return [(1.0 if self.sequence else min(1.0, (j + 1) / self.layers), wt * (self.lowest if j == 0 else 1.0))
                for j, wt in _layers(self.layers)]

    def _share(self, update_bits: float, k: float) -> float:
        return self.count * sum(wt * (1 - self.escaped(r * update_bits, k)) for r, wt in self._layers())

    def counted(self, update_bits: float, k: float) -> tuple[float, float]:
        """The MACs and output entries the kind counts: those no block computes from θ, the update and the inputs
        within k bits."""
        share = self._share(update_bits, k)
        return share * self.macs, share * self.entries

    @property
    def total(self) -> float:
        return self.count * self.macs * sum(wt for _, wt in self._layers())


def dense_fresh(sh: Shape, gemm: str, tokens: float, span: float, seq: tuple, batch_tokens: float | None = None,
                layered: bool = True) -> list[Fresh]:
    """A dense model's fresh products, every weight fresh: each layer's matrices at `gemm` and its attention over `span`
    in bf16 (a forward, whose operands need the update up to the layer: `layered`, else all of it, as a code that
    isn't per layer) and the head above them; with `batch_tokens` (training, a step's tokens) the input gradients, the
    weight gradients (each entry a step's sum) and attention's backward. `seq` is (positions, inputs, τ)."""
    n = sh.n_layer if layered else 1.0
    d, head, heads, attn = sh.d_model, sh.vocab * sh.d_model, sh.d_model / HEAD_DIM, sh.attention(span) / 2

    def layer(kind: str, macs: float, entries: float) -> Fresh:
        return Fresh(kind, macs / n, entries / n, n, tokens, *seq)

    def whole(kind: str, macs: float, entries: float) -> Fresh:
        return Fresh(kind, macs, entries, 1.0, tokens, *seq, sequence=True)

    out = [layer(gemm, sh.matrix_params, sh.n_layer * (5 + 2 * MLP) * d), Fresh(gemm, head, sh.vocab, 1.0, tokens, *seq),
           layer("gemm/bf16", attn, sh.n_layer * heads * span), layer("gemm/bf16", attn, sh.n_layer * d)]
    if batch_tokens is not None:
        weights = sh.matrix_params + head
        out += [whole(gemm, weights, sh.n_layer * (6 + MLP) * d + d), whole(gemm, weights, weights / batch_tokens),
                whole("gemm/bf16", attn, sh.n_layer * heads * span), whole("gemm/bf16", 3 * attn, 3 * sh.n_layer * d)]
    return out


def adapter_fresh(sh: Shape, ad: Adapter, tokens: float, span: float, seq: tuple,
                  batch_tokens: float | None = None) -> list[Fresh]:
    """A LoRA's fresh products: its base products read θ, so only the adapter's and attention's at each adapted layer
    (at the lowest, whose input is registered work, the shares `LOWEST_INPUT` and `LOWEST_ATTENTION` give): x·A over
    d_in, (x·A)·B over the rank, QK^T and PV; with `batch_tokens` dy·B^T over d_out, (dy·B^T)·A^T over the rank, the
    factors' gradients over a step's tokens and attention's backward."""
    d, r, n = sh.d_model, ad.rank, ad.layers
    d_in, d_out, matrices = TARGET_DIMS[ad.targets]
    fresh_in = LOWEST_INPUT[ad.targets]
    (qk, pv), (dov, rest) = LOWEST_ATTENTION[ad.targets]
    heads, unit = d / HEAD_DIM, d * span

    def layer(macs: float, entries: float, lowest: float) -> Fresh:
        return Fresh("gemm/bf16", macs, entries, n, tokens, *seq, lowest=lowest)

    out = [layer(r * d_in * d, r * matrices, fresh_in), layer(r * d_out * d, d_out * d, 1.0),
           layer(unit, heads * span, qk), layer(unit, d, pv)]
    if batch_tokens is not None:
        above = max(0.0, n - 1)

        def whole(macs: float, entries: float, lowest: float) -> Fresh:
            return Fresh("gemm/bf16", macs * (above + lowest), entries * (above + lowest), 1.0, tokens, *seq,
                         sequence=True)

        out += [whole(r * d_out * d, r * matrices, 1.0), whole(r * d_out * d, r * d_out * d / batch_tokens, 1.0),
                whole(r * d_in * d, d_in * d, fresh_in), whole(r * d_in * d, r * d_in * d / batch_tokens, fresh_in),
                whole(unit, heads * span, dov), whole(3 * unit, 3 * d, rest / 3)]
    return out


def muon_fresh(sh: Shape, optimizer: str, steps: float, seq: tuple) -> list[Fresh]:
    """Muon's Newton-Schulz products on the momentum, both operands fresh, each step: per iteration on an a x b matrix
    X X^T (a² entries), A·A (a²) and B·X (a·b), every matrix of every layer."""
    if optimizer != "muon":
        return []
    entries = NEWTON_SCHULZ_STEPS * sh.n_layer * sh.d_model ** 2 * (4 * 3 + 3 * (2 + MLP))
    return [Fresh("gemm/bf16", newton_schulz_macs(sh), entries, 1.0, steps, *seq, sequence=True)]


@dataclass(frozen=True)
class Steps:
    """`count` optimizer steps of one kind, each summing b partial gradients of `delta_g` bits and applying them to
    `delta_t` bits of weights: b is a step's token blocks, `tokens`·w/c (its tokens over s; at most its tokens, each a
    coordinate's partial where a block holds less than a token), or `partials` where a step sums a fixed number (a
    DiLoCo sync's workers)."""
    count: float
    delta_g: float
    delta_t: float
    tokens: float = 0.0
    w: float = 0.0
    partials: float = 0.0

    def b(self, c: float) -> float:
        return self.partials or self.tokens * min(1.0, self.w / c)


# ---- the model a workload gives --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    """A workload at its dials: the token kind's inputs (`measured`), the trained or served model's forward per token
    (`w_fwd`), the fresh products and optimizer steps, and the quantities they came from (`derived`)."""
    workload: str
    dials: dict
    measured: formalism.Measured
    w_fwd: float
    fresh: tuple[Fresh, ...] = ()
    steps: tuple[Steps, ...] = ()
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
            "decoder_gates": "model: none, E stores theta' itself"} | extra


#: d's label for an E holding a LoRA (`Adapter.decoder_gates`)
LORA_DECODER = ("model: the harness's decoder for a LoRA E (prepare.rebuild_macs), r + 2 fp32 multiply-adds an adapted "
                f"entry and r (d_in + d_out) to dequantize factors below 16 bits, at {formalism.DECODER_DEFINITION}'s "
                "gates (te_core 4.2's witness)")


def _dense_t(t: float | None) -> float:
    return T_DENSE if t is None else float(t)


def standard(n_params: float = REFERENCE_PARAMS, precision: str = "bf16", seq_len: float = 1024,
             batch_tokens: float = 65_536, optimizer: str = "adamw", t: float | None = None,
             tokens_per_param: float = 20.0, vocab: float = VOCAB) -> Model:
    """Standard dense pretraining: every weight trained, each token's forward and backward in token blocks, the
    optimizer in none. u(1) is every weight as the GEMMs read it, Δ θ' at bf16 (fp32 latents for Δ_t where the GEMMs
    are narrower), no decoder. Every product is fresh."""
    sh = dense(n_params, vocab=vocab)
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
    m = _measured([part], sh.tau, ACT_BITS * sh.d_model, n_params * WEIGHT_BITS[precision], tf, 16 * n_params, labels,
                  delta_t=32 * n_params if latents else None)
    seq = (seq_len, seq_len, sh.tau)
    fresh = dense_fresh(sh, f"gemm/{precision}", tokens, seq_len, seq, batch_tokens) + muon_fresh(sh, optimizer, steps, seq)
    return Model("standard", _dials(standard, locals()), m, w_fwd(sh, precision, seq_len), tuple(fresh),
                 (Steps(steps, GRAD_BITS * n_params, m.delta_t, batch_tokens, tok.w),),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "w_gates": tok.w, "t": tf, "t_counted": tok.t,
                  "optimizer_gates_per_step": opt})


def diloco(n_params: float = REFERENCE_PARAMS, precision: str = "bf16", seq_len: float = 1024,
           batch_tokens: float = 65_536, optimizer: str = "adamw", t: float | None = None,
           inner_steps: float = 500, syncs: float = 28, sync_bits: float | None = None, workers: float = 4,
           vocab: float = VOCAB) -> Model:
    """DiLoCo-style pretraining, the record's prefix S syncs: each a phase of H inner steps of B tokens (all `workers`'
    M), standard training on each worker with its own optimizer, then an outer step from the workers' pseudo-gradients,
    each `sync_bits` b once compressed (None: 32 N, uncompressed), summed over the M workers. Its token blocks are
    standard training's, and so are its fresh products and token kind. Only where the syncs and a phase's tokens
    describe every step's weights in fewer bits than 16 N does Δ change: S·b + H·B·τ, decoded by replaying the phase
    from the synced weights (d = H·B·w), which needs the inner optimizer to restart at each sync. That decoder counts
    only where it fits a block (te_core.formalism.at's d < c), and then a phase's inner gradients never leave its
    block, which passes on only its compressed delta (Δ_grad = b, an outer step without momentum: Nesterov's would
    carry 32 N); a code that isn't per layer leaves the fresh products no layer's share of it to read."""
    sh = dense(n_params, vocab=vocab)
    tok = training(sh, precision, seq_len)
    phase = inner_steps * batch_tokens
    tokens = syncs * phase
    steps = syncs * inner_steps
    opt = optimizer_gates(sh, optimizer)
    outer = sh.params * FUSED_OPTIMIZER_MACS["_fused_sgd_"] * per_mac("elementwise/f32")
    b = 32 * n_params if sync_bits is None else float(sync_bits)
    code = syncs * b + phase * sh.tau
    compressed = code < 16 * n_params
    dg = b if compressed else delta_grad(sh, optimizer)
    part = formalism.Part(tokens * tok.w + workers * steps * opt + syncs * outer, tokens, tokens * tok.w,
                          tokens * tok.alpha, max(1, round(steps)), dg)
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
    m = _measured([part], sh.tau, ACT_BITS * sh.d_model, n_params * WEIGHT_BITS[precision], _dense_t(t),
                  code if compressed else 16 * n_params, labels, decoder=phase * tok.w if compressed else 0.0,
                  delta_t=32 * n_params if latents else None)
    seq = (seq_len, seq_len, sh.tau)
    fresh = (dense_fresh(sh, f"gemm/{precision}", tokens, seq_len, seq, batch_tokens / workers, layered=not compressed)
             + muon_fresh(sh, optimizer, workers * steps, seq))
    st = (Steps(workers * steps, GRAD_BITS * n_params, m.delta_t, batch_tokens / workers, tok.w),
          Steps(syncs, b, m.delta_t, partials=workers))
    return Model("diloco", _dials(diloco, locals()), m, w_fwd(sh, precision, seq_len), tuple(fresh), st,
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "w_gates": tok.w, "tokens": tokens,
                  "code_bits": code, "compressed": compressed, "phase_tokens": phase, "phase_gates": phase * tok.w})


def rl(n_params: float = REFERENCE_PARAMS, update: str = "full", rank: float = 16, targets: str = "all",
       layers: float | None = None, rollout_precision: str = "fp4", prompt_tokens: float = 256,
       generated_tokens: float = 768, samples_per_trained: float = 1.0, optimizer: str = "adamw",
       batch_tokens: float = 65_536, t: float | None = None, tokens: float = 1e8, vocab: float = VOCAB) -> Model:
    """RL on the policy being trained: rollouts (prefill and generation) of the current policy at `rollout_precision`,
    `samples_per_trained` of them for each sequence a learner trains on in bf16 (every position of each, prompt and
    generated). Every token is affected (the generated ids are sampled from θ'), and only the prompts' ids are inputs.
    `update` "full" trains every weight (u(1) the least of the two kinds', the rollouts' narrower weights; every product
    fresh, the rollouts' too, since they read θ'); "lora" a LoRA kept apart from the base (u(1) = Δ, its values; d the
    harness's decoder for it, `Adapter.decoder_gates`; the base products read θ). One profile serves both kinds, at
    the larger free share and the smaller u(1), as te_core's single A1 needs; `sensitivity` gives the token kind's
    bounds that count each kind's tokens apart."""
    sh = dense(n_params, vocab=vocab)
    seq = prompt_tokens + generated_tokens
    span = decode_span(prompt_tokens, generated_tokens)
    n_l = tokens
    n_r = samples_per_trained * n_l
    steps = n_l / batch_tokens
    sq = (seq, prompt_tokens, sh.tau)
    if update == "full":
        roll, learn = forward(sh, rollout_precision, span), training(sh, "bf16", seq)
        u1s = (n_params * WEIGHT_BITS[rollout_precision], 16 * n_params)
        ts = (roll.t, T_DENSE)
        delta, dg, opt = 16 * n_params, delta_grad(sh, optimizer), optimizer_gates(sh, optimizer)
        decoder, trained = 0.0, n_params
        fresh = (dense_fresh(sh, f"gemm/{rollout_precision}", n_r, span, sq)
                 + dense_fresh(sh, "gemm/bf16", n_l, seq, sq, batch_tokens) + muon_fresh(sh, optimizer, steps, sq))
    else:
        ad = Adapter.of(sh, rank, targets, layers)
        roll, learn = lora_forward(sh, ad, rollout_precision, span), lora_training(sh, ad, seq)
        delta = 16 * ad.values
        u1s = (delta, delta)
        ts = (roll.t, learn.t)
        state = "sgd" if optimizer == "sgd" else "adamw"
        dg = (GRAD_BITS + STATE_BITS[state]) * ad.values
        opt = ad.values * FUSED_OPTIMIZER_MACS[f"_fused_{state}_"] * per_mac("elementwise/f32")
        decoder, trained = ad.decoder_gates(), ad.values
        fresh = adapter_fresh(sh, ad, n_r, span, sq) + adapter_fresh(sh, ad, n_l, seq, sq, batch_tokens)
    if t is not None:
        ts = (float(t), float(t))
    tf = max(ts)
    parts = [formalism.Part(n_r * roll.w, n_r, n_r * roll.w, n_r * roll.alpha, 1, 0.0),
             formalism.Part(n_l * learn.w + steps * opt, n_l, n_l * learn.w, n_l * learn.alpha, max(1, round(steps)), dg)]
    tau = sh.tau * prompt_tokens / seq
    labels = _labels(tau_bits="model: the prompts' ids, ceil(log2 V) bits each over the sequence's tokens (generated ids "
                              "are computed)",
                     u1_bits=("model: the least of the two kinds' weights as their GEMMs read them" if update == "full"
                              else "model: the adapter's values at 16 bits, Delta"),
                     delta_bits="model: theta' at bf16, 16 N" if update == "full" else "model: the adapter at 16 bits",
                     delta_grad_bits="model: the gradient at 32 bits a value and the optimizer's state")
    if update != "full":
        labels["decoder_gates"] = LORA_DECODER
    m = _measured(parts, tau, ACT_BITS * sh.d_model, min(u1s), tf, delta, labels, decoder=decoder)
    return Model("rl", _dials(rl, locals()), m, w_fwd(sh, rollout_precision, span), tuple(fresh),
                 (Steps(steps, GRAD_BITS * trained, m.delta_t, batch_tokens, learn.w),),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "rollout_w": roll.w, "learner_w": learn.w,
                  "rollout_tokens": n_r, "learner_tokens": n_l,
                  "rollout_share": n_r * roll.w / (n_r * roll.w + n_l * learn.w), "t": tf,
                  "kinds": [{"kind": kind, "tokens": n, "u1_bits": u1, "t": tk}
                            for kind, n, u1, tk in zip(("rollout", "learner"), (n_r, n_l), u1s, ts)]})


def inference(n_params: float = REFERENCE_PARAMS, precision: str = "fp4", prompt_tokens: float = 4096,
              generated_tokens: float = 1024, t: float | None = None, tokens: float = 1e9,
              vocab: float = VOCAB) -> Model:
    """Serving an unregistered model: its weights are no part of θ, so every one is update bits, as if θ' (Δ = its
    weights, d = 0, no training: Δ_grad = 0, so Fit is running θ', Δ + s·τ <= k), and every product is fresh. σ = 1;
    t is the share of a token's work in ops that read no weight, attention and the elementwise ops."""
    sh = dense(n_params, vocab=vocab)
    span = decode_span(prompt_tokens, generated_tokens)
    tok = forward(sh, precision, span)
    tf = tok.t if t is None else float(t)
    u1 = n_params * WEIGHT_BITS[precision]
    part = formalism.Part(tokens * tok.w, tokens, tokens * tok.w, tokens * tok.alpha, 1, 0.0)
    labels = _labels(tau_bits="model: the prompt's ids, ceil(log2 V) bits each over the session's tokens",
                     u1_bits=f"model: every weight as its GEMMs read it, N x {WEIGHT_BITS[precision]:g} bits",
                     delta_bits=f"model: the served weights, N x {WEIGHT_BITS[precision]:g} bits",
                     delta_grad_bits="model: none (no training)")
    m = _measured([part], sh.tau * prompt_tokens / (prompt_tokens + generated_tokens), ACT_BITS * sh.d_model, u1, tf,
                  u1, labels)
    sq = (prompt_tokens + generated_tokens, prompt_tokens, sh.tau)
    return Model("inference", _dials(inference, locals()), m, w_fwd(sh, precision, span),
                 tuple(dense_fresh(sh, f"gemm/{precision}", tokens, span, sq)), (),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "span": span, "t_counted": tok.t})


def lora(n_params: float = REFERENCE_PARAMS, rank: float = 8, targets: str = "all", layers: float | None = None,
         adapter_bits: float = 16, optimizer: str = "adamw", seq_len: float = 1024, batch_tokens: float = 65_536,
         t: float | None = None, tokens: float = 1e8, vocab: float = VOCAB) -> Model:
    """Fine-tuning a registered model with a LoRA on `targets` of its top `layers` layers (None: all): Δ the adapter's
    values at `adapter_bits` (fp32 latents for Δ_t below 16), d the harness's decoder for it (`Adapter.decoder_gates`),
    Δ_grad their gradient at 32 bits and the optimizer's state, u(1) = Δ, and t the share of the affected work that
    reads no adapter value. The base products read θ, so the fresh products are the adapter's and attention's above
    the lowest adapted layer (`adapter_fresh`)."""
    sh = dense(n_params, vocab=vocab)
    ad = Adapter.of(sh, rank, targets, layers)
    tok = lora_training(sh, ad, seq_len)
    tf = tok.t if t is None else float(t)
    steps = tokens / batch_tokens
    opt = ad.values * FUSED_OPTIMIZER_MACS[f"_fused_{optimizer}_"] * per_mac("elementwise/f32")
    delta = ad.values * adapter_bits
    dg = (GRAD_BITS + STATE_BITS[optimizer]) * ad.values
    part = formalism.Part(tokens * tok.w + steps * opt, tokens, tokens * tok.w, tokens * tok.alpha, max(1, round(steps)),
                          dg)
    labels = _labels(u1_bits="model: the adapter's values, Delta",
                     delta_bits=f"model: the adapter's values at {adapter_bits:g} bits",
                     delta_grad_bits=f"model: their gradient at 32 bits and {optimizer}'s state",
                     decoder_gates=LORA_DECODER)
    latents = adapter_bits < 16
    if latents:
        labels["delta_t_bits"] = "model: fp32 latents, 32 bits a value (a straight-through run)"
    m = _measured([part], sh.tau, ACT_BITS * sh.d_model, delta, tf, delta, labels, decoder=ad.decoder_gates(latents),
                  delta_t=32 * ad.values if latents else None)
    seq = (seq_len, seq_len, sh.tau)
    return Model("lora", _dials(lora, locals()), m, w_fwd(sh, "bf16", seq_len),
                 tuple(adapter_fresh(sh, ad, tokens, seq_len, seq, batch_tokens)),
                 (Steps(steps, GRAD_BITS * ad.values, m.delta_t, batch_tokens, tok.w),),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "adapter_values": ad.values, "w_gates": tok.w,
                  "t_counted": tok.t})


def distill(student_params: float = REFERENCE_PARAMS, teacher_params: float = 10 * REFERENCE_PARAMS,
            teacher_share: float = 1.0, teacher_precision: str = "bf16", seq_len: float = 1024,
            batch_tokens: float = 65_536, optimizer: str = "adamw", t: float | None = None,
            tokens_per_param: float = 20.0, vocab: float = VOCAB) -> Model:
    """Distillation from a registered teacher: the student's dense training, and the teacher's forward on
    `teacher_share` of its tokens in the same token blocks. The teacher reads θ and the tokens alone, so its work is
    outside A and none of its products is fresh: it raises w and leaves the affected count, u, t, a and the fresh
    products the student's."""
    sh, te = dense(student_params, vocab=vocab), dense(teacher_params, vocab=vocab)
    stu, teach = training(sh, "bf16", seq_len), forward(te, teacher_precision, seq_len)
    tok = Token(_add(stu.macs, {k: teacher_share * v for k, v in teach.macs.items()}), stu.affected, stu.reading)
    tokens = tokens_per_param * student_params
    steps = tokens / batch_tokens
    opt = optimizer_gates(sh, optimizer)
    part = formalism.Part(tokens * tok.w + steps * opt, tokens, tokens * tok.w, tokens * tok.alpha, max(1, round(steps)),
                          delta_grad(sh, optimizer))
    labels = _labels(u1_bits="model: the student's weights at bf16, 16 N_s", delta_bits="model: the student at bf16, 16 N_s",
                     delta_grad_bits=f"model: the student's gradient at 32 bits and {optimizer}'s state")
    m = _measured([part], sh.tau, ACT_BITS * sh.d_model, 16 * student_params, _dense_t(t), 16 * student_params, labels)
    seq = (seq_len, seq_len, sh.tau)
    fresh = dense_fresh(sh, "gemm/bf16", tokens, seq_len, seq, batch_tokens) + muon_fresh(sh, optimizer, steps, seq)
    return Model("distill", _dials(distill, locals()), m, w_fwd(sh, "bf16", seq_len), tuple(fresh),
                 (Steps(steps, GRAD_BITS * student_params, m.delta_t, batch_tokens, tok.w),),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "student_w": stu.w, "teacher_w": teach.w,
                  "student_share": stu.w / tok.w})


def _dials(fn, scope: dict) -> dict:
    return {name: scope[name] for name in inspect.signature(fn).parameters}


# ---- ρ, the kinds' terms and the console's columns ---------------------------------------------------------------------

def columns(m: Model) -> dict:
    """The console's columns for the model's token kind: `C_gates`, te_core.algorithm_policy's `INPUTS` (u and u_in
    capped at Δ where d is 0; `page_rho` caps them wherever d < c, as formalism.at reads them), `w_fit_gates` and
    `alpha_ratio`."""
    me = m.measured
    return ({"C_gates": me.gates} | dict(zip(algorithm_policy.INPUTS, algorithm_policy.inputs(me, m.w_fwd)))
            | {"w_fit_gates": me.w_fit, "alpha_ratio": me.alpha_ratio})


def update_bits(me: formalism.Measured, c: float) -> float:
    """The fewest bits a block reads to hold the whole update: u(1), or the code Δ where its decoder fits a block."""
    d = me.decoder_gates if me.decoder_gates is not None else math.inf
    return min(me.u1_bits, me.delta_bits) if d < c and me.delta_bits is not None else me.u1_bits


def menu(m: Model) -> dict:
    """The other kinds' inputs, as `rho` reads them: the fresh products (their widths and gates a MAC by kind are the
    reference's `widths` and `gates_per_mac`), their share of |C| before any escapes, and the optimizer steps."""
    me = m.measured
    return {"u1_bits": me.u1_bits, "delta_bits": me.delta_bits, "decoder_gates": me.decoder_gates,
            "fresh": [asdict(f) for f in m.fresh],
            "fresh_share": sum(f.total * per_mac(f.kind) for f in m.fresh) / me.gates,
            "steps": [asdict(s) for s in m.steps]}


def _finite(v: float) -> float | None:
    return None if math.isinf(v) else v


def rho(m: Model, c: float, k: float) -> dict:
    """ρ at (c, k): every kind's term (`KINDS`; None infinite, absent where the kind counts nothing), their max over 1
    (`bound`) and the binding kind with what it rests on; ρ is Fit's c/(c − d) where Fit holds and the menu's bound
    doesn't exceed it, else the bound. Beside them: te_core.formalism.at's ρ (the token kind with Fit, reading B),
    Fit's thresholds (the most Δ its compute block admits, k − (c/w_fit)·τ, and the most Δ_grad its reduction block
    admits, k/2), each step kind's b, and the shares of |C| in the fresh products counted and charged."""
    me = m.measured
    v = formalism.at(me, c, k)
    fit = v["fit"]
    d = me.decoder_gates if me.decoder_gates is not None else math.inf
    ub = update_bits(me, c)
    counted = [f.counted(ub, k) for f in m.fresh]
    charged = [fresh_blocks(n, e, k, WIDTHS[f.kind]) for f, (n, e) in zip(m.fresh, counted)]
    terms = {}
    if sum(charged):
        terms["fresh_products"] = c * sum(charged) / me.gates
    terms["token"] = v["slowdown"]
    bs = [s.b(c) for s in m.steps]
    load = [(s, b) for s, b in zip(m.steps, bs) if b > 1]
    if not fit and load:
        terms["reduction"] = c / me.gates * sum(s.count * (b - 1) * s.delta_g for s, b in load) / k
        terms["update"] = c / me.gates * sum(s.count * (s.delta_t + s.delta_g) for s, _ in load) / k
    if 0 < d < c and me.delta_bits < me.u1_bits and all(p["fit"]["compute"] for p in v["parts"]):
        terms["decoding"] = c / (c - d)
    bound = max(1.0, *terms.values())
    binding = max(terms, key=terms.get) if bound > 1 else None
    conflict = fit and bound > v["rho_fit"] * (1 + 1e-9)
    by_fit = fit and not conflict
    fails = sorted({clause for p in v["parts"] for clause, ok in p["fit"].items() if not ok})
    rests = (FIT if by_fit else KINDS[binding] if binding else
             f"nothing: no kind's term exceeds 1 and Fit fails ({', '.join(fails)}), so rho >= 1 is all the menu gives")
    out = {"rho": _finite(v["rho_fit"] if by_fit else bound), "basis": "fit" if by_fit else binding or "none",
           "binding_kind": binding, "rests_on": rests,
           "bound": _finite(bound), "terms": {kind: _finite(x) for kind, x in terms.items()},
           "te_core": {"rho": _finite(v["rho"]), "basis": "fit" if fit else "target"},
           "fit": fit, "fit_fails": fails, "rho_fit": _finite(v["rho_fit"]), "s": v["s"], "x0": v["x0"],
           "s_star": v["s_star"], "b": bs,
           "fresh_share": sum(n * per_mac(f.kind) for f, (n, _) in zip(m.fresh, counted)) / me.gates,
           "fresh_charged_share": sum(b * fresh_cap(k, WIDTHS[f.kind]) * per_mac(f.kind)
                                      for f, b in zip(m.fresh, charged)) / me.gates,
           "fit_delta_bits": k - c / me.w_fit * me.tau_bits, "fit_delta_grad_bits": k / 2}
    if conflict:
        out["conflict"] = (f"Fit holds, but the {binding} kind's term exceeds its c/(c - d): Fit's compute block, s "
                           "tokens computed from their inputs, can't compute their attention without their sequence")
    return out


def sensitivity(m: Model, c: float, k: float) -> dict:
    """The token kind's bounds beside te_core's for a record whose token blocks differ (RL's two kinds), from the
    counting of tokens' worth that s* bounds: every token with affected work leaves (1 − x0) of a token's worth to
    blocks that do at most s* each, so ρ >= (c/|C|)·n_A(1 − x0)/s* (`tokens_worth`, te_core's x0 and s*: its form
    without ᾱ/α_max), and with each kind's own profile ρ >= (c/|C|)·Σ_j n_j(1 − x0_j)/s*_j (`per_kind`). Each is
    max(1, ·) where Fit fails."""
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
    t literal; `RECORD_SOURCE`), on which page_rho gives the record's token kind (te_core's ρ) at each policy. te_core
    4.2 refuses the records' traces, made before its capture listed read-backs, until the runs are traced again."""
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


RECORD_SOURCE = ("te_core 4.1's console columns for the record (algorithm_policy, quantities measured, t literal; 4.2 "
                 "refuses the trace, made before its capture listed read-backs, until the run is traced again); "
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
                         "sets the optimizer's share, 1 - sigma, and b, a step's token blocks; the pilot's 64 x 1,024"),
    "optimizer": Dial("optimizer", "rule", "adamw", ("adamw", "muon", "sgd"),
                      "sets Delta_grad (the state), the optimizer's gates and Muon's fresh Newton-Schulz products"),
    "t_dense": Dial("t", "share of a token's affected work", None, (0.0, 0.9),
                    f"A1's free share, the token kind's: None is te_core's literal t on the 45M trace, {T_DENSE}; 0 is "
                    "evenly spread, which overcharges a block"),
    "tokens_per_param": Dial("tokens_per_param", "tokens / N", 20.0, (1.0, 2000.0),
                             "sets |C| only (Chinchilla's 20 by default)"),
    "vocab": Dial("vocab", "ids", VOCAB, (1000, 262_144),
                  "the vocabulary: the embedding's size and tau's bits an id; 32,000 is the v3 starts' (s-t20)"),
}

WORKLOADS = {w.name: w for w in (
    Workload("standard", "Standard dense pretraining",
             "Every weight of a dense decoder trained by backpropagation, its optimizer outside the token blocks.",
             standard,
             (COMMON["n_params"], COMMON["precision"], COMMON["seq_len"], COMMON["batch_tokens"], COMMON["optimizer"],
              COMMON["t_dense"], COMMON["tokens_per_param"], COMMON["vocab"]),
             (("default", {}), ("FP8 GEMMs", {"precision": "e4m3"}), ("7B", {"n_params": 7e9})),
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
              Dial("batch_tokens", "tokens an inner step", 65_536, (256, 1e7), "B, an inner step's tokens, all workers'"),
              COMMON["optimizer"], COMMON["t_dense"],
              Dial("inner_steps", "steps", 500, (1, 10_000), "H, the inner steps between syncs"),
              Dial("syncs", "syncs", 28, (1, 1e5),
                   "S, the record's outer steps so far; its tokens are S H B (28: 20 a parameter at 45M)"),
              Dial("sync_bits", "bits a sync", None, (1e2, 1.5e9),
                   "b, one worker's compressed outer delta; None is 32 N, uncompressed"),
              Dial("workers", "workers", 4, (1, 1024),
                   "M, the workers each inner step's tokens split over and each sync sums (the island's 4)"),
              COMMON["vocab"]),
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
              Dial("rollout_precision", "GEMM format", "fp4", ("bf16", "e4m3", "fp4"),
                   "the rollouts' weight GEMMs (fp4: W4A4)"),
              Dial("prompt_tokens", "tokens", 256, (16, 32768), "a rollout's prompt, the only inputs"),
              Dial("generated_tokens", "tokens", 768, (16, 32768), "a rollout's generated tokens"),
              Dial("samples_per_trained", "sequences", 1.0, (0.01, 16.0),
                   "rollouts generated for each one trained on (1: each once; 4: one in four kept; 0.01: rounds every "
                   "100 steps, as the island's)"),
              COMMON["optimizer"],
              Dial("batch_tokens", "learner tokens a step", 65_536, (1e4, 1e7),
                   "sets the optimizer's share, 1 - sigma, and b, a step's token blocks"),
              Dial("t", "share of a token's affected work", None, (0.0, 0.999),
                   "A1's free share, the token kind's: None is the larger of the two kinds' (the learner's te_core "
                   "literal t, or a LoRA's modelled one); 0 is evenly spread"),
              Dial("tokens", "learner tokens", 1e8, (1e6, 1e11), "sets |C| only"), COMMON["vocab"]),
             (("default", {}), ("LoRA rank 16", {"update": "lora"}),
              ("LoRA in one layer", {"update": "lora", "rank": 1, "targets": "q", "layers": 1}))),
    Workload("inference", "Inference on an unregistered model",
             "Serving a model whose weights are not registered, so all of them are update bits.",
             inference,
             (Dial("n_params", "parameters", REFERENCE_PARAMS, N_RANGE, "the served model"),
              Dial("precision", "GEMM format", "fp4", ("bf16", "e4m3", "fp4"), "the served weights' format (fp4: W4A4)"),
              Dial("prompt_tokens", "tokens", 4096, (16, 131072), "a session's prompt, the only inputs"),
              Dial("generated_tokens", "tokens", 1024, (1, 32768), "a session's generated tokens"),
              Dial("t", "share of a token's work", None, (0.0, 0.95),
                   "A1's free share, the token kind's: None is attention's modelled share; 0 is evenly spread"),
              Dial("tokens", "tokens served", 1e9, (1e6, 1e13), "sets |C| only"), COMMON["vocab"]),
             (("default", {}), ("bf16", {"precision": "bf16"}), ("7B", {"n_params": 7e9}))),
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
                   "A1's free share, the token kind's: None is the modelled share reading no adapter value; 0 is "
                   "evenly spread"),
              Dial("tokens", "tokens", 1e8, (1e6, 1e11), "sets |C| only"), COMMON["vocab"]),
             (("default", {}), ("top layer", {"layers": 1}), ("MLP, top layer", {"targets": "mlp", "layers": 1}),
              ("rank 1 on W_q", {"rank": 1, "targets": "q"}),
              ("rank 1 on W_q, one layer", {"rank": 1, "targets": "q", "layers": 1})),
             (Check("45M sparse/lora-alloc", {"rank": 4, "targets": "qv"}, "r20261102-132959-286b@1720",
                    {"w_gates": 1.66764e11, "sigma": 0.999972, "tau_bits": 16.0156, "a_bits": 8192,
                     "delta_bits": 788712, "delta_t_bits": 788712, "d_gates": 0, "delta_grad_bits": 4657152,
                     "w_fit_gates": 1.6639e11, "alpha_ratio": 0.998823,
                     "u_points": "0:0;0.999235:0;0.999235:788712;1:788712",
                     "u_in_points": "0:0;0.00105837:32768;0.00315535:65536;0.100031:262144;1:788712"},
                    "48,512 trained values against the model's 49,152; u_in front-loaded (a token's forward reads every "
                    "adapter value in dependency order before most of its work), where the model's A1' is even; d 0, "
                    "te_core 4.1's stub, where the model's is the harness's decoder (2.4e11 gates, 0.07% of c)"),)),
    Workload("distill", "Distillation from a registered teacher",
             "A student trained on a registered teacher's outputs, the teacher's forward in each token block.",
             distill,
             (Dial("student_params", "parameters", REFERENCE_PARAMS, N_RANGE, "the trained student"),
              Dial("teacher_params", "parameters", 10 * REFERENCE_PARAMS, (1e7, 1e12), "the registered teacher"),
              Dial("teacher_share", "share of tokens", 1.0, (0.0, 1.0), "the student tokens the teacher runs on"),
              Dial("teacher_precision", "GEMM format", "bf16", ("bf16", "e4m3", "fp4"), "the teacher's weight GEMMs"),
              COMMON["seq_len"], COMMON["batch_tokens"], COMMON["optimizer"], COMMON["t_dense"],
              COMMON["tokens_per_param"], COMMON["vocab"]),
             (("default", {}), ("no teacher", {"teacher_share": 0.0}), ("100x teacher", {"teacher_params": 100 * 45_029_888}))),
)}


def model(name: str, **dials) -> Model:
    return WORKLOADS[name].fn(**dials)


def point(m: Model) -> dict:
    """A model's reference point: its dials, the token kind's columns and what they're estimated from, the other
    kinds' inputs (`menu`), ρ at each of `POLICIES`, the token kind's sensitivity bounds where its token blocks differ,
    and what it derived."""
    out = {"dials": m.dials, "inputs": columns(m), "estimated": algorithm_policy.estimated(m.measured), "menu": menu(m),
           "rho": {name: rho(m, c, k) for name, (c, k) in POLICIES.items()}, "derived": m.derived}
    if m.derived.get("kinds"):
        out["sensitivity"] = {name: sensitivity(m, c, k) for name, (c, k) in POLICIES.items()}
    return out


def _check(ck: Check, workload: str) -> dict:
    """A pilot record beside its model (`Check`): the token kind's ρ (te_core's) by page_rho on the record's columns at
    each policy, and the model's at the check's dials."""
    m = model(workload, **ck.dials)
    return {"record": ck.record, "source": RECORD_SOURCE, "dials": m.dials, "differs": ck.differs,
            "record_inputs": ck.columns, "model_inputs": columns(m),
            "rho": {name: {"record": _finite(algorithm_policy.page_rho(ck.columns, c, k)[0]),
                           "model": rho(m, c, k)["te_core"]["rho"]} for name, (c, k) in POLICIES.items()}}


def reference() -> dict:
    """Every workload's reference points and checks at `POLICIES` (module docstring): `workloads.json`."""
    out = {"id": "te_core.workloads", "te_core": VERSION, "reading": "B", "spec": NOTE,
           "policies": {name: {"c": c, "k": k, "what": POLICY_NOTES[name]} for name, (c, k) in POLICIES.items()},
           "kinds": KINDS, "fit": FIT, "widths": {kind: list(w) for kind, w in WIDTHS.items()},
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
    return "inf" if v is None else f"{v:.3g}"


def table(ref: dict) -> str:
    """Each policy's table: every point's term of each kind (n/a where the kind counts nothing), the Fit clauses that
    fail, ρ and its basis."""
    lines = []
    heads = [f"{kind} ({ref['reading']})" if kind == "token" else kind for kind in KINDS]
    for q in POLICIES:
        lines += [f"{q}: {ref['policies'][q]['what']}", "",
                  "| workload | point | " + " | ".join(heads) + " | Fit fails | rho | basis |",
                  "|---|---|" + "---|" * (len(KINDS) + 3)]
        for name, w in ref["workloads"].items():
            for pname, p in w["points"].items():
                r = p["rho"][q]
                cells = [_fmt(r["terms"][kind]) if kind in r["terms"] else "n/a" for kind in KINDS]
                fails = ", ".join(r["fit_fails"]) or "none"
                lines.append(f"| {name} | {pname} | " + " | ".join(cells)
                             + f" | {fails} | {_fmt(r['rho'])} | {r['basis']} |")
        lines.append("")
    lines += ["checks, the token kind (te_core's rho):", "", "| workload | check | " + " | ".join(POLICIES) + " |",
              "|---|---|" + "---|" * len(POLICIES)]
    for name, w in ref["workloads"].items():
        for cname, ck in w["checks"].items():
            cells = [f"{_fmt(ck['rho'][q]['record'])} record, {_fmt(ck['rho'][q]['model'])} model" for q in POLICIES]
            lines.append(f"| {name} | {cname} | " + " | ".join(cells) + " |")
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
