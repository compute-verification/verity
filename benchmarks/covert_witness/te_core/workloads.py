"""Covert workloads under the policy, each a model: from a workload's dials to the inputs of the one bound's kinds, and
ρ at a policy (c, k), for the console's *Standard training under the policy* plot.

    python -m te_core.workloads [--json OUT]    # every workload's reference points at the console's policy

A workload (`WORKLOADS`) is a function from its dials (`Dial`: name, unit, default, range, why) to a `Model`: the inputs
of te_core.formalism's one bound as a `Measured` (|C|, w, σ, τ, a, the profiles u and u_in, Δ, d, Δ_grad and Δ_t, and
each part's partial gradient and gradient cut, each labelled with where it comes from), its products of two fresh
operands (`Fresh`), w_fwd, and the most update bits one of its ops reads (`op_update_bits`). `rho` gives every kind's
term at a policy (c, k), their max, the binding kind and the sensitivities and target beside them; `columns` gives the
target's inputs as the console's columns (te_core.algorithm_policy's `INPUTS`, with `w_fit_gates`, `alpha_ratio` and
`C_gates`), on which `algorithm_policy.page_rho` gives reading B's token form, and `menu` the other inputs, the fresh
products' among them. `reference` evaluates each workload's reference points (`Workload.points`) at `POLICIES` and
its `Check`s, measured records beside the model (the 45M pilot's, and the standard-workloads island's rows from s-t20
and m-t20 in `RECORDS`), each made a model of its own
(`recorded`) so that every kind's term is on both; `input_status` gives each input's source, "measured: " and the
record where one at the point's dials measures the model's value, else "assumed: " and why. `workloads.json` is its
output, which the console's tests pin. The spec, each mapping with its reasons, is the note `NOTE` names.

**The one bound** (the coordinator's 06:51Z ruling; Lean's `Counting` and `Slowdown`): ρ ≥ the max over a fixed menu
of kinds K of `kindBound`, (|B_K|/|C|)·c/cap_K(k), cap_K(k) the most of K's work one block does reading at most k bits
beyond θ. Only kinds whose terms are lower bounds on their own go into the max (the coordinator's 08:26Z and 10:21Z
rulings; te_core 4.5's `formalism.KINDS`): where Fit holds its construction gives c/(c − d); else ρ is max(1, the
measured terms), labelled with its binding kind (none where no measured term exceeds 1) and partial where a kind is
unmeasured, and null only where no kind is measured, never 1 (the coordinator's 09:46Z and 11:18Z rulings, as
te_core.formalism.at gives it). `basis` names fit, the binding kind, none or unmeasured, and `RESTS_ON[basis]` what it
rests on. The kinds (`VALID`):

- **fresh products** (`Fresh`): products whose two operands both depend on θ', priced by te_core.formalism's
  `fresh_products`, te_core.io_bound's `bound` with Λ proved (the Loomis-Whitney cap per block, the larger of its root
  form, (k/3)^1.5/sqrt(b_A b_B b_C) on chains entering as partial sums plus (k/2)/sqrt(b_A b_B) per sqrt of the chains
  a block starts, and its linear form, the cap's concave envelope, times Λ for blocks doing several products at once),
  on the model's products as io_bound reads a captured step's fresh rows (`products`, `Measured.products`): each GEMM
  by the standard algorithm at its widths as passed (`WIDTHS`), its Definition's gates a MAC and a gadget, attention's
  products uncounted (io_bound counts none). io_bound's cap has the chain order's constant (its premise P-chain), and
  the tree order's applies where the order isn't given (the coordinator's 11:18Z ruling); a model gives none
  (`ORDER`), as no capture does, so the term is unmeasured until io_bound has the tree constant, with io_bound's chain
  figures beside it. Where measured, the term is io_bound's proved figure without P-cell (`FRESH_FIGURE`), which needs
  no premise on how a partition splits a multiplier's gates (the coordinator's 06:57Z obligation 1). The chain
  figures, without P-cell and with it (every block holding whole GEMM gadgets), are sensitivities beside it
  (`fresh_chain` and `fresh_cell`: the second 19 times the first on the dense default, 3 times on the LoRA's). It
  rests on no A1. A product with an operand in θ (a frozen weight, a registered teacher's) isn't fresh, nor is the
  lowest layer's attention input projection's (its input is gathered embedding rows, which io_bound's P-width prices
  below their width), nor Newton-Schulz's products of an array with itself; and a block computes a product's operands
  itself where it can read the update up to the product's layer and the input ids up to its token (a forward) or of
  its whole sequence (a backward, whose gradient reads every later token): those products escape (`Fresh.counted`),
  since io_bound's P-width needs every operand paid at its width.
- **token** (A1 at reading A): te_core 4.5's packer over a trace's `entries` and its op log's matrix-product and
  weight-gradient tiles (te_core.tiles), finite at every k. A model has neither, so the term is unmeasured, never ∞
  or 1 (`reading_a`, its verdict beside each point).
- **decoding** (assumption (D), with A1 at reading A): the same packer with a block reading the update code running
  the whole decoder; unmeasured here for the same reason.

Beside ρ, never in its max (`SENSITIVITIES`, each with what it rests on, and `TARGET`): reduction and update,
(c/|C|)·Σ steps·(b − 1)·partial/k and (c/|C|)·Σ steps·γ/k over each part whose step's token work fills b = ⌈N·w/c⌉ ≥ 2
blocks (assumptions (R) and (U), which hold only for partitions split by tokens; the fresh-products kind covers the
weight-gradient product dW = δ^T X, whose partial sums across token blocks are the reduction); reading B's token form
and decoding; reading A's per-gate figure, which needs a trace; io_bound's chain figures; and the target,
te_core.formalism.published at reading B, Fit's c/(c − d) or Slowdown's token form, which a row without a trace
publishes: a target, not a bound. A DiLoCo sync has no tokens, so no term prices its workers' pseudo-gradients
(`diloco`).

The conventions every workload shares:

- **Policy** (`POLICIES`): the console's only, the one the dashboard reports rows at (its 10:23Z ruling): `default`,
  te_core.policy's `CONSOLE`, one full session of the 45M reference a block, 4,096 tokens in and 1,024 out, attention
  and the LM head included (`policy.session_gates`, 5.36682e14 gates, 1.5739 times the 5,120 x 2 x 45e6 x 740 that
  left attention out), and k = 65,536 bits. It holds c/w = 2,294 of the model's 45M training tokens (w = 2.3396e11
  gates, 7.0 FLOPs a parameter, 3.51 session tokens' 2N); the old c held 1,457.5, where TRN-40's 1,250 took a
  training token as 4.1 session tokens' 2N, 17% above the count.
- **Shapes** (`dense`): a dense decoder of N parameters with the reference model's d/L and SwiGLU width, d solved from N
  and the vocabulary (`vocab`: the 45M reference's 50,304, s-t20's 32,000), τ its ⌈log2 V⌉ bits an id.
- **Work**: a token's MACs at te_core.gates' prices, its weight GEMMs at the workload's precision and its attention in
  bf16 at te_core.flops' count (the whole span, so a training token at context T has 2·L·d·T), with `ELEMENTWISE` bf16
  elementwise MACs per MAC. Attention's QK^T and dO·V^T sum over a head's `HEAD_DIM` values, its other products over
  the span. a is the bf16 residual stream, 16·d, or a LoRA's rank outputs at its lowest adapted layer where narrower
  (`Adapter.a_bits`). A partial gradient and the gradient cut are 32 bits a trained value (`GRAD_BITS`), every
  worker's in a DiLoCo step. A served sequence of p prompt and g generated tokens runs its forward at p + g − 1
  positions and its head at the g it samples (`served`), as the island's generation rows count their MACs.
- **Full-weight training** is calibrated on the records: u(1) is every weight as a step's ops read it, the matrices
  at the GEMMs' width and the embedding and norms in fp32 (`Shape.weight_bits`), and Δ = Δ_t is θ' in fp32, 32 N,
  each the records' exactly (Δ less its code's 5 KB header), as Δ_grad is.
- **The decoder** d, by te_core 4.2's rule: 0 where E stores θ' itself (every full-weight workload), the harness's
  decoder for an E holding a LoRA (`Adapter.decoder_gates`), and compressed DiLoCo's replay of a phase.
- **A1's reading** (the dial `t`, which only the target and reading B's decoding read; reading A's token kind reads a
  trace): u is 0 up to the free share t and x·u(1) beyond it, u_in x·u(1) (`profile`: `formalism.with_tail` of
  `formalism.even`). Dense training's default t is `T_DENSE`, te_core's
  literal t measured on the 45M pilot's trace; elsewhere t is the modelled share of a token's affected work that reads
  no update value (`Token.t`). t = 0, the evenly spread reading, overcharges a block (the coordinator, 07:05Z).
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import astuple, dataclass, field, fields, replace
from pathlib import Path

from . import VERSION, algorithm_policy, formalism, policy
from .gates import PRICES, gates, per_mac
from .opnames import FUSED_OPTIMIZER_MACS

NOTE = "note:training-20261103T0600Z-draft-workload-models"
REFERENCE = Path(__file__).with_name("workloads.json")
VOCAB = 50_304
#: the console's served model: the 45M dense starts' parameters (starts_store.json's s50 and s100)
REFERENCE_PARAMS = policy.REFERENCE_PARAMS
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
#: the records the model is calibrated on: the 45M pilot's AdamW and Muon records (te_core 4.1's traces) and the
#: standard-workloads island's rows (te_core 4.4) from s-t20 (d 512, 6 layers, V = 32,000): dense Muon, DiLoCo, LoRAs
#: of rank 16 on attention and on every matrix, and distillation into a student of d 256 and 6 layers
PILOT_ADAMW, PILOT_MUON = "r20261102-140347-a1c2", "r20261102-134356-ee8f"
ISLAND_STANDARD, ISLAND_DILOCO, ISLAND_DISTILL = "r20261103-053241-99bf", "r20261103-064814-40d6", "r20261103-064849-6854"
ISLAND_LORA_ATTN, ISLAND_LORA_ALL = "r20261103-053921-7400", "r20261103-053912-3133"
#: the island's generation rows (te_core 4.4, untraced): RL rollouts of s-t20 with a rank-16 LoRA on every matrix and
#: fully fine-tuned, and an unregistered model of its shape served, each NVFP4 W4A4 rounds of 64 sequences
ISLAND_RL_LORA, ISLAND_RL_FULL, ISLAND_INFERENCE = ("r20261103-073616-b950", "r20261103-073633-f483",
                                                    "r20261103-073641-d1f4")
#: the island's rows from m-t20 (its 60M start): dense Muon and distillation into a student of d 384 and 8 layers
#: (traced), and fully fine-tuned RL rollouts and an unregistered model of its shape served (untraced)
ISLAND_STANDARD_60M, ISLAND_DISTILL_60M = "r20261103-091929-b455", "r20261103-091446-973e"
ISLAND_RL_FULL_60M, ISLAND_INFERENCE_60M = "r20261103-091516-fb8d", "r20261103-091453-e42e"
#: m-t20's parameters and width, as te_core.policy states its shape: d 768 and 8 layers, its SwiGLU 2,048 (8/3 d),
#: V = 32,000
M_T20_PARAMS = 81_212_160
M_T20 = policy.SHAPES[M_T20_PARAMS]["d_model"]
#: te_core 4.1's literal t on the 45M pilot's traced full-training step (every ref-optim record's): the share of a
#: training token's affected work in ops that read no update value
T_DENSE = 0.692646
NEWTON_SCHULZ_STEPS = 5
#: optimizer state bits per trained value (Muon's is its momentum on matrices, AdamW's on everything else)
STATE_BITS = {"adamw": 64, "sgd": 0, "muon": 32}
#: the lf family's attention head, d_model / n_head at every size (s 512/8, m 768/12, l 1024/16)
HEAD_DIM = 64
#: a layer's matrices, each (d_in, d_out) in d_models
MATRICES = {"q": (1.0, 1.0), "k": (1.0, 1.0), "v": (1.0, 1.0), "o": (1.0, 1.0), "gate": (1.0, MLP), "up": (1.0, MLP),
            "down": (MLP, 1.0)}
#: each LoRA target set's matrices in a layer
TARGET_MATRICES = {"q": ("q",), "qv": ("q", "v"), "attn": ("q", "k", "v", "o"), "mlp": ("gate", "up", "down"),
                   "all": tuple(MATRICES)}
#: each LoRA target set's (Σ d_in, Σ d_out) in d_models, and how many matrices
TARGET_DIMS = {name: (sum(MATRICES[x][0] for x in ms), sum(MATRICES[x][1] for x in ms), len(ms))
               for name, ms in TARGET_MATRICES.items()}
#: (d_in + d_out) per rank per layer of each LoRA target set, in d_models
TARGETS = {name: d_in + d_out for name, (d_in, d_out, _) in TARGET_DIMS.items()}
#: the entries of the matrices each LoRA target set adapts, per layer, in d_model²
TARGET_ENTRIES = {name: sum(MATRICES[x][0] * MATRICES[x][1] for x in ms) for name, ms in TARGET_MATRICES.items()}
#: each LoRA target set's adapted matrices reading its lowest adapted layer's first input (W_q, W_k and W_v read the
#: layer's; the MLP's gate and up the norm's after attention), whose rank outputs are a token's first affected state
FIRST_ADAPTED = {name: sum(x in ("q", "k", "v") for x in ms) or sum(x in ("gate", "up") for x in ms)
                 for name, ms in TARGET_MATRICES.items()}
#: the widest dimension of each LoRA target set's matrices, in d_models (a factor is it by the rank)
TARGET_WIDEST = {name: max(max(MATRICES[x]) for x in ms) for name, ms in TARGET_MATRICES.items()}
#: te_core.io_bound's commit in this tree, whose `bound` prices the fresh products (te_core 4.6's, by
#: te_core.formalism.fresh_products)
IO_BOUND = "cursor/te-io-bound-4207 at e4fc638622"
#: the order a model's products are summed in (te_core.formalism.ORDERS; None: not given): a model, like every capture
#: so far, doesn't give the kernels' order, so the tree constant applies (coordinator 11:18Z)
ORDER = None
#: io_bound's proved figures: the fresh-products kind's term, with no premise on the partition, and the one under
#: P-cell beside it
FRESH_FIGURE, FRESH_CELL = "rho_no_cell", "rho"
#: why reading A's kinds are unmeasured here
NO_TRACE = ("unmeasured: te_core 4.5's packer reads a trace's entries and op log (its runs and matrix-product tiles), "
            "which a model hasn't and the records here don't carry")
#: the one bound's kinds (te_core.formalism.KINDS), each a lower bound on its own (coordinator 08:26Z and 10:21Z), with
#: what its term rests on
VALID = {
    "fresh_products": "theorem for this circuit: te_core.io_bound's Loomis-Whitney bound with Λ proved "
                      f"({IO_BOUND}, by te_core.formalism.fresh_products) on the model's products of two fresh "
                      "operands, each operand and partial sum at its passed width, without P-cell, so with no "
                      "premise on the partition (a block may split a multiplier's gates; coordinator 06:57Z, "
                      "obligation 1), given P-width (a premise on a reference circuit's wiring, false on a declared "
                      "one), P-share and the standard algorithm, at the constant of the order the products are summed "
                      "in, the tree's where it isn't given (coordinator 11:18Z), as on a model; attention's products "
                      "uncounted",
    "token": f"{formalism.RESTS_ON['token']}; {NO_TRACE}",
    "decoding": f"{formalism.RESTS_ON['decoding']}; {NO_TRACE}",
}
#: io_bound's chain figures beside ρ, sensitivities: the term under P-chain, and under P-cell too
CHAIN = ("sensitivity, premise P-chain (te_core.io_bound): the products summed in one chain an output entry, the "
         "constant io_bound has, which no model or capture gives")
P_CELL = ("sensitivity, premises P-chain and P-cell (te_core.io_bound): every block holds whole GEMM gadgets, the cap "
          "for word-level MACs, which a partition splitting a multiplier's gates escapes (coordinator 06:57Z, "
          "obligation 1); te_core.formalism.fresh_products' figure")
#: what the bound publishes beside ρ, never in its max (te_core.formalism.SENSITIVITIES), and io_bound's chain figures
SENSITIVITIES = dict(formalism.SENSITIVITIES) | {"fresh_chain": CHAIN, "fresh_cell": P_CELL}
#: reading B's token form beside ρ: te_core.formalism.published's ρ for a row without a trace, a target, not a bound
TARGET = (f"{formalism.RESTS_ON_CELL['target']}: te_core.formalism.published at reading B, Fit's c/(c - d) where it "
          f"holds, else Slowdown's token form ({formalism.SENSITIVITIES['token_form']})")
FIT = "Fit's construction (te_core.formalism): every block reads Δ and its tokens' ids, and runs the decoder"
#: what a point's ρ rests on, by its basis (`rho`): a valid kind's `VALID` entry, Fit's construction, or nothing
RESTS_ON = VALID | {
    "fit": FIT,
    "none": "no measured kind > 1: Fit fails and every measured kind's term is at most 1, so rho is 1, partial where "
            "a kind is unmeasured (coordinator 11:18Z)",
    "unmeasured": "unmeasured: Fit fails and no kind is measured, so rho is null, never 1 (coordinator 09:46Z and "
                  "11:18Z): the fresh products need io_bound's tree constant, the token and decoding kinds a trace",
}
#: reading A's verdicts on the token and decoding kinds (`reading_a`); their terms are unmeasured under each
VERDICTS_A = {"unmodelled": "finite (te_core 4.5's tiles split an op, so no op's update bits alone leave it infinite), "
                            "from a trace's entries and op log, which a model hasn't",
              "refused": "a record whose trace's entries and op log aren't here"}


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

    def weight_bits(self, precision: str) -> float:
        """Every weight as a training step's ops read it: the matrices at the GEMMs' width, the embedding and the norms
        in fp32 (the records' u(1) exactly: 1,132,675,072 bits at the 45M reference, 832,782,336 at s-t20 and
        339,320,832 at the island's student)."""
        return WEIGHT_BITS[precision] * self.matrix_params + 32 * (self.params - self.matrix_params)

    def op_bits(self, precision: str, head_bits: float = 32) -> float:
        """The most update bits one op reads where every weight is θ': the head's GEMM over the tied embedding (fp32
        in training, as u(1) reads it) or an MLP matrix at the GEMMs' width."""
        return max(head_bits * self.vocab * self.d_model, WEIGHT_BITS[precision] * MLP * self.d_model ** 2)

    def attention(self, span: float) -> float:
        """QK and PV MACs of one token attending over `span` positions, every layer."""
        return 2 * self.n_layer * self.d_model * span


def dense(params: float, aspect: float = ASPECT, vocab: float = VOCAB, d_model: float | None = None) -> Shape:
    """The dense decoder of `params` parameters at d/L = `aspect`: N = L·((4 + 3·MLP)·d² + 2d) + (V + 1)·d (each layer's
    matrices and two norms, a tied embedding and the final norm), with L = d/aspect a cubic in d, solved by Cardano's
    formula (one real root); or, given `d_model`, L from N. 45,029,888 gives the reference model, d = 512 and L = 6, as
    35,658,240 does at V = 32,000 (s-t20); 13,012,224 at d = 256 and V = 32,000 has 6 layers (the island's student)."""
    if d_model is not None:
        d = float(d_model)
        return Shape(float(params), d, (params - (vocab + 1) * d) / ((4 + 3 * MLP) * d * d + 2 * d), float(vocab))
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


#: the policy the rows are reported at, the console's only (the dashboard's 10:23Z ruling): te_core.policy's `CONSOLE`
POLICIES = {"default": policy.CONSOLE}
POLICY_NOTES = {
    "default": "the console's policy (te_core.policy.CONSOLE, the dashboard's 10:23Z ruling): one full session of the "
               "45M reference a block, 4,096 tokens in and 1,024 out, attention and the LM head included "
               "(policy.session_gates), k = 65,536 bits",
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


def forward(sh: Shape, precision: str, span: float, head_share: float = 1.0) -> Token:
    """An inference token: every weight's forward at `precision` (the head's at the `head_share` of tokens whose logits
    are sampled) and attention over `span` in bf16, all affected."""
    gemm, weights = f"gemm/{precision}", sh.params - (1 - head_share) * sh.vocab * sh.d_model
    macs = _add({gemm: weights}, {"gemm/bf16": sh.attention(span)})
    return Token(macs, macs, {gemm: weights})


def served(prompt: float, generated: float) -> tuple[float, float]:
    """A served sequence's forward positions, the prompt and every generated token but the last (sampled, never read
    back), and the share of them whose logits are sampled, where the head's GEMM runs: the island's rounds count
    exactly these (their fp4 MACs: every matrix at each position, the head at each sampled one)."""
    positions = prompt + generated - 1
    return positions, generated / positions


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

    def a_bits(self, sh: Shape) -> float:
        """a, the narrowest cut across a token's affected path: its lowest adapted layer's first adapters' rank outputs
        in bf16 (`FIRST_ADAPTED`), or the residual stream where narrower. The island's rank-16 rows measure 768 bits,
        3 x 16 x 16, on attention and on every matrix."""
        return min(ACT_BITS * sh.d_model, ACT_BITS * self.rank * FIRST_ADAPTED[self.targets])

    def op_bits(self, sh: Shape, bits: float = 16) -> float:
        """The most update bits one op reads: its widest factor at `bits` a value (`TARGET_WIDEST`)."""
        return bits * self.rank * TARGET_WIDEST[self.targets] * sh.d_model

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


def lora_forward(sh: Shape, ad: Adapter, precision: str, span: float, head_share: float = 1.0) -> Token:
    """An inference token through a LoRA kept apart from its base: the base at `precision` (its head at `head_share`
    of the tokens, as `forward`), attention and the adapter in bf16; the layers below the lowest adapted one
    unaffected, and the adapter's ops read its values."""
    frac = ad.layers / sh.n_layer
    macs = _add({f"gemm/{precision}": sh.params - (1 - head_share) * sh.vocab * sh.d_model},
                {"gemm/bf16": sh.attention(span) + ad.macs})
    below = _add({f"gemm/{precision}": (sh.n_layer - ad.layers) * sh.layer_params},
                 {"gemm/bf16": (1 - frac) * sh.attention(span)})
    return Token(macs, _add(macs, {k: -v for k, v in below.items()}), {"gemm/bf16": ad.macs})


# ---- the fresh products ------------------------------------------------------------------------------------------------

def _layers(n: float) -> list[tuple[int, float]]:
    """Each layer of a stack of n from the bottom, with its weight (the top one's n's fraction where n isn't whole)."""
    return [(j, min(1.0, n - j)) for j in range(math.ceil(n - 1e-9))]


@dataclass(frozen=True)
class Fresh:
    """Products of two fresh operands: `count` tokens (or steps) of `macs` MACs of GEMM kind `kind`, summing into
    `entries` output entries, at each of `layers` layers of the update (the share of the update up to layer j is
    (j + 1)/layers), in sequences of `positions` tokens whose first `inputs` read an input id of `tau` bits each;
    `sequence` where a block computing their operands needs the whole sequence and update (a backward: the gradient
    reads every later token), not only the update up to the layer and the ids up to the token (a forward);
    `attention` for attention's products, two token-indexed dims, which te_core.io_bound counts none of; `paid`, per
    layer from the bottom, the largest k at which both operands pass io_bound's P-width (paid at their width: `_paid`;
    empty, at every k)."""
    kind: str
    macs: float
    entries: float
    layers: float
    count: float
    positions: float
    inputs: float
    tau: float
    sequence: bool = False
    attention: bool = False
    paid: tuple = ()

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

    def _layers(self, k: float) -> list[tuple[float, float]]:
        """(the update's share a block reads for the layer, the layer's weight at k) for each layer."""
        return [(1.0 if self.sequence else min(1.0, (j + 1) / self.layers),
                 wt if not self.paid or k <= self.paid[j] else 0.0) for j, wt in _layers(self.layers)]

    def counted(self, update_bits: float, k: float) -> tuple[float, float]:
        """The MACs and output entries io_bound counts at k: those whose operands pass P-width and that no block
        computes from θ, the update and the inputs within k bits."""
        share = self.count * sum(wt * (1 - self.escaped(r * update_bits, k)) for r, wt in self._layers(k))
        return share * self.macs, share * self.entries

    def total(self, k: float) -> float:
        """The MACs at k before any escapes."""
        return self.count * self.macs * sum(wt for _, wt in self._layers(k))


def dense_fresh(sh: Shape, gemm: str, tokens: float, span: float, seq: tuple, batch_tokens: float | None = None,
                layered: bool = True, head_share: float = 1.0) -> list[Fresh]:
    """A dense model's fresh products, every weight fresh: each layer's matrices at `gemm` and its attention over `span`
    in bf16 (a forward, whose operands need the update up to the layer: `layered`, else all of it, as a code that
    isn't per layer) and the head above them (at `head_share` of the tokens, `served`'s); with `batch_tokens`
    (training, a step's tokens) the input gradients, the weight gradients (each entry a step's sum) and attention's
    backward. The lowest layer's W_q, W_k and W_v read gathered embedding rows, which te_core.io_bound's P-width
    prices below their width (its `expansion`: 12 of its captured dense step's 96 such forwards and weight
    gradients), so their forward and weight gradient aren't among them. `seq` is (positions, inputs, τ)."""
    n = sh.n_layer if layered else 1.0
    d, head, heads, attn = sh.d_model, sh.vocab * sh.d_model, sh.d_model / HEAD_DIM, sh.attention(span) / 2
    gathered = 3 * d * d

    def layer(kind: str, macs: float, entries: float, attention: bool = False) -> Fresh:
        return Fresh(kind, macs / n, entries / n, n, tokens, *seq, attention=attention)

    def whole(kind: str, macs: float, entries: float, attention: bool = False) -> Fresh:
        return Fresh(kind, macs, entries, 1.0, tokens, *seq, sequence=True, attention=attention)

    out = [layer(gemm, sh.matrix_params - gathered, sh.n_layer * (5 + 2 * MLP) * d - 3 * d),
           Fresh(gemm, head, sh.vocab, 1.0, tokens * head_share, *seq),
           layer("gemm/bf16", attn, sh.n_layer * heads * span, True), layer("gemm/bf16", attn, sh.n_layer * d, True)]
    if batch_tokens is not None:
        weights = sh.matrix_params + head
        out += [whole(gemm, weights, sh.n_layer * (6 + MLP) * d + d),
                whole(gemm, weights - gathered, (weights - gathered) / batch_tokens),
                whole("gemm/bf16", attn, sh.n_layer * heads * span, True),
                whole("gemm/bf16", 3 * attn, 3 * sh.n_layer * d, True)]
    return out


#: the bits an entry a LoRA's GEMMs pass (bf16), each operand's width in io_bound's P-width
ADAPTER_BITS = 16.0


def _stream(matrices: tuple, n: int, lam: float, b: float = ADAPTER_BITS) -> list[dict]:
    """te_core.io_bound's rates (`_Rates`) on a LoRA's n adapted layers from the lowest, a model's: the bits a block
    pays an entry of each matrix's input, at most its width b, where the adapter's second product (K = r) passes
    `lam` an entry. The lowest's input is θ's and token data's (0); q, k and v pass the larger of their input's rate
    and lam where adapted, attention's output their sum (QK^T passes d_h/span of each, PV span/d_h of that and V's);
    the residual after attention and after the MLP the largest of its terms'; the MLP's hidden its input's over MLP
    (the frozen gate and up widen it) or lam where they're adapted; the frozen down projection's output the hidden's
    times MLP; and `out` the residual stream's after the layer. The backward's gradients of the residual stream pass
    the top's `out` (the frozen head widens the stream into the logits by V/d, and its gradient narrows them back)."""
    out, r = [], 0.0
    for _ in range(n):
        rq, rk, rv = (min(b, max(r, lam)) if x in matrices else r for x in ("q", "k", "v"))
        att = min(b, rq + rk + rv)
        h = min(b, max(r, att, lam if "o" in matrices else 0.0))
        g = min(b, max(h / MLP, lam if "gate" in matrices or "up" in matrices else 0.0))
        r_in = r
        r = min(b, max(h, g * MLP, lam if "down" in matrices else 0.0))
        out.append({"q": r_in, "k": r_in, "v": r_in, "o": att, "gate": h, "up": h, "down": g, "out": r})
    return out


def _paid(matrices: tuple, n: int, rank: float, x: str) -> tuple:
    """Per adapted layer, the largest k at which matrix x's input passes its width (`_stream`), where the adapter's
    second product passes lam = min(b, 4 r² b²/k) an entry (io_bound's rate of a K = r product with a trained side):
    ∞ where it passes at every k, 0 at none; by bisection on lam, which every rate grows with."""
    b = ADAPTER_BITS

    def ok(lam: float, i: int) -> bool:
        return _stream(matrices, i + 1, lam)[i][x] >= 0.999 * b

    out = []
    for i in range(n):
        if not ok(b, i) or ok(1e-12, i):
            out.append(math.inf if ok(1e-12, i) else 0.0)
            continue
        lo, hi = 1e-12, b
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            lo, hi = (lo, mid) if ok(mid, i) else (mid, hi)
        out.append(4 * rank * rank * b * b / hi)
    return tuple(out)


def adapter_fresh(sh: Shape, ad: Adapter, tokens: float, span: float, seq: tuple,
                  batch_tokens: float | None = None) -> list[Fresh]:
    """A LoRA's fresh products: its base products read θ, so only the adapter's at each adapted layer, as io_bound's
    kinds count them on its captured rank-16 step: x·A over d_in and (with `batch_tokens`) A's gradient over a step's
    tokens where the matrix's input passes its width (`_paid`: never at the lowest adapted layer, whose input is θ's,
    nor while the stream above it passes fewer bits, nor at the down projection, whose input the frozen gate and up
    widen, until the adapter's second product passes its width); (x·A)·B and (dy·B^T)·A^T over the rank (but the
    lowest layer's input gradients that feed no trained value); dy·B^T over d_out and B's gradient where dy passes
    its width (the residual stream's and attention's gradients where the stream leaves the top adapted layer at its
    width; gate's and up's, which the frozen down projection widens, only where the adapter's second product passes
    its width). Attention's products at the adapted layers are listed, which io_bound counts none of."""
    d, r, n = sh.d_model, ad.rank, ad.layers
    ms, count, b = TARGET_MATRICES[ad.targets], len(_layers(ad.layers)), ADAPTER_BITS
    top = _paid(ms, count, r, "out")[-1]
    full = (min(top, 4 * r * r * b),) * count if "down" in ms else (0.0,) * count
    feeds = {"q": False, "k": False, "v": False, "o": any(x in ms for x in "qkv"),
             "gate": any(x in ms for x in ("q", "k", "v", "o")), "up": any(x in ms for x in ("q", "k", "v", "o")),
             "down": "gate" in ms or "up" in ms}
    groups: dict = {}

    def add(name: str, macs: float, entries: float, paid: tuple = (), sequence: bool = False) -> None:
        m0, e0 = groups.get((name, paid, sequence), (0.0, 0.0))
        groups[(name, paid, sequence)] = (m0 + macs, e0 + entries)

    for x in ms:
        d_in, d_out = (v * d for v in MATRICES[x])
        paid_in = _paid(ms, count, r, x)
        paid_dy = full if x in ("gate", "up") else (top,) * count
        add("x A", r * d_in, r, paid_in)
        add("(x A) B", r * d_out, d_out)
        if batch_tokens is not None:
            add("dy B^T", r * d_out, r, paid_dy, True)
            add("dB", r * d_out, r * d_out / batch_tokens, paid_dy, True)
            add("(dy B^T) A^T", r * d_in, d_in, () if feeds[x] else (0.0,) + (math.inf,) * (count - 1), True)
            add("dA", r * d_in, r * d_in / batch_tokens, paid_in, True)
    out = [Fresh("gemm/bf16", macs, entries, n, tokens, *seq, sequence=sequence, paid=paid)
           for (_, paid, sequence), (macs, entries) in groups.items()]
    heads = d / HEAD_DIM
    out.append(Fresh("gemm/bf16", 2 * d * span, heads * span + d, n, tokens, *seq, attention=True))
    if batch_tokens is not None:
        out.append(Fresh("gemm/bf16", 4 * d * span, heads * span + 3 * d, n, tokens, *seq, sequence=True,
                         attention=True))
    return out


def muon_fresh(sh: Shape, optimizer: str, steps: float, seq: tuple) -> list[Fresh]:
    """Muon's Newton-Schulz products with two fresh operands, each step: B·X on every matrix of every layer (a² b MACs
    and a·b entries an iteration on an a x b matrix, a ≤ b), B = bA + cA·A. X X^T and A·A read one array twice, which
    io_bound's P-share leaves uncounted (its captured dense step's `self` rows)."""
    if optimizer != "muon":
        return []
    per = NEWTON_SCHULZ_STEPS * sh.n_layer * (4 + 3 * MLP)
    return [Fresh("gemm/bf16", per * sh.d_model ** 3, per * sh.d_model ** 2, 1.0, steps, *seq, sequence=True)]


# ---- the model a workload gives --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    """A workload at its dials: te_core.formalism.bound's inputs (`measured`), the trained or served model's forward
    per token (`w_fwd`), the fresh products, the most update bits one of its ops reads (`op_update_bits`; None: a
    record's, unknown), and the quantities they came from (`derived`)."""
    workload: str
    dials: dict
    measured: formalism.Measured
    w_fwd: float
    fresh: tuple[Fresh, ...] = ()
    op_update_bits: float | None = None
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
            "affected_gates": "model: each token's MACs that depend on theta', at te_core.gates' prices",
            "tau_bits": "model: the input ids a token block reads, ceil(log2 V) bits each",
            "a_bits": "upper bound (witness): d_model x 16, the bf16 residual stream (measured on both pilot starts)",
            "u": "assumption (A1): 0 up to the free share t, x u(1) beyond it",
            "u_in": "assumption (A1'): evenly spread, x u(1)",
            "decoder_gates": "model: none, E stores theta' itself"} | extra


#: the labels of a training part's partial gradient and gradient cut, 32 bits a trained value
GRADIENT_LABELS = {
    "grad_bits": f"model: the gradient cut, 32 bits a trained value (the traces' Delta_grad less the state: "
                 f"{ISLAND_STANDARD}, {ISLAND_DISTILL}, both LoRAs', and every worker's in {ISLAND_DILOCO})",
    "partial_bits": f"model: one block's partial gradient, 32 bits a trained value, a worker's in DiLoCo (the island's "
                    f"traces measure exactly this as te_core 4.4's partial_cut_bits: {ISLAND_STANDARD}, "
                    f"{ISLAND_DISTILL}, both LoRAs', {ISLAND_DILOCO}'s a worker's)",
}


def _full_labels(precision: str, optimizer: str, **extra) -> dict:
    """The labels of full-weight training, every weight trained from fp32 masters, each form with the records it was
    measured on."""
    state = {"adamw": f"AdamW's 96 N: {PILOT_ADAMW}, {ISLAND_DISTILL}", "muon": f"Muon's: {PILOT_MUON}, {ISLAND_STANDARD}",
             "sgd": "no record runs SGD"}[optimizer]
    return _labels(u1_bits=f"model: every weight as a step's ops read it, the matrices at {WEIGHT_BITS[precision]:g} "
                           f"bits and the embedding and norms in fp32 (the records' u(1) exactly: {PILOT_ADAMW}, "
                           f"{ISLAND_STANDARD}, {ISLAND_DISTILL})",
                   delta_bits=f"model: theta' in fp32, 32 N (the records' code less its 5 KB header: {PILOT_ADAMW}, "
                              f"{ISLAND_DISTILL})",
                   delta_grad_bits=f"model: the gradient at 32 bits a value and {optimizer}'s state (the traces' "
                                   f"exactly; {state})") | GRADIENT_LABELS | extra


#: a's label for a LoRA (`Adapter.a_bits`)
LORA_A = (f"model: the lowest adapted layer's first adapters' rank outputs, 16 r bits each, or the residual stream where "
          f"narrower ({ISLAND_LORA_ATTN} and {ISLAND_LORA_ALL} measure 3 x 16 x 16 = 768 bits at rank 16)")
#: d's label for an E holding a LoRA (`Adapter.decoder_gates`)
LORA_DECODER = ("model: the harness's decoder for a LoRA E (prepare.rebuild_macs), r + 2 fp32 multiply-adds an adapted "
                f"entry and r (d_in + d_out) to dequantize factors below 16 bits, at {formalism.DECODER_DEFINITION}'s "
                "gates (te_core 4.2's witness)")


def _dense_t(t: float | None) -> float:
    return T_DENSE if t is None else float(t)


def standard(n_params: float = REFERENCE_PARAMS, precision: str = "bf16", seq_len: float = 1024,
             batch_tokens: float = 65_536, optimizer: str = "adamw", t: float | None = None,
             tokens_per_param: float = 20.0, vocab: float = VOCAB, d_model: float | None = None) -> Model:
    """Standard dense pretraining: every weight trained, each token's forward and backward in token blocks, the
    optimizer in none. u(1) is every weight as a step's ops read it (`Shape.weight_bits`), Δ (and Δ_t) θ' in fp32, the
    records' code, and no decoder. Every product is fresh."""
    sh = dense(n_params, vocab=vocab, d_model=d_model)
    tok = training(sh, precision, seq_len)
    tokens = tokens_per_param * n_params
    steps = tokens / batch_tokens
    opt = optimizer_gates(sh, optimizer)
    tf = _dense_t(t)
    grad = GRAD_BITS * n_params
    part = formalism.Part(tokens * tok.w + steps * opt, tokens, tokens * tok.w, tokens * tok.alpha, max(1, round(steps)),
                          delta_grad(sh, optimizer), partial_bits=grad, grad_bits=grad)
    m = _measured([part], sh.tau, ACT_BITS * sh.d_model, sh.weight_bits(precision), tf, 32 * n_params,
                  _full_labels(precision, optimizer))
    seq = (seq_len, seq_len, sh.tau)
    fresh = dense_fresh(sh, f"gemm/{precision}", tokens, seq_len, seq, batch_tokens) + muon_fresh(sh, optimizer, steps, seq)
    return Model("standard", _dials(standard, locals()), m, w_fwd(sh, precision, seq_len), tuple(fresh),
                 sh.op_bits(precision),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "w_gates": tok.w, "t": tf, "t_counted": tok.t,
                  "optimizer_gates_per_step": opt})


def diloco(n_params: float = REFERENCE_PARAMS, precision: str = "bf16", seq_len: float = 1024,
           batch_tokens: float = 65_536, optimizer: str = "adamw", t: float | None = None,
           inner_steps: float = 500, syncs: float = 28, sync_bits: float | None = None, workers: float = 4,
           vocab: float = VOCAB, d_model: float | None = None) -> Model:
    """DiLoCo-style pretraining, the record's prefix S syncs: each a phase of H inner steps of B tokens (all `workers`'
    M), standard training on each worker with its own optimizer, then an outer step from the workers' pseudo-gradients,
    each `sync_bits` b once compressed (None: 32 N, uncompressed), summed over the M workers. Its token blocks are
    standard training's, and so are its fresh products and token kind. Only where the syncs and a phase's tokens
    describe every step's weights in fewer bits than θ' (32 N) does Δ change: S·b + H·B·τ, decoded by replaying the phase
    from the synced weights (d = H·B·w), which needs the inner optimizer to restart at each sync. That decoder counts
    only where it fits a block (te_core.formalism.at's d < c), and then a phase's inner gradients never leave its
    block, which passes on only its compressed delta (Δ_grad = b, an outer step without momentum: Nesterov's would
    carry 32 N); a code that isn't per layer leaves the fresh products no layer's share of it to read. Otherwise a
    step's Δ_grad is every worker's gradient and state, as the island's traced step measures (4 x 96 N), and its
    gradient cut every worker's gradient, one block's partial a worker's. Wherever a step's token work fills two
    blocks the decoder fits none (d = H·B·w ≥ B·w > c), so the reduction and update kinds read the uncompressed
    gradients. te_core's reduction kind sums a step's b partials into one gradient, where each of the M workers sums
    its own, M·(⌈b/M⌉ − 1) partials, up to M − 1 fewer than b − 1. A sync has no tokens, and te_core's kinds don't
    price its M pseudo-gradients: (M − 1)/(H·(b − 1)), about 0.02% of the inner steps' reduction at the default's
    H = 500 and 0.2% at the island's 50."""
    sh = dense(n_params, vocab=vocab, d_model=d_model)
    tok = training(sh, precision, seq_len)
    phase = inner_steps * batch_tokens
    tokens = syncs * phase
    steps = syncs * inner_steps
    opt = optimizer_gates(sh, optimizer)
    outer = sh.params * FUSED_OPTIMIZER_MACS["_fused_sgd_"] * per_mac("elementwise/f32")
    b = 32 * n_params if sync_bits is None else float(sync_bits)
    code = syncs * b + phase * sh.tau
    compressed = code < 32 * n_params
    dg = b if compressed else workers * delta_grad(sh, optimizer)
    part = formalism.Part(tokens * tok.w + workers * steps * opt + syncs * outer, tokens, tokens * tok.w,
                          tokens * tok.alpha, max(1, round(steps)), dg, partial_bits=GRAD_BITS * n_params,
                          grad_bits=workers * GRAD_BITS * n_params)
    labels = _full_labels(precision, optimizer)
    if not compressed:
        labels["delta_grad_bits"] = (f"model: every worker's gradient at 32 bits a value and {optimizer}'s state "
                                     f"(M x 96 N for AdamW: {ISLAND_DILOCO}'s traced step exactly)")
    if compressed:
        labels |= {"delta_bits": "model: S syncs of b bits and the phase's token ids, S b + H B tau",
                   "delta_grad_bits": "model: the phase's compressed delta, b (its inner gradients stay in its block "
                                      "wherever the decoder fits one)",
                   "decoder_gates": "model: replaying the phase from the synced weights, H B w"}
    m = _measured([part], sh.tau, ACT_BITS * sh.d_model, sh.weight_bits(precision), _dense_t(t),
                  code if compressed else 32 * n_params, labels, decoder=phase * tok.w if compressed else 0.0)
    seq = (seq_len, seq_len, sh.tau)
    fresh = (dense_fresh(sh, f"gemm/{precision}", tokens, seq_len, seq, batch_tokens / workers, layered=not compressed)
             + muon_fresh(sh, optimizer, workers * steps, seq))
    return Model("diloco", _dials(diloco, locals()), m, w_fwd(sh, precision, seq_len), tuple(fresh),
                 sh.op_bits(precision),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "w_gates": tok.w, "tokens": tokens,
                  "code_bits": code, "compressed": compressed, "phase_tokens": phase, "phase_gates": phase * tok.w})


def rl(n_params: float = REFERENCE_PARAMS, update: str = "full", rank: float = 16, targets: str = "all",
       layers: float | None = None, rollout_precision: str = "fp4", prompt_tokens: float = 256,
       generated_tokens: float = 768, samples_per_trained: float = 1.0, optimizer: str = "adamw",
       batch_tokens: float = 65_536, t: float | None = None, tokens: float = 1e8, vocab: float = VOCAB,
       rounds: bool = False, d_model: float | None = None) -> Model:
    """RL on the policy being trained: rollouts (prefill and generation) of the current policy at `rollout_precision`,
    `samples_per_trained` of them for each sequence a learner trains on in bf16 (every position of each, prompt and
    generated). Every token is affected (the generated ids are sampled from θ'), and only the prompts' ids are inputs.
    `update` "full" trains every weight (u(1) the least of the two kinds', the rollouts' narrower weights; every product
    fresh, the rollouts' too, since they read θ'); "lora" a LoRA kept apart from the base (u(1) = Δ, its values; d the
    harness's decoder for it, `Adapter.decoder_gates`; the base products read θ). One profile serves both kinds, at
    the larger free share and the smaller u(1), as te_core's single A1 needs; `sensitivity` gives the token kind's
    bounds that count each kind's tokens apart. With `rounds`, the rollouts alone, `tokens` of them, as the island's
    generation rows record them (the learner's steps aren't in the row): a round serves θ' as it stands, so Δ is the
    served weights (N at the rollouts' width) or the adapter, with no gradient and no learner's products."""
    sh = dense(n_params, vocab=vocab, d_model=d_model)
    seq = prompt_tokens + generated_tokens
    pos, head = served(prompt_tokens, generated_tokens)
    span = decode_span(prompt_tokens, generated_tokens)
    n_l = 0.0 if rounds else tokens
    n_r = tokens if rounds else samples_per_trained * n_l
    steps = n_l / batch_tokens
    sq_r, sq_l = (pos, prompt_tokens, sh.tau), (seq, prompt_tokens, sh.tau)
    if update == "full":
        roll, learn = forward(sh, rollout_precision, span, head), training(sh, "bf16", seq)
        weights = n_params * WEIGHT_BITS[rollout_precision]
        u1s = (weights, sh.weight_bits("bf16"))
        ts = (roll.t, T_DENSE)
        delta, dg, opt = 32 * n_params, delta_grad(sh, optimizer), optimizer_gates(sh, optimizer)
        decoder, trained, a = 0.0, n_params, ACT_BITS * sh.d_model
        op = sh.op_bits("bf16")
        fresh = dense_fresh(sh, f"gemm/{rollout_precision}", n_r, span, sq_r, head_share=head)
        if rounds:
            delta, op = weights, sh.op_bits(rollout_precision, WEIGHT_BITS[rollout_precision])
        else:
            fresh += dense_fresh(sh, "gemm/bf16", n_l, seq, sq_l, batch_tokens) + muon_fresh(sh, optimizer, steps, sq_l)
    else:
        ad = Adapter.of(sh, rank, targets, layers)
        roll, learn = lora_forward(sh, ad, rollout_precision, span, head), lora_training(sh, ad, seq)
        delta = 16 * ad.values
        u1s = (delta, delta)
        ts = (roll.t, learn.t)
        state = "sgd" if optimizer == "sgd" else "adamw"
        dg = (GRAD_BITS + STATE_BITS[state]) * ad.values
        opt = ad.values * FUSED_OPTIMIZER_MACS[f"_fused_{state}_"] * per_mac("elementwise/f32")
        decoder, trained, a = ad.decoder_gates(), ad.values, ad.a_bits(sh)
        op = ad.op_bits(sh)
        fresh = adapter_fresh(sh, ad, n_r, span, sq_r)
        if not rounds:
            fresh += adapter_fresh(sh, ad, n_l, seq, sq_l, batch_tokens)
    if rounds:
        u1s, ts = u1s[:1], ts[:1]
    if t is not None:
        ts = tuple(float(t) for _ in ts)
    tf = max(ts)
    grad = GRAD_BITS * trained
    parts = [formalism.Part(n_r * roll.w, n_r, n_r * roll.w, n_r * roll.alpha, 1, 0.0, partial_bits=0.0, grad_bits=0.0)]
    if not rounds:
        parts.append(formalism.Part(n_l * learn.w + steps * opt, n_l, n_l * learn.w, n_l * learn.alpha,
                                    max(1, round(steps)), dg, partial_bits=grad, grad_bits=grad))
    tau = sh.tau * prompt_tokens * (n_r / pos + n_l / seq) / (n_r + n_l)
    ids = ("model: the prompts' ids, ceil(log2 V) bits each over the tokens of their sequences (a rollout's forward "
           "positions, every generated token but the last; generated ids are computed)")
    none = "model: none (the rollouts alone)"
    if update == "full" and rounds:
        labels = _labels(tau_bits=ids, delta_grad_bits=none,
                         u1_bits=f"model: every weight as the rollouts' GEMMs read it, N x "
                                 f"{WEIGHT_BITS[rollout_precision]:g} bits",
                         delta_bits=f"model: the served weights, N x {WEIGHT_BITS[rollout_precision]:g} bits (a round "
                                    "serves theta' as it stands)")
    elif update == "full":
        labels = _full_labels("bf16", optimizer, tau_bits=ids,
                              u1_bits=f"model: the least of the two kinds' weights as their ops read them, the "
                                      f"rollouts' N x {WEIGHT_BITS[rollout_precision]:g} bits")
    else:
        labels = _labels(tau_bits=ids, u1_bits="model: the adapter's values at 16 bits, Delta",
                         delta_bits="model: the adapter at 16 bits", decoder_gates=LORA_DECODER, a_bits=LORA_A,
                         delta_grad_bits=none if rounds else "model: the gradient at 32 bits a value and the optimizer's "
                                                             "state") | ({} if rounds else GRADIENT_LABELS)
    m = _measured(parts, tau, a, min(u1s), tf, delta, labels, decoder=decoder)
    derived = {"d_model": sh.d_model, "n_layer": sh.n_layer, "rollout_w": roll.w, "learner_w": learn.w,
               "rollout_tokens": n_r, "learner_tokens": n_l,
               "rollout_share": n_r * roll.w / (n_r * roll.w + n_l * learn.w), "t": tf}
    if not rounds:
        derived["kinds"] = [{"kind": kind, "tokens": n, "u1_bits": u1, "t": tk}
                            for kind, n, u1, tk in zip(("rollout", "learner"), (n_r, n_l), u1s, ts)]
    return Model("rl", _dials(rl, locals()), m, w_fwd(sh, rollout_precision, span), tuple(fresh), op, derived)


def inference(n_params: float = REFERENCE_PARAMS, precision: str = "fp4", prompt_tokens: float = 4096,
              generated_tokens: float = 1024, t: float | None = None, tokens: float = 1e9,
              vocab: float = VOCAB, d_model: float | None = None) -> Model:
    """Serving an unregistered model: its weights are no part of θ, so every one is update bits, as if θ' (Δ = its
    weights, d = 0, no training: Δ_grad = 0, so Fit is running θ', Δ + s·τ <= k), and every product is fresh. σ = 1;
    t is the share of a token's work in ops that read no weight, attention and the elementwise ops."""
    sh = dense(n_params, vocab=vocab, d_model=d_model)
    pos, head = served(prompt_tokens, generated_tokens)
    span = decode_span(prompt_tokens, generated_tokens)
    tok = forward(sh, precision, span, head)
    tf = tok.t if t is None else float(t)
    u1 = n_params * WEIGHT_BITS[precision]
    part = formalism.Part(tokens * tok.w, tokens, tokens * tok.w, tokens * tok.alpha, 1, 0.0, partial_bits=0.0,
                          grad_bits=0.0)
    labels = _labels(tau_bits="model: the prompt's ids, ceil(log2 V) bits each over the session's forward positions "
                              "(every generated token but the last)",
                     u1_bits=f"model: every weight as its GEMMs read it, N x {WEIGHT_BITS[precision]:g} bits",
                     delta_bits=f"model: the served weights, N x {WEIGHT_BITS[precision]:g} bits",
                     delta_grad_bits="model: none (no training)")
    m = _measured([part], sh.tau * prompt_tokens / pos, ACT_BITS * sh.d_model, u1, tf, u1, labels)
    sq = (pos, prompt_tokens, sh.tau)
    return Model("inference", _dials(inference, locals()), m, w_fwd(sh, precision, span),
                 tuple(dense_fresh(sh, f"gemm/{precision}", tokens, span, sq, head_share=head)),
                 sh.op_bits(precision, WEIGHT_BITS[precision]),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "span": span, "t_counted": tok.t})


def lora(n_params: float = REFERENCE_PARAMS, rank: float = 8, targets: str = "all", layers: float | None = None,
         adapter_bits: float = 16, optimizer: str = "adamw", seq_len: float = 1024, batch_tokens: float = 65_536,
         t: float | None = None, tokens: float = 1e8, vocab: float = VOCAB, d_model: float | None = None) -> Model:
    """Fine-tuning a registered model with a LoRA on `targets` of its top `layers` layers (None: all): Δ the adapter's
    values at `adapter_bits` (fp32 latents for Δ_t below 16), d the harness's decoder for it (`Adapter.decoder_gates`),
    Δ_grad their gradient at 32 bits and the optimizer's state, u(1) = Δ, and t the share of the affected work that
    reads no adapter value. The base products read θ, so the fresh products are the adapter's and attention's above
    the lowest adapted layer (`adapter_fresh`)."""
    sh = dense(n_params, vocab=vocab, d_model=d_model)
    ad = Adapter.of(sh, rank, targets, layers)
    tok = lora_training(sh, ad, seq_len)
    tf = tok.t if t is None else float(t)
    steps = tokens / batch_tokens
    opt = ad.values * FUSED_OPTIMIZER_MACS[f"_fused_{optimizer}_"] * per_mac("elementwise/f32")
    delta = ad.values * adapter_bits
    dg = (GRAD_BITS + STATE_BITS[optimizer]) * ad.values
    grad = GRAD_BITS * ad.values
    part = formalism.Part(tokens * tok.w + steps * opt, tokens, tokens * tok.w, tokens * tok.alpha, max(1, round(steps)),
                          dg, partial_bits=grad, grad_bits=grad)
    labels = _labels(u1_bits="model: the adapter's values, Delta",
                     delta_bits=f"model: the adapter's values at {adapter_bits:g} bits",
                     delta_grad_bits=f"model: their gradient at 32 bits and {optimizer}'s state",
                     decoder_gates=LORA_DECODER, a_bits=LORA_A) | GRADIENT_LABELS
    latents = adapter_bits < 16
    if latents:
        labels["delta_t_bits"] = "model: fp32 latents, 32 bits a value (a straight-through run)"
    m = _measured([part], sh.tau, ad.a_bits(sh), delta, tf, delta, labels, decoder=ad.decoder_gates(latents),
                  delta_t=32 * ad.values if latents else None)
    seq = (seq_len, seq_len, sh.tau)
    return Model("lora", _dials(lora, locals()), m, w_fwd(sh, "bf16", seq_len),
                 tuple(adapter_fresh(sh, ad, tokens, seq_len, seq, batch_tokens)), ad.op_bits(sh, adapter_bits),
                 {"d_model": sh.d_model, "n_layer": sh.n_layer, "adapter_values": ad.values, "w_gates": tok.w,
                  "t_counted": tok.t})


def distill(student_params: float = REFERENCE_PARAMS, teacher_params: float = 10 * REFERENCE_PARAMS,
            teacher_share: float = 1.0, teacher_precision: str = "bf16", seq_len: float = 1024,
            batch_tokens: float = 65_536, optimizer: str = "adamw", t: float | None = None,
            tokens_per_param: float = 20.0, vocab: float = VOCAB, student_d_model: float | None = None,
            teacher_d_model: float | None = None) -> Model:
    """Distillation from a registered teacher: the student's dense training, and the teacher's forward on
    `teacher_share` of its tokens in the same token blocks. The teacher reads θ and the tokens alone, so its work is
    outside A and none of its products is fresh: it raises w and leaves the affected count, u, t, a and the fresh
    products the student's."""
    sh = dense(student_params, vocab=vocab, d_model=student_d_model)
    te = dense(teacher_params, vocab=vocab, d_model=teacher_d_model)
    stu, teach = training(sh, "bf16", seq_len), forward(te, teacher_precision, seq_len)
    tok = Token(_add(stu.macs, {k: teacher_share * v for k, v in teach.macs.items()}), stu.affected, stu.reading)
    tokens = tokens_per_param * student_params
    steps = tokens / batch_tokens
    opt = optimizer_gates(sh, optimizer)
    grad = GRAD_BITS * student_params
    part = formalism.Part(tokens * tok.w + steps * opt, tokens, tokens * tok.w, tokens * tok.alpha, max(1, round(steps)),
                          delta_grad(sh, optimizer), partial_bits=grad, grad_bits=grad)
    m = _measured([part], sh.tau, ACT_BITS * sh.d_model, sh.weight_bits("bf16"), _dense_t(t), 32 * student_params,
                  _full_labels("bf16", optimizer))
    seq = (seq_len, seq_len, sh.tau)
    fresh = dense_fresh(sh, "gemm/bf16", tokens, seq_len, seq, batch_tokens) + muon_fresh(sh, optimizer, steps, seq)
    return Model("distill", _dials(distill, locals()), m, w_fwd(sh, "bf16", seq_len), tuple(fresh),
                 sh.op_bits("bf16"),
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


#: a fresh-product group's fields, in the order `menu` lists them
FRESH_FIELDS = [f.name for f in fields(Fresh)]
#: what te_core.formalism.bound's reduction and update kinds read of each part (`menu`)
PART_FIELDS = ("steps", "tokens", "token_gates", "partial_bits", "grad_bits")


def _listed(f: Fresh) -> list:
    return [[_finite(x) for x in v] if isinstance(v, tuple) else v for v in astuple(f)]


def menu(m: Model) -> dict:
    """The other kinds' inputs, as `rho` reads them: the fresh products (each group's `FRESH_FIELDS`, `paid`'s ∞ None;
    their widths and gates a MAC and a gadget by kind are the reference's `widths`, `gates_per_mac` and `cell_gates`,
    K_whole d_model), what te_core.formalism.bound's reduction and update kinds read of each part (`PART_FIELDS`), and
    the most update bits one op reads (`reading_a`'s input)."""
    me = m.measured
    return {"u1_bits": me.u1_bits, "delta_bits": me.delta_bits, "decoder_gates": me.decoder_gates,
            "d_model": m.derived["d_model"], "fresh": [_listed(f) for f in m.fresh],
            "parts": [{key: getattr(p, key) for key in PART_FIELDS} for p in me.parts],
            "op_update_bits": m.op_update_bits}


def _finite(v):
    return None if isinstance(v, float) and math.isinf(v) else v


def products(m: Model, c: float, k: float) -> list[dict]:
    """The model's fresh products as te_core.io_bound's `bound` reads a captured step's fresh rows (`Row.product`): each
    group's MACs and output entries io_bound counts at k (`Fresh.counted`; attention's none), its chain K (MACs over
    outputs), d_model the shortest chain that makes an operand whole by itself (as on io_bound's captured steps), its
    widths (`WIDTHS`) and its Definition's gates a MAC and a gadget."""
    ub = update_bits(m.measured, c)
    out = []
    for f in m.fresh:
        macs, entries = (0.0, 0.0) if f.attention else f.counted(ub, k)
        if macs > 0:
            b_a, b_b, b_c = WIDTHS[f.kind]
            out.append({"macs": macs, "outputs": entries, "k_dim": macs / entries, "k_whole": m.derived["d_model"],
                        "b_a": b_a, "b_b": b_b, "b_c": b_c, "gates_per_mac": per_mac(f.kind),
                        "cell_gates": float(gates(PRICES[f.kind][0][0]))})
    return out


def fresh(m: Model, c: float, k: float) -> tuple[float | None, str | None, dict, dict]:
    """The fresh-products kind at (c, k): te_core.formalism.fresh_products on `products`, the model's `Measured.products`
    at k and summed in `ORDER`; its term where formalism measures it, io_bound's proved figure without P-cell
    (`FRESH_FIGURE`), else None with formalism's reason; io_bound's chain figures beside it, the sensitivities
    `fresh_chain` (without P-cell) and `fresh_cell` (with it, `FRESH_CELL`, formalism's own term); and Λ (None: void,
    past whole_chain's r = 1/4, its figures 0), the counted share of |C|, attention's share (uncounted) and the chain
    figure at the numerical Λ."""
    me = replace(m.measured, products={"k": float(k), "order": ORDER, "products": products(m, c, k)})
    _, why, got = formalism.fresh_products(me, c, k)
    b = got["io_bound"]
    p = b["proved"]
    chain = {"fresh_chain": p[FRESH_FIGURE], "fresh_cell": p[FRESH_CELL]}
    return None if why else p[FRESH_FIGURE], why, chain, {
        "order": ORDER, "lam": p["lam"], "share": b["share"], "numerical": b["numerical"][FRESH_FIGURE],
        "attention_share": sum(f.total(k) * per_mac(f.kind) for f in m.fresh if f.attention) / me.gates}


def reading_a(m: Model) -> str:
    """Reading A's verdict on the token and decoding kinds (module docstring; `VERDICTS_A`), whose terms are unmeasured
    under each: te_core 4.5's packer splits an op into matrix-product tiles that read only the coordinates they use,
    so its term is finite however many update bits one op reads (`Model.op_update_bits`, the menu's), from a trace's
    entries and op log, which a model hasn't. A record's ops aren't here."""
    return "refused" if m.op_update_bits is None else "unmodelled"


def rho(m: Model, c: float, k: float, target: tuple[float, str] | None = None) -> dict:
    """ρ at (c, k), the published bound (coordinator 08:26Z, 10:21Z and 11:18Z, as te_core.formalism.at gives it):
    Fit's c/(c − d) where Fit holds and no valid kind's term exceeds it, else max(1, the `VALID` kinds' measured terms)
    (the fresh products' by `fresh`; the token and decoding kinds' by te_core.formalism.bound, None without a trace's
    entries and op log, `reading_a`'s verdict beside them), and None where no kind is measured, never 1 (coordinator
    09:46Z), with `bound` that max (None where none is measured) beside it; `basis` names fit, the binding kind, none
    (no measured term exceeds 1) or unmeasured, `RESTS_ON[basis]` what it rests on, `partial` whether a valid kind is
    unmeasured beside a ρ the measured ones give (the dashboard's mark), and `why` each unmeasured kind's reason.
    Beside it, never in the max: the `TARGET` (te_core.formalism.published at reading B, or for a record `target`,
    page_rho's on its columns, which then also gives Fit) and the `SENSITIVITIES` (bound's, None where unmeasured,
    and io_bound's chain figures); the kinds and sensitivities unmeasured (an infinite term among them: never ∞);
    Fit's clauses that fail and its thresholds (the most Δ its compute block admits, k − (c/w_fit)·τ, and the most
    Δ_grad its reduction block admits, k/2); and each part's token blocks b."""
    me = m.measured
    v, one = formalism.at(me, c, k), formalism.bound(me, c, k)
    fit = v["fit"] if target is None else target[1] == "fit"
    term, why, chain, fr = fresh(m, c, k)
    terms = {kind: _finite(term if kind == "fresh_products" else one["terms"][kind]) for kind in formalism.KINDS}
    priced = {kind: x for kind, x in terms.items() if x is not None}
    top = max(priced, key=priced.get) if priced else None
    bound = max([1.0, *priced.values()]) if priced else None
    binding = top if bound is not None and bound > 1 else None
    conflict = fit and bound is not None and bound > v["rho_fit"] * (1 + 1e-9)
    by_fit = fit and not conflict
    if target is None:
        pub = formalism.published(me, c, k, "B")
        target = (pub["rho"], pub["binding_kind"])
    sens = {name: _finite(one["sensitivities"][name]) for name in formalism.SENSITIVITIES} | chain
    reasons = {"fresh_products": why} | {kind: one["unmeasured"].get(kind) for kind in ("token", "decoding")}
    fails = sorted({clause for p in v["parts"] for clause, ok in p["fit"].items() if not ok})
    out = {"rho": _finite(v["rho_fit"]) if by_fit else bound,
           "basis": "fit" if by_fit else binding or ("none" if priced else "unmeasured"), "binding_kind": binding,
           "bound": bound, "terms": terms, "partial": not by_fit and bool(priced) and len(priced) < len(terms),
           "why": {kind: reasons[kind] for kind in terms if terms[kind] is None},
           "fresh": fr, "target": {"rho": _finite(target[0]), "basis": target[1]}, "sensitivities": sens,
           "token_A": reading_a(m),
           "unmeasured": [name for name in (*terms, *sens) if (terms | sens)[name] is None],
           "fit": fit, "fit_fails": fails, "rho_fit": _finite(v["rho_fit"]), "s": v["s"], "x0": v["x0"],
           "s_star": v["s_star"], "b": one["inputs"]["reduction"]["blocks"],
           "fit_delta_bits": k - c / me.w_fit * me.tau_bits, "fit_delta_grad_bits": k / 2}
    if conflict:
        out["conflict"] = (f"Fit holds, but the {binding} kind's term exceeds its c/(c - d): Fit's compute block, s "
                           "tokens computed from their inputs, can't compute their backward without their sequence")
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


RECORD_SOURCE = ("te_core 4.1's console columns for the record (algorithm_policy, quantities measured, t literal; 4.2 "
                 "refuses the trace, made before its capture listed read-backs, until the run is traced again); "
                 "delta_grad_bits its trace's gradient cut and state; u_in_points the trace's u_in at the policies' k")


@dataclass(frozen=True)
class Check:
    """A measured record beside the model at `dials`: the record's console columns, on which page_rho gives its token
    kind at each policy, and each input's source (`status`: "measured" or "assumed: " and why, the standard-workloads
    panel's input_status; none given, every column measured); `menu` the record's own inputs of the other kinds
    (`grad_bits` and `partial_bits`, from its trace) and `parts` its parts in te_core.formalism.Part's fields (an
    island row's; none, one part from the columns); `recorded` makes it a model beside the model's."""
    name: str
    dials: dict
    record: str
    columns: dict
    differs: str
    status: dict = field(default_factory=dict)
    menu: dict = field(default_factory=dict)
    source: str = RECORD_SOURCE
    parts: tuple = ()


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


#: the standard-workloads island's measured rows (`records`: its panel's cells, each with its parts and traced steps)
RECORDS = Path(__file__).with_name("workloads_records.json")
#: how each island row differs from the model at its dials (the target, reading B's token form, goes as 1/t)
FRONT_LOADED = "u_in front-loaded ({} at x = 0) where the model's A1' is even"
#: how a generation row differs: a round isn't traced, so te_core takes its stubs
UNTRACED = ("a round isn't traced, so the record's A1 is te_core's stub, t 0 with u even to Delta and u_in = u, against "
            "the model's t {t}, and its affected work only the ops that read an update tensor (te_core's untraced "
            "convention, {share} of its work) where the model's is every op the update reaches, which reading B's "
            "decoding form reads; 1 - sigma the record's {sigma}, {outside} gates outside the token blocks (the same "
            "in each generation row from its start), where the model's rounds have none; w_fwd the start's bf16 "
            "forward at 1,024 tokens, which no policy here reads")
#: m-t20's layers at its width in the model's shape (its 2.75 d MLP against m-t20's 8/3 d): the matrices' parameters
#: m-t20's to 1e-5, attention's MACs 2% below its 8 layers'
M_T20_LAYERS = dense(M_T20_PARAMS, vocab=policy.SHAPES[M_T20_PARAMS]["vocab"], d_model=M_T20).n_layer
#: how an m-t20 row's shape differs
M_T20_SHAPE = f"m-t20 at d 768 in the model's shape, {M_T20_LAYERS:.2f} layers of 2.75 d MLPs where it has 8 of 8/3 d"
ISLAND_DIFFERS = {
    ISLAND_STANDARD: f"t the trace's 0.732 against the model's 0.693 (the 45M trace's literal t), which puts the "
                     f"model's target 6% above the record's; 1 - sigma the trace's 0.0036 against the model's 0.0075, "
                     f"whose five Newton-Schulz iterations count twice the trace's optimizer gates; "
                     f"{FRONT_LOADED.format('the fp32 embedding')}; w_fit its eval steps' forward",
    ISLAND_DILOCO: f"t the trace's 0.732 against the model's 0.693, the model's target 6% above the record's; the "
                   f"outer step Nesterov SGD where the model's is plain, and its 8 syncs' pseudo-gradients priced by "
                   f"no kind on either; {FRONT_LOADED.format('the fp32 embedding')}",
    ISLAND_LORA_ATTN: f"t the trace's 0.9967 against the model's 0.9916 (its count of the affected work reading no "
                      f"adapter value); {FRONT_LOADED.format('131,072 bits')}",
    ISLAND_LORA_ALL: f"t the trace's 0.9920 against the model's 0.9802; {FRONT_LOADED.format('262,144 bits')}",
    ISLAND_DISTILL: "t the trace's 0.781 against the model's 0.693 (the 45M trace's literal t), which puts the model's "
                    "target 13% above the record's; u_in front-loaded (the student's fp32 embedding at x = 0) where "
                    "the model's A1' is even; w_fwd the start's (the s-t20 teacher's) forward where the model's is the "
                    "student's, which no policy here reads",
    ISLAND_RL_LORA: f"{UNTRACED.format(t='0.916', share='8.6%', sigma='0.0051', outside='1.23e13')}, and a the stub's "
                    "8,192 bits (the residual stream) against the model's 768 (the rank outputs, as the island's traced "
                    "LoRAs measure); w 3% above the record's",
    ISLAND_RL_FULL: f"{UNTRACED.format(t='0.338', share='69%', sigma='0.0054', outside='1.23e13')}; Delta the served "
                    "codes, 0.05% above the model's N x 4.5 bits (the norms in bf16)",
    ISLAND_INFERENCE: f"{UNTRACED.format(t='0.522', share='50%', sigma='0.0042', outside='1.23e13')}; w 1.1% below the "
                      "record's; Delta as the full-weight RL row's",
    ISLAND_STANDARD_60M: f"{M_T20_SHAPE}; t the trace's 0.712 against the model's 0.693, which puts the model's target "
                         f"3% above the record's; 1 - sigma the trace's 0.015 against the model's 0.029 (the model's "
                         f"five Newton-Schulz iterations, as on s-t20); w 2.1% below the record's; "
                         f"{FRONT_LOADED.format('the fp32 embedding')}",
    ISLAND_DISTILL_60M: f"the teacher {M_T20_SHAPE}, and the student at d 384 with as many layers where it has 8; t "
                        f"the trace's 0.779 against the model's 0.693, which puts the model's target 12% above the "
                        f"record's; u_in front-loaded (the student's fp32 embedding at x = 0) where the model's A1' is "
                        f"even; w_fwd the start's (the m-t20 teacher's) forward where the model's is the student's, "
                        f"which no policy here reads",
    ISLAND_RL_FULL_60M: f"{UNTRACED.format(t='0.271', share='76%', sigma='0.0091', outside='2.81e13')}; Delta the "
                        f"served codes, 0.04% above the model's N x 4.5 bits (the norms in bf16); w 1.5% above the "
                        f"record's; {M_T20_SHAPE}",
    ISLAND_INFERENCE_60M: f"{UNTRACED.format(t='0.449', share='57%', sigma='0.0041', outside='2.81e13')}; w 0.7% above "
                          f"the record's; Delta as the full-weight RL rows'; {M_T20_SHAPE}",
}
#: the gradient inputs' sources on an island row: its main part's, te_core 4.4's partial_cut_bits and grad_cut_bits
ISLAND_MENU_STATUS = {"grad_bits": "measured: the row's main part's gradient cut (te_core 4.4's grad_cut_bits)",
                      "partial_bits": "measured: the row's main part's partial gradient, one block's (te_core 4.4's "
                                      "partial_cut_bits; a worker's in DiLoCo)"}
#: the island's generation rows, the rollouts alone (`rl`'s `rounds`)
ISLAND_ROUNDS = (ISLAND_RL_LORA, ISLAND_RL_FULL, ISLAND_RL_FULL_60M)
#: each island start's width beyond its rows' dials (the panel names none; s-t20's is the reference model's d/L)
ISLAND_WIDTH = {"60M": M_T20}


def records(tables: dict, art: str) -> dict:
    """`RECORDS` from the island's `training/standard-workloads` panel (`tables`, its uncut tables `art`): each row's
    cells (its dials, run, about, `rho_inputs`, `input_status`, ρ at the island's two points, reading and te_core)
    with its read's steps and checkpoint, its parts (each with its partial gradient and gradient cut) and its traced
    steps' measured facts, or a generation row's rounds."""
    cols, rows = tables["columns"], []
    for cells, det in zip(tables["rows"], tables["details"]):
        r = dict(zip(cols, cells))
        run, rest = r["run"].split(" ", 1)
        ref, about = rest.split(" ", 1)
        rows.append({"workload": r["workload"], "model": r["model"], "run": run, "art": ref,
                     "about": about.strip().removeprefix("(").removesuffix(")"), "dials": json.loads(r["dials"]),
                     "rho_inputs": json.loads(r["rho_inputs"]), "input_status": json.loads(r["input_status"]),
                     "rho_default": r["rho_default"], "rho_console": r["rho_console"], "reading": r["reading"],
                     "te_core": r["te_core"], "steps": det["read"]["steps"], "checkpoint": det["read"]["checkpoint"],
                     "invalid": det["read"]["invalid"], "parts": det["parts"]}
                    | ({"traces": [{"step": t["step"]} | t["measured"] for t in det["read"]["traces"]]}
                       if "traces" in det["read"] else {"rounds": det["read"]["rounds"]}))
    return {"source": f"{tables['id']} ({art}, {tables['updated_at']}): {tables['source']}", "tables": art,
            "rows": rows}


def island_checks(workload: str) -> tuple[Check, ...]:
    """The island's measured rows of `workload` (`RECORDS`) as checks: their dials, with their start's width
    (`ISLAND_WIDTH`, the teacher's in distillation) and, for a generation row of RL's, the rollouts alone (`rounds`,
    its tokens); their columns and input status; and their parts, each with its partial gradient and gradient cut
    (`Check.menu` its main part's). None before `--records` first writes the file (`workloads.json`'s pin then
    fails)."""
    if not RECORDS.exists():
        return ()
    data = json.loads(RECORDS.read_text())
    out = []
    for r in data["rows"]:
        if r["workload"] != workload:
            continue
        width = ISLAND_WIDTH.get(r["model"])
        dials = (r["dials"] | ({} if width is None else {"teacher_d_model" if workload == "distill" else "d_model": width})
                 | ({"rounds": True, "tokens": sum(p["tokens"] for p in r["parts"])} if r["run"] in ISLAND_ROUNDS else {}))
        main = max(r["parts"], key=lambda p: p["token_gates"])
        note = f"; {'; '.join(r['invalid'])}" if r["invalid"] else ""
        out.append(Check(f"{r['model']} standard-workloads {r['run']}", dials, f"{r['run']}@{r['checkpoint']}",
                         r["rho_inputs"], ISLAND_DIFFERS.get(r["run"], ""), r["input_status"] | ISLAND_MENU_STATUS,
                         {"grad_bits": main["grad_bits"], "partial_bits": main["partial_bits"]},
                         f"the standard-workloads island's row ({r['art']}; te_core {r['te_core']}, reading "
                         f"{r['reading']}; {data['tables']}): {r['about']}{note}", tuple(r["parts"])))
    return tuple(out)


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
    "d_model": Dial("d_model", "d_model", None, (64, 8192),
                    "the model's width, n_layer following from N; None is the reference model's d/L (`dense`). The "
                    f"island's 60M rows are m-t20's 768, which puts {M_T20_LAYERS:.2f} layers where it has 8 (its "
                    "SwiGLU is 8/3 d, the model's 2.75 d)"),
}

WORKLOADS = {w.name: w for w in (
    Workload("standard", "Standard dense pretraining",
             "Every weight of a dense decoder trained by backpropagation, its optimizer outside the token blocks.",
             standard,
             (COMMON["n_params"], COMMON["precision"], COMMON["seq_len"], COMMON["batch_tokens"], COMMON["optimizer"],
              COMMON["t_dense"], COMMON["tokens_per_param"], COMMON["vocab"], COMMON["d_model"]),
             (("default", {}), ("FP8 GEMMs", {"precision": "e4m3"}), ("7B", {"n_params": 7e9})),
             (Check("45M ref-optim/adamw-rewarm-mild", {}, "r20261102-140347-a1c2@946",
                    FULL_RECORD | FULL_U | {"w_gates": 2.31908e11, "sigma": 0.999823, "w_fit_gates": 1.5441e11,
                                            "alpha_ratio": 0.995234, "delta_grad_bits": 4322869248},
                    "w 0.9% below the model's; w_fit and alpha_ratio the rewarm's phases, where the model has one; "
                    "u_in all of u(1) at x = 0, where the model's A1' is even"),
              Check("45M ref-optim/cooldown-forks", {"optimizer": "muon"}, "r20261102-134356-ee8f@694",
                    FULL_RECORD | FULL_U | {"w_gates": 2.32623e11, "sigma": 0.993392, "w_fit_gates": 1.03235e11,
                                            "alpha_ratio": 0.992069, "delta_grad_bits": 3706306560},
                    "as adamw-rewarm-mild; sigma the trace's 0.99339 against the model's 0.98774, five bf16 "
                    "Newton-Schulz iterations on every matrix")) + island_checks("standard")),
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
              COMMON["vocab"], COMMON["d_model"]),
             (("default", {}), ("1-bit syncs", {"sync_bits": REFERENCE_PARAMS}),
              ("Fit's corner", {"inner_steps": 2, "batch_tokens": 256, "syncs": 16, "sync_bits": 1024})),
             island_checks("diloco")),
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
              Dial("tokens", "learner tokens", 1e8, (1e6, 1e11), "sets |C| only (with rounds, the rollouts')"),
              COMMON["vocab"],
              Dial("rounds", "mode", False, (False, True),
                   "the rollouts alone, `tokens` of them, as the island's generation rows record them: no learner, "
                   "Delta the served weights or the adapter"), COMMON["d_model"]),
             (("default", {}), ("LoRA rank 16", {"update": "lora"}),
              ("LoRA in one layer", {"update": "lora", "rank": 1, "targets": "q", "layers": 1})),
             island_checks("rl")),
    Workload("inference", "Inference on an unregistered model",
             "Serving a model whose weights are not registered, so all of them are update bits.",
             inference,
             (Dial("n_params", "parameters", REFERENCE_PARAMS, N_RANGE, "the served model"),
              Dial("precision", "GEMM format", "fp4", ("bf16", "e4m3", "fp4"), "the served weights' format (fp4: W4A4)"),
              Dial("prompt_tokens", "tokens", 4096, (16, 131072), "a session's prompt, the only inputs"),
              Dial("generated_tokens", "tokens", 1024, (1, 32768), "a session's generated tokens"),
              Dial("t", "share of a token's work", None, (0.0, 0.95),
                   "A1's free share, the token kind's: None is attention's modelled share; 0 is evenly spread"),
              Dial("tokens", "tokens served", 1e9, (1e6, 1e13), "sets |C| only"), COMMON["vocab"], COMMON["d_model"]),
             (("default", {}), ("bf16", {"precision": "bf16"}), ("7B", {"n_params": 7e9})),
             island_checks("inference")),
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
              Dial("tokens", "tokens", 1e8, (1e6, 1e11), "sets |C| only"), COMMON["vocab"], COMMON["d_model"]),
             (("default", {}), ("top layer", {"layers": 1}), ("MLP, top layer", {"targets": "mlp", "layers": 1}),
              ("rank 1 on W_q", {"rank": 1, "targets": "q"}),
              ("rank 1 on W_q, one layer", {"rank": 1, "targets": "q", "layers": 1})),
             (Check("45M sparse/lora-alloc", {"rank": 4, "targets": "qv"}, "r20261102-132959-286b@1720",
                    {"w_gates": 1.66764e11, "sigma": 0.999972, "tau_bits": 16.0156, "a_bits": 8192,
                     "delta_bits": 788712, "delta_t_bits": 788712, "d_gates": 0, "delta_grad_bits": 4657152,
                     "w_fit_gates": 1.6639e11, "alpha_ratio": 0.998823,
                     "u_points": "0:0;0.999235:0;0.999235:788712;1:788712",
                     "u_in_points": "0:0;0.00105837:32768;0.00315535:65536;0.100031:262144;1:788712"},
                    "a 8,192, te_core 4.1's cut (the residual stream), where the model's is the rank outputs, 2 x 4 x "
                    "16 = 128 bits, the cut te_core 4.2 measures on the island's rank-16 rows (768): the record's token "
                    "kind about 70 times the model's; 48,512 trained values against the model's 49,152; u_in front-loaded "
                    "(a token's forward reads every adapter value in dependency order before most of its work), where "
                    "the model's A1' is even; d 0, te_core 4.1's stub, where the model's is the harness's decoder "
                    "(2.4e11 gates, 0.07% of c)"),)
             + island_checks("lora")),
    Workload("distill", "Distillation from a registered teacher",
             "A student trained on a registered teacher's outputs, the teacher's forward in each token block.",
             distill,
             (Dial("student_params", "parameters", REFERENCE_PARAMS, N_RANGE, "the trained student"),
              Dial("teacher_params", "parameters", 10 * REFERENCE_PARAMS, (1e7, 1e12), "the registered teacher"),
              Dial("teacher_share", "share of tokens", 1.0, (0.0, 1.0), "the student tokens the teacher runs on"),
              Dial("teacher_precision", "GEMM format", "bf16", ("bf16", "e4m3", "fp4"), "the teacher's weight GEMMs"),
              COMMON["seq_len"], COMMON["batch_tokens"], COMMON["optimizer"], COMMON["t_dense"],
              COMMON["tokens_per_param"], COMMON["vocab"],
              Dial("student_d_model", "d_model", None, (64, 8192),
                   "the student's width; None is the reference model's d/L (the island's half-width students: 256 of "
                   "s-t20, 384 of m-t20)"),
              Dial("teacher_d_model", "d_model", None, (64, 8192),
                   "the teacher's width; None is the reference model's d/L (m-t20's 768 on the island's 60M row)")),
             (("default", {}), ("no teacher", {"teacher_share": 0.0}), ("100x teacher", {"teacher_params": 100 * 45_029_888})),
             island_checks("distill")),
)}


def model(name: str, **dials) -> Model:
    return WORKLOADS[name].fn(**dials)


#: the console's columns, each with the te_core quantities whose labels give its source (te_core.standard_workloads'
#: `KEYS` and `LABELLED`: algorithm_policy's, with w_fwd and the two 4.x terms)
KEYS = ("C_gates", *algorithm_policy.INPUTS, "w_fit_gates", "alpha_ratio")
LABELLED = algorithm_policy.LABELLED | {"w_gates": ("token_gates", "tokens"), "w_fwd_gates": ("w_fwd_gates",),
                                        "w_fit_gates": ("token_gates", "tokens"), "alpha_ratio": ("affected_gates",)}
#: how near a record's value a model input must be to count as the record's (σ by the optimizer's share, 1 − σ): the
#: code's 5 KB header is 3e-5 of 32 N and the pilot's τ adds 0.1% of other inputs to the ids' 16 bits
MEASURED_REL = 2e-3
#: the other kinds' inputs, each with where it comes from (an island row's `Check.menu` measures `grad_bits`)
MENU_STATUS = {
    "fresh": f"assumed: the shape's products by the standard algorithm (dense_fresh, adapter_fresh, muon_fresh), each "
             f"counted as te_core.io_bound's kinds count a captured step's (its s-shape dense and rank-16 LoRA steps at "
             f"{IO_BOUND}: the counted gates exactly, rho within 1% and 2.4%, all of the gap |C|'s); no trace of these "
             f"workloads lists its products yet",
    "widths": "assumed: each product's operands at its GEMM's width and its partial sums in fp32, as passed (WIDTHS)",
    "parts": "assumed: a step's tokens the dials' batch, so b = ceil(N w / c) (te_core.formalism's _blocks)",
    "grad_bits": f"assumed: the gradient at 32 bits a trained value, every worker's in a DiLoCo step (the island's "
                 f"traces measure exactly this as grad_cut_bits: {ISLAND_STANDARD}, {ISLAND_DILOCO}, "
                 f"{ISLAND_LORA_ATTN}, {ISLAND_LORA_ALL}, {ISLAND_DISTILL})",
    "partial_bits": f"assumed: one block's partial at 32 bits a trained value, a worker's in DiLoCo (the island's "
                    f"traces measure exactly this as partial_cut_bits: {ISLAND_STANDARD}, {ISLAND_DILOCO}, "
                    f"{ISLAND_LORA_ATTN}, {ISLAND_LORA_ALL}, {ISLAND_DISTILL})",
    "op_update_bits": "assumed: the widest op's weight or factor at the width its ops read it (Shape.op_bits, "
                      "Adapter.op_bits): reading A's test, no trace's entries",
}


def _profile_agrees(model_points: str, record_points: str, tail: bool) -> bool:
    """Whether two profiles agree to `MEASURED_REL`: u (`tail`) by its free share and u(1), which are all s* reads
    where u(1) > k; u_in point by point."""
    a, b = algorithm_policy.profile(model_points), algorithm_policy.profile(record_points)
    if tail:
        a, b = ((next((x for x, v in p if v > 0), 1.0), p[-1][1]) for p in (a, b))
        return all(math.isclose(x, y, rel_tol=MEASURED_REL) for x, y in zip(a, b))
    return len(a) == len(b) and all(math.isclose(x, y, rel_tol=MEASURED_REL, abs_tol=1e-9)
                                    for p, q in zip(a, b) for x, y in zip(p, q))


def _agrees(key: str, a, b) -> bool:
    if key in ("u_points", "u_in_points"):
        return _profile_agrees(a, b, key == "u_points")
    if a is None or b is None:
        return a is b
    if key == "sigma":
        a, b = 1 - a, 1 - b
    return math.isclose(a, b, rel_tol=MEASURED_REL)


def _said(key: str, v) -> str:
    """A record's value as `input_status` quotes it: a profile by what s* and x0 read (its free share and u(1), or
    its value at x = 0)."""
    if key == "u_points":
        u = algorithm_policy.profile(v)
        return f"t {_fmt(next((x for x, b in u if b > 0), 1.0))} and u(1) {_fmt(u[-1][1])}"
    if key == "u_in_points":
        u = algorithm_policy.profile(v)
        return f"u_in {_fmt(max(b for x, b in u if x == 0))} at x = 0 of u(1) {_fmt(u[-1][1])}"
    return v if isinstance(v, str) else _fmt(v)


def input_status(m: Model, checks: tuple[Check, ...] = ()) -> dict:
    """Each column's source in the standard-workloads panel's form: "measured: " and the record where a check at the
    model's dials measures the model's value (to `MEASURED_REL`), else "assumed: " and the model's label, with what
    the records at its dials measure. The other kinds' inputs are `MENU_STATUS`'s, at every point."""
    me, col = m.measured, columns(m)
    same = [ck for ck in checks if model(m.workload, **ck.dials).dials == m.dials]
    labels = me.labels | {"w_fwd_gates": "model: a token's forward at te_core.gates' prices, no elementwise share"}
    out = {}
    for key in KEYS:
        keys = LABELLED["delta_bits"] if key == "delta_t_bits" and me.delta_t_bits is None else LABELLED[key]
        said = [labels[q] for q in keys if q in labels]
        label = next((x for x in said if algorithm_policy.kind(x)), said[0] if said else "model: no label")
        got = [(ck, ck.columns[key]) for ck in same if ck.columns.get(key) is not None]
        hit = next((ck for ck, v in got if _agrees(key, col[key], v) and ck.status.get(key, "measured") == "measured"),
                   None)
        out[key] = (f"measured: {hit.record}" if hit else f"assumed: {label.removeprefix('model: ')}"
                    + "".join(f"; {ck.record} measures {_said(key, v)}" for ck, v in got))
    return out


def point(m: Model, checks: tuple[Check, ...] = ()) -> dict:
    """A model's reference point: its dials, the token kind's columns, what they're estimated from and each input's
    source (`input_status`, beside the workload's `checks`), the other kinds' inputs (`menu`), ρ at each of
    `POLICIES`, the token kind's sensitivity bounds where its token blocks differ, and what it derived."""
    out = {"dials": m.dials, "inputs": columns(m), "estimated": algorithm_policy.estimated(m.measured),
           "input_status": input_status(m, checks), "menu": menu(m),
           "rho": {name: rho(m, c, k) for name, (c, k) in POLICIES.items()}, "derived": m.derived}
    if m.derived.get("kinds"):
        out["sensitivity"] = {name: sensitivity(m, c, k) for name, (c, k) in POLICIES.items()}
    return out


def recorded(ck: Check, m: Model) -> Model:
    """`ck`'s record as a model beside `m`, the model at its dials: the record's parts where the check has them (an
    island row's), else its columns in one part (its |C|, else the model's tokens at the record's w and σ; its tokens
    σ|C|/w, the model's steps, the record's Δ_grad, and the gradient inputs `Check.menu` gives, else the model's), with
    the columns' τ, a, profiles, Δ, d and Δ_t, and the model's fresh products on the record's tokens (each group's
    count by the record's tokens over the model's: no trace lists its products); a record's ops are unknown
    (`reading_a`). `rho` takes its token kind and Fit from page_rho on the columns, which one part can't give where
    the record's w_fit or ᾱ/α_max differ."""
    col = ck.columns
    if ck.parts:
        parts = tuple(formalism.Part(**p) for p in ck.parts)
    else:
        if "C_gates" in col:
            gates, tokens = col["C_gates"], col["sigma"] * col["C_gates"] / col["w_gates"]
        else:
            tokens = m.measured.tokens
            gates = tokens * col["w_gates"] / col["sigma"]
        main = max(m.measured.parts, key=lambda p: p.token_gates)
        parts = (formalism.Part(gates, tokens, col["sigma"] * gates, col["sigma"] * gates,
                                sum(p.steps for p in m.measured.parts), col["delta_grad_bits"],
                                ck.menu.get("partial_bits", main.partial_bits), ck.menu.get("grad_bits", main.grad_bits)),)
    u = algorithm_policy.profile(col["u_points"])
    me = formalism.Measured(parts, col["tau_bits"], col["a_bits"], u, algorithm_policy.profile(col["u_in_points"]),
                            col["delta_bits"], col["d_gates"], {}, next((x for x, b in u if b > 0), 0.0), None,
                            col["delta_t_bits"])
    scale = me.tokens / m.measured.tokens
    return replace(m, measured=me, w_fwd=col.get("w_fwd_gates", m.w_fwd), op_update_bits=None,
                   fresh=tuple(replace(f, count=f.count * scale) for f in m.fresh))


def _check(ck: Check, workload: str) -> dict:
    """A measured record beside its model (`Check`): at each policy `rho` on the record (`recorded`, its target by
    page_rho on its columns) and on the model at the check's dials, each kind's term, ρ, the target and the
    sensitivities (`CHECKED`); each record input's source (its `status`; the model's are its point's
    `input_status`)."""
    m = model(workload, **ck.dials)
    rec = recorded(ck, m)
    out = {"record": ck.record, "source": ck.source, "dials": m.dials, "differs": ck.differs,
           "record_inputs": ck.columns | ck.menu,
           "record_status": {key: ck.status.get(key, "measured") for key in (*ck.columns, *ck.menu)},
           "model_inputs": columns(m), "rho": {}}
    for name, (c, k) in POLICIES.items():
        sides = (("record", rho(rec, c, k, algorithm_policy.page_rho(ck.columns, c, k))), ("model", rho(m, c, k)))
        out["rho"][name] = {side: {key: x[key] for key in CHECKED} for side, x in sides}
    return out


#: what a check keeps of `rho` on its record and on the model
CHECKED = ("rho", "basis", "binding_kind", "terms", "partial", "target", "sensitivities", "fit_fails", "token_A",
           "unmeasured", "b")


def reference() -> dict:
    """Every workload's reference points and checks at `POLICIES` (module docstring): `workloads.json`."""
    out = {"id": "te_core.workloads", "te_core": VERSION, "reading": "A", "spec": NOTE, "io_bound": IO_BOUND,
           "policies": {name: {"c": c, "k": k, "what": POLICY_NOTES[name]} for name, (c, k) in POLICIES.items()},
           "valid": list(VALID), "sensitivities": SENSITIVITIES, "target": TARGET, "rests_on": RESTS_ON,
           "reading_a": VERDICTS_A, "menu_status": MENU_STATUS, "fresh_fields": FRESH_FIELDS,
           "part_fields": list(PART_FIELDS), "records": RECORDS.name,
           "widths": {kind: list(w) for kind, w in WIDTHS.items()},
           "gates_per_mac": {kind: per_mac(kind) for kind in ("gemm/bf16", "gemm/e4m3", "gemm/fp4", "gemm/f32",
                                                              "elementwise/bf16", "elementwise/f32")},
           "cell_gates": {kind: gates(PRICES[kind][0][0]) for kind in WIDTHS}, "workloads": {}}
    for w in WORKLOADS.values():
        out["workloads"][w.name] = {
            "title": w.title, "what": w.what,
            "dials": [{"name": d.name, "unit": d.unit, "default": d.default, "range": list(d.range), "why": d.why}
                      for d in w.dials],
            "points": {name: point(model(w.name, **settings), w.checks) for name, settings in w.points},
            "checks": {ck.name: _check(ck, w.name) for ck in w.checks}}
    return out


#: the significant digits `dumps` keeps of a float
DIGITS = 7


def _rounded(x):
    if isinstance(x, float) and math.isfinite(x):
        return float(f"{x:.{DIGITS}g}")
    if isinstance(x, dict):
        return {key: _rounded(v) for key, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rounded(v) for v in x]
    return x


def dumps(x, depth: int = 0, width: int = 160) -> str:
    """`x` as JSON, each float to `DIGITS` significant digits, indented by one space a level, each container that fits
    `width` on one line (`REFERENCE` stays under the repository's 256 KiB blob limit)."""
    if depth == 0:
        x = _rounded(x)
    flat = json.dumps(x, allow_nan=False)
    if not isinstance(x, (dict, list)) or len(flat) + depth <= width:
        return flat
    pad = " " * (depth + 1)
    body = ([f"{pad}{json.dumps(key)}: {dumps(v, depth + 1, width)}" for key, v in x.items()] if isinstance(x, dict)
            else [pad + dumps(v, depth + 1, width) for v in x])
    ends = "{}" if isinstance(x, dict) else "[]"
    return ends[0] + "\n" + ",\n".join(body) + "\n" + " " * depth + ends[1]


def _fmt(v) -> str:
    return "unmeasured" if v is None else f"{v:.3g}"


#: how the table writes reading A's verdict on the token and decoding kinds, whose terms are unmeasured under each
VERDICTS = {"unmodelled": "unmeasured", "refused": "refused"}
#: the table's columns of a point's `rho` at a policy: (heading, how to read it)
COLUMNS = (("fresh products", lambda r: r["terms"]["fresh_products"]),
           ("fresh, chain (sens.)", lambda r: r["sensitivities"]["fresh_chain"]),
           ("fresh, chain, P-cell (sens.)", lambda r: r["sensitivities"]["fresh_cell"]),
           ("target (B)", lambda r: r["target"]["rho"]),
           ("decoding, B (sens.)", lambda r: r["sensitivities"]["decoding_form"]),
           ("reduction (sens.)", lambda r: r["sensitivities"]["reduction"]),
           ("update (sens.)", lambda r: r["sensitivities"]["update"]),
           ("fresh share", lambda r: r["fresh"]["share"]))


def table(ref: dict) -> str:
    """Each policy's table: every point's terms (the valid kinds', the fresh products' without P-cell, io_bound's chain
    figures, the target and the sensitivities), the token kind's reading-A verdict, the Fit clauses that fail, ρ and
    its basis; then the
    checks', each on the record and on the model; then what each basis rests on and what each verdict means."""
    lines = []
    heads = [h for h, _ in COLUMNS] + ["token, decoding (A)"]
    for q in POLICIES:
        lines += [f"{q}: {ref['policies'][q]['what']}", "",
                  "| workload | point | " + " | ".join(heads) + " | Fit fails | rho | basis |",
                  "|---|---|" + "---|" * (len(heads) + 3)]
        for name, w in ref["workloads"].items():
            for pname, p in w["points"].items():
                r = p["rho"][q]
                cells = [_fmt(get(r)) for _, get in COLUMNS] + [VERDICTS[r["token_A"]]]
                fails = ", ".join(r["fit_fails"]) or "none"
                lines.append(f"| {name} | {pname} | " + " | ".join(cells)
                             + f" | {fails} | {_fmt(r['rho'])} | {r['basis']} |")
        lines.append("")
    for q in POLICIES:
        lines += [f"{q}, the checks: each term on the record / on the model at its dials", "",
                  "| workload | check | " + " | ".join(heads[:-2]) + " | rho | basis |",
                  "|---|---|" + "---|" * len(heads)]
        for name, w in ref["workloads"].items():
            for cname, ck in w["checks"].items():
                rm = [ck["rho"][q][side] for side in ("record", "model")]
                cells = [" / ".join(_fmt(get(r)) for r in rm) for _, get in COLUMNS[:-1]]
                lines.append(f"| {name} | {cname} | " + " | ".join(cells) + f" | {' / '.join(_fmt(r['rho']) for r in rm)}"
                             f" | {' / '.join(r['basis'] for r in rm)} |")
        lines.append("")
    lines += [f"- {basis}: {why}" for basis, why in ref["rests_on"].items()]
    lines += [f"- target (B): {ref['target']}"]
    lines += [f"- {kind} (sens.): {why}" for kind, why in ref["sensitivities"].items()]
    lines += [f"- token, decoding (A) {VERDICTS[v]}: {why}" for v, why in ref["reading_a"].items()]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path, help=f"write the reference points here ({REFERENCE.name} is the pinned copy)")
    ap.add_argument("--records", nargs=2, metavar=("TABLES", "ART"),
                    help=f"only write {RECORDS.name} from the island's training/standard-workloads tables file and "
                         "its art id (then run again for the reference points, whose checks read it)")
    a = ap.parse_args(argv)
    if a.records:
        path, art = a.records
        RECORDS.write_text(json.dumps(records(json.loads(Path(path).read_text()), art), indent=1) + "\n")
        print(f"{RECORDS}: {RECORDS.stat().st_size:,} bytes", file=sys.stderr)
        return 0
    ref = reference()
    if a.json:
        a.json.write_text(dumps(ref) + "\n")
    print(table(ref))
    return 0


if __name__ == "__main__":
    sys.exit(main())
