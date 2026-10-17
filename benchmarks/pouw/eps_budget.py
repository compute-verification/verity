"""Dither's practical ε as served, per model, context, phase and design (EPS-BUDGET, cmp-13; Derived).

    uv run python benchmarks/pouw/eps_budget.py [--json OUT] [--md OUT]

ε = (honest work a cheater can skip) / (the cheater's unavoidable work), the brief's definition
(note:20261017T0013Z-report-epsilon-program). Each component of honest work is classified per design as unavoidable (a
transcript Verify recomputes, a commitment, a draw, noise the cheater must expand, a value a check recomputes from the
transcript) or skippable (unchecked: the strip of a low-rank noise, the epilogue, the quantizer's floor, decode's clean
products, an unnoised lm_head). The cheater is the zero instance (A* = B* = 0, chosen before the draw), which makes every
strip, every served output and everything derived from it free unless a check recomputes it from the transcript.

- MAC-weighted: tensor MACs only, strips at their logical rank (L per output word).
- Time-weighted: every component's GPU time on one RTX PRO 6000 (sm_120) at its read clock, from the measured rates in
  RATES (each with its note and run), executed strips (two int8 limbs), epilogue and softmax included. Decode is
  memory-bound, the cheater's transcript is not (B̃ = S is generated from U, and the zero instance streams no weights),
  so decode's time ratio is a bandwidth gap no noise design closes; it is reported, not optimized.

Designs are cumulative: D0 the design of record (decisions 29, 36, 44, 46, 51; register note:20261016T0200Z), then
D1 Q Kᵀ dithered on both operands, D2 the linears dithered on both operands, D3 lm_head noised, D4 decode serving the
edited product with per-stage coins (no deferral), D5 every stage's operand derived by C1′ from the previous stage's
served output (the epilogue and quantizer recomputed by checks).

SKIP (note:20261017T0310Z-report-eps1-program) prices whole designs per component, in executed int8 products and honest
seconds, with γ: `record`, A both-sided bit dither (adopted-bd), B CO-Dither at r = k/2, B32 CO-Dither at r = 32, C
NCP-INT on k32 IMMA (`SKIP_DESIGNS`, `component_table`, `skip_table`).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

L = 32                      # transcript stride and the low-rank noise's rank
CONTEXTS = (2048, 8192, 32768)
PHASES = ("prefill", "decode")
DESIGNS = ("D0", "D1", "D2", "D3", "D4", "D5")
DESIGN_NAMES = {"D0": "of record", "D1": "+ QKᵀ dithered on both operands", "D2": "+ linears dithered on both operands",
                "D3": "+ lm_head noised", "D4": "+ decode served edited", "D5": "+ operands derived by C1′"}


@dataclass(frozen=True)
class Shape:
    h: int
    layers: int
    heads: int
    kv: int
    d: int
    inter: int
    vocab: int
    tp: int                 # GPUs a decode replica uses (the decode step below is per replica)

    @property
    def linears(self) -> list[tuple[str, int, int]]:
        """(name, k, n) per layer; one GEMM per stage (q/k/v fused, gate/up fused)."""
        return [("qkv", self.h, (self.heads + 2 * self.kv) * self.d), ("o", self.heads * self.d, self.h),
                ("gate_up", self.h, 2 * self.inter), ("down", self.inter, self.h)]


MODELS = {"llama-3.1-8b": Shape(4096, 32, 32, 8, 128, 14336, 128256, 1),
          "llama-3.1-70b": Shape(8192, 80, 64, 8, 128, 28672, 128256, 2)}

#: Rates, each measured (with its note) unless marked estimate.
RATES = {
    "peak": 188 * 1014 * 2.092e9,   # IMMA s8 MAC/s at the read clock (FP-EXACT, note:20261016T1931Z)
    "eta_lin": 0.80,     # prefill int8 GEMMs alone, 32 layers x 6 stages at 2K: 46.4 ms (STAGE-COST r20261016-095504-5463)
    "eta_att": 0.70,     # fused int8 attention, 60-70% of int8 peak (STAGE-COST r20261016-100833-1b7c)
    "eta_def": 0.71,     # deferred noised pass plateau, 0.97 µs a token-layer (STAGE-COST, same run)
    "r_receipt": 1.638,  # edited GEMM kernels / plain at S = 128: 1.668 (COMMIT-GPU r5 r20261016-183051-a379), less the
                         # 2-atom tail's 0.030 (EGEMM r20261016-090505-596b: 4 atoms 0.024 ms of 0.405 at o_proj)
    "tail": 1.9,         # a strip MAC costs 1.9 mainloop MACs (that 4-atom tail: 0.006 ms an atom against 32/4096 of 0.405)
    "t_quant": 0.56e-12,     # s an element, plain activation quantizer (COMMIT-GPU r5: +23 ms at 8 x 4K, 40.8 G elements)
    "t_dither": 0.48e-12,    # s an element, the dither's bits (EGEMM r20261016-104404-700b: qdither - qplain 0.016 ms, 8192 x 4096)
    "t_wedit": 0.86e-12,     # s a weight element a draw, W' = W + U written (EGEMM: wedit 0.0144 ms, 4096 x 4096)
    "t_xcommit": 1.10e-12,   # s an operand element, SHA-512 tile leaves R = 128 (COMMIT-GPU r5: +45 ms at 8 x 4K)
    "receipt_commit": 0.016,  # of plain prefill time, S = 128 (COMMIT-GPU r5)
    "draws": 0.019,           # of plain prefill time, 0.3 ms simulated draws at 8 x 4K (COMMIT-GPU r5)
    "tokens_per_draw": 32768,  # prefill stage set a GPU (STAGE-COST's recommendation)
    "softmax": 0.25,     # estimate: exp, max, sum and P's quantize per (query head, key) against 256 MACs a pair; MUFU at
                         # 16 a SM a clock against 1014 MACs gives 0.25 of the attention's tensor time (range 0.12-0.5)
    "nongemm_token": 4.68e-6,  # s a token of non-GEMM kernels at prefill, Llama-3.1-8B, vLLM (note:20261016T0215Z §2)
    "nongemm_step": 1.09e-3,   # s a decode step floor of the same kernels (same note)
    "bw": 1.41e12,       # B/s, decode attention's measured KV read (same note, dcap r20261016-023229-185e)
    "lin_bw_b32": 6.98e9 / 5.9e-3,  # B/s, 32-layer int8 linears at M = 32: 5.64 ms (M = 1) to 6.18 ms (M = 64) (same run)
    "step_fixed": 0.293e-3,    # s a decode step, the step model's constant (same note: within 1.7% of 14 steps)
    "comm_tp2_b32": 5.2e-3,    # s a TP-2 decode step of all-reduces, calibrated to HC-COMM r6's base step at B 32, 2K
    "deferred_commit": 0.012,  # of the decode step, committing the stored operands (STAGE-COST: 0.9-1.4%)
    "spot_certified": 0.998,   # the certified fraction's statistical factor φψ (note:20261016T0711Z-report-rs-definition-v3)
}
DECODE_BATCH = 32


def epilogue_bytes(s: Shape) -> int:
    """Bytes a token-layer the non-GEMM kernels move (bf16, fused): RoPE on Q and K, two residual adds with RMSNorm,
    SiLU x up, the KV cache write. Only its ratio between models is used: the time is vLLM's measured 8B figure."""
    return 4 * (s.heads + s.kv) * s.d + 12 * s.h + 6 * s.inter + 2 * s.kv * s.d


@dataclass
class Point:
    model: str
    ctx: int
    phase: str
    design: str
    macs_skip: float
    macs_need: float
    time_skip: float
    time_need: float
    parts: dict

    @property
    def eps_mac(self) -> float:
        return self.macs_skip / self.macs_need

    @property
    def eps_time(self) -> float:
        return self.time_skip / self.time_need


def decode_step(name: str, ctx: int, rates: dict = RATES) -> dict:
    """A plain decode step at batch DECODE_BATCH, in GPU-seconds a token-layer: `step` the whole, `gemm_dram` the part
    that streams the linears' weights and the KV cache (the time the GEMMs' tensor work hides under)."""
    s, r = MODELS[name], rates
    tokens = DECODE_BATCH * s.layers
    lin = sum(k * n for _, k, n in s.linears)
    att_bytes = DECODE_BATCH * ctx * s.layers * 2 * s.kv * s.d
    dram = (s.layers * lin / r["lin_bw_b32"] + att_bytes / r["bw"]) / s.tp
    step = dram + r["step_fixed"] + r["nongemm_step"] + (r["comm_tp2_b32"] if s.tp == 2 else 0)
    return {"step": step * s.tp / tokens, "gemm_dram": dram * s.tp / tokens}


def at_least(design: str, d: str) -> bool:
    return DESIGNS.index(design) >= DESIGNS.index(d)


def point(name: str, ctx: int, phase: str, design: str, rates: dict = RATES) -> Point:
    """One (model, context, phase, design): per token-layer MACs and seconds, split into skippable and unavoidable."""
    s, r = MODELS[name], rates
    keys = (ctx + 1) / 2 if phase == "prefill" else ctx
    lin = sum(k * n for _, k, n in s.linears)
    qk = pv = s.heads * s.d * keys
    lm = s.vocab * s.h / s.layers / (ctx if phase == "prefill" else 1)  # prefill serves the last position's logits
    weights = lin
    d1, d2, d3, d4, d5 = (at_least(design, x) for x in ("D1", "D2", "D3", "D4", "D5"))
    strip = {"linears": 0 if d2 else L * sum(n for *_, n in s.linears),
             "qk": 0 if d1 else min(L / s.d, 1) * qk,
             "pv": min(L + 1, keys) * s.heads * s.d}
    t_lin = 1 / (r["eta_lin"] * r["peak"])
    t_att = 1 / (r["eta_att"] * r["peak"])
    t_def = 1 / (r["eta_def"] * r["peak"])
    act = 3 * s.h + s.inter  # quantized linear inputs a token-layer (qkv, o, gate_up, down)
    qkv_el = (s.heads + 2 * s.kv) * s.d
    skip_m, need_m, skip_t, need_t = {}, {}, {}, {}

    noised = lin + qk + pv + (lm if d3 else 0)
    if phase == "prefill" or d4:
        need_m["transcripts"] = noised
        skip_m.update({f"strip {k}": v for k, v in strip.items() if v})
        if not d3:
            skip_m["lm_head"] = lm
    else:
        need_m["deferred transcripts"] = noised
        skip_m["clean products"] = lin + qk + pv
        skip_m["lm_head"] = lm

    plain = lin * t_lin + (qk + pv) * t_att
    if phase == "prefill":
        need_t["transcripts linears"] = lin * t_lin * r["r_receipt"]
        need_t["transcripts attention"] = (qk + pv) * t_att * r["r_receipt"]
        if d3:
            need_t["transcript lm_head"] = lm * t_lin * r["r_receipt"]
        else:
            skip_t["lm_head"] = lm * t_lin
        if strip["linears"]:
            skip_t["strip linears"] = 2 * strip["linears"] * r["tail"] * t_lin
        if strip["qk"]:
            skip_t["strip qk"] = 2 * strip["qk"] * t_att
        skip_t["strip pv"] = 2 * strip["pv"] * r["tail"] * t_att
        (need_t if d1 else skip_t)["softmax"] = r["softmax"] * (qk + pv) * t_att
        quant = r["t_quant"] * (act + qkv_el)
        (need_t if d5 else skip_t)["quantizer"] = quant
        epi = r["nongemm_token"] / MODELS["llama-3.1-8b"].layers * epilogue_bytes(s) / epilogue_bytes(MODELS["llama-3.1-8b"])
        (need_t if d5 else skip_t)["epilogue"] = max(epi - quant, 0)
        dithered = act + s.heads * s.d * 2 + s.kv * s.d + (s.heads * s.d if d1 else 0)
        need_t["dither bits"] = r["t_dither"] * (dithered + (weights / r["tokens_per_draw"] if d2 else 0))
        skip_t["noised weight write"] = r["t_wedit"] * weights / r["tokens_per_draw"]
        need_t["operand commitments"] = r["t_xcommit"] * (act + qkv_el)
        need_t["receipt commitments"] = r["receipt_commit"] * plain
        need_t["draws"] = r["draws"] * plain
    else:
        gpu_s = decode_step(name, ctx, r)["step"]
        transcripts = (lin + (lm if d3 else 0)) * t_def * r["r_receipt"] + (qk + pv) * t_att * r["r_receipt"]
        need_t["transcripts (compute-bound)"] = transcripts
        need_t["operand commitments"] = r["deferred_commit"] * gpu_s
        if d4:
            skip_t["served step beyond the transcripts (weight and KV streams, epilogue, comm)"] = max(gpu_s - transcripts, 0)
            skip_t["strips"] = (2 * strip["linears"] * r["tail"] * t_lin + 2 * strip["qk"] * t_att
                                + 2 * strip["pv"] * r["tail"] * t_att)
        else:
            skip_t["clean step"] = gpu_s
    return Point(name, ctx, phase, design, sum(skip_m.values()), sum(need_m.values()), sum(skip_t.values()),
                 sum(need_t.values()), {"macs_skip": skip_m, "macs_need": need_m, "time_skip": skip_t, "time_need": need_t})


def compiled_extra(rates: dict = RATES) -> float:
    """What the compiled protocol adds beyond the ideal one: the spot checks certify φψ of the transcript words, so up to
    1/φψ - 1 of them can be skipped at δ. Receipts (duo under H-μ⁺) and commitments add nothing a cheater skips."""
    return 1 / rates["spot_certified"] - 1


# ------------------------------------------------- SKIP: ε_skip per component and γ (note:20261017T0310Z-report-eps1-program)

SKIP_DESIGNS = {
    "record": "Dither of record: U on the weights and on Q (peels), P's peel, lm_head clean, forming unchecked, decode "
              "deferred (decision 36)",
    "A": "both operands dithered with fresh bits and no U (adopted-bd), the served output the last checked word, "
         "lm_head the same, P V as recorded with its output checked (its whole strip free at a one-hot P row)",
    "B": "CO-Dither at r = k/2 with a 7-bit served side: U on the peeled side, bits on both, the served output a "
         "checked word, one limb; P V as in A",
    "B32": "CO-Dither at r = 32 with an 8-bit served side: the strip's second int8 limb unforced, r/(k + r)",
    "Bc": "CHECKED-STRIP's CO-Dither: co32c on the linears and lm_head (r = 32, the residue sums' base-128 digits "
          "committed, so every limb is checked; 7-bit weights, 8-bit activations), B (cohalf) on Q Kᵀ, and P V with "
          "its strip's digits committed the same way (P peeled with U and a fresh bit, V served, its residue sums' "
          "digits committed)",
    "Ap": "A with Bc's P V (its strip's digits committed)",
    "C": "NCP-INT on k32 IMMA: three noised copies cancelling in one depth-3k product, the exact output the last "
         "checked word, so at the zero instance the last segment (L of 3k) is known",
    "CB": "NCP-INT on the linears and lm_head, CO-Dither at r = d/2 (7-bit) on Q Kᵀ, P V as in A: wide noise on every "
          "product",
}
#: hybrids: (design for the linears and lm_head, for Q Kᵀ, for P V); `pv` is P V with its strip's digits committed
HYBRIDS = {"CB": ("C", "B", "A"), "Bc": ("co32c", "B", "pv"), "Ap": ("A", "A", "pv")}
#: co32c's digit limbs: the residue sums of an 8-bit served side over k / 32 terms take up to 3 base-128 digits
CO32C_DIGITS = 3


def pv_digits(keys: int) -> int:
    """m, the base-128 digits of P V's Ycum (CHECKED-STRIP's P3): 2 up to 2K keys, 3 up to 128K."""
    return 2 if keys <= 2048 else 3
#: Every SKIP design but `record` derives each stage's operand by C1′ from the previous stage's served output (the
#: epilogue and quantizer are checked) and serves decode edited, with coins drawn per round once the weights and the
#: round's inputs are committed (no deferred pass, no per-stage round trip).
COMPONENTS = ("linears", "lm_head", "QKᵀ", "P·V", "softmax", "forming", "noise, commitments, draws")


def leaf(design: str, comp: str) -> str:
    """The design that serves `comp`: a hybrid's part, else the design itself."""
    return HYBRIDS[design][{"QKᵀ": 1, "P·V": 2}.get(comp, 0)] if design in HYBRIDS else design


def products(design: str, comp: str, k: int) -> tuple[float, float, float]:
    """(transcript, strip, skippable) int8 products an output word of depth k executes. A strip runs on two int8 limbs
    when its operand is a residue sum that leaves int8 (L = 32 classes of k / L terms), on one when it fits (CO-Dither's
    r = k/2 with a 7-bit served side: two terms), on checked digit limbs when the sums' base-128 digits are committed
    (co32c; pv, (m + 1)·L for m digits). Adopted P V's whole strip is free: at a one-hot P row its output is a copy
    (CHECKED-STRIP's Result 3)."""
    design = leaf(design, comp)
    if comp == "P·V":
        if design == "C":
            return 3 * k, 0, L
        if design == "pv":
            return k, (pv_digits(k) + 1) * L, 0
        return k, 2 * (L + 1), 2 * (L + 1)
    if design == "record":
        return k, 2 * L, 2 * L
    if design == "A":
        return k, 0, 0
    if design == "B":
        return k, k // 2, 0
    if design == "B32":
        return k, 2 * L, L
    if design == "co32c":
        return k, CO32C_DIGITS * L, 0
    if design == "C":
        return 3 * k, 0, L
    raise ValueError(design)


@dataclass
class Part:
    component: str
    macs: float          # int8 products a token-layer the honest prover executes
    macs_skip: float     # of them, the ones Verify doesn't require
    seconds: float       # honest GPU-seconds a token-layer
    seconds_skip: float
    plain: float         # GPU-seconds of the same component served as plain int8


def parts(name: str, ctx: int, phase: str, design: str, rates: dict = RATES) -> list[Part]:
    """Per component, a token-layer: honest int8 products, the skippable ones, honest seconds, skippable seconds and plain
    int8 seconds. Decode's seconds are the GEMMs' compute, which the DRAM-bound step hides (`decode_gamma`); decode's
    binding facet is DRAM (`decode_dram`)."""
    s, r = MODELS[name], rates
    keys = (ctx + 1) / 2 if phase == "prefill" else ctx
    t_lin = 1 / (r["eta_lin"] * r["peak"])
    t_att = 1 / (r["eta_att"] * r["peak"])
    deferred = design == "record" and phase == "decode"
    acc: dict[str, list[float]] = {}

    def gemm(comp, words, k, t):
        if design == "record" and comp == "lm_head":
            c = words * k
            row = [c, c, c * t, c * t, c * t]       # served plain, never proved
        else:
            tr, st, sk = products(design, comp, k)
            short = st and (leaf(design, comp) != "B" or st <= 2 * (L + 1))     # limbs of L or L + 1 terms
            strip_t = t * (r["tail"] if short else 1)
            row = [words * (tr + st), words * sk, words * (tr * t * r["r_receipt"] + st * strip_t),
                   words * sk * (strip_t if st else t * r["r_receipt"]), words * k * t]
            if deferred:                            # the served clean product, whose noised twin the deferred pass proves
                row = [row[0] + words * k, row[1] + words * k, row[2] + words * k * t, row[3] + words * k * t, row[4]]
        acc[comp] = [a + b for a, b in zip(acc.get(comp, [0] * 5), row)]

    for _, k, n in s.linears:
        gemm("linears", n, k, t_lin)
    gemm("lm_head", s.vocab / s.layers / (ctx if phase == "prefill" else 1), s.h, t_lin)
    gemm("QKᵀ", s.heads * keys, s.d, t_att)
    gemm("P·V", s.heads * s.d, int(keys), t_att)
    out = [Part(c, *v) for c, v in acc.items()]
    att_plain = 2 * s.heads * s.d * keys * t_att
    soft = r["softmax"] * att_plain
    out.append(Part("softmax", 0, 0, soft, soft if design == "record" else 0, soft))
    act = 3 * s.h + s.inter
    qkv_el = (s.heads + 2 * s.kv) * s.d
    epi = r["nongemm_token"] / MODELS["llama-3.1-8b"].layers * epilogue_bytes(s) / epilogue_bytes(MODELS["llama-3.1-8b"])
    form = max(epi, r["t_quant"] * (act + qkv_el))
    out.append(Part("forming", 0, 0, form, form if design == "record" else 0, form))
    lin_plain = sum(k * n for _, k, n in s.linears) * t_lin
    weights = sum(k * n for _, k, n in s.linears)
    copies = 3 if design in ("C", "CB") else 1
    extra = (r["t_dither"] * copies * (act + qkv_el + s.heads * s.d + weights / r["tokens_per_draw"])
             + r["t_xcommit"] * (act + qkv_el) + (r["receipt_commit"] + r["draws"]) * (lin_plain + att_plain))
    if design == "record":
        extra += r["t_wedit"] * weights / r["tokens_per_draw"]
    out.append(Part("noise, commitments, draws", 0, 0, extra, 0, 0))
    return out


def decode_dram(name: str, ctx: int, rates: dict = RATES) -> float:
    """What the zero-weight, zero-KV instance (B = 0 with B̃ generated from the coins, K = V = 0) saves on decode's binding
    facet, DRAM, as a fraction of its own (compute-bound) transcripts: the plain step beyond the transcripts. Binding B to
    a registered weight matrix with every operand derived by C1′ leaves no zero instance, and this term goes to 0."""
    p = point(name, ctx, "decode", "D4", rates)
    return (p.parts["time_skip"]["served step beyond the transcripts (weight and KV streams, epilogue, comm)"]
            / p.parts["time_need"]["transcripts (compute-bound)"])


@dataclass
class Summary:
    model: str
    ctx: int
    phase: str
    design: str
    eps_mac: float
    eps_time: float
    gamma: float
    rows: list


def summary(name: str, ctx: int, phase: str, design: str, rates: dict = RATES) -> Summary:
    """ε_skip = skippable / (honest - skippable), by int8 products and by honest seconds, whole and per component (each
    component's skippable share over the same whole denominator); γ = honest / plain int8 time."""
    ps = parts(name, ctx, phase, design, rates)
    macs, secs = sum(p.macs for p in ps), sum(p.seconds for p in ps)
    skip_m, skip_t = sum(p.macs_skip for p in ps), sum(p.seconds_skip for p in ps)
    rows = [{"component": p.component, "mac_share": p.macs / macs, "eps_mac": p.macs_skip / (macs - skip_m),
             "eps_time": p.seconds_skip / (secs - skip_t), "gamma": p.seconds / p.plain if p.plain else None}
            for p in ps]
    if phase == "decode":
        gemm = sum(p.seconds for p in ps if p.component in ("linears", "lm_head", "QKᵀ", "P·V"))
        gamma = decode_gamma(name, ctx, design, gemm, rates)
    else:
        gamma = secs / sum(p.plain for p in ps)
    return Summary(name, ctx, phase, design, skip_m / (macs - skip_m), skip_t / (secs - skip_t), gamma, rows)


def decode_gamma(name: str, ctx: int, design: str, gemm: float, rates: dict = RATES) -> float:
    """Honest decode step / plain decode step (DRAM-bound, `decode_step`). The edited step adds the GEMMs' compute where
    it exceeds the DRAM time it hides under, and the operand commitments; `record` runs its deferred pass on top."""
    st = decode_step(name, ctx, rates)
    commit = rates["deferred_commit"] * st["step"]
    if design == "record":
        return (st["step"] + gemm + commit) / st["step"]
    return (st["step"] + max(gemm - st["gemm_dram"], 0) + commit) / st["step"]


def skip_table(designs=tuple(SKIP_DESIGNS), rates: dict = RATES) -> str:
    head = "| model | context | phase | " + " | ".join(f"{d}: ε_skip MAC / time, γ" for d in designs) + " |"
    lines = [head, "|---|---:|---|" + "---:|" * len(designs)]
    for m in MODELS:
        for c in CONTEXTS:
            for ph in PHASES:
                cells = []
                for d in designs:
                    q = summary(m, c, ph, d, rates)
                    cells.append(f"{100 * q.eps_mac:.3g}% / {100 * q.eps_time:.3g}%, {q.gamma:.2f}")
                lines.append(f"| {m} | {c // 1024}K | {ph} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def component_table(name: str, ctx: int, phase: str, designs=tuple(SKIP_DESIGNS), rates: dict = RATES) -> str:
    sums = {d: summary(name, ctx, phase, d, rates) for d in designs}
    lines = [(f"{name}, {ctx // 1024}K, {phase}: per component, ε_skip by int8 products / by time (each over the whole), "
              "component γ"), "", "| component | MAC share (A) | " + " | ".join(designs) + " |",
             "|---|---:|" + "---:|" * len(designs)]
    for i, comp in enumerate(COMPONENTS):
        cells = []
        for d in designs:
            row = sums[d].rows[i]
            g = f", γ {row['gamma']:.2f}" if row["gamma"] else ""
            cells.append(f"{100 * row['eps_mac']:.3g}% / {100 * row['eps_time']:.3g}%{g}")
        share = sums["A"].rows[i]["mac_share"] if "A" in sums else sums[designs[0]].rows[i]["mac_share"]
        lines.append(f"| {comp} | {100 * share:.3g}% | " + " | ".join(cells) + " |")
    lines.append("| **whole** | 100% | " + " | ".join(
        f"**{100 * sums[d].eps_mac:.3g}% / {100 * sums[d].eps_time:.3g}%, γ {sums[d].gamma:.2f}**" for d in designs)
        + " |")
    return "\n".join(lines)


def table(points: list[Point], designs=DESIGNS) -> str:
    head = "| model | context | phase | " + " | ".join(f"{d} ε MAC / time" for d in designs) + " |"
    lines = [head, "|---|---:|---|" + "---:|" * len(designs)]
    by = {(p.model, p.ctx, p.phase, p.design): p for p in points}
    for m in MODELS:
        for c in CONTEXTS:
            for ph in PHASES:
                cells = [f"{100 * by[m, c, ph, d].eps_mac:.3g}% / {100 * by[m, c, ph, d].eps_time:.3g}%" for d in designs]
                lines.append(f"| {m} | {c // 1024}K | {ph} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def all_points(rates: dict = RATES) -> list[Point]:
    return [point(m, c, ph, d, rates) for m in MODELS for c in CONTEXTS for ph in PHASES for d in DESIGNS]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--json")
    p.add_argument("--md")
    a = p.parse_args(argv)
    pts = all_points()
    sums = [summary(m, c, ph, d) for m in MODELS for c in CONTEXTS for ph in PHASES for d in SKIP_DESIGNS]
    text = "\n".join(["Designs: " + "; ".join(f"{k} {v}" for k, v in DESIGN_NAMES.items()), "", table(pts), "",
                      f"Compiled protocol adds {100 * compiled_extra():.2f}% (spot checks).", "",
                      "SKIP designs: " + "; ".join(f"{k}: {v}" for k, v in SKIP_DESIGNS.items()), "", skip_table(), "",
                      *(component_table(m, c, ph) + "\n" for m in MODELS for c in (2048, 32768) for ph in PHASES),
                      "Decode's zero-weight instance saves, on DRAM, this multiple of its transcripts' compute: "
                      + ", ".join(f"{m} {c // 1024}K {decode_dram(m, c):.2f}" for m in MODELS for c in CONTEXTS)])
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps({"rates": RATES, "points": [
            {**asdict(q), "eps_mac": q.eps_mac, "eps_time": q.eps_time} for q in pts],
            "skip": [asdict(q) for q in sums],
            "decode_zero_weight_gap": {f"{m}/{c}": decode_dram(m, c) for m in MODELS for c in CONTEXTS}}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
