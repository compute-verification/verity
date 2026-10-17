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
        tokens = DECODE_BATCH * s.layers
        att_bytes = DECODE_BATCH * ctx * s.layers * 2 * s.kv * s.d
        step = (s.layers * lin / r["lin_bw_b32"] + att_bytes / r["bw"]) / s.tp + r["step_fixed"] \
            + r["nongemm_step"] + (r["comm_tp2_b32"] if s.tp == 2 else 0)
        gpu_s = step * s.tp / tokens          # GPU-seconds a token-layer
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
    text = "\n".join([f"Designs: " + "; ".join(f"{k} {v}" for k, v in DESIGN_NAMES.items()), "", table(pts), "",
                      f"Compiled protocol adds {100 * compiled_extra():.2f}% (spot checks)."])
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps({"rates": RATES, "points": [
            {**asdict(q), "eps_mac": q.eps_mac, "eps_time": q.eps_time} for q in pts]}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
