"""Serving under the audit: per GPU, the throughput, TPOT and TTFT of a prompt/generation mix, decode and prefill
priced apart, the SM-partitioned filler at its measured cost, and the KV cache's design options each with its γ under
Conservation.

    uv run python benchmarks/overhead_frontier/serving.py --out DIR      writes DIR/serving.json, prints the tables

A decode step is ``model.step``'s audited step (no top-up: the filler is priced as measured) at the weights' codec,
its attention and KV append at the KV's, plus the engine's own ops a step (``engine``); stock is the plain step plus
the same. A prompt token costs stock's measured chunk (the GEMMs at ``phi_prefill``, attention at ``phi_fa`` over
the prompt before it, the rest), and under the audit the weights' decode to scratch a chunk (``scratch``) and the KV
encode, k·x₁_h read-times a byte, which ra-vllm-encoded-stack measured at 164.5 µs a BF16 token (74% of a chunk).

A replica serves the mix's requests at λ by continuous batching: it decodes whenever a sequence runs and prefills
each prompt in chunks between steps, so in the steady state λ·(prefill + Ḡ·t_d(B)/B) = 1 with B = λ·Ḡ·TPOT running
sequences (Little). Throughput is the largest λ with B ≤ ``b_max`` (and the KV in HBM): 1/(prefill + Ḡ·t_d(B)/B).
Each arm is put beside stock at stock's utilization ρ₀ on trace 1 (``lambda0`` on one GPU): it takes r =
λ_max(stock)/λ_max(arm) times the GPUs, and its TPOT and TTFT are at ρ₀ of its own throughput, a burst of
``burst``/r requests a replica, the middle one's TTFT ((n + 1)/2)·P̄ prompt tokens at t_pf + t_d/chunk each. A filler
multiplies every GPU time by its measured TPOT factor and leaves its measured η.

γ (against R·T, the KV option's part): μ_rebuild of the weakest registered class, weights or KV (masked, so no
μ_partial); μ_kv, the share of R·T spent reading KV that isn't registered (options b, c, d) and, unregistered, its
writes; μ_scratch, the prefill's weight scratch (written and read a chunk, unregistered by design). The rest of γ
(the frontier's per-pass terms, sampling, λ) and η are the same for every option and are reported beside them.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inputs as I  # noqa: E402
from model import SHAPES, Params, Shape, Workload, _rebuild, at_width, decode, gamma_cap, interp, step  # noqa: E402

BLOCK = 4096
K_S = 16
STATED_W = (3, 1024)
REST = ("mu_chip", "mu_unreg", "mu_timing", "mu_sample", "mu_serve", "lambda", "mu_sketch")

SERVING: dict[str, I.Input] = {
    "phi_prefill": I.Input(648e12, "FLOP/s per GPU, FP8", "stock vLLM's CUTLASS FP8 GEMMs in a 2,048-token chunk of "
                           "Llama-3.1-8B, 44.10 ms (ra-vllm-encoded-stack r20261013-084112-b76a, art:fc1431a7)",
                           "measured"),
    "phi_fa": I.Input(200e12, "FLOP/s per GPU", "FlashAttention in the same chunks, 6.13 ms at 0–2,048 tokens of "
                      "context and 15.18 at 2,048–4,096 (179 and 217 TFLOP/s; the stack's the same)", "measured"),
    "prefill_rest": I.Input(24.4e-3, "s a 2,048-token chunk at 8B", "stock's 74.97 ms chunk less its GEMMs and "
                            "attention: norms, rotary, the KV write and 18 ms the profile leaves unnamed (the stack's "
                            "chunk has the same); scaled by layers × hidden / tp elsewhere, an estimate", "measured"),
    "scratch": I.Input(39.83e-3 / 8.03e9, "s a weight byte a chunk", "the stack's 451.6 ms chunk less its KV encode "
                       "(336.8) less stock's chunk (74.97): 39.8 ms for 8.03 GB of weight words decoded to scratch "
                       "(row-major 24.73, the LM head 5.76, the KV gather and the scratch's traffic) at the stated "
                       "codec (r20261013-085555-3005, art:77776cf6); another codec by its weight decode a byte "
                       "(`model.decode`)", "measured"),
    "engine": I.Input(1.43e-3, "s a decode step at 32 layers", "stock vLLM's own ops (norms, activation "
                      "quantization, rotary, SiLU, sampling, copies) 1.04 ms and its idle between kernels 0.39 at "
                      "32 × 1536 (r20261013-084112-b76a; the stack's 1.95 less its integration); by layers elsewhere, "
                      "an estimate", "measured"),
    "chunk": I.Input(2048, "tokens", "--max-batched-tokens in every measured run", "policy"),
    "b_max": I.Input(128, "sequences", "the stack captures decode graphs to 128 rows (da19b47ff); stock the same, "
                     "for a like comparison", "policy"),
    "lambda0": I.Input(4.0, "requests/s", "ra-kv-filler's trace 1 on one GPU: bursts of mean 12 at exponential gaps "
                       "of mean 3 s (4.4k prompt tokens a second)", "measured"),
    "burst": I.Input(12.0, "requests", "trace 1's mean burst", "measured"),
}

# The filler as measured on the stated stack at 32 sequences (ra-vllm-encoded-stack milestone 4): its TPOT over the
# unfilled stack's 67.88 ms, and η after it on the GPU's clock; stock's filler (#1918) for stock.
FILLERS: dict[str, tuple[float, float, str]] = {
    "none": (1.0, float("nan"), "no filler; η as the engine leaves it (stack 0.87, stock 0.42)"),
    "F48/16w": (101.00 / 67.88, 0.118, "green-context partition, 48 SMs, 16 warps (r20261013-073858-a598, "
                "art:27e10f0f)"),
    "F64/16w": (117.24 / 67.88, 0.087, "64 SMs, 16 warps (r20261013-073858-a598)"),
    "F64/8w": (107.46 / 67.88, 0.099, "64 SMs, 8 warps (r20261013-080630-2524, art:720da8fd)"),
    "F96/8w": (136.28 / 67.88, 0.078, "96 SMs, 8 warps, the least η (r20261013-080630-2524)"),
}
STOCK_FILLER = (16.88 / 10.69, 0.064, "stock's seq@32/16, unpartitioned (#1918; r20261013-080534-e16e, "
                "art:1f418937)")

# What the model reproduces (8B, BF16 KV, one GPU), ra-vllm-encoded-stack after its integration fixes.
MEASURED = {
    "stack decode 32x1536": (32, 1536, 40.39e-3, "r20261013-085555-3005"),
    "stack decode 32x1280": (32, 1280, 38.34e-3, "r20261013-085555-3005"),
    "stack decode 64x1536": (64, 1536, 61.43e-3, "r20261013-090958-eec7"),
    "stack decode 128x1536": (128, 1536, 95.84e-3, "r20261013-090958-eec7: two weight passes at 128 rows"),
    "stock decode 32x1536": (32, 1536, 12.40e-3, "r20261013-084112-b76a"),
    "stock decode 32x1280": (32, 1280, 11.60e-3, "r20261013-082452-ebe5"),
    "stack chunk": (2048, 0, 451.6e-3, "r20261013-085555-3005: of it the KV encode 336.8"),
    "stock chunk": (2048, 0, 74.97e-3, "r20261013-084112-b76a"),
    "stock chunk 2": (2048, 2048, 83.11e-3, "r20261013-084112-b76a: the second chunk, at 2,048 of context"),
    "stock TPOT": (0, 0, 10.69e-3, "trace 1, 32 seqs (r20261013-080534-e16e)"),
    "stock TTFT": (0, 0, 0.24, "trace 1, median"),
    "stack TPOT 32": (32, 0, 60.22e-3, "trace 1, 32 seqs (r20261013-085529-e4f4, art:c031ce4c): slot-bound"),
    "stack TTFT 32": (32, 0, 16.88, "trace 1, 32 seqs, median: the wait for a slot, over a 40 s trace"),
}

# Each mix is classes (prompt tokens, generated tokens, weight), every prompt token computed (no prefix cache).
MIXES: dict[str, list[tuple[int, int, float]]] = {
    "trace 1": [(p, g, 1.0) for p in (256, 1024, 2048) for g in (64, 256, 512)],
    "decode-heavy": [(p, g, 1.0) for p in (256, 1024) for g in (1024, 2048)],
    "prompt-heavy": [(p, g, 1.0) for p in (4096, 8192) for g in (128, 256)],
}
RATIOS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
MODELS: dict[str, int] = {"8B": 1, "70B": 1, "405B": 8}


@dataclass(frozen=True)
class Arm:
    name: str
    what: str
    audited: bool = True
    k_w: int = 3
    bits_w: int = 1024
    k_kv: int = 3
    bits_kv: int = 1024
    kv_dtype: int = 2
    kv: str = "registered"
    after: int = 0
    lag: float = 0.0
    host: bool = False


ARMS: list[Arm] = [
    Arm("stated", "masked p1024 at k = 3, weights and KV, KV registered at its append (as measured)"),
    Arm("a KV k=2", "the KV at k = 2, the weights at k = 3", k_kv=2),
    Arm("a all k=2", "the fallback: the weights and the KV at k = 2", k_w=2, k_kv=2),
    Arm("a KV p512 k=3", "the KV at 512 bits, k = 3, the weights at p1024", bits_kv=512),
    Arm("a all p512 k=3", "the whole store at 512 bits, k = 3", bits_w=512, bits_kv=512),
    Arm("a all p512 k=2", "the whole store at 512 bits, k = 2", k_w=2, bits_w=512, k_kv=2, bits_kv=512),
    Arm("b after 64", "KV registered once read 64 times, unregistered and plain till then", kv="after", after=64),
    Arm("b after 256", "KV registered once read 256 times", kv="after", after=256),
    Arm("c KV unregistered", "the weights registered at k = 3, the KV plain and unregistered", kv="plain"),
    Arm("c KV unreg, k=2", "the weights at k = 2, the KV plain and unregistered", k_w=2, kv="plain"),
    Arm("d deferred 0.5 s", "the prompt's KV written plain and encoded off the TTFT path within 0.5 s",
        kv="deferred", lag=0.5),
    Arm("d deferred 2 s", "the same within 2 s", kv="deferred", lag=2.0),
    Arm("e1 FP8 KV", "the stated codec over an FP8 KV cache (the operator's dtype)", kv_dtype=1),
    Arm("e1 FP8 KV, all k=2", "FP8 KV, the weights and KV at k = 2", k_w=2, k_kv=2, kv_dtype=1),
    Arm("e2 host encodes", "the host's IFMA takes its share of the encode (h of the GPUs' squaring)", host=True),
    Arm("rec: all k=2 + d 0.5 s", "the fallback, the prompt's encode deferred within 0.5 s", k_w=2, k_kv=2,
        kv="deferred", lag=0.5),
    Arm("rec + FP8 KV", "the same over an FP8 KV cache", k_w=2, k_kv=2, kv_dtype=1, kv="deferred", lag=0.5),
]
STOCK = Arm("stock", "vLLM, no audit", audited=False)
STOCK_FP8 = Arm("stock FP8 KV", "vLLM, no audit, FP8 KV", audited=False, kv_dtype=1)
PROJECTED = ("stated", "a all k=2", "c KV unregistered", "e1 FP8 KV, all k=2", "rec: all k=2 + d 0.5 s",
             "rec + FP8 KV")
SWEPT = ("stated", "a all k=2", "a all p512 k=2", "c KV unregistered", "e1 FP8 KV, all k=2", "rec: all k=2 + d 0.5 s")


def values() -> dict[str, Any]:
    return {**I.values(), **I.values(SERVING)}


def kv_shape(shape: Shape, dtype: int) -> Shape:
    return replace(shape, kv_heads=shape.kv_heads * dtype)


def _at(v: dict[str, Any], bits: int) -> dict[str, Any]:
    return {**v, "bits": bits, "masked": True, "p_over_o": 0.0}


def weight_decode(k: int, bits: int, v: dict[str, Any]) -> float:
    """A weight byte's read and decode at (k, bits), in read-times."""
    return 1.0 + decode(interp(v["o_w"][k], 32), k, _at(v, bits))


def encode_s(arm: Arm, v: dict[str, Any]) -> float:
    """The honest encode's seconds a KV byte: k·x₁_h read-times, the host's share taken off with ``host``."""
    vb = _at(v, arm.bits_kv)
    x1_h = v["x1_h"] * at_width(vb, "x1") / at_width(vb, "x1", 512) / at_width(vb, "sq_speed")
    h = at_width(vb, "h_host_sq") if arm.host else 0.0
    return arm.k_kv * x1_h / (v["R"] * 1e9) / (1.0 + h)


def decode_step(shape: Shape, tp: int, batch: float, ctx: float, arm: Arm, v: dict[str, Any], u: float = 0.0,
                e: float = 1.0) -> float:
    """A decode step's seconds: ``u`` of the KV reads unregistered (plain, unreceipted) and ``e`` of the appended
    tokens encoded in the step."""
    w = Workload(kv_shape(shape, arm.kv_dtype), batch, int(ctx), tp)
    engine = v["engine"] * shape.layers / 32
    st = step(w, Params(arm.k_w, BLOCK, K_S, 0), _at(v, arm.bits_w))
    if not arm.audited:
        return st["plain"] + engine
    if arm.kv == "plain":
        u, e = 1.0, 0.0
    vk = _at(v, arm.bits_kv)
    sk = st if (arm.k_kv, arm.bits_kv) == (arm.k_w, arm.bits_w) else step(w, Params(arm.k_kv, BLOCK, K_S, 0), vk)
    h = at_width(vk, "h_host_sq") if arm.host else 0.0
    attn = (1.0 - u) * sk["attn"] + u * sk["t_kv"]
    return st["lin"] + e * sk["enc"] / (1.0 + h) + attn + st["t_act"] + st["t_ar"] + st["gap"] + engine


def prefill_token(shape: Shape, tp: int, prompt: float, arm: Arm, v: dict[str, Any]) -> tuple[float, float]:
    """A prompt token's seconds on each GPU of a replica at prompt length ``prompt``, and of it the KV encode."""
    L, h, chunk = shape.layers, shape.hidden, v["chunk"]
    t = 2 * shape.linear_params / (tp * v["phi_prefill"]) + 4 * L * h * (prompt / 2) / (tp * v["phi_fa"])
    t += v["prefill_rest"] / chunk * L * h / (32 * 4096) / tp
    if tp > 1:
        t += 2 * L * (2 * (tp - 1) / tp * h * 2 / (v["b_p2p"] * 1e9) + v["t_allreduce"] / chunk)
    if not arm.audited:
        return t, 0.0
    t += shape.weight_bytes / tp * v["scratch"] * weight_decode(arm.k_w, arm.bits_w, v) / \
        weight_decode(*STATED_W, v) / chunk
    if arm.kv == "plain":
        return t, 0.0
    enc = kv_shape(shape, arm.kv_dtype).kv_bytes_per_token / tp * encode_s(arm, v)
    return t + enc, enc


def _below(n: float, g: int) -> float:
    """Σ_{r<g} min(n, r): the reads a request's generated tokens get before ``n`` each."""
    if g - 1 <= n:
        return g * (g - 1) / 2
    return n * (n + 1) / 2 + n * (g - 1 - n)


@dataclass(frozen=True)
class Mix:
    classes: tuple[tuple[int, int, float], ...]

    @property
    def weight(self) -> float:
        return sum(w for _, _, w in self.classes)

    def mean(self, f) -> float:
        return sum(w * f(p, g) for p, g, w in self.classes) / self.weight

    @property
    def prompt(self) -> float:
        return self.mean(lambda p, g: p)

    @property
    def generated(self) -> float:
        return self.mean(lambda p, g: g)

    @property
    def ctx(self) -> float:
        """The mean context a decode step reads, over the generated tokens."""
        return self.mean(lambda p, g: g * (p + g / 2)) / self.generated

    @property
    def reads(self) -> float:
        """KV token-reads a request: its prompt's G times, its generated tokens' the steps after each."""
        return self.mean(lambda p, g: p * g + g * (g - 1) / 2)


def kv_fractions(mix: Mix, arm: Arm, n_lag: float = 0.0) -> dict[str, float]:
    """For the KV option: ``u`` the unregistered share of KV reads, ``e_gen`` the share of generated tokens encoded,
    ``skip`` the prompt tokens a request never encodes (their request ends first)."""
    if not arm.audited or arm.kv == "registered":
        return {"u": 0.0, "e_gen": 1.0, "skip": 0.0, "n": 0.0}
    if arm.kv == "plain":
        return {"u": 1.0, "e_gen": 0.0, "skip": 0.0, "n": 0.0}
    n = arm.after if arm.kv == "after" else n_lag
    prompt_unreg = mix.mean(lambda p, g: p * min(n, g))
    skip = mix.mean(lambda p, g: p if g <= n else 0.0)
    if arm.kv == "deferred":
        return {"u": prompt_unreg / mix.reads, "e_gen": 1.0, "skip": skip, "n": n}
    gen_unreg = mix.mean(lambda p, g: _below(n, g))
    e_gen = mix.mean(lambda p, g: max(0.0, g - 1 - n)) / mix.generated
    return {"u": (prompt_unreg + gen_unreg) / mix.reads, "e_gen": e_gen, "skip": skip, "n": n}


class Server:
    """One replica of ``tp`` GPUs serving ``mix`` under ``arm``, every GPU time × ``slow`` (a filler's)."""

    def __init__(self, model: str, mix: Mix, arm: Arm, v: dict[str, Any], slow: float = 1.0):
        self.shape, self.tp, self.mix, self.arm, self.v, self.slow = SHAPES[model], MODELS[model], mix, arm, v, slow
        self.classes = [(p, g, w, *prefill_token(self.shape, self.tp, p, arm, v)) for p, g, w in mix.classes]
        self.kv_tok = kv_shape(self.shape, arm.kv_dtype).kv_bytes_per_token
        room = self.tp * v["H"] - self.shape.weight_bytes
        self.b_max = min(float(v["b_max"]), room / (mix.ctx * self.kv_tok))
        self.enc_tok = self.classes[0][4]

    def fractions(self, tpot: float) -> dict[str, float]:
        return kv_fractions(self.mix, self.arm, self.arm.lag / tpot if self.arm.kv == "deferred" else 0.0)

    def prefill(self, fr: dict[str, float]) -> float:
        """Each GPU's prefill seconds a request, less the encode of prompts whose request ends before it's due."""
        pf = sum(w * p * t for p, g, w, t, _ in self.classes) / self.mix.weight
        return (pf - fr["skip"] * self.enc_tok) * self.slow

    def t_d(self, batch: float, fr: dict[str, float]) -> float:
        return decode_step(self.shape, self.tp, batch, self.mix.ctx, self.arm, self.v, fr["u"], fr["e_gen"]) * \
            self.slow

    def capacity(self) -> dict[str, float]:
        """The largest request rate a replica sustains: every slot running, B = b_max."""
        tpot = 0.05
        for _ in range(50):
            fr = self.fractions(tpot)
            td = self.t_d(self.b_max, fr)
            lam = 1.0 / (self.prefill(fr) + self.mix.generated * td / self.b_max)
            tpot, last = self.b_max / (lam * self.mix.generated), tpot
            if abs(tpot - last) < 1e-9:
                break
        return {"lambda": lam, "tpot": tpot, "batch": self.b_max, "t_d": td, **fr}

    def operate(self, lam: float) -> dict[str, Any] | None:
        """The steady state at λ requests/s a replica: B from λ·Ḡ·t_d(B)/B = 1 − λ·prefill, or None if overloaded."""
        tpot = 0.05
        for _ in range(50):
            fr = self.fractions(tpot)
            left = 1.0 - lam * self.prefill(fr)
            if left <= 0 or lam * self.mix.generated * self.t_d(self.b_max, fr) / self.b_max > left:
                return None
            lo, hi = 1e-3, self.b_max
            for _ in range(60):
                mid = (lo + hi) / 2
                if lam * self.mix.generated * self.t_d(mid, fr) / mid > left:
                    lo = mid
                else:
                    hi = mid
            tpot_new = hi / (lam * self.mix.generated)
            if abs(tpot_new - tpot) < 1e-9:
                break
            tpot = tpot_new
        return {"lambda": lam, "batch": hi, "tpot": tpot_new, "t_d": self.t_d(hi, fr), "prefill": self.prefill(fr),
                **fr}

    def ttft(self, op: dict[str, Any], burst: float) -> float:
        """The middle request of a burst of ``burst`` arriving together: ((n + 1)/2)·P̄ tokens ahead of its first,
        each a prompt token on the critical path (a deferred encode isn't) and its share of the decode steps."""
        late = self.arm.kv in ("after", "deferred")
        crit = sum(w * p * (t - (enc if late else 0.0)) for p, g, w, t, enc in self.classes)
        crit /= sum(w * p for p, g, w, *_ in self.classes)
        tok = crit * self.slow + op["t_d"] / (self.v["chunk"] - op["batch"])
        return (max(1.0, burst) + 1) / 2 * self.mix.prompt * tok

    def gamma(self, op: dict[str, Any]) -> dict[str, float]:
        """The KV option's γ at the operating point, a GPU's shares of R·T."""
        v, arm, tp = self.v, self.arm, self.tp
        R = v["R"] * 1e9
        if not arm.audited:
            return {"mu_rebuild": 0.0, "mu_kv": 0.0, "mu_scratch": 0.0, "gamma_kv": 0.0}
        classes = [(arm.k_w, arm.bits_w)] + ([] if arm.kv == "plain" else [(arm.k_kv, arm.bits_kv)])
        rebuild = max(_rebuild(Params(k, BLOCK, K_S, 0), _at(v, b)) for k, b in classes)
        lam = op["lambda"]
        kv_read = lam * self.mix.reads * self.kv_tok / tp
        kv = op["u"] * kv_read / R
        if arm.kv == "plain":
            kv += lam * (self.mix.prompt + self.mix.generated) * self.kv_tok / tp * v["write_bus"] / R
        scratch = lam * self.mix.prompt / v["chunk"] * 2 * self.shape.weight_bytes / tp / R
        w = Workload(kv_shape(self.shape, arm.kv_dtype), op["batch"], int(self.mix.ctx), tp)
        p = Params(arm.k_w, BLOCK, K_S, 0)
        st = {**step(w, p, _at(v, arm.bits_w)), "cap": op["t_d"], "unfilled": 0.0}
        g = gamma_cap(w, p, {**_at(v, arm.bits_w), "eta_counted": False}, st)
        rest = sum(g[n] for n in REST)
        return {"mu_rebuild": rebuild, "mu_kv": kv, "mu_scratch": scratch, "gamma_kv": rebuild + kv,
                "kv_read_share": kv_read / R, "rest": rest, "gamma": rebuild + kv + scratch + rest}

    def encode_share(self, op: dict[str, Any]) -> float:
        """The share of the GPU's time spent encoding KV: the prompts' encoded in prefill and the appended tokens'."""
        fr = self.fractions(op["tpot"])
        tokens = self.mix.prompt - fr["skip"] + self.mix.generated * fr["e_gen"]
        return op["lambda"] * tokens * self.enc_tok * self.slow


def mix_of(name: str) -> Mix:
    return Mix(tuple(MIXES[name]))


def ratio_mix(r: float) -> Mix:
    return Mix(((max(1, round(256 * r)), 256, 1.0),))


def compare(model: str, mix: Mix, arms: list[Arm], v: dict[str, Any], filler: str = "F64/8w") -> list[dict[str, Any]]:
    """Each arm beside stock (at the arm's KV dtype) at stock's utilization on trace 1, without and with ``filler``."""
    rho0 = v["lambda0"] / Server("8B", mix_of("trace 1"), STOCK, v).capacity()["lambda"]
    slow, eta, _ = FILLERS[filler]
    out = []
    stocks = {d: Server(model, mix, replace(STOCK, kv_dtype=d), v) for d in (1, 2)}
    stock_cap = {d: s.capacity()["lambda"] for d, s in stocks.items()}
    stock_filled = {d: Server(model, mix, replace(STOCK, kv_dtype=d), v, STOCK_FILLER[0]).capacity()["lambda"]
                    for d in (1, 2)}
    for arm in [STOCK, STOCK_FP8] + arms:
        srv = Server(model, mix, arm, v)
        cap = srv.capacity()
        base = stock_cap[arm.kv_dtype]
        if arm.kv == "deferred":
            need = v["burst"] * cap["lambda"] / base * mix.prompt * srv.enc_tok
            if need > arm.lag:
                arm = replace(arm, lag=need)
                srv = Server(model, mix, arm, v)
                cap = srv.capacity()
        r = base / cap["lambda"]
        op = srv.operate(rho0 * cap["lambda"])
        filled = Server(model, mix, arm, v, STOCK_FILLER[0] if not arm.audited else slow)
        cap_f = filled.capacity()
        op_f = filled.operate(rho0 * cap_f["lambda"])
        g = srv.gamma(op) if op else {}
        same = srv.operate(rho0 * base)
        row = {
            "arm": arm.name, "what": arm.what, "model": model, "tp": srv.tp, "kv_dtype": arm.kv_dtype,
            "b_max": srv.b_max, "rho0": rho0,
            "capacity_req_s_gpu": cap["lambda"] / srv.tp,
            "capacity_tok_s_gpu": cap["lambda"] * (mix.prompt + mix.generated) / srv.tp,
            "gpus_x": r, "gpus_x_filled": stock_cap[arm.kv_dtype] / cap_f["lambda"],
            "gpus_x_filled_vs_filled_stock": stock_filled[arm.kv_dtype] / cap_f["lambda"],
            "tpot_ms": op["tpot"] * 1e3 if op else None, "batch": op["batch"] if op else None,
            "ttft_s": srv.ttft(op, v["burst"] / r) if op else None,
            "tpot_ms_filled": op_f["tpot"] * 1e3 if op_f else None,
            "ttft_s_filled": filled.ttft(op_f, v["burst"] * cap_f["lambda"] / base) if op_f else None,
            "tpot_sat_ms": cap["tpot"] * 1e3, "decode_step_ms": op["t_d"] * 1e3 if op else None,
            "prefill_s_req": srv.prefill(srv.fractions(op["tpot"] if op else cap["tpot"])),
            "same_gpus": "overloaded" if same is None else {"tpot_ms": same["tpot"] * 1e3,
                                                           "ttft_s": srv.ttft(same, v["burst"])},
            "eta_filled": STOCK_FILLER[1] if not arm.audited else eta,
            "encode_share": srv.encode_share(op) / rho0 if op else None, "lag_s": arm.lag,
            **g,
        }
        out.append(row)
    return out


def sweep(v: dict[str, Any]) -> list[dict[str, Any]]:
    """The GPUs each arm takes beside stock as the mix moves from decode- to prompt-heavy: G = 256, P = r·G."""
    rows = []
    for r in RATIOS:
        mix = ratio_mix(r)
        for model in MODELS:
            stocks = {d: Server(model, mix, replace(STOCK, kv_dtype=d), v).capacity()["lambda"] for d in (1, 2)}
            for name in SWEPT:
                arm = next(a for a in ARMS if a.name == name)
                cap = Server(model, mix, arm, v).capacity()["lambda"]
                rows.append({"ratio": r, "model": model, "arm": name, "gpus_x": stocks[arm.kv_dtype] / cap})
    return rows


def validate(v: dict[str, Any]) -> dict[str, dict[str, float]]:
    """The model beside each measurement it reproduces."""
    s = SHAPES["8B"]
    stated = ARMS[0]
    out = {}
    for name, (b, ctx, measured, src) in MEASURED.items():
        if "decode" in name:
            got = decode_step(s, 1, b, ctx, stated if name.startswith("stack") else STOCK, v)
        elif "chunk" in name:
            arm = stated if name.startswith("stack") else STOCK
            got = b * prefill_token(s, 1, 2 * ctx + b, arm, v)[0]
        else:
            continue
        out[name] = {"model": got, "measured": measured, "ratio": got / measured, "source": src}
    t1 = mix_of("trace 1")
    stock = Server("8B", t1, STOCK, v)
    op = stock.operate(v["lambda0"])
    out["stock TPOT"] = {"model": op["tpot"], "measured": MEASURED["stock TPOT"][2]}
    out["stock TTFT"] = {"model": stock.ttft(op, v["burst"]), "measured": MEASURED["stock TTFT"][2]}
    at32 = Server("8B", t1, stated, {**v, "b_max": 32}).capacity()
    out["stack TPOT 32"] = {"model": at32["tpot"], "measured": MEASURED["stack TPOT 32"][2],
                            "note": "slot-bound at 32: the model's saturated TPOT"}
    out["stack encode per token"] = {"model": kv_shape(s, 2).kv_bytes_per_token * encode_s(stated, v),
                                     "measured": 336.8e-3 / 2048}
    for row in out.values():
        row["ratio"] = row["model"] / row["measured"]
    return out


def encode_price(v: dict[str, Any]) -> list[dict[str, float]]:
    """What a registered KV byte costs to encode at each codec, against the γ it buys: k·x₁_h read-times a byte
    times μ_rebuild is about the same at every k (x₁_h/x₁ · s·(1 + h)), so a cheaper encode is μ_rebuild spent."""
    out = []
    for k, bits in ((3, 1024), (2, 1024), (3, 512), (2, 512)):
        arm = Arm("", "", k_kv=k, bits_kv=bits)
        read_times = encode_s(arm, v) * v["R"] * 1e9
        mu = _rebuild(Params(k, BLOCK, K_S, 0), _at(v, bits))
        out.append({"k": k, "bits": bits, "read_times": read_times, "mu_rebuild": mu, "product": read_times * mu,
                    "us_8b_bf16_token": read_times * kv_shape(SHAPES["8B"], 2).kv_bytes_per_token / (v["R"] * 1e3)})
    return out


def table(rows: list[dict[str, Any]]) -> str:
    head = (f"  {'arm':24} {'GPUs×':>6} {'+fill':>6} {'tok/s':>6} {'enc':>4} {'TPOT':>7} {'TTFT':>6} {'TPOTf':>7} "
            f"{'TTFTf':>6} {'μ_reb':>6} {'μ_kv':>6} {'γ_kv':>6} {'scr':>5} {'rest':>5} {'γ':>6}  same GPUs")
    lines = [head]
    for r in rows:
        same = r["same_gpus"]
        same = same if isinstance(same, str) else f"TPOT {same['tpot_ms']:.1f} ms, TTFT {same['ttft_s']:.2f} s"

        def f(x, w, d, scale=1.0):
            return f"{'—':>{w}}" if x is None else f"{x * scale:{w}.{d}f}"
        lines.append(
            f"  {r['arm']:24} {r['gpus_x']:6.2f} {r['gpus_x_filled']:6.2f} {r['capacity_tok_s_gpu']:6.0f} "
            f"{f(r['encode_share'], 4, 0, 100)} "
            f"{f(r['tpot_ms'], 7, 1)} {f(r['ttft_s'], 6, 2)} {f(r['tpot_ms_filled'], 7, 1)} "
            f"{f(r['ttft_s_filled'], 6, 2)} {f(r.get('mu_rebuild'), 6, 3, 100)} {f(r.get('mu_kv'), 6, 3, 100)} "
            f"{f(r.get('gamma_kv'), 6, 3, 100)} {f(r.get('mu_scratch'), 5, 2, 100)} {f(r.get('rest'), 5, 2, 100)} "
            f"{f(r.get('gamma'), 6, 2, 100)}  {same}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    v = values()
    checks = validate(v)
    results = {}
    for model in MODELS:
        names = None if model == "8B" else PROJECTED
        arms = [a for a in ARMS if names is None or a.name in names]
        results[model] = {m: compare(model, mix_of(m), arms, v) for m in MIXES}
    fillers = {f: compare("8B", mix_of("trace 1"), [a for a in ARMS if a.name in PROJECTED], v, f)
               for f in FILLERS if f != "none"}
    slots32 = compare("8B", mix_of("trace 1"), ARMS, {**v, "b_max": 32})
    doc = {"schema": "verity/overhead-serving/v1", "inputs": {n: asdict(i) for n, i in SERVING.items()},
           "fillers": FILLERS, "stock_filler": STOCK_FILLER, "mixes": MIXES, "models": MODELS,
           "arms": [asdict(a) for a in ARMS], "validation": checks, "results": results, "fillers_8B_trace1": fillers,
           "slots32_8B_trace1": slots32, "ratio_sweep": sweep(v), "encode_price": encode_price(v)}
    (args.out / "serving.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print("== the model beside the measurements (8B, BF16 KV, one GPU)")
    for name, c in checks.items():
        print(f"  {name:24} model {c['model']:10.5g}  measured {c['measured']:10.5g}  ×{c['ratio']:.3f}")
    print("\n== a registered KV byte's encode against the γ it buys (read-times a byte × μ_rebuild)")
    for e in doc["encode_price"]:
        print(f"  p{e['bits']} k={e['k']}: {e['read_times']:6.0f} read-times a byte, μ_rebuild {100 * e['mu_rebuild']:.3f}%,"
              f" product {e['product']:.2f}; {e['us_8b_bf16_token']:.1f} µs an 8B BF16 token")
    for model, by_mix in results.items():
        for m, rows in by_mix.items():
            mix = mix_of(m)
            print(f"\n== {model} (TP{MODELS[model]}) {m}: P̄ {mix.prompt:.0f}, Ḡ {mix.generated:.0f}, "
                  f"P/G {mix.prompt / mix.generated:.2f}, decode ctx {mix.ctx:.0f}; at ρ₀ {rows[0]['rho0']:.2f} of "
                  f"each arm's throughput, filler F64/8w (×{FILLERS['F64/8w'][0]:.3f}, η {FILLERS['F64/8w'][1]})")
            print(table(rows))
    for f, rows in fillers.items():
        print(f"\n== 8B trace 1, filler {f} (×{FILLERS[f][0]:.3f}, η {FILLERS[f][1]}): {FILLERS[f][2]}")
        print(table(rows))
    print("\n== 8B trace 1 at 32 slots, the measured engine's (b_max = 32)")
    print(table(slots32))
    print("\n== GPUs × stock as the prompt share grows (G = 256, P = ratio × G; no filler)")
    sw = doc["ratio_sweep"]
    for model in MODELS:
        print(f"  {model}: {'P/G':>6} " + " ".join(f"{n[:18]:>18}" for n in SWEPT))
        for r in RATIOS:
            each = [x["gpus_x"] for x in sw if x["model"] == model and x["ratio"] == r]
            print(f"  {'':5} {r:6g} " + " ".join(f"{x:18.2f}" for x in each))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
