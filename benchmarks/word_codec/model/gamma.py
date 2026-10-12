"""The rate audit's γ bound as arithmetic: every term, for a model shape on one RTX PRO 6000.

γ_bw = 1 − cert_bw / Q_honest, where Q_honest is every byte a decode step reads from device memory (E, the KV cache
and activations) and cert_bw the chip's reads from HBM the audit certifies (the actual-circuit model, v2.4). The
terms, each a fraction of the certified store E (or of E plus the KV cache when the KV cache is encoded and certified
too):

- `mu_chip`: bits held on chip across windows are never read again; the capacity is M_cores = S_chip.
- `mu_rebuild`: words the adversary doesn't read but re-encodes from a compressed source, paid in work under the
  direct-sum conjecture W. On-chip sources and HBM sources give the same saving per unit of compute (see `saving`),
  so the term is s_word · f_cap, s_word the share of its slot a rebuilt word saves (`s_word`). s_c, the payload's
  compressibility, is an assumption about data the operator chooses: a compressor bounds it from one side only. W can
  be registered with its s_c; the prompts can't, so the KV cache's payloads count as free (`rebuild`), and the KV
  words may take a codec and rounds of their own (`Point.kv_codec`, `Point.kv_rounds`). The adversary's costs
  default to the honest's measured kernels; `Point.adversary` puts them elsewhere, such as at the squaring floor
  WorkHardness is stated at (`at_floor`, `sq_floor`).
- `mu_slot`: slot bits that carry no information (the top 3 bits of a word mod 2^61 − 1), which an adversary drops.
- `mu_sample`: wrong sketches the verifier's spot checks miss, for a time-averaged claim with total rejection.
- `mu_lambda`: the random oracle's guessing allowance per tile (AuditRO's λ).
- `mu_side`: zero under the trusted host's isolation (no peer-to-peer, IOMMU), and not certifiable without it.
The uncertified KV and activation reads enter through Q_honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

# RTX PRO 6000 Blackwell Server (sm_120).
R_READ = 1.488e12  # B/s, measured (art:51a27942); the datasheet's 1.597e12
S_CHIP = 239_992_832  # B: registers + shared memory + L2 + the red team's allowance (#1805)
M_DEVICE = 103_079_215_104  # B
CLOCK_LOCKED_MHZ = 2092
CLOCK_MAX_MHZ = 2430


@dataclass(frozen=True)
class Shape:
    name: str
    hidden: int
    layers: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int

    def linears(self) -> int:
        q = self.heads * self.head_dim
        kv = 2 * self.kv_heads * self.head_dim
        per_layer = self.hidden * (q + kv) + q * self.hidden + self.hidden * 2 * self.intermediate
        per_layer += self.intermediate * self.hidden
        return self.layers * per_layer

    def lm_head(self) -> int:
        return self.vocab * self.hidden

    def weights_read(self) -> int:
        """FP8 weights a decode step reads: the linears and lm_head (the embedding table is gathered, not read)."""
        return self.linears() + self.lm_head()

    def kv_bytes_per_token(self, bytes_per_value: int = 2) -> int:
        return self.layers * 2 * self.kv_heads * self.head_dim * bytes_per_value

    def activation_bytes(self, batch: int, bytes_per_value: int = 2) -> int:
        """Inputs to qkv, o, gate_up and down per step: an upper bound on activation reads (most hit L2)."""
        q = self.heads * self.head_dim
        return batch * self.layers * (2 * self.hidden + q + self.intermediate) * bytes_per_value


LLAMA_8B = Shape("Llama-3.1-8B", 4096, 32, 14336, 32, 8, 128, 128256)
MISTRAL_24B = Shape("Mistral-Small-24B", 5120, 40, 32768, 32, 8, 128, 131072)
LLAMA_70B = Shape("Llama-3.1-70B", 8192, 80, 28672, 64, 8, 128, 128256)
SHAPES = (LLAMA_8B, MISTRAL_24B, LLAMA_70B)


def e_bytes(shape: Shape, payload_bits: int = 64, slot_bits: int = 64) -> float:
    """Stored bytes of E: FP8 weights packed `payload_bits` to a `slot_bits` word."""
    return shape.weights_read() * slot_bits / payload_bits


def mu_chip(e: float, s_chip: float = S_CHIP) -> float:
    return min(1.0, s_chip / e)


def mu_slot(p_bits: float, slot_bits: int = 64) -> float:
    return (slot_bits - p_bits) / slot_bits


def e4m3_values() -> list[float]:
    out = set()
    for b in range(256):
        e, m = (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            continue
        x = (m / 8) * 2.0**-6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        out.add(-x if b & 0x80 else x)
    return sorted(out)


def e4m3_entropy(cdf, scale: float) -> float:
    """Order-0 entropy in bits of round-to-nearest E4M3 of a distribution with the given CDF and scale."""
    v = e4m3_values()
    edges = [(a + b) / 2 for a, b in zip(v, v[1:])]
    h, lo = 0.0, 0.0
    for e in edges + [math.inf]:
        hi = 1.0 if e == math.inf else cdf(e, scale)
        p = hi - lo
        if p > 0:
            h -= p * math.log2(p)
        lo = hi
    return h


def gauss_cdf(x: float, s: float) -> float:
    return 0.5 * (1 + math.erf(x / (s * math.sqrt(2))))


def laplace_cdf(x: float, b: float) -> float:
    return 0.5 * math.exp(x / b) if x < 0 else 1 - 0.5 * math.exp(-x / b)


def f_cap_keep_decode(y: float, tau: float, eps: float, r: float, kappa: float = 1.0, theta: float = 0.0) -> float:
    """Fraction of words an adversary can re-encode when it must decode the words it keeps (real weights).

    Compute over the window: y·κ·(1 + θ) native passes, of which τ (tag) and ε per kept word are owed; a rebuilt
    word costs rε and saves its ε of decode. Matmul and decompression costs are left out, so this is an upper bound.
    """
    return max(0.0, min(1.0, (y * kappa * (1 + theta) - tau - eps) / (eps * (r - 1))))


def f_cap_decode_free(y: float, tau: float, eps: float, r: float, kappa: float = 1.0, theta: float = 0.0) -> float:
    """The same when every payload is free to the adversary (PRG payloads): it never decodes."""
    return max(0.0, min(1.0, (y * kappa * (1 + theta) - tau) / (r * eps)))


def s_word(c: "Codec", s_c: float) -> float:
    """What a rebuilt word saves, as a share of its slot, beyond the slot slack every word loses (`mu_slot`): it reads
    its payload compressed, payload_bits · (1 − s_c) bits, in place of log₂ p. s_c is per payload byte."""
    return (c.p_bits - c.payload_bits * (1 - s_c)) / c.slot_bits


def saving(mu_chip_: float, f_cap: float, s_c: float) -> float:
    """Most bytes of E an adversary avoids reading with on-chip capacity μ_chip and compute for f_cap re-encodes.

    The LP: hold encoded words on chip (u1, free), hold compressed payloads on chip (u2, H/8 of the space, one
    re-encode each), or re-encode from a compressed copy in HBM (u3, saves s_c of the word). Per re-encode,
    u2 over u1 and u3 both gain s_c, so the optimum is μ_chip + s_c · min(f_cap, 1 − μ_chip).
    """
    return mu_chip_ + s_c * min(f_cap, 1.0 - mu_chip_)


def rebuild(budget: float, keep: float, options: list[tuple[float, float, float]]) -> float:
    """Bytes an adversary avoids reading by re-encoding, as a share of the certified store: the LP over regions.

    The budget, κ·d·(1 + θ) − τ passes, first pays `keep`, the decode owed on words whose payloads the adversary
    needs. Each option is (share of the store, passes per pass to rebuild it, share of a word it saves), the cost net
    of the decode a rebuilt word no longer owes; the LP spends what is left on the best saving per pass first.
    """
    left = budget - keep
    saved = 0.0
    for share, cost, s in sorted(options, key=lambda o: o[2] / o[1], reverse=True):
        if share <= 0 or left <= 0:
            continue
        f = min(1.0, left / (cost * share))
        saved += share * f * s
        left -= cost * share * f
    return saved


def mu_sample(delta: float, samples_per_window: int, windows: int) -> float:
    """Skipped fraction f with (1 − f)^(sN) ≥ δ: the most a prover cheats in a time-averaged claim that rejects on
    any wrong opened sketch."""
    return math.log(1 / delta) / (samples_per_window * windows)


def mu_lambda(tile_bytes: int, lam_bits: float = 46.0) -> float:
    return lam_bits / (8 * tile_bytes)


@dataclass(frozen=True)
class Codec:
    """A codec's measured costs in native passes of E: decode ε, encode/decode ratio r, fused step y, tag τ."""

    eps: float
    r: float
    y: float
    tau: float
    p_bits: float = 64.0  # information bits a stored word carries, log2 p
    payload_bits: int = 64
    slot_bits: int = 64


# Measured at 2^61 − 1 (art:51a27942, 60 payload bits in 64-bit slots, batch 1–8).
MEASURED = {
    1: Codec(0.28, 23, 1.16, 0.15, p_bits=61, payload_bits=60),
    2: Codec(0.46, 29, 1.24, 0.15, p_bits=61, payload_bits=60),
    3: Codec(0.65, 33, 1.28, 0.15, p_bits=61, payload_bits=60),
    4: Codec(0.84, 36, 1.38, 0.15, p_bits=61, payload_bits=60),
    5: Codec(1.03, 38, 1.55, 0.15, p_bits=61, payload_bits=60),
}


def packed(c: Codec) -> Codec:
    """The same codec with its words bit-packed to ⌈log₂ p⌉ bits: a pass of E shrinks by that over the slot, and the
    measured costs, unchanged in seconds, grow by the inverse in passes."""
    bits = math.ceil(c.p_bits)
    g = c.slot_bits / bits
    return replace(c, eps=c.eps * g, y=c.y * g, tau=c.tau * g, slot_bits=bits)


# Wide words at k = 1, measured by ra-codec-tensor (r20261012-073450-68ab, r20261012-080525-4dcf): ε is the decode's
# CUDA-core time per native pass, y the fused GEMM's today (warp-specialized, no tag) at batch 1–8, weighted by bytes
# over Llama-3.1's four linears with the best variant for each.
WIDE = {
    "p255": Codec(0.71, 135, 1.51, 0.15, p_bits=255, payload_bits=248, slot_bits=256),
    "p511": Codec(0.84, 295, 1.49, 0.15, p_bits=511, payload_bits=504, slot_bits=512),
    "m521": Codec(0.87, 292, 1.56, 0.15, p_bits=521, payload_bits=520, slot_bits=528),
    "p767": Codec(0.98, 503, 1.75, 0.15, p_bits=767, payload_bits=760, slot_bits=768),
}
# The same y at batch 32 (same runs, same weighting). Nothing is measured above 32, so these stand in, as a lower
# bound, for larger batches.
MEASURED_Y32 = {1: 1.18, 2: 1.25, 3: 1.34, 4: 1.44, 5: 1.61}
WIDE_Y32 = {"p255": 1.59, "p511": 1.74, "m521": 1.66, "p767": 1.79}


def at_batch(c: Codec, y32: float, batch: int) -> Codec:
    """The codec with its fused step measured at the batch: the batch-1–8 figure below 32, else the batch-32 one."""
    return c if batch < 32 else replace(c, y=y32)
# The adversary's tensor-core help on encode, x_cuda / x_add − 1 (same runs), and the standalone decode-from-HBM
# ratio, native 1,420 GB/s over the decode's: what a kernel that hides decode under the read could reach.
WIDE_THETA = {"p255": 85 / 83 - 1, "p511": 233 / 219 - 1, "m521": 0.0, "p767": 449 / 407 - 1}
WIDE_HIDDEN_Y = {"p255": max(1.05, 1420 / 1429), "p511": 1420 / 1195, "m521": 1420 / 1100, "p767": 1420 / 930}


def extrapolated(bits: int) -> Codec:
    """A full-slot word wider than any measured, from p511 and p767: ε linear in the bits, r at p767's r per bit (r
    per bit has grown with the width, so rε, the adversary's encode, is if anything low), y p767's with ε's
    difference. 2^1536 − 3453 and 2^2048 − 1557 (one head's BF16 K or V row a word) are the full-slot primes."""
    p767 = WIDE["p767"]
    eps = p767.eps + (p767.eps - WIDE["p511"].eps) * (bits - 768) / 256
    return Codec(eps, p767.r * bits / 768, p767.y + eps - p767.eps, p767.tau, bits, bits, bits)


SMS = 188
ISSUE_LANES = 128  # an SM's four schedulers, one 32-lane instruction each a clock


def sq_floor(bits: float, ops: float | None = None, clock_mhz: float = CLOCK_LOCKED_MHZ) -> float:
    """A squaring mod p (⌈log₂ p⌉ = bits) at the device's issue bound, in passes of the words squared [estimate].

    Schoolbook on L = bits/32 limbs: L(L + 1)/2 products (IMAD.WIDE with accumulate) and as many carry adds, 2L to
    double the cross terms and 3L for a pseudo-Mersenne fold, so L(L + 1) + 5L instructions, 720 at 768 bits;
    Karatsuba trades products for adds and issues no fewer there. A pipe slower than the issue rate only raises it,
    so as the adversary's cost it is low unless some mix issues fewer instructions; tensor cores enter as θ. At the
    locked clock, as the measured ε and r are."""
    limbs = bits / 32
    n = limbs * (limbs + 1) + 5 * limbs if ops is None else ops
    return R_READ * n / (bits / 8 * SMS * clock_mhz * 1e6 * ISSUE_LANES)


def at_floor(c: Codec, sq: float, rounds: int = 1, eta: float = 0.1) -> Codec:
    """The adversary's codec under WorkHardness(p, k, η) with a squaring every `sq` passes: its decode is `rounds`
    squarings and its encode (1 − η)·rounds·(⌈log₂ p⌉ − 2) of them."""
    return replace(c, eps=rounds * sq, r=(1 - eta) * (math.ceil(c.p_bits) - 2))


@dataclass(frozen=True)
class Point:
    shape: Shape
    batch: int
    context: int
    kv_certified: bool
    codec: Codec
    s_c: float = 0.19  # FP8 weights per payload byte: measured 0.17 (8B, art:23672ef9), 0.19 (70B, art:f989de1d)
    kv_rounds: int = 1  # Sloth rounds on the KV cache's words, k for E: its payloads are free, so it may want more
    kv_codec: Codec | None = None  # the KV words' codec, at one round (E's when None); its ε, r and slot are used
    out_tokens: int = 0  # tokens a sequence generates, over which its prompt's KV encode is spread (0: not counted)
    adversary: Codec | None = None  # the adversary's ε and r on E's words (`at_floor`); the honest's when None
    kv_adversary: Codec | None = None  # the same on the KV words, at one round; E's adversary's when no kv_codec
    kappa: float = 1.0
    theta: float = 0.0
    other: float = 0.0  # μ_sample + μ_λ (+ μ_side, zero under isolation)
    idle: float = 0.0  # the step's time outside reads (attention math, norms, sampling, launches), in passes of E
    act_crossing: float = 0.0  # share of activation bytes that miss L2 and cross

    def terms(self) -> dict[str, float]:
        c = self.codec
        e = e_bytes(self.shape, c.payload_bits, c.slot_bits)
        kv = self.batch * self.context * self.shape.kv_bytes_per_token()
        act = self.act_crossing * self.shape.activation_bytes(self.batch)
        cert = e + kv if self.kv_certified else e
        a, b = e / cert, (kv / cert if self.kv_certified else 0.0)
        kc = self.kv_codec or c
        enc_kv = self.kv_rounds * kc.r * kc.eps  # passes to encode a pass of KV words
        new = self.batch * self.shape.kv_bytes_per_token() if self.kv_certified else 0.0
        # d, the window in passes of the certified bytes: E's fused pass, the KV cache's (E's fused step with the KV
        # codec's decode in place of E's) or its uncertified read at the native rate, the encode of the step's new
        # KV words, and the idle time; the adversary may spend all of it.
        y_kv = c.y + self.kv_rounds * kc.eps - c.eps if self.kv_certified else 1.0
        d = (c.y * e + y_kv * kv + enc_kv * new + self.idle * e) / cert
        # The prompts' KV words, encoded at prefill, outside the decode windows: passes of the certified bytes.
        prefill = enc_kv * (kv - new) / cert
        budget = d * self.kappa * (1 + self.theta) - c.tau
        chip = mu_chip(cert)
        slot = a * mu_slot(c.p_bits, c.slot_bits) + b * mu_slot(kc.p_bits, kc.slot_bits)
        ca = self.adversary or c
        ka = self.kv_adversary or (kc if self.kv_codec else ca)
        enc_kv_a = self.kv_rounds * ka.r * ka.eps
        # W is registered with its s_c, so the E words kept are decoded; the prompts aren't, so no KV word is.
        rebuilt = min(1.0 - chip, rebuild(budget, ca.eps * a, [(a, (ca.r - 1) * ca.eps, s_word(c, self.s_c)),
                                                               (b, enc_kv_a, s_word(kc, 1.0))]))
        # Without the assumption on s_c (an unvetted model): every payload free, nothing decoded.
        rebuilt_free = min(1.0 - chip, rebuild(budget, 0.0, [(a, ca.r * ca.eps, s_word(c, 1.0)),
                                                              (b, enc_kv_a, s_word(kc, 1.0))]))
        q_honest = e + kv + act
        stock = (e + kv) / e + self.idle
        served = d + (prefill / self.out_tokens if self.out_tokens else 0.0)

        def gamma(lost: float) -> float:
            return 1 - cert * (1 - min(1.0, lost)) / q_honest

        return {
            "slowdown": (d * cert / e) / stock,
            # With each prompt's encode spread over the tokens it generates; stock's prefill is left out, so high.
            "slowdown_served": (served * cert / e) / stock,
            "prefill_passes": prefill,
            "E_GB": e / 1e9,
            "kv_GB": kv / 1e9,
            "fits": e + kv <= M_DEVICE,
            "d": d,
            "mu_chip": chip,
            "mu_rebuild": rebuilt,
            "mu_slot": slot,
            "uncertified_reads": 1 - cert / q_honest,
            "gamma_bw": gamma(chip + rebuilt + slot + self.other),
            "gamma_decode_free": gamma(chip + rebuilt_free + slot + self.other),
            # The joint view's compute residual: the window's ALU beyond what decoding E and the tag force.
            "alu_idle": max(0.0, 1 - (c.eps * a + c.tau) / (self.kappa * d * (1 + self.theta))),
        }
