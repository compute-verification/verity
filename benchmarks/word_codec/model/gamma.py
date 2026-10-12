"""The rate audit's γ bound as arithmetic: every term, for a model shape on one RTX PRO 6000.

γ_bw = 1 − cert_in / Q_honest, where Q_honest is every byte a decode step reads from device memory (E, the KV cache
and activations) and cert_in the cores' ingress from HBM the audit certifies (the actual-circuit model, v2.2). The
terms, each a fraction of the certified store E (or of E plus the KV cache when the KV cache is encoded and certified
too):

- `mu_chip`: bits held on chip across windows are never read again; the capacity is M_cores = S_chip.
- `mu_rebuild`: words the adversary doesn't read but re-encodes from a compressed source, paid in work under the
  direct-sum conjecture W. On-chip sources and HBM sources give the same saving per unit of compute (see `saving`),
  so the term is s_word · f_cap, s_word the share of its slot a rebuilt word saves (`s_word`). s_c, the payload's
  compressibility, is an assumption about data the operator chooses: a compressor bounds it from one side only.
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
# CUDA-core time per native pass, y the fused GEMM's today (warp-specialized, no tag) at batch 1–16, lower end.
WIDE = {
    "p255": Codec(0.71, 135, 1.51, 0.15, p_bits=255, payload_bits=248, slot_bits=256),
    "p511": Codec(0.84, 295, 1.47, 0.15, p_bits=511, payload_bits=504, slot_bits=512),
    "m521": Codec(0.87, 292, 1.55, 0.15, p_bits=521, payload_bits=520, slot_bits=528),
    "p767": Codec(0.98, 503, 1.75, 0.15, p_bits=767, payload_bits=760, slot_bits=768),
}
# The adversary's tensor-core help on encode, x_cuda / x_add − 1 (same runs), and the standalone decode-from-HBM
# ratio, native 1,420 GB/s over the decode's: what a kernel that hides decode under the read could reach.
WIDE_THETA = {"p255": 85 / 83 - 1, "p511": 233 / 219 - 1, "m521": 0.0, "p767": 449 / 407 - 1}
WIDE_HIDDEN_Y = {"p255": max(1.05, 1420 / 1429), "p511": 1420 / 1195, "m521": 1420 / 1100, "p767": 1420 / 930}


@dataclass(frozen=True)
class Point:
    shape: Shape
    batch: int
    context: int
    kv_certified: bool
    codec: Codec
    s_c: float = 0.19  # FP8 weights per payload byte: measured 0.17 (8B, art:23672ef9), 0.19 (70B, art:f989de1d)
    s_c_kv: float = 0.30  # BF16 KV, an estimate
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
        # d, the window in passes of the certified bytes: their fused pass, the uncertified KV read at the native
        # rate, and the idle time, all of which the adversary may spend on re-encoding.
        d = c.y + ((0.0 if self.kv_certified else kv) + self.idle * e) / cert
        f = f_cap_keep_decode(d, c.tau, c.eps, c.r, self.kappa, self.theta)
        f_free = f_cap_decode_free(d, c.tau, c.eps, c.r, self.kappa, self.theta)
        s_c = (e * s_word(c, self.s_c) + (kv * s_word(c, self.s_c_kv) if self.kv_certified else 0.0)) / cert
        chip = mu_chip(cert)
        lost = min(1.0, saving(chip, f, s_c) + mu_slot(c.p_bits, c.slot_bits) + self.other)
        q_honest = e + kv + act
        stock = (e + kv) / e + self.idle
        return {
            "slowdown": (d * cert / e) / stock,
            "E_GB": e / 1e9,
            "kv_GB": kv / 1e9,
            "fits": e + kv <= M_DEVICE,
            "d": d,
            "mu_chip": chip,
            "mu_rebuild": saving(chip, f, s_c) - chip,
            "mu_slot": mu_slot(c.p_bits, c.slot_bits),
            "uncertified_reads": 1 - cert / q_honest,
            "gamma_bw": 1 - cert * (1 - lost) / q_honest,
            # Without the assumption on s_c (an unvetted model, duplicated prompts): every payload free, nothing decoded.
            "gamma_decode_free": 1 - cert * (1 - min(1.0, saving(chip, f_free, 1.0) + mu_slot(c.p_bits, c.slot_bits)
                                                     + self.other)) / q_honest,
            # The joint view's compute residual: the window's ALU that decode and tag don't force.
            "alu_idle": max(0.0, 1 - (c.eps + c.tau) / (self.kappa * d * (1 + self.theta))),
        }
