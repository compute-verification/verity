"""γ_bw at the squaring floor (the red team's F17): the frontier and the operating point of the design note.

`python -m model.frontier [OPS] [BATCH] [MODEL]` from `benchmarks/word_codec` prints, for each E codec, γ_bw with W
registered / with no assumption on W, the step slowdown and the served one, with the KV option the 5% rule picks;
1023 bits at 6.58x the issue bound is the measured thread-per-word kernel; 1536 and 2048 need a kernel not yet built,
fused thread-per-word (extrapolated, past the measured bracket) or with its decode apart (from measured squarings).
OPS multiplies the issue-bound instruction count (1.0 the issue bound, 1.67 IMAD.WIDE at half rate). Costs are in
passes at the locked clock, as measured; the adversary runs at the max clock (κ = KAPPA), the honest at either.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace

from model import gamma as g

IDLE = {"Llama-3.1-8B": 0.37, "Mistral-Small-24B": 0.15, "Llama-3.1-70B": 0.10}
SHAPES = {"70": g.LLAMA_70B, "24": g.MISTRAL_24B, "8": g.LLAMA_8B}
OUT_TOKENS = 512
TAG = 0.227  # ra-sketch-tag's sketch (0.117) and its layer sequence (0.11), on top of the fused decode
# ra-codec-tensor's decodes fused into the 70B layer at k = 1 (r20261012-095837-c1eb), tag not included, batch 1 / 32:
# 2^768 − 825 1.159 / 1.386, 2^512 − 569 1.051 / 1.445, 2^1023 − 361 2.025 / 1.961 (thread-per-word throughout).
FUSED_768_Y = 1.159
FUSED_768_Y32 = 1.386
FUSED_512_Y = 1.051
FUSED_1023_Y = 2.025
# Standalone squarings in passes of the words squared, locked clock [measured]: 768 bits ra-word-codec's 2.81e10/s,
# 1,536 and 2,048 the red team's F22 (r20261012-102430-ee3a: 8.3e9/s, 5.05e9/s with one level of Karatsuba).
SQUARING = {768: g.R_READ / (96 * 2.81e10), 1536: 0.93, 2048: g.R_READ / (256 * 5.05e9)}
KAPPA = g.CLOCK_MAX_MHZ / g.CLOCK_LOCKED_MHZ  # Φ_cores is declared at the max clock


def _square(limbs: int) -> int:
    """Instructions to square `limbs` limbs without the fold, the fewer of schoolbook (products, their carry adds and
    the doubling) and a level of Karatsuba (three squarings of half, the sum h, its carry 2h, the combine 4h)."""
    school = limbs * limbs + 3 * limbs
    if limbs < 4 or limbs % 2:
        return school
    h = limbs // 2
    return min(school, 3 * _square(h) + 7 * h)


def issue_count(bits: int) -> int:
    """The fewest instructions a known squaring mod a pseudo-Mersenne p issues, the fold 3L included [estimate]: the
    schoolbook L(L + 1) + 5L up to 512 bits, Karatsuba's below it from 768 (696, 1,120, 2,184 and 3,488 at 768,
    1,023, 1,536 and 2,048 bits against 720, 1,216, 2,592 and 4,480)."""
    limbs = math.ceil(bits / 32)
    return _square(limbs) + 3 * limbs


def sq(bits: int, ops: float = 1.0) -> float:
    return g.sq_floor(bits, ops=ops * issue_count(bits))


# The fused decode's ALU over the issue bound, ε / sq(bits), from the measured steps: 4.57 at 768 bits (the best),
# 5.17 at 512, 7.14 at 1023, where 32 limbs a thread leave room for 6 decode warps against 768's 10. Wider fields
# are bracketed by the best and the worst measured; past 1023 bits a thread per word is outside the bracket.
OFF_ISSUE = (g.WIDE["p767"].eps / sq(768), (FUSED_1023_Y - FUSED_768_Y + g.WIDE["p767"].eps) / sq(1023))


def apart_off(bits: int) -> tuple[float, float]:
    """`off` for a decode that keeps a standalone squaring's registers (several threads a word, or decode warps that
    don't share the GEMM's register file) [estimate]: 768's fused decode with its squaring replaced by the measured
    one at `bits`, the rest of its ALU a byte held (low) or scaled with the squaring (high)."""
    e768 = g.WIDE["p767"].eps
    low = SQUARING[bits] + e768 - SQUARING[768]
    return low / sq(bits), e768 * SQUARING[bits] / SQUARING[768] / sq(bits)


def decode_eps(bits: int, off: float, clock_mhz: float = g.CLOCK_LOCKED_MHZ) -> float:
    """The fused decode's ALU in passes, `off` times the issue bound [estimate past 1023 bits]."""
    return off * sq(bits) * g.CLOCK_LOCKED_MHZ / clock_mhz


def y_of_eps(eps: float, batch: int) -> float:
    """The fused step against its decode's ALU, through the measured points (ε 0.84 → 1.051, 0.98 → 1.159, 1.85 →
    2.025 at batch 1): past 0.98 the decode is at the edge of the read and each further pass is a pass of step. At
    batch 32 the GEMM's own floor (768's 1.386) or the batch-1 step, whichever is larger: 1023 bits measured 1.961."""
    e512, e768 = g.WIDE["p511"].eps, g.WIDE["p767"].eps
    if batch >= 32:
        return max(FUSED_768_Y32, y_of_eps(eps, 1))
    if eps >= e768:
        return FUSED_768_Y + eps - e768
    return FUSED_512_Y + max(0.0, eps - e512) * (FUSED_768_Y - FUSED_512_Y) / (e768 - e512)


def fused_y(bits: int, batch: int, off: float, clock_mhz: float = g.CLOCK_LOCKED_MHZ) -> float:
    return y_of_eps(decode_eps(bits, off, clock_mhz), batch)


def e_codec(bits: int, batch: int, off: float, clock_mhz: float = g.CLOCK_LOCKED_MHZ) -> g.Codec:
    if bits == 768:
        c0 = g.WIDE["p767"]
        c = replace(c0, tau=0.011, p_bits=768, payload_bits=768)
    else:
        c = replace(g.extrapolated(bits), tau=0.011)
    return replace(c, y=fused_y(bits, batch, off, clock_mhz))


def kv_codec(bits: int, batch: int, off: float = OFF_ISSUE[0]) -> g.Codec:
    """A KV prime decoded beside attention: p767's unfused step with ε at `off` times the issue bound."""
    p767 = g.WIDE["p767"]
    eps = decode_eps(bits, off)
    c = replace(g.extrapolated(bits), eps=eps, y=p767.y + eps - p767.eps, tau=0.011)
    return replace(c, y=c.y + 0.04) if batch >= 32 else c


def options(shape: g.Shape, batch: int, c: g.Codec, ops: float, kv_widths=(1536, 2048), kappa: float = KAPPA,
            off: float = OFF_ISSUE[0]) -> dict[str, dict]:
    """Every KV option at E's codec `c` (its own prime at k_KV rounds, or a wider KV prime at one, its kernel `off`
    times the issue bound), the adversary at the floor on both."""
    bits = math.ceil(c.p_bits)
    adv = g.at_floor(c, sq(bits, ops))
    c = replace(c, y=c.y + TAG)
    common = dict(idle=IDLE[shape.name], theta=g.WIDE_THETA["p767"], out_tokens=OUT_TOKENS, adversary=adv,
                  kappa=kappa)
    out = {f"k{kr}": g.Point(shape, batch, 4096, True, c, kv_rounds=kr, **common).terms() for kr in (1, 2, 4)}
    for n in kv_widths:
        kc = kv_codec(n, batch, off)
        out[f"w{n}"] = g.Point(shape, batch, 4096, True, c, kv_codec=kc, kv_adversary=g.at_floor(kc, sq(n, ops)),
                               **common).terms()
    return out


def pick(opts: dict[str, dict], allowed=None, tol: float = 1.05) -> tuple[str, dict]:
    """The 5% rule: the lowest γ_bw within `tol` of the step at one KV round on E's prime."""
    base = opts["k1"]["slowdown"]
    ok = [k for k in opts if (allowed is None or k in allowed) and opts[k]["slowdown"] <= tol * base]
    k = min(ok, key=lambda x: opts[x]["gamma_bw"])
    return k, opts[k]


def row(t: dict) -> str:
    return (f"{100 * t['gamma_bw']:.2f} / {100 * t['gamma_decode_free']:.2f}% at {t['slowdown']:.2f}x"
            f" ({t['slowdown_served']:.2f}x served)")


def main(argv: list[str]) -> None:
    ops = float(argv[0]) if argv else 1.0
    batch = int(argv[1]) if len(argv) > 1 else 1
    shape = SHAPES[argv[2] if len(argv) > 2 else "70"]
    print(f"{shape.name} batch {batch}, squaring at {ops}x the issue bound, the adversary at {g.CLOCK_MAX_MHZ} MHz;"
          " W registered / no assumption")
    for clock in (g.CLOCK_LOCKED_MHZ, g.CLOCK_MAX_MHZ):
        print(f" the honest at {clock} MHz")
        c = e_codec(768, batch, OFF_ISSUE[0], clock)
        k, t = pick(options(shape, batch, c, ops), allowed=("k1", "k2", "k4"))
        print(f"  768 fused (runs today), y {c.y:.2f} (measured at the locked clock), KV {k}: {row(t)}")
        for bits in (1023, 1536, 2048):
            offs = [(off, "thread-per-word fused") for off in OFF_ISSUE]
            if bits in SQUARING:
                offs += [(off, "decode apart") for off in apart_off(bits)]
            for off, design in offs:
                c = e_codec(bits, batch, off, clock)
                k, t = pick(options(shape, batch, c, ops, off=off))
                print(f"  {bits} {design}, y {c.y:.2f} ({off:.2f}x the issue bound), KV {k}: {row(t)}")


def today(shape: g.Shape = g.LLAMA_70B, batch: int = 1, ops: float = 1.0,
          clock_mhz: float = g.CLOCK_LOCKED_MHZ) -> dict:
    """What runs today: E on 2^768 − 825 fused at k = 1, the KV cache on the same prime at the 5% rule's rounds (4 at
    batch 1)."""
    c = e_codec(768, batch, OFF_ISSUE[0], clock_mhz)
    return pick(options(shape, batch, c, ops), allowed=("k1", "k2", "k4"))[1]


def point(shape: g.Shape = g.LLAMA_70B, batch: int = 1, ops: float = 1.0, off: float | None = None,
          clock_mhz: float = g.CLOCK_LOCKED_MHZ) -> dict:
    """The operating point: E on 2^1536 − 3453 at k = 1, its decode `off` times the issue bound (by default the low
    end of `apart_off`), the KV cache at the 5% rule's pick."""
    off = apart_off(1536)[0] if off is None else off
    return pick(options(shape, batch, e_codec(1536, batch, off, clock_mhz), ops, off=off))[1]


if __name__ == "__main__":
    main(sys.argv[1:])
