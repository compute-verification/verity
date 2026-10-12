"""γ_bw at the squaring floor (the red team's F17): the frontier and the operating point of the design note.

`python -m model.frontier [OPS] [BATCH] [MODEL]` from `benchmarks/word_codec` prints, for each E codec, γ_bw with W
registered / with no assumption on W, the step slowdown and the served one, with the KV option the 5% rule picks.
OPS multiplies the issue-bound instruction count (1.0 the issue bound, 1.67 IMAD.WIDE at half rate).
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
# ra-codec-tensor's 2^768 − 825 decode fused into the 70B layer at k = 1 (r20261012-095837-c1eb), tag not included.
FUSED_768_Y = 1.159
FUSED_768_Y32 = 1.386
# How far today's squaring kernels sit above the issue bound (61, 255, 521, 768 bits: 1.9, 3.3, 3.0, 2.3).
OFF_ISSUE = (2.3, 3.3)


def sq(bits: int, ops: float = 1.0) -> float:
    limbs = bits / 32
    return g.sq_floor(bits, ops=ops * (limbs * (limbs + 1) + 5 * limbs))


def fused_y(bits: int, batch: int, off: float) -> float:
    """A fused decode's y scaled from the 768-bit kernel: its overhead over the read, on a decode whose ALU is
    `off` times the issue bound and no longer hides once it passes a pass [estimate]."""
    over = (FUSED_768_Y32 if batch >= 32 else FUSED_768_Y) - 1
    return max(1.0, off * sq(bits)) + over


def e_codec(bits: int, batch: int, off: float) -> g.Codec:
    if bits == 768:
        c0 = g.WIDE["p767"]
        c = replace(c0, tau=0.011, p_bits=768, payload_bits=768)
    else:
        c = replace(g.extrapolated(bits), tau=0.011)
    return replace(c, y=fused_y(bits, batch, off))


def kv_codec(bits: int, batch: int) -> g.Codec:
    c = replace(g.extrapolated(bits), tau=0.011)
    return replace(c, y=c.y + 0.04) if batch >= 32 else c


def options(shape: g.Shape, batch: int, c: g.Codec, ops: float, kv_widths=(1536, 2048)) -> dict[str, dict]:
    """Every KV option at E's codec `c` (its own prime at k_KV rounds, or a wider KV prime at one), the adversary
    at the floor on both."""
    bits = math.ceil(c.p_bits)
    adv = g.at_floor(c, sq(bits, ops))
    c = replace(c, y=c.y + TAG)
    common = dict(idle=IDLE[shape.name], theta=g.WIDE_THETA["p767"], out_tokens=OUT_TOKENS, adversary=adv)
    out = {f"k{kr}": g.Point(shape, batch, 4096, True, c, kv_rounds=kr, **common).terms() for kr in (1, 2, 4)}
    for n in kv_widths:
        kc = kv_codec(n, batch)
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
    print(f"{shape.name} batch {batch}, squaring at {ops}x the issue bound; W registered / no assumption")
    k, t = pick(options(shape, batch, replace(e_codec(768, batch, 2.3), y=FUSED_768_Y32 if batch >= 32 else
                                              FUSED_768_Y), ops), allowed=("k1", "k2", "k4"))
    print(f"  768 fused (measured y), KV {k}: {row(t)}")
    for bits in (1536, 2048):
        for off in OFF_ISSUE:
            c = e_codec(bits, batch, off)
            k, t = pick(options(shape, batch, c, ops))
            print(f"  {bits} y {c.y:.2f} (kernel {off}x the issue bound), KV {k}: {row(t)}")
    for off in OFF_ISSUE:
        print(f"  point, kernel {off}x the issue bound: {row(point(shape, batch, ops, off))}")


def point(shape: g.Shape = g.LLAMA_70B, batch: int = 1, ops: float = 1.0, off: float = OFF_ISSUE[0]) -> dict:
    """The operating point: E on 2^1536 − 3453 and the KV cache on 2^2048 − 1557 (one head's BF16 K or V row a
    word), both at k = 1."""
    return options(shape, batch, e_codec(1536, batch, off), ops, kv_widths=(2048,))["w2048"]


if __name__ == "__main__":
    main(sys.argv[1:])
