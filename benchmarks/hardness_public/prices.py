"""W1 prices (CORE.md §2a: one dense int8 MAC = 1 unit) of the exact-multiply routes a split W = L + S can use.

Rates are the card files' (`verity/hardware/cards/{h200-sxm-141gb,rtx-pro-6000-server}.toml`, on
`cursor/lean-h17-fc8d`; not on main yet), except where a line says *recalled*. Every price is generous to the cheater:
the low-rank part is priced at one int8 MAC per factor MAC (its s32 intermediate really needs about 3 bytes), FP4 pieces
are taken to accumulate exactly, and IDP4A is priced per register written with no gather cost.
"""

import math

CHIPS = {
    "h200": {
        # tensor 4096 s8 MAC/SM-clock; imad 64 lanes/SM-clock; b1 BGMMA measured at 8x s8 (weight.b1 = 0.125)
        "b1": 0.125,
        "fp4": None,  # weight.e2m1 = 1: no FP4 MMA on sm_90a
        "sparse24": 0.5,  # recalled: H200 data sheet, INT8 sparse 3,958 TOPS = 2x dense
        # IDP4A writes one register for 4 MACs, an s8 MMA one for 32: 8 units a MAC (HT-PREMISE's price, per register
        # written; at the lane rate in time it would be 16)
        "alu_mac": 8.0,
    },
    "sm120": {
        # tensor 1024 s8 MAC/SM-clock; imad 64 lanes/SM-clock; no b1 MMA (weight.b1 = 1); nvf4/mxf4 0.5
        "b1": None,
        "fp4": 0.5,
        "sparse24": 0.5,  # recalled: Blackwell sparse MMA at 2x dense, so sparse FP4 is 0.25
        "alu_mac": 8.0,  # per register written, as on the H200 (at the lane rate in time it would be 4)
    },
}

ZBITS = (1, 2, 8)  # the operand Z = x + E: x = 0 leaves E's b bits (HT-PROTO recommends b = 2); 8 is full int8 x


def planes(a: int) -> int:
    """Two's-complement bit planes for the integers in [-a, a]."""
    return 0 if a == 0 else math.ceil(math.log2(2 * a + 1))


def fp4_pieces(a: int) -> int:
    """e2m1 pieces (balanced base-8 digits in [-4, 4], each exact in e2m1 under a power-of-two block scale) for [-a, a]."""
    if a == 0:
        return 0
    n, reach = 1, 4
    while reach < a:
        n += 1
        reach = reach * 8 + 4
    return n


def z_fp4_pieces(b: int) -> int:
    return 1 if b <= 2 else fp4_pieces(2 ** (b - 1))


def dense_part(chip: str, a: int, b: int) -> float:
    """Cheapest dense route for a residual with entries in [-a, a] against a b-bit Z."""
    c = CHIPS[chip]
    if a == 0:
        return 0.0
    best = 1.0 if a <= 127 else 2.0
    if c["b1"] is not None:
        best = min(best, planes(a) * b * c["b1"])
    if c["fp4"] is not None:
        best = min(best, fp4_pieces(a) * z_fp4_pieces(b) * c["fp4"])
    return best


def sparse24_part(chip: str, a: int, b: int, excess: float) -> float:
    """2:4 route: the tensor pass at half the dense price, plus the groups' entries beyond two on the ALU."""
    c = CHIPS[chip]
    return c["sparse24"] * dense_part(chip, a, b) + excess * c["alu_mac"]


def alu_part(chip: str, density: float) -> float:
    return density * CHIPS[chip]["alu_mac"]


def lowrank_part(r: int, n: int, k: int) -> float:
    return r * (n + k) / (n * k)
