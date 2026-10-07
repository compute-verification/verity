"""Pearl-C's device record: the hardware facts a Pearl-C instance is stated on, as one scheme parameter.

Every checked value goes through two tensor-core steps (``verity.ml.circuits.silicon`` models): the FP8 atom (E4M3 x E4M3 into an FP32
accumulator, k = 32) that C̃, the fused A′·F_B and B̃·F_A, G and the forming atom run on, and the BF16 step (k = 16) that
the peel accumulates into C̃'s word.  Beside them the record holds the promotion period G (atoms per fresh +0 accumulator,
promoted by one FP32 nearest-even add), the W1 prices (`Prices`) that `wref`, `credit_of`, `job_cost` and the debit read,
the domain's k bound, and the debit cap ρ (`cap`: a tile whose debit passes ρ times its credited cells' credit is
rejected, and an accepted unit is credited (1 − ρ)·`creditDevRev1`).  `pearl_c`, `pearl_c_debit`, `pearl_c_u` and
`pearl_c_work` read the record; nothing else in them names hardware.

* `H100` is the record every ``pearl-c-h100-*`` output was pinned on: `HOPPER_E4M3_K32`, `HOPPER_BF16_WGMMA_K16`, G = 4,
  Rounds 10-11's W1 prices, k <= 2^16 (the pure-chain freezing bound)
  and ρ = 1/400 (TT_OUT(1/400)).
* `SM120` is the RTX PRO 6000 Blackwell's (``mma.sync``: ``kind::f8f6f4`` m16n8k32 E4M3 and m16n8k16 BF16).  The atom is the
  card's captured step, `BLACKWELL_SM120_E4M3_M16N8K32` (one group of 32 in a 26-bit adder), and the prices are
  `SM120_PRICES` (`Prices.sm120`, measured on node 2); the other fields are stand-ins (`standins` names each):
  `HOPPER_BF16_M16N8K16` for the peel, and the H100's G = 4 and k bound.  Its cap is ρ = 1/1,000 (Daniel, 9:43 AM PDT on
  2 Oct; Lean `Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192` and its twins).  G, the prices, the k
  bound and the cap are statement parameters: they change through the coordinator and the cheap-binding and Lean lanes,
  not here.
* `SM120_UNPROMOTED` is `SM120` without promotion (G is None: one accumulator from +0 over the whole of k, and the word is
  the last atom's output).  It is the panel's ``pearl-c-sm120`` v2, built for the kernels and the twin; no scheme
  is registered on it and its debit raises until the theory and Lean lanes restate Pearl-C without promotion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

from verity.ml.circuits.silicon.models import (BLACKWELL_SM120_E4M3_M16N8K32, HOPPER_BF16_M16N8K16, HOPPER_BF16_WGMMA_K16,
                                 HOPPER_E4M3_K32, GroupSum)


@dataclass(frozen=True, slots=True)
class Prices:
    """A device's W1 prices, in units of one dense FP8 MAC on that device (the device statement's `Prices`: `add`,
    `bf16`, `Costs`'s `fs` and `qa`).  Each credited step is at its cheapest bit-exact implementation on the device."""

    device: str
    fadd: int                # one FP32 add per word: a promotion, and P_B's FMA chain per step
    bf16_mac: int            # one BF16 MAC (the peel)
    forming_credited: int    # fs per activation element: the noise atom and the cast (depend on the salt)
    forming_a_only: int      # qa per activation element: ρ's squared accumulate (reads the activations alone)

    @property
    def forming(self) -> int:
        return self.forming_credited + self.forming_a_only


@dataclass(frozen=True, slots=True)
class Device:
    """One device's record (module docstring)."""

    name: str                # the scheme name's device part: pearl-c-<name>-v1
    atom: GroupSum           # the FP8 atom, E4M3 k32
    peel: GroupSum           # the BF16 peel step, k16
    g: int | None            # atoms per promotion (`Params.pi.G`); None: no promotion
    prices: Prices
    k_max: int               # the domain's bound on k
    standins: tuple[str, ...] = ()   # the fields that are stand-ins, each with what it stands in for
    cap: Fraction = Fraction(1, 400)  # ρ: the tile rule's debit cap and the credit's (1 − ρ)

    def __post_init__(self) -> None:
        if (self.atom.dtype, self.atom.k, self.peel.dtype, self.peel.k) != ("e4m3", 32, "bf16", 16):
            raise ValueError(f"{self.name}: Pearl-C takes an E4M3 k32 atom and a BF16 k16 peel step")
        if self.g is not None and self.g < 1:
            raise ValueError(f"{self.name}: G is at least one atom")
        if not (isinstance(self.cap, Fraction) and 0 < self.cap < 1):
            raise ValueError(f"{self.name}: the cap ρ is a Fraction in (0, 1)")

    @property
    def window(self) -> int:
        """k's alignment: products per promotion (G atoms of k = 32), or one atom when nothing is promoted."""
        return self.atom.k * (self.g or 1)

    def promotions(self, k: int) -> int:
        """FP32 promotion adds per word of depth k."""
        return k // self.window if self.g else 0

    @property
    def pipelines(self) -> dict[str, tuple[tuple[int, ...], int, int]]:
        """Lean's `Pipeline` of each step: (groups, adder width, exponent floor)."""
        return {s: (m.groups, m.width, m.zero_exponent) for s, m in (("atom", self.atom), ("peel", self.peel))}


#: sm_90, the H100 (Rounds 10–11, `Prices.h100`): FADD 32, a BF16 MAC 2, and `Costs.adopted`'s forming, 64 + 4
H100_PRICES = Prices("sm_90", fadd=32, bf16_mac=2, forming_credited=64, forming_a_only=4)
#: sm_120, the RTX PRO 6000 (`Prices.sm120`; node 2, locked 2,100 MHz, `r20260930-063213-c94a` and
#: `r20260930-061717-3e84`): FADD 8, issue-bound (8.38 in the measured 64-add loop); a BF16 MAC 2; the k32 noise atom
#: (32) and the E4M3 cast alone (8; 12.31 in the packing loop), 40; ρ's one FFMA per 8 elements, 1
SM120_PRICES = Prices("sm_120", fadd=8, bf16_mac=2, forming_credited=40, forming_a_only=1)
PRICES = {p.device: p for p in (H100_PRICES, SM120_PRICES)}

H100 = Device("h100", HOPPER_E4M3_K32, HOPPER_BF16_WGMMA_K16, 4, H100_PRICES, 1 << 16)

SM120 = Device(
    "sm120", BLACKWELL_SM120_E4M3_M16N8K32, HOPPER_BF16_M16N8K16, 4, SM120_PRICES, 1 << 16,
    standins=(
        "peel: HOPPER_BF16_M16N8K16, matched on the RTX 5090's mma.sync m16n8k16 bf16 (204,800/204,800) and exact in the "
        "vLLM lanes' RTX PRO 6000 cuBLASLt and MoE correspondences; mma.sync m16n8k16 not probed on the RTX PRO 6000 itself",
        "g: 4, the H100 statement's promotion period (it answers Hopper's 14-bit adder); P1 and the debit not yet re-derived "
        "for the 26-bit atom",
        "k_max: 2^16, the H100's pure-chain freezing bound, until re-derived on the captured atom",
    ),
    cap=Fraction(1, 1000),
)

SM120_UNPROMOTED = replace(
    SM120, name="sm120-unpromoted", g=None,
    standins=tuple(s for s in SM120.standins if not s.startswith("g:")) + (
        "g: none, the panel's pearl-c-sm120 v2; Pearl-C's statement (P1, the debit, TT_OUT) is not restated without "
        "promotion",),
)

DEVICES = {d.name: d for d in (H100, SM120, SM120_UNPROMOTED)}
