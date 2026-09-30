"""Pearl-C's device record: the hardware facts a Pearl-C instance is stated on, as one scheme parameter.

Every checked value goes through two tensor-core steps (``verity.ml.tc`` models): the FP8 atom (E4M3 x E4M3 into an FP32
accumulator, k = 32) that C̃, the fused A′·F_B and B̃·F_A, G and the forming atom run on, and the BF16 step (k = 16) that
the peel accumulates into C̃'s word.  Beside them the record holds the promotion period G (atoms per fresh +0 accumulator,
promoted by one FP32 nearest-even add), the W1 prices (`Prices`) that `wref`, `credit_of`, `job_cost` and the debit read,
and the domain's k bound.  `pearl_c`, `pearl_c_debit`, `pearl_c_u` and `pearl_c_work` read the record; nothing else in them
names hardware.

* `H100` is the record every ``pearl-c-h100-*`` output was pinned on: `HOPPER_E4M3_K32`, `HOPPER_BF16_WGMMA_K16`, G = 4,
  Rounds 10-11's W1 prices (research store ``docs/pouw/cheap-binding.md`` §1) and k <= 2^16 (the pure-chain freezing bound).
* `SM120` is the RTX PRO 6000 Blackwell's (``mma.sync``: ``kind::f8f6f4`` m16n8k32 E4M3 and m16n8k16 BF16).  **Until the
  card's FP8 capture lands, every field is a stand-in** (`standins` names each): Ada's `ADA_E4M3_M16N8K32` as the first
  hypothesis for the atom, `HOPPER_BF16_M16N8K16` (which matched the RTX 5090's ``mma.sync``) for the peel, and G = 4, the
  H100's prices and k bound as placeholders.  G, the prices and the k bound are statement parameters: they change through
  the coordinator and the cheap-binding and Lean lanes, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.ml.tc.models import (ADA_E4M3_M16N8K32, HOPPER_BF16_M16N8K16, HOPPER_BF16_WGMMA_K16, HOPPER_E4M3_K32,
                                 GroupSum)


@dataclass(frozen=True, slots=True)
class Prices:
    """W1 prices, in units of one dense FP8 MAC on the device."""

    fadd: int                # one FP32 add (a promotion)
    bf16_mac: int            # one BF16 MAC (the peel)
    forming_credited: int    # `Costs.adopted`'s fs per operand element: the noise atom and the noisy cast
    forming_a_only: int      # `Costs.adopted`'s qa per operand element: ρ, which reads the activations alone
    hash_byte: int           # one TurboSHAKE128 byte; reported beside the unit, never in W_ref (the random oracle's)


@dataclass(frozen=True, slots=True)
class Device:
    """One device's record (module docstring)."""

    name: str                # the scheme name's device part: pearl-c-<name>-v1
    atom: GroupSum           # the FP8 atom, E4M3 k32
    peel: GroupSum           # the BF16 peel step, k16
    g: int                   # atoms per promotion (`Params.pi.G`)
    prices: Prices
    k_max: int               # the domain's bound on k
    standins: tuple[str, ...] = ()   # the fields that are stand-ins, each with what it stands in for

    def __post_init__(self) -> None:
        if (self.atom.dtype, self.atom.k, self.peel.dtype, self.peel.k) != ("e4m3", 32, "bf16", 16):
            raise ValueError(f"{self.name}: Pearl-C takes an E4M3 k32 atom and a BF16 k16 peel step")
        if self.g < 1:
            raise ValueError(f"{self.name}: G is at least one atom")

    @property
    def window(self) -> int:
        """Products per promotion: G atoms of k = 32."""
        return self.g * self.atom.k

    @property
    def pipelines(self) -> dict[str, tuple[tuple[int, ...], int, int]]:
        """Lean's `Pipeline` of each step: (groups, adder width, exponent floor)."""
        return {s: (m.groups, m.width, m.zero_exponent) for s, m in (("atom", self.atom), ("peel", self.peel))}


H100_PRICES = Prices(fadd=32, bf16_mac=2, forming_credited=64, forming_a_only=4, hash_byte=826)

H100 = Device("h100", HOPPER_E4M3_K32, HOPPER_BF16_WGMMA_K16, 4, H100_PRICES, 1 << 16)

SM120 = Device(
    "sm120", ADA_E4M3_M16N8K32, HOPPER_BF16_M16N8K16, 4, H100_PRICES, 1 << 16,
    standins=(
        "atom: ADA_E4M3_M16N8K32 (RTX 4090 mma.sync m16n8k32 e4m3), the first hypothesis until the RTX PRO 6000's "
        "kind::f8f6f4 m16n8k32 E4M3 capture",
        "peel: HOPPER_BF16_M16N8K16, matched on the RTX 5090's mma.sync m16n8k16 bf16 (204,800/204,800); not yet "
        "rechecked on the RTX PRO 6000",
        "g: 4, the H100's promotion period (answers Hopper's 14-bit adder), until the capture says whether chains freeze",
        "prices: the H100's W1 prices (fadd 32, bf16 MAC 2, fs 64, qa 4, 826 per hashed byte), until measured on the card",
        "k_max: 2^16, the H100's pure-chain freezing bound, until re-derived on the captured atom",
    ),
)

DEVICES = {d.name: d for d in (H100, SM120)}
