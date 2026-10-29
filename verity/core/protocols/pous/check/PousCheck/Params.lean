/-!
The deployed P2 parameters (P2 v4, profile `p2-2054B`): the prime, the block geometry, the audit's
challenge count and its deadline.
-/

namespace PousCheck

/-- `p = 2^16448 - 21065`, prime by the ECPP certificate the protocol pins. -/
def P : Nat := 2 ^ 16448 - 21065

/-- Bits and bytes of one encoded block. -/
def W_BITS : Nat := 16448
def W_BYTES : Nat := 2056

/-- The deployed audit: `k` sequential challenges, each answered within `Δ + RTT`, with the round-trip allowance capped. -/
def K : Nat := 105
def DELTA_NS : Nat := 500000
def RTT_CAP_NS : Nat := 354000

end PousCheck
