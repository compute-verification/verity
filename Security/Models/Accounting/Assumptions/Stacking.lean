import Definitions.Accounting.Carrier

/-!
# Stacking's named assumption: encoding in an exact-integer circuit costs roundings (trusted)

What the stacking guarantee (`Models/Accounting/Guarantees/Stacking.lean`) assumes beyond the remote audit's own, a
named `Prop` taken as a hypothesis, never an `axiom`:

* `CircuitEncodeHardness b rEnc cost`: computing `x` of a block's `b` encoded bits in an exact-integer circuit costs
  at least `x/b` of `rEnc` roundings (claim `circuit-encode-hardness`).
-/

namespace Accounting.Assumptions

/-- **`circuit-encode-hardness`: an exact-integer circuit needs `rEnc` roundings to encode a fresh block, `x/b` of
them for `x` of its `b` bits.** For every `x ≤ b`:

  `x · rEnc ≤ b · cost x`,

where `cost x` is the fewest roundings in which a circuit of exact-integer operations (integer matrix products that
neither wrap nor clamp, linear maps, roundings) computes `x` bits of a block's encoding (any `x`-bit function of it)
from the block's payload and anything else it holds, the block's other `b − x` bits among them, one block at a time
(no amortization across blocks). It is what `RateSketch.Provenance.roundings` charges an `encode` step of `x` bits.
With a budget of `R` roundings over a carrier's life it bounds the bits a circuit encodes by `b·R/rEnc`: at the
property's `c_ρ` roundings per byte of certified read, `q ≤ c_ρ·reads·L/rEnc` blocks over a life of `L` windows.

**Why it should hold.** Products are exact and free in this count; what costs is a rounding, the one operation that
discards information (a floor, a right shift, a modular reduction step). An `n`-limb modular squaring needs at least
`n − O(1)` roundings that each read a bounded window of limbs, even with free products (the carry lower bound,
note:20261017T0810Z-report-restricted-loopholes, section (7)), and encoding a word is about `rounds · (b_word − 2)`
modular squarings (`Accounting.Assumptions.WorkHardness`'s price at the stated codec's `rounds`), so `rEnc` should be
at least their product per word, times the block's words. It is conjectured, not proved or measured: the carry bound is
per squaring, not per encoding, and the partial form must also hold against a completion from the block's other bits,
as the red team's lattice completion is against `WorkHardness` (`ra-rate-redteam`). An exact-integer circuit is
necessary: with FP8 or BF16 matrix products the FP32 accumulator gives about one free floor per product, and the
assumption is false for such circuits. -/
def CircuitEncodeHardness (b rEnc : ℕ) (cost : ℕ → ℕ) : Prop :=
  ∀ x, x ≤ b → x * rEnc ≤ b * cost x

end Accounting.Assumptions
