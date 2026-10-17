import Definitions.Accounting.Carrier

/-!
# Stacking's named assumptions: encoding in an exact-integer circuit costs roundings, and bound keys (trusted)

What the stacking guarantees (`Models/Accounting/Guarantees/Stacking.lean`) assume beyond the remote audit's own, named
`Prop`s taken as hypotheses, never `axiom`s:

* `CircuitEncodeHardness b rEnc cost`: computing `x` of a block's `b` encoded bits in an exact-integer circuit costs
  at least `x/b` of `rEnc` roundings (claim `circuit-encode-hardness`).
* `KeyBinding δ store`: a store whose blocks written after the seed are encoded under bound keys is, on every event,
  at most `δ` likelier than a uniform store (claim `key-binding`).
-/

namespace Accounting.Assumptions

open Pous (pr)
open scoped ENNReal

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

/-- **`key-binding`: a store whose blocks written after the seed are under bound keys is a uniform store, but for
`δ`.** For every set `ev` of stores, the programmed store `store ω`, at a uniform `ω`, is in `ev` with probability at
most `δ` more than a uniform store:

  `Pr_ω[store ω ∈ ev] ≤ Pr_E[E ∈ ev] + δ`.

`ω` is everything the store's contents depend on: the codec, its seed, and every writer's choices. A block registered
before the seed is its payload encoded under the seed's key. A block written after it (a KV page: its payload is a
circuit's value, which may depend on other blocks' encodings) is its payload `p_j` encoded under the bound key
`K_j = H(seed ‖ j ‖ d_j)`, `d_j = sha512/row/v2(p_j)`, and it decodes only under the digest it was written with.

**Why it should hold.** In the ideal encoding model a block's encoding is uniform given its key and payload, and a
bound block's key is fresh for each payload, so a payload chosen in any way from other blocks leaves the block's
encoding uniform and independent of them; what the payload carries about other blocks is a carrier
(`RateSketch.Carrier`), charged apart. Choosing the encoding instead needs a payload that decodes from it under the key
of its own digest, a SHA-512 search, so `δ` is at most the writers' searches over `2^512`. Under the seed's key the
writer picks the encoding by writing its decoding: measured in
`verity/core/protocols/remote_bandwidth/tests/test_key_binding.py`, which also fails 64 tries under the bound key.
Not covered: a writer that chooses among several payloads by their encodings (grinding) moves its block's
distribution, by about the logarithm of its tries in bits; and a block written during a session, whose payload may
depend on the session's coins. -/
def KeyBinding {St Ω : Type} [Fintype St] [Fintype Ω] (δ : ℝ≥0∞) (store : Ω → St) : Prop :=
  ∀ ev : Set St, pr (fun ω => store ω ∈ ev) ≤ pr (fun E : St => E ∈ ev) + δ

end Accounting.Assumptions
