import Mathlib.Data.FinEnum
import Definitions.Core.Game
import Definitions.Hash.Basic

/-!
# The hash area's named assumption: keyed SHA-512 is a pseudorandom function

`Sha512Prf` (`prf/sha-512`): SHA-512 keyed by secret bytes right after a public prefix, at inputs none of which is a
prefix of another, passes a test as a random function does, up to `ε_prf(t, Q)`. An assumed advantage bound, allowed
by Daniel's ruling on ASK-184 (B, 28 Oct): a statement that carries a bound from a model in which SHA-512's keyed
outputs are uniform to the deployed SHA-512 takes it and adds its `ε_prf`.
-/

namespace Hash.Assumptions

open scoped ENNReal

/-- **`prf/sha-512`: keyed SHA-512 is a pseudorandom function, with advantage at most `ε t Q`.**

The key is `κ ≥ 32` uniform bytes `K` placed after a public prefix `pre`, the two within SHA-512's first 1024-bit
message block; the inputs are `Q` public strings `xs i`, none a prefix of another; the function is
`F_K(x) = H(pre ‖ K ‖ x)`. Its answers pass the test `T` with at most the probability that `Q` independent uniform
digests do (a random function's answers at `Q` distinct inputs), plus `ε t Q`, where `t` is the test's time and `Q`
(`Fintype.card ι`) its queries. The test has uniform coins of its own (`Y`), as a randomized distinguisher does.

What it assumes of SHA-512 is what the `prf/sha-512` claim names (`verity.core.claims`): its compression function is a
PRF keyed by its chaining value, which makes the cascade a PRF on prefix-free inputs (Bellare, Canetti and Krawczyk
1996), and a dual PRF keyed by the secret bytes of its first message block, as the HMAC analyses use. Prefix-free
inputs keep length extension out: no padded input is a prefix of another. `ε` is the claim's advantage function, the
number a statement that takes this premise adds to its bound. A4 (`FlockSoundness.Assumptions.KeyedStreamsUniform`),
the same claim stated on `verity.randomness`'s keyed streams, puts its advantage at most `2^-128` for its tests at a
32-byte key.

It is a predicate on one test, with the test's time declared (`t`), as A4 and `SHA512CRExpected` are: a statement for
every test would be refutable, since `2^(8κ)` keys give at most that many tuples of answers, and Lean has no cost model
of a test. The theorems take it at the tests they build from the prover, at the prover's declared time. -/
def Sha512Prf (H : Msg → Digest) (ε : ℕ → ℕ → ℝ≥0∞) (pre : Msg) (κ : ℕ) {ι : Type} [Fintype ι] [DecidableEq ι]
    (xs : ι → Msg)
    {Y : Type} [Fintype Y] (T : Y → (ι → Digest) → Prop) (t : ℕ) : Prop :=
  32 ≤ κ → pre.length + κ ≤ 128 → (∀ i j, xs i <+: xs j → i = j) →
    Verity.Game.prCoin (fun z : (Fin κ → UInt8) × Y => T z.2 fun i => H (pre ++ List.ofFn z.1 ++ xs i)) ≤
      Verity.Game.prCoin (fun z : (ι → Digest) × Y => T z.2 z.1) + ε t (Fintype.card ι)

end Hash.Assumptions
