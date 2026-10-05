import FlockSoundness.Assumptions
import FlockSoundness.Protocol.CROnly.Hm96Defs
import FlockSoundness.Game.Expect
import FlockSoundness.Randomness

/-!
# hm96 hiding from the universal hash: the definitions

`Assumptions.Hm96Hiding My c δ₁` bounds, for every test, how much the pair `(My y, c y)` of a uniform salt `y` is told
apart from `(U, c y)` with `U` uniform. Its least `δ₁` is the statistical distance of the two pairs, `hidingGap My c`.
hm96's `My` is the Hankel matrix of its key (`hankel`, `verity/commitments/hm96/PROTOCOL.md` §2), a universal family
(`Universal`), and the leftover hash lemma bounds the gap's mean over a uniform key (§3).
-/

namespace FlockSoundness.CROnly

open Finset

/-- **The hiding gap**: the statistical distance between `(My y, c y)` for `y` uniform on `Y` and `(U, c y)` for
`(U, y)` uniform on `B × Y`, a sum over the pairs either takes, `B × c(Y)`. -/
noncomputable def hidingGap {Y B C : Type} [Fintype Y] [Fintype B] [DecidableEq B] [DecidableEq C] (My : Y → B)
    (c : Y → C) : ℝ :=
  1 / 2 * ∑ v ∈ (univ : Finset B) ×ˢ univ.image c,
    |(#{y | (My y, c y) = v} : ℝ) / Fintype.card Y - (#{p : B × Y | (p.1, c p.2) = v} : ℝ) / Fintype.card (B × Y)|

/-- **A universal family** of maps `M k : Y → B`: two distinct inputs collide under at most a `1/|B|` fraction of the
keys. -/
def Universal {K Y B : Type} [Fintype K] [Fintype B] [DecidableEq B] (M : K → Y → B) : Prop :=
  ∀ y y', y ≠ y' → #{k | M k y = M k y'} * Fintype.card B ≤ Fintype.card K

end FlockSoundness.CROnly
