import Mathlib.Basic.Real.Basic
import RunModel.Protocol

/-!
# Named assumptions (trusted)

The primitive assumptions the run's guarantees rest on, stated over the executable's own SHA-512 and hm96 functions.
They are C-Flock's soundness package's (`FlockSoundness.Assumptions.SHA512CRStrict`, `Hm96Hiding`), restated because
that package needs ArkLib; both are written in Mathlib's and core's constants and the executable verifier's, so their
home is core's `Verity.Assumptions` (open question in the memo). A3 is embodied by `fresh`, not assumed.
-/

namespace RunModel.Assumptions

open Run
open scoped ENNReal

/-- **`cr/sha-512`, for one finder**: a finder (a game of core's played by a fixed strategy) that makes at most `q`
SHA-512 evaluations on every outcome outputs a collision of `Flock.Sha512.hash` with probability at most `q² / 2^513`.
A predicate on one finder, taken for the finders built from the developer. -/
def Sha512CR {β : Type} (g : Verity.Game β) (s : Verity.Game.Strategy g) (out : β → Option (ByteArray × ByteArray))
    (cost : β → ℝ) (q : ℝ) : Prop :=
  (∀ b, cost b ≤ q) →
    (Verity.Game.prob (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) g
      s).toReal ≤ q ^ 2 / (2 : ℝ) ^ 513

/-- `cr/sha-512` for a `Finder`. -/
def FinderCR (f : Finder) (q : ℝ) : Prop := Sha512CR f.game f.strat f.out f.cost q

/-- What `cr/sha-512` charges for `n` finders at budget `q` each. -/
noncomputable def collisions (n : ℕ) (q : ℝ) : ℝ≥0∞ := n * ENNReal.ofReal (q ^ 2 / (2 : ℝ) ^ 513)

/-- The salt's own hash, `c = H(pad(SALT_TAG) ‖ y)`. -/
def saltHash (y : ByteArray) : ByteArray := Flock.Sha512.hash (Flock.Hm96.Default512.saltPrefix ++ y)

/-- **hm96-sha512/v1 hides, up to `δ₁` per leaf**: for a uniform salt `y`, `(M·y, c(y))` and `(U, c(y))` with `U`
uniform are within `δ₁` on every event, so a leaf (a function of `x ⊕ M·y` and `c(y)`) tells nothing of `x` beyond
`δ₁`. Statistical, under the pinned key (`hash-derived-key`, `δ₁ = 2^-193`). -/
def Hm96Hiding (δ₁ : ℝ≥0∞) : Prop :=
  ∀ E : Set (ByteArray × ByteArray),
    ((fresh 192).map fun y => (Flock.Hm96.Default512.mask y, saltHash y)).toOuterMeasure E ≤
      ((fresh 64).bind fun u => (fresh 192).map fun y => (u, saltHash y)).toOuterMeasure E + δ₁

end RunModel.Assumptions
