import Proofs.Flock.Soundness.Assumptions

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

/-- **hm96's keyed hash** (`verity/commitments/hm96/PROTOCOL.md` §2): the Hankel matrix of a key `κ` of `n + l − 1`
symbols applied to a salt of `l`, `(M(κ)·y)_i = Σ_j κ_{i+j}·y_j`. hm96 takes the symbols in GF(2), with `n = 512` and
`l = 1536` for `hm96-sha512`. -/
def hankel {F : Type} [Field F] {n l : ℕ} (κ : Fin (n + l - 1) → F) (y : Fin l → F) (i : Fin n) : F :=
  ∑ j : Fin l, κ ⟨i.val + j.val, by have := i.isLt; have := j.isLt; omega⟩ * y j

end FlockSoundness.CROnly
