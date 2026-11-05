import Proofs.Flock.Soundness.ZK.RealLeaves

/-!
# An opening's empty class at one hm96 leaf

An opening of a coin tree's leaf releases the leaf's first coins, which the verifier compares with live coins drawn
after the leaf was committed. The opening's class at those coins (`OpeningAt.classOf`) holds each draw with the leaf's
hash that releases them, and `openEmpty` is the probability that it is empty. In the abstract: a leaf
`H(x(ω) + M·y, c(y))` of uniform coins `ω` and salt `y`, a release `P ω`, and live coins `u` whose bits `bits u` have
the release's law. `fiberMisses_le` bounds by `2·δ₁` the probability that no coins and salt with the leaf's hash
release `bits u`. That is T1's bound twice. T1 makes the leaf ideal, so the event no longer reads the coins, and
`bits u` may be swapped for the release `P ω`. T1 on the complement then makes the leaf real again, where the leaf's
own coins and salt release `P ω`.
-/

namespace Verity.Game

open scoped ENNReal

/-- No coins `ω` and salt `y` whose leaf `H(x(ω) + M·y, c(y))` is `ℓ` release `r`: at a leaf whose hash is `ℓ`, the
class of an opening that releases `P ω` is empty at `r`. -/
def FiberMisses {Ω Y B C D R : Type} [Add B] (My : Y → B) (c : Y → C) (H : B × C → D) (x : Ω → B) (P : Ω → R)
    (ℓ : D) (r : R) : Prop :=
  ∀ ω y, H (x ω + My y, c y) = ℓ → P ω ≠ r

/-- **An opening's class is empty with probability at most `2·δ₁`** at one hm96 leaf. Over the leaf's uniform coins `ω`
and salt `y`, and live coins `u` drawn independently whose bits `bits u` have the law of the release `P ω` (`hlaw`),
this is the probability that no coins and salt with the leaf's hash release `bits u`. -/
theorem fiberMisses_le {Ω Q Y B C D R : Type} [Fintype Ω] [Nonempty Ω] [Fintype Q] [Nonempty Q] [Fintype Y]
    [Nonempty Y] [AddCommGroup B] [Fintype B] (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D) (x : Ω → B) (P : Ω → R) (bits : Q → R)
    (hlaw : ∀ g : R → ℝ≥0∞, avg (fun ω => g (P ω)) = avg fun u => g (bits u)) :
    prCoin (fun p : (Ω × Q) × Y => FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (bits p.1.2))
      ≤ 2 * δ₁ := by
  have : Nonempty B := ⟨0⟩
  have h1 := FlockSoundness.ZK.ideal_leaf_swap My c hT1 H (fun w : Ω × Q => x w.1)
    (fun w ℓ => FiberMisses My c H x P ℓ (bits w.2))
  have h2 := FlockSoundness.ZK.ideal_leaf_swap My c hT1 H (fun w : Ω × Q => x w.1)
    (fun w ℓ => ¬ FiberMisses My c H x P ℓ (P w.1))
  have hno : ∀ p : (Ω × Q) × Y, ¬ FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (P p.1.1) :=
    fun p h => h p.1.1 p.2 rfl rfl
  have hown : prCoin (fun p : (Ω × Q) × Y => ¬ FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (P p.1.1))
      = 1 := by
    simp [prCoin, hno, FlockSoundness.Game.avg_const]
  have hswap : prCoin (fun p : (Ω × Q) × (B × Y) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (bits p.1.2))
      = prCoin (fun p : (Ω × Q) × (B × Y) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1)) := by
    rw [FlockSoundness.ZK.prCoin_prod_split
        (fun (w : Ω × Q) (u : B × Y) => FiberMisses My c H x P (H (u.1, c u.2)) (bits w.2)),
      FlockSoundness.ZK.prCoin_prod_split
        (fun (w : Ω × Q) (u : B × Y) => FiberMisses My c H x P (H (u.1, c u.2)) (P w.1)),
      FlockSoundness.ZK.avg_prod_split, FlockSoundness.ZK.avg_prod_split]
    simp only [FlockSoundness.Game.avg_const]
    exact (hlaw fun r => prCoin fun u : B × Y => FiberMisses My c H x P (H (u.1, c u.2)) r).symm
  have hideal : prCoin (fun p : (Ω × Q) × (B × Y) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1))
      ≤ δ₁ := by
    have a := FlockSoundness.ZK.prCoin_add_not
      (fun p : (Ω × Q) × (B × Y) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1))
    have hfin := FlockSoundness.ZK.prCoin_ne_top
      (fun p : (Ω × Q) × (B × Y) => ¬ FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1))
    refine (ENNReal.add_le_add_iff_right hfin).1 ?_
    rw [a, add_comm δ₁]
    exact hown.symm.le.trans h2
  calc prCoin (fun p : (Ω × Q) × Y => FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (bits p.1.2))
      ≤ prCoin (fun p : (Ω × Q) × (B × Y) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (bits p.1.2)) + δ₁ := h1
    _ = prCoin (fun p : (Ω × Q) × (B × Y) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1)) + δ₁ := by
        rw [hswap]
    _ ≤ δ₁ + δ₁ := add_le_add hideal le_rfl
    _ = 2 * δ₁ := (two_mul δ₁).symm

end Verity.Game
