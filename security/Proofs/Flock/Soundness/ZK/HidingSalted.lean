module

public import Definitions.Core.Game
public import Mathlib.Data.Fintype.Pi
public import Proofs.Flock.Soundness.ZK.Hiding
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Combinatorics.Enumerative.DoubleCounting
import Mathlib.Data.Nat.Bitwise
import Mathlib.Tactic.Positivity.Finset
import Proofs.Flock.Soundness.Assumptions
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.Rewinding
import Std.Data.TreeMap.Raw.Lemmas

/-!
# Zero knowledge: T1 over leaves whose digests read the earlier salts

`ideal_leaves_swap` swaps `n` hm96 leaves for ideal ones when each salt is read by its own leaf alone. A worker that is
answered each salt after its commitment may read it in every later message, so digest `i` reads the salts before `i`
(and through them every earlier leaf). The same bound `n·δ₁` holds, swapping the leaves last to first: the last salt is
read by the last leaf alone, so `ideal_leaf_swap` replaces that leaf, with the earlier salts part of the experiment. The
ideal leaf reads its salt only through `c(y)`, so the leaves before it form an experiment of the same shape, one leaf
shorter.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Game
open scoped ENNReal

/-- **T1 over `n` leaves whose digests read the earlier salts**: in an experiment on uniform `(ω, y₀ … y_{n−1})` whose leaf
`i` is `H(x_i(ω, y₀ … y_{i−1}) + M·y_i, c(y_i))`, replacing every leaf by an ideal one `H(U_i, c(y_i))` with fresh uniform
`U_i` changes the probability of any event by at most `n·δ₁`. -/
theorem ideal_leaves_salted {Y B C D : Type} [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B]
    (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D) :
    ∀ (n : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] (x : Ω → (i : Fin n) → (Fin i → Y) → B)
      (G : Ω → (Fin n → D) → Prop),
      prCoin (fun p : Ω × (Fin n → Y) =>
          G p.1 fun i => H (x p.1 i (fun j => p.2 (Fin.castLE i.isLt.le j)) + My (p.2 i), c (p.2 i)))
        ≤ prCoin (fun p : Ω × (Fin n → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) + n * δ₁ := by
  intro n
  induction n with
  | zero =>
    intro Ω _ _ x G
    let e : Ω × (Fin 0 → Y) ≃ Ω × (Fin 0 → B × Y) := (Equiv.refl Ω).prodCongr
      ⟨fun _ => Fin.elim0, fun _ => Fin.elim0, fun _ => funext fun i => i.elim0, fun _ => funext fun i => i.elim0⟩
    have h := prCoin_congr_equiv e (fun p : Ω × (Fin 0 → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2))
    have e1 : (fun p : Ω × (Fin 0 → Y) =>
          G p.1 fun i => H (x p.1 i (fun j => p.2 (Fin.castLE i.isLt.le j)) + My (p.2 i), c (p.2 i)))
        = fun p => G (e p).1 fun i => H (((e p).2 i).1, c ((e p).2 i).2) := by
      funext p
      congr 1
      funext i
      exact i.elim0
    rw [e1, h]
    simp
  | succ n ih =>
    intro Ω _ _ x G
    -- the leaves before the last, at the first `n` salts
    let L : Ω × (Fin n → Y) → Fin n → D := fun q i =>
      H (x q.1 i.castSucc (fun j => q.2 (Fin.castLE i.isLt.le j)) + My (q.2 i), c (q.2 i))
    let e₁ : Ω × (Fin (n + 1) → Y) ≃ (Ω × (Fin n → Y)) × Y :=
      { toFun := fun p => ((p.1, Fin.init p.2), p.2 (Fin.last n))
        invFun := fun q => (q.1.1, Fin.snoc q.1.2 q.2)
        left_inv := fun p => by simp [Fin.snoc_init_self]
        right_inv := fun q => by simp }
    let eB : (Ω × (Fin n → Y)) × (B × Y) ≃ (Ω × (B × Y)) × (Fin n → Y) :=
      { toFun := fun p => ((p.1.1, p.2), p.1.2)
        invFun := fun p => ((p.1.1, p.2), p.1.2)
        left_inv := fun _ => rfl
        right_inv := fun _ => rfl }
    let eC : (Ω × (B × Y)) × (Fin n → B × Y) ≃ Ω × (Fin (n + 1) → B × Y) :=
      { toFun := fun p => (p.1.1, Fin.snoc p.2 p.1.2)
        invFun := fun q => ((q.1, q.2 (Fin.last n)), Fin.init q.2)
        left_inv := fun p => by simp
        right_inv := fun q => by simp [Fin.snoc_init_self] }
    -- the last leaf, with the earlier salts part of the experiment: no other leaf reads its salt
    have step₁ := ideal_leaf_swap My c hT1 H (fun q : Ω × (Fin n → Y) => x q.1 (Fin.last n) q.2)
      (fun q d => G q.1 (Fin.snoc (α := fun _ => D) (L q) d))
    -- the other `n`, with the last leaf's ideal seed part of the experiment
    have step₂ := ih (Ω := Ω × (B × Y)) (fun q i => x q.1 i.castSucc)
      (fun q ds => G q.1 (Fin.snoc (α := fun _ => D) ds (H (q.2.1, c q.2.2))))
    have hA : prCoin (fun p : Ω × (Fin (n + 1) → Y) =>
          G p.1 fun i => H (x p.1 i (fun j => p.2 (Fin.castLE i.isLt.le j)) + My (p.2 i), c (p.2 i)))
        = prCoin (fun q : (Ω × (Fin n → Y)) × Y =>
            G q.1.1 (Fin.snoc (α := fun _ => D) (L q.1) (H (x q.1.1 (Fin.last n) q.1.2 + My q.2, c q.2)))) := by
      rw [← prCoin_congr_equiv e₁]
      congr 1
      funext p
      congr 1
      funext i
      induction i using Fin.lastCases with
      | last =>
        rw [Fin.snoc_last]
        rfl
      | cast i =>
        rw [Fin.snoc_castSucc]
        rfl
    have hB : prCoin (fun p : (Ω × (Fin n → Y)) × (B × Y) =>
          G p.1.1 (Fin.snoc (α := fun _ => D) (L p.1) (H (p.2.1, c p.2.2))))
        = prCoin (fun p : (Ω × (B × Y)) × (Fin n → Y) =>
          G p.1.1 (Fin.snoc (α := fun _ => D) (L (p.1.1, p.2)) (H (p.1.2.1, c p.1.2.2)))) :=
      prCoin_congr_equiv eB (fun p : (Ω × (B × Y)) × (Fin n → Y) =>
        G p.1.1 (Fin.snoc (α := fun _ => D) (L (p.1.1, p.2)) (H (p.1.2.1, c p.1.2.2))))
    have hC : prCoin (fun p : (Ω × (B × Y)) × (Fin n → B × Y) =>
          G p.1.1 (Fin.snoc (α := fun _ => D) (fun i => H ((p.2 i).1, c (p.2 i).2)) (H (p.1.2.1, c p.1.2.2))))
        = prCoin (fun p : Ω × (Fin (n + 1) → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) := by
      rw [← prCoin_congr_equiv eC]
      congr 1
      funext p
      congr 1
      funext i
      induction i using Fin.lastCases with
      | last => simp [eC]
      | cast i => simp [eC]
    calc prCoin (fun p : Ω × (Fin (n + 1) → Y) =>
          G p.1 fun i => H (x p.1 i (fun j => p.2 (Fin.castLE i.isLt.le j)) + My (p.2 i), c (p.2 i)))
        = prCoin (fun q : (Ω × (Fin n → Y)) × Y =>
            G q.1.1 (Fin.snoc (α := fun _ => D) (L q.1) (H (x q.1.1 (Fin.last n) q.1.2 + My q.2, c q.2)))) :=
          hA
      _ ≤ prCoin (fun p : (Ω × (Fin n → Y)) × (B × Y) =>
            G p.1.1 (Fin.snoc (α := fun _ => D) (L p.1) (H (p.2.1, c p.2.2)))) + δ₁ :=
          step₁
      _ = prCoin (fun p : (Ω × (B × Y)) × (Fin n → Y) =>
            G p.1.1 (Fin.snoc (α := fun _ => D) (L (p.1.1, p.2)) (H (p.1.2.1, c p.1.2.2)))) + δ₁ := by
          rw [hB]
      _ ≤ (prCoin (fun p : (Ω × (B × Y)) × (Fin n → B × Y) =>
            G p.1.1 (Fin.snoc (α := fun _ => D) (fun i => H ((p.2 i).1, c (p.2 i).2)) (H (p.1.2.1, c p.1.2.2))))
              + n * δ₁) + δ₁ :=
          add_le_add step₂ le_rfl
      _ = prCoin (fun p : Ω × (Fin (n + 1) → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2))
            + ((n + 1 : ℕ) : ℝ≥0∞) * δ₁ := by
          rw [hC]
          push_cast
          ring

end FlockSoundness.ZK
