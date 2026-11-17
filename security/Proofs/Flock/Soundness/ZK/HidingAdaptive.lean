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

/-! Zero knowledge: T1 over leaves whose digests read the leaves before them (`ideal_leaves_adaptive`). -/

@[expose] public section

namespace FlockSoundness.ZK

open Game
open scoped ENNReal

/-- **The leaves of an adaptive experiment**: leaf `i` is `H(x_i(the leaves before it) + m_i, c_i)`, where `π i = (m_i, c_i)`
is its mask and the hash of its salt. -/
def adaptiveLeaves {n : ℕ} {B C D : Type} [Add B] (H : B × C → D) (x : (i : Fin n) → (Fin i → D) → B)
    (π : Fin n → B × C) (i : Fin n) : D :=
  H (x i (fun j => adaptiveLeaves H x π (Fin.castLE i.isLt.le j)) + (π i).1, (π i).2)
termination_by i.val
decreasing_by exact j.isLt

theorem adaptiveLeaves_eq {n : ℕ} {B C D : Type} [Add B] (H : B × C → D) (x : (i : Fin n) → (Fin i → D) → B)
    (π : Fin n → B × C) (i : Fin n) :
    adaptiveLeaves H x π i = H (x i (fun j => adaptiveLeaves H x π (Fin.castLE i.isLt.le j)) + (π i).1, (π i).2) := by
  rw [adaptiveLeaves]

theorem adaptiveLeaves_unique {n : ℕ} {B C D : Type} [Add B] (H : B × C → D) (x : (i : Fin n) → (Fin i → D) → B)
    (π : Fin n → B × C) (f : Fin n → D)
    (hf : ∀ i, f i = H (x i (fun j => f (Fin.castLE i.isLt.le j)) + (π i).1, (π i).2)) :
    f = adaptiveLeaves H x π := by
  suffices h : ∀ k (i : Fin n), i.val = k → f i = adaptiveLeaves H x π i from funext fun i => h _ i rfl
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    intro i hi
    have e : (fun j : Fin i => f (Fin.castLE i.isLt.le j))
        = fun j => adaptiveLeaves H x π (Fin.castLE i.isLt.le j) :=
      funext fun j => ih j.val (lt_of_lt_of_eq j.isLt hi) (Fin.castLE i.isLt.le j) rfl
    rw [hf i, adaptiveLeaves_eq H x π i]
    exact congrArg (fun F => H (x i F + (π i).1, (π i).2)) e

theorem ideal_leaves_adaptive {Y B C D : Type} [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B]
    (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D)
    (n : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] (x : Ω → (i : Fin n) → (Fin i → D) → B)
    (G : Ω → (Fin n → D) → Prop) :
    prCoin (fun p : Ω × (Fin n → Y) => G p.1 (adaptiveLeaves H (x p.1) fun i => (My (p.2 i), c (p.2 i))))
      ≤ prCoin (fun p : Ω × (Fin n → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) + n * δ₁ := by
  let pairs : (Fin n → B × Y) → Fin n → B × C := fun u i => ((u i).1, c (u i).2)
  let ideal : (Fin n → B × Y) → Fin n → D := fun v i => H ((v i).1, c (v i).2)
  let φ : Ω → (Fin n → B × Y) → Fin n → B × Y := fun ω u i =>
    (x ω i (fun j => adaptiveLeaves H (x ω) (pairs u) (Fin.castLE i.isLt.le j)) + (u i).1, (u i).2)
  let ψ : Ω → (Fin n → B × Y) → Fin n → B × Y := fun ω v i =>
    ((v i).1 - x ω i (fun j => ideal v (Fin.castLE i.isLt.le j)), (v i).2)
  have hφ : ∀ ω u, ideal (φ ω u) = adaptiveLeaves H (x ω) (pairs u) := fun ω u =>
    funext fun i => (adaptiveLeaves_eq H (x ω) (pairs u) i).symm
  have hψ : ∀ ω v, adaptiveLeaves H (x ω) (pairs (ψ ω v)) = ideal v := fun ω v =>
    (adaptiveLeaves_unique H (x ω) (pairs (ψ ω v)) (ideal v) fun i => by
      show H ((v i).1, c (v i).2) = H (x ω i (fun j => ideal v (Fin.castLE i.isLt.le j))
        + ((v i).1 - x ω i (fun j => ideal v (Fin.castLE i.isLt.le j))), c (v i).2)
      rw [add_sub_cancel]).symm
  have hψφ : ∀ ω u, ψ ω (φ ω u) = u := fun ω u => funext fun i => by
    refine Prod.ext ?_ rfl
    show x ω i (fun j => adaptiveLeaves H (x ω) (pairs u) (Fin.castLE i.isLt.le j)) + (u i).1
        - x ω i (fun j => ideal (φ ω u) (Fin.castLE i.isLt.le j)) = (u i).1
    rw [hφ ω u]
    exact add_sub_cancel_left _ _
  have hφψ : ∀ ω v, φ ω (ψ ω v) = v := fun ω v => funext fun i => by
    refine Prod.ext ?_ rfl
    show x ω i (fun j => adaptiveLeaves H (x ω) (pairs (ψ ω v)) (Fin.castLE i.isLt.le j))
        + ((v i).1 - x ω i (fun j => ideal v (Fin.castLE i.isLt.le j))) = (v i).1
    rw [hψ ω v]
    exact add_sub_cancel _ _
  let e : Ω × (Fin n → B × Y) ≃ Ω × (Fin n → B × Y) :=
    { toFun := fun q => (q.1, φ q.1 q.2)
      invFun := fun q => (q.1, ψ q.1 q.2)
      left_inv := fun q => Prod.ext rfl (hψφ q.1 q.2)
      right_inv := fun q => Prod.ext rfl (hφψ q.1 q.2) }
  have h1 : prCoin (fun p : Ω × (Fin n → Y) => G p.1 (adaptiveLeaves H (x p.1) fun i => (My (p.2 i), c (p.2 i))))
      ≤ prCoin (fun p : Ω × (Fin n → B × Y) => G p.1 (adaptiveLeaves H (x p.1) (pairs p.2))) + n * δ₁ := by
    simpa only [id, zero_add] using
      ideal_leaves_swap My c hT1 (id : B × C → B × C) n (Ω := Ω) (fun _ _ => (0 : B))
        (fun ω π => G ω (adaptiveLeaves H (x ω) π))
  have h2 : prCoin (fun p : Ω × (Fin n → B × Y) => G p.1 (adaptiveLeaves H (x p.1) (pairs p.2)))
      = prCoin (fun p : Ω × (Fin n → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) := by
    refine Eq.trans ?_ (prCoin_equiv e fun p : Ω × (Fin n → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2))
    congr 1
    funext q
    show G q.1 (adaptiveLeaves H (x q.1) (pairs q.2)) = G q.1 (ideal (φ q.1 q.2))
    exact congrArg (G q.1) (hφ q.1 q.2).symm
  rw [← h2]
  exact h1

end FlockSoundness.ZK
