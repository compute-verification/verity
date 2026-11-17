module

public import Definitions.Core.ValueSalts
public import Proofs.Core.Game.Basic
public import Proofs.Flock.Soundness.ZK.Hiding
public import Proofs.Flock.Soundness.CROnly.Hiding

/-! A registration's ideal record hides its values at hm96 commitments, within `N·2δ₁` and with no premise (`ideal_hm96`). -/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game Finset
open scoped ENNReal

theorem hm96_rev_gap {Y B C : Type} [Fintype Y] [Fintype B] [DecidableEq B] [DecidableEq C] [Nonempty Y] [Nonempty B]
    (My : Y → B) (c : Y → C) (E : B × C → Prop) :
    prCoin (fun p : B × Y => E (p.1, c p.2)) ≤
      prCoin (fun y : Y => E (My y, c y)) + ENNReal.ofReal (CROnly.hidingGap My c) := by
  refine (CROnly.prCoin_comp_le (fun p : B × Y => (p.1, c p.2)) (fun y => (My y, c y))
    ((univ : Finset B) ×ˢ univ.image c) (fun p => mem_product.2 ⟨mem_univ _, mem_image_of_mem c (mem_univ p.2)⟩)
    (fun y => mem_product.2 ⟨mem_univ _, mem_image_of_mem c (mem_univ y)⟩) E).trans
    (add_le_add le_rfl (ENNReal.ofReal_le_ofReal (le_of_eq ?_)))
  unfold CROnly.hidingGap
  congr 1
  exact Finset.sum_congr rfl fun v _ => abs_sub_comm _ _

theorem ideal_leaf_swap_rev {Ω Y B C D : Type} [Fintype Ω] [Nonempty Ω] [Fintype Y] [Nonempty Y] [AddCommGroup B]
    [Fintype B] [Nonempty B] [DecidableEq B] [DecidableEq C] (My : Y → B) (c : Y → C) (H : B × C → D) (x : Ω → B)
    (G : Ω → D → Prop) :
    prCoin (fun p : Ω × (B × Y) => G p.1 (H (p.2.1, c p.2.2)))
      ≤ prCoin (fun p : Ω × Y => G p.1 (H (x p.1 + My p.2, c p.2))) + ENNReal.ofReal (CROnly.hidingGap My c) := by
  rw [prCoin_prod_split (fun ω (u : B × Y) => G ω (H (u.1, c u.2))),
    prCoin_prod_split (fun ω y => G ω (H (x ω + My y, c y)))]
  calc avg (fun ω => prCoin fun u : B × Y => G ω (H (u.1, c u.2)))
      ≤ avg (fun ω => prCoin (fun y => G ω (H (x ω + My y, c y))) + ENNReal.ofReal (CROnly.hidingGap My c)) := by
        refine avg_mono fun ω => ?_
        have e : prCoin (fun u : B × Y => G ω (H (x ω + u.1, c u.2))) = prCoin fun u : B × Y => G ω (H (u.1, c u.2)) :=
          prCoin_equiv ((Equiv.addLeft (x ω)).prodCongr (Equiv.refl Y)) (fun u : B × Y => G ω (H (u.1, c u.2)))
        exact (le_of_eq e.symm).trans (hm96_rev_gap My c fun bc => G ω (H (x ω + bc.1, bc.2)))
    _ = avg (fun ω => prCoin fun y => G ω (H (x ω + My y, c y))) + ENNReal.ofReal (CROnly.hidingGap My c) := by
        rw [avg_add, avg_const]

theorem leaves_hybrid {Y X D : Type} [Fintype Y] [Nonempty Y] (L₁ L₂ : X → Y → D) {δ : ℝ≥0∞}
    (h1 : ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] (x : Ω → X) (G : Ω → D → Prop),
      prCoin (fun p : Ω × Y => G p.1 (L₁ (x p.1) p.2)) ≤ prCoin (fun p : Ω × Y => G p.1 (L₂ (x p.1) p.2)) + δ) :
    ∀ (n : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] (x : Ω → Fin n → X) (G : Ω → (Fin n → D) → Prop),
      prCoin (fun p : Ω × (Fin n → Y) => G p.1 fun i => L₁ (x p.1 i) (p.2 i))
        ≤ prCoin (fun p : Ω × (Fin n → Y) => G p.1 fun i => L₂ (x p.1 i) (p.2 i)) + n * δ := by
  intro n
  induction n with
  | zero =>
    intro Ω _ _ x G
    have e : (fun p : Ω × (Fin 0 → Y) => G p.1 fun i => L₁ (x p.1 i) (p.2 i))
        = fun p => G p.1 fun i => L₂ (x p.1 i) (p.2 i) := by
      funext p
      congr 1
      funext i
      exact i.elim0
    rw [e]
    simp
  | succ n ih =>
    intro Ω _ _ x G
    let e₁ : Ω × (Fin (n + 1) → Y) ≃ (Ω × (Fin n → Y)) × Y :=
      { toFun := fun p => ((p.1, Fin.tail p.2), p.2 0)
        invFun := fun q => (q.1.1, Fin.cons q.2 q.1.2)
        left_inv := fun p => by simp [Fin.cons_self_tail]
        right_inv := fun q => by simp }
    let eB : (Ω × (Fin n → Y)) × Y ≃ (Ω × Y) × (Fin n → Y) :=
      { toFun := fun p => ((p.1.1, p.2), p.1.2)
        invFun := fun p => ((p.1.1, p.2), p.1.2)
        left_inv := fun _ => rfl
        right_inv := fun _ => rfl }
    let eC : (Ω × Y) × (Fin n → Y) ≃ Ω × (Fin (n + 1) → Y) :=
      { toFun := fun p => (p.1.1, Fin.cons p.1.2 p.2)
        invFun := fun q => ((q.1, q.2 0), Fin.tail q.2)
        left_inv := fun p => by simp
        right_inv := fun q => by simp [Fin.cons_self_tail] }
    have step₁ := h1 (fun q : Ω × (Fin n → Y) => x q.1 0)
      (fun q d => G q.1 (Fin.cons d fun i => L₁ (x q.1 i.succ) (q.2 i)))
    have step₂ := ih (Ω := Ω × Y) (fun q i => x q.1 i.succ)
      (fun q ds => G q.1 (Fin.cons (L₂ (x q.1 0) q.2) ds))
    have hA : prCoin (fun p : Ω × (Fin (n + 1) → Y) => G p.1 fun i => L₁ (x p.1 i) (p.2 i))
        = prCoin (fun q : (Ω × (Fin n → Y)) × Y =>
            G q.1.1 (Fin.cons (L₁ (x q.1.1 0) q.2) fun i => L₁ (x q.1.1 i.succ) (q.1.2 i))) := by
      rw [← prCoin_congr_equiv e₁]
      congr 1
      funext p
      congr 1
      funext i
      refine Fin.cases ?_ (fun i => ?_) i <;> rfl
    have hB : prCoin (fun p : (Ω × (Fin n → Y)) × Y =>
          G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.2) fun i => L₁ (x p.1.1 i.succ) (p.1.2 i)))
        = prCoin (fun p : (Ω × Y) × (Fin n → Y) =>
          G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.1.2) fun i => L₁ (x p.1.1 i.succ) (p.2 i))) :=
      prCoin_congr_equiv eB (fun p : (Ω × Y) × (Fin n → Y) =>
        G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.1.2) fun i => L₁ (x p.1.1 i.succ) (p.2 i)))
    have hC : prCoin (fun p : (Ω × Y) × (Fin n → Y) =>
          G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.1.2) fun i => L₂ (x p.1.1 i.succ) (p.2 i)))
        = prCoin (fun p : Ω × (Fin (n + 1) → Y) => G p.1 fun i => L₂ (x p.1 i) (p.2 i)) := by
      rw [← prCoin_congr_equiv eC]
      congr 1
      funext p
      congr 1
      funext i
      refine Fin.cases ?_ (fun i => ?_) i <;> rfl
    calc prCoin (fun p : Ω × (Fin (n + 1) → Y) => G p.1 fun i => L₁ (x p.1 i) (p.2 i))
        = prCoin (fun q : (Ω × (Fin n → Y)) × Y =>
            G q.1.1 (Fin.cons (L₁ (x q.1.1 0) q.2) fun i => L₁ (x q.1.1 i.succ) (q.1.2 i))) := hA
      _ ≤ prCoin (fun p : (Ω × (Fin n → Y)) × Y =>
            G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.2) fun i => L₁ (x p.1.1 i.succ) (p.1.2 i))) + δ := step₁
      _ = prCoin (fun p : (Ω × Y) × (Fin n → Y) =>
            G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.1.2) fun i => L₁ (x p.1.1 i.succ) (p.2 i))) + δ := by
          rw [hB]
      _ ≤ (prCoin (fun p : (Ω × Y) × (Fin n → Y) =>
            G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.1.2) fun i => L₂ (x p.1.1 i.succ) (p.2 i))) + n * δ) + δ :=
          add_le_add step₂ le_rfl
      _ = prCoin (fun p : Ω × (Fin (n + 1) → Y) => G p.1 fun i => L₂ (x p.1 i) (p.2 i))
            + ((n + 1 : ℕ) : ℝ≥0∞) * δ := by
          rw [hC]
          push_cast
          ring

theorem leaves_swap_values {Y B C D : Type} [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B] [Nonempty B]
    [DecidableEq B] [DecidableEq C] (My : Y → B) (c : Y → C) (H : B × C → D) (n : ℕ) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (x x' : Ω → Fin n → B) (G : Ω → (Fin n → D) → Prop) :
    prCoin (fun p : Ω × (Fin n → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i)))
      ≤ prCoin (fun p : Ω × (Fin n → Y) => G p.1 fun i => H (x' p.1 i + My (p.2 i), c (p.2 i))) +
        n * (2 * ENNReal.ofReal (CROnly.hidingGap My c)) :=
  leaves_hybrid (δ := 2 * ENNReal.ofReal (CROnly.hidingGap My c)) (fun (a : B × B) y => H (a.1 + My y, c y))
    (fun (a : B × B) y => H (a.2 + My y, c y))
    (fun z G' => ((ideal_leaf_swap My c (CROnly.hm96Hiding_gap My c) H (fun ω => (z ω).1) G').trans
      (add_le_add (ideal_leaf_swap_rev My c H (fun ω => (z ω).2) G') le_rfl)).trans_eq
        (by rw [add_assoc, ← two_mul]))
    n (fun ω i => (x ω i, x' ω i)) G

open Verity.ValueSalts in
theorem ideal_hm96 {K S Ys B C D V O T R : Type} [Fintype S] [Nonempty S] [Fintype Ys] [Nonempty Ys] [Fintype T]
    [Nonempty T] [Fintype R] [Nonempty R] [AddCommGroup B] [Fintype B] [Nonempty B] [DecidableEq B] [DecidableEq C]
    {N : ℕ} (sv : Service K S (Fin N → Ys) V O T R) (My : Ys → B) (c : Ys → C) (H : B × C → D)
    (dig : V → Fin N → B) (g : (Fin N → D) → O)
    (hout : ∀ v y, sv.out v y = g fun i => H (dig v i + My (y i), c (y i))) (n m : ℕ) (v v₀ : V)
    (A : RegistrationRecord S O T R → Prop) :
    sv.ideal n m v A ≤ sv.ideal n m v₀ A + N * (2 * ENNReal.ofReal (CROnly.hidingGap My c)) := by
  let e : (S × T × R) × (Fin N → Ys) ≃ S × (Fin N → Ys) × T × R :=
    { toFun := fun q => (q.1.1, q.2, q.1.2.1, q.1.2.2)
      invFun := fun p => ((p.1, p.2.2.1, p.2.2.2), p.2.1)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have hv : ∀ u : V, sv.ideal n m u A = prCoin (fun q : (S × T × R) × (Fin N → Ys) =>
      A (n, m, (q.1.1, g fun i => H (dig u i + My (q.2 i), c (q.2 i)), q.1.2.1), q.1.2.2)) := fun u => by
    unfold Service.ideal
    simp only [hout]
    exact (prCoin_congr_equiv e _).symm
  rw [hv v, hv v₀]
  exact leaves_swap_values My c H N (fun (_ : S × T × R) i => dig v i) (fun _ i => dig v₀ i)
    (fun ω ls => A (n, m, (ω.1, g ls, ω.2.1), ω.2.2))

end FlockSoundness.ZK
