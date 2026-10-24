import Definitions.Accounting.Guarantees.TS
import Proofs.Pouw.RelativeSecurity

/-!
# Binding from the fork form

`Accounting.SecurityProofs.TS.BindingOfFork : Accounting.Guarantees.TS.BindingOfFork`, C-Flock's rewinding step
(`Proofs/Flock/Recursive/Fork.lean`) over the TS model's uniform coins. Fix the coins `x` before round `w`; its
continuations are `splice w x z` over every `z`. With `n` of them proved and `n_v` opening round `w`'s leaves to `v`,
the reference opening has the most, `n_*`, so `x`'s moved continuations are `n − n_*`, and its forked pairs are
`n² − Σ_v n_v² ≥ n² − n·n_* ≥ (n − n_*)²` (`moved_sq_le`). Cauchy–Schwarz over `x` gives the square of the moved mass
at most the forked mass. Sums over continuations are sums over paths because `(x, z) ↦ (splice w x z, splice w z x)`
is an involution (`splice_swap`, `sum_splice`).
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Pouw.RelativeSecurity

/-- A continuation's continuation from the same round is a continuation of the first path. -/
theorem splice_left {Ω : Type} : ∀ {n : ℕ} (w : ℕ) (x z z' : Paths Ω n), splice w (splice w x z) z' = splice w x z'
  | 0, _, _, _, _ => rfl
  | _ + 1, 0, _, _, _ => rfl
  | _ + 1, w + 1, x, z, z' => by simp only [splice, splice_left w x.2 z.2 z'.2]

/-- Swapping two paths' coins from round `w` on, twice, gives the first back. -/
theorem splice_swap {Ω : Type} : ∀ {n : ℕ} (w : ℕ) (x z : Paths Ω n), splice w (splice w x z) (splice w z x) = x
  | 0, _, _, _ => rfl
  | _ + 1, 0, _, _ => rfl
  | _ + 1, w + 1, x, z => by simp only [splice, splice_swap w x.2 z.2]

/-- Summing over every path and every continuation of it is summing over paths, each `|Paths|` times. -/
theorem sum_splice {Ω : Type} [Fintype Ω] {n : ℕ} (w : ℕ) (q : Paths Ω n → ℕ) :
    ∑ x : Paths Ω n, ∑ z : Paths Ω n, q (splice w x z) = Fintype.card (Paths Ω n) * ∑ y, q y := by
  let Φ : Paths Ω n × Paths Ω n → Paths Ω n × Paths Ω n := fun p => (splice w p.1 p.2, splice w p.2 p.1)
  have hΦ : Function.Involutive Φ := fun p => by simp only [Φ, splice_swap]
  have h := Equiv.sum_comp (hΦ.toPerm Φ) (fun p => q p.1)
  simp only [Function.Involutive.coe_toPerm, Φ] at h
  rw [← Fintype.sum_prod_type' (fun x z => q (splice w x z)), h, Fintype.sum_prod_type (fun p => q p.1),
    Finset.mul_sum]
  simp [Finset.sum_const, Finset.card_univ, mul_comm]

namespace ProofLayer

variable {Ω : Type} {S nπ : ℕ} {V : Type} (pf : Accounting.TS.ProofLayer Ω S nπ V)

open Classical

theorem provedAt_splice [Fintype Ω] (w : Fin S) (x z : Paths Ω S) :
    pf.provedAt (splice w x z) w = pf.provedAt x w := by
  ext z'; simp [Accounting.TS.ProofLayer.provedAt, splice_left]

theorem weight_splice [Fintype Ω] (w : Fin S) (x z z' : Paths Ω S) :
    pf.weight (splice w x z) z' w = pf.weight x z' w := by
  simp only [Accounting.TS.ProofLayer.weight, provedAt_splice, splice_left]

/-- Every continuation of `x` from round `w` reads round `w`'s commitment at `x`'s reference continuation. -/
theorem ref_splice [Fintype Ω] (w : Fin S) (x z : Paths Ω S) (h : (pf.provedAt x w).Nonempty) :
    pf.ref (splice w x z) w = pf.ref x w := by
  have hfun : (fun z' => pf.weight (splice w x z) z' w) = (fun z' => pf.weight x z' w) :=
    funext fun z' => weight_splice pf w x z z'
  have key : ∀ (t t' : Finset (Paths Ω S)) (f g : Paths Ω S → ℕ) (h1 : t.Nonempty) (h2 : t'.Nonempty), t = t' →
      f = g → (t.exists_max_image f h1).choose = (t'.exists_max_image g h2).choose := by
    rintro t t' f g h1 h2 rfl rfl; rfl
  have h' : (pf.provedAt (splice w x z) w).Nonempty := by rwa [provedAt_splice]
  unfold Accounting.TS.ProofLayer.ref
  rw [dite_eq_left h', dite_eq_left h, splice_left]
  exact congrArg _ (key _ _ _ _ h' h (provedAt_splice pf w x z) hfun)

/-- **At one node**: the moved continuations of `x` from round `w`, squared, are at most its forked pairs. -/
theorem moved_sq_le [Fintype Ω] (w : Fin S) (x : Paths Ω S) :
    (∑ z, if pf.Moved (splice w x z) w then 1 else 0 : ℕ) ^ 2 ≤
      ∑ z1, ∑ z2, if pf.Forked w (splice w x z1) z2 then 1 else 0 := by
  set s := pf.provedAt x w with hs
  have hP : ∀ z, pf.Proved (splice w x z) w ↔ z ∈ s := fun z => by
    simp [hs, Accounting.TS.ProofLayer.provedAt]
  rcases s.eq_empty_or_nonempty with h0 | hne
  · have : ∀ z, ¬ pf.Moved (splice w x z) w := fun z hm => by
      have := (hP z).mp hm.1
      simp [h0] at this
    simp [this]
  obtain ⟨-, hmax⟩ := (s.exists_max_image (fun z => pf.weight x z w) hne).choose_spec
  set zs := (s.exists_max_image (fun z => pf.weight x z w) hne).choose with hzs
  have href : pf.ref x w = splice w x zs := by
    unfold Accounting.TS.ProofLayer.ref; rw [dite_eq_left hne]
  have hcomm : ∀ z i, pf.committed (splice w x z) w i = pf.opened (splice w x zs) w i := fun z i => by
    simp only [Accounting.TS.ProofLayer.committed, ref_splice pf w x z hne, href]
  set W := pf.weight x zs w with hW
  have hA : (∑ z, if pf.Moved (splice w x z) w then 1 else 0 : ℕ) + W = s.card := by
    rw [← Finset.card_filter, hW, Accounting.TS.ProofLayer.weight, ← hs]
    rw [← Finset.card_filter_add_card_filter_not (s := s) (fun z' => ∀ i, pf.opened (splice w x z') w i =
      pf.opened (splice w x zs) w i), add_comm]
    congr 2
    ext z
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Accounting.TS.ProofLayer.Moved, hcomm, ← hP,
      not_forall]
  have hF : (∑ z1, ∑ z2, if pf.Forked w (splice w x z1) z2 then 1 else 0 : ℕ) + ∑ z1 ∈ s, pf.weight x z1 w =
      s.card * s.card := by
    have hrow : ∀ z1, (∑ z2, if pf.Forked w (splice w x z1) z2 then 1 else 0 : ℕ) =
        if z1 ∈ s then s.card - pf.weight x z1 w else 0 := fun z1 => by
      rw [← Finset.card_filter]
      split_ifs with h1
      · rw [Accounting.TS.ProofLayer.weight, ← hs, ← Finset.card_filter_add_card_filter_not (s := s)
          (fun z' => ∀ i, pf.opened (splice w x z') w i = pf.opened (splice w x z1) w i), add_tsub_cancel_left]
        congr 1
        ext z2
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Accounting.TS.ProofLayer.Forked, splice_left, hP,
          not_forall]
        constructor
        · rintro ⟨-, h2, i, hi⟩; exact ⟨h2, i, Ne.symm hi⟩
        · rintro ⟨h2, i, hi⟩; exact ⟨h1, h2, i, Ne.symm hi⟩
      · rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
        intro z2 _ hf
        exact h1 ((hP z1).mp hf.1)
    have hle : ∀ z1, pf.weight x z1 w ≤ s.card := fun z1 => by
      rw [Accounting.TS.ProofLayer.weight, ← hs]; exact Finset.card_filter_le _ _
    simp only [hrow]
    rw [← Finset.sum_filter, Finset.filter_mem_eq_inter, Finset.univ_inter, ← Finset.sum_add_distrib,
      Finset.sum_congr rfl fun z1 _ => Nat.sub_add_cancel (hle z1)]
    simp
  have hsum : ∑ z1 ∈ s, pf.weight x z1 w ≤ s.card * W := by
    simpa using Finset.sum_le_card_nsmul s (fun z1 => pf.weight x z1 w) W fun z1 h1 => hmax z1 h1
  generalize (∑ z, if pf.Moved (splice w x z) w then 1 else 0 : ℕ) = A at hA ⊢
  generalize (∑ z1, ∑ z2, if pf.Forked w (splice w x z1) z2 then 1 else 0 : ℕ) = F at hF ⊢
  generalize ∑ z1 ∈ s, pf.weight x z1 w = sw at hF hsum
  rw [← hA] at hF hsum
  nlinarith [Nat.zero_le (A * W)]

/-- Binding's failure at round `w`, squared, is at most its fork form's. -/
theorem moved_le_forked [Fintype Ω] (w : Fin S) :
    prob (fun x => pf.Moved x w) ^ 2 ≤ prob (fun p : Paths Ω S × Paths Ω S => pf.Forked w p.1 p.2) := by
  set C := Fintype.card (Paths Ω S) with hC
  set M : ℕ := ∑ y, if pf.Moved y w then 1 else 0 with hM
  set Fk : ℕ := ∑ y, ∑ z, if pf.Forked w y z then 1 else 0 with hFk
  have hA : ∑ x, (∑ z, if pf.Moved (splice w x z) w then 1 else 0 : ℕ) = C * M :=
    sum_splice w (fun y => if pf.Moved y w then 1 else 0)
  have hF : ∑ x, (∑ z1, ∑ z2, if pf.Forked w (splice w x z1) z2 then 1 else 0 : ℕ) = C * Fk := by
    calc ∑ x, (∑ z1, ∑ z2, if pf.Forked w (splice w x z1) z2 then 1 else 0 : ℕ)
        = ∑ x, ∑ z2, ∑ z1, (if pf.Forked w (splice w x z1) z2 then 1 else 0 : ℕ) :=
          Finset.sum_congr rfl fun x _ => Finset.sum_comm
      _ = ∑ z2, ∑ x, ∑ z1, (if pf.Forked w (splice w x z1) z2 then 1 else 0 : ℕ) := Finset.sum_comm
      _ = ∑ z2, C * ∑ y, (if pf.Forked w y z2 then 1 else 0 : ℕ) :=
          Finset.sum_congr rfl fun z2 _ => sum_splice w (fun y => if pf.Forked w y z2 then 1 else 0)
      _ = C * Fk := by rw [← Finset.mul_sum, hFk, Finset.sum_comm]
  have hCS : (∑ x, (∑ z, if pf.Moved (splice w x z) w then 1 else 0 : ℕ)) ^ 2 ≤
      C * ∑ x, (∑ z, if pf.Moved (splice w x z) w then 1 else 0 : ℕ) ^ 2 := by
    simpa using sq_sum_le_card_mul_sum_sq (s := Finset.univ)
      (f := fun x => (∑ z, if pf.Moved (splice w x z) w then 1 else 0 : ℕ))
  have h2 : (C * M) ^ 2 ≤ C * (C * Fk) := by
    rw [← hA, ← hF]
    exact hCS.trans (Nat.mul_le_mul_left _ (Finset.sum_le_sum fun x _ => moved_sq_le pf w x))
  have hcM : (Finset.univ.filter fun x => pf.Moved x w).card = M := by rw [Finset.card_filter]
  have hcF : (Finset.univ.filter fun p : Paths Ω S × Paths Ω S => pf.Forked w p.1 p.2).card = Fk := by
    rw [Finset.card_filter, Fintype.sum_prod_type]
  unfold prob
  rw [hcM, hcF, Fintype.card_prod, ← hC]
  rcases Nat.eq_zero_or_pos C with h0 | hpos
  · simp [h0]
  have hMF : M ^ 2 ≤ Fk := by
    have : C ^ 2 * M ^ 2 ≤ C ^ 2 * Fk := by nlinarith [h2]
    exact Nat.le_of_mul_le_mul_left this (by positivity)
  rw [div_pow, Nat.cast_mul, ← sq]
  exact div_le_div_of_nonneg_right (by exact_mod_cast hMF) (by positivity)

end ProofLayer

theorem BindingOfFork : Accounting.Guarantees.TS.BindingOfFork := fun _ _ _ _ pf w =>
  ProofLayer.moved_le_forked pf w

end Accounting.SecurityProofs.TS
