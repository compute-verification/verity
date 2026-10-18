import Models.Accounting.Guarantees.TS
import Proofs.Pouw.RelativeSecurity

/-!
# (I8) for a sampled window

`Accounting.SecurityProofs.TS.ProofsSoundOfDrawn : Accounting.Guarantees.TS.ProofsSoundOfDrawn`: the event `E` of
`ProofsSound` is "in some round, some drawn unit's binding fails or its proof is unsound", which the union bound puts at
`S·(δ_bind + s·δ_ZK)`. Off it, on an accepted path, every drawn unit's proof is accepted (`Of`), so it opens the values
committed before the draws (not `Moved`) and they satisfy its relation (not `Unsound`), so they are right (`Pins`): the
ideal game accepts (`Of`) and the fixed words follow (`RightFixedDrawn`).

`DrawnBindingOfFork`: `BindingOfFork`'s rewinding step (`Proofs.Accounting.TS.Fork`), here for any continuation map
`sp` with `sp (sp x z) z' = sp x z'` and `sp (sp x z) (sp z x) = x` (`Plurality.moved_le_forked`), at `spliceDraw w`,
one unit at a time; Cauchy–Schwarz over the units gives the `|J|²`.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- A continuation's continuation from the same draw is a continuation of the first path. -/
theorem spliceDraw_left {A B : Type} : ∀ {n : ℕ} (w : ℕ) (x z z' : Paths (A × B) n),
    spliceDraw w (spliceDraw w x z) z' = spliceDraw w x z'
  | 0, _, _, _, _ => rfl
  | _ + 1, 0, _, _, _ => rfl
  | _ + 1, w + 1, x, z, z' => by simp only [spliceDraw, spliceDraw_left w x.2 z.2 z'.2]

/-- Swapping two paths' coins from round `w`'s draws on, twice, gives the first back. -/
theorem spliceDraw_swap {A B : Type} : ∀ {n : ℕ} (w : ℕ) (x z : Paths (A × B) n),
    spliceDraw w (spliceDraw w x z) (spliceDraw w z x) = x
  | 0, _, _, _ => rfl
  | _ + 1, 0, _, _ => rfl
  | _ + 1, w + 1, x, z => by simp only [spliceDraw, spliceDraw_swap w x.2 z.2]

namespace Plurality

variable {X V : Type} [Fintype X] {sp : X → X → X} {acc : X → Prop} {op : X → V}

/-- Summing over every path and every continuation of it is summing over paths, each `|X|` times. -/
theorem sum_sp (hs : ∀ x z, sp (sp x z) (sp z x) = x) (q : X → ℕ) :
    ∑ x, ∑ z, q (sp x z) = Fintype.card X * ∑ y, q y := by
  let Φ : X × X → X × X := fun p => (sp p.1 p.2, sp p.2 p.1)
  have hΦ : Function.Involutive Φ := fun p => by simp only [Φ, hs]
  have h := Equiv.sum_comp (hΦ.toPerm Φ) (fun p => q p.1)
  simp only [Function.Involutive.coe_toPerm, Φ] at h
  rw [← Fintype.sum_prod_type' (fun x z => q (sp x z)), h, Fintype.sum_prod_type (fun p => q p.1),
    Finset.mul_sum]
  simp [Finset.sum_const, Finset.card_univ, mul_comm]

open Classical

theorem accAt_sp (hl : ∀ x z z', sp (sp x z) z' = sp x z') (x z : X) :
    accAt sp acc (sp x z) = accAt sp acc x := by
  ext z'; simp [accAt, hl]

theorem agree_sp (hl : ∀ x z z', sp (sp x z) z' = sp x z') (x z z' : X) :
    agree sp acc op (sp x z) z' = agree sp acc op x z' := by
  simp only [agree, accAt_sp hl, hl]

/-- Every continuation of `x` has `x`'s plurality continuation. -/
theorem plurality_sp (hl : ∀ x z z', sp (sp x z) z' = sp x z') (x z : X) (h : (accAt sp acc x).Nonempty) :
    plurality sp acc op (sp x z) = plurality sp acc op x := by
  have hfun : agree sp acc op (sp x z) = agree sp acc op x := funext fun z' => agree_sp hl x z z'
  have key : ∀ (t t' : Finset X) (f g : X → ℕ) (h1 : t.Nonempty) (h2 : t'.Nonempty), t = t' →
      f = g → (t.exists_max_image f h1).choose = (t'.exists_max_image g h2).choose := by
    rintro t t' f g h1 h2 rfl rfl; rfl
  have h' : (accAt sp acc (sp x z)).Nonempty := by rwa [accAt_sp hl]
  unfold plurality
  rw [dite_eq_left h', dite_eq_left h, hl]
  exact congrArg _ (key _ _ _ _ h' h (accAt_sp hl x z) hfun)

/-- Binding's failure at `x`: `acc` holds and `op` isn't the plurality continuation's. -/
def Mov (sp : X → X → X) (acc : X → Prop) (op : X → V) (x : X) : Prop := acc x ∧ op x ≠ op (plurality sp acc op x)

/-- Two runs off one node, both with `acc`, with different values of `op`. -/
def Frk (sp : X → X → X) (acc : X → Prop) (op : X → V) (x z : X) : Prop := acc x ∧ acc (sp x z) ∧ op x ≠ op (sp x z)

/-- **At one node**: the moved continuations of `x`, squared, are at most its forked pairs. With `n` continuations
accepted and `n_v` of them at `v`, the plurality has `n_*`, the moved are `n − n_*`, and the forked pairs are
`n² − Σ_v n_v² ≥ n² − n·n_* ≥ (n − n_*)²`. -/
theorem moved_sq_le (hl : ∀ x z z', sp (sp x z) z' = sp x z') (x : X) :
    (∑ z, if Mov sp acc op (sp x z) then 1 else 0 : ℕ) ^ 2 ≤
      ∑ z1, ∑ z2, if Frk sp acc op (sp x z1) z2 then 1 else 0 := by
  set s := accAt sp acc x with hs
  have hP : ∀ z, acc (sp x z) ↔ z ∈ s := fun z => by simp [hs, accAt]
  rcases s.eq_empty_or_nonempty with h0 | hne
  · have : ∀ z, ¬ acc (sp x z) := fun z hm => by
      have := (hP z).mp hm
      simp [h0] at this
    simp [this, Mov, Frk]
  obtain ⟨-, hmax⟩ := (s.exists_max_image (agree sp acc op x) hne).choose_spec
  set zs := (s.exists_max_image (agree sp acc op x) hne).choose with hzs
  have href : plurality sp acc op x = sp x zs := by
    unfold plurality; rw [dite_eq_left hne]
  have hcomm : ∀ z, op (plurality sp acc op (sp x z)) = op (sp x zs) := fun z => by
    rw [plurality_sp hl x z hne, href]
  set W := agree sp acc op x zs with hW
  have hA : (∑ z, if Mov sp acc op (sp x z) then 1 else 0 : ℕ) + W = s.card := by
    rw [← Finset.card_filter, hW, agree, ← hs]
    rw [← Finset.card_filter_add_card_filter_not (s := s) (fun z' => op (sp x z') = op (sp x zs)), add_comm]
    congr 2
    ext z
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Mov, hcomm, ← hP]
  have hF : (∑ z1, ∑ z2, if Frk sp acc op (sp x z1) z2 then 1 else 0 : ℕ) + ∑ z1 ∈ s, agree sp acc op x z1 =
      s.card * s.card := by
    have hrow : ∀ z1, (∑ z2, if Frk sp acc op (sp x z1) z2 then 1 else 0 : ℕ) =
        if z1 ∈ s then s.card - agree sp acc op x z1 else 0 := fun z1 => by
      rw [← Finset.card_filter]
      split_ifs with h1
      · rw [agree, ← hs, ← Finset.card_filter_add_card_filter_not (s := s)
          (fun z' => op (sp x z') = op (sp x z1)), add_tsub_cancel_left]
        congr 1
        ext z2
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Frk, hl, hP]
        constructor
        · rintro ⟨-, h2, hi⟩; exact ⟨h2, Ne.symm hi⟩
        · rintro ⟨h2, hi⟩; exact ⟨h1, h2, Ne.symm hi⟩
      · rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
        intro z2 _ hf
        exact h1 ((hP z1).mp hf.1)
    have hle : ∀ z1, agree sp acc op x z1 ≤ s.card := fun z1 => by
      rw [agree, ← hs]; exact Finset.card_filter_le _ _
    simp only [hrow]
    rw [← Finset.sum_filter, Finset.filter_mem_eq_inter, Finset.univ_inter, ← Finset.sum_add_distrib,
      Finset.sum_congr rfl fun z1 _ => Nat.sub_add_cancel (hle z1)]
    simp
  have hsum : ∑ z1 ∈ s, agree sp acc op x z1 ≤ s.card * W := by
    simpa using Finset.sum_le_card_nsmul s (fun z1 => agree sp acc op x z1) W fun z1 h1 => hmax z1 h1
  generalize (∑ z, if Mov sp acc op (sp x z) then 1 else 0 : ℕ) = M at hA ⊢
  generalize (∑ z1, ∑ z2, if Frk sp acc op (sp x z1) z2 then 1 else 0 : ℕ) = F at hF ⊢
  generalize ∑ z1 ∈ s, agree sp acc op x z1 = sw at hF hsum
  rw [← hA] at hF hsum
  nlinarith [Nat.zero_le (M * W)]

/-- Binding's failure, squared, is at most its fork form's. -/
theorem moved_le_forked (hl : ∀ x z z', sp (sp x z) z' = sp x z') (hs : ∀ x z, sp (sp x z) (sp z x) = x) :
    prob (fun x => Mov sp acc op x) ^ 2 ≤ prob (fun p : X × X => Frk sp acc op p.1 p.2) := by
  set C := Fintype.card X with hC
  set M : ℕ := ∑ y, if Mov sp acc op y then 1 else 0 with hM
  set Fk : ℕ := ∑ y, ∑ z, if Frk sp acc op y z then 1 else 0 with hFk
  have hA : ∑ x, (∑ z, if Mov sp acc op (sp x z) then 1 else 0 : ℕ) = C * M :=
    sum_sp hs (fun y => if Mov sp acc op y then 1 else 0)
  have hF : ∑ x, (∑ z1, ∑ z2, if Frk sp acc op (sp x z1) z2 then 1 else 0 : ℕ) = C * Fk := by
    calc ∑ x, (∑ z1, ∑ z2, if Frk sp acc op (sp x z1) z2 then 1 else 0 : ℕ)
        = ∑ x, ∑ z2, ∑ z1, (if Frk sp acc op (sp x z1) z2 then 1 else 0 : ℕ) :=
          Finset.sum_congr rfl fun x _ => Finset.sum_comm
      _ = ∑ z2, ∑ x, ∑ z1, (if Frk sp acc op (sp x z1) z2 then 1 else 0 : ℕ) := Finset.sum_comm
      _ = ∑ z2, C * ∑ y, (if Frk sp acc op y z2 then 1 else 0 : ℕ) :=
          Finset.sum_congr rfl fun z2 _ => sum_sp hs (fun y => if Frk sp acc op y z2 then 1 else 0)
      _ = C * Fk := by rw [← Finset.mul_sum, hFk, Finset.sum_comm]
  have hCS : (∑ x, (∑ z, if Mov sp acc op (sp x z) then 1 else 0 : ℕ)) ^ 2 ≤
      C * ∑ x, (∑ z, if Mov sp acc op (sp x z) then 1 else 0 : ℕ) ^ 2 := by
    simpa using sq_sum_le_card_mul_sum_sq (s := Finset.univ)
      (f := fun x => (∑ z, if Mov sp acc op (sp x z) then 1 else 0 : ℕ))
  have h2 : (C * M) ^ 2 ≤ C * (C * Fk) := by
    rw [← hA, ← hF]
    exact hCS.trans (Nat.mul_le_mul_left _ (Finset.sum_le_sum fun x _ => moved_sq_le hl x))
  have hcM : (Finset.univ.filter fun x => Mov sp acc op x).card = M := by rw [Finset.card_filter]
  have hcF : (Finset.univ.filter fun p : X × X => Frk sp acc op p.1 p.2).card = Fk := by
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

end Plurality

namespace DrawnLayer

variable {A B : Type} [Fintype A] [Fintype B] {S s : ℕ} {J V : Type} (pf : Accounting.TS.DrawnLayer A B S s J V)

/-- Binding's failure for one unit at round `w`, squared, is at most its fork form's at the draw. -/
theorem moved_le_forked (w : Fin S) (j : J) :
    prob (fun x => pf.Moved x w j) ^ 2 ≤ prob (fun p : Paths (A × B) S × Paths (A × B) S => pf.Forked w j p.1 p.2) :=
  Plurality.moved_le_forked (sp := spliceDraw w) (acc := fun y => pf.accepted y w j)
    (op := fun y => pf.opened y w j) (spliceDraw_left w) (spliceDraw_swap w)

/-- Off binding failures and unsound proofs of drawn units, an accepted path's drawn units are right at the values
committed before the draws. -/
theorem right_committed (x : Paths (A × B) S) (hacc : ∀ w i, pf.accepted x w (pf.drawnAt x w i))
    (hM : ∀ w i, ¬ pf.Moved x w (pf.drawnAt x w i)) (hU : ∀ w i, ¬ pf.Unsound x w (pf.drawnAt x w i))
    (hPin : pf.Pins) (w : Fin S) (i : Fin s) :
    pf.right w (pf.drawnAt x w i) (pf.committed x w (pf.drawnAt x w i)) := by
  have heq : pf.opened x w (pf.drawnAt x w i) = pf.committed x w (pf.drawnAt x w i) := by
    by_contra hne
    exact hM w i ⟨hacc w i, hne⟩
  apply hPin
  rw [← heq]
  by_contra hr
  exact hU w i ⟨hacc w i, hr⟩

end DrawnLayer

/-- (I8) for a sampled window, from its drawn units' binding at `δ_bind`, each draw's proof soundness at `δ_ZK`, each
unit's pinning and E1(c) with the drawn units right. -/
theorem ProofsSoundOfDrawn : Accounting.Guarantees.TS.ProofsSoundOfDrawn := by
  intro K T L U J V _ _ _ _ A B _ _ S s P pf Nw δbind δzk hOf hB hZ hPin hRF
  refine ⟨fun x => ∃ w ∈ (Finset.univ : Finset (Fin S)), (∃ i, pf.Moved x w (pf.drawnAt x w i)) ∨
    ∃ i ∈ (Finset.univ : Finset (Fin s)), pf.Unsound x w (pf.drawnAt x w i), ?_, ?_⟩
  · have hw : ∀ w, prob (fun x => (∃ i, pf.Moved x w (pf.drawnAt x w i)) ∨
        ∃ i ∈ (Finset.univ : Finset (Fin s)), pf.Unsound x w (pf.drawnAt x w i)) ≤ δbind + s * δzk := fun w => by
      refine (prob_or_le _ _).trans (add_le_add (hB w) ((prob_exists_le _ _).trans ?_))
      calc ∑ i : Fin s, prob (fun x => pf.Unsound x w (pf.drawnAt x w i)) ≤ ∑ _i : Fin s, δzk :=
            Finset.sum_le_sum fun i _ => hZ w i
        _ = s * δzk := by simp
    calc prob _ ≤ ∑ w : Fin S, prob (fun x => (∃ i, pf.Moved x w (pf.drawnAt x w i)) ∨
          ∃ i ∈ (Finset.univ : Finset (Fin s)), pf.Unsound x w (pf.drawnAt x w i)) := prob_exists_le _ _
      _ ≤ ∑ _w : Fin S, (δbind + s * δzk) := Finset.sum_le_sum fun w _ => hw w
      _ = S * (δbind + s * δzk) := by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  · intro x hx hE
    have hright := DrawnLayer.right_committed pf x (hOf.1 x hx)
      (fun w i h => hE ⟨w, Finset.mem_univ _, Or.inl ⟨i, h⟩⟩)
      (fun w i h => hE ⟨w, Finset.mem_univ _, Or.inr ⟨i, Finset.mem_univ _, h⟩⟩) hPin
    exact ⟨hOf.2 x hright, hRF x hx hright⟩

/-- Binding for a round's drawn units: per unit, the fork form at the draw bounds its square
(`DrawnLayer.moved_le_forked`), and Cauchy–Schwarz over the units gives `|J|²`. -/
theorem DrawnBindingOfFork : Accounting.Guarantees.TS.DrawnBindingOfFork := by
  intro J V _ A B _ _ S s pf w
  set F := prob (fun p : Paths (A × B) S × Paths (A × B) S => ∃ j, pf.Forked w j p.1 p.2) with hF
  have hm : ∀ j, prob (fun x => pf.Moved x w j) ^ 2 ≤ F := fun j =>
    (DrawnLayer.moved_le_forked pf w j).trans (prob_mono fun p h => ⟨j, h⟩)
  have h1 : prob (fun x => ∃ i, pf.Moved x w (pf.drawnAt x w i)) ≤ ∑ j, prob (fun x => pf.Moved x w j) :=
    (prob_mono fun x ⟨_, h⟩ => ⟨_, Finset.mem_univ _, h⟩).trans (prob_exists_le _ _)
  have h2 : (∑ j, prob (fun x => pf.Moved x w j)) ^ 2 ≤
      Fintype.card J * ∑ j, prob (fun x => pf.Moved x w j) ^ 2 := by
    simpa using sq_sum_le_card_mul_sum_sq (s := (Finset.univ : Finset J))
      (f := fun j => prob (fun x => pf.Moved x w j))
  have h3 : ∑ j : J, prob (fun x => pf.Moved x w j) ^ 2 ≤ Fintype.card J * F := by
    calc ∑ j : J, prob (fun x => pf.Moved x w j) ^ 2 ≤ ∑ _j : J, F := Finset.sum_le_sum fun j _ => hm j
      _ = Fintype.card J * F := by simp
  calc prob (fun x => ∃ i, pf.Moved x w (pf.drawnAt x w i)) ^ 2
      ≤ (∑ j, prob (fun x => pf.Moved x w j)) ^ 2 := pow_le_pow_left₀ (prob_nonneg _) h1 2
    _ ≤ Fintype.card J * ∑ j, prob (fun x => pf.Moved x w j) ^ 2 := h2
    _ ≤ Fintype.card J * (Fintype.card J * F) := mul_le_mul_of_nonneg_left h3 (by positivity)
    _ = (Fintype.card J : ℝ) ^ 2 * F := by ring

end Accounting.SecurityProofs.TS
