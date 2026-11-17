module

public import Proofs.Pouw.TS.Sampled
public import Proofs.ResourceAccounting.Lemmas.TS.Fixed

/-! `UnitFixed` from a fixed word, and the additive lemma. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.TS

open ResourceAccounting.TS ResourceAccounting.Assumptions.TS Pouw.RelativeSecurity

theorem splice_self {Ω : Type} : ∀ {n : ℕ} (k : ℕ) (x : Paths Ω n), splice k x x = x
  | 0, _, _ => rfl
  | _ + 1, 0, _ => rfl
  | _ + 1, k + 1, x => by simp only [splice, splice_self k x.2, Prod.mk.eta]

theorem splice_mono {Ω : Type} : ∀ {n : ℕ} (b c : ℕ) (x z : Paths Ω n), b ≤ c →
    splice b x (splice c x z) = splice c x z
  | 0, _, _, _, _, _ => rfl
  | _ + 1, 0, _, _, _, _ => rfl
  | _ + 1, _ + 1, 0, _, _, h => by omega
  | _ + 1, b + 1, c + 1, x, z, h => by
    simp only [splice, splice_mono b c x.2 z.2 (Nat.le_of_succ_le_succ h)]

theorem spliceAt_mono {C B : Type} {m : ℕ} : ∀ {n : ℕ} (a b w c : ℕ) (x z : Paths (Paths C m × B) n),
    (a < w ∨ a = w ∧ b ≤ c) → spliceAt a b x (spliceAt w c x z) = spliceAt w c x z
  | 0, _, _, _, _, _, _, _ => rfl
  | _ + 1, 0, b, 0, c, x, z, h => by
    have hbc : b ≤ c := h.elim (fun h => absurd h (Nat.lt_irrefl 0)) And.right
    simp only [spliceAt, splice_mono b c x.1.1 z.1.1 hbc]
  | _ + 1, 0, b, _ + 1, _, x, _, _ => by simp only [spliceAt, splice_self b x.1.1, Prod.mk.eta]
  | _ + 1, _ + 1, _, 0, _, _, _, h =>
    h.elim (fun h => absurd h (Nat.not_lt_zero _)) fun h => absurd h.1 (Nat.succ_ne_zero _)
  | _ + 1, a + 1, b, w + 1, c, x, z, h => by
    have h' : a < w ∨ a = w ∧ b ≤ c := h.imp Nat.lt_of_succ_lt_succ fun h => ⟨Nat.succ.inj h.1, h.2⟩
    simp only [spliceAt, spliceAt_mono a b w c x.2 z.2 h']

theorem committed_spliceAt {C B : Type} [Fintype C] [Fintype B] {S m s : ℕ} {J Q Pos V : Type} [Nonempty V]
    (pf : SampledLayer C B S m s J Q Pos V) (x z : Paths (Paths C m × B) S) (ℓ : Pos) (w : Fin S) (c : Fin m)
    (h : (pf.cut ℓ).1 < w ∨ (pf.cut ℓ).1 = w ∧ (pf.cut ℓ).2 ≤ c.castSucc) :
    pf.committed (spliceAt w c x z) ℓ = pf.committed x ℓ := by
  have hm : spliceAt (pf.cut ℓ).1 (pf.cut ℓ).2 x (spliceAt w c x z) = spliceAt w c x z := by
    apply spliceAt_mono
    rcases h with h | ⟨h1, h2⟩
    · exact Or.inl h
    · refine Or.inr ⟨congrArg Fin.val h1, ?_⟩
      exact h2
  unfold SampledLayer.committed
  rw [← hm]
  exact Plurality.plural_sp (sp := pf.cont ℓ) (opens := fun y v => pf.Opens y ℓ v)
    (spliceAt_left (pf.cut ℓ).1 (pf.cut ℓ).2) x _

theorem steers_late {C B : Type} {S m s : ℕ} {J Q Pos V N W : Type} (pf : WordLayer C B S m s J Q Pos V N W)
    {t u : Fin S × J} (h : pf.Steers t u) : Relation.ReflTransGen pf.Late t u := by
  obtain ⟨v, hv, k, -, hs, hb⟩ := h
  refine hv.tail ⟨k, hs, ?_⟩
  unfold WordLayer.Bound at hb
  rcases lt_trichotomy (pf.cut (pf.pos u.1 u.2 k)).1 u.1 with h1 | h1 | h1
  · exact absurd (Or.inl h1) hb
  · refine Or.inr ⟨h1.symm, ?_⟩
    by_contra h2
    exact hb (Or.inr ⟨h1, not_lt.1 h2⟩)
  · exact Or.inl h1

theorem fixedWords_of_good {C B : Type} [Fintype C] [Fintype B] {S m s : ℕ} {J Q Pos V N W : Type} [Fintype J]
    [Nonempty V] [AddCommGroup W] (pf : WordLayer C B S m s J Q Pos V N W) {wt : Fin S → J → ℝ}
    (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (hM : ModesRun pf mode) (x : Paths (Paths C m × B) S)
    (u : Fin S × J) (hF : pf.Fixed wt u) (hG : pf.Good mode x u.1 u.2) : pf.FixedWords wt x u :=
  ⟨hF, (hM x u).1 (hG u Relation.ReflTransGen.refl), fun t ht => (hM x t).1 (hG t (steers_late pf ht))⟩

open Classical in
theorem UnitFixedOfWords : ResourceAccounting.Guarantees.TS.UnitFixedOfWords := by
  intro K T L U _ _ _ _ C B _ _ S m s J Q Pos V N W _ _ _ P pf mode wt folded hwt hfix hM hC _ x hx
  obtain ⟨h1, h2⟩ := hC x hx
  refine ⟨le_trans (div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg S)) h1,
    le_trans (div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg S)) h2⟩
  · refine Finset.sum_le_sum fun w _ => Finset.sum_le_sum fun j _ => ?_
    show (if pf.Good mode x w j then wt w j else 0) ≤ if pf.FixedWords wt x (w, j) then wt w j else 0
    split_ifs with hg hfw <;> first
      | exact le_rfl
      | exact hwt w j
      | exact le_of_eq (Classical.byContradiction fun hw =>
          hfw (fixedWords_of_good pf mode hM x (w, j) (hfix w j hw) hg))
  · refine Finset.sum_le_sum fun w _ => Finset.sum_le_sum fun j _ => ?_
    show (if mode x (w, j) = .whole then 0 else wt w j) ≤ if pf.Folds x (w, j) then wt w j else 0
    by_cases hw : mode x (w, j) = .whole
    · refine (ite_eq_left hw).trans_le ?_
      by_cases hf : pf.Folds x (w, j)
      · exact (hwt w j).trans_eq (ite_eq_left hf).symm
      · exact le_of_eq (ite_eq_right hf).symm
    · exact le_of_eq ((ite_eq_right hw).trans (ite_eq_left ((hM x (w, j)).2 hw)).symm)

theorem AdditiveFixed : ResourceAccounting.Guarantees.TS.AdditiveFixed := by
  intro C B _ _ S m s J Q Pos V N W _ _ pf u hb x hC
  refine ⟨?_, fun z => ?_⟩
  · unfold WordLayer.work
    rw [hC.1]
    exact add_sub_cancel_right _ _
  · funext k
    exact committed_spliceAt pf.toSampledLayer x z (pf.pos u.1 u.2 k.1) u.1 (pf.coin u.1 u.2) (hb k.1 k.2)

end ResourceAccounting.SecurityProofs.TS
