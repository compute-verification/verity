import Definitions.Pous.Model.FreeBlocks
import Proofs.Pous.TrackA.A3Prime

/-!
# A3′ with free blocks

`PousTrackA.a3_prime` for `seqAcceptsFree`: a round passes when its index is free or its answer is right. Lemma A's
event `E` is unchanged (it counts right answers only), and off `E` a round's passing indices are the at most `g − 1`
right ones and the at most `F` free ones, so the chain rule gives `((g − 1 + F)/B)^k`. The free set needs no advice:
the responder is not told it, and the bound holds for every free set the setup determines.
-/

namespace Pous.FreeBlocks

open Pous PousThm1 PousTrackA
open scoped ENNReal

variable {M : IdealModel}

/-- Every round of a sequential run from round `j` in state `σ` satisfies `P` at its index and answer. -/
def seqAll {ℓ S B : ℕ} (f : M.Qry → M.Ans) (step : ℕ → Bits S → Fin B → Prog M (Bits ℓ × Bits S))
    (P : Fin B → Bits ℓ → Prop) : ℕ → List (Fin B) → Bits S → Prop
  | _, [], _ => True
  | j, i :: is, σ => P i ((step j σ i).run f).1 ∧ seqAll f step P (j + 1) is ((step j σ i).run f).2

theorem seqAll_of_getElem? {ℓ S B : ℕ} (f : M.Qry → M.Ans) (step : ℕ → Bits S → Fin B → Prog M (Bits ℓ × Bits S))
    (P : Fin B → Bits ℓ → Prop) :
    ∀ (j : ℕ) (is : List (Fin B)) (σ : Bits S),
      (∀ (n : ℕ) (hn : n < is.length) (a : Bits ℓ), (seqAnswers f step j is σ)[n]? = some a → P is[n] a) →
      seqAll f step P j is σ
  | _, [], _, _ => trivial
  | j, i :: is, σ, h => by
      refine ⟨h 0 (by simp) _ (by simp [seqAnswers]), seqAll_of_getElem? f step P (j + 1) is _ fun n hn a ha => ?_⟩
      have := h (n + 1) (by simp only [List.length_cons]; omega) a (by simpa [seqAnswers] using ha)
      simpa using this

/-- **Chain rule for a per-round predicate.** `PousThm1.pr_seq_le` with "the answer is right" replaced by `P`. -/
theorem pr_seqAll_le {ℓ S B : ℕ} [NeZero B] (f : M.Qry → M.Ans)
    (step : ℕ → Bits S → Fin B → Prog M (Bits ℓ × Bits S)) (P : Fin B → Bits ℓ → Prop) (c : ℝ≥0∞) :
    ∀ (k j : ℕ) (σ : Bits S),
      (∀ h : List (Fin B), h.length < k →
        pr (fun i : Fin B => P i ((step (j + h.length) (seqState f step j h σ) i).run f).1) ≤ c) →
      pr (fun I : Fin k → Fin B => seqAll f step P j (List.ofFn I) σ) ≤ c ^ k
  | 0, _, _, _ => by rw [pow_zero]; exact pr_le_one _
  | k + 1, j, σ, hyp => by
      rw [← pr_equiv (Fin.consEquiv fun _ : Fin (k + 1) => Fin B :
        Fin B × (Fin k → Fin B) ≃ (Fin (k + 1) → Fin B))]
      simp only [Fin.consEquiv_apply, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, seqAll]
      rw [pow_succ']
      refine pr_and_le (α := Fin B) (β := Fin k → Fin B) (fun i₀ => P i₀ ((step j σ i₀).run f).1)
        (fun i₀ I => seqAll f step P (j + 1) (List.ofFn fun i => I i) ((step j σ i₀).run f).2) c (c ^ k) ?_
        fun i₀ => ?_
      · simpa [seqState] using hyp [] (Nat.succ_pos k)
      · refine pr_seqAll_le f step P c k (j + 1) _ fun h hh => ?_
        have := hyp (i₀ :: h) (by simp only [List.length_cons]; omega)
        simpa [seqState, Nat.add_assoc, Nat.add_comm 1] using this

/-- **A3′ with `F` free blocks.** `Pr[pass] ≤ ((g − 1 + F)/B)^k + |Hist|·Γ + Pr[not G]`. -/
theorem a3_prime_free (sch : Scheme M) (S D Q k g F : ℕ) (Γ : ℝ≥0∞)
    (G : Bits sch.n → M.Ω × sch.Coins → Prop) (hA : LemmaA sch S D Q g Γ G)
    (W : Bits sch.n) (A₁ : Preprocessor sch S) (Fr : FreeSet sch F) (A₂ : SeqResponder sch S D Q) :
    pr (seqAcceptsFree sch k W A₁ Fr A₂) ≤
      (((g - 1 + F : ℕ) : ℝ≥0∞) / sch.B) ^ k + (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ +
        pr (fun a => ¬ G W a) := by
  classical
  let st : List (Fin sch.B) → Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits S :=
    fun h W pp C ω => seqState (M.answer ω) (A₂.step W pp) 0 h (A₁ W pp C ω)
  let pred : List (Fin sch.B) → TimedProdResponder sch S D Q := fun h =>
    ⟨fun W pp σ i => progMap Prod.fst (A₂.step W pp h.length σ i),
      fun W pp σ i => bounded_progMap _ (A₂.bounded _ _ _ _ _)⟩
  let good : List (Fin sch.B) → M.Ω × sch.Coins → Fin sch.B → Prop := fun h a i =>
    ((pred h).prog W (sch.enc W a.1 a.2).1 (st h W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
      (M.answer a.1) = (sch.enc W a.1 a.2).2 i
  let fr : M.Ω × sch.Coins → Finset (Fin sch.B) := fun a =>
    Fr.free W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1
  let E : M.Ω × sch.Coins → Prop := fun a => ¬ G W a ∨
    ∃ j ∈ Finset.range k, ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
      G W a ∧ g ≤ (Finset.univ.filter fun i => good (List.ofFn v) a i).card
  let pass : (M.Ω × sch.Coins) → (Fin k → Fin sch.B) → Prop := fun a I =>
    seqAll (M.answer a.1) (A₂.step W (sch.enc W a.1 a.2).1)
      (fun i o => i ∈ fr a ∨ o = (sch.enc W a.1 a.2).2 i) 0 (List.ofFn I)
      (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1)
  have hre : pr (seqAcceptsFree sch k W A₁ Fr A₂) ≤ pr (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) =>
      pass x.1 x.2) := by
    refine le_trans (pr_mono fun x hx => ?_)
      (pr_equiv (Equiv.prodAssoc _ _ _).symm (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) => pass x.1 x.2)).le
    simp only [seqAcceptsFree] at hx
    show pass (x.1, x.2.1) x.2.2
    refine seqAll_of_getElem? _ _ _ 0 _ _ fun n hn a ha => ?_
    simp only [List.length_ofFn] at hn
    simp only [List.getElem_ofFn]
    by_cases hf : x.2.2 ⟨n, hn⟩ ∈ fr (x.1, x.2.1)
    · exact Or.inl hf
    · refine Or.inr ?_
      have h1 := hx ⟨n, hn⟩ hf
      rw [ha] at h1
      exact Option.some.inj h1
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  refine hre.trans ((pr_prod_le (α := M.Ω × sch.Coins) (β := Fin k → Fin sch.B) pass E
    ((((g - 1 + F : ℕ) : ℝ≥0∞) / sch.B) ^ k) fun a ha => ?_).trans ?_)
  · -- off `E`, a round passes on at most `g − 1` right indices and `F` free ones
    refine pr_seqAll_le _ _ _ _ k 0 _ fun h hh => ?_
    have hG : G W a := by by_contra hG; exact ha (Or.inl hG)
    have hlt : (Finset.univ.filter fun i => good h a i).card ≤ g - 1 := by
      have : ¬ g ≤ (Finset.univ.filter fun i => good h a i).card := by
        intro hg
        refine ha (Or.inr ⟨h.length, Finset.mem_range.2 hh, h.get, Finset.mem_univ _, hG, ?_⟩)
        rwa [List.ofFn_get]
      omega
    have hfr : (Finset.univ.filter fun i => i ∈ fr a).card ≤ F := by
      rw [Finset.filter_mem_eq_inter, Finset.univ_inter]
      exact Fr.card_le _ _ _ _
    have hcard : (Finset.univ.filter fun i => i ∈ fr a ∨ good h a i).card ≤ g - 1 + F := by
      have hsub : (Finset.univ.filter fun i => i ∈ fr a ∨ good h a i) ⊆
          (Finset.univ.filter fun i => i ∈ fr a) ∪ (Finset.univ.filter fun i => good h a i) := by
        intro i hi
        simp only [Finset.mem_filter, Finset.mem_union, Finset.mem_univ, true_and] at hi ⊢
        exact hi
      exact (Finset.card_le_card hsub).trans ((Finset.card_union_le _ _).trans (by omega))
    have hpr : pr (fun i => i ∈ fr a ∨ good h a i) ≤ ((g - 1 + F : ℕ) : ℝ≥0∞) / sch.B := by
      rw [pr_def, Fintype.card_fin]
      exact ENNReal.div_le_div_right (by exact_mod_cast hcard) _
    simpa only [good, pred, st, run_progMap, Nat.zero_add] using hpr
  · -- a union over the histories, Lemma A for each
    have hE : pr E ≤ pr (fun a => ¬ G W a) + (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ := by
      refine (pr_or_le _ _).trans (add_le_add le_rfl ?_)
      calc pr (fun a => ∃ j ∈ Finset.range k, ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
            G W a ∧ g ≤ (Finset.univ.filter fun i => good (List.ofFn v) a i).card)
          ≤ ∑ j ∈ Finset.range k, pr (fun a => ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
            G W a ∧ g ≤ (Finset.univ.filter fun i => good (List.ofFn v) a i).card) := pr_exists_le _ _
        _ ≤ ∑ j ∈ Finset.range k, ∑ _v : Fin j → Fin sch.B, Γ := by
            refine Finset.sum_le_sum fun j _ => (pr_exists_le _ _).trans (Finset.sum_le_sum fun v _ => ?_)
            have := hA W (st (List.ofFn v)) (pred (List.ofFn v))
            convert this using 3
        _ = (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ := by
            rw [Finset.sum_mul]
            refine Finset.sum_congr rfl fun j _ => ?_
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin,
              nsmul_eq_mul, Nat.cast_pow]
    calc pr E + (((g - 1 + F : ℕ) : ℝ≥0∞) / sch.B) ^ k
        ≤ pr (fun a => ¬ G W a) + (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ +
          (((g - 1 + F : ℕ) : ℝ≥0∞) / sch.B) ^ k := add_le_add hE le_rfl
      _ = _ := by ring

/-- `a3_prime_free` in the review's form: if `|Hist|·Γ ≤ 2^-λ'`, then
`Pr[pass] ≤ ((g − 1 + F)/B)^k + 2^-λ' + Pr[not G]`. -/
theorem a3_prime_free_lam (sch : Scheme M) (S D Q k g F lam : ℕ) (Γ : ℝ≥0∞)
    (G : Bits sch.n → M.Ω × sch.Coins → Prop) (hA : LemmaA sch S D Q g Γ G)
    (hΓ : (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ ≤ (2⁻¹ : ℝ≥0∞) ^ lam)
    (W : Bits sch.n) (A₁ : Preprocessor sch S) (Fr : FreeSet sch F) (A₂ : SeqResponder sch S D Q) :
    pr (seqAcceptsFree sch k W A₁ Fr A₂) ≤
      (((g - 1 + F : ℕ) : ℝ≥0∞) / sch.B) ^ k + (2⁻¹ : ℝ≥0∞) ^ lam + pr (fun a => ¬ G W a) :=
  (a3_prime_free sch S D Q k g F Γ G hA W A₁ Fr A₂).trans (by gcongr)

end Pous.FreeBlocks
