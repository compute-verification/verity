import Definitions.Pouw.Assumptions.Dimension.Mixing
import Proofs.Pouw.Dimension.Proofs

/-!
# The mixing lemma: proofs

* `span1_slots_le`: the slots lie in the span of the inputs, `B`'s outputs and `R`'s outputs.
* `mixingLemma`: `cost ≥ 16·|B| + c_s·|R|`, `|B| ≥ D − dim W`, `|R| ≥ L(W)`.
* `algBound4090`: with A1, `cost ≥ 16·(1 − γ₁)·D`.
* `alg4090Witness`: the new hypotheses are satisfiable.
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.NCP.RelationProofs Pouw.Dimension.Assumptions

variable {G : Type}

theorem span1_slots_le (P : List (Instr G)) (L : List (G → ℚ)) :
    span1 (slots P L) ≤
      span1 L ⊔ Submodule.span ℚ {f | f ∈ bigOuts P} ⊔ Submodule.span ℚ {f | f ∈ prodOuts P} := by
  induction P generalizing L with
  | nil => exact le_sup_left.trans le_sup_left
  | cons ins P ih =>
    rw [slots_cons]
    refine (ih _).trans ?_
    rw [span1_append]
    cases ins with
    | free c w =>
      rw [sup_eq_right.2 ((Submodule.span_singleton_le_iff_mem _ _).2 (out_free_mem c w L))]
      exact le_rfl
    | priced p f =>
      show Submodule.span ℚ {f} ⊔ span1 L ⊔ _ ⊔ _ ≤ _
      by_cases hp : 16 ≤ p
      · simp only [bigOuts, prodOuts, if_pos hp]
        refine sup_le (sup_le (sup_le ?_ (le_sup_left.trans le_sup_left)) ?_) le_sup_right
        · exact (Submodule.span_mono (by simp)).trans (le_sup_right.trans le_sup_left)
        · exact (Submodule.span_mono fun g hg => List.mem_cons_of_mem _ hg).trans (le_sup_right.trans le_sup_left)
      · simp only [bigOuts, prodOuts, if_neg hp]
        refine sup_le (sup_le (sup_le ?_ (le_sup_left.trans le_sup_left)) (le_sup_right.trans le_sup_left)) ?_
        · exact (Submodule.span_mono (by simp)).trans le_sup_right
        · exact (Submodule.span_mono fun g hg => List.mem_cons_of_mem _ hg).trans le_sup_right

theorem finrank_span_list_le (l : List (G → ℚ)) :
    Module.finrank ℚ (Submodule.span ℚ {f | f ∈ l}) ≤ l.length := by
  classical
  have h := finrank_span_finset_le_card (R := ℚ) l.toFinset
  rw [Set.finrank] at h
  have hs : ((l.toFinset : Finset (G → ℚ)) : Set (G → ℚ)) = {f | f ∈ l} := by ext; simp
  rw [hs] at h
  exact h.trans (List.toFinset_card_le l)

theorem mem_prodOuts {P : List (Instr G)} {f : G → ℚ} (h : f ∈ prodOuts P) :
    ∃ p, ¬ 16 ≤ p ∧ Instr.priced p f ∈ P := by
  induction P with
  | nil => simp [prodOuts] at h
  | cons ins P ih =>
    cases ins with
    | free c w =>
      obtain ⟨p, hp, hm⟩ := ih h
      exact ⟨p, hp, List.mem_cons_of_mem _ hm⟩
    | priced p' f' =>
      by_cases hp : 16 ≤ p'
      · simp only [prodOuts, if_pos hp] at h
        obtain ⟨p, hp, hm⟩ := ih h
        exact ⟨p, hp, List.mem_cons_of_mem _ hm⟩
      · simp only [prodOuts, if_neg hp, List.mem_cons] at h
        rcases h with rfl | h
        · exact ⟨p', hp, by simp⟩
        · obtain ⟨p, hp, hm⟩ := ih h
          exact ⟨p, hp, List.mem_cons_of_mem _ hm⟩

theorem h4090_cost {N m n k cs : ℕ} {P : List (Instr (Noise N m n k))} (h : H4090 cs P) :
    16 * (bigOuts P).length + cs * (prodOuts P).length ≤ cost P := by
  induction P with
  | nil => simp [bigOuts, prodOuts, cost]
  | cons ins P ih =>
    have ih' := ih fun p f hm => h p f (List.mem_cons_of_mem _ hm)
    rw [cost_cons]
    cases ins with
    | free c w =>
      simp only [bigOuts, prodOuts, Instr.price, zero_add]
      exact ih'
    | priced p f =>
      have hp := h p f (by simp)
      by_cases h16 : 16 ≤ p
      · simp only [bigOuts, prodOuts, if_pos h16, List.length_cons, Instr.price, Nat.mul_succ]
        linarith
      · have hcs : cs ≤ p := (hp.resolve_left h16).1
        simp only [bigOuts, prodOuts, if_neg h16, List.length_cons, Instr.price, Nat.mul_succ]
        linarith

theorem finrank_target {k m n N : ℕ} {R : Mat} (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    {U : Finset ℕ} (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N) :
    Module.finrank ℚ (targetSpan k m n N R A B U) = U.card * (m * n) * (3 * k / 16 - 1) := by
  have hli : LinearIndependent ℚ
      (fun x : ↥(U ×ˢ range (nonFinal k m n)) => word k m n N R A B x.1.1 x.1.2) :=
    Fintype.linearIndependent_iff.2 fun c hc => indep_sub A B hind hU c (hc ▸ Submodule.zero_mem _)
  rw [targetSpan, finrank_span_eq_card hli, Fintype.card_coe, card_product, card_range, nonFinal]
  ring

theorem range_get_prodOuts (P : List (Instr G)) :
    Set.range (fun i => (prodOuts P).get i) = {f | f ∈ prodOuts P} := by
  ext f
  simp only [Set.mem_range, Set.mem_setOf_eq, List.mem_iff_get]

theorem prodFree_get {N m n k cs : ℕ} {P : List (Instr (Noise N m n k))} (hH : H4090 cs P)
    (i : Fin (prodOuts P).length) : ProdFree ((prodOuts P).get i) := by
  obtain ⟨p, hp, hm⟩ := mem_prodOuts (List.get_mem _ _)
  exact ((hH p _ hm).resolve_left hp).2

/-- **The mixing lemma.** -/
theorem mixingLemma : MixingLemma := by
  intro k m n N R A B U L P cs hind hU hL hH hout
  have hD := finrank_target (m := m) (N := N) A B hind hU
  refine ⟨hD, h4090_cost hH, ?_, ?_⟩
  · set T := targetSpan k m n N R A B U
    set Sb := Submodule.span ℚ {f | f ∈ bigOuts P}
    set X := Submodule.span ℚ {f | f ∈ prodOuts P} ⊔ V0 N m n k
    have hT : T ≤ X ⊔ Sb := by
      have h1 : T ≤ span1 (slots P L) := by
        refine Submodule.span_le.2 ?_
        rintro _ ⟨x, rfl⟩
        exact Submodule.subset_span (Set.mem_insert_of_mem _
          (hout x.1.1 (mem_product.1 x.2).1 x.1.2 (mem_range.1 (mem_product.1 x.2).2)))
      refine h1.trans ((span1_slots_le P L).trans ?_)
      exact sup_le (sup_le ((span1_le_V0 hL).trans (le_sup_right.trans le_sup_left)) le_sup_right)
        (le_sup_left.trans le_sup_left)
    have e1 := Submodule.finrank_sup_add_finrank_inf_eq T X
    have e2 := Submodule.finrank_mono (sup_le hT (le_sup_left : X ≤ X ⊔ Sb))
    have e3 := Submodule.finrank_add_le_finrank_add_finrank X Sb
    have e4 : Module.finrank ℚ Sb ≤ (bigOuts P).length := finrank_span_list_le (bigOuts P)
    rw [← hD]
    omega
  · refine Nat.sInf_le ⟨fun i => (prodOuts P).get i, prodFree_get hH, ?_⟩
    rw [range_get_prodOuts]
    exact inf_le_right

/-- **Theorem D in AlgM_4090.** -/
theorem algBound4090 : AlgBound4090 := by
  intro k m n N R A B U L P cs c γ₁ hind hU hL hH hout hA1 hc hγ0 hγ1
  obtain ⟨hD, hcost, hB, -⟩ := mixingLemma k m n N R A B U L P cs hind hU hL hH hout
  set T := targetSpan k m n N R A B U
  set W := T ⊓ (Submodule.span ℚ {f | f ∈ prodOuts P} ⊔ V0 N m n k)
  have hA := hA1 m n N A B U hU W inf_le_left (prodOuts P).length (fun i => (prodOuts P).get i)
    (prodFree_get hH) (by rw [range_get_prodOuts]; exact inf_le_right)
  have hWT : Module.finrank ℚ W ≤ Module.finrank ℚ T := Submodule.finrank_mono inf_le_left
  set D := U.card * (m * n) * (3 * k / 16 - 1)
  have hcostR : (16 : ℝ) * (bigOuts P).length + cs * (prodOuts P).length ≤ cost P := by exact_mod_cast hcost
  have hBR : (D : ℝ) ≤ (bigOuts P).length + Module.finrank ℚ W := by exact_mod_cast hB
  have hWD : (Module.finrank ℚ W : ℝ) ≤ D := by rw [← hD]; exact_mod_cast hWT
  have h1 : (cs : ℝ) * (c * (1 - γ₁) * Module.finrank ℚ W) ≤ cs * (prodOuts P).length :=
    mul_le_mul_of_nonneg_left hA (Nat.cast_nonneg _)
  have h2 : 16 * ((1 - γ₁) * Module.finrank ℚ W) ≤ (c * cs) * ((1 - γ₁) * Module.finrank ℚ W) :=
    mul_le_mul_of_nonneg_right hc (mul_nonneg (by linarith) (Nat.cast_nonneg _))
  have h3 : γ₁ * Module.finrank ℚ W ≤ γ₁ * D := mul_le_mul_of_nonneg_left hWD hγ0
  linarith

/-! ## Witnesses -/

theorem one_mem_V0 {N m n k : ℕ} : (1 : Noise N m n k → ℚ) ∈ V0 N m n k :=
  Submodule.mem_sup_left (Submodule.subset_span (Set.mem_insert _ _))

theorem prodOuts_append (P Q : List (Instr G)) : prodOuts (P ++ Q) = prodOuts P ++ prodOuts Q := by
  induction P with
  | nil => rfl
  | cons ins P ih =>
    cases ins with
    | free c w => simp only [List.cons_append, prodOuts, ih]
    | priced p f =>
      simp only [List.cons_append, prodOuts]
      split_ifs <;> simp [ih]

theorem prodOuts_eq_nil {P : List (Instr G)} (h : ∀ p f, Instr.priced p f ∈ P → 16 ≤ p) : prodOuts P = [] := by
  induction P with
  | nil => rfl
  | cons ins P ih =>
    have ih' := ih fun p f hm => h p f (List.mem_cons_of_mem _ hm)
    cases ins with
    | free c w => simpa [prodOuts] using ih'
    | priced p f => simp [prodOuts, h p f (by simp), ih']

/-- **H4090's witness, with a scalar product.** -/
theorem alg4090Witness : Alg4090Witness := by
  intro k hk h16 m n N A B U _
  obtain ⟨R, hadm, hind⟩ := RelationWitness.independentWitness k hk h16
  have hbig : ∀ p f, Instr.priced p f ∈ honest k m n N R A B U 16 → 16 ≤ p := fun p f hm => by
    obtain ⟨u, _, q, _, heq⟩ := mem_honest.1 hm
    exact (Instr.priced.inj heq).1.le
  refine ⟨R, honest k m n N R A B U 16 ++ [Instr.priced 8 1], hadm, hind, ?_, ?_, ?_, ?_⟩
  · intro p f hm
    rcases List.mem_append.1 hm with hm | hm
    · exact Or.inl (hbig p f hm)
    · simp only [List.mem_singleton, Instr.priced.injEq] at hm
      obtain ⟨rfl, rfl⟩ := hm
      exact Or.inr ⟨le_rfl, 1, one_mem_V0, 1, one_mem_V0, (mul_one 1).symm⟩
  · intro u hu p hp
    exact mem_slots_of_priced (List.mem_append_left _ (mem_honest.2 ⟨u, hu, p, hp, rfl⟩)) _
  · rw [prodOuts_append, prodOuts_eq_nil hbig]
    simp [prodOuts]
  · rw [cost_append, honest_cost]
    simp [cost, Instr.price]

/-- **`A1_W1`'s witness.** -/
theorem a1W1Witness : A1W1Witness := fun k R γ₁ h hind _ => a1Witness k R 2 γ₁ hind (by linarith)

end Pouw.Dimension
