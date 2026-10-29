import Definitions.Pouw.Assumptions.Dimension.RouteUPipe
import Proofs.Pouw.Dimension.PipeProofs
import Proofs.Pouw.Dimension.ForestProofs
import Proofs.Pouw.Dimension.RouteUProofs

/-!
# `A1_cost` in the closed scalar pipe on route U: proofs

* `pipe_coreU` (`pipe_core` on route-U words): the words are outside `V₀` and distinct (`pipe_wordU_nmem`,
  `pipe_wordU_injOn`, from `indep_subU`); the count is `pipe_core`'s, with `A1_cost_U` on the FP32 outputs.
* `algBoundPipeU`, `algBoundQuadU`.
* `a1CostQuadUHolds` (Theorem 1 on route U: `Forest.a1CostQuadNear` at `w = wordU β`, near `word` by `wordUSubV0`),
  and with it `algBoundQuad8U`.
* `a1CostWitnessU`, `pipeWitnessU`.
-/

namespace Pouw.Dimension.RouteUProofs

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.NCP.RelationProofs Pouw.NCP.GameProofs Pouw.NCP.RelationWitness
  Pouw.Dimension Pouw.Dimension.Assumptions

theorem pipe_wordU_nmem (β : ℤ) {k m n N : ℕ} {R : Mat} (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    {U : Finset ℕ} (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N) {x : ℕ × ℕ}
    (hx : x ∈ U ×ˢ range (nonFinal k m n)) : wordU β k m n N R A B x.1 x.2 ∉ V0 N m n k := by
  classical
  intro h
  have := indep_subU β A B hind hU (fun y => if y = ⟨x, hx⟩ then 1 else 0) (by simpa [ite_smul] using h) ⟨x, hx⟩
  simp at this

theorem pipe_wordU_injOn (β : ℤ) {k m n N : ℕ} {R : Mat} (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    {U : Finset ℕ} (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N) :
    Set.InjOn (fun x : ℕ × ℕ => wordU β k m n N R A B x.1 x.2) ↑(U ×ˢ range (nonFinal k m n)) := by
  classical
  intro x hx y hy hxy
  by_contra hne
  have hne' : (⟨x, hx⟩ : ↥(U ×ˢ range (nonFinal k m n))) ≠ ⟨y, hy⟩ := fun h => hne (congrArg Subtype.val h)
  simp only at hxy
  have := indep_subU β A B hind hU (fun z => (if z = ⟨x, hx⟩ then 1 else 0) - (if z = ⟨y, hy⟩ then 1 else 0))
    (by simp [sub_smul, Finset.sum_sub_distrib, ite_smul, hxy]) ⟨x, hx⟩
  simp [hne'] at this

/-- The closed-pipe bound on route U for any class `C` of FP32 pipes that `A1_cost_U` covers. -/
theorem pipe_coreU (β : ℤ) (C : ∀ {N m n k : ℕ}, List (Noise N m n k → ℚ) → Prop) {k m n N : ℕ} {R : Mat}
    {A : (Fin k → Fin n → Fin 64) → ℕ → Mat} {B : ℕ → Mat} {U : Finset ℕ} {L : List (Noise N m n k → ℚ)}
    {P : List (Instr (Noise N m n k))} {cs : ℕ} {c γ : ℝ} (hind : CheckpointIndependent k 16 R)
    (hU : U ⊆ range N) (hL : ∀ f ∈ L, f ∈ V0 N m n k) (hfree : ∀ c w, Instr.free c w ∉ P)
    (hpr : ∀ p f, Instr.priced p f ∈ P → 16 ≤ p ∨ cs ≤ p) (hC : C (prodOuts P))
    (hout : OutputsU β k m n N R A B U P L) (hA1 : A1CostOnU β C c γ k R) (hc : 16 ≤ c * cs) :
    16 * (1 - γ) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P := by
  classical
  set T := U ×ˢ range (nonFinal k m n) with hT
  set SR := T.filter (fun x => wordU β k m n N R A B x.1 x.2 ∈ prodOuts P) with hSR
  set SB := T.filter (fun x => wordU β k m n N R A B x.1 x.2 ∈ bigOuts P) with hSB
  have hinj := pipe_wordU_injOn (m := m) β A B hind hU
  have hcov : T ⊆ SR ∪ SB := by
    intro x hx
    have hx' := mem_product.1 hx
    rcases pipe_mem_slots P L hfree _ (hout x.1 hx'.1 x.2 (mem_range.1 hx'.2)) with h | h | h
    · exact absurd (hL _ h) (pipe_wordU_nmem β A B hind hU hx)
    · exact mem_union_right _ (mem_filter.2 ⟨hx, h⟩)
    · exact mem_union_left _ (mem_filter.2 ⟨hx, h⟩)
  have hBc : SB.card ≤ (bigOuts P).length := by
    rw [← card_image_of_injOn (hinj.mono (coe_subset.2 (filter_subset _ _)))]
    refine (card_le_card ?_).trans (List.toFinset_card_le _)
    intro f hf
    obtain ⟨x, hx, rfl⟩ := mem_image.1 hf
    exact List.mem_toFinset.2 (mem_filter.1 hx).2
  have hTc : T.card ≤ SR.card + SB.card := (card_le_card hcov).trans (card_union_le _ _)
  rw [hT, pipe_card U] at hTc
  have hA := hA1 m n N A B U hU (prodOuts P) hC SR (filter_subset _ _) (fun x hx => (mem_filter.1 hx).2)
  have hcost : (16 : ℝ) * (bigOuts P).length + cs * (prodOuts P).length ≤ cost P := by
    exact_mod_cast pipe_cost hpr
  have hBc' : (SB.card : ℝ) ≤ (bigOuts P).length := by exact_mod_cast hBc
  have hTc' : ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ SR.card + SB.card := by exact_mod_cast hTc
  set D : ℝ := ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ)
  have hR : 16 * ((SR.card : ℝ) - γ * D) ≤ cs * (prodOuts P).length := by
    rcases le_or_gt 0 ((SR.card : ℝ) - γ * D) with h0 | h0
    · have h1 : (cs : ℝ) * (c * SR.card - c * γ * D) ≤ cs * (prodOuts P).length :=
        mul_le_mul_of_nonneg_left hA (Nat.cast_nonneg _)
      have h2 : 16 * ((SR.card : ℝ) - γ * D) ≤ (c * cs) * ((SR.card : ℝ) - γ * D) :=
        mul_le_mul_of_nonneg_right hc h0
      linarith
    · have : (0 : ℝ) ≤ cs * (prodOuts P).length := by positivity
      linarith
  linarith

/-- **Theorem D with a closed scalar pipe on route U.** -/
theorem algBoundPipeU : AlgBoundPipeU := by
  intro β k m n N R A B U L P cs c γ hind hMR hU hL hP hout hA1 hc
  exact pipe_coreU β IsPipe hind hU hL hP.1 hP.2.1 hP.2.2 hout (hA1 hind hMR) hc

/-- **Theorem D with a quadratic closed pipe on route U.** -/
theorem algBoundQuadU : AlgBoundQuadU := by
  intro β k m n N R A B U L P cs c γ hind hMR hU hL hP hout hA1 hc
  exact pipe_coreU β IsQuadPipe hind hU hL hP.1.1 hP.1.2.1 hP.2 hout (hA1 hind hMR) hc

/-- **Theorem 1 on route U**: `A1_cost_quad_U β 2 0` holds for every `β`, `k` and `R`, the instance `w = wordU β` of
`Forest.a1CostQuadNear`. -/
theorem a1CostQuadUHolds : A1CostQuadUHolds := by
  intro β k R hind hmr m n N A B U hU Rs hRs S hS hSw
  have h := Forest.a1CostQuadNear k m n N R A B U (wordU β k m n N R A B) hind hmr hU
    (fun u _ p _ => wordUSubV0 β k m n N R A B u p) Rs hRs S hS hSw
  simp only [mul_zero, zero_mul, sub_zero]
  exact_mod_cast h

theorem algBoundQuad8U : AlgBoundQuad8U := by
  intro β k m n N R A B U L P hind hMR hU hL hP hout
  have h := algBoundQuadU β k m n N R A B U L P 8 2 0 hind hMR hU hL hP hout (a1CostQuadUHolds β k R) (by norm_num)
  rw [sub_zero, mul_one] at h
  exact_mod_cast h

/-- One step per output word: `|S| ≤ #steps` and `|S| ≤ D`, so `2·|S| − D ≤ #steps`. -/
theorem pipe_a1_halfU (β : ℤ) (C : ∀ {N m n k : ℕ}, List (Noise N m n k → ℚ) → Prop) {k : ℕ} {R : Mat}
    (hind : CheckpointIndependent k 16 R) : A1CostOnU β C 2 (1 / 2) k R := by
  classical
  intro m n N A B U hU Rs _ S hS hSw
  have hinj := (pipe_wordU_injOn (m := m) β A B hind hU).mono (coe_subset.2 hS)
  have h1 : S.card ≤ Rs.length := by
    rw [← card_image_of_injOn hinj]
    refine (card_le_card ?_).trans (List.toFinset_card_le _)
    intro f hf
    obtain ⟨x, hx, rfl⟩ := mem_image.1 hf
    exact List.mem_toFinset.2 (hSw x hx)
  have h2 : S.card ≤ U.card * (m * n) * (3 * k / 16 - 1) := (card_le_card hS).trans (pipe_card U).le
  have h1' : (S.card : ℝ) ≤ Rs.length := by exact_mod_cast h1
  have h2' : (S.card : ℝ) ≤ ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) := by exact_mod_cast h2
  linarith

/-- **`A1_cost_U`'s witness.** -/
theorem a1CostWitnessU : A1CostWitnessU := fun β _ _ =>
  ⟨fun hind _ => pipe_a1_halfU β IsPipe hind, fun hind _ => pipe_a1_halfU β IsQuadPipe hind⟩

/-- **The closed pipe's witness on route U.** -/
theorem pipeWitnessU : PipeWitnessU := by
  intro β k hk h16 m n N A B U _
  have hbig : ∀ p f, Instr.priced p f ∈ honestU β k m n N (shift24 k) A B U 16 → 16 ≤ p := fun p f hm => by
    obtain ⟨u, _, q, _, heq⟩ := mem_honestU.1 hm
    exact (Instr.priced.inj heq).1.le
  have hpo : prodOuts (honestU β k m n N (shift24 k) A B U 16 ++ [Instr.priced 8 (1 : Noise N m n k → ℚ)]) = [1] := by
    rw [prodOuts_append, prodOuts_eq_nil hbig]
    simp [prodOuts]
  refine ⟨shift24 k, honestU β k m n N (shift24 k) A B U 16 ++ [Instr.priced 8 1], shiftBy24_admissible hk h16,
    indep_of_fresh (fresh hk h16), minRankGe_shift16 hk h16,
    ⟨⟨?_, ?_, ?_⟩, ?_⟩, ?_, ?_, ?_⟩
  · intro c w hm
    rcases List.mem_append.1 hm with hm | hm
    · obtain ⟨u, _, q, _, heq⟩ := mem_honestU.1 hm
      cases heq
    · simp at hm
  · intro p f hm
    rcases List.mem_append.1 hm with hm | hm
    · exact Or.inl (hbig p f hm)
    · simp only [List.mem_singleton, Instr.priced.injEq] at hm
      exact Or.inr hm.1.ge
  · rw [hpo]
    intro i h
    obtain rfl : i = 0 := by simpa using h
    exact ⟨1, 1, 0, Or.inl one_mem_V0, Or.inl one_mem_V0, Or.inl (Submodule.zero_mem _), by simp⟩
  · rw [hpo]
    intro i h
    obtain rfl : i = 0 := by simpa using h
    exact ⟨1, 1, 0, Or.inl (Submodule.zero_mem _), by simp, Or.inl ⟨one_mem_V0, one_mem_V0⟩⟩
  · intro u hu p hp
    exact mem_slots_of_priced (List.mem_append_left _ (mem_honestU.2 ⟨u, hu, p, hp, rfl⟩)) _
  · rw [hpo]
    rfl
  · rw [cost_append, honestU_cost]
    simp [cost, Instr.price]

end Pouw.Dimension.RouteUProofs
