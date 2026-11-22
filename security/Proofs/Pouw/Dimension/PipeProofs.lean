module

public import Definitions.Pouw.Assumptions.Dimension.Pipe
public import Definitions.Pouw.Assumptions.Dimension.Mixing
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.Dimension.Proofs
import Proofs.Pouw.NCP.GameProofs
import Proofs.Pouw.NCP.RelationWitness

/-!
# `A1_cost` in the closed scalar pipe: proofs

* `algBoundPipe`, `algBoundQuad` (`pipe_core`): with no free instruction every checked word is an input, a big output
  or an FP32 output (`pipe_mem_slots`); the words are outside `V₀` and distinct (`pipe_word_nmem`, `pipe_word_injOn`),
  so `|S_B| + |S_R| ≥ D` with `|S_B| ≤ #big`, `cost ≥ 16·#big + c_s·#steps` (`pipe_cost`), and `A1_cost` on `S_R`
  gives `c_s·#steps ≥ 16·(|S_R| − γ·D)`.
* `a1CostOfSplit`, `a1CostWitness`.
* `minRankGeFacts`: MinRank is monotone in `ρ`, and MinRank ≥ 2 gives `MinRank2`.
-/

@[expose] public section

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.NCP.RelationProofs Pouw.NCP.GameProofs Pouw.NCP.RelationWitness
  Pouw.Dimension.Assumptions

variable {G : Type}

theorem pipe_mem_slots (P : List (Instr G)) :
    ∀ L : List (G → ℚ), (∀ c w, Instr.free c w ∉ P) → ∀ f ∈ slots P L, f ∈ L ∨ f ∈ bigOuts P ∨ f ∈ prodOuts P := by
  induction P with
  | nil => intro L _ f hf; exact Or.inl hf
  | cons ins P ih =>
    intro L hP f hf
    rw [slots_cons] at hf
    have hP' : ∀ c w, Instr.free c w ∉ P := fun c w h => hP c w (List.mem_cons_of_mem _ h)
    cases ins with
    | free c w => exact (hP c w (by simp)).elim
    | priced p g =>
      rcases ih _ hP' f hf with h | h | h
      · rcases List.mem_append.1 h with h | h
        · exact Or.inl h
        · have hg : f = g := by simpa [Instr.out] using h
          subst hg
          by_cases hp : 16 ≤ p
          · simp [bigOuts, hp]
          · simp [prodOuts, hp]
      · by_cases hp : 16 ≤ p
        · simp [bigOuts, hp, h]
        · simp [bigOuts, hp, h]
      · by_cases hp : 16 ≤ p
        · simp [prodOuts, hp, h]
        · simp [prodOuts, hp, h]

theorem pipe_cost {P : List (Instr G)} {cs : ℕ} (h : ∀ p f, Instr.priced p f ∈ P → 16 ≤ p ∨ cs ≤ p) :
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
      · simp only [bigOuts, prodOuts, h16, ↓reduceIte, List.length_cons, Instr.price, Nat.mul_succ]
        linarith
      · have hcs : cs ≤ p := hp.resolve_left h16
        simp only [bigOuts, prodOuts, h16, ↓reduceIte, List.length_cons, Instr.price, Nat.mul_succ]
        linarith

theorem pipe_word_nmem {k m n N : ℕ} {R : Mat} (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    {U : Finset ℕ} (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N) {x : ℕ × ℕ}
    (hx : x ∈ U ×ˢ range (nonFinal k m n)) : word k m n N R A B x.1 x.2 ∉ V0 N m n k := by
  classical
  intro h
  have := indep_sub A B hind hU (fun y => if y = ⟨x, hx⟩ then 1 else 0) (by simpa [ite_smul] using h) ⟨x, hx⟩
  simp at this

theorem pipe_word_injOn {k m n N : ℕ} {R : Mat} (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    {U : Finset ℕ} (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N) :
    Set.InjOn (fun x : ℕ × ℕ => word k m n N R A B x.1 x.2) ↑(U ×ˢ range (nonFinal k m n)) := by
  classical
  intro x hx y hy hxy
  by_contra hne
  have hne' : (⟨x, hx⟩ : ↥(U ×ˢ range (nonFinal k m n))) ≠ ⟨y, hy⟩ := fun h => hne (congrArg Subtype.val h)
  simp only at hxy
  have := indep_sub A B hind hU (fun z => (if z = ⟨x, hx⟩ then 1 else 0) - (if z = ⟨y, hy⟩ then 1 else 0))
    (by simp [sub_smul, Finset.sum_sub_distrib, ite_smul, hxy]) ⟨x, hx⟩
  simp [hne'] at this

theorem pipe_card {k m n : ℕ} (U : Finset ℕ) :
    (U ×ˢ range (nonFinal k m n)).card = U.card * (m * n) * (3 * k / 16 - 1) := by
  rw [card_product, card_range, nonFinal]
  ring

/-- The closed-pipe bound for any class `C` of FP32 pipes that `A1_cost` covers. -/
theorem pipe_core (C : ∀ {N m n k : ℕ}, List (Noise N m n k → ℚ) → Prop) {k m n N : ℕ} {R : Mat}
    {A : (Fin k → Fin n → Fin 64) → ℕ → Mat} {B : ℕ → Mat} {U : Finset ℕ} {L : List (Noise N m n k → ℚ)}
    {P : List (Instr (Noise N m n k))} {cs : ℕ} {c γ : ℝ} (hind : CheckpointIndependent k 16 R)
    (hU : U ⊆ range N) (hL : ∀ f ∈ L, f ∈ V0 N m n k) (hfree : ∀ c w, Instr.free c w ∉ P)
    (hpr : ∀ p f, Instr.priced p f ∈ P → 16 ≤ p ∨ cs ≤ p) (hC : C (prodOuts P))
    (hout : Outputs k m n N R A B U P L) (hA1 : A1CostOn C c γ k R) (hc : 16 ≤ c * cs) :
    16 * (1 - γ) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P := by
  classical
  set T := U ×ˢ range (nonFinal k m n) with hT
  set SR := T.filter (fun x => word k m n N R A B x.1 x.2 ∈ prodOuts P) with hSR
  set SB := T.filter (fun x => word k m n N R A B x.1 x.2 ∈ bigOuts P) with hSB
  have hinj := pipe_word_injOn (m := m) A B hind hU
  have hcov : T ⊆ SR ∪ SB := by
    intro x hx
    have hx' := mem_product.1 hx
    rcases pipe_mem_slots P L hfree _ (hout x.1 hx'.1 x.2 (mem_range.1 hx'.2)) with h | h | h
    · exact absurd (hL _ h) (pipe_word_nmem A B hind hU hx)
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

/-- **Theorem D with a closed scalar pipe.** -/
theorem algBoundPipe : AlgBoundPipe := by
  intro k m n N R A B U L P cs c γ hind hMR hU hL hP hout hA1 hc
  exact pipe_core IsPipe hind hU hL hP.1 hP.2.1 hP.2.2 hout (hA1 hind hMR) hc

/-- **Theorem D with a quadratic closed pipe.** -/
theorem algBoundQuad : AlgBoundQuad := by
  intro k m n N R A B U L P cs c γ hind hMR hU hL hP hout hA1 hc
  exact pipe_core IsQuadPipe hind hU hL hP.1.1 hP.1.2.1 hP.2 hout (hA1 hind hMR) hc

/-- **`A1_cost` splits.** -/
theorem a1CostOfSplit : A1CostOfSplit := by
  intro c γ k R hq hn hind hMR m n N A B U hU Rs hRs S hS hSw
  by_cases h : IsQuadPipe Rs
  · exact hq hind hMR m n N A B U hU Rs h S hS hSw
  · exact hn hind hMR m n N A B U hU Rs ⟨hRs, h⟩ S hS hSw

/-- One step per output word: `|S| ≤ #steps` and `|S| ≤ D`, so `2·|S| − D ≤ #steps`. -/
theorem pipe_a1_half (C : ∀ {N m n k : ℕ}, List (Noise N m n k → ℚ) → Prop) {k : ℕ} {R : Mat}
    (hind : CheckpointIndependent k 16 R) : A1CostOn C 2 (1 / 2) k R := by
  classical
  intro m n N A B U hU Rs _ S hS hSw
  have hinj := (pipe_word_injOn (m := m) A B hind hU).mono (coe_subset.2 hS)
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

/-- **The `A1_cost` family's witness.** -/
theorem a1CostWitness : A1CostWitness := fun _ _ =>
  ⟨fun hind _ => pipe_a1_half IsPipe hind, fun hind _ => pipe_a1_half IsQuadPipe hind,
    fun hind _ => pipe_a1_half (fun Rs => IsPipe Rs ∧ ¬ IsQuadPipe Rs) hind⟩

/-! ## MinRank -/

/-- The row of column `b` on the shift's diagonal `b + 24 mod k`. -/
def shRow24 (k b : ℕ) : ℕ := if b + 24 < k then b + 24 else b + 24 - k

theorem minRankGe_mono {k : ℕ} {R : Mat} {ρ ρ' : ℕ} (h : ρ' ≤ ρ) (hR : MinRankGe k R ρ) : MinRankGe k R ρ' := by
  intro q a ha b hb hne
  obtain ⟨r, s, hr, hs, hlt, hdiag, htri⟩ := hR q a ha b hb hne
  exact ⟨r ∘ Fin.castLE h, s ∘ Fin.castLE h, hr.comp (Fin.castLE_injective h), hs.comp (Fin.castLE_injective h),
    fun i => hlt _, fun i => hdiag _, fun i l hil => htri _ _ hil⟩

theorem minRank2_of_ge {k : ℕ} {R : Mat} (h : MinRankGe k R 2) : MinRank2 k R := by
  intro q x y hxy
  by_contra hne
  simp only [not_or, not_forall] at hne
  obtain ⟨⟨a0, ha0, hx0⟩, ⟨b0, hb0, hy0⟩⟩ := hne
  obtain ⟨r, s, -, -, hlt, hdiag, htri⟩ :=
    h q a0 ha0 b0 hb0 (by rw [hxy a0 ha0 b0 hb0]; exact mul_ne_zero hx0 hy0)
  have e00 := hdiag 0
  have e11 := hdiag 1
  have e01 := htri 0 1 (by decide)
  rw [hxy _ (hlt 0).1 _ (hlt 0).2] at e00
  rw [hxy _ (hlt 1).1 _ (hlt 1).2] at e11
  rw [hxy _ (hlt 0).1 _ (hlt 1).2] at e01
  exact mul_ne_zero (left_ne_zero_of_mul e00) (right_ne_zero_of_mul e11) e01

/-- **MinRank is monotone in `ρ`, and MinRank ≥ 2 gives `MinRank2`.** -/
theorem minRankGeFacts : MinRankGeFacts :=
  ⟨fun _ _ _ _ h hR => minRankGe_mono h hR, fun _ _ h => minRank2_of_ge h⟩

/-! ## Witnesses -/

/-- The partial sums `Σ_{l ≤ i} x l · y l`. -/
def psums (x y : ℕ → G → ℚ) (i : ℕ) : G → ℚ := ∑ l ∈ range (i + 1), x l * y l

end Pouw.Dimension
