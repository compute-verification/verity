import Models.Pouw.Assumptions.Dimension.Forest
import Proofs.Pouw.Dimension.PipeProofs
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.LinearAlgebra.Matrix.Block
import Mathlib.Tactic.Module

/-!
# Theorem 1 of `a1-cost.md` (the forest argument): proofs

* **The forest count** (`forestBound`). The new V-gates are the values in `Z` outside the span of the earlier ones
  (`news`); they span what the V-gates span. A new FP32 V-gate's operands are earlier non-V values with nonzero
  coefficients (`att_of_new`): one plus a private atom (a pendant edge) or two (an edge). Group the non-V values by
  their line modulo `Z` (`cls`); an edge joins two values of one line (`cls_eq_of_edge`), and two pendant atoms on one
  line differ by an element of `Z`, so by `TwoAtomFree` they are proportional. So the new FP32 V-gates attached to
  one line lie in `(span of its vertices ⊔ one atom) ⊓ Z`, which has dimension at most the number of vertices
  (`class_bound`). Summing over lines, `dim ≤ #new priced V-gates + #non-V values`, and the count follows.
  This replaces the forest and its trees (`a1-cost.md` steps 5–8) by the lines modulo `Z`, which carry the same count.
* **The grid** (`gridTwoAtom`). For `a, b ∈ V₀` the four-point difference kills `a` and `b`, so
  `sd(a·b) = Δ_E a·Δ_F b + Δ_F a·Δ_E b` (`sd_mul`), a matrix of rank ≤ 2 in the probe's `(a, b)`. A combination of
  checked words equal to two atoms modulo `V₀` would give each unit and entry a combination of checkpoint matrices of
  rank ≤ 4; MinRank ≥ 5 gives a lower-triangular 5 × 5 minor with a nonzero diagonal (`no_rank4`), so the combination
  of matrices is `0`, and independence makes the coefficients `0` (`comb_two_atoms`).
* **The pipe** (`a1CostQuadHolds`). A quadratic closed pipe is an FP32 program on Q-parts (`opnd_of_pipeOpnd`), all
  its steps FP32; its outputs' words are independent modulo `V₀`, so `2·|S| ≤ #steps`.
-/

namespace Pouw.Dimension.Forest

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.NCP.RelationProofs Pouw.Dimension Pouw.Dimension.Assumptions


open scoped Classical

noncomputable section

variable {W : Type} [AddCommGroup W] [Module ℚ W]

/-- The V-gates among the first `N` values. -/
def vgs (Z : Submodule ℚ W) (y : ℕ → W) (N : ℕ) : Finset ℕ := (range N).filter (fun i => y i ∈ Z)

/-- The new V-gates: not in the span of the earlier V-gates. -/
def news (Z : Submodule ℚ W) (y : ℕ → W) (N : ℕ) : Finset ℕ :=
  (vgs Z y N).filter (fun i => y i ∉ Submodule.span ℚ (y '' ↑((vgs Z y N).filter (· < i))))

theorem mem_span_news (Z : Submodule ℚ W) (y : ℕ → W) (N : ℕ) :
    ∀ i, i ∈ vgs Z y N → y i ∈ Submodule.span ℚ (y '' ↑(news Z y N)) := by
  intro i
  induction i using Nat.strong_induction_on with
  | _ i ih =>
    intro hi
    by_cases hn : i ∈ news Z y N
    · exact Submodule.subset_span ⟨i, hn, rfl⟩
    · have h : y i ∈ Submodule.span ℚ (y '' ↑((vgs Z y N).filter (· < i))) := by
        by_contra h
        exact hn (mem_filter.2 ⟨hi, h⟩)
      refine (Submodule.span_le.2 ?_) h
      rintro _ ⟨j, hj, rfl⟩
      rw [coe_filter, Set.mem_ofPred_eq] at hj
      exact ih j hj.2 hj.1

/-- The shape of a new FP32 V-gate: its first operand `j` is an earlier non-V value, with a private atom (a pendant
edge) or with a second earlier non-V operand (an edge), every coefficient nonzero. -/
def Att (Z : Submodule ℚ W) (At : Set W) (y : ℕ → W) (i j : ℕ) : Prop :=
  j < i ∧ y j ∉ Z ∧ ((∃ q ∈ At, ∃ β : ℚ, β ≠ 0 ∧ y i = q + β • y j) ∨
    (∃ γ₁ γ₂ : ℚ, ∃ j₂ < i, γ₁ ≠ 0 ∧ γ₂ ≠ 0 ∧ y j₂ ∉ Z ∧ y i = γ₁ • y j + γ₂ • y j₂))

theorem opnd_cases {Z : Submodule ℚ W} {y : ℕ → W} {N i : ℕ} (hi : i < N) {c : W} (h : Opnd y i c) :
    c ∈ Submodule.span ℚ (y '' ↑((vgs Z y N).filter (· < i))) ∨
      ∃ j < i, ∃ β : ℚ, β ≠ 0 ∧ y j ∉ Z ∧ c = β • y j := by
  rcases h with rfl | ⟨j, hj, β, rfl⟩
  · exact Or.inl (Submodule.zero_mem _)
  · by_cases hβ : β = 0
    · left; rw [hβ, zero_smul]; exact Submodule.zero_mem _
    by_cases hz : y j ∈ Z
    · left
      refine Submodule.smul_mem _ _ (Submodule.subset_span ⟨j, ?_, rfl⟩)
      rw [coe_filter, Set.mem_ofPred_eq]
      exact ⟨mem_filter.2 ⟨mem_range.2 (hj.trans hi), hz⟩, hj⟩
    · exact Or.inr ⟨j, hj, β, hβ, hz, rfl⟩

theorem span_early_le (Z : Submodule ℚ W) (y : ℕ → W) (N i : ℕ) :
    Submodule.span ℚ (y '' ↑((vgs Z y N).filter (· < i))) ≤ Z := by
  refine Submodule.span_le.2 ?_
  rintro _ ⟨j, hj, rfl⟩
  rw [coe_filter, Set.mem_ofPred_eq] at hj
  exact (mem_filter.1 hj.1).2

theorem att_of_new {Z : Submodule ℚ W} {At : Set W} (hsm : ∀ q ∈ At, ∀ μ : ℚ, μ • q ∈ At)
    (h2 : TwoAtomFree Z At) {y : ℕ → W} {N i : ℕ} (hi : i ∈ news Z y N) (hs : FStep At y i) :
    ∃ j, Att Z At y i j := by
  obtain ⟨hv, hnew⟩ := mem_filter.1 hi
  obtain ⟨hiN, hZ⟩ := mem_filter.1 hv
  have hiN := mem_range.1 hiN
  have hSpZ := span_early_le Z y N i
  have hcases := fun {c : W} (h : Opnd y i c) => opnd_cases (Z := Z) hiN h
  generalize Submodule.span ℚ (y '' ↑((vgs Z y N).filter (· < i))) = Sp at hnew hSpZ hcases
  rcases hs with ⟨q, hq, c, hc, hy⟩ | ⟨α, b, c, hb, hc, hy⟩
  · rcases hcases hc with hc | ⟨j, hj, β, hβ, hjZ, rfl⟩
    · exfalso
      have hq' : q = y i - c := by rw [hy]; abel
      have hqZ : q + (0 : ℚ) • q ∈ Z := by
        rw [zero_smul, add_zero, hq']
        exact Z.sub_mem hZ (hSpZ hc)
      have h0 := h2 q hq _ (hsm q hq 0) hqZ
      rw [zero_smul, add_zero] at h0
      have hv : y i = c := by rw [hy, h0, zero_add]
      exact hnew (hv ▸ hc)
    · exact ⟨j, hj, hjZ, Or.inl ⟨q, hq, β, hβ, hy⟩⟩
  · have hnot : ∀ {x : W}, x ∈ Sp → y i - x ∈ Sp → False := fun {x} h1 h3 => by
      have e : x + (y i - x) = y i := by abel
      exact hnew (e ▸ Sp.add_mem h1 h3)
    rcases hcases hb with hb | ⟨j₁, hj₁, β₁, hβ₁, hj₁Z, rfl⟩
    · exfalso
      rcases hcases hc with hc | ⟨j₂, hj₂, β₂, hβ₂, hj₂Z, rfl⟩
      · have e : y i - α • b = c := by rw [hy]; abel
        exact hnot (Sp.smul_mem α hb) (e ▸ hc)
      · have e : β₂ • y j₂ = y i - α • b := by rw [hy]; abel
        have : β₂ • y j₂ ∈ Z := e ▸ Z.sub_mem hZ (hSpZ (Sp.smul_mem α hb))
        exact hj₂Z ((Z.smul_mem_iff hβ₂).1 this)
    · rcases hcases hc with hc | ⟨j₂, hj₂, β₂, hβ₂, hj₂Z, rfl⟩
      · exfalso
        by_cases hα : α = 0
        · have e : y i - 0 = c := by rw [hy, hα, zero_smul, zero_add, sub_zero]
          exact hnot (x := 0) Sp.zero_mem (e ▸ hc)
        · have e : (α * β₁) • y j₁ = y i - c := by rw [hy, mul_smul]; abel
          have : (α * β₁) • y j₁ ∈ Z := e ▸ Z.sub_mem hZ (hSpZ hc)
          exact hj₁Z ((Z.smul_mem_iff (mul_ne_zero hα hβ₁)).1 this)
      · by_cases hα : α = 0
        · exfalso
          have e : y i = β₂ • y j₂ := by rw [hy, hα, zero_smul, zero_add]
          exact hj₂Z ((Z.smul_mem_iff hβ₂).1 (e ▸ hZ))
        · exact ⟨j₁, hj₁, hj₁Z, Or.inr ⟨α * β₁, β₂, j₂, hj₂, mul_ne_zero hα hβ₁, hβ₂, hj₂Z,
            by rw [hy, mul_smul]⟩⟩

/-- The class of a value: its line modulo `Z`. -/
def cls (Z : Submodule ℚ W) (y : ℕ → W) (j : ℕ) : Submodule ℚ W := Submodule.span ℚ {y j} ⊔ Z

theorem cls_eq_of_edge {Z : Submodule ℚ W} {y : ℕ → W} {i j j₂ : ℕ} {γ₁ γ₂ : ℚ} (hZ : y i ∈ Z)
    (h₁ : γ₁ ≠ 0) (h₂ : γ₂ ≠ 0) (hy : y i = γ₁ • y j + γ₂ • y j₂) : cls Z y j₂ = cls Z y j := by
  have key : ∀ (a b : ℕ) (x y' : ℚ), x ≠ 0 → y i = y' • y a + x • y b → y b ∈ cls Z y a := by
    intro a b x y' hx h
    rw [show y b = x⁻¹ • (y i - y' • y a) by rw [h, add_sub_cancel_left, inv_smul_smul₀ hx]]
    exact Submodule.smul_mem _ _ (Submodule.sub_mem _ (Submodule.mem_sup_right hZ)
      (Submodule.mem_sup_left (Submodule.smul_mem _ _ (Submodule.mem_span_singleton_self _))))
  apply le_antisymm
  · exact sup_le (Submodule.span_singleton_le_iff_mem _ _ |>.2 (key j j₂ γ₂ γ₁ h₂ hy)) le_sup_right
  · exact sup_le (Submodule.span_singleton_le_iff_mem _ _ |>.2
      (key j₂ j γ₁ γ₂ h₁ (by rw [hy, add_comm]))) le_sup_right

theorem prop_of_cls {Z : Submodule ℚ W} {y : ℕ → W} {j j' : ℕ} (h : cls Z y j = cls Z y j') :
    ∃ l : ℚ, y j - l • y j' ∈ Z := by
  have hj : y j ∈ cls Z y j' := h ▸ Submodule.mem_sup_left (Submodule.mem_span_singleton_self _)
  obtain ⟨s, hs, z, hz, hsz⟩ := Submodule.mem_sup.1 hj
  obtain ⟨l, rfl⟩ := Submodule.mem_span_singleton.1 hs
  exact ⟨l, by rw [← hsz, add_sub_cancel_left]; exact hz⟩

theorem finrank_span_image_le (y : ℕ → W) (s : Finset ℕ) :
    Module.finrank ℚ (Submodule.span ℚ (y '' ↑s)) ≤ s.card := by
  have h := finrank_span_finset_le_card (R := ℚ) (s.image y)
  rw [Set.finrank, coe_image] at h
  exact h.trans card_image_le

variable [FiniteDimensional ℚ W]

/-- **One class of the forest count**: the new V-gates `E'` attached to the vertices `V'` of one line modulo `Z`
span at most `|V'|` dimensions. -/
theorem class_bound {Z : Submodule ℚ W} {At : Set W} (hsm : ∀ q ∈ At, ∀ μ : ℚ, μ • q ∈ At)
    (h2 : TwoAtomFree Z At) (y : ℕ → W) (E' V' : Finset ℕ) (hE : ∀ i ∈ E', y i ∈ Z)
    (hV : ∀ j ∈ V', y j ∉ Z) (hprop : ∀ j ∈ V', ∀ j' ∈ V', ∃ l : ℚ, y j - l • y j' ∈ Z)
    (hshape : ∀ i ∈ E', ∃ j ∈ V', (∃ q ∈ At, ∃ β : ℚ, β ≠ 0 ∧ y i = q + β • y j) ∨
      y i ∈ Submodule.span ℚ (y '' ↑V')) :
    Module.finrank ℚ (Submodule.span ℚ (y '' ↑E')) ≤ V'.card := by
  rcases E'.eq_empty_or_nonempty with rfl | ⟨i₀, hi₀⟩
  · rw [coe_empty, Set.image_empty, Submodule.span_empty, finrank_bot]; exact Nat.zero_le _
  obtain ⟨j₀, hj₀, -⟩ := hshape i₀ hi₀
  set Qs : Set W := {q | q ∈ At ∧ ∃ i ∈ E', ∃ j ∈ V', ∃ β : ℚ, β ≠ 0 ∧ y i = q + β • y j}
  have hQ : Module.finrank ℚ (Submodule.span ℚ Qs) ≤ 1 := by
    rcases Qs.eq_empty_or_nonempty with h | ⟨q₀, hq₀⟩
    · rw [h, Submodule.span_empty, finrank_bot]; exact zero_le_one
    obtain ⟨hq₀A, i₁, hi₁, j₁, hj₁, β₁, hβ₁, hy₁⟩ := hq₀
    have hle : Submodule.span ℚ Qs ≤ Submodule.span ℚ ({q₀} : Finset W) := by
      refine Submodule.span_le.2 ?_
      rintro q ⟨hqA, i, hi, j, hj, β, hβ, hy⟩
      obtain ⟨l, hl⟩ := hprop j hj j₁ hj₁
      set μ : ℚ := β * l / β₁
      have hμ : μ * β₁ = β * l := by simp only [μ]; field_simp
      have hmem : q + (-μ) • q₀ ∈ Z := by
        have e : q + (-μ) • q₀ = y i + (-μ) • y i₁ + (-β) • (y j - l • y j₁) + (μ * β₁ - β * l) • y j₁ := by
          rw [show q = y i - β • y j by rw [hy]; abel, show q₀ = y i₁ - β₁ • y j₁ by rw [hy₁]; abel]
          module
        rw [e, hμ, sub_self, zero_smul, add_zero]
        exact Z.add_mem (Z.add_mem (hE i hi) (Z.smul_mem _ (hE i₁ hi₁))) (Z.smul_mem _ hl)
      have h0 := h2 q hqA _ (hsm q₀ hq₀A (-μ)) hmem
      rw [coe_singleton, SetLike.mem_coe, Submodule.mem_span_singleton]
      exact ⟨μ, by rw [neg_smul] at h0; exact (eq_of_add_neg_eq_zero h0).symm⟩
    exact (Submodule.finrank_mono hle).trans ((finrank_span_finset_le_card _).trans (card_singleton _).le)
  set Y := Submodule.span ℚ (y '' ↑V') ⊔ Submodule.span ℚ Qs
  have hEY : Submodule.span ℚ (y '' ↑E') ≤ Y ⊓ Z := by
    refine Submodule.span_le.2 ?_
    rintro _ ⟨i, hi, rfl⟩
    refine ⟨?_, hE i hi⟩
    obtain ⟨j, hj, h | h⟩ := hshape i hi
    · obtain ⟨q, hq, β, hβ, hy⟩ := h
      rw [SetLike.mem_coe, hy]
      exact Y.add_mem (Submodule.mem_sup_right (Submodule.subset_span ⟨hq, i, hi, j, hj, β, hβ, hy⟩))
        (Submodule.mem_sup_left (Submodule.smul_mem _ _ (Submodule.subset_span ⟨j, hj, rfl⟩)))
    · exact Submodule.mem_sup_left h
  have hlt : Y ⊓ Z < Y := by
    have hy0 : y j₀ ∈ Y := Submodule.mem_sup_left (Submodule.subset_span ⟨j₀, hj₀, rfl⟩)
    refine lt_of_le_of_ne inf_le_left fun h => hV j₀ hj₀ ?_
    rw [← h] at hy0
    exact hy0.2
  have h1 := Submodule.finrank_lt_finrank_of_lt hlt
  have h3 : Module.finrank ℚ Y ≤ _ :=
    Submodule.finrank_add_le_finrank_add_finrank (Submodule.span ℚ (y '' ↑V')) (Submodule.span ℚ Qs)
  have h4 := finrank_span_image_le y V'
  have h5 := Submodule.finrank_mono hEY
  omega

theorem finrank_span_biUnion_le (y : ℕ → W) {ι : Type} (T : Finset ι) (t : ι → Finset ℕ) :
    Module.finrank ℚ (Submodule.span ℚ (y '' ↑(T.biUnion t))) ≤
      ∑ κ ∈ T, Module.finrank ℚ (Submodule.span ℚ (y '' ↑(t κ))) := by
  induction T using Finset.induction_on with
  | empty => rw [biUnion_empty, coe_empty, Set.image_empty, Submodule.span_empty, finrank_bot]; exact Nat.zero_le _
  | insert a T ha ih =>
    rw [biUnion_insert, sum_insert ha, coe_union, Set.image_union, Submodule.span_union]
    exact (Submodule.finrank_add_le_finrank_add_finrank _ _).trans (Nat.add_le_add_left ih _)

/-- **Theorem 1 with Corollary 2's pricing** (the forest count). -/
theorem forestBound : ForestBound := by
  intro W _ _ _ Z At hsm h2 y N F hF hstep
  set V := vgs Z y N
  set E := news Z y N
  set Non := (range N).filter (fun j => y j ∉ Z)
  set EF := E.filter (· ∈ F)
  set EB := E.filter (· ∉ F)
  have hEV : E ⊆ V := filter_subset _ _
  have hEZ : ∀ i ∈ E, y i ∈ Z := fun i hi => (mem_filter.1 (hEV hi)).2
  have hEN : ∀ i ∈ E, i < N := fun i hi => mem_range.1 (mem_filter.1 (hEV hi)).1
  -- the attachment of each new FP32 V-gate
  have hatt : ∀ i ∈ EF, ∃ j, Att Z At y i j := fun i hi =>
    att_of_new hsm h2 (mem_filter.1 hi).1 (hstep i (mem_filter.1 hi).2)
  let att : ℕ → ℕ := fun i => if h : ∃ j, Att Z At y i j then h.choose else 0
  have hatt' : ∀ i ∈ EF, Att Z At y i (att i) := fun i hi => by
    simp only [att]
    split_ifs with h
    · exact h.choose_spec
    · exact absurd (hatt i hi) h
  let T := EF.image (fun i => cls Z y (att i))
  let Ek : Submodule ℚ W → Finset ℕ := fun κ => EF.filter (fun i => cls Z y (att i) = κ)
  let Vk : Submodule ℚ W → Finset ℕ := fun κ => Non.filter (fun j => cls Z y j = κ)
  have hk : ∀ κ ∈ T, Module.finrank ℚ (Submodule.span ℚ (y '' ↑(Ek κ))) ≤ (Vk κ).card := by
    intro κ _
    refine class_bound hsm h2 y (Ek κ) (Vk κ) (fun i hi => hEZ i (mem_filter.1 (mem_filter.1 hi).1).1)
      (fun j hj => (mem_filter.1 (mem_filter.1 hj).1).2) (fun j hj j' hj' => prop_of_cls
        (((mem_filter.1 hj).2).trans ((mem_filter.1 hj').2).symm)) ?_
    intro i hi
    obtain ⟨hiEF, hκ⟩ := mem_filter.1 hi
    have hiE := (mem_filter.1 hiEF).1
    obtain ⟨hlt, hnZ, h⟩ := hatt' i hiEF
    have hmemV : att i ∈ Vk κ :=
      mem_filter.2 ⟨mem_filter.2 ⟨mem_range.2 (hlt.trans (hEN i hiE)), hnZ⟩, hκ⟩
    refine ⟨att i, hmemV, ?_⟩
    rcases h with h | ⟨γ₁, γ₂, j₂, hj₂, h₁, h₂, hj₂Z, hy⟩
    · exact Or.inl h
    · right
      have hj₂V : j₂ ∈ Vk κ := mem_filter.2 ⟨mem_filter.2 ⟨mem_range.2 (hj₂.trans (hEN i hiE)), hj₂Z⟩,
        (cls_eq_of_edge (hEZ i hiE) h₁ h₂ hy).trans hκ⟩
      rw [hy]
      exact Submodule.add_mem _ (Submodule.smul_mem _ _ (Submodule.subset_span ⟨att i, hmemV, rfl⟩))
        (Submodule.smul_mem _ _ (Submodule.subset_span ⟨j₂, hj₂V, rfl⟩))
  have hEFsum : Module.finrank ℚ (Submodule.span ℚ (y '' ↑EF)) ≤ Non.card := by
    have hbu : T.biUnion Ek = EF := biUnion_filter_eq_of_maps_to fun i hi => mem_image_of_mem _ hi
    have hdisj : (↑T : Set (Submodule ℚ W)).PairwiseDisjoint Vk := by
      intro a _ b _ hab
      exact disjoint_filter.2 fun j _ h1 h2 => hab (h1.symm.trans h2)
    have hsub : T.biUnion Vk ⊆ Non := biUnion_subset.2 fun κ _ => filter_subset _ _
    calc Module.finrank ℚ (Submodule.span ℚ (y '' ↑EF))
        = Module.finrank ℚ (Submodule.span ℚ (y '' ↑(T.biUnion Ek))) := by rw [hbu]
      _ ≤ ∑ κ ∈ T, Module.finrank ℚ (Submodule.span ℚ (y '' ↑(Ek κ))) := finrank_span_biUnion_le y T Ek
      _ ≤ ∑ κ ∈ T, (Vk κ).card := sum_le_sum hk
      _ = (T.biUnion Vk).card := (card_biUnion hdisj).symm
      _ ≤ Non.card := card_le_card hsub
  -- the dimension of the V-gates
  have hVset : y '' {i | i < N ∧ y i ∈ Z} = y '' ↑V := by
    congr 1; ext i; simp [V, vgs]
  have hd1 : Module.finrank ℚ (Submodule.span ℚ (y '' ↑V)) ≤
      Module.finrank ℚ (Submodule.span ℚ (y '' ↑E)) :=
    Submodule.finrank_mono (Submodule.span_le.2 (by rintro _ ⟨i, hi, rfl⟩; exact mem_span_news Z y N i hi))
  have hEsplit : E = EB ∪ EF := by
    ext i; simp only [EB, EF, mem_union, mem_filter]; tauto
  have hd2 : Module.finrank ℚ (Submodule.span ℚ (y '' ↑E)) ≤ EB.card + Non.card := by
    rw [hEsplit, coe_union, Set.image_union, Submodule.span_union]
    exact (Submodule.finrank_add_le_finrank_add_finrank _ _).trans
      (Nat.add_le_add (finrank_span_image_le y EB) hEFsum)
  have hd3 : Module.finrank ℚ (Submodule.span ℚ (y '' ↑E)) ≤ EB.card + EF.card := by
    have h := finrank_span_image_le y E
    have hc : EF.card + EB.card = E.card := card_filter_add_card_filter_not _
    omega
  -- the counts
  set NonB := Non.filter (· ∉ F)
  set NonF := Non.filter (· ∈ F)
  have hNon : Non.card = NonF.card + NonB.card := (card_filter_add_card_filter_not _).symm
  have hdisjEN : ∀ s t : Finset ℕ, s ⊆ E → t ⊆ Non → Disjoint s t := fun s t hs ht =>
    disjoint_left.2 fun i his hit => (mem_filter.1 (ht hit)).2 (hEZ i (hs his))
  have hB : EB.card + NonB.card ≤ N - F.card := by
    rw [← card_union_of_disjoint (hdisjEN _ _ (filter_subset _ _) (filter_subset _ _)), ← card_range N,
      ← card_sdiff_of_subset hF]
    refine card_le_card fun i hi => ?_
    rcases mem_union.1 hi with hi | hi
    · exact mem_sdiff.2 ⟨mem_range.2 (hEN i (mem_filter.1 hi).1), (mem_filter.1 hi).2⟩
    · exact mem_sdiff.2 ⟨(mem_filter.1 (mem_filter.1 hi).1).1, (mem_filter.1 hi).2⟩
  have hFc : EF.card + NonF.card ≤ F.card := by
    rw [← card_union_of_disjoint (hdisjEN _ _ (filter_subset _ _) (filter_subset _ _))]
    refine card_le_card fun i hi => ?_
    rcases mem_union.1 hi with hi | hi
    · exact (mem_filter.1 hi).2
    · exact (mem_filter.1 hi).2
  rw [hVset]
  omega

end

/-! ## Grid -/

section Grid

theorem pt_TF (N m n k u i a b j : ℕ) : pt N m n k true false u i a b j = pt N m n k true false u i a 0 0 := by
  simp [pt]

theorem pt_FT (N m n k u i a b j : ℕ) : pt N m n k false true u i a b j = pt N m n k false true 0 0 0 b j := by
  simp [pt]

theorem pt_FF (N m n k u i a b j : ℕ) : pt N m n k false false u i a b j = pt N m n k false false 0 0 0 0 0 := by
  simp [pt]

/-- The mixed difference of a product of two functions that it kills. -/
theorem sd_mul {N m n k u i a b j : ℕ} {φ ψ : Noise N m n k → ℚ} (hφ : sd N m n k u i a b j φ = 0)
    (hψ : sd N m n k u i a b j ψ = 0) :
    sd N m n k u i a b j (φ * ψ) =
      (φ (pt N m n k true false u i a 0 0) - φ (pt N m n k false false 0 0 0 0 0)) *
        (ψ (pt N m n k false true 0 0 0 b j) - ψ (pt N m n k false false 0 0 0 0 0)) +
      (φ (pt N m n k false true 0 0 0 b j) - φ (pt N m n k false false 0 0 0 0 0)) *
        (ψ (pt N m n k true false u i a 0 0) - ψ (pt N m n k false false 0 0 0 0 0)) := by
  rw [sd_apply] at hφ hψ ⊢
  rw [pt_TF, pt_FT, pt_FF] at hφ hψ ⊢
  simp only [Pi.mul_apply]
  linear_combination ψ (pt N m n k true true u i a b j) * hφ +
    (φ (pt N m n k true false u i a 0 0) + φ (pt N m n k false true 0 0 0 b j) -
      φ (pt N m n k false false 0 0 0 0 0)) * hψ

/-- A lower-triangular `5 × 5` matrix with a nonzero diagonal is not a sum of four rank-one matrices. -/
theorem no_rank4 (P : Fin 5 → Fin 5 → ℚ) (X : Fin 5 → Fin 4 → ℚ) (Y : Fin 4 → Fin 5 → ℚ)
    (hP : ∀ i l, P i l = ∑ r, X i r * Y r l) (hd : ∀ i, P i i ≠ 0) (hu : ∀ i l, i < l → P i l = 0) : False := by
  have hM : Matrix.of P = Matrix.of X * Matrix.of Y := by
    ext i l; simp [Matrix.mul_apply, hP]
  have hdet : (Matrix.of P).det ≠ 0 := by
    rw [Matrix.det_of_isLowerTriangular (Matrix.of P) (fun i l h => hu i l h)]
    exact Finset.prod_ne_zero_iff.2 fun i _ => hd i
  have h1 := Matrix.rank_of_isUnit (Matrix.of P) ((Matrix.isUnit_iff_isUnit_det _).2 (isUnit_iff_ne_zero.2 hdet))
  have h2 := Matrix.rank_mul_le_right (Matrix.of X) (Matrix.of Y)
  have h3 := Matrix.rank_le_card_height (Matrix.of Y)
  rw [← hM, h1] at h2
  simp only [Fintype.card_fin] at h2 h3
  omega

/-- The mixed difference of a combination of checked words reads its coefficients on the checkpoint matrices. -/
theorem sd_comb {k m n N : ℕ} (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) {U : Finset ℕ}
    (hU : U ⊆ range N) (c : ℕ → ℕ → ℚ) {u0 e a b : ℕ} (hu0 : u0 ∈ U) (he : e < m * n) (ha : a < k) (hb : b < k) :
    sd N m n k u0 (e / n) a b (e % n) (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • word k m n N R A B u p) =
      ∑ t ∈ range (3 * k / 16 - 1), c u0 (t * (m * n) + e) * (ckMat k R (16 * (t + 1)) a b : ℚ) := by
  have hN : u0 < N := mem_range.1 (hU hu0)
  rw [map_sum, sum_eq_single u0 (fun u _ hne => by
      rw [map_sum]
      exact sum_eq_zero fun p _ => by rw [map_smul, sd_word_ne R A B _ _ _ _ hne, smul_zero])
    (fun h => absurd hu0 h), map_sum]
  simp only [map_smul, smul_eq_mul, sd_word_self R A B hN he ha hb, coefAt]
  push_cast
  rw [nonFinal]
  simp only [mul_ite, mul_zero]
  rw [sum_blocks (m * n) e he (fun p => c u0 p * (ckMat k R (16 * (p / (m * n) + 1)) a b : ℚ))]
  refine sum_congr rfl fun t _ => ?_
  rw [div_blocks he]

/-- **Rank ≥ 5 against two atoms**: if a combination of non-final checked words differs from a sum of two products
of free values by a free value, its coefficients are `0`. -/
theorem comb_two_atoms {k m n N : ℕ} {R : Mat} (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    {U : Finset ℕ} (hind : CheckpointIndependent k 16 R) (hmr : MinRankGe k R 5) (hU : U ⊆ range N)
    (c : ℕ → ℕ → ℚ) {a₁ b₁ a₂ b₂ : Noise N m n k → ℚ} (ha₁ : a₁ ∈ V0 N m n k) (hb₁ : b₁ ∈ V0 N m n k)
    (ha₂ : a₂ ∈ V0 N m n k) (hb₂ : b₂ ∈ V0 N m n k)
    (hc : (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • word k m n N R A B u p) - (a₁ * b₁ + a₂ * b₂) ∈
      V0 N m n k) :
    ∀ u ∈ U, ∀ p < nonFinal k m n, c u p = 0 := by
  intro u hu p hp
  have hmn : 0 < m * n := Nat.pos_of_ne_zero fun h => by simp [nonFinal, h] at hp
  have hp' : p / (m * n) < 3 * k / 16 - 1 := (Nat.div_lt_iff_lt_mul hmn).2 hp
  have hT : 0 < 3 * k / 16 - 1 := lt_of_le_of_lt (Nat.zero_le _) hp'
  have hk : 16 * (3 * k / 16 - 1) < 3 * k := by omega
  set e := p % (m * n) with he_def
  have he : e < m * n := Nat.mod_lt _ hmn
  have hzero : ∀ a < k, ∀ b < k,
      ∑ t ∈ range (3 * k / 16 - 1), c u (t * (m * n) + e) * (ckMat k R (16 * (t + 1)) a b : ℚ) = 0 := by
    intro a ha b hb
    by_contra hne
    obtain ⟨r, s, -, -, hrs, hdiag, hup⟩ := hmr (fun t => c u (t * (m * n) + e)) a ha b hb hne
    let Pd : (Noise N m n k → ℚ) → ℕ → ℚ := fun φ a' =>
      φ (pt N m n k true false u (e / n) a' 0 0) - φ (pt N m n k false false 0 0 0 0 0)
    let Rd : (Noise N m n k → ℚ) → ℕ → ℚ := fun φ b' =>
      φ (pt N m n k false true 0 0 0 b' (e % n)) - φ (pt N m n k false false 0 0 0 0 0)
    have hval : ∀ i l : Fin 5, ∑ t ∈ range (3 * k / 16 - 1),
        c u (t * (m * n) + e) * (ckMat k R (16 * (t + 1)) (r i) (s l) : ℚ) =
        Pd a₁ (r i) * Rd b₁ (s l) + Rd a₁ (s l) * Pd b₁ (r i) +
          (Pd a₂ (r i) * Rd b₂ (s l) + Rd a₂ (s l) * Pd b₂ (r i)) := by
      intro i l
      have h0 := sd_V0 u (e / n) (r i) (s l) (e % n) hc
      rw [map_sub, sub_eq_zero, sd_comb R A B hU c hu he (hrs i).1 (hrs l).2, map_add,
        sd_mul (sd_V0 _ _ _ _ _ ha₁) (sd_V0 _ _ _ _ _ hb₁), sd_mul (sd_V0 _ _ _ _ _ ha₂) (sd_V0 _ _ _ _ _ hb₂)] at h0
      exact h0
    refine no_rank4 (fun i l => ∑ t ∈ range (3 * k / 16 - 1),
        c u (t * (m * n) + e) * (ckMat k R (16 * (t + 1)) (r i) (s l) : ℚ))
      (fun i => ![Pd a₁ (r i), Pd b₁ (r i), Pd a₂ (r i), Pd b₂ (r i)])
      (fun q l => ![Rd b₁ (s l), Rd a₁ (s l), Rd b₂ (s l), Rd a₂ (s l)] q) (fun i l => ?_) hdiag
      (fun i l h => hup i l h)
    rw [hval i l, Fin.sum_univ_four]
    simp
    ring
  have := indepQ hind (3 * k / 16 - 1) hk (fun t => c u (t * (m * n) + e)) hzero (p / (m * n)) hp'
  rwa [he_def, Nat.div_add_mod'] at this

theorem mk_sum_smul {N m n k : ℕ} {ι : Type} (s : Finset ι) (g : ι → ℚ) (f : ι → Noise N m n k → ℚ) :
    (∑ x ∈ s, g x • (Submodule.Quotient.mk (f x) : QPart N m n k)) =
      Submodule.Quotient.mk (∑ x ∈ s, g x • f x) := by
  rw [← Submodule.mkQ_apply, map_sum]
  simp [Submodule.mkQ_apply]

theorem sum_sub_eq {k m n N : ℕ} {R : Mat} (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    {U : Finset ℕ} (c : ↥(U ×ˢ range (nonFinal k m n)) → ℚ) :
    ∃ c' : ℕ → ℕ → ℚ, (∀ x : ↥(U ×ˢ range (nonFinal k m n)), c' x.1.1 x.1.2 = c x) ∧
      ∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c' u p • word k m n N R A B u p =
        ∑ x : ↥(U ×ˢ range (nonFinal k m n)), c x • word k m n N R A B x.1.1 x.1.2 := by
  classical
  let c' : ℕ → ℕ → ℚ := fun u p => if h : (u, p) ∈ U ×ˢ range (nonFinal k m n) then c ⟨(u, p), h⟩ else 0
  refine ⟨c', fun x => by simp only [c', Prod.mk.eta, x.2, dite_true], ?_⟩
  rw [← sum_product (s := U) (t := range (nonFinal k m n)) (f := fun x => c' x.1 x.2 • word k m n N R A B x.1 x.2),
    ← sum_coe_sort (U ×ˢ range (nonFinal k m n))]
  refine sum_congr rfl fun x _ => ?_
  have hx : ((x : ℕ × ℕ).1, (x : ℕ × ℕ).2) ∈ U ×ˢ range (nonFinal k m n) := x.2
  simp only [c', hx, dite_true]

/-- **Rank ≥ 3 on the grid, in the form Theorem 1 uses** (from MinRank ≥ 5). -/
theorem gridTwoAtom : GridTwoAtom := by
  intro k m n N R A B U hind hmr hU q hq q' hq' hZ
  obtain ⟨a₁, ha₁, b₁, hb₁, rfl⟩ := hq
  obtain ⟨a₂, ha₂, b₂, hb₂, rfl⟩ := hq'
  obtain ⟨c, hc⟩ := (Submodule.mem_span_range_iff_exists_fun ℚ).1 hZ
  obtain ⟨c', hc', hsum⟩ := sum_sub_eq (m := m) (N := N) (R := R) A B c
  have hmk := mk_sum_smul (N := N) (m := m) (n := n) (k := k) univ c
    (fun x : ↥(U ×ˢ range (nonFinal k m n)) => word k m n N R A B x.1.1 x.1.2)
  rw [hc, ← Submodule.Quotient.mk_add, eq_comm, Submodule.Quotient.eq, ← hsum] at hmk
  have h0 := comb_two_atoms A B hind hmr hU c' ha₁ hb₁ ha₂ hb₂ hmk
  have hall : ∀ x : ↥(U ×ˢ range (nonFinal k m n)), c x = 0 := fun x => by
    rw [← hc' x]
    exact h0 _ (mem_product.1 x.2).1 _ (mem_range.1 (mem_product.1 x.2).2)
  rw [← hc]
  simp [hall]

theorem atoms_smul {N m n k : ℕ} : ∀ q ∈ atoms N m n k, ∀ μ : ℚ, μ • q ∈ atoms N m n k := by
  rintro _ ⟨a, ha, b, hb, rfl⟩ μ
  exact ⟨μ • a, Submodule.smul_mem _ μ ha, b, hb, by rw [smul_mul_assoc, Submodule.Quotient.mk_smul]⟩

theorem getD_of_lt {α : Type} {l : List α} {i : ℕ} (h : i < l.length) (a : α) : l.getD i a = l[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h, Option.getD_some]

theorem opnd_of_pipeOpnd {N m n k : ℕ} {Rs : List (Noise N m n k → ℚ)} {i : ℕ} {x : Noise N m n k → ℚ}
    (h : PipeOpnd (Rs.take i) x) :
    Opnd (fun j => (Submodule.Quotient.mk (Rs.getD j 0) : QPart N m n k)) i (Submodule.Quotient.mk x) := by
  rcases h with h | ⟨r, hr, β, rfl⟩
  · exact Or.inl ((Submodule.Quotient.mk_eq_zero _).2 h)
  · obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hr
    have hji : j < i := lt_of_lt_of_le hj (by simp)
    have hjl : j < Rs.length := lt_of_lt_of_le hj (by simp)
    refine Or.inr ⟨j, hji, β, ?_⟩
    rw [List.getElem_take]
    show Submodule.Quotient.mk (β • Rs[j]) = β • Submodule.Quotient.mk (Rs.getD j 0)
    rw [getD_of_lt hjl, Submodule.Quotient.mk_smul]

/-- **Theorem 1 for any word family** (the forest count on Q-parts, from independence and the two-atom property). -/
theorem theorem1Family : Theorem1Family := by
  intro N m n k w U h2 hli Rs hRs S hS hSw
  classical
  let y : ℕ → QPart N m n k := fun j => Submodule.Quotient.mk (Rs.getD j 0)
  have hF : ∀ i ∈ range Rs.length, FStep (atoms N m n k) y i := by
    intro i hi
    have hi := mem_range.1 hi
    obtain ⟨a, b, c, hc, hf, hab⟩ := hRs i hi
    have hyi : y i = Submodule.Quotient.mk (a * b + c) := by
      simp only [y, getD_of_lt hi, hf]
    rcases hab with ⟨ha, hb⟩ | ⟨⟨α, rfl⟩, hb⟩
    · exact Or.inl ⟨_, ⟨a, ha, b, hb, rfl⟩, _, opnd_of_pipeOpnd hc, by rw [hyi, Submodule.Quotient.mk_add]⟩
    · refine Or.inr ⟨α, _, _, opnd_of_pipeOpnd hb, opnd_of_pipeOpnd hc, ?_⟩
      rw [hyi, Submodule.Quotient.mk_add, ← Submodule.Quotient.mk_smul]
      congr 2
  have hB := forestBound (QPart N m n k) (familyQ w U) (atoms N m n k) atoms_smul h2 y Rs.length
    (range Rs.length) subset_rfl hF
  rw [card_range, Nat.sub_self, mul_zero, zero_add] at hB
  let wS : ↥S → QPart N m n k := fun x => Submodule.Quotient.mk (w x.1.1 x.1.2)
  have hliS : LinearIndependent ℚ wS :=
    hli.comp (fun x : ↥S => (⟨x.1, hS x.2⟩ : ↥(U ×ˢ range (nonFinal k m n))))
      (fun x x' h => Subtype.ext (Subtype.mk.inj h))
  have hle : Submodule.span ℚ (Set.range wS) ≤
      Submodule.span ℚ (y '' {i | i < Rs.length ∧ y i ∈ familyQ w U}) := by
    refine Submodule.span_le.2 ?_
    rintro _ ⟨x, rfl⟩
    obtain ⟨i, hi, hix⟩ := List.getElem_of_mem (hSw x.1 x.2)
    have hyi : y i = wS x := by simp only [y, wS, getD_of_lt hi, hix]
    refine Submodule.subset_span ⟨i, ⟨hi, ?_⟩, hyi⟩
    rw [hyi]
    exact Submodule.subset_span ⟨⟨x.1, hS x.2⟩, rfl⟩
  have hd := Submodule.finrank_mono hle
  rw [finrank_span_eq_card hliS, Fintype.card_coe] at hd
  omega

/-- **The grid gives Theorem 1's hypotheses to every family near the words.** -/
theorem gridNear : GridNear := by
  intro k m n N R A B U w hind hmr hU hw
  have hfun : (fun x : ↥(U ×ˢ range (nonFinal k m n)) =>
      (Submodule.Quotient.mk (w x.1.1 x.1.2) : QPart N m n k)) =
      fun x => Submodule.Quotient.mk (word k m n N R A B x.1.1 x.1.2) := by
    funext x
    obtain ⟨hu, hp⟩ := mem_product.1 x.2
    exact (Submodule.Quotient.eq _).2 (hw _ hu _ (mem_range.1 hp))
  refine ⟨?_, ?_⟩
  · have h := gridTwoAtom k m n N R A B U hind hmr hU
    unfold familyQ
    rw [hfun]
    exact h
  · unfold FamilyIndepQ
    rw [hfun]
    refine Fintype.linearIndependent_iff.2 fun g hg => indep_sub A B hind hU g ?_
    rw [mk_sum_smul] at hg
    exact (Submodule.Quotient.mk_eq_zero _).1 hg

/-- **Theorem 1 for every family near the words.** -/
theorem a1CostQuadNear : A1CostQuadNear := by
  intro k m n N R A B U w hind hmr hU hw
  obtain ⟨h2, hli⟩ := gridNear k m n N R A B U w hind hmr hU hw
  exact theorem1Family N m n k w U h2 hli

/-- **Theorem 1 on the grid** (`a1-cost.md` §3.2 with P8-2): `A1_cost_quad 2 0` holds for every `k` and `R`, the
instance `w = word` of `a1CostQuadNear`. -/
theorem a1CostQuadHolds : A1CostQuadHolds := by
  intro k R hind hmr m n N A B U hU Rs hRs S hS hSw
  have h := a1CostQuadNear k m n N R A B U (word k m n N R A B) hind hmr hU
    (fun u _ p _ => by rw [sub_self]; exact Submodule.zero_mem _) Rs hRs S hS hSw
  simp only [mul_zero, zero_mul, sub_zero]
  exact_mod_cast h

theorem a1CostOfNonquad : A1CostOfNonquad := fun k R h => a1CostOfSplit 2 0 k R (a1CostQuadHolds k R) h

theorem algBoundQuad8 : AlgBoundQuad8 := by
  intro k m n N R A B U L P hind hMR hU hL hP hout
  have h := algBoundQuad k m n N R A B U L P 8 2 0 hind hMR hU hL hP hout (a1CostQuadHolds k R) (by norm_num)
  rw [sub_zero, mul_one] at h
  exact_mod_cast h

theorem ttNCPQuadOfA2 : TTNCPQuadOfA2 := by
  intro k m n N R A B U M q γ₀ γ₂ h16 hk hind hMR hU hA2 hout hγ₂ hγ₀ hγ
  exact ttNCPOfQuadA2 k m n N R A B U M q γ₀ 0 γ₂ h16 hk hind hMR hU hA2 (a1CostQuadHolds k R) hout hγ₂ hγ₀
    (by rw [sub_zero, mul_one]; exact hγ)

theorem quadChainConsistent : QuadChainConsistent := by
  intro k hk h16 m n N A B U hU
  obtain ⟨R, hind, hMR, -, hA2q, -, -, -, hq, -⟩ := pipeChainConsistent k hk h16 m n N A B U hU
  refine ⟨R, hind, hMR, hA2q, hq, ?_⟩
  have hk' : (1024 : ℝ) ≤ k := by exact_mod_cast hk
  have h1 : 16 / (3 * (k : ℝ)) ≤ 1 / 2 := by
    rw [div_le_iff₀ (by positivity)]
    linarith
  linarith

theorem quadChainConsistentTight : QuadChainConsistentTight := by
  intro k hk h16 m n N A B U hU
  obtain ⟨R, hind, hMR, hA2q, hq, -⟩ := quadChainConsistent k hk h16 m n N A B U hU
  refine ⟨R, hind, hMR, hA2q, hq, one_pos, ?_, by ring⟩
  have hk' : (1024 : ℝ) ≤ k := by exact_mod_cast hk
  rw [div_lt_one (by positivity)]
  linarith

end Grid

end Pouw.Dimension.Forest
