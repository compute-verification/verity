import Definitions.Pouw.Assumptions.Dimension.Pipe
import Proofs.Pouw.Dimension.MixingProofs
import Proofs.Pouw.Dimension.MinRankProofs

/-!
# `A1_cost` in the closed scalar pipe: proofs

* `algBoundPipe`, `algBoundQuad` (`pipe_core`): with no free instruction every checked word is an input, a big output
  or an FP32 output (`pipe_mem_slots`); the words are outside `V₀` and distinct (`pipe_word_nmem`, `pipe_word_injOn`),
  so `|S_B| + |S_R| ≥ D` with `|S_B| ≤ #big`, `cost ≥ 16·#big + c_s·#steps` (`pipe_cost`), and `A1_cost` on `S_R`
  gives `c_s·#steps ≥ 16·(|S_R| − γ·D)`.
* `a1CostOfSplit`, `a1CostWitness`, `pipeWitness`, `pipeWordWitness` (the 16-FFMA chain `psums`).
* `minRankGeShift16`: a diagonal 16 × 16 minor in each of `minrank.md`'s three cases (`shMinor_diag`, `shMinor_48`,
  `shMinor_24`; the last uses `cycle_sum`); `minRankGeFacts`.
-/

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

/-! ## MinRank ≥ 16 for the shift by 24 -/

/-- `X[a, b]` for the combination `q` of the non-final checkpoint matrices. -/
noncomputable def ckComb (k : ℕ) (R : Mat) (q : ℕ → ℚ) (a b : ℕ) : ℚ :=
  ∑ t ∈ range (3 * k / 16 - 1), q t * (ckMat k R (16 * (t + 1)) a b : ℚ)

/-- The row of column `b` on the shift's diagonal `b + 24 mod k`. -/
def shRow24 (k b : ℕ) : ℕ := if b + 24 < k then b + 24 else b + 24 - k

/-- `γ` of `minrank.md` §1 on the step of `x`: the block-3 suffix sum of `q` from `x`. -/
noncomputable def shF (k : ℕ) (q : ℕ → ℚ) (x : ℕ) : ℚ :=
  ∑ t ∈ range (3 * k / 16 - 1), q t * ((if 2 * k + x < 16 * (t + 1) then 1 else 0 : ℤ) : ℚ)

/-- A `ρ × ρ` minor of `X` with nonzero diagonal whose off-diagonal entries lie off the shift's three diagonals. -/
def ShMinor (k ρ : ℕ) (X : ℕ → ℕ → ℚ) : Prop :=
  ∃ r s : Fin ρ → ℕ, Function.Injective r ∧ Function.Injective s ∧ (∀ i, r i < k ∧ s i < k) ∧
    (∀ i, X (r i) (s i) ≠ 0) ∧
    (∀ i l, i ≠ l → r i ≠ s l ∧ r i ≠ s l + 24 ∧ r i + k ≠ s l + 24 ∧ r i ≠ s l + 48 ∧ r i + k ≠ s l + 48)

theorem shF_step {c : ℕ} (q : ℕ → ℚ) {x y : ℕ} (h : x / 16 = y / 16) : shF (16 * c) q x = shF (16 * c) q y := by
  unfold shF
  refine sum_congr rfl fun t _ => ?_
  split_ifs <;> first | rfl | (exfalso; omega)

theorem comb24 {k b : ℕ} (hk : 1024 ≤ k) (hb : b < k) (q : ℕ → ℚ) :
    ckComb k (shiftBy 24 k) q (shRow24 k b) b = shF k q (shRow24 k b) - shF k q b := by
  unfold ckComb shF
  rw [← sum_sub_distrib]
  refine sum_congr rfl fun t _ => ?_
  rw [gen24 hk (by unfold shRow24; split_ifs <;> omega) hb (by unfold shRow24; split_ifs <;> omega)]
  split_ifs <;> push_cast <;> ring

theorem cycle_sum {K : ℕ} (hK : 0 < K) (f : ℕ → ℚ) :
    ∑ u ∈ range K, (f (if u + 1 < K then u + 1 else 0) - f u) = 0 := by
  obtain ⟨K', rfl⟩ : ∃ K', K = K' + 1 := ⟨K - 1, by omega⟩
  rw [sum_sub_distrib, sum_range_succ, sum_range_succ' f]
  have : ∑ u ∈ range K', f (if u + 1 < K' + 1 then u + 1 else 0) = ∑ u ∈ range K', f (u + 1) :=
    sum_congr rfl fun u hu => by
      have : u + 1 < K' + 1 := by simp at hu; omega
      simp [this]
  rw [this]
  simp

theorem shMinor_diag {c b0 : ℕ} (hk : 1024 ≤ 16 * c) (hb0 : b0 < 16 * c) (q : ℕ → ℚ)
    (hne : ckComb (16 * c) (shiftBy 24 (16 * c)) q b0 b0 ≠ 0) :
    ShMinor (16 * c) 16 (ckComb (16 * c) (shiftBy 24 (16 * c)) q) := by
  have hs : ∀ i : Fin 16, 16 * (b0 / 16) + (i : ℕ) < 16 * c := fun i => by have := i.isLt; omega
  refine ⟨fun i => 16 * (b0 / 16) + i, fun i => 16 * (b0 / 16) + i, fun i l h => Fin.ext (by simp only at h; omega),
    fun i l h => Fin.ext (by simp only at h; omega), fun i => ⟨hs i, hs i⟩, fun i => ?_, fun i l hil => ?_⟩
  · have hv : ∀ t, ckMat (16 * c) (shiftBy 24 (16 * c)) (16 * (t + 1)) (16 * (b0 / 16) + i) (16 * (b0 / 16) + i) =
        ckMat (16 * c) (shiftBy 24 (16 * c)) (16 * (t + 1)) b0 b0 := fun t => by
      have := i.isLt
      rw [gen0 hk (hs i), gen0 hk hb0]
      split_ifs <;> omega
    have e : ckComb (16 * c) (shiftBy 24 (16 * c)) q (16 * (b0 / 16) + i) (16 * (b0 / 16) + i) =
        ckComb (16 * c) (shiftBy 24 (16 * c)) q b0 b0 := by
      unfold ckComb
      exact sum_congr rfl fun t _ => by rw [hv t]
    exact fun h0 => hne (e.symm.trans h0)
  · have := i.isLt
    have := l.isLt
    have hil' : (i : ℕ) ≠ l := fun h => hil (Fin.ext h)
    beta_reduce
    refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

theorem shMinor_48 {c a0 b0 : ℕ} (hk : 1024 ≤ 16 * c) (ha0 : a0 < 16 * c) (hb0 : b0 < 16 * c) (q : ℕ → ℚ)
    (h : a0 = b0 + 48 ∨ a0 + 16 * c = b0 + 48) (hne : ckComb (16 * c) (shiftBy 24 (16 * c)) q a0 b0 ≠ 0) :
    ShMinor (16 * c) 16 (ckComb (16 * c) (shiftBy 24 (16 * c)) q) := by
  obtain ⟨j0, hj0, hbj0⟩ : ∃ j0 < 16 * c, b0 + 24 = j0 ∨ b0 + 24 = j0 + 16 * c := by
    by_cases hb : b0 + 24 < 16 * c
    · exact ⟨b0 + 24, hb, Or.inl rfl⟩
    · exact ⟨b0 + 24 - 16 * c, by omega, Or.inr (by omega)⟩
  set t0 := j0 / 16 with ht0
  set s : Fin 16 → ℕ := fun i => if 24 ≤ 16 * t0 + i then 16 * t0 + i - 24 else 16 * t0 + i + 16 * c - 24 with hs
  set r : Fin 16 → ℕ := fun i => if 16 * t0 + i + 24 < 16 * c then 16 * t0 + i + 24 else 16 * t0 + i + 24 - 16 * c
    with hr
  have hsd : ∀ i : Fin 16, s i < 16 * c ∧ (s i + 24 = 16 * t0 + i ∨ s i + 24 = 16 * t0 + i + 16 * c) := fun i => by
    have := i.isLt
    simp only [hs]
    split_ifs <;> omega
  have hrd : ∀ i : Fin 16, r i < 16 * c ∧ (r i = 16 * t0 + i + 24 ∨ r i + 16 * c = 16 * t0 + i + 24) := fun i => by
    have := i.isLt
    simp only [hr]
    split_ifs <;> omega
  refine ⟨r, s, fun i l h => Fin.ext ?_, fun i l h => Fin.ext ?_, fun i => ⟨(hrd i).1, (hsd i).1⟩, fun i => ?_,
    fun i l hil => ?_⟩
  · have := i.isLt; have := l.isLt; have := hrd i; have := hrd l; omega
  · have := i.isLt; have := l.isLt; have := hsd i; have := hsd l; omega
  · have hi := i.isLt
    have hv : ∀ t, ckMat (16 * c) (shiftBy 24 (16 * c)) (16 * (t + 1)) (r i) (s i) =
        ckMat (16 * c) (shiftBy 24 (16 * c)) (16 * (t + 1)) a0 b0 := fun t => by
      rw [gen48 (j := 16 * t0 + i) hk (hrd i).1 (hsd i).1 (by omega) (by have := hrd i; omega)
          (by have := hsd i; omega),
        gen48 (j := j0) hk ha0 hb0 hj0 (by omega) (by omega)]
      split_ifs <;> omega
    have e : ckComb (16 * c) (shiftBy 24 (16 * c)) q (r i) (s i) = ckComb (16 * c) (shiftBy 24 (16 * c)) q a0 b0 := by
      unfold ckComb
      exact sum_congr rfl fun t _ => by rw [hv t]
    exact fun h0 => hne (e.symm.trans h0)
  · have := i.isLt
    have := l.isLt
    have hil' : (i : ℕ) ≠ l := fun h => hil (Fin.ext h)
    have := hrd i
    have := hsd l
    refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

theorem shMinor_24 {c a0 b0 : ℕ} (hk : 1024 ≤ 16 * c) (hb0 : b0 < 16 * c) (q : ℕ → ℚ)
    (h : a0 = shRow24 (16 * c) b0) (hne : ckComb (16 * c) (shiftBy 24 (16 * c)) q a0 b0 ≠ 0) :
    ShMinor (16 * c) 16 (ckComb (16 * c) (shiftBy 24 (16 * c)) q) := by
  have hc : 64 ≤ c := by omega
  set nxt : ℕ → ℕ := fun u => if u + 1 < c then u + 1 else 0 with hnxt
  set H : ℕ → ℚ := fun u => shF (16 * c) q (16 * nxt u) - shF (16 * c) q (16 * u) with hH
  have hrow : ∀ u < c, ∀ i < 8, shF (16 * c) q (shRow24 (16 * c) (16 * u + i)) = shF (16 * c) q (16 * nxt u) :=
    fun u hu i hi => shF_step q (by simp only [hnxt, shRow24]; split_ifs <;> omega)
  obtain ⟨u0, hu0, hH0⟩ : ∃ u0 < c, H u0 ≠ 0 := by
    rw [h, comb24 hk hb0] at hne
    have hb : shF (16 * c) q b0 = shF (16 * c) q (16 * (b0 / 16)) := shF_step q (by omega)
    by_cases hi : b0 % 16 < 8
    · refine ⟨b0 / 16, by omega, ?_⟩
      have ha : shF (16 * c) q (shRow24 (16 * c) b0) = shF (16 * c) q (16 * nxt (b0 / 16)) :=
        shF_step q (by simp only [hnxt, shRow24]; split_ifs <;> omega)
      simp only [hH]
      rwa [← ha, ← hb]
    · have ha : shF (16 * c) q (shRow24 (16 * c) b0) = shF (16 * c) q (16 * nxt (nxt (b0 / 16))) :=
        shF_step q (by simp only [hnxt, shRow24]; split_ifs <;> omega)
      by_cases h0 : H (b0 / 16) = 0
      · refine ⟨nxt (b0 / 16), by simp only [hnxt]; split_ifs <;> omega, fun h1 => hne ?_⟩
        simp only [hH] at h0 h1
        rw [ha, hb]
        linarith
      · exact ⟨b0 / 16, by omega, h0⟩
  obtain ⟨u1, hu1, hne01, hH1⟩ : ∃ u1 < c, u1 ≠ u0 ∧ H u1 ≠ 0 := by
    by_contra hno
    simp only [not_exists, not_and, not_not] at hno
    have hsum : ∑ u ∈ range c, H u = H u0 :=
      sum_eq_single_of_mem u0 (mem_range.2 hu0) fun u hu hne => hno u (mem_range.1 hu) hne
    have hz := cycle_sum (by omega : 0 < c) (fun u => shF (16 * c) q (16 * u))
    exact hH0 (hsum.symm.trans hz)
  set s : Fin 16 → ℕ := fun i => if (i : ℕ) < 8 then 16 * u0 + i else 16 * u1 + i - 8 with hs
  have hsd : ∀ i : Fin 16, (s i = 16 * u0 + i ∧ (i : ℕ) < 8) ∨ (s i + 8 = 16 * u1 + i ∧ 8 ≤ (i : ℕ)) := fun i => by
    have := i.isLt
    simp only [hs]
    split_ifs <;> omega
  have hslt : ∀ i, s i < 16 * c := fun i => by have := i.isLt; have := hsd i; omega
  have hrd : ∀ i : Fin 16, shRow24 (16 * c) (s i) < 16 * c ∧
      (shRow24 (16 * c) (s i) = s i + 24 ∨ shRow24 (16 * c) (s i) + 16 * c = s i + 24) := fun i => by
    have := hslt i
    unfold shRow24
    split_ifs <;> omega
  refine ⟨fun i => shRow24 (16 * c) (s i), s, fun i l h => Fin.ext ?_, fun i l h => Fin.ext ?_,
    fun i => ⟨(hrd i).1, hslt i⟩, fun i => ?_, fun i l hil => ?_⟩
  · simp only at h
    have := i.isLt; have := l.isLt; have := hrd i; have := hrd l; have := hsd i; have := hsd l; omega
  · have := i.isLt; have := l.isLt; have := hsd i; have := hsd l; omega
  · simp only
    rw [comb24 hk (hslt i)]
    rcases hsd i with ⟨he, hi⟩ | ⟨he, hi⟩
    · rw [he, hrow u0 hu0 i hi, shF_step q (show (16 * u0 + i) / 16 = (16 * u0) / 16 by omega)]
      exact hH0
    · have he' : s i = 16 * u1 + (i - 8) := by omega
      rw [he', hrow u1 hu1 (i - 8) (by omega), shF_step q (show (16 * u1 + (i - 8)) / 16 = (16 * u1) / 16 by omega)]
      exact hH1
  · have := i.isLt
    have := l.isLt
    have hil' : (i : ℕ) ≠ l := fun h => hil (Fin.ext h)
    have := hrd i
    have := hsd i
    have := hsd l
    beta_reduce
    refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

theorem minRankGe_shift16 {k : ℕ} (hk : 1024 ≤ k) (h16 : 16 ∣ k) : MinRankGe k (shiftBy 24 k) 16 := by
  intro q a0 ha0 b0 hb0 hne
  have hd : a0 = b0 ∨ a0 = b0 + 24 ∨ a0 + k = b0 + 24 ∨ a0 = b0 + 48 ∨ a0 + k = b0 + 48 := by
    by_contra hc
    simp only [not_or] at hc
    obtain ⟨c0, c1, c2, c3, c4⟩ := hc
    exact hne (sum_eq_zero fun t _ => by rw [shift_support hk ha0 hb0 c0 c1 c2 c3 c4]; simp)
  change ckComb k (shiftBy 24 k) q a0 b0 ≠ 0 at hne
  suffices H : ShMinor k 16 (ckComb k (shiftBy 24 k) q) by
    obtain ⟨r, s, hr, hs, hlt, hdg, hoff⟩ := H
    refine ⟨r, s, hr, hs, hlt, hdg, fun i l hil => ?_⟩
    obtain ⟨h0, h1, h2, h3, h4⟩ := hoff i l hil.ne
    exact sum_eq_zero fun t _ => by simp [shift_support hk (hlt i).1 (hlt l).2 h0 h1 h2 h3 h4]
  obtain ⟨c, rfl⟩ := h16
  rcases hd with h | h | h | h | h
  · subst h
    exact shMinor_diag hk hb0 q hne
  · exact shMinor_24 hk hb0 q (by unfold shRow24; split_ifs <;> omega) hne
  · exact shMinor_24 hk hb0 q (by unfold shRow24; split_ifs <;> omega) hne
  · exact shMinor_48 hk ha0 hb0 q (Or.inl h) hne
  · exact shMinor_48 hk ha0 hb0 q (Or.inr h) hne

/-- **MinRank ≥ 16 for the shift by 24.** -/
theorem minRankGeShift16 : MinRankGeShift16 := fun _ hk h16 => minRankGe_shift16 hk h16

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

/-- **The closed pipe's witness, with an FP32 step.** -/
theorem pipeWitness : PipeWitness := by
  intro k hk h16 m n N A B U _
  have hbig : ∀ p f, Instr.priced p f ∈ honest k m n N (shift24 k) A B U 16 → 16 ≤ p := fun p f hm => by
    obtain ⟨u, _, q, _, heq⟩ := mem_honest.1 hm
    exact (Instr.priced.inj heq).1.le
  have hpo : prodOuts (honest k m n N (shift24 k) A B U 16 ++ [Instr.priced 8 (1 : Noise N m n k → ℚ)]) = [1] := by
    rw [prodOuts_append, prodOuts_eq_nil hbig]
    simp [prodOuts]
  refine ⟨shift24 k, honest k m n N (shift24 k) A B U 16 ++ [Instr.priced 8 1], shiftBy24_admissible hk h16,
    indep_of_fresh (fresh hk h16), minRankGe_shift16 hk h16,
    ⟨⟨?_, ?_, ?_⟩, ?_⟩, ?_, ?_, ?_⟩
  · intro c w hm
    rcases List.mem_append.1 hm with hm | hm
    · obtain ⟨u, _, q, _, heq⟩ := mem_honest.1 hm
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
    exact mem_slots_of_priced (List.mem_append_left _ (mem_honest.2 ⟨u, hu, p, hp, rfl⟩)) _
  · rw [hpo]
    rfl
  · rw [cost_append, honest_cost]
    simp [cost, Instr.price]

/-- The partial sums `Σ_{l ≤ i} x l · y l`. -/
def psums (x y : ℕ → G → ℚ) (i : ℕ) : G → ℚ := ∑ l ∈ range (i + 1), x l * y l

theorem psums_pipe {N m n k L : ℕ} {x y : ℕ → Noise N m n k → ℚ} (hx : ∀ l < L, x l ∈ V0 N m n k)
    (hy : ∀ l < L, y l ∈ V0 N m n k) :
    IsPipe ((List.range L).map (psums x y)) ∧ IsQuadPipe ((List.range L).map (psums x y)) := by
  have key : ∀ i (h : i < ((List.range L).map (psums x y)).length),
      PipeOpnd (((List.range L).map (psums x y)).take i) (if i = 0 then 0 else psums x y (i - 1)) ∧
      ((List.range L).map (psums x y))[i]'h = x i * y i + (if i = 0 then 0 else psums x y (i - 1)) ∧
      x i ∈ V0 N m n k ∧ y i ∈ V0 N m n k := by
    intro i h
    have hi : i < L := by simpa using h
    refine ⟨?_, ?_, hx i hi, hy i hi⟩
    · split_ifs with h0
      · exact Or.inl (Submodule.zero_mem _)
      · refine Or.inr ⟨psums x y (i - 1), ?_, 1, by simp⟩
        rw [← List.map_take, List.take_range]
        exact List.mem_map.2 ⟨i - 1, List.mem_range.2 (by omega), rfl⟩
    · simp only [List.getElem_map, List.getElem_range]
      split_ifs with h0
      · subst h0
        simp [psums]
      · obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
        rw [Nat.add_sub_cancel]
        simp only [psums]
        rw [Finset.sum_range_succ _ (j + 1), add_comm]
  refine ⟨fun i h => ?_, fun i h => ?_⟩
  · obtain ⟨hc, he, hxi, hyi⟩ := key i h
    exact ⟨x i, y i, _, Or.inl hxi, Or.inl hyi, hc, he⟩
  · obtain ⟨hc, he, hxi, hyi⟩ := key i h
    exact ⟨x i, y i, _, hc, he, Or.inl ⟨hxi, hyi⟩⟩

theorem e1_mem_V0 {N m n k u i a : ℕ} (hu : u < N) (hi : i < m) (ha : a < k) :
    (fun ω : Noise N m n k => (e1Of ω u i a : ℚ)) ∈ V0 N m n k := by
  refine Submodule.mem_sup_left (Submodule.subset_span (Set.mem_insert_of_mem _ ?_))
  show _ ∈ coords N m n k
  refine List.mem_append_left _ (List.mem_flatMap.2 ⟨⟨u, hu⟩, List.mem_finRange _, List.mem_flatMap.2 ⟨⟨i, hi⟩,
    List.mem_finRange _, List.mem_map.2 ⟨⟨a, ha⟩, List.mem_finRange _, ?_⟩⟩⟩)
  funext ω
  simp only [e1Of, dif_pos (show u < N ∧ i < m ∧ a < k from ⟨hu, hi, ha⟩)]
  push_cast
  rfl

theorem f1_mem_V0 {N m n k b j : ℕ} (hb : b < k) (hj : j < n) :
    (fun ω : Noise N m n k => (f1Of ω b j : ℚ)) ∈ V0 N m n k := by
  refine Submodule.mem_sup_left (Submodule.subset_span (Set.mem_insert_of_mem _ ?_))
  show _ ∈ coords N m n k
  refine List.mem_append_right _ (List.mem_flatMap.2 ⟨⟨b, hb⟩, List.mem_finRange _,
    List.mem_map.2 ⟨⟨j, hj⟩, List.mem_finRange _, ?_⟩⟩)
  funext ω
  simp only [f1Of, dif_pos (show b < k ∧ j < n from ⟨hb, hj⟩)]
  push_cast
  rfl

theorem fn2_mem_V0 {N m n k : ℕ} {φ : Noise N m n k → ℚ} (h : ∀ ω ω' : Noise N m n k, ω.2 = ω'.2 → φ ω = φ ω') :
    φ ∈ V0 N m n k :=
  Submodule.mem_sup_right (Submodule.subset_span h)

/-- **The closed pipe's witness, with a checked word in the pipe.** -/
theorem pipeWordWitness : PipeWordWitness := by
  intro k hk h16 m n N A B U u₀ hU hu₀ hm hn
  have hu₀N : u₀ < N := Finset.mem_range.1 (hU hu₀)
  set R := shift24 k
  let x : ℕ → Noise N m n k → ℚ := fun l ω => (stackX k (A ω.2 u₀) (e1Of ω u₀) R 0 l : ℚ)
  let y : ℕ → Noise N m n k → ℚ := fun l ω => (stackY k (B u₀) R (f1Of ω) l 0 : ℚ)
  have hx : ∀ l < 16, x l ∈ V0 N m n k := fun l hl => by
    have : x l = (fun ω : Noise N m n k => ((A ω.2 u₀ 0 l : ℤ) : ℚ)) + fun ω => ((e1Of ω u₀ 0 l : ℤ) : ℚ) := by
      funext ω
      simp [x, stackX, hcat3, show l < k by omega]
    rw [this]
    exact Submodule.add_mem _ (fn2_mem_V0 fun ω ω' h => by rw [h]) (e1_mem_V0 hu₀N hm (by omega))
  have hy : ∀ l < 16, y l ∈ V0 N m n k := fun l hl => by
    have : y l = (fun _ : Noise N m n k => ((B u₀ l 0 : ℤ) : ℚ)) + fun ω => ((f1Of ω l 0 : ℤ) : ℚ) := by
      funext ω
      simp [y, stackY, vcat3, show l < k by omega]
    rw [this]
    exact Submodule.add_mem _ (fn2_mem_V0 fun _ _ _ => rfl) (f1_mem_V0 (by omega) hn)
  have hword : psums x y 15 = word k m n N R A B u₀ 0 := by
    funext ω
    simp [psums, x, y, word, checked, ck, Finset.sum_apply]
  have hbig : ∀ p f, Instr.priced p f ∈ honest k m n N R A B U 16 → 16 ≤ p := fun p f hm => by
    obtain ⟨u, _, q, _, heq⟩ := mem_honest.1 hm
    exact (Instr.priced.inj heq).1.le
  have hpo : prodOuts (honest k m n N R A B U 16 ++ (List.range 16).map fun i => Instr.priced 8 (psums x y i)) =
      (List.range 16).map (psums x y) := by
    rw [prodOuts_append, prodOuts_eq_nil hbig, List.nil_append]
    generalize List.range 16 = l
    induction l with
    | nil => rfl
    | cons a l ih => simp [prodOuts, ih]
  obtain ⟨hpipe, hquad⟩ := psums_pipe hx hy
  refine ⟨R, honest k m n N R A B U 16 ++ (List.range 16).map fun i => Instr.priced 8 (psums x y i),
    shiftBy24_admissible hk h16, indep_of_fresh (fresh hk h16), minRankGe_shift16 hk h16,
    ⟨⟨?_, ?_, ?_⟩, ?_⟩, ?_, ?_, ?_, ?_, ?_⟩
  · intro c w hm
    rcases List.mem_append.1 hm with hm | hm
    · obtain ⟨u, _, q, _, heq⟩ := mem_honest.1 hm
      cases heq
    · simp at hm
  · intro p f hm
    rcases List.mem_append.1 hm with hm | hm
    · exact Or.inl (hbig p f hm)
    · obtain ⟨i, -, heq⟩ := List.mem_map.1 hm
      cases heq
      exact Or.inr le_rfl
  · rw [hpo]; exact hpipe
  · rw [hpo]; exact hquad
  · intro u hu p hp
    exact mem_slots_of_priced (List.mem_append_left _ (mem_honest.2 ⟨u, hu, p, hp, rfl⟩)) _
  · rw [hpo]; simp
  · rw [hpo, ← hword]
    exact List.mem_map.2 ⟨15, List.mem_range.2 (by norm_num), rfl⟩
  · refine pipe_word_nmem A B (indep_of_fresh (fresh hk h16)) hU (x := (u₀, 0)) ?_
    refine Finset.mem_product.2 ⟨hu₀, Finset.mem_range.2 ?_⟩
    exact Nat.mul_pos (by omega) (Nat.mul_pos hm hn)
  · have hc : cost ((List.range 16).map fun i => Instr.priced 8 (psums x y i)) = 16 * 8 := by
      simp [cost, Function.comp_def, Instr.price]
    rw [cost_append, honest_cost, hc]

end Pouw.Dimension
