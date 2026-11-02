import Definitions.Nci.Guarantees.Formalism
import Proofs.Core.Circuit
import Proofs.Nci.Formalism.Reconstruction
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Positivity

/-!
# The training formalism v1: Slowdown and Security

`Slowdown`: for a `(c, k)`-partition into `B` blocks `I`, with `X_I = max_T x_T(I)` and `W(I)` the dependent work `I`
does in the tokens it enters mid-path, `w` the work per token:

1. A1 at each block, with `|in(I) \ in(C)| ≤ |in(I) \ θ| ≤ k` (`computedReads_subset_unregistered`), gives
   `u(X_I) + a m(I) ≤ k` (`block_cut`).
2. Each token `I` enters holds at most `X_I w` of its dependent gates in `I` (`card_inter_tokenDep_le`), so
   `a W(I) ≤ a m(I) X_I w ≤ X_I (k - u(X_I)) w ≤ M w` (`block_mid_work_le`).
3. A block that holds a gate of `T` and reads no gate of `T` from outside holds every ancestor of that gate inside `T`
   (`mem_of_feedsWithin`), so under `TokensLinked` at most one such block meets `T`'s linked dependent gates
   (`quiet_unique`). That block has `u(X_I) ≤ k`, so its share is at most `x₀` (`BottomShare`, which at the empty block
   also gives `0 ≤ x₀`); the blocks that enter `T` mid-path do the rest,
   `Σ_{I enters T} |I ∩ T ∩ Dep| ≥ (1 - x₀ - ε) |T ∩ Dep|` (`token_mid_work_ge`).
4. Summed over the tokens, which are disjoint (`sum_card_tokenDep`), and swapped to a sum over the blocks:
   `a (1 - x₀ - ε) φ |C| ≤ Σ_I a W(I) ≤ B M w`, which is the bound on `B c / |C|`; it is `0` when `|C|`, `w` or `M` is.

`SlowdownEven` and `SlowdownFrozenTail` are `Slowdown` at the `M` and `x₀` their profiles give, and `Security` is
`Slowdown` with A2: `ρ̲ ≤ ρ ≤ B c / |C|`, `p H(ℓ*) ≤ |C|` and `B ≤ N n`.
-/

namespace Nci.Formalism

open Finset
open Verity (Declaration Partition)
open Nci.Formalism.Assumptions
open scoped NNReal ENNReal

variable {d : Declaration} {L : Type} [Fintype L]

theorem computedReads_subset_unregistered {I : Finset (Fin d.D.N)} (hI : GateSet d.D I) :
    computedReads d I ⊆ d.unregistered I := by
  intro v hv
  rw [computedReads, Finset.mem_sdiff] at hv
  rw [Verity.Declaration.unregistered, Finset.mem_sdiff]
  exact ⟨hv.1, fun hθ => hv.2 (mem_inputs_of_mem_θ hI hv.1 hθ)⟩

/-- A1 at a block of at most `c` gates that reads at most `k` values beyond `θ`. -/
theorem block_cut (A : Annotation d L) {c k : ℕ} {u : ℝ≥0 → ℝ} {a : ℝ} (hcut : CutCost d A c u a)
    {I : Finset (Fin d.D.N)} (hI : GateSet d.D I) (hc : I.card ≤ c) (hk : (d.unregistered I).card ≤ k) :
    u (A.maxShare I) + a * A.midEntries I ≤ k := by
  have h2 : (computedReads d I).card ≤ k := (card_le_card (computedReads_subset_unregistered hI)).trans hk
  calc u (A.maxShare I) + a * A.midEntries I ≤ (computedReads d I).card := hcut I hI hc
    _ ≤ k := by exact_mod_cast h2

omit [Fintype L] in
theorem share_le_one (A : Annotation d L) (T : L) (I : Finset (Fin d.D.N)) : A.share T I ≤ 1 := by
  unfold Annotation.share
  by_cases h : ((A.tokenDep T).card : ℝ≥0) = 0
  · rw [h, div_zero]
    exact zero_le_one
  · rw [div_le_one (pos_iff_ne_zero.2 h)]
    exact_mod_cast card_le_card inter_subset_right

theorem maxShare_le_one (A : Annotation d L) (I : Finset (Fin d.D.N)) : A.maxShare I ≤ 1 :=
  Finset.sup_le fun T _ => share_le_one A T I

theorem share_le_maxShare (A : Annotation d L) (T : L) (I : Finset (Fin d.D.N)) : A.share T I ≤ A.maxShare I :=
  Finset.le_sup (f := fun T => A.share T I) (mem_univ T)

omit [Fintype L] in
theorem card_inter_tokenDep_le (A : Annotation d L) (T : L) (I : Finset (Fin d.D.N)) :
    ((I ∩ A.tokenDep T).card : ℝ) ≤ (A.share T I : ℝ) * (A.tokenDep T).card := by
  unfold Annotation.share
  by_cases h : (A.tokenDep T).card = 0
  · have h0 : (I ∩ A.tokenDep T).card = 0 :=
      Nat.eq_zero_of_le_zero (h ▸ card_le_card inter_subset_right)
    simp [h0, h]
  · have h' : ((A.tokenDep T).card : ℝ) ≠ 0 := by exact_mod_cast h
    apply le_of_eq
    rw [NNReal.coe_div, NNReal.coe_natCast, NNReal.coe_natCast, div_mul_cancel₀ _ h']

open Classical in
/-- What a block does in the tokens it enters mid-path, `a W(I) ≤ M w`. -/
theorem block_mid_work_le (A : Annotation d L) {k : ℕ} {u : ℝ≥0 → ℝ} {a M : ℝ} (ha : 0 ≤ a)
    (hpeak : PeakBound u k M) {I : Finset (Fin d.D.N)} (hI : u (A.maxShare I) + a * A.midEntries I ≤ k) :
    a * ∑ T ∈ univ.filter (fun T => A.EntersMidPath I T), ((I ∩ A.tokenDep T).card : ℝ) ≤ M * A.work := by
  have hterm : ∀ T, ((I ∩ A.tokenDep T).card : ℝ) ≤ (A.maxShare I : ℝ) * A.work := by
    intro T
    have h1 : ((A.tokenDep T).card : ℝ) ≤ A.work := by
      exact_mod_cast (card_le_card (show A.tokenDep T ⊆ A.tokens.holds T from inter_subset_left)).trans
        (le_sup (f := fun T => (A.tokens.holds T).card) (mem_univ T))
    have h2 : (A.share T I : ℝ) ≤ A.maxShare I := by exact_mod_cast share_le_maxShare A T I
    calc ((I ∩ A.tokenDep T).card : ℝ) ≤ (A.share T I : ℝ) * (A.tokenDep T).card := card_inter_tokenDep_le A T I
      _ ≤ (A.maxShare I : ℝ) * A.work := by gcongr
  have hsum : ∑ T ∈ univ.filter (fun T => A.EntersMidPath I T), ((I ∩ A.tokenDep T).card : ℝ) ≤
      (A.midEntries I : ℝ) * ((A.maxShare I : ℝ) * A.work) := by
    calc _ ≤ ∑ T ∈ univ.filter (fun T => A.EntersMidPath I T), (A.maxShare I : ℝ) * A.work :=
          sum_le_sum fun T _ => hterm T
      _ = _ := by
          rw [sum_const, nsmul_eq_mul]
          rfl
  have hpk := hpeak (A.maxShare I) (maxShare_le_one A I)
  have ham : a * A.midEntries I ≤ k - u (A.maxShare I) := by linarith
  calc a * ∑ T ∈ univ.filter (fun T => A.EntersMidPath I T), ((I ∩ A.tokenDep T).card : ℝ)
      ≤ a * ((A.midEntries I : ℝ) * ((A.maxShare I : ℝ) * A.work)) := by gcongr
    _ = (a * A.midEntries I) * (A.maxShare I : ℝ) * A.work := by ring
    _ ≤ (k - u (A.maxShare I)) * (A.maxShare I : ℝ) * A.work := by gcongr
    _ = ((A.maxShare I : ℝ) * (k - u (A.maxShare I))) * A.work := by ring
    _ ≤ M * A.work := by gcongr

/-- A block that holds a gate and reads nothing of `S` from outside holds every ancestor of the gate inside `S`. -/
theorem mem_of_feedsWithin {I S : Finset (Fin d.D.N)} (hquiet : ∀ v ∈ d.D.into I, v ∉ S)
    {h g : Fin d.D.N} (hfw : FeedsWithin d.D S h g) : g ∈ I → h ∈ I := by
  induction hfw with
  | refl _ => exact id
  | @step h h' g hr hS _ ih =>
    intro hg
    by_contra hh
    refine hquiet h ?_ hS
    simp only [Verity.Circuit.into, mem_filter, mem_univ, true_and]
    exact ⟨hh, h', ih hg, hr⟩

omit [Fintype L] in
/-- Under `TokensLinked`, two blocks that read nothing of `T` from outside and meet `T`'s linked dependent gates are one
block. -/
theorem quiet_unique (A : Annotation d L) {B : ℕ} (P : Partition d.D B) {T : L} {R : Finset (Fin d.D.N)}
    (hlinked : ∀ g₁ ∈ A.tokenDep T \ R, ∀ g₂ ∈ A.tokenDep T \ R, ∃ h,
      FeedsWithin d.D (A.tokens.holds T) h g₁ ∧ FeedsWithin d.D (A.tokens.holds T) h g₂)
    {u₁ u₂ : Fin B} (hq₁ : ∀ v ∈ d.D.into (P.gates u₁), v ∉ A.tokens.holds T)
    (hq₂ : ∀ v ∈ d.D.into (P.gates u₂), v ∉ A.tokens.holds T)
    (h₁ : (P.gates u₁ ∩ (A.tokenDep T \ R)).Nonempty) (h₂ : (P.gates u₂ ∩ (A.tokenDep T \ R)).Nonempty) :
    u₁ = u₂ := by
  obtain ⟨g₁, hg₁⟩ := h₁
  obtain ⟨g₂, hg₂⟩ := h₂
  rw [mem_inter] at hg₁ hg₂
  obtain ⟨h, hf₁, hf₂⟩ := hlinked g₁ hg₁.2 g₂ hg₂.2
  have hh₁ := mem_of_feedsWithin hq₁ hf₁ hg₁.1
  have hh₂ := mem_of_feedsWithin hq₂ hf₂ hg₂.1
  rw [P.mem_gates] at hh₁ hh₂
  exact Option.some_injective _ (hh₁.symm.trans hh₂)

/-- A partition's blocks split any set of computed gates. -/
theorem sum_card_inter_gates {B : ℕ} (P : Partition d.D B) {S : Finset (Fin d.D.N)} (hS : S ⊆ d.D.computed) :
    ∑ v, (P.gates v ∩ S).card = S.card := by
  classical
  rw [← card_biUnion]
  · congr 1
    ext g
    simp only [mem_biUnion, mem_univ, true_and, mem_inter, P.mem_gates]
    constructor
    · rintro ⟨v, -, hg⟩
      exact hg
    · intro hg
      have hc := hS hg
      simp only [Verity.Circuit.computed, mem_filter, mem_univ, true_and] at hc
      have hne : P.unit g ≠ none := fun h => by
        rw [P.input_iff, hc] at h
        exact Bool.false_ne_true h
      obtain ⟨v, hv⟩ := Option.ne_none_iff_exists'.1 hne
      exact ⟨v, hv, hg⟩
  · intro v _ w _ hvw
    refine disjoint_left.2 fun g hv hw => hvw ?_
    rw [mem_inter, P.mem_gates] at hv hw
    exact Option.some_injective _ (hv.1.symm.trans hw.1)

/-- The tokens' dependent gates add up to `|Dep ∩ ⋃ 𝒯|`. -/
theorem sum_card_tokenDep (A : Annotation d L) (hA : A.Valid) :
    ∑ T, (A.tokenDep T).card = (A.dep ∩ A.tokenGates).card := by
  classical
  rw [← card_biUnion]
  · congr 1
    ext g
    simp only [Annotation.tokenDep, Annotation.tokenGates, mem_biUnion, mem_univ, true_and, mem_inter]
    constructor
    · rintro ⟨T, hT, hd⟩
      exact ⟨hd, T, hT⟩
    · rintro ⟨hd, T, hT⟩
      exact ⟨T, hT, hd⟩
  · intro T _ T' _ hTT'
    exact Disjoint.mono inter_subset_left inter_subset_left (hA.disjoint hTT')

open Classical in
/-- The blocks that enter `T` mid-path do all but `x₀ + ε` of `T`'s dependent work. -/
theorem token_mid_work_ge (A : Annotation d L) (hA : A.Valid) {B c k : ℕ} (P : Partition d.D B)
    (hP : IsCKPartition d P c k) {u : ℝ≥0 → ℝ} {a x₀ ε : ℝ} (ha : 0 ≤ a) (hcut : CutCost d A c u a)
    (hbottom : BottomShare u k x₀) (hx₀ : 0 ≤ x₀) (T : L) {R : Finset (Fin d.D.N)} (hR : R ⊆ A.tokenDep T)
    (hRε : (R.card : ℝ) ≤ ε * (A.tokenDep T).card)
    (hlinked : ∀ g₁ ∈ A.tokenDep T \ R, ∀ g₂ ∈ A.tokenDep T \ R, ∃ h,
      FeedsWithin d.D (A.tokens.holds T) h g₁ ∧ FeedsWithin d.D (A.tokens.holds T) h g₂) :
    (1 - x₀ - ε) * (A.tokenDep T).card ≤
      ∑ v ∈ univ.filter (fun v => A.EntersMidPath (P.gates v) T), ((P.gates v ∩ A.tokenDep T).card : ℝ) := by
  have hdepC : A.tokenDep T ⊆ d.D.computed := inter_subset_left.trans (hA.tokens_computed T)
  have htotal : ∑ v, ((P.gates v ∩ A.tokenDep T).card : ℝ) = (A.tokenDep T).card := by
    exact_mod_cast sum_card_inter_gates P hdepC
  have hsplit := sum_filter_add_sum_filter_not univ (fun v => A.EntersMidPath (P.gates v) T)
    (fun v => ((P.gates v ∩ A.tokenDep T).card : ℝ))
  have hterm : ∀ v, ((P.gates v ∩ A.tokenDep T).card : ℝ) ≤
      (P.gates v ∩ R).card + (P.gates v ∩ (A.tokenDep T \ R)).card := by
    intro v
    have hsub : P.gates v ∩ A.tokenDep T ⊆ (P.gates v ∩ R) ∪ (P.gates v ∩ (A.tokenDep T \ R)) := by
      intro g hg
      rw [mem_inter] at hg
      by_cases hgR : g ∈ R
      · exact mem_union_left _ (mem_inter.2 ⟨hg.1, hgR⟩)
      · exact mem_union_right _ (mem_inter.2 ⟨hg.1, mem_sdiff.2 ⟨hg.2, hgR⟩⟩)
    exact_mod_cast (card_le_card hsub).trans (card_union_le _ _)
  have hR' : ∑ v ∈ univ.filter (fun v => ¬ A.EntersMidPath (P.gates v) T), ((P.gates v ∩ R).card : ℝ) ≤ R.card := by
    have h1 : ∑ v ∈ univ.filter (fun v => ¬ A.EntersMidPath (P.gates v) T), ((P.gates v ∩ R).card : ℝ) ≤
        ∑ v, ((P.gates v ∩ R).card : ℝ) :=
      sum_le_sum_of_subset_of_nonneg (subset_univ _) (fun _ _ _ => Nat.cast_nonneg _)
    have h2 : ∑ v, ((P.gates v ∩ R).card : ℝ) = R.card := by
      exact_mod_cast sum_card_inter_gates P (hR.trans hdepC)
    linarith
  have hquietOf : ∀ v ∈ univ.filter (fun v => ¬ A.EntersMidPath (P.gates v) T),
      (P.gates v ∩ (A.tokenDep T \ R)).Nonempty → ∀ w ∈ d.D.into (P.gates v), w ∉ A.tokens.holds T := by
    intro v hv hne w hw hwT
    have hnot : ¬ A.EntersMidPath (P.gates v) T := (mem_filter.1 hv).2
    obtain ⟨g, hg⟩ := hne
    have hg' : g ∈ P.gates v ∩ A.tokenDep T := by
      rw [mem_inter] at hg ⊢
      exact ⟨hg.1, (mem_sdiff.1 hg.2).1⟩
    have h1 : 0 < (P.gates v ∩ A.tokenDep T).card := card_pos.2 ⟨g, hg'⟩
    have h2 : 0 < (A.tokenDep T).card := card_pos.2 ⟨g, (mem_inter.1 hg').2⟩
    have hpos : 0 < A.share T (P.gates v) := by
      unfold Annotation.share
      exact div_pos (by exact_mod_cast h1) (by exact_mod_cast h2)
    exact hnot ⟨hpos, w, hw, hwT⟩
  have hrest : ∑ v ∈ univ.filter (fun v => ¬ A.EntersMidPath (P.gates v) T),
      ((P.gates v ∩ (A.tokenDep T \ R)).card : ℝ) ≤ x₀ * (A.tokenDep T).card := by
    by_cases hex : ∃ v ∈ univ.filter (fun v => ¬ A.EntersMidPath (P.gates v) T),
        (P.gates v ∩ (A.tokenDep T \ R)).Nonempty
    · obtain ⟨q, hqQ, hqne⟩ := hex
      rw [sum_eq_single q]
      · have hqblock := block_cut A hcut (P.gates_subset_computed q) (hP.1 q) (hP.2 q)
        have hm : 0 ≤ a * (A.midEntries (P.gates q) : ℝ) := mul_nonneg ha (Nat.cast_nonneg _)
        have hX : ((A.maxShare (P.gates q) : ℝ≥0) : ℝ) ≤ x₀ :=
          hbottom _ (maxShare_le_one A _) (by linarith)
        have hs : (A.share T (P.gates q) : ℝ) ≤ x₀ :=
          (by exact_mod_cast share_le_maxShare A T _ : (A.share T (P.gates q) : ℝ) ≤ A.maxShare (P.gates q)).trans hX
        calc ((P.gates q ∩ (A.tokenDep T \ R)).card : ℝ) ≤ (P.gates q ∩ A.tokenDep T).card := by
              exact_mod_cast card_le_card (inter_subset_inter_left sdiff_subset)
          _ ≤ (A.share T (P.gates q) : ℝ) * (A.tokenDep T).card := card_inter_tokenDep_le A T _
          _ ≤ x₀ * (A.tokenDep T).card := by gcongr
      · intro v hv hvq
        by_contra hne0
        have hne : (P.gates v ∩ (A.tokenDep T \ R)).Nonempty := by
          rw [nonempty_iff_ne_empty]
          intro h
          apply hne0
          simp [h]
        exact hvq (quiet_unique A P hlinked (hquietOf v hv hne) (hquietOf q hqQ hqne) hne hqne)
      · intro hq
        exact absurd hqQ hq
    · simp only [not_exists, not_and] at hex
      have h0 : ∑ v ∈ univ.filter (fun v => ¬ A.EntersMidPath (P.gates v) T),
          ((P.gates v ∩ (A.tokenDep T \ R)).card : ℝ) = 0 := by
        apply sum_eq_zero
        intro v hv
        rw [not_nonempty_iff_eq_empty.1 (hex v hv)]
        simp
      rw [h0]
      positivity
  have hquiet : ∑ v ∈ univ.filter (fun v => ¬ A.EntersMidPath (P.gates v) T),
      ((P.gates v ∩ A.tokenDep T).card : ℝ) ≤ R.card + x₀ * (A.tokenDep T).card := by
    calc ∑ v ∈ univ.filter (fun v => ¬ A.EntersMidPath (P.gates v) T), ((P.gates v ∩ A.tokenDep T).card : ℝ)
        ≤ ∑ v ∈ univ.filter (fun v => ¬ A.EntersMidPath (P.gates v) T),
            (((P.gates v ∩ R).card : ℝ) + (P.gates v ∩ (A.tokenDep T \ R)).card) :=
          sum_le_sum fun v _ => hterm v
      _ = ∑ v ∈ univ.filter (fun v => ¬ A.EntersMidPath (P.gates v) T), ((P.gates v ∩ R).card : ℝ) +
            ∑ v ∈ univ.filter (fun v => ¬ A.EntersMidPath (P.gates v) T),
              ((P.gates v ∩ (A.tokenDep T \ R)).card : ℝ) := sum_add_distrib
      _ ≤ R.card + x₀ * (A.tokenDep T).card := add_le_add hR' hrest
  linarith

namespace SecurityProofs

theorem Slowdown : Guarantees.Slowdown := by
  classical
  intro d L _ A c k u a x₀ ε M hA hcut hlink hpeak hbottom ha
  have hM : 0 ≤ M := by simpa using hpeak 0 zero_le_one
  have hx₀ : 0 ≤ x₀ := by
    have h0 := block_cut A (k := k) hcut (empty_subset _) (by simp)
      (by simp [Verity.Declaration.unregistered, Verity.Circuit.into])
    have hm : 0 ≤ a * (A.midEntries ∅ : ℝ) := mul_nonneg ha (Nat.cast_nonneg _)
    exact (NNReal.coe_nonneg _).trans (hbottom _ (maxShare_le_one A ∅) (by linarith))
  unfold slowdown
  refine le_iInf fun B => le_iInf fun P => le_iInf fun hP => ?_
  have hstar : a * (1 - x₀ - ε) * ((A.dep ∩ A.tokenGates).card : ℝ) ≤ B * M * A.work := by
    have hT : ∀ T, (1 - x₀ - ε) * ((A.tokenDep T).card : ℝ) ≤
        ∑ v ∈ univ.filter (fun v => A.EntersMidPath (P.gates v) T), ((P.gates v ∩ A.tokenDep T).card : ℝ) := by
      intro T
      obtain ⟨R, hR, hRε, hl⟩ := hlink T
      exact token_mid_work_ge A hA P hP ha hcut hbottom hx₀ T hR hRε hl
    have hW : ∀ v, a * ∑ T ∈ univ.filter (fun T => A.EntersMidPath (P.gates v) T),
        ((P.gates v ∩ A.tokenDep T).card : ℝ) ≤ M * A.work :=
      fun v => block_mid_work_le A ha hpeak (block_cut A hcut (P.gates_subset_computed v) (hP.1 v) (hP.2 v))
    have hswap : ∑ T, ∑ v ∈ univ.filter (fun v => A.EntersMidPath (P.gates v) T),
          ((P.gates v ∩ A.tokenDep T).card : ℝ) =
        ∑ v, ∑ T ∈ univ.filter (fun T => A.EntersMidPath (P.gates v) T), ((P.gates v ∩ A.tokenDep T).card : ℝ) := by
      simp only [sum_filter]
      exact sum_comm
    have hsumT : ((A.dep ∩ A.tokenGates).card : ℝ) = ∑ T, ((A.tokenDep T).card : ℝ) := by
      exact_mod_cast (sum_card_tokenDep A hA).symm
    calc a * (1 - x₀ - ε) * ((A.dep ∩ A.tokenGates).card : ℝ)
        = a * ∑ T, (1 - x₀ - ε) * ((A.tokenDep T).card : ℝ) := by
          rw [hsumT, ← mul_sum]
          ring
      _ ≤ a * ∑ T, ∑ v ∈ univ.filter (fun v => A.EntersMidPath (P.gates v) T),
            ((P.gates v ∩ A.tokenDep T).card : ℝ) := by
          gcongr with T
          exact hT T
      _ = ∑ v, a * ∑ T ∈ univ.filter (fun T => A.EntersMidPath (P.gates v) T),
            ((P.gates v ∩ A.tokenDep T).card : ℝ) := by
          rw [hswap, mul_sum]
      _ ≤ ∑ _v : Fin B, M * A.work := sum_le_sum fun v _ => hW v
      _ = B * M * A.work := by
          rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
          ring
  by_cases hC0 : d.D.computed.card = 0
  · have h0 : A.depShare = 0 := by simp [Annotation.depShare, hC0]
    rw [h0]
    simp
  have hCpos : (0 : ℝ) < d.D.computed.card := by exact_mod_cast Nat.pos_of_ne_zero hC0
  have key : A.depShare * (1 - x₀ - ε) * A.tokensPerBlock c * a / M * (d.D.computed.card : ℝ) ≤ (B * c : ℝ) := by
    rcases (Nat.cast_nonneg (α := ℝ) A.work).eq_or_lt with hw | hw
    · have h0 : A.tokensPerBlock c = 0 := by simp [Annotation.tokensPerBlock, ← hw]
      rw [h0]
      simp only [mul_zero, zero_mul, zero_div]
      positivity
    rcases hM.eq_or_lt with hM0 | hM0
    · rw [← hM0]
      simp only [div_zero, zero_mul]
      positivity
    have hC' : (d.D.computed.card : ℝ) ≠ 0 := hCpos.ne'
    have hw' : (A.work : ℝ) ≠ 0 := hw.ne'
    have hM' : M ≠ 0 := hM0.ne'
    have heq : A.depShare * (1 - x₀ - ε) * A.tokensPerBlock c * a / M * (d.D.computed.card : ℝ) =
        (a * (1 - x₀ - ε) * ((A.dep ∩ A.tokenGates).card : ℝ)) * c / (A.work * M) := by
      unfold Annotation.depShare Annotation.tokensPerBlock
      field_simp
    rw [heq, div_le_iff₀ (mul_pos hw hM0)]
    calc a * (1 - x₀ - ε) * ((A.dep ∩ A.tokenGates).card : ℝ) * c ≤ (B * M * A.work) * c :=
          mul_le_mul_of_nonneg_right hstar (Nat.cast_nonneg c)
      _ = B * c * (A.work * M) := by ring
  have hCne : (d.D.computed.card : ℝ≥0∞) ≠ 0 := by exact_mod_cast hC0
  rw [ENNReal.le_div_iff_mul_le (Or.inl hCne) (Or.inl (ENNReal.natCast_ne_top _)),
    ← ENNReal.ofReal_natCast d.D.computed.card, ← ENNReal.ofReal_mul' (Nat.cast_nonneg _),
    ← ENNReal.ofReal_natCast (B * c)]
  exact ENNReal.ofReal_le_ofReal (by push_cast; exact key)

theorem SlowdownEven : Guarantees.SlowdownEven := by
  intro d L _ A c k u₁ a ε hA hcut hlink hk hku ha
  have hk0 : (0 : ℝ) < k := Nat.cast_pos.2 hk
  have hu : 0 < u₁ := hk0.trans_le hku
  have h := Slowdown d L A c k (fun x => (x : ℝ) * u₁) a ((k : ℝ) / u₁) ε ((k : ℝ) ^ 2 / (4 * u₁)) hA hcut hlink
    (fun x _ => by
      show (x : ℝ) * (k - (x : ℝ) * u₁) ≤ (k : ℝ) ^ 2 / (4 * u₁)
      rw [le_div_iff₀ (by linarith)]
      nlinarith [sq_nonneg ((k : ℝ) - 2 * u₁ * x)])
    (fun x _ hx => by
      rw [le_div_iff₀ hu]
      exact hx)
    ha
  have heq : A.depShare * (1 - (k : ℝ) / u₁ - ε) * A.tokensPerBlock c * a / ((k : ℝ) ^ 2 / (4 * u₁)) =
      4 * A.depShare * A.tokensPerBlock c * a * (u₁ - k - ε * u₁) / (k : ℝ) ^ 2 := by
    rw [div_div_eq_mul_div]
    congr 1
    have hcancel : (k : ℝ) / u₁ * u₁ = k := div_mul_cancel₀ _ hu.ne'
    linear_combination (-(4 * A.depShare * A.tokensPerBlock c * a)) * hcancel
  rw [← heq]
  exact h

theorem SlowdownFrozenTail : Guarantees.SlowdownFrozenTail := by
  rintro d L _ A c k u a t ε hA hcut hlink ⟨hu0, hut⟩ ht ha
  refine Slowdown d L A c k u a t ε (t * k) hA hcut hlink (fun x hx1 => ?_) (fun x hx1 hxk => ?_) ha
  · have hx0 : (0 : ℝ) ≤ x := x.2
    have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
    by_cases hxt : (x : ℝ) ≤ t
    · nlinarith [hu0 x]
    · have hku := hut x (not_le.1 hxt) hx1
      nlinarith
  · by_contra hxt
    exact absurd hxk (not_le.2 (hut x (not_le.1 hxt) hx1))

theorem Security : Guarantees.Security := by
  intro methods dec ℓ ℓstar D p r c k n N B P u a x₀ ε M hfloor hamong hA hwithin hreach hcut hlink hpeak hbottom ha
    hp hP hB
  have hslow := Slowdown r.d (Fin r.numTokens) r.A c k u a x₀ ε M hA hcut hlink hpeak hbottom ha
  set R := r.A.depShare * (1 - x₀ - ε) * r.A.tokensPerBlock c * a / M
  have hρ : slowdown r.d c k ≤ ((B * c : ℕ) : ℝ≥0∞) / (r.d.D.computed.card : ℝ≥0∞) := by
    unfold slowdown
    exact iInf_le_of_le B (iInf_le_of_le P (iInf_le_of_le hP le_rfl))
  have hRC : ENNReal.ofReal R * (r.d.D.computed.card : ℝ≥0∞) ≤ ((B * c : ℕ) : ℝ≥0∞) := by
    by_cases hC : (r.d.D.computed.card : ℝ≥0∞) = 0
    · rw [hC, mul_zero]
      exact zero_le
    · exact (ENNReal.le_div_iff_mul_le (Or.inl hC) (Or.inl (ENNReal.natCast_ne_top _))).1 (hslow.trans hρ)
  have hBc : ((B * c : ℕ) : ℝ≥0∞) ≤ ((N * n * c : ℕ) : ℝ≥0∞) := by
    exact_mod_cast Nat.mul_le_mul_right c hB
  have hfl := hfloor r hamong hA hwithin hreach
  by_cases hR0 : R ≤ 0
  · have h0 : R * p ≤ 0 := by nlinarith
    rw [ENNReal.ofReal_eq_zero.2 h0, zero_mul]
    exact zero_le
  · rw [ENNReal.ofReal_mul (not_le.1 hR0).le, mul_assoc]
    calc ENNReal.ofReal R * (ENNReal.ofReal p * honestCost methods ℓ ℓstar)
        ≤ ENNReal.ofReal R * (r.d.D.computed.card : ℝ≥0∞) := by gcongr
      _ ≤ ((B * c : ℕ) : ℝ≥0∞) := hRC
      _ ≤ ((N * n * c : ℕ) : ℝ≥0∞) := hBc

end SecurityProofs

end Nci.Formalism
