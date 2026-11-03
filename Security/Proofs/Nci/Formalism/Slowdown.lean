import Definitions.Nci.Guarantees.Formalism
import Proofs.Core.Circuit
import Proofs.Nci.Formalism.Reconstruction
import Proofs.Nci.Formalism.Fit
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-!
# The training formalism v2: the bounds on `ρ`, the claim and Security

Each lower bound `X ≤ ρ` is proved as `X ≤ B c / |C|` for a partition `P` into `B` inference blocks
(`le_slowdown_iff`), and each is `X · |C| ≤ B c · Y` for `X / Y` (`div_le_partitionSlowdown`):

1. **Counting.** `P`'s blocks split `S` (`sum_card_inter_gates`) and each holds at most `max_I |I ∩ S|` of it, so
   `|S| ≤ B max_I |I ∩ S|` (`counting_le`).
2. **The token form.** With `X_T = T ∩ A \ R`, each block does at most `s*` tokens' worth of unreachable work, and at
   least `|I ∩ X_T| / α` of each token when every `|T ∩ A| ≤ α` (`sum_inter_le_unreachableWork`), so summed over the
   blocks, `Σ_T |X_T| ≤ B α s*` (`sum_card_unreachable_le`). A token whose reachable share is at most `x₀` has
   `|X_T| ≥ |T ∩ A| (1 - x₀)` (`card_unreachable_ge`). With identical token blocks that is `|𝒯| (1 - x₀) ≤ B s*`, and
   `|⋃ 𝒯| ≤ |𝒯| w` gives `σ (1 - x₀) s |C| ≤ B c s*` (`token_core`).
3. **From A1.** A gate of `X_T` in an inference block `I` means `I` cuts `T`, so `I`'s unreachable work is at most
   `m(I) max_T x_T(I)` (`unreachableWork_le`); A1 with `|in(I) \ in(C)| ≤ |in(I) \ θ| ≤ k`
   (`computedReads_subset_unregistered`) gives `a m(I) ≤ k - u(max_T x_T(I))`, and so `a s* ≤ M` at the block that
   attains `s*`.
4. **From A1′.** A reachable gate `g` of `T` lies in an inference block `I` that computes `T` from its inputs. `I` holds
   every gate of `T` that feeds `g` through `T` (`mem_of_feedsWithin`), so `g`'s dependency share is at most `x_T(I)`,
   which A1′ and `|in(I) \ θ| ≤ k` put at most `ξ`: `T ∩ R` lies among the gates `D_T(ξ)` counts. The token blocks are
   disjoint, so `g`'s token is `T`.
5. **The claim** is 2 at the `x₀` that 4 gives, times 3 (`claim_core`); its two cases are it at the `M` and `ξ` their
   profiles give, and **Security** is the claim with A2: `ρ̲ p H(ℓ*) ≤ ρ̲ |C| ≤ B c ≤ N n c`.
-/

namespace Nci.Formalism

open Finset
open Verity (Declaration Partition)
open Nci.Formalism.Assumptions
open scoped NNReal ENNReal

variable {d : Declaration} {L : Type} [Fintype L]

/-- **A lower bound on `ρ`** is one on `|𝓘| c / |C|` for every partition `𝓘` into inference blocks. -/
theorem le_slowdown_iff {c k : ℕ} {X : ℝ≥0∞} :
    X ≤ slowdown d c k ↔ ∀ (B : ℕ) (P : Partition d.D B), P.IsCK d c k → X ≤ partitionSlowdown d B c := by
  simp only [slowdown, le_iInf_iff]

/-- `X / Y ≤ B c / |C|` from `X |C| ≤ B c Y`, with `X = 0` when `|C|` is. -/
theorem div_le_partitionSlowdown {B c : ℕ} {X Y : ℝ≥0}
    (h : X * d.D.computed.card ≤ ((B * c : ℕ) : ℝ≥0) * Y) (hC : d.D.computed.card = 0 → X = 0) :
    (X : ℝ≥0∞) / Y ≤ partitionSlowdown d B c := by
  by_cases hC0 : d.D.computed.card = 0
  · rw [hC hC0, ENNReal.coe_zero, ENNReal.zero_div]
    exact zero_le
  have hCpos : (0 : ℝ≥0) < d.D.computed.card := by exact_mod_cast Nat.pos_of_ne_zero hC0
  by_cases hY : Y = 0
  · have hX : X = 0 := by
      rw [hY, mul_zero] at h
      exact (mul_eq_zero.1 (le_antisymm h (zero_le))).resolve_right hCpos.ne'
    rw [hX, ENNReal.coe_zero, ENNReal.zero_div]
    exact zero_le
  rw [partitionSlowdown, ← ENNReal.coe_div hY, ← ENNReal.coe_natCast (B * c),
    ← ENNReal.coe_natCast d.D.computed.card, ← ENNReal.coe_div hCpos.ne', ENNReal.coe_le_coe,
    div_le_div_iff₀ (pos_iff_ne_zero.2 hY) hCpos]
  exact h

/-- A partition into inference blocks has inference blocks for blocks. -/
theorem inferenceBlock_gates {B c k : ℕ} {P : Partition d.D B} (hP : P.IsCK d c k) (u : Fin B) :
    InferenceBlock d c k (P.gates u) :=
  ⟨P.gates_subset_computed u, hP.card u, hP.reads u⟩

theorem inferenceBlock_empty (c k : ℕ) : InferenceBlock d c k ∅ := by
  refine ⟨empty_subset _, by simp, ?_⟩
  simp [Verity.Declaration.unregistered, Verity.Circuit.into]

theorem computedReads_subset_unregistered (I : Finset (Fin d.D.N)) : computedReads d I ⊆ d.unregistered I := by
  intro v hv
  rw [computedReads, mem_sdiff] at hv
  rw [Verity.Declaration.unregistered, mem_sdiff]
  exact ⟨hv.1, fun hθ => hv.2 (mem_inputs_of_mem_θ hθ)⟩

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

/-- **The counting step**: `(|S| / |C|) / (m / c) ≤ B c / |C|` from `|S| ≤ B m`. -/
theorem counting_le {B c a m : ℕ} (h : a ≤ B * m) (ha : a ≤ d.D.computed.card) :
    ((a : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)) / ((m : ℝ≥0∞) / (c : ℝ≥0∞)) ≤ partitionSlowdown d B c := by
  rcases Nat.eq_zero_or_pos a with rfl | ha0
  · simp
  have hC : 0 < d.D.computed.card := ha0.trans_le ha
  have hm : 0 < m := Nat.pos_of_ne_zero fun h0 => by
    rw [h0, mul_zero] at h
    omega
  rcases Nat.eq_zero_or_pos c with rfl | hc
  · rw [Nat.cast_zero, ENNReal.div_zero (by exact_mod_cast hm.ne'), ENNReal.div_top]
    exact zero_le
  have hC' : (d.D.computed.card : ℝ≥0) ≠ 0 := by exact_mod_cast hC.ne'
  have hc' : (c : ℝ≥0) ≠ 0 := by exact_mod_cast hc.ne'
  have hmc : (0 : ℝ≥0) < (m : ℝ≥0) / c := div_pos (by exact_mod_cast hm) (by exact_mod_cast hc)
  rw [partitionSlowdown, ← ENNReal.coe_natCast a, ← ENNReal.coe_natCast d.D.computed.card, ← ENNReal.coe_natCast m,
    ← ENNReal.coe_natCast c, ← ENNReal.coe_natCast (B * c), ← ENNReal.coe_div hC', ← ENNReal.coe_div hc',
    ← ENNReal.coe_div hmc.ne', ← ENNReal.coe_div hC', ENNReal.coe_le_coe,
    div_le_div_iff₀ hmc (pos_iff_ne_zero.2 hC'), div_mul_cancel₀ _ hC', Nat.cast_mul, mul_assoc,
    mul_div_cancel₀ _ hc']
  exact_mod_cast h

/-- `max_I |I ∩ S|` bounds what an inference block holds of `S`. -/
theorem card_le_maxInBlock {c k : ℕ} {S I : Finset (Fin d.D.N)} (hI : InferenceBlock d c k I) :
    (I ∩ S).card ≤ maxInBlock d c k S :=
  le_sup (f := fun I => (I ∩ S).card) (mem_filter.2 ⟨mem_univ I, hI⟩)

omit [Fintype L] in
theorem affected_subset (A : Annotation d L) : A.affected ⊆ d.D.computed := by
  classical
  exact filter_subset _ _

omit [Fintype L] in
theorem tokenAffected_subset (A : Annotation d L) (T : L) : A.tokenAffected T ⊆ d.D.computed :=
  inter_subset_right.trans (affected_subset A)

omit [Fintype L] in
theorem mem_reachable (A : Annotation d L) {c k : ℕ} {g : Fin d.D.N} :
    g ∈ A.reachable c k ↔
      g ∈ A.affected ∧ ∃ T, g ∈ A.tokens.holds T ∧ ∃ I, InferenceBlock d c k I ∧ g ∈ I ∧ A.FromInputs I T := by
  classical
  rw [Annotation.reachable, mem_filter]

omit [Fintype L] in
theorem share_le_one (A : Annotation d L) (T : L) (I : Finset (Fin d.D.N)) : A.share T I ≤ 1 := by
  unfold Annotation.share
  by_cases h : ((A.tokenAffected T).card : ℝ≥0) = 0
  · rw [h, div_zero]
    exact zero_le_one
  · rw [div_le_one₀ (pos_iff_ne_zero.2 h)]
    exact_mod_cast card_le_card inter_subset_right

theorem maxShare_le_one (A : Annotation d L) (I : Finset (Fin d.D.N)) : A.maxShare I ≤ 1 :=
  Finset.sup_le fun T _ => share_le_one A T I

theorem share_le_maxShare (A : Annotation d L) (T : L) (I : Finset (Fin d.D.N)) : A.share T I ≤ A.maxShare I :=
  Finset.le_sup (f := fun T => A.share T I) (mem_univ T)

theorem unreachableWork_le_capacity (A : Annotation d L) {c k : ℕ} {I : Finset (Fin d.D.N)}
    (hI : InferenceBlock d c k I) : A.unreachableWork c k I ≤ A.unreachableCapacity c k :=
  Finset.le_sup (f := A.unreachableWork c k) (mem_filter.2 ⟨mem_univ I, hI⟩)

/-- A block does at least `|I ∩ X_T| / α` of each token's unreachable work when every `|T ∩ A| ≤ α`. -/
theorem sum_inter_le_unreachableWork (A : Annotation d L) {c k β : ℕ}
    (hβ : ∀ T, (A.tokenAffected T).card ≤ β) (I : Finset (Fin d.D.N)) :
    (∑ T, ((I ∩ (A.tokenAffected T \ A.reachable c k)).card : ℝ≥0)) ≤ β * A.unreachableWork c k I := by
  rw [Annotation.unreachableWork, mul_sum]
  refine sum_le_sum fun T _ => ?_
  by_cases h0 : (A.tokenAffected T).card = 0
  · have : (I ∩ (A.tokenAffected T \ A.reachable c k)).card = 0 :=
      Nat.eq_zero_of_le_zero (h0 ▸ card_le_card (inter_subset_right.trans sdiff_subset))
    rw [this, Nat.cast_zero]
    exact zero_le
  · have hα : ((A.tokenAffected T).card : ℝ≥0) ≠ 0 := by exact_mod_cast h0
    calc ((I ∩ (A.tokenAffected T \ A.reachable c k)).card : ℝ≥0)
        = ((I ∩ (A.tokenAffected T \ A.reachable c k)).card : ℝ≥0) / (A.tokenAffected T).card *
            (A.tokenAffected T).card := (div_mul_cancel₀ _ hα).symm
      _ ≤ ((I ∩ (A.tokenAffected T \ A.reachable c k)).card : ℝ≥0) / (A.tokenAffected T).card * β := by
          gcongr
          exact_mod_cast hβ T
      _ = β * (((I ∩ (A.tokenAffected T \ A.reachable c k)).card : ℝ≥0) / (A.tokenAffected T).card) :=
          mul_comm _ _

/-- **The unreachable work of a partition into inference blocks**: `Σ_T |X_T| ≤ B α s*` when every `|T ∩ A| ≤ α`. -/
theorem sum_card_unreachable_le (A : Annotation d L) {B c k β : ℕ} {P : Partition d.D B} (hP : P.IsCK d c k)
    (hβ : ∀ T, (A.tokenAffected T).card ≤ β) :
    (∑ T, ((A.tokenAffected T \ A.reachable c k).card : ℝ≥0)) ≤ B * β * A.unreachableCapacity c k := by
  have hsplit : ∀ T, ((A.tokenAffected T \ A.reachable c k).card : ℝ≥0) =
      ∑ u, ((P.gates u ∩ (A.tokenAffected T \ A.reachable c k)).card : ℝ≥0) := fun T => by
    exact_mod_cast (sum_card_inter_gates P (sdiff_subset.trans (tokenAffected_subset A T))).symm
  calc (∑ T, ((A.tokenAffected T \ A.reachable c k).card : ℝ≥0))
      = ∑ T, ∑ u, ((P.gates u ∩ (A.tokenAffected T \ A.reachable c k)).card : ℝ≥0) :=
        sum_congr rfl fun T _ => hsplit T
    _ = ∑ u, ∑ T, ((P.gates u ∩ (A.tokenAffected T \ A.reachable c k)).card : ℝ≥0) := sum_comm
    _ ≤ ∑ _u : Fin B, (β : ℝ≥0) * A.unreachableCapacity c k := by
        refine sum_le_sum fun u _ => (sum_inter_le_unreachableWork A hβ _).trans ?_
        gcongr
        exact unreachableWork_le_capacity A (inferenceBlock_gates hP u)
    _ = B * β * A.unreachableCapacity c k := by
        rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, mul_assoc]

omit [Fintype L] in
/-- A token whose reachable share is at most `x` has `|T ∩ A \ R| ≥ |T ∩ A| (1 - x)`. -/
theorem card_unreachable_ge (A : Annotation d L) {c k : ℕ} {x : ℝ≥0} (T : L) (hx : A.reachableShare c k T ≤ x) :
    ((A.tokenAffected T).card : ℝ≥0) * (1 - x) ≤ ((A.tokenAffected T \ A.reachable c k).card : ℝ≥0) := by
  have hsplit := card_sdiff_add_card_inter (A.tokenAffected T) (A.reachable c k)
  have hR : ((A.tokenAffected T ∩ A.reachable c k).card : ℝ≥0) ≤ (A.tokenAffected T).card * x := by
    by_cases h0 : (A.tokenAffected T).card = 0
    · have : (A.tokenAffected T ∩ A.reachable c k).card = 0 :=
        Nat.eq_zero_of_le_zero (h0 ▸ card_le_card inter_subset_left)
      rw [this, Nat.cast_zero]
      exact zero_le
    · have hα : (0 : ℝ≥0) < (A.tokenAffected T).card := by exact_mod_cast Nat.pos_of_ne_zero h0
      rw [Annotation.reachableShare, div_le_iff₀ hα] at hx
      rw [mul_comm]
      exact hx
  rw [mul_tsub, mul_one, tsub_le_iff_right]
  calc ((A.tokenAffected T).card : ℝ≥0)
      = ((A.tokenAffected T \ A.reachable c k).card : ℝ≥0) + (A.tokenAffected T ∩ A.reachable c k).card := by
        exact_mod_cast hsplit.symm
    _ ≤ _ := by gcongr

/-- **The token core**, identical token blocks: `σ (1 - x₀) s |C| ≤ B c s*`. -/
theorem token_core (A : Annotation d L) {c k α : ℕ} {x₀ : ℝ≥0} {B : ℕ} {P : Partition d.D B}
    (hid : A.Identical α) (hR : A.ReachableAtMost c k x₀) (hP : P.IsCK d c k) :
    A.tokenShare * (1 - x₀) * A.tokensPerBlock c * d.D.computed.card ≤
      ((B * c : ℕ) : ℝ≥0) * A.unreachableCapacity c k := by
  obtain ⟨hα, hαT⟩ := hid
  have hsum := sum_card_unreachable_le A hP (β := α) fun T => (hαT T).le
  have hlow : ∀ T, (α : ℝ≥0) * (1 - x₀) ≤ ((A.tokenAffected T \ A.reachable c k).card : ℝ≥0) := fun T => by
    have := card_unreachable_ge A T (hR T)
    rwa [hαT T] at this
  have h1 : (α : ℝ≥0) * ((Fintype.card L : ℝ≥0) * (1 - x₀)) ≤ α * (B * A.unreachableCapacity c k) := by
    calc (α : ℝ≥0) * ((Fintype.card L : ℝ≥0) * (1 - x₀)) = ∑ _T : L, (α : ℝ≥0) * (1 - x₀) := by
          rw [sum_const, card_univ, nsmul_eq_mul]
          ring
      _ ≤ ∑ T, ((A.tokenAffected T \ A.reachable c k).card : ℝ≥0) := sum_le_sum fun T _ => hlow T
      _ ≤ B * α * A.unreachableCapacity c k := hsum
      _ = α * (B * A.unreachableCapacity c k) := by ring
  have h2 : (Fintype.card L : ℝ≥0) * (1 - x₀) ≤ B * A.unreachableCapacity c k :=
    le_of_mul_le_mul_left h1 (by exact_mod_cast hα)
  have h3 : A.tokenGates.card ≤ Fintype.card L * A.work := by
    calc A.tokenGates.card ≤ ∑ T, (A.tokens.holds T).card := card_biUnion_le
      _ ≤ ∑ _T : L, A.work := sum_le_sum fun T _ => le_sup (f := fun T => (A.tokens.holds T).card) (mem_univ T)
      _ = Fintype.card L * A.work := by rw [sum_const, card_univ, smul_eq_mul]
  have hσ : A.tokenShare * d.D.computed.card ≤ A.tokenGates.card := by
    unfold Annotation.tokenShare
    by_cases hC : (d.D.computed.card : ℝ≥0) = 0
    · rw [hC, mul_zero]
      exact zero_le
    · rw [div_mul_cancel₀ _ hC]
  have hs : (A.work : ℝ≥0) * A.tokensPerBlock c ≤ c := by
    unfold Annotation.tokensPerBlock
    by_cases hw : (A.work : ℝ≥0) = 0
    · rw [hw, zero_mul]
      exact zero_le
    · rw [mul_div_cancel₀ _ hw]
  calc A.tokenShare * (1 - x₀) * A.tokensPerBlock c * d.D.computed.card
      = (A.tokenShare * d.D.computed.card) * (1 - x₀) * A.tokensPerBlock c := by ring
    _ ≤ (A.tokenGates.card : ℝ≥0) * (1 - x₀) * A.tokensPerBlock c := by gcongr
    _ ≤ ((Fintype.card L : ℝ≥0) * A.work) * (1 - x₀) * A.tokensPerBlock c := by
        gcongr
        exact_mod_cast h3
    _ = ((Fintype.card L : ℝ≥0) * (1 - x₀)) * (A.work * A.tokensPerBlock c) := by ring
    _ ≤ (B * A.unreachableCapacity c k) * c := by gcongr
    _ = ((B * c : ℕ) : ℝ≥0) * A.unreachableCapacity c k := by
        push_cast
        ring

/-- An inference block's unreachable work is at most `m(I) max_T x_T(I)`: a gate of `X_T` in it means it cuts `T`. -/
theorem unreachableWork_le (A : Annotation d L) {c k : ℕ} {I : Finset (Fin d.D.N)} (hI : InferenceBlock d c k I) :
    A.unreachableWork c k I ≤ A.cuts I * A.maxShare I := by
  classical
  have hzero : ∀ T ∈ (univ : Finset L), T ∉ univ.filter (fun T => (I ∩ A.tokenAffected T).Nonempty ∧
      ¬ A.FromInputs I T) →
      ((I ∩ (A.tokenAffected T \ A.reachable c k)).card : ℝ≥0) / (A.tokenAffected T).card = 0 := by
    intro T _ hT
    rw [mem_filter, not_and] at hT
    have hempty : I ∩ (A.tokenAffected T \ A.reachable c k) = ∅ := by
      rw [eq_empty_iff_forall_notMem]
      intro g hg
      rw [mem_inter, mem_sdiff] at hg
      obtain ⟨hgI, hgA, hgR⟩ := hg
      have hfrom : A.FromInputs I T := by
        by_contra hn
        exact hT (mem_univ T) ⟨⟨g, mem_inter.2 ⟨hgI, hgA⟩⟩, hn⟩
      exact hgR ((mem_reachable A).2 ⟨mem_of_mem_inter_right hgA, T, mem_of_mem_inter_left hgA, I, hI, hgI, hfrom⟩)
    rw [hempty, card_empty, Nat.cast_zero, zero_div]
  calc A.unreachableWork c k I
      = ∑ T ∈ univ.filter (fun T => (I ∩ A.tokenAffected T).Nonempty ∧ ¬ A.FromInputs I T),
          ((I ∩ (A.tokenAffected T \ A.reachable c k)).card : ℝ≥0) / (A.tokenAffected T).card :=
        (sum_subset (subset_univ _) hzero).symm
    _ ≤ (univ.filter (fun T => (I ∩ A.tokenAffected T).Nonempty ∧ ¬ A.FromInputs I T)).card • A.maxShare I := by
        refine sum_le_card_nsmul _ _ _ fun T _ => (le_trans ?_ (share_le_maxShare A T I))
        unfold Annotation.share
        gcongr
        exact sdiff_subset
    _ = A.cuts I * A.maxShare I := by
        rw [nsmul_eq_mul]
        rfl

namespace SecurityProofs

theorem Counting : Guarantees.Counting := by
  intro d c k S B P hS hP
  have hle : S.card ≤ B * maxInBlock d c k S := by
    rw [← sum_card_inter_gates P hS]
    calc ∑ u, (P.gates u ∩ S).card ≤ ∑ _u : Fin B, maxInBlock d c k S :=
          sum_le_sum fun u _ => card_le_maxInBlock (inferenceBlock_gates hP u)
      _ = B * maxInBlock d c k S := by simp
  exact counting_le hle (card_le_card hS)

theorem Slowdown : Guarantees.Slowdown := fun d _ A c k B P hP =>
  Counting d c k (A.affected \ A.reachable c k) B P (sdiff_subset.trans (affected_subset A)) hP

theorem TokenForm : Guarantees.TokenForm := by
  intro d L _ A c k α x₀ B P hid hR hP
  exact div_le_partitionSlowdown (token_core A hid hR hP) fun hC => by simp [Annotation.tokenShare, hC]

theorem TokenFormVarying : Guarantees.TokenFormVarying := by
  intro d L _ A c k x₀ B P hR hP
  have hsum := sum_card_unreachable_le A hP (β := A.maxAffected)
    fun T => le_sup (f := fun T => (A.tokenAffected T).card) (mem_univ T)
  have hcover : (A.affected ∩ A.tokenGates).card ≤ ∑ T, (A.tokenAffected T).card := by
    calc (A.affected ∩ A.tokenGates).card ≤ (univ.biUnion A.tokenAffected).card := by
          apply card_le_card
          intro g hg
          rw [mem_inter, Annotation.tokenGates, mem_biUnion] at hg
          obtain ⟨hgA, T, -, hgT⟩ := hg
          exact mem_biUnion.2 ⟨T, mem_univ T, mem_inter.2 ⟨hgT, hgA⟩⟩
      _ ≤ ∑ T, (A.tokenAffected T).card := card_biUnion_le
  have hq : (1 - x₀) * ((A.affected ∩ A.tokenGates).card : ℝ≥0) ≤
      B * A.maxAffected * A.unreachableCapacity c k := by
    calc (1 - x₀) * ((A.affected ∩ A.tokenGates).card : ℝ≥0)
        ≤ (1 - x₀) * ∑ T, ((A.tokenAffected T).card : ℝ≥0) := by
          gcongr
          exact_mod_cast hcover
      _ = ∑ T, ((A.tokenAffected T).card : ℝ≥0) * (1 - x₀) := by
          rw [mul_sum]
          exact sum_congr rfl fun T _ => mul_comm _ _
      _ ≤ ∑ T, ((A.tokenAffected T \ A.reachable c k).card : ℝ≥0) :=
          sum_le_sum fun T _ => card_unreachable_ge A T (hR T)
      _ ≤ _ := hsum
  refine div_le_partitionSlowdown ?_ fun hC => ?_
  · calc (1 - x₀) * ((A.affected ∩ A.tokenGates).card : ℝ≥0) * c * d.D.computed.card
        = ((1 - x₀) * ((A.affected ∩ A.tokenGates).card : ℝ≥0)) * (c * d.D.computed.card) := by ring
      _ ≤ (B * A.maxAffected * A.unreachableCapacity c k) * (c * d.D.computed.card) := by gcongr
      _ = ((B * c : ℕ) : ℝ≥0) * (d.D.computed.card * A.maxAffected * A.unreachableCapacity c k) := by
          push_cast
          ring
  · have h0 : (A.affected ∩ A.tokenGates).card = 0 :=
      Nat.eq_zero_of_le_zero (hC ▸ card_le_card (inter_subset_left.trans (affected_subset A)))
    rw [h0, Nat.cast_zero, mul_zero, zero_mul]

theorem CapacityBound : Guarantees.CapacityBound := by
  intro d L _ A c k u a M hcut hpeak
  obtain ⟨I, hI, hsup⟩ := Finset.exists_mem_eq_sup (univ.filter (InferenceBlock d c k))
    ⟨∅, mem_filter.2 ⟨mem_univ _, inferenceBlock_empty c k⟩⟩ (A.unreachableWork c k)
  have hIB : InferenceBlock d c k I := (mem_filter.1 hI).2
  rw [Annotation.unreachableCapacity, hsup]
  have hk : ((computedReads d I).card : ℝ≥0) ≤ k := by
    exact_mod_cast (card_le_card (computedReads_subset_unregistered I)).trans hIB.2.2
  have hm : a * A.cuts I ≤ (k : ℝ≥0) - u (A.maxShare I) := le_tsub_of_add_le_left ((hcut I hIB).trans hk)
  calc a * A.unreachableWork c k I ≤ a * (A.cuts I * A.maxShare I) := by
        gcongr
        exact unreachableWork_le A hIB
    _ = A.maxShare I * (a * A.cuts I) := by ring
    _ ≤ A.maxShare I * ((k : ℝ≥0) - u (A.maxShare I)) := by gcongr
    _ ≤ M := hpeak _ (maxShare_le_one A I)

/-- A block that holds a gate and reads nothing of `S` from outside holds every gate that feeds it within `S`. -/
theorem mem_of_feedsWithin {d : Declaration} {I S : Finset (Fin d.D.N)} (hquiet : ∀ v ∈ d.D.into I, v ∉ S)
    {h g : Fin d.D.N} (hfw : FeedsWithin d.D S h g) : g ∈ I → h ∈ I := by
  induction hfw with
  | refl _ => exact id
  | @step h h' g hr hS _ ih =>
    intro hg
    by_contra hh
    refine hquiet h ?_ hS
    simp only [Verity.Circuit.into, mem_filter, mem_univ, true_and]
    exact ⟨hh, h', ih hg, hr⟩

theorem ReachableShareBound : Guarantees.ReachableShareBound := by
  classical
  intro d L A c k u_in ξ hA hin hlim T
  unfold Annotation.reachableShare Annotation.dependencyProfile
  refine div_le_div_of_nonneg_right ?_ (zero_le)
  refine Nat.cast_le.2 (card_le_card fun g hg => ?_)
  rw [mem_inter] at hg
  obtain ⟨hgA, hgR⟩ := hg
  rw [mem_filter]
  refine ⟨hgA, ?_⟩
  obtain ⟨-, T', hgT', I, hI, hgI, hfrom⟩ := (mem_reachable A).1 hgR
  obtain rfl : T' = T := by
    by_contra hne
    exact disjoint_left.1 (hA.disjoint hne) hgT' (mem_of_mem_inter_left hgA)
  have hsub : A.dependencies T' g ⊆ I ∩ A.tokenAffected T' := by
    intro h hh
    rw [Annotation.dependencies, mem_filter] at hh
    exact mem_inter.2 ⟨mem_of_feedsWithin hfrom hh.2 hgI, hh.1⟩
  have hshare : A.share T' I ≤ ξ :=
    hlim _ (share_le_one A T' I) ((hin I hI T' hfrom).trans (by exact_mod_cast hI.2.2))
  refine le_trans ?_ hshare
  unfold Annotation.dependencyShare Annotation.share
  exact div_le_div_of_nonneg_right (Nat.cast_le.2 (card_le_card hsub)) (zero_le)

/-- **The claim's core**: under A1 and A1′, identical token blocks, `σ (1 - x₀) s a |C| ≤ B c M`. -/
theorem claim_core {d : Declaration} {L : Type} [Fintype L] (A : Annotation d L) (hA : A.Valid) {c k α : ℕ}
    {u u_in : ℝ≥0 → ℝ≥0} {a M ξ x₀ : ℝ≥0} {B : ℕ} {P : Partition d.D B} (hid : A.Identical α)
    (hcut : CutCost d A c k u a) (hpeak : PeakBound u k M) (hin : InputCost d A c k u_in)
    (hlim : ReachLimit u_in k ξ) (hprof : A.ProfileAtMost ξ x₀) (hP : P.IsCK d c k) :
    A.tokenShare * (1 - x₀) * A.tokensPerBlock c * a * d.D.computed.card ≤ ((B * c : ℕ) : ℝ≥0) * M := by
  have hR : A.ReachableAtMost c k x₀ := fun T =>
    (ReachableShareBound d L A c k u_in ξ hA hin hlim T).trans (hprof T)
  have h1 := token_core A hid hR hP
  have h2 := CapacityBound d L A c k u a M hcut hpeak
  calc A.tokenShare * (1 - x₀) * A.tokensPerBlock c * a * d.D.computed.card
      = a * (A.tokenShare * (1 - x₀) * A.tokensPerBlock c * d.D.computed.card) := by ring
    _ ≤ a * (((B * c : ℕ) : ℝ≥0) * A.unreachableCapacity c k) := by gcongr
    _ = ((B * c : ℕ) : ℝ≥0) * (a * A.unreachableCapacity c k) := by ring
    _ ≤ ((B * c : ℕ) : ℝ≥0) * M := by gcongr

theorem SlowdownClaim : Guarantees.SlowdownClaim := by
  intro d L _ A c k α u u_in a M ξ x₀ B P hA hid hcut hpeak hin hlim hprof hP
  exact div_le_partitionSlowdown (claim_core A hA hid hcut hpeak hin hlim hprof hP)
    fun hC => by simp [Annotation.tokenShare, hC]

/-- An evenly spread update peaks at `k² / (4 u(1))`. -/
theorem peak_even {u₁ : ℝ≥0} (hu : 0 < u₁) (k : ℕ) : PeakBound (fun x => x * u₁) k ((k : ℝ≥0) ^ 2 / (4 * u₁)) := by
  intro x _
  show x * ((k : ℝ≥0) - x * u₁) ≤ (k : ℝ≥0) ^ 2 / (4 * u₁)
  rw [le_div_iff₀ (by positivity)]
  rcases le_total (x * u₁) (k : ℝ≥0) with h | h
  · rw [← NNReal.coe_le_coe]
    push_cast [NNReal.coe_sub h]
    nlinarith [sq_nonneg ((k : ℝ) - 2 * u₁ * x)]
  · rw [tsub_eq_zero_of_le h, mul_zero, zero_mul]
    exact zero_le

/-- An evenly spread update reaches within `k` up to the share `k / u(1)`. -/
theorem reach_even {u₁ : ℝ≥0} (hu : 0 < u₁) (k : ℕ) : ReachLimit (fun x => x * u₁) k ((k : ℝ≥0) / u₁) :=
  fun _ _ hx => (le_div_iff₀ hu).2 hx

theorem SlowdownEven : Guarantees.SlowdownEven := by
  intro d L _ A c k α u₁ a x₀ B P hA hid hcut hin hu hprof hP
  have h := claim_core A hA hid hcut (peak_even hu k) hin (reach_even hu k) hprof hP
  refine div_le_partitionSlowdown ?_ fun hC => by simp [Annotation.tokenShare, hC]
  have h4 : (4 * u₁) * ((k : ℝ≥0) ^ 2 / (4 * u₁)) = (k : ℝ≥0) ^ 2 := mul_div_cancel₀ _ (by positivity)
  calc 4 * A.tokenShare * (1 - x₀) * A.tokensPerBlock c * a * u₁ * d.D.computed.card
      = (4 * u₁) * (A.tokenShare * (1 - x₀) * A.tokensPerBlock c * a * d.D.computed.card) := by ring
    _ ≤ (4 * u₁) * (((B * c : ℕ) : ℝ≥0) * ((k : ℝ≥0) ^ 2 / (4 * u₁))) := by gcongr
    _ = ((B * c : ℕ) : ℝ≥0) * ((4 * u₁) * ((k : ℝ≥0) ^ 2 / (4 * u₁))) := by ring
    _ = ((B * c : ℕ) : ℝ≥0) * (k : ℝ≥0) ^ 2 := by rw [h4]

/-- Frozen layers above the share `t` put the peak at most `t k`. -/
theorem peak_frozen {u : ℝ≥0 → ℝ≥0} {k : ℕ} {t : ℝ≥0} (h : FrozenShare u k t) : PeakBound u k (t * k) := by
  intro x hx
  rcases le_or_gt x t with hxt | hxt
  · exact mul_le_mul' hxt tsub_le_self
  · rw [tsub_eq_zero_of_le (h x hxt hx).le, mul_zero]
    exact zero_le

theorem SlowdownFrozen : Guarantees.SlowdownFrozen := by
  intro d L _ A c k α u u_in a t ξ x₀ B P hA hid hcut hfrozen hin hlim hprof hP
  exact div_le_partitionSlowdown (claim_core A hA hid hcut (peak_frozen hfrozen) hin hlim hprof hP)
    fun hC => by simp [Annotation.tokenShare, hC]

theorem Security : Guarantees.Security := by
  intro methods dec ℓ ℓstar D p r c k n N α u u_in a M ξ x₀ B P hfloor hamong hA hwithin hreach hid hcut hpeak hin
    hlim hprof hP hB
  have hclaim := SlowdownClaim r.d (Fin r.numTokens) r.A c k α u u_in a M ξ x₀ B P hA hid hcut hpeak hin hlim hprof hP
  unfold partitionSlowdown at hclaim
  set X : ℝ≥0∞ := ((r.A.tokenShare * (1 - x₀) * r.A.tokensPerBlock c * a : ℝ≥0) : ℝ≥0∞) / (M : ℝ≥0∞)
  have hXC : X * (r.d.D.computed.card : ℝ≥0∞) ≤ ((B * c : ℕ) : ℝ≥0∞) := by
    by_cases hC : (r.d.D.computed.card : ℝ≥0∞) = 0
    · rw [hC, mul_zero]
      exact zero_le
    · exact (ENNReal.le_div_iff_mul_le (Or.inl hC) (Or.inl (ENNReal.natCast_ne_top _))).1 hclaim
  have hBc : ((B * c : ℕ) : ℝ≥0∞) ≤ ((N * n * c : ℕ) : ℝ≥0∞) := by
    exact_mod_cast Nat.mul_le_mul_right c hB
  have hfl := hfloor r hamong hA hwithin hreach
  rw [mul_assoc]
  calc X * (ENNReal.ofReal p * honestCost methods ℓ ℓstar) ≤ X * (r.d.D.computed.card : ℝ≥0∞) := by gcongr
    _ ≤ ((B * c : ℕ) : ℝ≥0∞) := hXC
    _ ≤ ((N * n * c : ℕ) : ℝ≥0∞) := hBc

end SecurityProofs

end Nci.Formalism
