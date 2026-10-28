import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Powerset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers
import Proofs.Pous.Thm1.Theorem1iTimed

/-!
# Report Theorem 1(ii), sequential: `PousProofs.theorem1_ii_seq`, as drafted

For a history `h` of fewer than `k` indices, `st h` is the state the responder reaches after answering `h`, and
round `|h|` predicts block `i` by `step |h| (st h) i`, a `(D, Q)`-bounded program. `E` is the event that, for
some history, the wrong predictions fit: `S + B + #wrong·ℓ ≤ β`.

1. Off `E`, every round is answered correctly with probability at most `p₀` given any history
   (`pr_good_le_p₀`). The chain rule over the `k` rounds (`pr_seq_le`) gives `p₀^k`, and conditioning on the
   setup (`pr_prod_le`) gives `Pr[accept] ≤ Pr[E] + p₀^k`.
2. For each fixed history, the patch compressor (`pr_fits_le`: the state and the wrong blocks, a `β`-bit code
   by counting) against `TimedINC` bounds the event by `π`. A union bound over the `B^j` histories of each
   length `j < k` (`pr_exists_le`) gives `Pr[E] ≤ (Σ_{j<k} B^j)·π`.
-/

namespace PousThm1

open Pous
open scoped ENNReal

/-! ### Uniform probability -/

theorem pr_le_one {α : Type} [Fintype α] (P : α → Prop) : pr P ≤ 1 := by
  classical
  rw [pr_def]
  refine ENNReal.div_le_of_le_mul ?_
  rw [one_mul]
  exact_mod_cast (Finset.card_filter_le _ _).trans (Finset.card_univ).le

/-- **Union bound.** -/
theorem pr_exists_le {α ι : Type} [Fintype α] (s : Finset ι) (P : ι → α → Prop) :
    pr (fun a => ∃ x ∈ s, P x a) ≤ ∑ x ∈ s, pr (P x) := by
  classical
  simp only [pr_def, div_eq_mul_inv]
  rw [← Finset.sum_mul]
  gcongr
  have : (Finset.univ.filter fun a => ∃ x ∈ s, P x a) = s.biUnion fun x => Finset.univ.filter (P x) := by
    ext a; simp
  rw [this]
  exact_mod_cast Finset.card_biUnion_le

/-- **Conditioning on the first coordinate.** If every fibre off `E` has probability at most `c`, the product event
has probability at most `Pr[E] + c`. -/
theorem pr_prod_le {α β : Type} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β]
    (P : α → β → Prop) (E : α → Prop) (c : ℝ≥0∞) (h : ∀ a, ¬ E a → pr (P a) ≤ c) :
    pr (fun x : α × β => P x.1 x.2) ≤ pr E + c := by
  classical
  have hα0 : (Fintype.card α : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hβ0 : (Fintype.card β : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hαt : (Fintype.card α : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hβt : (Fintype.card β : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hfib : ∀ a, ((Finset.univ.filter (P a)).card : ℝ≥0∞) ≤
      (if E a then 1 else 0) * Fintype.card β + c * Fintype.card β := by
    intro a
    by_cases ha : E a
    · simp only [ha, ite_true, one_mul]
      refine le_add_right ?_
      exact_mod_cast (Finset.card_filter_le _ _).trans (Finset.card_univ).le
    · simp only [ha, ite_false, zero_mul, zero_add]
      have := h a ha
      rwa [pr_def, ENNReal.div_le_iff hβ0 hβt] at this
  have hcount : ((Finset.univ.filter fun x : α × β => P x.1 x.2).card : ℝ≥0∞) =
      ∑ a, ((Finset.univ.filter (P a)).card : ℝ≥0∞) := by
    rw [Finset.natCast_card_filter, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [Finset.natCast_card_filter]
  have hE : ((Finset.univ.filter E).card : ℝ≥0∞) = ∑ a, (if E a then 1 else 0 : ℝ≥0∞) :=
    Finset.natCast_card_filter _ _
  rw [pr_def, pr_def, hcount, Fintype.card_prod, Nat.cast_mul, ENNReal.div_le_iff (mul_ne_zero hα0 hβ0)
    (ENNReal.mul_ne_top hαt hβt)]
  calc ∑ a, ((Finset.univ.filter (P a)).card : ℝ≥0∞)
      ≤ ∑ a, ((if E a then 1 else 0) * Fintype.card β + c * Fintype.card β) :=
        Finset.sum_le_sum fun a _ => hfib a
    _ = ((Finset.univ.filter E).card : ℝ≥0∞) * Fintype.card β +
          Fintype.card α * (c * Fintype.card β) := by
        rw [Finset.sum_add_distrib, ← Finset.sum_mul, ← hE, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    _ = (((Finset.univ.filter E).card : ℝ≥0∞) / Fintype.card α + c) *
          (Fintype.card α * Fintype.card β) := by
        rw [add_mul, ← mul_assoc (_ / (Fintype.card α : ℝ≥0∞)), ENNReal.div_mul_cancel hα0 hαt]
        ring

/-- **Chain rule, as a bound.** A first-coordinate event of probability at most `c₁`, then a second-coordinate
event of conditional probability at most `c₂`. -/
theorem pr_and_le {α β : Type} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β]
    (P : α → Prop) (R : α → β → Prop) (c₁ c₂ : ℝ≥0∞) (h₁ : pr P ≤ c₁) (h₂ : ∀ a, pr (R a) ≤ c₂) :
    pr (fun x : α × β => P x.1 ∧ R x.1 x.2) ≤ c₁ * c₂ := by
  classical
  have hα0 : (Fintype.card α : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hβ0 : (Fintype.card β : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hαt : (Fintype.card α : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hβt : (Fintype.card β : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hfib : ∀ a, ((Finset.univ.filter fun b => P a ∧ R a b).card : ℝ≥0∞) ≤
      (if P a then 1 else 0) * (c₂ * Fintype.card β) := by
    intro a
    by_cases ha : P a
    · simp only [ha, true_and, ite_true, one_mul]
      have := h₂ a
      rwa [pr_def, ENNReal.div_le_iff hβ0 hβt] at this
    · simp [ha]
  have hcount : ((Finset.univ.filter fun x : α × β => P x.1 ∧ R x.1 x.2).card : ℝ≥0∞) =
      ∑ a, ((Finset.univ.filter fun b => P a ∧ R a b).card : ℝ≥0∞) := by
    rw [Finset.natCast_card_filter, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [Finset.natCast_card_filter]
  have hP : ((Finset.univ.filter P).card : ℝ≥0∞) = ∑ a, (if P a then 1 else 0 : ℝ≥0∞) :=
    Finset.natCast_card_filter _ _
  have h1 := h₁
  rw [pr_def, ENNReal.div_le_iff hα0 hαt] at h1
  rw [pr_def, hcount, Fintype.card_prod, Nat.cast_mul, ENNReal.div_le_iff (mul_ne_zero hα0 hβ0)
    (ENNReal.mul_ne_top hαt hβt)]
  calc ∑ a, ((Finset.univ.filter fun b => P a ∧ R a b).card : ℝ≥0∞)
      ≤ ∑ a, (if P a then 1 else 0) * (c₂ * Fintype.card β) := Finset.sum_le_sum fun a _ => hfib a
    _ = ((Finset.univ.filter P).card : ℝ≥0∞) * (c₂ * Fintype.card β) := by rw [← Finset.sum_mul, ← hP]
    _ ≤ (c₁ * Fintype.card α) * (c₂ * Fintype.card β) := by gcongr
    _ = c₁ * c₂ * (Fintype.card α * Fintype.card β) := by ring

/-! ### The threshold `p₀` -/

/-- If `g` of `B` blocks are right and the other `b` do not fit, `S + B + b·ℓ > β`, then `g ≤ p₀·B`. -/
theorem good_le_p₀ (B ℓ β S g b : ℕ) (hB : 0 < B) (hgb : g + b = B) (h : β < S + B + b * ℓ) :
    (g : ℝ) ≤ p₀ β S B ℓ * B := by
  unfold p₀
  rcases Nat.eq_zero_or_pos ℓ with rfl | hℓ
  · simp only [Nat.cast_zero, mul_zero, div_zero, sub_zero, one_mul]
    exact_mod_cast (by omega : g ≤ B)
  · have hB' : (0 : ℝ) < B := by exact_mod_cast hB
    have hℓ' : (0 : ℝ) < ℓ := by exact_mod_cast hℓ
    have hgb' : (g : ℝ) + b = B := by exact_mod_cast hgb
    have h' : (β : ℝ) < S + B + b * ℓ := by exact_mod_cast h
    have key : (1 - ((β : ℝ) - S - B) / (B * ℓ)) * B = B - ((β : ℝ) - S - B) / ℓ := by
      field_simp
    rw [key]
    have : ((β : ℝ) - S - B) / ℓ < b := by
      rw [div_lt_iff₀ hℓ']; linarith
    linarith

open Classical in
/-- One block: if the wrong blocks do not fit, a uniform block is right with probability at most `p₀`. -/
theorem pr_good_le_p₀ (B ℓ β S : ℕ) (hB : 0 < B) (good : Fin B → Prop)
    (h : ¬ S + B + (Finset.univ.filter fun i => ¬ good i).card * ℓ ≤ β) :
    pr good ≤ ENNReal.ofReal (p₀ β S B ℓ) := by
  have hgb := Finset.card_filter_add_card_filter_not (s := Finset.univ) good
  rw [Finset.card_univ, Fintype.card_fin] at hgb
  have hr := good_le_p₀ B ℓ β S _ _ hB hgb (by omega)
  have hB0 : (B : ℝ≥0∞) ≠ 0 := by exact_mod_cast hB.ne'
  rw [pr_def, Fintype.card_fin, ENNReal.div_le_iff hB0 (ENNReal.natCast_ne_top _)]
  calc ((Finset.univ.filter good).card : ℝ≥0∞)
      = ENNReal.ofReal ((Finset.univ.filter good).card : ℝ) := (ENNReal.ofReal_natCast _).symm
    _ ≤ ENNReal.ofReal (p₀ β S B ℓ * B) := ENNReal.ofReal_le_ofReal hr
    _ = ENNReal.ofReal (p₀ β S B ℓ) * B := by
        rw [ENNReal.ofReal_mul' (Nat.cast_nonneg _), ENNReal.ofReal_natCast]

/-! ### Programs -/

section progMap

variable {M : IdealModel} {α β : Type} (g : α → β) (f : M.Qry → M.Ans)

end progMap

/-! ### Compression against `TimedINC` -/

/-- **Codes from counting.** A set of at most `2^β` points has a `β`-bit code that decodes every point of it. -/
theorem exists_code {X : Type} [Fintype X] [Nonempty X] (T : Finset X) (β : ℕ) (h : T.card ≤ 2 ^ β) :
    ∃ (c : X → Bits β) (d : Bits β → X), ∀ x ∈ T, d (c x) = x := by
  classical
  obtain ⟨e⟩ := Function.Embedding.nonempty_of_card_le (α := T) (β := Bits β) (by simpa using h)
  refine ⟨fun x => if hx : x ∈ T then e ⟨x, hx⟩ else fun _ => false,
    fun w => if hw : ∃ y, e y = w then hw.choose.1 else Classical.arbitrary X, fun x hx => ?_⟩
  have hex : ∃ y, e y = e ⟨x, hx⟩ := ⟨_, rfl⟩
  simp only [hx, dite_true, hex]
  exact congrArg Subtype.val (e.injective hex.choose_spec)

/-- **Counting patches.** Assignments of an optional `ℓ`-bit block to each of `B` positions, with at most `L` bits
of blocks in total: at most `2^B · 2^L`. -/
theorem card_patches_le (B ℓ L : ℕ) :
    (Finset.univ.filter fun g : Fin B → Option (Bits ℓ) =>
      (Finset.univ.filter fun i => g i ≠ none).card * ℓ ≤ L).card ≤ 2 ^ B * 2 ^ L := by
  classical
  let t : Finset (Fin B) → (i : Fin B) → Finset (Option (Bits ℓ)) := fun D i =>
    if i ∈ D then Finset.univ.erase none else {none}
  have hsub : (Finset.univ.filter fun g : Fin B → Option (Bits ℓ) =>
      (Finset.univ.filter fun i => g i ≠ none).card * ℓ ≤ L) ⊆
      (Finset.univ.filter fun D : Finset (Fin B) => D.card * ℓ ≤ L).biUnion
        fun D => Fintype.piFinset (t D) := by
    intro g hg
    rw [Finset.mem_filter] at hg
    rw [Finset.mem_biUnion]
    refine ⟨Finset.univ.filter fun i => g i ≠ none, by simpa using hg.2, ?_⟩
    rw [Fintype.mem_piFinset]
    intro i
    by_cases hi : g i = none <;> simp [t, hi]
  refine (Finset.card_le_card hsub).trans ((Finset.card_biUnion_le).trans ?_)
  have hD : ∀ D ∈ (Finset.univ.filter fun D : Finset (Fin B) => D.card * ℓ ≤ L),
      (Fintype.piFinset (t D)).card ≤ 2 ^ L := by
    intro D hD
    rw [Finset.mem_filter] at hD
    rw [Fintype.card_piFinset]
    have : ∀ i, (t D i).card = if i ∈ D then 2 ^ ℓ else 1 := by
      intro i
      by_cases hi : i ∈ D
      · simp [t, hi, Finset.card_erase_of_mem, Fintype.card_option]
      · simp [t, hi]
    simp_rw [this]
    rw [Finset.prod_ite_mem, Finset.univ_inter, Finset.prod_const, ← pow_mul]
    exact Nat.pow_le_pow_right (by norm_num) (by rw [mul_comm]; exact hD.2)
  refine (Finset.sum_le_card_nsmul _ _ _ hD).trans ?_
  rw [smul_eq_mul]
  refine Nat.mul_le_mul_right _ ?_
  refine (Finset.card_filter_le _ _).trans ?_
  rw [Finset.card_univ, Fintype.card_finset, Fintype.card_fin]

/-- The compressor's message: the responder's state, and a replacement for each block it gets wrong. -/
abbrev Desc (S B ℓ : ℕ) : Type := Bits S × (Fin B → Option (Bits ℓ))

/-- The messages that fit in `β` bits: `S` for the state, `B` for the set of replaced blocks, `ℓ` per replaced
block. -/
def fits (S B ℓ β : ℕ) : Finset (Desc S B ℓ) :=
  Finset.univ.filter fun x => S + B + (Finset.univ.filter fun i => x.2 i ≠ none).card * ℓ ≤ β

/-- At most `2^β` messages fit: `2^S` states times `2^B·2^(β−S−B)` patches (none if `β < S + B`). -/
theorem card_fits_le (S B ℓ β : ℕ) : (fits S B ℓ β).card ≤ 2 ^ β := by
  by_cases hβ : S + B ≤ β
  · have hsub : fits S B ℓ β ⊆ Finset.univ ×ˢ (Finset.univ.filter fun g : Fin B → Option (Bits ℓ) =>
        (Finset.univ.filter fun i => g i ≠ none).card * ℓ ≤ β - S - B) := by
      intro x hx
      rw [fits, Finset.mem_filter] at hx
      rw [Finset.mem_product, Finset.mem_filter]
      exact ⟨Finset.mem_univ _, Finset.mem_univ _, by omega⟩
    refine (Finset.card_le_card hsub).trans ?_
    rw [Finset.card_product, Finset.card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]
    calc 2 ^ S * _ ≤ 2 ^ S * (2 ^ B * 2 ^ (β - S - B)) := Nat.mul_le_mul_left _ (card_patches_le B ℓ _)
      _ = 2 ^ β := by rw [← pow_add, ← pow_add]; congr 1; omega
  · have : fits S B ℓ β = ∅ := by
      ext x
      rw [fits, Finset.mem_filter]
      simp only [Finset.mem_univ, true_and, Finset.notMem_empty, iff_false]
      omega
    rw [this, Finset.card_empty]
    exact Nat.zero_le _

/-- Replace every block where the prediction `P` differs from `C`. -/
def patch {B ℓ : ℕ} (P C : Fin B → Bits ℓ) : Fin B → Option (Bits ℓ) :=
  fun i => if P i = C i then none else some (C i)

/-- The patched block where there is one, else the prediction: this is `C`. -/
theorem patch_getD {B ℓ : ℕ} (P C : Fin B → Bits ℓ) (i : Fin B) : (patch P C i).getD (P i) = C i := by
  by_cases h : P i = C i <;> simp [patch, h]

/-- If the blocks where `P` differs from `C` fit (`S + B + #wrong·ℓ ≤ β`), so does the message `(σ, patch P C)`. -/
theorem mem_fits_patch {S B ℓ β : ℕ} (σ : Bits S) (P C : Fin B → Bits ℓ) {_ : DecidablePred fun i => ¬ P i = C i}
    (h : S + B + (Finset.univ.filter fun i => ¬ P i = C i).card * ℓ ≤ β) : (σ, patch P C) ∈ fits S B ℓ β := by
  rw [fits, Finset.mem_filter]
  refine ⟨Finset.mem_univ _, ?_⟩
  convert h using 4
  ext i
  by_cases hi : P i = C i <;> simp [patch, hi]

open Classical in
/-- **The timed patch compressor.** Let `st` pick an `S`-bit state from the setup (any function of it) and `pred`
predict each block from `(W, pp, state)` within `(D, Q)`. Then the wrong predictions fit in `β` bits,
`S + B + #wrong·ℓ ≤ β`, with probability at most `π` under `TimedINC D Q β π`. -/
theorem pr_fits_le {M : IdealModel} (sch : Scheme M) (D Q β S : ℕ) (π : ℝ≥0∞) (hinc : TimedINC sch D Q β π)
    (W : Bits sch.n) (st : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits S)
    (pred : Bits sch.n → Bits sch.t → Bits S → Fin sch.B → OracleProg M (Bits sch.ℓ))
    (hpred : ∀ W pp σ i, (pred W pp σ i).Bounded D Q) :
    pr (fun a : M.Ω × sch.Coins => S + sch.B + (Finset.univ.filter fun i =>
      ¬ (pred W (sch.enc W a.1 a.2).1 (st W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
        (M.answer a.1) = (sch.enc W a.1 a.2).2 i).card * sch.ℓ ≤ β) ≤ π := by
  obtain ⟨c, d, hcd⟩ := exists_code (fits S sch.B sch.ℓ β) β (card_fits_le S sch.B sch.ℓ β)
  let cmp : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits β :=
    fun W pp C ω => c (st W pp C ω, patch (fun i => (pred W pp (st W pp C ω) i).run (M.answer ω)) C)
  let exp : Bits sch.n → Bits sch.t → Bits β → Fin sch.B → OracleProg M (Bits sch.ℓ) :=
    fun W pp w i => ((d w).2 i).elim (pred W pp (d w).1 i) OracleProg.ret
  have hexp : ∀ W pp w i, (exp W pp w i).Bounded D Q := by
    intro W pp w i
    cases h : (d w).2 i with
    | none => simp only [exp, h, Option.elim]; exact hpred _ _ _ _
    | some b => simp only [exp, h, Option.elim]; exact ret_bounded _ _ _
  refine le_trans (pr_mono fun a ha => ?_) (hinc W cmp exp hexp)
  have hfit := mem_fits_patch (st W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1)
    (fun i => (pred W (sch.enc W a.1 a.2).1 (st W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
      (M.answer a.1))
    (sch.enc W a.1 a.2).2 ha
  show ∀ i, (exp W _ (cmp W _ _ a.1) i).run (M.answer a.1) = _
  intro i
  simp only [exp, cmp, hcd _ hfit]
  rw [← patch_getD (fun i => (pred W (sch.enc W a.1 a.2).1
    (st W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run (M.answer a.1)) (sch.enc W a.1 a.2).2 i]
  cases patch _ _ i <;> rfl

/-! ### Sequential responders -/

/-- The state a sequential responder reaches after answering the history `h` from round `j` in state `σ`. -/
def seqState {M : IdealModel} {ℓ S B : ℕ} (f : M.Qry → M.Ans)
    (step : ℕ → Bits S → Fin B → OracleProg M (Bits ℓ × Bits S)) : ℕ → List (Fin B) → Bits S → Bits S
  | _, [], σ => σ
  | j, i :: is, σ => seqState f step (j + 1) is ((step j σ i).run f).2

/-- **Chain rule for sequential audits.** If after every history of fewer than `k` indices a uniform next index is
answered correctly with probability at most `c`, all `k` rounds are with probability at most `c^k`. -/
theorem pr_seq_le {M : IdealModel} {ℓ S B : ℕ} [NeZero B] (f : M.Qry → M.Ans)
    (step : ℕ → Bits S → Fin B → OracleProg M (Bits ℓ × Bits S)) (C : Fin B → Bits ℓ) (c : ℝ≥0∞) :
    ∀ (k j : ℕ) (σ : Bits S),
      (∀ h : List (Fin B), h.length < k →
        pr (fun i : Fin B => ((step (j + h.length) (seqState f step j h σ) i).run f).1 = C i) ≤ c) →
      pr (fun I : Fin k → Fin B => seqAnswers f step j (List.ofFn I) σ = List.ofFn fun t => C (I t)) ≤ c ^ k
  | 0, _, _, _ => by rw [pow_zero]; exact pr_le_one _
  | k + 1, j, σ, hyp => by
      rw [← pr_equiv (Fin.consEquiv fun _ : Fin (k + 1) => Fin B :
        Fin B × (Fin k → Fin B) ≃ (Fin (k + 1) → Fin B))]
      simp only [Fin.consEquiv_apply, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, seqAnswers, List.cons.injEq]
      rw [pow_succ']
      refine pr_and_le (α := Fin B) (β := Fin k → Fin B) (fun i₀ => ((step j σ i₀).run f).1 = C i₀)
        (fun i₀ I => seqAnswers f step (j + 1) (List.ofFn fun i => I i) ((step j σ i₀).run f).2 =
          List.ofFn fun i => C (I i)) c (c ^ k) ?_ fun i₀ => ?_
      · simpa [seqState] using hyp [] (Nat.succ_pos k)
      · refine pr_seq_le f step C c k (j + 1) _ fun h hh => ?_
        have := hyp (i₀ :: h) (by simp only [List.length_cons]; omega)
        simpa [seqState, Nat.add_assoc, Nat.add_comm 1] using this

end PousThm1

namespace PousProofs

open Pous PousThm1
open scoped ENNReal

/-- Theorem 1(ii) for stateful sequential responders: one compressor per history of fewer than `k` indices, then a
union bound (the plug-in of tradeoff Theorem 1). -/
theorem theorem1_ii_seq {M : IdealModel} (sch : Scheme M) (D Q β S k : ℕ) (π : ℝ≥0∞)
    (hinc : TimedINC sch D Q β π) (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q) :
    pr (seqAccepts sch k W A₁ A₂) ≤
      (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * π + ENNReal.ofReal (p₀ β S sch.B sch.ℓ) ^ k := by
  classical
  -- the state after history `h`, and whether round `|h|` answers block `i` correctly from it
  let st : List (Fin sch.B) → Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits S :=
    fun h W pp C ω => seqState (M.answer ω) (A₂.step W pp) 0 h (A₁ W pp C ω)
  let pred : List (Fin sch.B) → Bits sch.n → Bits sch.t → Bits S → Fin sch.B → OracleProg M (Bits sch.ℓ) :=
    fun h W pp σ i => progMap Prod.fst (A₂.step W pp h.length σ i)
  let good : List (Fin sch.B) → M.Ω × sch.Coins → Fin sch.B → Prop := fun h a i =>
    (pred h W (sch.enc W a.1 a.2).1 (st h W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
      (M.answer a.1) = (sch.enc W a.1 a.2).2 i
  let E : M.Ω × sch.Coins → Prop := fun a => ∃ j ∈ Finset.range k, ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
    S + sch.B + (Finset.univ.filter fun i => ¬ good (List.ofFn v) a i).card * sch.ℓ ≤ β
  have hre : pr (seqAccepts sch k W A₁ A₂) = pr (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) =>
      seqAnswers (M.answer x.1.1) (A₂.step W (sch.enc W x.1.1 x.1.2).1) 0 (List.ofFn x.2)
        (A₁ W (sch.enc W x.1.1 x.1.2).1 (sch.enc W x.1.1 x.1.2).2 x.1.1) =
      List.ofFn fun t => (sch.enc W x.1.1 x.1.2).2 (x.2 t)) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) =>
      seqAnswers (M.answer x.1.1) (A₂.step W (sch.enc W x.1.1 x.1.2).1) 0 (List.ofFn x.2)
        (A₁ W (sch.enc W x.1.1 x.1.2).1 (sch.enc W x.1.1 x.1.2).2 x.1.1) =
      List.ofFn fun t => (sch.enc W x.1.1 x.1.2).2 (x.2 t))
  rw [hre]
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  refine (pr_prod_le (α := M.Ω × sch.Coins) (β := Fin k → Fin sch.B)
    (fun a I => seqAnswers (M.answer a.1) (A₂.step W (sch.enc W a.1 a.2).1) 0 (List.ofFn I)
        (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) =
      List.ofFn fun t => (sch.enc W a.1 a.2).2 (I t))
    E (ENNReal.ofReal (p₀ β S sch.B sch.ℓ) ^ k) fun a ha => ?_).trans (add_le_add ?_ le_rfl)
  · -- no history's wrong blocks fit: every round is right with probability at most `p₀`
    refine pr_seq_le _ _ _ _ k 0 _ fun h hh => ?_
    have hnf : ¬ S + sch.B + (Finset.univ.filter fun i => ¬ good h a i).card * sch.ℓ ≤ β := by
      intro hf
      refine ha ⟨h.length, Finset.mem_range.2 hh, h.get, Finset.mem_univ _, ?_⟩
      rwa [List.ofFn_get]
    have := pr_good_le_p₀ sch.B sch.ℓ β S sch.B_pos (good h a) (by convert hnf)
    simpa only [good, pred, st, run_progMap, Nat.zero_add] using this
  · -- one compressor per history, then a union bound
    calc pr E ≤ ∑ j ∈ Finset.range k, pr (fun a => ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
          S + sch.B + (Finset.univ.filter fun i => ¬ good (List.ofFn v) a i).card * sch.ℓ ≤ β) :=
          pr_exists_le _ _
      _ ≤ ∑ j ∈ Finset.range k, ∑ _v : Fin j → Fin sch.B, π := by
          refine Finset.sum_le_sum fun j _ => (pr_exists_le _ _).trans (Finset.sum_le_sum fun v _ => ?_)
          exact pr_fits_le sch D Q β S π hinc W (st (List.ofFn v)) (pred (List.ofFn v))
            fun W pp σ i => bounded_progMap _ (A₂.bounded _ _ _ _ _)
      _ = (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * π := by
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin,
            nsmul_eq_mul, Nat.cast_pow]

end PousProofs
