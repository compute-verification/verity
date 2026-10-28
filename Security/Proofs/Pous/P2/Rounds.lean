import Definitions.Pous.Assumptions.P2
import Proofs.Pous.Sanity

/-!
# From the per-round conjecture to the graded bound and to today's whole-audit form

`P2RingRoundSecure p ε₀ D Q L` bounds each round at a boundary held at most `S₀` bits by `1 − p`, outside an event
`bad` of mass at most `k·ε₀`. Chaining it over the `k` rounds (`pr_chain`; a round at a boundary above `S₀` still
never raises acceptance, `varPassesOn_of_succ`) gives `(1 − p)^m + k·ε₀` with `m` the low boundaries (`graded`). A
`SeqResponderOn` of today's game is a `VarResponderOn` with the constant state size (`ofSeq`), accepted exactly when
all its answers are correct (`varPassesOn_ofSeq_iff`), so at `2^19` blocks and `k = 105` the graded bound is
`P2RingAuditSecure`'s once `(1 − p)^105 + 105·ε₀ ≤ 1% + 2^-129 + 2^-130` (`auditSecure_of_roundSecure`), which
`p = 0.043` and `ε₀ = 2^-137` meet (`floor_043_eps`).
-/

namespace PousP2Rounds

open Pous Pous.P2.Assumptions
open scoped ENNReal

section Prob

variable {α : Type} [Fintype α]

theorem pr_le_one' (P : α → Prop) : pr P ≤ 1 := by
  classical
  rw [pr_def]
  exact ENNReal.div_le_of_le_mul (by rw [one_mul]; exact_mod_cast Finset.card_le_univ _)

/-- An event is at most its part outside `Q` and `Q`. -/
theorem pr_le_and_not_add (P Q : α → Prop) : pr P ≤ pr (fun x => P x ∧ ¬ Q x) + pr Q := by
  classical
  rw [pr_def, pr_def, pr_def, ENNReal.div_add_div_same]
  gcongr
  have hsub : Finset.univ.filter P ⊆ (Finset.univ.filter fun x => P x ∧ ¬ Q x) ∪ Finset.univ.filter Q :=
    fun x hx => by by_cases hq : Q x <;> simp_all
  exact_mod_cast (Finset.card_le_card hsub).trans (Finset.card_union_le _ _)

theorem card_filter_range_succ (low : ℕ → Prop) [DecidablePred low] (j : ℕ) :
    ((Finset.range (j + 1)).filter low).card = ((Finset.range j).filter low).card + if low j then 1 else 0 := by
  rw [Finset.range_add_one, Finset.filter_insert]
  split_ifs with h
  · rw [Finset.card_insert_of_notMem (by simp)]
  · rfl

/-- **Chaining rounds**: events `G j` shrinking in `j`, each step a factor `q` at a low round, are at most
`q^(number of low rounds)`. -/
theorem pr_chain (G : ℕ → α → Prop) (low : ℕ → Prop) [DecidablePred low] (q : ℝ≥0∞) (k : ℕ)
    (hmono : ∀ j x, G (j + 1) x → G j x) (hstep : ∀ j < k, low j → pr (G (j + 1)) ≤ q * pr (G j)) :
    ∀ j ≤ k, pr (G j) ≤ q ^ ((Finset.range j).filter low).card
  | 0, _ => by simpa using pr_le_one' (G 0)
  | j + 1, hj => by
      have ih := pr_chain G low q k hmono hstep j (by omega)
      rw [card_filter_range_succ]
      split_ifs with h
      · rw [pow_succ, mul_comm]
        exact (hstep j (by omega) h).trans (mul_le_mul' le_rfl ih)
      · simpa using (pr_mono (hmono j)).trans ih

end Prob

section Game

variable {B w p : ℕ} [NeZero p]

theorem varPassesOn_zero (sch : Scheme (ringRO B w p)) {Sz : ℕ → ℕ} {D Q L k : ℕ} (W : Bits sch.n)
    (A₁ : Preprocessor sch (Sz 0)) (A₂ : VarResponderOn sch Sz D Q L) (x : Outcome sch k) :
    varPassesOn sch k 0 W A₁ A₂ x := by
  simp [varPassesOn]

/-- One more correct answer is a stronger event. -/
theorem varPassesOn_of_succ (sch : Scheme (ringRO B w p)) {Sz : ℕ → ℕ} {D Q L k j : ℕ} (W : Bits sch.n)
    (A₁ : Preprocessor sch (Sz 0)) (A₂ : VarResponderOn sch Sz D Q L) (x : Outcome sch k)
    (h : varPassesOn sch k (j + 1) W A₁ A₂ x) : varPassesOn sch k j W A₁ A₂ x := by
  have h' := congrArg (List.take j) h
  simp only [List.take_take, Nat.min_eq_left (Nat.le_succ j)] at h'
  exact h'

/-- A `SeqResponderOn` is a `VarResponderOn` with the same state size at every boundary. -/
def ofSeq {sch : Scheme (ringRO B w p)} {S D Q L : ℕ} (A₂ : SeqResponderOn sch S D Q L) :
    VarResponderOn sch (fun _ => S) D Q L :=
  ⟨A₂.step, A₂.bounded⟩

theorem varAnswers_const {M : IdealModel} {ℓ S B' : ℕ} (f : M.Qry → M.Ans)
    (step : ℕ → Bits S → Fin B' → OracleProg M (Bits ℓ × Bits S)) :
    ∀ (j : ℕ) (is : List (Fin B')) (σ : Bits S),
      varAnswers (Sz := fun _ => S) f step j is σ = seqAnswers f step j is σ
  | _, [], _ => rfl
  | j, i :: is, σ => by simp only [varAnswers, seqAnswers, varAnswers_const f step (j + 1) is]

theorem length_seqAnswers {M : IdealModel} {ℓ S B' : ℕ} (f : M.Qry → M.Ans)
    (step : ℕ → Bits S → Fin B' → OracleProg M (Bits ℓ × Bits S)) :
    ∀ (j : ℕ) (is : List (Fin B')) (σ : Bits S), (seqAnswers f step j is σ).length = is.length
  | _, [], _ => rfl
  | j, i :: is, σ => by simp only [seqAnswers, List.length_cons, length_seqAnswers f step (j + 1) is]

/-- All `k` answers of `ofSeq A₂` are correct exactly when today's audit accepts `A₂`. -/
theorem varPassesOn_ofSeq_iff (sch : Scheme (ringRO B w p)) {S D Q L k : ℕ} (W : Bits sch.n)
    (A₁ : Preprocessor sch S) (A₂ : SeqResponderOn sch S D Q L) (x : Outcome sch k) :
    varPassesOn sch k k W A₁ (ofSeq A₂) x ↔ seqAcceptsOn sch k W A₁ A₂ x := by
  unfold varPassesOn seqAcceptsOn ofSeq
  simp only
  rw [varAnswers_const, List.take_of_length_le (by rw [length_seqAnswers]; simp),
    List.take_of_length_le (by simp)]

theorem lowBoundaries_const (S k : ℕ) : lowBoundaries (fun _ => S) S k = k := by
  simp [lowBoundaries]

end Game

/-- **The graded bound** (`Pous.Guarantees.P2RingGradedOfRoundSecure`). -/
theorem graded : Pous.Guarantees.P2RingGradedOfRoundSecure := by
  intro p ε₀ D Q L h N _ h1 h2 Sz k W A₁ A₂
  obtain ⟨bad, hbad, hstep⟩ := h N h1 h2 Sz k W A₁ A₂
  classical
  have hc := pr_chain (fun j x => varPassesOn _ k j W A₁ A₂ x ∧ ¬ bad x)
    (fun j => Sz j ≤ (p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) (1 - p) k
    (fun j x hx => ⟨varPassesOn_of_succ _ W A₁ A₂ x hx.1, hx.2⟩) (fun j hj hlow => hstep j hj hlow) k le_rfl
  calc pr (varPassesOn _ k k W A₁ A₂)
      ≤ pr (fun x => varPassesOn _ k k W A₁ A₂ x ∧ ¬ bad x) + pr bad := pr_le_and_not_add _ _
    _ ≤ _ := add_le_add hc hbad

/-- **Today's whole-audit form from the per-round one** (`Pous.Guarantees.P2RingAuditSecureOfRoundSecure`). -/
theorem auditSecure_of_roundSecure : Pous.Guarantees.P2RingAuditSecureOfRoundSecure := by
  intro p ε₀ D Q L hnum h W A₁ A₂
  have hg := graded p ε₀ D Q L h (2 ^ 19) le_rfl (by norm_num) _ 105 W A₁ (ofSeq A₂)
  rw [lowBoundaries_const, Nat.cast_ofNat] at hg
  calc pr (seqAcceptsOn _ 105 W A₁ A₂)
      = pr (varPassesOn _ 105 105 W A₁ (ofSeq A₂)) :=
        congrArg pr (funext fun x => propext (varPassesOn_ofSeq_iff _ W A₁ A₂ x).symm)
    _ ≤ _ := hg
    _ ≤ _ := hnum

/-- **`p = 0.043` meets the derivation's condition** with up to `2^-130` of bad-event mass: `0.957^105 ≈ 0.0099 ≤ 1%`. -/
theorem one_sub_043 : (1 - 43 / 1000 : ℝ≥0∞) = ENNReal.ofReal (957 / 1000) := by
  rw [ENNReal.ofReal_div_of_pos (by norm_num), ENNReal.ofReal_ofNat, ENNReal.ofReal_ofNat]
  refine ENNReal.sub_eq_of_eq_add (by simp [ENNReal.div_eq_top]) ?_
  rw [ENNReal.div_add_div_same, show (957 + 43 : ℝ≥0∞) = 1000 by norm_num]
  exact (ENNReal.div_self (by norm_num) (by norm_num)).symm

theorem floor_043 (e : ℝ≥0∞) (he : e ≤ (2⁻¹ : ℝ≥0∞) ^ 130) :
    (1 - 43 / 1000 : ℝ≥0∞) ^ 105 + e ≤ ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) := by
  have h2 : (1 - 43 / 1000 : ℝ≥0∞) ^ 105 ≤ ENNReal.ofReal Params.δ := by
    rw [one_sub_043, ← ENNReal.ofReal_pow (by norm_num)]
    exact ENNReal.ofReal_le_ofReal (by norm_num [Params.δ])
  exact add_le_add h2 (he.trans le_add_self)

/-- **`p = 0.043` and `ε₀ = 2^-137` meet the derivation's condition**: `105·2^-137 < 128·2^-137 = 2^-130`. -/
theorem floor_043_eps :
    (1 - 43 / 1000 : ℝ≥0∞) ^ 105 + 105 * (2⁻¹ : ℝ≥0∞) ^ 137 ≤
      ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) := by
  refine floor_043 _ ?_
  calc 105 * (2⁻¹ : ℝ≥0∞) ^ 137 ≤ 2 ^ 7 * (2⁻¹ : ℝ≥0∞) ^ 137 := by gcongr; norm_num
    _ = (2⁻¹ : ℝ≥0∞) ^ 130 := by
      rw [show (137 : ℕ) = 7 + 130 by norm_num, pow_add, ← mul_assoc, ← mul_pow,
        ENNReal.mul_inv_cancel (by norm_num) (by norm_num), one_pow, one_mul]

theorem half_pow (n : ℕ) : (2⁻¹ : ℝ≥0∞) ^ n = ENNReal.ofReal ((1 / 2) ^ n) := by
  rw [ENNReal.ofReal_pow (by norm_num), one_div, ENNReal.ofReal_inv_of_pos two_pos, ENNReal.ofReal_ofNat]

set_option exponentiation.threshold 320 in
/-- **`p = 0.043` and `ε₀ = 2^-137` at 320 rounds**: `0.957^320 ≈ 7.8·10^-7`, and with `320·2^-137` it stays below
`2^-20 ≈ 9.5·10^-7`. -/
theorem floor_043_eps_320 :
    (1 - 43 / 1000 : ℝ≥0∞) ^ 320 + 320 * (2⁻¹ : ℝ≥0∞) ^ 137 ≤ (2⁻¹ : ℝ≥0∞) ^ 20 := by
  rw [one_sub_043, ← ENNReal.ofReal_pow (by norm_num), half_pow, half_pow, ← ENNReal.ofReal_ofNat 320,
    ← ENNReal.ofReal_mul (by norm_num), ← ENNReal.ofReal_add (by positivity) (by positivity)]
  exact ENNReal.ofReal_le_ofReal (by norm_num)

end PousP2Rounds
