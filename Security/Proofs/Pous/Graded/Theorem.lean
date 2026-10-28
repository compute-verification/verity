import Definitions.Pous.Accounting.GradedMeets
import Proofs.Pous.Graded.Tail

/-!
# Theorem 1(ii) for the graded audit

The game is `Security/Definitions/Pous/Model/Graded.lean`. `theorem1_ii_graded` is `theorem1_ii_seq`
with the tail rule in place of the chain rule:

`Pr[accept] ≤ (Σ_{j<k} B^j)·π + P[Bin(k, c) ≥ k − t]` for any `c ≤ 1` with `p₀ ≤ c`.

A late round's program may exceed `(D, Q)`, so the compressor cannot run it. It runs `cut` instead: the round's program
stopped after `D` rounds or `Q` queries. `cut` is `(D, Q)`-bounded (`cut_bounded`), and on a round that is on time it
returns what the program returns (`cut_run`). So with `E`, `st` and the per-history compressor of `theorem1_ii_seq`
built from the cut programs, `Pr[E] ≤ (Σ_{j<k} B^j)·π` as before, and off `E` each round's cut program is right with
probability at most `p₀` given any history. When every answer is correct, a round whose cut program is wrong was late
(`misses_le_late`), so accepting with at most `t` late rounds needs at most `t` misses, which
`pr_hits_ge_le_binomial` bounds. The proof never runs a late program, so it holds for every hard bound
`(D_late, Q_late)`: `GradedResponder.bounded` only restricts the responder.

`gradedAuditSecure_zero_iff`: with no grace the graded audit is the sequential audit, for `D ≤ D_late` and
`Q ≤ Q_late`.
-/

namespace PousGraded

open Pous PousThm1
open scoped ENNReal

variable {M : IdealModel}

/-! ### A program stopped at `(D, Q)` -/

/-- `p` stopped after `D` rounds or before the round that would exceed `Q` queries, answering `dflt` there. -/
def cut {α : Type} (dflt : α) : ℕ → ℕ → OracleProg M α → OracleProg M α
  | _, _, .ret a => .ret a
  | 0, _, .round _ _ => .ret dflt
  | D + 1, Q, .round qs k =>
      if qs.length ≤ Q then .round qs fun as => cut dflt D (Q - qs.length) (k as) else .ret dflt

theorem cut_depth_work {α : Type} (dflt : α) (f : M.Qry → M.Ans) (p : OracleProg M α) :
    ∀ D Q : ℕ, (cut dflt D Q p).depth f ≤ D ∧ (cut dflt D Q p).work f ≤ Q := by
  induction p with
  | ret a => intro D Q; cases D <;> simp [cut, OracleProg.depth, OracleProg.work]
  | round qs k ih =>
      intro D Q
      cases D with
      | zero => simp [cut, OracleProg.depth, OracleProg.work]
      | succ D =>
          by_cases h : qs.length ≤ Q
          · have := ih (qs.map f) D (Q - qs.length)
            simp only [cut, h, ↓reduceIte, OracleProg.depth, OracleProg.work]
            omega
          · simp [cut, h, OracleProg.depth, OracleProg.work]

theorem cut_bounded {α : Type} (dflt : α) (D Q : ℕ) (p : OracleProg M α) : (cut dflt D Q p).Bounded D Q :=
  fun ω => cut_depth_work dflt (M.answer ω) p D Q

theorem bounded_mono {α : Type} {p : OracleProg M α} {D Q D' Q' : ℕ} (h : p.Bounded D Q) (hD : D ≤ D') (hQ : Q ≤ Q') :
    p.Bounded D' Q' :=
  fun ω => ⟨(h ω).1.trans hD, (h ω).2.trans hQ⟩

/-- On a path within `(D, Q)` the stopped program returns what the program returns. -/
theorem cut_run {α : Type} (dflt : α) (f : M.Qry → M.Ans) (p : OracleProg M α) :
    ∀ D Q : ℕ, p.depth f ≤ D → p.work f ≤ Q → (cut dflt D Q p).run f = p.run f := by
  induction p with
  | ret a => intro D Q _ _; cases D <;> simp [cut]
  | round qs k ih =>
      intro D Q hD hQ
      simp only [OracleProg.depth, OracleProg.work] at hD hQ
      cases D with
      | zero => omega
      | succ D =>
          have h : qs.length ≤ Q := by omega
          simp only [cut, h, ↓reduceIte, OracleProg.run]
          exact ih (qs.map f) D (Q - qs.length) (by omega) (by omega)

/-! ### Late rounds and misses -/

/-- When every answer is correct, the rounds whose stopped program is wrong are among the late ones. -/
theorem misses_le_late {ℓ S B : ℕ} (f : M.Qry → M.Ans) (D Q : ℕ) (dflt : Bits ℓ × Bits S)
    (step : ℕ → Bits S → Fin B → OracleProg M (Bits ℓ × Bits S)) (C : Fin B → Bits ℓ) (is : List (Fin B)) :
    ∀ (j : ℕ) (σ : Bits S), (gradedAnswers f D Q step j is σ).map Prod.fst = is.map C →
      misses (fun j σ i => ((step j σ i).run f).2) (fun j σ i => ((cut dflt D Q (step j σ i)).run f).1 = C i)
          j is σ ≤ (gradedAnswers f D Q step j is σ).countP (fun a => !a.2) := by
  induction is with
  | nil => intro j σ _; simp [misses]
  | cons i is ih =>
      intro j σ h
      simp only [gradedAnswers, List.map_cons, List.cons.injEq] at h
      have ih' := ih (j + 1) ((step j σ i).run f).2 h.2
      simp only [misses, gradedAnswers, List.countP_cons]
      by_cases hon : (step j σ i).depth f ≤ D ∧ (step j σ i).work f ≤ Q
      · have hok : ((cut dflt D Q (step j σ i)).run f).1 = C i := by
          rw [cut_run dflt f _ D Q hon.1 hon.2]; exact h.1
        simp only [hok, ↓reduceIte, decide_eq_true hon, Bool.not_true, Bool.false_eq_true]
        omega
      · simp only [decide_eq_false hon, Bool.not_false, ↓reduceIte]
        split_ifs <;> omega

theorem map_fst_gradedAnswers {ℓ S B : ℕ} (f : M.Qry → M.Ans) (D Q : ℕ)
    (step : ℕ → Bits S → Fin B → OracleProg M (Bits ℓ × Bits S)) (is : List (Fin B)) :
    ∀ (j : ℕ) (σ : Bits S), (gradedAnswers f D Q step j is σ).map Prod.fst = seqAnswers f step j is σ := by
  induction is with
  | nil => intro j σ; rfl
  | cons i is ih => intro j σ; simp only [gradedAnswers, seqAnswers, List.map_cons, ih]

theorem countP_late_eq_zero {ℓ S B : ℕ} (f : M.Qry → M.Ans) (D Q : ℕ)
    (step : ℕ → Bits S → Fin B → OracleProg M (Bits ℓ × Bits S))
    (hb : ∀ j σ i, (step j σ i).depth f ≤ D ∧ (step j σ i).work f ≤ Q) (is : List (Fin B)) :
    ∀ (j : ℕ) (σ : Bits S), (gradedAnswers f D Q step j is σ).countP (fun a => !a.2) = 0 := by
  induction is with
  | nil => intro j σ; rfl
  | cons i is ih =>
      intro j σ
      simp only [gradedAnswers, List.countP_cons, decide_eq_true (hb j σ i), Bool.not_true,
        Bool.false_eq_true, ↓reduceIte, ih, Nat.add_zero]

/-- With no late round, the stopped responder answers as the responder does. -/
theorem seqAnswers_cut {ℓ S B : ℕ} (f : M.Qry → M.Ans) (D Q : ℕ) (dflt : Bits ℓ × Bits S)
    (step : ℕ → Bits S → Fin B → OracleProg M (Bits ℓ × Bits S)) (is : List (Fin B)) :
    ∀ (j : ℕ) (σ : Bits S), (gradedAnswers f D Q step j is σ).countP (fun a => !a.2) = 0 →
      seqAnswers f (fun j σ i => cut dflt D Q (step j σ i)) j is σ = (gradedAnswers f D Q step j is σ).map Prod.fst := by
  induction is with
  | nil => intro j σ _; rfl
  | cons i is ih =>
      intro j σ h
      simp only [gradedAnswers, List.countP_cons] at h
      have hon : (step j σ i).depth f ≤ D ∧ (step j σ i).work f ≤ Q := by
        by_contra hon
        simp only [decide_eq_false hon, Bool.not_false, ↓reduceIte] at h
        omega
      simp only [decide_eq_true hon, Bool.not_true, Bool.false_eq_true, ↓reduceIte, Nat.add_zero] at h
      simp only [seqAnswers, gradedAnswers, List.map_cons, cut_run dflt f _ D Q hon.1 hon.2, ih _ _ h]

/-! ### Sanity: no grace is the sequential audit -/

/-- **With `t = 0` the graded audit is the sequential audit**, at every hard bound `(D_late, Q_late)` at least
`(D, Q)`. A bounded responder is a graded one and is never late, and a graded responder that is never late answers as
its stopped (bounded) form does. -/
theorem gradedAuditSecure_zero_iff (sch : Scheme M) (S D Q D_late Q_late k : ℕ) (hD : D ≤ D_late) (hQ : Q ≤ Q_late)
    (δ ε : ℝ≥0∞) :
    GradedAuditSecure sch S D Q D_late Q_late k 0 δ ε ↔ AuditSecure sch .sequential S D Q k δ ε := by
  constructor
  · intro h W A₁ A₂
    refine le_trans (pr_mono fun x hx => ?_)
      (h W A₁ ⟨A₂.step, fun W pp j σ i => bounded_mono (A₂.bounded W pp j σ i) hD hQ⟩)
    simp only [seqAccepts] at hx
    simp only [gradedAccepts]
    refine ⟨(map_fst_gradedAnswers _ _ _ _ _ _ _).trans hx, ?_⟩
    rw [countP_late_eq_zero _ _ _ _ fun j σ i => A₂.bounded _ _ j σ i x.1]
  · intro h W A₁ A₂
    let dflt : Bits sch.ℓ × Bits S := (fun _ => false, fun _ => false)
    refine le_trans (pr_mono fun x hx => ?_)
      (h W A₁ ⟨fun W pp j σ i => cut dflt D Q (A₂.step W pp j σ i), fun _ _ _ _ _ => cut_bounded _ _ _ _⟩)
    simp only [gradedAccepts] at hx
    simp only [seqAccepts]
    exact (seqAnswers_cut _ _ _ _ _ _ _ _ (Nat.le_zero.1 hx.2)).trans hx.1

theorem gradedMeets_zero_iff (sch : Scheme M) (D Q D_late Q_late k : ℕ) (hD : D ≤ D_late) (hQ : Q ≤ Q_late)
    (ε : ℝ≥0∞) : GradedMeets sch D Q D_late Q_late k 0 ε ↔ Meets sch .sequential D Q k ε := by
  unfold GradedMeets Meets
  rw [gradedAuditSecure_zero_iff sch _ D Q D_late Q_late k hD hQ]

/-! ### Theorem 1(ii), graded -/

/-- **Theorem 1(ii) for the graded audit.** As `theorem1_ii_seq`, with `P[Bin(k, c) ≥ k − t]` in place of `p₀^k`, for
any `c ≤ 1` at least `p₀`, at every hard bound `(D_late, Q_late)`. -/
theorem theorem1_ii_graded (sch : Scheme M) (D Q β S k t : ℕ) {D_late Q_late : ℕ} (π c : ℝ≥0∞) (hc : c ≤ 1)
    (hp : ENNReal.ofReal (p₀ β S sch.B sch.ℓ) ≤ c) (hinc : TimedINC sch D Q β π) (W : Bits sch.n)
    (A₁ : Preprocessor sch S) (A₂ : GradedResponder sch S D_late Q_late) :
    pr (gradedAccepts sch D Q k t W A₁ A₂) ≤
      (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * π + binTail k t c := by
  classical
  let dflt : Bits sch.ℓ × Bits S := (fun _ => false, fun _ => false)
  let nxt : Bits sch.n → Bits sch.t → M.Ω → ℕ → Bits S → Fin sch.B → Bits S :=
    fun W pp ω j σ i => ((A₂.step W pp j σ i).run (M.answer ω)).2
  let st : List (Fin sch.B) → Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits S :=
    fun h W pp C ω => after (nxt W pp ω) 0 h (A₁ W pp C ω)
  let pred : List (Fin sch.B) → Bits sch.n → Bits sch.t → Bits S → Fin sch.B → OracleProg M (Bits sch.ℓ) :=
    fun h W pp σ i => progMap Prod.fst (cut dflt D Q (A₂.step W pp h.length σ i))
  let good : List (Fin sch.B) → M.Ω × sch.Coins → Fin sch.B → Prop := fun h a i =>
    (pred h W (sch.enc W a.1 a.2).1 (st h W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
      (M.answer a.1) = (sch.enc W a.1 a.2).2 i
  let E : M.Ω × sch.Coins → Prop := fun a => ∃ j ∈ Finset.range k,
    ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
      S + sch.B + (Finset.univ.filter fun i => ¬ good (List.ofFn v) a i).card * sch.ℓ ≤ β
  have hre : pr (gradedAccepts sch D Q k t W A₁ A₂) =
      pr (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) =>
        gradedAccepts sch D Q k t W A₁ A₂ (x.1.1, x.1.2, x.2)) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) =>
      gradedAccepts sch D Q k t W A₁ A₂ (x.1.1, x.1.2, x.2))
  rw [hre]
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  refine (pr_prod_le (α := M.Ω × sch.Coins) (β := Fin k → Fin sch.B)
    (fun a I => gradedAccepts sch D Q k t W A₁ A₂ (a.1, a.2, I)) E (binTail k t c) fun a ha => ?_).trans
    (add_le_add ?_ le_rfl)
  · -- off `E`: a round's stopped program is right with probability at most `p₀` given any history, and a wrong
    -- one was late
    refine le_trans (pr_mono fun I hI => ?_) (pr_hits_ge_le_binomial (nxt W (sch.enc W a.1 a.2).1 a.1)
      (fun j σ i => ((cut dflt D Q (A₂.step W (sch.enc W a.1 a.2).1 j σ i)).run (M.answer a.1)).1 =
        (sch.enc W a.1 a.2).2 i) c hc k t 0 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1)
      fun h hh => ?_)
    · obtain ⟨h1, h2⟩ := hI
      refine le_trans ?_ h2
      exact misses_le_late (M.answer a.1) D Q dflt (A₂.step W (sch.enc W a.1 a.2).1) (sch.enc W a.1 a.2).2
        (List.ofFn I) 0 _ (h1.trans (List.map_ofFn (f := I) (g := (sch.enc W a.1 a.2).2)).symm)
    · have hnf : ¬ S + sch.B + (Finset.univ.filter fun i => ¬ good h a i).card * sch.ℓ ≤ β := by
        intro hf
        refine ha ⟨h.length, Finset.mem_range.2 hh, h.get, Finset.mem_univ _, ?_⟩
        rwa [List.ofFn_get]
      have := (pr_good_le_p₀ sch.B sch.ℓ β S sch.B_pos (good h a) (by convert hnf)).trans hp
      simpa only [good, pred, st, run_progMap, Nat.zero_add] using this
  · -- one compressor per history, then a union bound
    calc pr E ≤ ∑ j ∈ Finset.range k, pr (fun a => ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
          S + sch.B + (Finset.univ.filter fun i => ¬ good (List.ofFn v) a i).card * sch.ℓ ≤ β) :=
          pr_exists_le _ _
      _ ≤ ∑ j ∈ Finset.range k, ∑ _v : Fin j → Fin sch.B, π := by
          refine Finset.sum_le_sum fun j _ => (pr_exists_le _ _).trans (Finset.sum_le_sum fun v _ => ?_)
          exact pr_fits_le sch D Q β S π hinc W (st (List.ofFn v)) (pred (List.ofFn v))
            fun W pp σ i => bounded_progMap _ (cut_bounded _ _ _ _)
      _ = (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * π := by
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin,
            nsmul_eq_mul, Nat.cast_pow]

end PousGraded
