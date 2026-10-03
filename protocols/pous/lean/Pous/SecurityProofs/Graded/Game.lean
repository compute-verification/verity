import Pous.Protocol

/-!
# DRAFT: the graded sequential audit (new trusted surface, not reviewed, not pinned)

**Status: draft for statement review (@lean) and a red team.** These definitions are new trusted surface: a guarantee
stated with them means what they say, and nothing here has been reviewed. They sit under `Pous/SecurityProofs/Graded/`
so that the trusted layer, `TRUSTED.sha256` and the grader stay exactly as reviewed. Once accepted, `GradedResponder`,
`gradedAnswers`, `gradedAccepts` and `GradedAuditSecure` move to `Pous/Protocol/Model/` beside `SeqResponder`,
`GradedMeets` to `Pous/Protocol/Accounting/`, and the statements of `Graded/Band.lean` to `Pous/Guarantees.lean`, where
they are pinned.

**The game.** It is the sequential audit (`Pous/Protocol/Model/Audit.lean`) with a grace of `t` late rounds:

1. Setup and preprocessing are as in the sequential audit: `W` arbitrary, `(pp, C) := enc W ω r`, and an `S`-bit state
   `σ := A₁ W pp C ω`.
2. Round `j` receives the index `i_j`, the round number and the state left by round `j − 1`, and runs
   `step W pp j σ i_j`, a program with no bound: it returns a block and the next `S`-bit state.
3. The round is **on time** iff its program, on the path the primitive answers, takes at most `D` rounds of queries and
   at most `Q` queries (`depth ≤ D ∧ work ≤ Q`); otherwise it is **late**. A late round's program may make any number
   of queries in any number of rounds.
4. The next index arrives as soon as the answer is returned, late or not: a late round delays no other round's
   deadline, and the state it hands on is any `S`-bit function of what it computed.
5. The audit **accepts** iff every returned block is correct (a wrong answer rejects, on time or late) and at most `t`
   rounds are late.

At `t = 0` it is the sequential audit: `PousGraded.gradedAuditSecure_zero_iff` (`Graded/Theorem.lean`) proves
`GradedAuditSecure … k 0 δ ε ↔ AuditSecure … .sequential … k δ ε`.

**What a reviewer should check.** That "late" is measured on the program's own path (`Prog.depth`, `Prog.work` under
`M.answer ω`), as `Prog.Bounded` is, rather than against every primitive; that the state stays `S` bits across a late
round (a late round can recompute anything, but keeps only `S` bits of it); and that the verifier's count of late
rounds is what the implementation's deadline check counts.
-/

namespace PousGraded

open Pous
open scoped ENNReal

variable {M : IdealModel}

/-- DRAFT. A responder for the graded audit: `step W pp j σ i` answers index `i` in round `j` from state `σ`, and
returns the block and the next state. Unlike `SeqResponder`, a round's program has no bound. -/
structure GradedResponder (sch : Scheme M) (S : ℕ) : Type where
  step : Bits sch.n → Bits sch.t → ℕ → Bits S → Fin sch.B → Prog M (Bits sch.ℓ × Bits S)

/-- DRAFT. The blocks a graded responder returns for the indices `is`, from round `j` in state `σ`, each with whether
its round was on time: at most `D` rounds and `Q` queries on the path `f` answers. -/
def gradedAnswers {ℓ S B : ℕ} (f : M.Qry → M.Ans) (D Q : ℕ)
    (step : ℕ → Bits S → Fin B → Prog M (Bits ℓ × Bits S)) : ℕ → List (Fin B) → Bits S → List (Bits ℓ × Bool)
  | _, [], _ => []
  | j, i :: is, σ =>
      (((step j σ i).run f).1, decide ((step j σ i).depth f ≤ D ∧ (step j σ i).work f ≤ Q)) ::
        gradedAnswers f D Q step (j + 1) is ((step j σ i).run f).2

/-- DRAFT. The graded audit with `k` rounds and grace `t` accepts: every returned block is correct and at most `t`
rounds are late. -/
def gradedAccepts (sch : Scheme M) {S : ℕ} (D Q k t : ℕ) (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (A₂ : GradedResponder sch S) (x : Outcome sch k) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  let ans := gradedAnswers (M.answer ω) D Q (A₂.step W pp) 0 (List.ofFn x.2.2) (A₁ W pp C ω)
  ans.map Prod.fst = List.ofFn (fun j => C (x.2.2 j)) ∧ ans.countP (fun a => !a.2) ≤ t

/-- DRAFT. **The graded security target.** For every `W`, every `S`-bit preprocessing and every responder, the
`k`-round audit with grace `t` accepts with probability at most `δ + ε`. -/
def GradedAuditSecure (sch : Scheme M) (S D Q k t : ℕ) (δ ε : ℝ≥0∞) : Prop :=
  ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : GradedResponder sch S),
    pr (gradedAccepts sch D Q k t W A₁ A₂) ≤ δ + ε

/-- DRAFT. **The requirement with grace `t`**: `Meets` with the graded audit in place of the sequential one. At `t = 0`
it is `Meets … .sequential` (`gradedMeets_zero_iff`). -/
def GradedMeets (sch : Scheme M) (D Q k t : ℕ) (ε : ℝ≥0∞) : Prop :=
  ε ≤ Params.εMax ∧ sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧
    GradedAuditSecure sch (sch.stateBound Params.ρ) D Q k t (ENNReal.ofReal Params.δ) ε

end PousGraded
