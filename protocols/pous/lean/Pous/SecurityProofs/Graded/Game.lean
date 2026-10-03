import Pous.Protocol

/-!
# DRAFT: the graded sequential audit (new trusted surface, not reviewed, not pinned)

**Status: draft for statement review (@lean) and a red team.** These definitions are new trusted surface: a guarantee
stated with them means what they say, and nothing here has been reviewed. They sit under `Pous/SecurityProofs/Graded/`
so that the trusted layer, `TRUSTED.sha256` and the grader stay exactly as reviewed. Once accepted, `GradedResponder`,
`gradedAnswers`, `gradedAccepts` and `GradedAuditSecure` move to `Pous/Protocol/Model/` beside `SeqResponder`,
`GradedMeets` to `Pous/Protocol/Accounting/`, and the statements of `Graded/Band.lean` to `Pous/Guarantees.lean`, where
they are pinned.

**The game.** It is the sequential audit (`Pous/Protocol/Model/Audit.lean`) with a grace of `t` late rounds, each
within a hard bound `(D_late, Q_late)` with `D ≤ D_late` and `Q ≤ Q_late`:

1. Setup and preprocessing are as in the sequential audit: `W` arbitrary, `(pp, C) := enc W ω r`, and an `S`-bit state
   `σ := A₁ W pp C ω`.
2. Round `j` receives the index `i_j`, the round number and the state left by round `j − 1`, and runs
   `step W pp j σ i_j`, a `(D_late, Q_late)`-bounded program (`GradedResponder.bounded`): it returns a block and the
   next `S`-bit state.
3. The round is **on time** iff its program, on the path the primitive answers, takes at most `D` rounds of queries and
   at most `Q` queries (`depth ≤ D ∧ work ≤ Q`); otherwise it is **late**. A late round's program may take up to
   `D_late` rounds of queries and `Q_late` queries.
4. The next index arrives as soon as the answer is returned, late or not: a late round delays no other round's
   deadline, and the state it hands on is any `S`-bit function of what it computed.
5. The audit **accepts** iff every returned block is correct (a wrong answer rejects, on time or late) and at most `t`
   rounds are late.

At `t = 0` it is the sequential audit: `PousGraded.gradedAuditSecure_zero_iff` (`Graded/Theorem.lean`) proves
`GradedAuditSecure … k 0 δ ε ↔ AuditSecure … .sequential … k δ ε` whenever `D ≤ D_late` and `Q ≤ Q_late`.

**Why late rounds are bounded.** With no bound the game is not a conservative model of a storage server at `t ≥ 1`. A
server that keeps nothing of `C` at rest spends its one late round re-encoding `C` from the free `W`, then answers
every later round by lookup, so the graded rule accepts with probability 1. The game's `S`-bit state across the late
round excludes that only by assumption, because a late answer pushes the next arrival back for as long as the
re-encoding takes (requirement 1). The `S`-bit state at each arrival models storage at rest only while the whole window,
`k − t` rounds within `(D, Q)` and `t` rounds within `(D_late, Q_late)`, is too short to re-encode `C`. That is the
sequential audit's own link from the `S`-bit state to storage at rest; an instance states it as a condition on its
parameters (`Graded/Band.lean`: `D_late ≤ d` and `D_late < 2^9`).

**Requirements on the protocol.** The game assumes each of these, and a bound proved in it says nothing about a
protocol that breaks one. When the definitions move to `Pous/Protocol/Model/` and the statements to `Pous/Guarantees/`,
`PROTOCOL.md` must state them as requirements of the graded audit.

1. **No pipelining.** The verifier sends challenge `j + 1` only after it receives answer `j`, and each round's deadline
   runs from its own challenge's send time. The game reveals index `i_{j+1}` only after answer `j`, and times each round
   alone.
2. **A hard deadline.** An answer past `Δ_late` rejects, and a wrong or missing answer rejects. `(D_late, Q_late)` is
   what the adversary completes within `Δ_late`, as `(D, Q)` is within `Δ`. A responder that runs past
   `(D_late, Q_late)` is rejected; the same responder stopped at `(D_late, Q_late)` (`cut`, `Graded/Theorem.lean`) is a
   `GradedResponder` and answers as it does on every round the verifier accepts, so `GradedResponder.bounded` is the
   deadline.
3. **Every answer past `Δ` is counted late.** A round over `(D, Q)` on its own path answers past `Δ`, so the verifier
   counts at least the game's late rounds, and accepting with at most `t` counted implies the game's acceptance.

**What a reviewer should check.** That "late" is measured on the program's own path (`Prog.depth`, `Prog.work` under
`M.answer ω`), as `Prog.Bounded` is, rather than against every primitive; that the state stays `S` bits across a late
round (a late round can recompute anything within `(D_late, Q_late)`, but keeps only `S` bits of it); that
`(D_late, Q_late)` is what the adversary completes within `Δ_late`; and that the verifier's count of late rounds is
what the implementation's deadline check counts.
-/

namespace PousGraded

open Pous
open scoped ENNReal

variable {M : IdealModel}

/-- DRAFT. A responder for the graded audit: `step W pp j σ i` answers index `i` in round `j` from state `σ`, and
returns the block and the next state, within `(D_late, Q_late)`: the hard deadline (requirement 2). Whether a round is
on time is measured against the audit's `(D, Q)` on its own path (`gradedAnswers`). -/
structure GradedResponder (sch : Scheme M) (S D_late Q_late : ℕ) : Type where
  step : Bits sch.n → Bits sch.t → ℕ → Bits S → Fin sch.B → Prog M (Bits sch.ℓ × Bits S)
  bounded : ∀ W pp j σ i, (step W pp j σ i).Bounded D_late Q_late

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
def gradedAccepts (sch : Scheme M) {S D_late Q_late : ℕ} (D Q k t : ℕ) (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (A₂ : GradedResponder sch S D_late Q_late) (x : Outcome sch k) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  let ans := gradedAnswers (M.answer ω) D Q (A₂.step W pp) 0 (List.ofFn x.2.2) (A₁ W pp C ω)
  ans.map Prod.fst = List.ofFn (fun j => C (x.2.2 j)) ∧ ans.countP (fun a => !a.2) ≤ t

/-- DRAFT. **The graded security target.** For every `W`, every `S`-bit preprocessing and every responder within
`(D_late, Q_late)`, the `k`-round audit with grace `t` accepts with probability at most `δ + ε`. -/
def GradedAuditSecure (sch : Scheme M) (S D Q D_late Q_late k t : ℕ) (δ ε : ℝ≥0∞) : Prop :=
  ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : GradedResponder sch S D_late Q_late),
    pr (gradedAccepts sch D Q k t W A₁ A₂) ≤ δ + ε

/-- DRAFT. **The requirement with grace `t`**: `Meets` with the graded audit in place of the sequential one. At `t = 0`
it is `Meets … .sequential` (`gradedMeets_zero_iff`, for `D ≤ D_late` and `Q ≤ Q_late`). -/
def GradedMeets (sch : Scheme M) (D Q D_late Q_late k t : ℕ) (ε : ℝ≥0∞) : Prop :=
  ε ≤ Params.εMax ∧ sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧
    GradedAuditSecure sch (sch.stateBound Params.ρ) D Q D_late Q_late k t (ENNReal.ofReal Params.δ) ε

end PousGraded
