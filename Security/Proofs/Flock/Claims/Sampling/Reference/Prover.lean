import Proofs.Flock.Claims.Complete.Zk.Honest
import Proofs.Flock.Claims.Sampling.Path

/-!
# The sampled reference prover

The honest sampled prover the deployment runs (`refP`), on its tapes (`Tape`, one Flock tape per unit), as a
`Claims.Prover` against the sampled verifier.

The trace is committed before the run: the statement's roots, hm96 leaves under the verifier's registered
frame-v3-sha512 path check (`SamplingDeploy.commit`, `pathCommit` at `hm96Leaf`), and the witness holds the openings.
The prover waits for the selection coin, saying what the run's framing fixes, and in each checked unit's session runs
Flock's masked reference prover (`FlockComplete.honestPZk`) at the session's own history, on the Flock witness that
encodes what it opens: at each position the session reads, the witness's first opening there that
passes its own check under the roots (`revealOC`), with its own `H` calls. Its framing is the code's:

* `step`: where the run is at a history, for the prover: unit `u`'s session at its own history under its statement
  (`Step.session`), or a message the framing fixes outside the sessions (`Step.say`);
* `reads`: the positions a checked unit's session opens under the session's statement;
* `wit`: the Flock witness that encodes what a unit's session reveals, under the session's statement.

Both read the session's statement, so one session may prove several drawn units, its statement theirs.

What it reveals opens and is the committed value `Wit.val` reads (`found_opens`, `found_val`), and covers every read
position with a committed value (`found_covers`).

`sessErrAt`: the largest of Flock's errors, capped at `1`, over the statements of the sessions at `x`; the record's
completeness error is the proof units' count times it at Flock's `δcZk`.
-/

namespace Claims.Sampling.Reference

open Claims Claims.Sampling

/-- **Where a run is, for the reference prover**, at a history: unit `u`'s session at its own history `hS` under the
session's statement `xS`, or a message the run's framing fixes outside the sessions. -/
inductive Step where
  | session (u : ℕ × ℕ) (xS : Msg) (hS : List (Msg × Msg))
  | say (m : Msg)

/-- **The reference prover's tape**: a Flock tape per unit, its session's. -/
abbrev Tape := ℕ × ℕ → FlockComplete.TapeZk

/-- **The opening revealed at `g`**: the first of `os` at `g` whose check under `roots` passes, each check its own `H`
calls. -/
def firstOC (Cm : Commitment Bool) (roots : List Msg) (g : ℕ) : List (Opening Bool) → OC (Option (Opening Bool))
  | [] => .ret none
  | o :: os =>
    if o.pos = g then ocBind (Cm.opens roots o) fun ok => if ok then .ret (some o) else firstOC Cm roots g os
    else firstOC Cm roots g os

/-- **The openings revealed at the positions `gs`**: at each, the witness's first opening there that passes. -/
def revealOC (Cm : Commitment Bool) (roots : List Msg) (w : Wit) : List ℕ → OC (List (Opening Bool))
  | [] => .ret []
  | g :: gs => ocBind (firstOC Cm roots g w.vals) fun o =>
      ocBind (revealOC Cm roots w gs) fun os => .ret (o.toList ++ os)

/-- **What it reveals at the positions `gs`**, against `H`. -/
def found (H : Msg → Msg) (Cm : Commitment Bool) (roots : List Msg) (w : Wit) (gs : List ℕ) : List (Opening Bool) :=
  gs.flatMap fun g => (w.vals.find? fun o => o.pos == g && Cm.Opens H roots o).toList

/-- **The sampled reference prover** at the witness `w`: what the framing fixes outside the sessions; in unit `u`'s
session, Flock's masked reference prover at the session's history, on the Flock witness of what it reveals and the
unit's tape. -/
def refP (step : Msg → List (Msg × Msg) → Step) (parse : Msg → Option Stmt) (reads : Stmt → ℕ × ℕ → Msg → List ℕ)
    (Cm : Commitment Bool) (wit : Stmt → ℕ × ℕ → Msg → List (Opening Bool) → Claims.Flock.Wit) (w : Wit) :
    Prover Tape :=
  ⟨fun x hist ω =>
    match step x hist with
    | .say m => .ret m
    | .session u xS hS =>
      match parse x with
      | some s => ocBind (revealOC Cm s.roots w (reads s u xS)) fun os =>
          (FlockComplete.honestPZk (wit s u xS os)).next xS hS (ω u)
      | none => .ret []⟩

/-- **The sessions' error at `x`**: the largest of `δ`, capped at `1`, over the statements of the sessions `x` sets up;
`0` where it sets up none. -/
noncomputable def sessErrAt (session : Msg → ℕ × ℕ → Seen → Option (Msg × Seen)) (δ : Msg → ℝ) (x : Msg) : ℝ :=
  ⨆ xS : {xS : Msg // ∃ u v vS, session x u v = some (xS, vS)}, min 1 (δ xS.1)

/-! ## What it reveals -/

theorem firstOC_eval (H : Msg → Msg) (Cm : Commitment Bool) (roots : List Msg) (g : ℕ) :
    ∀ os : List (Opening Bool),
      ((firstOC Cm roots g os).eval H).1 = os.find? fun o => o.pos == g && Cm.Opens H roots o
  | [] => rfl
  | o :: os => by
    have ih := firstOC_eval H Cm roots g os
    by_cases hp : o.pos = g
    · have hb : (o.pos == g) = true := beq_iff_eq.2 hp
      unfold Commitment.Opens at ih ⊢
      cases h : ((Cm.opens roots o).eval H).1
      · simp only [firstOC, ite_eq_left hp, ocBind_eval, h, List.find?_cons, hb, Bool.true_and, Bool.false_eq_true,
          ite_false]
        exact ih
      · simp [firstOC, hp, ocBind_eval, h, OC.eval]
    · have hb : (o.pos == g) = false := by simpa using hp
      simp only [firstOC, ite_eq_right hp, List.find?_cons, hb, Bool.false_and]
      exact ih

theorem revealOC_eval (H : Msg → Msg) (Cm : Commitment Bool) (roots : List Msg) (w : Wit) :
    ∀ gs : List ℕ, ((revealOC Cm roots w gs).eval H).1 = found H Cm roots w gs
  | [] => rfl
  | g :: gs => by
    simp only [revealOC, ocBind_eval, OC.eval, firstOC_eval, revealOC_eval H Cm roots w gs, found, List.flatMap_cons]

/-- **What it reveals opens** under the roots: the honest commitment's check passes. -/
theorem found_opens {H : Msg → Msg} {Cm : Commitment Bool} {roots : List Msg} {w : Wit} {gs : List ℕ}
    {o : Opening Bool} (h : o ∈ found H Cm roots w gs) : Cm.Opens H roots o = true := by
  obtain ⟨g, -, hg⟩ := List.mem_flatMap.1 h
  have hf : w.vals.find? (fun o => o.pos == g && Cm.Opens H roots o) = some o := by
    cases he : w.vals.find? (fun o => o.pos == g && Cm.Opens H roots o) with
    | none => simp [he] at hg
    | some o' =>
      simp only [he, Option.toList_some, List.mem_singleton] at hg
      rw [hg]
  have := List.find?_some hf
  simp only [Bool.and_eq_true] at this
  exact this.2

/-- **What it reveals is the committed value `Wit.val` reads** at its position. -/
theorem found_val {H : Msg → Msg} {Cm : Commitment Bool} {s : Stmt} {w : Wit} {gs : List ℕ} {o : Opening Bool}
    (h : o ∈ found H Cm s.roots w gs) : w.val H Cm s o.pos = some o.val := by
  obtain ⟨g, -, hg⟩ := List.mem_flatMap.1 h
  cases hf : w.vals.find? (fun o => o.pos == g && Cm.Opens H s.roots o) with
  | none => simp [hf] at hg
  | some o' =>
    simp only [hf, Option.toList_some, List.mem_singleton] at hg
    subst hg
    have hpos := List.find?_some hf
    simp only [Bool.and_eq_true, beq_iff_eq] at hpos
    unfold Wit.val
    rw [hpos.1, hf]
    rfl

/-- **It covers every read position with a committed value.** -/
theorem found_covers {H : Msg → Msg} {Cm : Commitment Bool} {s : Stmt} {w : Wit} {gs : List ℕ} {g : ℕ} {b : Bool}
    (hg : g ∈ gs) (hb : w.val H Cm s g = some b) : ∃ o ∈ found H Cm s.roots w gs, o.pos = g := by
  unfold Wit.val at hb
  cases hf : w.vals.find? (fun o => o.pos == g && Cm.Opens H s.roots o) with
  | none => simp [hf] at hb
  | some o =>
    refine ⟨o, List.mem_flatMap.2 ⟨g, hg, by simp [hf]⟩, ?_⟩
    have := List.find?_some hf
    simp only [Bool.and_eq_true, beq_iff_eq] at this
    exact this.1

/-! ## Its messages -/

/-- **In a session, its message is Flock's reference prover's** at the session's history, on the Flock witness of what
it reveals. -/
theorem refP_session (H : Msg → Msg) (step : Msg → List (Msg × Msg) → Step) (parse : Msg → Option Stmt)
    (reads : Stmt → ℕ × ℕ → Msg → List ℕ) (Cm : Commitment Bool)
    (wit : Stmt → ℕ × ℕ → Msg → List (Opening Bool) → Claims.Flock.Wit)
    (w : Wit) {x : Msg} {s : Stmt} {hist : List (Msg × Msg)} {u : ℕ × ℕ} {xS : Msg} {hS : List (Msg × Msg)}
    (hs : parse x = some s) (hstep : step x hist = .session u xS hS) (ω : Tape) :
    (((refP step parse reads Cm wit w).next x hist ω).eval H).1 =
      (((FlockComplete.honestPZk (wit s u xS (found H Cm s.roots w (reads s u xS)))).next xS hS (ω u)).eval H).1 := by
  simp only [refP, hstep, hs, ocBind_eval, revealOC_eval]

/-- **Outside the sessions, it says what the framing fixes.** -/
theorem refP_say (H : Msg → Msg) (step : Msg → List (Msg × Msg) → Step) (parse : Msg → Option Stmt)
    (reads : Stmt → ℕ × ℕ → Msg → List ℕ) (Cm : Commitment Bool)
    (wit : Stmt → ℕ × ℕ → Msg → List (Opening Bool) → Claims.Flock.Wit)
    (w : Wit) {x : Msg} {hist : List (Msg × Msg)} {m : Msg} (hstep : step x hist = .say m) (ω : Tape) :
    (((refP step parse reads Cm wit w).next x hist ω).eval H).1 = m := by
  simp only [refP, hstep, OC.eval]

end Claims.Sampling.Reference
