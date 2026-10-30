import Proofs.Flock.Claims.Complete.ZkTables.Faithful
import Proofs.Flock.Claims.Sampling.Path

/-!
# The sampled reference prover

The honest sampled prover the deployment runs (`refP PF`), on its tapes (one Flock tape per session statement), as a
`Claims.Prover` against the sampled verifier, at a Flock reference prover `PF`: the masked record's
(`FlockComplete.ZkTables.honestPR`, on `Tape`) or the plain record's (`FlockComplete.honestP`).

The trace is committed before the run: the statement's roots, hm96 leaves under the verifier's registered
frame-v3-sha512 path check (`SamplingDeploy.commit`, `pathCommit` at `hm96Leaf`), and the witness holds the openings.
The prover waits for the selection coin, saying what the run's framing fixes, and in each session runs
`PF` at the session's own history, on the Flock witness
that encodes what it opens: at each position the session reads, the witness's first opening there that
passes its own check under the roots (`revealOC`), with its own `H` calls. Its framing is the code's:

* `step`: where the run is at a history, for the prover: a session at its own history under its statement
  (`Step.session`), or a message the framing fixes outside the sessions (`Step.say`);
* `reads`: the positions a session opens, by its statement;
* `wit`: the Flock witness that encodes what a session reveals, by its statement.

A session is named by its statement, so one session may prove several drawn units, its statement theirs.

What it reveals opens and is the committed value `Wit.val` reads (`found_opens`, `found_val`), and covers every read
position with a committed value (`found_covers`).

`sessErrAt`: the largest of Flock's errors, capped at `1`, over the statements of the reference prover's sessions at
`x`, at an honest witness and coins that check the session's unit; the record's completeness error is the sessions'
count times it at the Flock record's `δc`.
-/

namespace Claims.Sampling.Reference

open Claims Claims.Sampling

/-- **Where a run is, for the reference prover**, at a history: the session under the statement `xS` at its own
history `hS`, or a message the run's framing fixes outside the sessions. -/
inductive Step where
  | session (xS : Msg) (hS : List (Msg × Msg))
  | say (m : Msg)

/-- **The masked reference prover's tape**: a masked Flock tape per session statement. -/
abbrev Tape := Msg → FlockComplete.TapeZk

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

/-- **The sampled reference prover** at the witness `w`: what the framing fixes outside the sessions; in the session
under `xS`, the Flock reference prover `PF` at the session's history, on the Flock witness of what it reveals and
`xS`'s tape. -/
def refP {ΩF : Type} (PF : Claims.Flock.Wit → Prover ΩF) (step : Msg → List (Msg × Msg) → Step)
    (parse : Msg → Option Stmt) (reads : Stmt → Msg → List ℕ) (Cm : Commitment Bool)
    (wit : Stmt → Msg → List (Opening Bool) → Claims.Flock.Wit) (w : Wit) : Prover (Msg → ΩF) :=
  ⟨fun x hist ω =>
    match step x hist with
    | .say m => .ret m
    | .session xS hS =>
      match parse x with
      | some s => ocBind (revealOC Cm s.roots w (reads s xS)) fun os => (PF (wit s xS os)).next xS hS (ω xS)
      | none => .ret []⟩

/-- **The sessions' error at `x`**: the largest of `δ`, capped at `1`, over the statements of the sessions of `P`'s runs
at a witness `R` relates to `x` and coins that check the session's unit; `0` where there are none. -/
noncomputable def sessErrAt {Coin Ω : Type} [Inhabited Coin] (H : Msg → Msg) (R : Msg → Wit → Prop)
    (P : Wit → Prover Ω) (V : Verifier Coin) (checked : (x : Msg) → (Fin (V.k x) → Coin) → List (ℕ × ℕ))
    (session : Msg → ℕ × ℕ → Seen → Option (Msg × Seen)) (δ : Msg → ℝ) (x : Msg) : ℝ :=
  ⨆ xS : {xS : Msg // ∃ w ω r u vS, R x w ∧ u ∈ checked x r ∧
      session x u (run H (P w) V x ω r).seen = some (xS, vS)}, min 1 (δ xS.1)

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
theorem refP_session (H : Msg → Msg) {ΩF : Type} (PF : Claims.Flock.Wit → Prover ΩF)
    (step : Msg → List (Msg × Msg) → Step) (parse : Msg → Option Stmt) (reads : Stmt → Msg → List ℕ)
    (Cm : Commitment Bool) (wit : Stmt → Msg → List (Opening Bool) → Claims.Flock.Wit)
    (w : Wit) {x : Msg} {s : Stmt} {hist : List (Msg × Msg)} {xS : Msg} {hS : List (Msg × Msg)}
    (hs : parse x = some s) (hstep : step x hist = .session xS hS) (ω : Msg → ΩF) :
    (((refP PF step parse reads Cm wit w).next x hist ω).eval H).1 =
      (((PF (wit s xS (found H Cm s.roots w (reads s xS)))).next xS hS (ω xS)).eval H).1 := by
  simp only [refP, hstep, hs, ocBind_eval, revealOC_eval]

/-- **Outside the sessions, it says what the framing fixes.** -/
theorem refP_say (H : Msg → Msg) {ΩF : Type} (PF : Claims.Flock.Wit → Prover ΩF)
    (step : Msg → List (Msg × Msg) → Step) (parse : Msg → Option Stmt) (reads : Stmt → Msg → List ℕ)
    (Cm : Commitment Bool) (wit : Stmt → Msg → List (Opening Bool) → Claims.Flock.Wit)
    (w : Wit) {x : Msg} {hist : List (Msg × Msg)} {m : Msg} (hstep : step x hist = .say m) (ω : Msg → ΩF) :
    (((refP PF step parse reads Cm wit w).next x hist ω).eval H).1 = m := by
  simp only [refP, hstep, OC.eval]

end Claims.Sampling.Reference
