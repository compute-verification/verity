import FlockSoundness.Protocol.Refine.LiveCompiled
import FlockSoundness.Protocol.Discharge.Exec.FlockLinkL
import FlockSoundness.Protocol.Discharge.Exec.StrictCRL
import FlockSoundness.Protocol.Headline
import FlockSoundness.Protocol.Discharge.Exec.Event

/-!
# R11 for one-table sessions: the live audit simulated in the salted audit

`ExecRefines` (`Event.lean`) is R11 at one strategy pair. This file proves it, with `bad = ⊥`, for the live audit whose
session is one table's live session (`liveTable`, the coin server with retained round bytes, `Refine/Live.lean`) written
out as a session directory by `wr`, against the audit with the salted session `batchedSessionL` on one-table plans. What
it takes is `Decodes` (`Refine/LiveSim.lean`) at each draw and registration: on every live outcome flock-verify
accepts, each rep's game runs on the tape decoded from its stream and its proof file, and the salted table's verdict on
those runs holds.

* `sessionBL_one`: the salted session on one table is `modelTable`'s shape, with the reps of `tableAfterL` and the
  output `[tableOutL …]`.
* `live_le_sessionBL`: against any live prover of `liveTable`, the live verdict is no likelier than the salted
  session's acceptance against the simulated prover (`live_le` at `sessionBL_one`).
* `simAudit`, `execRefines_one`: the audit strategy built from a live one (the registration it sends, and at each draw
  the simulated session strategy), and R11 at it, `ExecRefines … (LiveAccepts I) ⊥ (audit … (batchedSessionL …)) …`.
* `Prog.flock_headline_live`: `Prog.flock_headline_execL` with `hRef` discharged by `execRefines_one`. For one-table
  plans and the live session `liveTable` written by `wr`, flock-verify accepts with at least `K` wrong units of the
  simulated prover's committed values with probability at most `miss(K) + ksAvgStrict + linkBoundE`, given `Decodes`.

What R11 still needs past this: `Decodes` itself, which is deterministic (R11b, the framing: the round bytes the coin
server retains decode to the messages the proof files replay, and `wr` is the server's record); sessions of more than
one table (M0's default is one, `--session-tables 1`); and that the coin server's session is `liveTable`, whose coins
are uniform and drawn after each round's bytes (A6).
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Refine Audit Audit.Partition
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## Strategies through relabellings and casts -/

section Plumbing

variable {α β : Type}

theorem map_map' (f : α → β) {γ : Type} (f' : β → γ) (g : Game α) : (g.map f).map f' = g.map fun a => f' (f a) := by
  unfold Game.map
  rw [Game.bind_assoc]
  rfl

theorem bind_map {γ : Type} (g : Game α) (K : α → Game β) (f : β → γ) :
    (g.bind K).map f = g.bind fun a => (K a).map f := by
  unfold Game.map
  rw [Game.bind_assoc]

end Plumbing

/-! ## One table's salted session as `modelTable` -/

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

open Classical in
/-- The salted table's output, `tableAfterL`'s, as `modelTable`'s. -/
noncomputable def tableOutL (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (cap : Cap D) (pend : PendL 𝔽 D × PendL 𝔽 D) (os : Opens 𝔽 D × Salts Sl) : OutCL 𝔽 D Sl :=
  ⟨pend.1.verdict os.1 && pend.2.verdict os.1 &&
    decide (OpensOKL H E Lf cap 0 pend.1 os.1 os.2 ∧ OpensOKL H E Lf cap 1 pend.2 os.1 os.2), cap, pend, os.1, os.2⟩

/-- The reps of table `T`, at link points `pts`. -/
noncomputable abbrev repsOf (A : Arith 𝔽 K) (T : TabSpec mPts) (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) :
    Game (PendL 𝔽 D) :=
  repC A r.val T.S T.sch T.hm (extraClaims T.S T.ptLocal mPts pts)

omit [Fintype K] in
/-- **The salted session on one table is `modelTable`'s shape**: `Commit`'s caps, the link points, the reps
concurrently, the salted openings, and the one table's output. -/
theorem sessionBL_one (A : Arith 𝔽 K) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (T : TabSpec mPts) :
    modelTable (repsOf (D := D) A T) (fun (caps : Fin 1 → Cap D) pend os => [tableOutL H E Lf (caps 0) pend os]) =
      sessionBL A H E Lf [T] := by
  show Game.send (Fin 1 → Cap D) (fun caps => Game.coin (Fin 2 → Fin mPts → 𝔽) fun pts =>
      (interleave (repsOf (D := D) A T pts 0) (repsOf (D := D) A T pts 1)).bind fun pend =>
        Game.send (Opens 𝔽 D × Salts Sl) fun os => Game.ret [tableOutL H E Lf (caps 0) pend os]) =
    Game.send (Fin 1 → Cap D) fun caps => Game.coin (Fin 2 → Fin mPts → 𝔽) fun pts =>
      batch ((gamesBL A H E Lf [T] caps).map (· pts))
  congr 1
  funext caps
  congr 1
  funext pts
  simp only [gamesBL, List.ofFn_succ, List.ofFn_zero, List.map_cons, List.map_nil, batch, Plays.interleave_ret,
    map_map']
  rw [show (fun a : OutCL 𝔽 D Sl => (a, ([] : List (OutCL 𝔽 D Sl))).1 :: (a, ([] : List (OutCL 𝔽 D Sl))).2) =
    fun a => [a] from rfl]
  unfold tabAfterL tableAfterL
  rw [bind_map]
  rfl

/-! ## The transfer at one table -/

variable [Nonempty D] [Inhabited D] [Inhabited Sl]

omit [Fintype K] [Nonempty D] in
/-- `Decodes` for the one table's output, as `Decodes` for the session's list of one output. -/
theorem decodes_one {H : List UInt8 → D} {E : Merkle.Enc (List 𝔽) D} {Lf : Merkle.Leaf (List 𝔽) Sl D H}
    {T : TabSpec mPts} {A : Arith 𝔽 K} {pL : LiveOut mPts → Prop} {opn : Array ByteArray → Opens 𝔽 D × Salts Sl}
    {dec : ℕ → ByteArray → List Msg} {one : ℕ → Bool} {fin : Fin 2 → Array ByteArray → List Msg}
    {capOf : ByteArray → (Fin 1 → Cap D)}
    (hdec : Decodes (fun (caps : Fin 1 → Cap D) pend os => tableOutL H E Lf (caps 0) pend os) pL (·.accepted = true)
      opn dec one fin capOf (repsOf (D := D) A T)) :
    Decodes (fun (caps : Fin 1 → Cap D) pend os => [tableOutL H E Lf (caps 0) pend os]) pL (acceptedBL [T]) opn dec
      one fin capOf (repsOf (D := D) A T) := fun o ho => by
  obtain ⟨pend, h0, h1, hacc⟩ := hdec o ho
  refine ⟨pend, h0, h1, fun i => ?_⟩
  obtain ⟨i, hi⟩ := i
  obtain rfl : i = 0 := by simp at hi; omega
  exact hacc

/-- The simulated session strategy against a live prover of `liveTable`. -/
noncomputable def simSessL (A : Arith 𝔽 K) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (T : TabSpec mPts) (pL : LiveOut mPts → Prop)
    (opn : Array ByteArray → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray) (dec : ℕ → ByteArray → List Msg)
    (one : ℕ → Bool) (fin : Fin 2 → Array ByteArray → List Msg) (capOf : ByteArray → (Fin 1 → Cap D)) (N : ℕ)
    (hdec : Decodes (fun (caps : Fin 1 → Cap D) pend os => tableOutL H E Lf (caps 0) pend os) pL (·.accepted = true)
      opn dec one fin capOf (repsOf (D := D) A T))
    (Pr : Strategy (liveTable dg mPts N)) : Strategy (sessionBL A H E Lf [T]) :=
  sessionBL_one A H E Lf T ▸
    (simTable (fun (caps : Fin 1 → Cap D) pend os => [tableOutL H E Lf (caps 0) pend os]) pL (acceptedBL [T]) opn dg
      dec one fin capOf N (repsOf (D := D) A T) (fun _ _ => Model.allInh_repC A _ T.S T.sch T.hm _)
      (decodes_one hdec)).strategy Pr

/-! ## The audit -/

variable {n : ℕ} {Reg : Type}

/-- **The live audit's session at one table**: `liveTable`, its outcome written out as the session directory by `wr`
(the record the coin server writes, and the proof files). -/
def liveOf (dg : ByteArray → ByteArray) (mPts N : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
    (S : Finset (Fin n)) (R : Reg) : Game Transcript :=
  (liveTable dg mPts N).map (wr S R)

variable (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
  (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
  (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)) (N : ℕ)

/-- **The decoder is right on the sessions flock-verify accepts** (R11b, named): at every draw and registration, on
every live outcome whose session directory flock-verify accepts, each rep's game runs on the tape decoded from its
stream and its proof file, and the salted table's verdict on those runs, with the cap `root_B` decodes to and the
openings and salts the proof files decode to, holds. -/
def DecodesAll : Prop :=
  ∀ S R, Decodes (fun (caps : Fin 1 → Cap D) pend os => tableOutL H E Lf (caps 0) pend os)
    (fun o => Accepts I (wr S R o)) (·.accepted = true) (opn S R) (dec S R) (one S R) (fin S R) (capOf S R)
    (repsOf (D := D) execArith (T S R))

variable {L : Law n}

/-! ## The headline at the executable, for one-table sessions -/

variable {N' : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N'))

end FlockSoundness.Discharge.Exec
