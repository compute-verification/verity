import Proofs.Flock.C3.Compose.Family
import Proofs.Flock.VBridge.Family.Commits

/-!
# n0: candidate 3's honest family

`C3Family.honest` is the family the guarantee's `_hbr` is discharged at, at its sessions (`C3Family.checked`: V*'s
checked sessions, `checkedC3`). Its commit strings, digests and messages are G's forms at the session's reads (`skOf`,
`cmtOf`, `wmsgOf` and `msgOf` at a decoder `dec`), and:

* `run`: the inner transcript's steps (V*'s layers' messages zipped with the rounds' coins) decoded stage by stage, in
  the chain's order (`decodeRun`): the level-0 cap and the link points, the two reps on the prefix both play
  (`repsOf`, rep 0's steps first), each rep's sparse phase, R's opening and R's Ligerito each on the prefix of what is
  left that its game plays (`stageOf`), and the openings. A game plays on at most one prefix of a tape, so each stage's
  decode is the play's where one plays.
* `row`: the `b ‖ c` the verifier holds at a registered value's row, by `RecVStar.fixedRow`'s prefixes at its hold of
  the inner transcript (`innerCommitsAt`, the event's), and at the inner shape's sparse entry (`heldRow`).

R's opening is at `7 ≤ mReg`, which the padding gives (`seven_le_mReg`).
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.InnerAccepts (firstSteps)
open FlockVBridge.Family (skOf cmtOf wmsgOf msgOf)

/-- R's table has at least the opening's seven variables: it is padded to `PAD_TO`. -/
theorem seven_le_mReg (sh : SparseShape) : 7 ≤ sh.mReg :=
  show 7 ≤ max SparseShape.padTo _ from le_trans (by decide) (le_max_left _ _)

/-- **The default run**: zeros, empty steps and rejecting pending results. -/
noncomputable def c3Default (e : ExecHolo) : C3Run e :=
  ⟨fun _ => H512 [], fun _ _ => 0, fun _ => [],
    fun _ => (⟨0, fun _ => 0, fun _ => 0, 0, 0, 0⟩, ⟨⟨0, fun _ => 0, fun _ => 0⟩, 0, fun _ => 0, fun _ => 0⟩),
    fun _ => PendL.reject, fun _ => [], fun _ => ⟨0, fun _ => 0, fun _ => 0, fun _ => 0⟩, [], ⟨0, fun _ => 0⟩, [],
    PendL.reject, (fun _ _ _ => ([], []), fun _ _ _ => [])⟩

open Classical in
/-- **A stage's decode**: a prefix of the tape `l` the stage's game `g` plays on, and its output; `([], d)` where it
plays on none. -/
noncomputable def stageOf {α : Type} (g : Game α) (l : List Step) (d : α) : List Step × α :=
  if h : ∃ p : List Step × α, p.1 <+: l ∧ PlaysT g p.1 p.2 then h.choose else ([], d)

open Classical in
/-- **The two reps' decode**: rep 0's and rep 1's steps, rep 0's first (`firstSteps`, the reps' interleaving), on a
prefix of `l`, with their outputs; `d` where none plays. -/
noncomputable def repsOf {α β : Type} (g₀ : Game α) (g₁ : Game β) (l : List Step) (d : List Step × α × List Step × β) :
    List Step × α × List Step × β :=
  if h : ∃ q : List Step × α × List Step × β,
      firstSteps q.1 ++ q.2.2.1 <+: l ∧ PlaysT g₀ q.1 q.2.1 ∧ PlaysT g₁ q.2.2.1 q.2.2.2 then h.choose else d

open Classical in
/-- **A prover step's value** at the type `α`; `d` where the step is none of `α`'s. -/
noncomputable def proverAt (α : Type) (s : Option Step) (d : α) : α :=
  if h : ∃ a : α, s = some (.inl ⟨α, a⟩) then h.choose else d

open Classical in
/-- **A coin step's value** at the type `α`; `d` where the step is none of `α`'s. -/
noncomputable def coinAt (α : Type) (s : Option Step) (d : α) : α :=
  if h : ∃ a : α, s = some (.inr ⟨α, a⟩) then h.choose else d

/-- **The inner transcript's steps decoded stage by stage**, in the chain's order (`holoSteps`): the level-0 cap, the
link points, the reps, rep 0's and rep 1's sparse phases on their reps' outputs, R's opening on the two claims, R's
Ligerito on its output, and the openings with their salts. -/
noncomputable def decodeRun (e : ExecHolo) (l : List Step) : C3Run e :=
  let d := c3Default e
  let S := e.cls.pubStatement e.regs
  let cap₀ := proverAt (Cap Dig) l[0]? d.cap₀
  let pts := coinAt (Fin 2 → Fin e.cls.mPts → GF128) l[1]? d.pts
  let l₁ := l.drop 2
  let rs := repsOf (holoRepC (D := Dig) execArith 0 S e.sch e.cls.hm (extraClaims S e.cls.ptLocal e.cls.mPts pts))
    (holoRepC (D := Dig) execArith 1 S e.sch e.cls.hm (extraClaims S e.cls.ptLocal e.cls.mPts pts)) l₁
    ([], (none, PendL.reject), [], (none, PendL.reject))
  let rep : Fin 2 → ZcOut GF128 S.m × HoloLcOut GF128 S.m (S.kLog - 6) :=
    ![rs.2.1.1.getD (d.rep 0), rs.2.2.2.1.getD (d.rep 1)]
  let sparse := fun i => sparsePhase execArith e.cls.shape (holoPub S e.cls.shape e.cls.units (rep i).1 (rep i).2)
    (fun σ => (e.τ σ).val) (rep i).2.h
  let l₂ := l₁.drop (2 * rs.1.length + rs.2.2.1.length)
  let sp₀ := stageOf (sparse 0) l₂ none
  let l₃ := l₂.drop sp₀.1.length
  let sp₁ := stageOf (sparse 1) l₃ none
  let l₄ := l₃.drop sp₁.1.length
  let claim : Fin 2 → RegClaim GF128 e.cls.shape := ![sp₀.2.getD (d.claim 0), sp₁.2.getD (d.claim 1)]
  let op := stageOf (opening execArith (seven_le_mReg e.cls.shape) [(claim 0).toClaim, (claim 1).toClaim]) l₄ none
  let l₅ := l₄.drop op.1.length
  let opened := op.2.getD d.opened
  let lg := stageOf (ligeritoC (D := Dig) execArith 2 e.rsch.sch opened.T opened.b) l₅ PendL.reject
  ⟨cap₀, pts, ![rs.1, rs.2.2.1], rep, ![rs.2.1.2, rs.2.2.2.2], ![sp₀.1, sp₁.1], claim, op.1, opened, lg.1, lg.2,
    proverAt (Opens GF128 Dig × Salts (List UInt8)) (l₅.drop lg.1.length)[0]? d.os⟩

/-- **The `b ‖ c` the verifier holds at a registered value's row**: `RecVStar.fixedRow` at its hold of the inner
transcript `cs`, by the value's prefix, at the inner shape's sparse entry; none where it holds none. -/
def heldRow (cs : Except String RecVStar.InnerCommits) (sparse : Option RecSparse.Shape) (value : String) (x : ℕ) :
    Option (List UInt8) :=
  match cs with
  | .ok cs => match RecVStar.fixedRow cs value x sparse with
    | some (.ok (_, c)) => some c.toList
    | _ => none
  | .error _ => none

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(n0) The honest family**: at the verifier's list `ss` of the inner shape `ish`, V*'s statements `Vs` whose files
parse (`hob`), V*'s sessions `zs`, G's commit strings, digests and messages at the sessions' reads `own`, the stage
selector `sel`, the commitments' `b ‖ c` (`bc`) and the decoder `dec`, the run decoded stage by stage (`decodeRun`) and
the rows the verifier holds of its own inputs `tr`, `record` and `reg` (`heldRow`). -/
noncomputable def C3Family.honest (ish : RecVStar.Inner) {ss : Array VStar.Stmt}
    (hss : RecVStar.statements ish = .ok ss) (bind : Fin (vstarCount ish) → VStar.Bindings)
    {Vs : Fin (vstarCount ish) → VStmt} (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).Outer Reg')
    (own : Fin (vstarCount ish) → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg)
    (tr : Lean.Json) (record : String) (reg : Lean.Json) : C3Family T CT st hp xR L Reg Cm Reg' Reg₂ where
  ish := ish
  stmts := ss
  hstmts := hss
  bind := bind
  Vs := Vs
  hob := hob
  zs := zs
  wmsg := wmsgOf Vs (skOf Vs own sel) (holoInnerAt T CT st hp).r dec
  Sk := skOf Vs own sel
  cmt := cmtOf Vs bc
  msg := msgOf Vs (skOf Vs own sel) dec
  run := fun _ _ t Lb => decodeRun (holoAt st hp)
    (schedSteps T CT 0 ((wmsgOf Vs (skOf Vs own sel) (holoInnerAt T CT st hp).r dec Lb).zip (t.map Prod.snd)))
  row := fun v x _ => heldRow (innerCommitsAt tr record st reg) ish.sparse v x

/-- **(n0) The checked family**: the honest family at the guarantee's sessions, V*'s checked sessions (`checkedC3`) of
the prover's `zr` under the records' custody, at the verifier's hold of the inner transcript, check 5's source `coins`
and the shared rows `rows`. -/
noncomputable abbrev C3Family.checked (ish : RecVStar.Inner) {ss : Array VStar.Stmt}
    (hss : RecVStar.statements ish = .ok ss) (bind : Fin (vstarCount ish) → VStar.Bindings)
    {Vs : Fin (vstarCount ish) → VStmt} (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    (zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg')
    (hCust : Flock.Premises.RecordCustody zr) (h : Typed zr) (coins : Option Flock.RecCoins.Source)
    (rows : Option (Array Flock.Registered.Port))
    (own : Fin (vstarCount ish) → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg)
    (tr : Lean.Json) (record : String) (reg : Lean.Json) : C3Family T CT st hp xR L Reg Cm Reg' Reg₂ :=
  C3Family.honest ish hss bind hob
    (Flock.Guarantees.checkedC3 ish (innerCommitsAt tr record st reg).toOption coins bind rows zr hCust h) own sel bc
    dec tr record reg

end FlockC3.Compose
