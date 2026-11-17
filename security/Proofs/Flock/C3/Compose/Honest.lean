module

public import Proofs.Flock.C3.Compose.Family
public import Proofs.Flock.C3.Compose.Leaf
public import Proofs.Flock.Recursive.AuditC3
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-! Candidate 3's honest family (`C3Family.honest`), at which the guarantee's `_hbr` is discharged. -/

@[expose] public section

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.InnerAccepts (firstSteps)
open FlockVBridge.Family (skOf cmtOf wmsgOf msgOf)

theorem seven_le_mReg (sh : SparseShape) : 7 ≤ sh.mReg :=
  show 7 ≤ max SparseShape.padTo _ from le_trans (by decide) (le_max_left _ _)

noncomputable def c3Default (e : ExecHolo) : C3Run e :=
  ⟨fun _ => H512 [], fun _ _ => 0, fun _ => [],
    fun _ => (⟨0, fun _ => 0, fun _ => 0, 0, 0, 0⟩, ⟨⟨0, fun _ => 0, fun _ => 0⟩, 0, fun _ => 0, fun _ => 0⟩),
    fun _ => PendL.reject, fun _ => [], fun _ => ⟨0, fun _ => 0, fun _ => 0, fun _ => 0⟩, [], ⟨0, fun _ => 0⟩, [],
    PendL.reject, (fun _ _ _ => ([], []), fun _ _ _ => [])⟩

open Classical in
noncomputable def stageOf {α : Type} (g : Game α) (l : List Step) (d : α) : List Step × α :=
  if h : ∃ p : List Step × α, p.1 <+: l ∧ PlaysT g p.1 p.2 then h.choose else ([], d)

open Classical in
noncomputable def repsOf {α β : Type} (g₀ : Game α) (g₁ : Game β) (l : List Step) (d : List Step × α × List Step × β) :
    List Step × α × List Step × β :=
  if h : ∃ q : List Step × α × List Step × β,
      firstSteps q.1 ++ q.2.2.1 <+: l ∧ PlaysT g₀ q.1 q.2.1 ∧ PlaysT g₁ q.2.2.1 q.2.2.2 then h.choose else d

open Classical in
noncomputable def proverAt (α : Type) (s : Option Step) (d : α) : α :=
  if h : ∃ a : α, s = some (.inl ⟨α, a⟩) then h.choose else d

open Classical in
noncomputable def coinAt (α : Type) (s : Option Step) (d : α) : α :=
  if h : ∃ a : α, s = some (.inr ⟨α, a⟩) then h.choose else d

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

export FlockSoundness.Assumptions (heldRow)

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

noncomputable def C3Family.honest (ish : RecVStar.Inner) {ss : Array VStar.Stmt}
    (hss : RecVStar.statements ish = .ok ss) (bind : Fin (vstarCount ish) → VStar.Bindings)
    {Vs : Fin (vstarCount ish) → VStmt} (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).Outer Reg')
    (own : Fin (vstarCount ish) → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg)
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json) :
    C3Family T CT st hp xR L Reg Cm Reg' Reg₂ where
  ish := ish
  stmts := ss
  hstmts := hss
  bind := bind
  Vs := Vs
  hob := hob
  zs := zs
  wmsg := wmsgOf Vs (skLeaf Vs own sel (holoInnerAt T CT st hp).r) (holoInnerAt T CT st hp).r dec
  Sk := skOf Vs own sel
  cmt := cmtOf Vs bc
  msg := msgOf Vs (skLeaf Vs own sel (holoInnerAt T CT st hp).r) dec
  run := fun _ _ t Lb => decodeRun (holoAt st hp) (schedSteps T CT 0
    ((wmsgOf Vs (skLeaf Vs own sel (holoInnerAt T CT st hp).r) (holoInnerAt T CT st hp).r dec Lb).zip (t.map Prod.snd)))
  row := fun v x t => heldRow (innerCommitsAt (wr t).1 (wr t).2 st reg) ish.sparse v x

noncomputable abbrev C3Family.checked (ish : RecVStar.Inner) {ss : Array VStar.Stmt}
    (hss : RecVStar.statements ish = .ok ss) (bind : Fin (vstarCount ish) → VStar.Bindings)
    {Vs : Fin (vstarCount ish) → VStmt} (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    (zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg')
    (hCust : Flock.Premises.RecordCustody zr) (h : Typed zr) (coins : String → Option Flock.RecCoins.Source)
    (rows : Option (Array Flock.Registered.Port))
    (own : Fin (vstarCount ish) → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg)
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json) :
    C3Family T CT st hp xR L Reg Cm Reg' Reg₂ :=
  C3Family.honest ish hss bind hob
    (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows zr
      hCust h) own sel bc dec wr reg

end FlockC3.Compose
