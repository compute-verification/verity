module

public import Proofs.Flock.C3.PubRows.Shape
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
import Proofs.Flock.C3.RegLevels.AccReads
import Proofs.Flock.C3.RegLevels.OpensRegs
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup

/-! `SharedOwnRoot`: two sessions register a shared value under one root, from `checkShared`. -/

@[expose] public section

namespace FlockC3.RegLevels

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion

def SharedValue (rows : Option (Array Flock.Registered.Port)) (v : String) : Prop :=
  (rows.bind (·.find? (·.value == v))).isSome ∨ (RecVStar.stackedTopLevel v).isSome

theorem site_of_checkShared {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (h : RecVStar.checkShared s own rows = .ok ())
    {p : Flock.Registered.Port} (hp : p ∈ own) (hv : SharedValue rows p.value) :
    ∃ w, rows.bind (·.find? (·.value == p.value)) = some w ∧ RecVStar.portDiff? p w false = none := by
  unfold RecVStar.checkShared at h
  rw [← Array.forIn_toList] at h
  obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
  refine FlockC3.PubRows.rows_forIn_each (fun p => SharedValue rows p.value →
      ∃ w, rows.bind (·.find? (·.value == p.value)) = some w ∧ RecVStar.portDiff? p w false = none)
    ?_ _ h p (Array.mem_toList_iff.2 hp) hv
  intro p r hr
  split at hr
  · rename_i w hw
    split at hr
    · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
      cases h2
    · rename_i hn
      refine ⟨fun _ => ⟨w, hw, ?_⟩, ?_⟩
      · cases hd : RecVStar.portDiff? p w false with
        | none => rfl
        | some d => exact absurd hd (hn d)
      · cases hr; rfl
  · rename_i hw
    have hs : (RecVStar.stackedTopLevel p.value).isSome = false := by
      cases hs : (RecVStar.stackedTopLevel p.value).isSome
      · rfl
      · first | (rw [hs] at hr; cases hr) | (simp only [hs] at hr; cases hr) | simp [hs] at hr
    have ht : (RecVStar.topLevel p.value).isSome = false := by
      cases ht : (RecVStar.topLevel p.value).isSome
      · rfl
      · first | (rw [hs, ht] at hr; cases hr) | (simp only [hs, ht] at hr; cases hr) | simp [hs, ht] at hr
    first | rw [hs, ht] at hr | simp only [hs, ht, Bool.false_eq_true, ↓reduceIte] at hr
    refine ⟨fun hv => ?_, by cases hr; rfl⟩
    rcases hv with hv | hv
    · rw [hw] at hv; cases hv
    · rw [hs] at hv; cases hv

theorem shared_of_checkShared {s s' : VStar.Stmt} {own own' : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (h : RecVStar.checkShared s own rows = .ok ())
    (h' : RecVStar.checkShared s' own' rows = .ok ()) {p p' : Flock.Registered.Port} (hp : p ∈ own) (hp' : p' ∈ own')
    (hv : p.value = p'.value) (hs : SharedValue rows p.value) :
    p.root = p'.root ∧ p.leaves = p'.leaves ∧ p.domain = p'.domain ∧ p.schema = p'.schema := by
  obtain ⟨w, hw, hd⟩ := site_of_checkShared h hp hs
  obtain ⟨w', hw', hd'⟩ := site_of_checkShared h' hp' (hv ▸ hs)
  rw [← hv, hw] at hw'
  cases hw'
  have e := (site_of_portDiff hd).trans (site_of_portDiff hd').symm
  simp only [FlockVBridge.Family.site, Prod.mk.injEq] at e
  exact ⟨e.2.1, e.2.2.1, e.2.2.2.2.2.1, e.2.2.2.2.2.2⟩

section Family

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

/-- Only a value the verifier's shared rows `rows` register, or R's tops (`SharedValue`); any other value is
not covered. -/
def SharedOwnRoot (rows : Option (Array Flock.Registered.Port)) (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) :
    Prop :=
  ∀ R ω t R₂ j j', (∃ tr, AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr) →
    (∃ tr, AcceptsZKR (F.zs R ω t R₂ j').I (F.zs R ω t R₂ j').own tr) →
    ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j').own, p.value = p'.value → SharedValue rows p.value →
      p.root = p'.root ∧ p.leaves = p'.leaves ∧ p.domain = p'.domain ∧ p.schema = p'.schema

theorem sharedOwnRoot_of_checks {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hca : FlockC3.PubRows.ChecksAccepted F rows wr reg) : SharedOwnRoot rows F := by
  intro R ω t R₂ j j' hex hex' p hp p' hp' hv hs
  have one : ∀ j, (∃ tr, AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr) →
      ∃ s, RecVStar.checkShared s (F.zs R ω t R₂ j).own rows = .ok () := by
    intro j hex
    obtain ⟨tr, -, -, spec, sts, hs, hfm⟩ := hca R ω t R₂ j hex
    obtain ⟨h0, -⟩ := setup_zero_c (F.zs R ω t R₂ j).ht (F.zs R ω t R₂ j).hTags.1 (F.zs R ω t R₂ j).hcH hs
    obtain ⟨-, s, -, -, -, -, h⟩ := regs_of_checkStmt (array_forM_ok hfm 0 h0)
    exact ⟨s, h⟩
  obtain ⟨s, h⟩ := one j hex
  obtain ⟨s', h'⟩ := one j' hex'
  exact shared_of_checkShared h h' hp hp' hv hs

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}

theorem sharedOwnRoot_checked {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} :
    SharedOwnRoot rows (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) :=
  sharedOwnRoot_of_checks (FlockC3.PubRows.checksAccepted_checked (coins := coins) (rows := rows) (zr := zr)
    (hCust := hCust) (h := h))

end Family

end FlockC3.RegLevels
