module

public import Proofs.Flock.C3.Compose.Tied
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
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Coins

/-! The source's rep steps (`StepsOfVals`). -/

@[expose] public section

namespace FlockC3.Reps

noncomputable section

section unit

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

open Flock FlockSoundness FlockSoundness.Refine FlockC3.HoloValues
open Flock.RecAlgebra (hOfW)
open FlockVBridge.Compiled (RepData)
open FlockSoundness.Discharge.Exec (kindOf weave)
open FlockSoundness.Discharge.PlaysCoin (coinArr toT_exact)

def TapeAlong (σ : List FlockSoundness.Discharge.Exec.Kind) (ms cs : List Game.Msg) : Prop :=
  ∃ l : List Game.Step, l.map kindOf <+: σ ∧ l.filterMap Sum.getLeft? = ms ∧ l.filterMap Sum.getRight? = cs

theorem tapeAlong_of_plays {α : Type} {g : Game α} {σ : List FlockSoundness.Discharge.Exec.Kind}
    (hs : FlockSoundness.Discharge.Exec.Spine g σ) {ms : List Game.Msg} {cs : List (Array Flock.F128)} {a : α}
    (h : Game.Plays fOf g ms cs a) : TapeAlong σ ms (cs.map (coinArr fOf)) := by
  obtain ⟨l, hp, h1, h2⟩ := toT_exact fOf h
  exact ⟨l, FlockSoundness.Discharge.Exec.Spine.prefix hp hs, h1, h2⟩

/-- The tape conjunct is needed: `weave` drops what runs past the shape and reads no message's type. -/
def Woven (σ : List FlockSoundness.Discharge.Exec.Kind) (ms cs : List Game.Msg) (ts : List Game.Step) : Prop :=
  ts = weave σ ms cs ∧ TapeAlong σ ms cs

theorem Woven.steps {σ : List FlockSoundness.Discharge.Exec.Kind} {ms cs : List Game.Msg} {ts : List Game.Step}
    (h : Woven σ ms cs ts) :
    ts.map kindOf <+: σ ∧ ts.filterMap Sum.getLeft? = ms ∧ ts.filterMap Sum.getRight? = cs := by
  obtain ⟨rfl, l, hk, rfl, rfl⟩ := h
  rw [FlockVBridge.Family.weave_of_prefix hk]
  exact ⟨hk, rfl, rfl⟩

def UnitWoven (S : Model.Statement) (sch₀ : FlockSoundness.Accounting.Schedule) (nX : ℕ) (ts : List Game.Step) (sh : Flock.Shape)
    (d : RepData) (v : UnitVals) : Prop :=
  Woven (holoRepShape 𝔽 𝕂 Dig S.m S.kLog nX sch₀) (holoRepMsgs v.st v.sch (2 + nX) v.p (hOfW sh d.w) capDig)
    ((consumed v.t.rounds v.t.k (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6)) ++
      consumed v.t.rounds (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6) + 1) v.t₄.k).map (coinArr fOf)) ts

theorem unitSteps_of_woven {S : Model.Statement} {sch₀ : FlockSoundness.Accounting.Schedule} {extra : List (Model.Claim 𝔽 S.m)}
    {cap₀ : Model.Cap Dig} {os : Model.Opens 𝔽 Dig} {r : ℕ} {ts : List Game.Step} {sh : Flock.Shape} {d : RepData}
    {v : UnitVals} (h : UnitWoven S sch₀ extra.length ts sh d v) (hcap : capDig v.p.cap = cap₀)
    (hos : ∀ ℓ j, os r ℓ j = openRow v.p.ligerito v.sch ℓ j) : UnitSteps S sch₀ extra cap₀ os r ts sh d v :=
  have h' := Woven.steps h
  ⟨h'.1, h'.2.1, h'.2.2, hcap, hos⟩

end unit

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

def RepStepsWoven (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals) : Prop :=
  AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
    UnitWoven ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch
      (extraClaims ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts (stp R ω t Lb).pts).length ((stp R ω t Lb).ts i) sh (rep R ω t Lb i)
        (vals R ω t Lb i)

theorem stepsOfVals_of_woven {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    (h : RepStepsWoven zs dg N k Rw stp sh rep vals)
    (hcap : AtRep zs dg N k Rw sh rep fun R ω t Lb i => capDig (vals R ω t Lb i).p.cap = (stp R ω t Lb).cap₀)
    (hos : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      ∀ ℓ j, (stp R ω t Lb).os.1 i ℓ j = openRow (vals R ω t Lb i).p.ligerito (vals R ω t Lb i).sch ℓ j) :
    StepsOfVals zs dg N k Rw (tiedSrc stp) sh rep vals := fun R ω t R₂ τ o hA i halg hacc =>
  unitSteps_of_woven (h R ω t R₂ τ o hA i halg hacc) (hcap R ω t R₂ τ o hA i halg hacc)
    (hos R ω t R₂ τ o hA i halg hacc)

end

end FlockC3.Reps
