import Proofs.Flock.C3.Reps.Tie
import Proofs.Flock.C3.Compose.Tied

/-!
# The source's rep steps (`StepsOfVals`)

`stepsOfVals_of_woven`: `StepsOfVals` at the tied source `tiedSrc stp`, whose steps, link points, cap and openings are
`stp`'s, from `RepStepsWoven`, the cap (`hcap`) and the openings (`hos`). `RepStepsWoven`: each rep's steps in `stp`
are `UnitSteps`' messages and coin steps at the values, woven along the rep's shape, and those are a tape's along it
(`TapeAlong`), since `weave` alone drops what runs past the shape and reads no message's type. A tape along a shape is
its messages and coins woven along it (`weave_of_prefix`), so the steps take the tape's kinds, messages and coin steps
(`Woven.steps`, `unitSteps_of_woven`). A run on the values' messages and coins gives the tape (`tapeAlong_of_plays`).
-/

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

/-- `ms` and `cs` are the messages and coin steps of a tape whose kinds are a prefix of `σ`. -/
def TapeAlong (σ : List FlockSoundness.Discharge.Exec.Kind) (ms cs : List Game.Msg) : Prop :=
  ∃ l : List Game.Step, l.map kindOf <+: σ ∧ l.filterMap Sum.getLeft? = ms ∧ l.filterMap Sum.getRight? = cs

/-- **A run on messages and coins gives a tape along its game's spine** (`toT_exact`, `Spine.prefix`). -/
theorem tapeAlong_of_plays {α : Type} {g : Game α} {σ : List FlockSoundness.Discharge.Exec.Kind}
    (hs : FlockSoundness.Discharge.Exec.Spine g σ) {ms : List Game.Msg} {cs : List (Array Flock.F128)} {a : α}
    (h : Game.Plays fOf g ms cs a) : TapeAlong σ ms (cs.map (coinArr fOf)) := by
  obtain ⟨l, hp, h1, h2⟩ := toT_exact fOf h
  exact ⟨l, FlockSoundness.Discharge.Exec.Spine.prefix hp hs, h1, h2⟩

/-- Steps `ts` are `ms` and `cs` woven along `σ`, and those are a tape's along `σ`. -/
def Woven (σ : List FlockSoundness.Discharge.Exec.Kind) (ms cs : List Game.Msg) (ts : List Game.Step) : Prop :=
  ts = weave σ ms cs ∧ TapeAlong σ ms cs

/-- **Woven steps are the tape**: their kinds are a prefix of `σ`, their messages `ms` and their coin steps `cs`. -/
theorem Woven.steps {σ : List FlockSoundness.Discharge.Exec.Kind} {ms cs : List Game.Msg} {ts : List Game.Step}
    (h : Woven σ ms cs ts) :
    ts.map kindOf <+: σ ∧ ts.filterMap Sum.getLeft? = ms ∧ ts.filterMap Sum.getRight? = cs := by
  obtain ⟨rfl, l, hk, rfl, rfl⟩ := h
  rw [FlockVBridge.Family.weave_of_prefix hk]
  exact ⟨hk, rfl, rfl⟩

/-- **A rep's steps woven from its unit's values**: `UnitSteps`' messages (`holoRepMsgs`) and coin steps at `v`,
woven along `holoRepShape` at `S`, `nX` extra claims and `sch₀`. -/
def UnitWoven (S : Model.Statement) (sch₀ : Accounting.Schedule) (nX : ℕ) (ts : List Game.Step) (sh : Flock.Shape)
    (d : RepData) (v : UnitVals) : Prop :=
  Woven (holoRepShape 𝔽 𝕂 Dig S.m S.kLog nX sch₀) (holoRepMsgs v.st v.sch (2 + nX) v.p (hOfW sh d.w) capDig)
    ((consumed v.t.rounds v.t.k (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6)) ++
      consumed v.t.rounds (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6) + 1) v.t₄.k).map (coinArr fOf)) ts

/-- **`UnitSteps` from woven steps**, the cap and the openings. -/
theorem unitSteps_of_woven {S : Model.Statement} {sch₀ : Accounting.Schedule} {extra : List (Model.Claim 𝔽 S.m)}
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
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The step source's rep `i` is the values' proof, woven**: its steps are `holoRepMsgs` of `(vals R ω t Lb i).p` and
the coins the record `(vals R ω t Lb i).t` consumes but `z_partial`'s, woven along `holoRepShape` at the run's
statement, schedule and extra claims, and those are a tape's along it (`UnitWoven`). -/
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

/-- **`StepsOfVals` at the tied source** from `stp`'s woven rep steps, its cap and its openings, which `tie` keeps. -/
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

#print axioms FlockC3.Reps.TapeAlong
#print axioms FlockC3.Reps.tapeAlong_of_plays
#print axioms FlockC3.Reps.Woven
#print axioms FlockC3.Reps.Woven.steps
#print axioms FlockC3.Reps.UnitWoven
#print axioms FlockC3.Reps.unitSteps_of_woven
#print axioms FlockC3.Reps.RepStepsWoven
#print axioms FlockC3.Reps.stepsOfVals_of_woven
