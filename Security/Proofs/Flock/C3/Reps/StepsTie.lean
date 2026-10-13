import Proofs.Flock.C3.Reps.Steps

/-!
# `RepStepsWoven` and the openings from the reps' parts

`repStepsWoven_of_tie`: `RepStepsWoven` at a step source `stp` from the reps' other parts at the values
(`ProofOfVals`, `SetupOfVals`, `FramesOfVals`) and the source's weave (`RepWeaveAt`). Its tape conjunct is the rep's
run on the values' messages and coins (`tapeAlong_of_parts`: `rep_of_unit`'s run at the proof's rows as openings, then
`tapeAlong_of_plays`). Its weave conjunct and the openings (`hos`) are no tie's: `SrcRepTie` is made only from
`RepUnitOfAlg`, whose steps' part is `StepsOfVals` itself, and `RepsMsgsRead` says only that two accepted proofs' reps
have the same messages. They are the source's definition at the reps, so they hold by `rfl` at `repsOfVals`, `stp`
with its reps' steps and openings read off the values (`repWeaveAt_repsOfVals`, `hos_repsOfVals`); there `StepsOfVals`
needs only the other parts and the cap (`stepsOfVals_repsOfVals`).
-/

namespace FlockC3.Reps

noncomputable section

section unit

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

open Flock FlockSoundness FlockSoundness.Refine FlockC3.HoloValues
open Flock.RecAlgebra (hOfW)
open FlockVBridge.Compiled (RepData AlgebraHolds AccHolds)
open FlockSoundness.Discharge.Exec (weave)
open FlockSoundness.Discharge.PlaysCoin (coinArr)

/-- **A rep unit's messages and coin steps at `v`, woven along `holoRepShape`** at `S`, `nX` extra claims and `sch₀`
(`UnitWoven`'s steps). -/
def unitWeave (S : Model.Statement) (sch₀ : Accounting.Schedule) (nX : ℕ) (sh : Flock.Shape) (d : RepData)
    (v : UnitVals) : List Game.Step :=
  weave (holoRepShape 𝔽 𝕂 Dig S.m S.kLog nX sch₀) (holoRepMsgs v.st v.sch (2 + nX) v.p (hOfW sh d.w) capDig)
    ((consumed v.t.rounds v.t.k (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6)) ++
      consumed v.t.rounds (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6) + 1) v.t₄.k).map (coinArr fOf))

/-- **A rep's messages and coins are a tape along its shape** from its unit's other parts: the program runs
(`holoProg_of_zclc`), refines to `holoRepC` (`holoRep_refines`) at the proof's rows as openings, and that run is a
tape along `holoRepC`'s spine (`tapeAlong_of_plays`). -/
theorem tapeAlong_of_parts {S : Model.Statement} {sch₀ : Accounting.Schedule} {extra : List (Model.Claim 𝔽 S.m)}
    {sh : Flock.Shape} {d : RepData} {v : UnitVals} (hm : 13 ≤ S.m) (alg : AlgebraHolds sh d) (acc : AccHolds sh d)
    (hp : UnitProof sh d v) (hs : UnitSetup S sch₀ extra sh v) (hf : UnitFrames sh d v) :
    TapeAlong (holoRepShape 𝔽 𝕂 Dig S.m S.kLog extra.length sch₀)
      (holoRepMsgs v.st v.sch (2 + extra.length) v.p (hOfW sh d.w) capDig)
      ((consumed v.t.rounds v.t.k (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6)) ++
        consumed v.t.rounds (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6) + 1) v.t₄.k).map (coinArr fOf)) := by
  have h22 := exec_fast100_ge hs.fast
  obtain ⟨hprog, zc₁, t₁, h₁, hk₁⟩ := holoProg_of_zclc sh d alg acc v.st v.sch v.p v.extras v.t hf.k0 hp.coins hf.wf
    hs.m (by omega) hs.kL hp.zp hp.lp hp.foldPts hf.n₀ hf.n₁ hf.n₂ hf.nr hf.a₀ hf.a₁ hf.a₂ hf.ar hf.lc hf.cl hp.rs
    hp.rsShape hp.rsVal hf.rsRounds hf.lg
  have hdims := fast100_dims v.st.m v.sch hs.fast
  have hlv : ∀ i < v.sch.levels.size, LevelSeam (v.sch.levels.getD i default) := fun i hi =>
    fast100_level v.st.m v.sch hs.fast _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  obtain ⟨-, kz, -, -, -, hkz, _, _, _, hpl, -⟩ := holoRep_refines v.st S hm (by have := hs.mS; omega) v.sch hs.mS
    hs.kS hs.fold extra v.extras hs.extrasLen hs.extrasRel hdims.1 hlv
    (fun i hi => fast100_alpha v.st.m v.sch hs.fast _
      (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi))
    (hdims.2.trans hs.mS) v.p (hOfW sh d.w) 0 v.t v.t₄ hf.wf hprog capDig
    (fun _ ℓ j => openRow v.p.ligerito v.sch ℓ j) (fun _ _ _ _ => rfl)
  have hkz' := hkz zc₁ t₁ h₁
  rw [hk₁] at hkz'
  subst hkz'
  rw [hs.sched] at hpl
  exact tapeAlong_of_plays (holoRepC_spine (D := Dig) execArith 0 S sch₀ hm extra) hpl

end unit

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {stp : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
  {sh : Flock.Shape}
  {rep : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
  {vals : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Hypothesis: the step source's rep steps are the values' weave** (`RepStepsWoven`'s first conjunct): rep `i`'s
steps in `stp` are `unitWeave` of `vals R ω t Lb i` at the run's statement, schedule and extra claims. No tie reads
it; it is the source's definition at the reps, 0 lines at `repsOfVals`. -/
def RepWeaveAt (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals) : Prop :=
  AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
    (stp R ω t Lb).ts i = unitWeave ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch
      (extraClaims ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts (stp R ω t Lb).pts).length sh (rep R ω t Lb i) (vals R ω t Lb i)

/-- **`RepStepsWoven` from the reps' parts**: the weave is the source's (`RepWeaveAt`), and the values' messages and
coins are a tape along the shape by the rep's run on them (`tapeAlong_of_parts`). -/
theorem repStepsWoven_of_tie (hw : RepWeaveAt zs dg N k Rw stp sh rep vals)
    (hpr : ProofOfVals zs dg N k Rw sh rep vals) (hs : SetupOfVals zs dg N k Rw (tiedSrc stp) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals) : RepStepsWoven zs dg N k Rw stp sh rep vals :=
  fun R ω t R₂ τ o hA i halg hacc => ⟨hw R ω t R₂ τ o hA i halg hacc,
    tapeAlong_of_parts (holoAt st hp).cls.hm halg hacc (hpr R ω t R₂ τ o hA i halg hacc)
      (hs R ω t R₂ τ o hA i halg hacc) (hf R ω t R₂ τ o hA i halg hacc)⟩

/-- **The step source with its reps read off the values**: `stp` with rep `i`'s steps the values' weave
(`unitWeave`) and its openings at reps 0 and 1 the values' proof's rows (`openRow`); the rest is `stp`'s. -/
def repsOfVals (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals) :
    Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp) := fun R ω t Lb =>
  { stp R ω t Lb with
    ts := fun i => unitWeave ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch
      (extraClaims ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts (stp R ω t Lb).pts).length sh (rep R ω t Lb i) (vals R ω t Lb i)
    os := (fun q ℓ j => if h : q < 2 then openRow (vals R ω t Lb ⟨q, h⟩).p.ligerito (vals R ω t Lb ⟨q, h⟩).sch ℓ j
      else (stp R ω t Lb).os.1 q ℓ j, (stp R ω t Lb).os.2) }

theorem repWeaveAt_repsOfVals : RepWeaveAt zs dg N k Rw (repsOfVals stp sh rep vals) sh rep vals :=
  fun _ _ _ _ _ _ _ _ _ _ => rfl

/-- **The openings at `repsOfVals`' reps are the values' proof's rows** (`StepsOfVals`' `hos`). -/
theorem hos_repsOfVals : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
    ∀ ℓ j, (repsOfVals stp sh rep vals R ω t Lb).os.1 i ℓ j =
      openRow (vals R ω t Lb i).p.ligerito (vals R ω t Lb i).sch ℓ j :=
  fun _ _ _ _ _ _ _ i _ _ _ _ => dite_eq_left i.isLt

/-- **`StepsOfVals` at the source that reads its reps off the values**, from the reps' other parts and the cap. -/
theorem stepsOfVals_repsOfVals (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hs : SetupOfVals zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hcap : AtRep zs dg N k Rw sh rep fun R ω t Lb i => capDig (vals R ω t Lb i).p.cap = (stp R ω t Lb).cap₀) :
    StepsOfVals zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep vals :=
  stepsOfVals_of_woven (repStepsWoven_of_tie repWeaveAt_repsOfVals hpr hs hf) hcap hos_repsOfVals

end

end FlockC3.Reps

#print axioms FlockC3.Reps.unitWeave
#print axioms FlockC3.Reps.tapeAlong_of_parts
#print axioms FlockC3.Reps.RepWeaveAt
#print axioms FlockC3.Reps.repStepsWoven_of_tie
#print axioms FlockC3.Reps.repsOfVals
#print axioms FlockC3.Reps.repWeaveAt_repsOfVals
#print axioms FlockC3.Reps.hos_repsOfVals
#print axioms FlockC3.Reps.stepsOfVals_repsOfVals
