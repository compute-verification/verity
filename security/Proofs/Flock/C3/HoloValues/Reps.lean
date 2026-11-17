module

public import Proofs.Flock.C3.Compose.Family
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

/-! The holo values: each rep's decoded steps play `holoRepC` to the decoded output its sparse phase reads
(`HReps`). -/

@[expose] public section

namespace FlockC3.HoloValues

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

def HReps (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    let e := holoAt st hp
    let S := e.cls.pubStatement e.regs
    let u := F.run R ω t Lb
    ∀ i : Fin 2,
      PlaysT (holoRepC (D := Dig) execArith i S e.sch e.cls.hm (extraClaims S e.cls.ptLocal e.cls.mPts u.pts))
        (u.ts i) (some (u.rep i), u.pend i) ∧
      (u.pend i).verdict u.os.1 = true ∧ OpensOKL H512 enc512 leafPlain u.cap₀ i (u.pend i) u.os.1 u.os.2

end FlockC3.HoloValues
