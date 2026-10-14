import Proofs.Flock.C3.Compose.Family

/-!
# Piece (f): R's levels on the chain

`HRegLevels`: at every accepted outcome, R's decoded Ligerito steps (`rec-top-stacked-L<l>` and R's levels, which the
chain runs after both reps and R's opening) play `ligeritoC` on stream 2 at R's registered schedule to R's pending
result, which the openings accept and whose level-0 paths verify against the registered cap (`xR.cap`, the record's
`stacked` entry by `StackedTop`): R's tops and its binding round.
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(f) R's levels play to R's result**, which the openings accept against the registered cap. -/
def HRegLevels (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    let u := F.run R ω t Lb
    PlaysT (ligeritoC (D := Dig) execArith 2 (holoAt st hp).rsch.sch u.opened.T u.opened.b) u.lg u.pendR ∧
      u.pendR.verdict u.os.1 = true ∧ OpensOKL H512 enc512 leafPlain xR.cap 2 u.pendR u.os.1 u.os.2

end FlockC3.RegLevels
