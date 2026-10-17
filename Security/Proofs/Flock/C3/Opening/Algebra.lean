import Proofs.Flock.C3.Compose.Family

/-!
# Piece (e): RecOpening's algebra at opening-only shape

`HOpening`: at every accepted outcome, R's decoded opening steps play `opening` on the two registered claims
(`STACKED_CLAIMS`, in rep order) to the output Ligerito on R takes. The `rec-opening` units are opening-only: no
zerocheck or lincheck, so their algebra is O1–O6 at `m = mReg`, with `7 ≤ mReg` from the shape.
-/

namespace FlockC3.Opening

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(e) R's opening plays to its output** on the two registered claims. -/
def HOpening (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    let u := F.run R ω t Lb
    ∃ hreg : 7 ≤ (holoAt st hp).cls.shape.mReg,
      PlaysT (opening execArith hreg [(u.claim 0).toClaim, (u.claim 1).toClaim]) u.op (some u.opened)

end FlockC3.Opening
