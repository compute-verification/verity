import Proofs.Flock.C3.Compose.Family

/-!
# Piece (d): RecSparse at session level

`HSparse`: at every accepted outcome, each rep's decoded sparse steps play `sparsePhase` on that rep's public values
(`holoPub` of its zerocheck and lincheck outputs, its `h`) to the registered claim R's opening takes. The unit level is
done (`VBridge/Sparse/`: `check_sparse`, `laid_sparse`, `sparse_refines`, `sparsePhase_plays`); this piece lifts it to
the sessions' accepted layers, as `HRuns` lifts `run_of_algebra`.
-/

namespace FlockC3.Sparse

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(d) Each rep's sparse phase plays to its registered claim**, on that rep's public values. -/
def HSparse (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    let e := holoAt st hp
    let u := F.run R ω t Lb
    ∀ i : Fin 2,
      PlaysT (sparsePhase execArith e.cls.shape
          (holoPub (e.cls.pubStatement e.regs) e.cls.shape e.cls.units (u.rep i).1 (u.rep i).2)
          (fun σ => (e.τ σ).val) (u.rep i).2.h)
        (u.sp i) (some (u.claim i))

end FlockC3.Sparse
