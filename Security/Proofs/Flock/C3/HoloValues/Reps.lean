import Proofs.Flock.C3.Compose.Family

/-!
# Piece (c): the holo values

`HReps`: at every accepted outcome, each rep's decoded steps play `holoRepC` (the hidden units' lincheck,
`holoLincheck`, in place of the plain one) to the decoded output its sparse phase reads, and the openings accept the
rep's result and verify its level-0 paths against the session's cap. G's analogue is `HShape`, `HResL` and `HRuns`
through `algebraHolds_of_decodeL` and `run_of_algebra`, every one of which takes `sh.holo = false` or
`holoOf c = .ok false`; this piece is those lemmas with the side condition dropped (the gap list names them) and the
holo lincheck's values (`Algebra/Holo.lean`) in the rep's run.
-/

namespace FlockC3.HoloValues

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(c) Each rep plays `holoRepC` to its decoded output**, whose result the openings accept and whose level-0 paths
verify against the session's cap. -/
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
