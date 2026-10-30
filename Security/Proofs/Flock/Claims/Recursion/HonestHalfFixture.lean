import Proofs.Flock.Claims.Recursion.HonestHalfTops
import Proofs.Flock.Claims.Recursion.Assumptions.HonestFixture

/-!
# The honest half at the honest run (`honest_half_fixture`)

`honest_half_tops` at the one run `HonestFixture` names: some deployed recursive run at some statement `x` has every
part `s < nS x` accepting (`PartAccepts` at `RecursiveKSAt`'s record and layout), the honest witness of `RecursiveKS`'s
non-vacuity, with every premise in the one closed assumption.
-/

namespace RecDeploy.HonestHalfFixture

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy.VStarBuild
open RecDeploy.VStarHonest
open RecDeploy.HonestHalf

/-- **At the honest run, every V\* part accepts**: `HonestFixture`'s run, at the deployed brings at `x` under `deployedB`
and V\*'s layout of record `layR elemR`, has `PartAccepts` at every part `s < nS x` (`honest_half_tops`). -/
theorem honest_half_fixture (h : HonestFixture) :
    ∃ (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round)
      (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
      (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
      (w : Claims.Flock.Wit) (ω : Tape (bLAt mask Ir b0 x aux nOf hd recRest))
      (r : Fin ((DAt mask Ir b0 x aux nOf hd recRest).V.k x) → Coin),
      ∀ s < nS x,
        Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 (DAt mask Ir b0 x aux nOf hd recRest)
          (deployZkRec _root_.Flock.Readable.relation) (DeployedR.layR OpenRows.elemR) Claims.Recursion.Form.Dense
          (showsPart recRead ({ bringsAt mask Ir (deployedB b0) x with tr := recTr aux nOf hd recRest } : Brings))
          (cmtTr recRead) s x (Claims.coinSeq r)
          ((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w)
            (DAt mask Ir b0 x aux nOf hd recRest).V x ω r).msgs.map Prod.fst) := by
  obtain ⟨mask, Ir, b0, x, aux, nOf, hd, recRest, decI, w, st, ω, r, hR, hW, hN, hI, hT⟩ := h
  exact ⟨mask, Ir, b0, x, aux, nOf, hd, recRest, decI, w, ω, r,
    HonestHalfTops.honest_half_tops mask Ir b0 x aux nOf hd recRest decI ω r hR hW hN hI hT⟩

end RecDeploy.HonestHalfFixture
