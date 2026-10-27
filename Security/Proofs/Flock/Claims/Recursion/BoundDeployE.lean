import Proofs.Flock.Claims.Recursion.BoundDeploy
import Proofs.Flock.Claims.DeployE

/-!
# The bound record at the element-level verifier's steps (DecompB, option (b))

`RecDeploy.KSBoundDeployed` and its ∀-form `KSBoundRoute` take the element-level verifier's steps `nextE` as a
parameter. Option (b) instantiates them with the element-level inner verifier's (`FlockDeploy.VE`, the deployed plain
verifier with its digests of encodings dropped). What that reads, and what it doesn't:

* **The acceptance a bound reads is the deployed one at every `nextE`.** `KSBound` bounds `Pr[the compiled verifier
  accepts ∧ E* extracts no witness]`. The compiled verifier reads its inner verifier only for its rounds, which `deE`
  pins to the deployed plain verifier's (`FlockDeploy.V false`'s `k`), so it is one verifier at every `nextE`, columns,
  weights, joins, digest input and last-message reading (`recBound_V_bdDeployed`). The relation the witness is to meet
  is `deployPlainTrim`'s parser's, also at every `nextE`.
* **`nextE` moves only which inner views `E*` reads as accepted**, `IVStar`'s acceptance of them. At `VE` that
  acceptance is no stronger than at the deployed plain verifier's steps: `VE` accepts every message and coin list, and
  every run, the deployed plain verifier accepts, at every oracle (`FlockDeploy.nextE_acc_of_next`,
  `FlockDeploy.run_out_VE`).

So a bound proved at `nextE := VE.next` is a bound on the deployed compiled verifier's acceptance. Its premises at
`VE` are (P1) and (P4) at `VE`'s steps, not yet proved (`plainKSAbort_bd` and `lastBindsEC_deployed_sha512_closed` are
at the deployed plain verifier's own steps).
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **The compiled verifier is one verifier at every choice of the bound record**: it reads `IVStar` only for its
rounds, the deployed plain verifier's. So `KSBoundDeployed`'s acceptance event is the same at every element-level
verifier's steps, the element-level one (`FlockDeploy.VE`) among them. -/
theorem recBound_V_bdDeployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE nextE' : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE colsE' : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE wE' : Msg → ℝ)
    (join join' : List Msg → Msg) (rowIn rowIn' : Msg → Msg)
    (final final' : Msg → Option (List (List (Msg × Salt)) × Msg)) (w w' : Msg → ℝ) :
    (recBound rel b L (bdDeployed rel nextE colsE wE join rowIn final w).IVStar).V =
      (recBound rel b L (bdDeployed rel nextE' colsE' wE' join' rowIn' final' w').IVStar).V :=
  rfl

/-- At the element-level verifier's steps, `DE`'s verifier is `FlockDeploy.VE`. -/
theorem deE_VE (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) :
    (deE rel FlockDeploy.VE.next colsE wE).V = FlockDeploy.VE :=
  rfl

end RecDeploy
