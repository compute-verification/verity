import Proofs.Flock.Claims.Recursion.ResLowerDD
import Proofs.Flock.Claims.Recursion.ResChainW
import Proofs.Flock.Claims.Recursion.ResStepsW
import Proofs.Flock.Claims.Recursion.ResEndsW
import Proofs.Flock.Claims.Recursion.OpenCoefTie
import Proofs.Flock.Claims.Recursion.OpenLevW
import Proofs.Flock.Claims.Recursion.RestVerdictBasesHold
import Proofs.Flock.Claims.Recursion.RestOpenedSized

/-!
# KS's residual leaf at the deployed readers (`hres_deployed`)

`hres_deployed` is the `ResLoweredPD` that `recursiveKSAt_of_leaves_deployed` takes as `hres`: at the deployed reader
`DeployedR.rdR`, its last message's reading `DeployedR.finalR`, the elements `OpenRows.elemR`, the deployed brings
`bB (deployedB b) …`, the dense form, `recRead` and `cmtTr recRead`. It is `resLowered_DB` there, with

* the base checks closed (`basesHold_spec`, at every reader);
* `ReadsAt` closed (`RestElem.readsAt_elemR_closed`);
* P3 from its steps and ends (`accPorts_of_tie`):
  * the steps (`StepsW`, `stepsW_of_commits`) from each row D copy's outputs (`levelOutsW`) and openings
    (`openCopyW_deployed`), both closed, its coefficients (`copyCoefTieRP_deployed`, closed), the chain's commit
    strings (`AccCommitW`) and the next rows' owners (`NextHeldW`);
  * the ends (`EndsW`, `endsW_of`) from the algebra parts' commit tie (`InnerAccTie`);
* P2 (`FramePorts`).

Open, each a named hypothesis: `hF` (`FramePorts`), `hCm` (`AccCommitW`), `hN` (`NextHeldW`) and `hT`
(`InnerAccTie`). `hres_deployed_W` is the same leaf from `StepsW` and `EndsW` whole, and `hres_deployed_all` the form
`recursiveKS_of_resLowered` takes, at the deployed weights.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-- **KS's residual leaf at the deployed readers, from P2 and P3's halves**: `ResLoweredPD` at `DeployedR.rdR`,
`DeployedR.finalR`, `OpenRows.elemR` and the deployed brings, from the frame ports (`hF`), the steps half (`hS`) and the
ends half (`hE`); the base checks and `ReadsAt` are closed. -/
theorem hres_deployed_W (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hF : FramePorts DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest)
      OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead))
    (hS : StepsW DeployedR.rdR key wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest)
      OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead))
    (hE : EndsW DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest)
      OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead)) :
    ResLoweredPD DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR
      Claims.Recursion.Form.Dense recRead (cmtTr recRead) DecompBR.resPD DecompBR.ownRD
      (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead :=
  resLowered_DB DeployedR.rdR key wE DeployedR.finalR w b aux nOf hd recRest OpenRows.elemR
    (basesHold_spec DeployedR.rdR) RestElem.readsAt_elemR_closed hF
    (accPorts_of_W DeployedR.rdR key wE DeployedR.finalR w _ OpenRows.elemR recRead (cmtTr recRead) hS hE)

/-- **`StepsW` at the deployed readers** from the chain's commit strings (`hCm`) and the next rows' owners (`hN`): each
row D copy's outputs (`levelOutsW`), openings (`openCopyW_deployed`) and coefficients (`copyCoefTieRP_deployed`) are
closed. -/
theorem stepsW_deployed (wE w : Msg → ℝ) (b : Brings) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8)
    (hCm : AccCommitW DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest)
      OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead))
    (hN : NextHeldW DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest)
      OpenRows.elemR Claims.Recursion.Form.Dense) :
    StepsW DeployedR.rdR key wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR
      Claims.Recursion.Form.Dense recRead (cmtTr recRead) := by
  have hL := levelOutsW DeployedR.rdR key wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest)
    OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead) (fun _ h => h)
  have hO := openCopyW_deployed key wE w (DecompBR.bB (deployedB b) aux nOf hd recRest) recRead (cmtTr recRead)
  have hC := OpenCoef.copyCoefTieRP_deployed DeployedR.rdR wE DeployedR.finalR w b aux nOf hd recRest OpenRows.elemR
    (cmtTr recRead) key
  unfold LevelOutsW at hL
  unfold OpenCopyW at hO
  unfold OpenCoef.CopyCoefTieRP at hC
  exact stepsW_of_commits DeployedR.rdR key wE DeployedR.finalR w _ OpenRows.elemR recRead (cmtTr recRead) hL hO hC
    hCm hN

/-- **KS's residual leaf at the deployed readers**: `ResLoweredPD` at `DeployedR.rdR`, `DeployedR.finalR`,
`OpenRows.elemR` and the deployed brings, from the frame ports (`hF`), the chain's commit strings (`hCm`), the next rows'
owners (`hN`) and the algebra parts' commit tie (`hT`). -/
theorem hres_deployed (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hF : FramePorts DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest)
      OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead))
    (hCm : AccCommitW DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest)
      OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead))
    (hN : NextHeldW DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest)
      OpenRows.elemR Claims.Recursion.Form.Dense)
    (hT : InnerAccTie DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest)
      OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead)) :
    ResLoweredPD DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR
      Claims.Recursion.Form.Dense recRead (cmtTr recRead) DecompBR.resPD DecompBR.ownRD
      (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead :=
  hres_deployed_W wE w b aux nOf hd recRest key hF (stepsW_deployed wE w b key aux nOf hd recRest hCm hN)
    (endsW_of DeployedR.rdR wE DeployedR.finalR w _ OpenRows.elemR recRead (cmtTr recRead) hT)

/-- **`hres_deployed` for every brings and record**, at the deployed weights and one key: the hypothesis
`recursiveKS_of_resLowered` takes. -/
theorem hres_deployed_all (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hF : ∀ (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      FramePorts DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR
        (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR Claims.Recursion.Form.Dense recRead
        (cmtTr recRead))
    (hCm : ∀ (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      AccCommitW DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR
        (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR Claims.Recursion.Form.Dense recRead
        (cmtTr recRead))
    (hN : ∀ (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      NextHeldW DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR
        (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR Claims.Recursion.Form.Dense)
    (hT : ∀ (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      InnerAccTie DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR
        (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR Claims.Recursion.Form.Dense recRead
        (cmtTr recRead)) :
    ∀ (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      ∃ key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ,
        ResLoweredPD DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR
          (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR Claims.Recursion.Form.Dense recRead
          (cmtTr recRead) DecompBR.resPD DecompBR.ownRD (Rows.elemReadC (Top.accRC DeployedR.rdR key))
          Rows.resVRead :=
  fun b aux nOf hd recRest => ⟨key, hres_deployed DeployedR.wER DeployedR.wR b aux nOf hd recRest key
    (hF b aux nOf hd recRest) (hCm b aux nOf hd recRest) (hN b aux nOf hd recRest) (hT b aux nOf hd recRest)⟩

end RecDeploy.ResLower
