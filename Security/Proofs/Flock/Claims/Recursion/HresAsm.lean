import Proofs.Flock.Claims.Recursion.ResLowerDD
import Proofs.Flock.Claims.Recursion.ResChainW
import Proofs.Flock.Claims.Recursion.RestVerdictBasesHold
import Proofs.Flock.Claims.Recursion.RestOpenedSized

/-!
# KS's residual leaf at the deployed readers (`hres_deployed`)

`hres_deployed` is the `ResLoweredPD` that `recursiveKSAt_of_leaves_deployed` takes as `hres`: at the deployed reader
`DeployedR.rdR`, its last message's reading `DeployedR.finalR`, the elements `OpenRows.elemR`, the deployed brings
`bB (deployedB b) …`, the dense form, `recRead` and `cmtTr recRead`. It is `resLowered_DB` there, with

* the base checks closed (`basesHold_spec`, at every reader);
* `ReadsAt` closed (`RestElem.readsAt_elemR_closed`);
* P3 from its two halves (`accPorts_of_W`): the steps (`StepsW`) and the ends (`EndsW`);
* P2 (`FramePorts`).

Open, each a named hypothesis: `hF` (P2), `hS` (`StepsW`) and `hE` (`EndsW`). `hres_deployed_all` is the form
`recursiveKS_of_resLowered` takes, at the deployed weights.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-- **KS's residual leaf at the deployed readers**: `ResLoweredPD` at `DeployedR.rdR`, `DeployedR.finalR`,
`OpenRows.elemR` and the deployed brings, from the frame ports (`hF`), the steps half (`hS`) and the ends half (`hE`);
the base checks and `ReadsAt` are closed. -/
theorem hres_deployed (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round)
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

/-- **`hres_deployed` for every brings and record**, at the deployed weights and one key: the hypothesis
`recursiveKS_of_resLowered` takes. -/
theorem hres_deployed_all (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hF : ∀ (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      FramePorts DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR
        (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR Claims.Recursion.Form.Dense recRead
        (cmtTr recRead))
    (hS : ∀ (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      StepsW DeployedR.rdR key DeployedR.wER DeployedR.finalR DeployedR.wR
        (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR Claims.Recursion.Form.Dense recRead
        (cmtTr recRead))
    (hE : ∀ (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      EndsW DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR
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
    (hF b aux nOf hd recRest) (hS b aux nOf hd recRest) (hE b aux nOf hd recRest)⟩

end RecDeploy.ResLower
