import Proofs.Flock.Claims.Recursion.ResStepsWV
import Proofs.Flock.Claims.Recursion.OpenCoefTie

/-!
# `StepsW` at the deployed readers and brings

`stepsWV_deployedR` at KS's brings (`DecompBR.bB (deployedB b) …`, `recRead`), with the copy's coefficients
(`OpenCoef.copyCoefTieRP_deployed`) and the registered setup (`regSetup_deployedB`) closed. What is left is the next
rows' owners (`NextHeldW`).
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-- **`StepsWG` at `XsW … AccV` at the deployed readers and brings**, from the next rows' owners (`hN`). -/
theorem stepsWV_deployed (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE w : Msg → ℝ) (b : Brings)
    (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record)
    (rest : Msg → List (Msg × Claims.Out) → List UInt8) (cmt : Msg → ℕ → Msg → Msg)
    (hN : NextHeldW DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest)
      OpenRows.elemR Claims.Recursion.Form.Dense) :
    StepsWG DeployedR.rdR key wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest) OpenRows.elemR
      Claims.Recursion.Form.Dense recRead cmt
      (XsW DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest) OpenRows.elemR
        (AccV DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest) OpenRows.elemR
          Claims.Recursion.Form.Dense recRead cmt)) :=
  stepsWV_deployedR key wE w (DecompBR.bB (deployedB b) aux nOf hd rest) recRead cmt
    (OpenCoef.copyCoefTieRP_deployed DeployedR.rdR wE DeployedR.finalR w b aux nOf hd rest OpenRows.elemR cmt key)
    (regSetup_with_tr (regSetup_deployedB b) _) hN

end RecDeploy.ResLower
