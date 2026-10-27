import Proofs.Flock.Claims.Recursion.ResStepsW
import Proofs.Flock.Claims.Recursion.AccTies
import Proofs.Flock.Claims.Recursion.OpenLevW

/-!
# `StepsW` at the 22:25Z ruling's side condition (`XsW` at `AccV`)

`stepsWG_of_commits` where the chain's commit strings are `accCommitV` (the brings list the registered reads,
`RegSetup`), and at the deployed readers with the copy's outputs (`levelOutsW`) and openings (`openCopyW_deployed`)
closed. What is left is the copy's coefficients (`CoefW`, RECR's `copyCoefTieRP_deployed`) and the next rows' owners
(`NextHeldW`). `accPorts_of_G` (`ResEndsW`) takes the result to P3.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-- **`StepsWG` at `XsW … AccV`**, wherever the brings list the registered reads (`hRS`), from the copy's outputs
(`hL`), openings (`hO`), coefficients (`hC`) and the next rows' owners (`hN`). -/
theorem stepsWV_of (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (untr : TrReader)
    (cmt : Msg → ℕ → Msg → Msg)
    (hL : OpenLower.AtPiecesP FlockDeploy.sha512 (DRL rd wE final w b elem) Claims.Recursion.Form.Dense untr b cmt
      (DeployedR.layR elem) (Rows.rowDPieces key) fun x _ _ LANES HH _ _ wt p ht q =>
        _root_.Flock.Readable.LevelOuts
          (OpenLower.copyR (Claims.Recursion.xI (DRL rd wE final w b elem) x) wt p ht q) LANES HH)
    (hO : OpenLower.AtPiecesP FlockDeploy.sha512 (DRL rd wE final w b elem) Claims.Recursion.Form.Dense untr b cmt
      (DeployedR.layR elem) (Rows.rowDPieces key) fun x c ws LANES HH _ _ wt p ht q =>
        AtDecW rd wE final w b elem x c ws fun ms =>
          OpenCopy (Rows.sideAt rd (Claims.Recursion.xI (DRL rd wE final w b elem) x) ms q) q LANES HH
            (OpenLower.copyR (Claims.Recursion.xI (DRL rd wE final w b elem) x) wt p ht q))
    (hC : CoefW rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt) (hRS : RegSetup b)
    (hN : NextHeldW rd wE final w b elem Claims.Recursion.Form.Dense) :
    StepsWG rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt
      (XsW rd wE final w b elem (AccV rd wE final w b elem Claims.Recursion.Form.Dense untr cmt)) :=
  stepsWG_of_commits rd key wE final w b elem untr cmt _ hL hO hC
    (fun x st hx hf c ws hJ hΦ => by
      obtain ⟨xs, hxs, hcol, hms⟩ := hΦ
      exact accCommitV rd wE final w b elem untr cmt hRS x st hx hf c ws hJ xs hxs hcol hms)
    hN

/-- **`StepsWG` at `XsW … AccV` at the deployed readers**: `stepsWV_of` with the copy's outputs (`levelOutsW`) and
openings (`openCopyW_deployed`). -/
theorem stepsWV_deployedR (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE w : Msg → ℝ) (b : Brings) (untr : TrReader)
    (cmt : Msg → ℕ → Msg → Msg)
    (hC : CoefW DeployedR.rdR key wE DeployedR.finalR w b OpenRows.elemR Claims.Recursion.Form.Dense untr cmt)
    (hRS : RegSetup b) (hN : NextHeldW DeployedR.rdR wE DeployedR.finalR w b OpenRows.elemR Claims.Recursion.Form.Dense) :
    StepsWG DeployedR.rdR key wE DeployedR.finalR w b OpenRows.elemR Claims.Recursion.Form.Dense untr cmt
      (XsW DeployedR.rdR wE DeployedR.finalR w b OpenRows.elemR
        (AccV DeployedR.rdR wE DeployedR.finalR w b OpenRows.elemR Claims.Recursion.Form.Dense untr cmt)) :=
  stepsWV_of DeployedR.rdR key wE DeployedR.finalR w b OpenRows.elemR untr cmt
    (levelOutsW DeployedR.rdR key wE DeployedR.finalR w b OpenRows.elemR _ untr cmt (fun _ h => h))
    (openCopyW_deployed key wE w b untr cmt) hC hRS hN

end RecDeploy.ResLower
