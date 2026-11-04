import Proofs.Flock.Claims.Recursion.ResPiecesD
import Proofs.Flock.Claims.Recursion.ResLowerDeployed

/-!
# `ResLowJ` at the concrete `rec-residuals` pieces, from `ResLowered`

`ResLowJ` (the residual groups' share of `hlowP`, at the parts' own messages) at `resPD` and `ownRD` follows from
the per-part share `ResLower.ResLowered` at the deployed layout: `ownRD` gives a piece only to a statement V\* lists
as an algebra part (`.inner`), which is `resLowered_parts`' `hin`, and a `Joint` acceptance is every part's.

* `ownRD_inner`: the part `ownRD` names is an algebra part of V\*'s list at the inner shape `readX` read.
* `resLowJ_of_resLowered`: `ResLowJ` at `DeployedR.layR elem`, `resPD` and `ownRD`.
-/

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

theorem ownRD_inner {x : Msg} {q : ResPiece} {s : ℕ} (h : ownRD x q = some s) :
    ∃ (r : XR) (ss : Array VStar.Stmt) (j : ℕ) (nm : String) (pt : VStar.Part), readX x = some r ∧
      RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some (.inner j nm pt) := by
  unfold ownRD at h
  cases hr : readX x with
  | none => rw [hr] at h; cases h
  | some r =>
    rw [hr] at h
    cases hss : RecVStar.statements r.shape with
    | error e => simp [hss, Except.toOption] at h
    | ok ss =>
      simp only [hss, Except.toOption, Option.bind_some] at h
      obtain ⟨hsz, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 h
      refine ⟨r, ss, ?_⟩
      cases he : ss[s] with
      | inner j nm pt => exact ⟨j, nm, pt, rfl, hss, by rw [Array.getElem?_eq_getElem hsz, he]⟩
      | _ => rw [he] at hp; simp [isResOf] at hp

/-- **`ResLowJ` from the per-part share**: at the deployed layout `DeployedR.layR elem`, wherever each algebra
part's accepting circuit computes its pieces (`ResLowered`), every part's does at a `Joint` acceptance. -/
theorem resLowJ_of_resLowered (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (h : ResLower.ResLowered rd wE final w (bB b aux nOf hd rest) elem Claims.Recursion.Form.Dense recRead
      (cmtTr recRead) resPD ownRD elemAt vvAt) :
    ResLowJ rd b aux nOf hd rest wE final w (DeployedR.layR elem) resPD elemAt vvAt ownRD :=
  fun _ _ _ _ _ _ hJ s hs =>
    ResLower.resLowered_parts h (fun _ _ _ _ ho => ownRD_inner ho) (hJ.2.2.2 s hs)

end RecDeploy.DecompBR
