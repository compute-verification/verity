import Proofs.Flock.Claims.Recursion.DecompBDeployed
import Proofs.Flock.Claims.Recursion.PartAcceptsPin

/-!
# `ResLowJ` with each part's check at its own witness

`ResLowJP` is `ResLowJ` at `JointP`: every part's `PartAcceptsP` at its own witness `ws[s]`, all of them at the parts'
own messages, with the recorded calls of every part true and free of collisions together. That is the acceptance
`RestLowR`'s assembly holds (`hJ`); `ResLowJ` weakens it to `Joint` (`joint_of_jointP`), which loses what ties one
part's copy to another's: the copy of a residual piece reads the frame values at rows only another part opens, and the
accumulator rows only the joint chain ties (`ResLower.PortsAt`).

* `ResLowJP`: the residual groups' share of `hlowP` at `JointP`.
* `resLowJP_of_resLowJ`: `ResLowJ` gives it.
-/

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB
open RecDeploy.OpenLower (JointP joint_of_jointP)
open RecDeploy.RestRead (DenseX kI)

noncomputable section

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (L : LayoutB)
  (resP : Msg → List ResPiece)
  (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (ownR : Msg → ResPiece → Option ℕ)

/-- What each part's circuit accepts at layout `L`, at its own witness. -/
abbrev acceptsLP :=
  PartAcceptsP FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
    (deployZkRec _root_.Flock.Readable.relation) L Claims.Recursion.Form.Dense (showsPart recRead (bB b aux nOf hd rest))
    (cmtTr recRead)

/-- **The residual groups' share at the parts' own witnesses and messages** (`hlowP`'s `rec-residuals` kind): where
every part accepts at its own witness and the parts' own messages (`JointP`), each group `ownR` gives part `s` holds. -/
def ResLowJP : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), parse x = some st → Claims.Recursion.Form.Dense st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
      JointP FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) L
        (acceptsLP rd b aux nOf hd rest wE final w L) x c ws →
      ∀ s < nS x, PiecesComputes (bdR rd wE final w) (recL rd b aux nOf hd rest wE final w L) resP
        (resHolds elemAt vvAt) ownR s x c (L.msgs x ws)

theorem resLowJP_of_resLowJ (h : ResLowJ rd b aux nOf hd rest wE final w L resP elemAt vvAt ownR) :
    ResLowJP rd b aux nOf hd rest wE final w L resP elemAt vvAt ownR :=
  fun x st hx hf c ws hJ => h x st hx hf c ws (joint_of_jointP hJ)

end

end RecDeploy.DecompBR
