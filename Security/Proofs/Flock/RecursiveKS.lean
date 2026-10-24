import Proofs.Flock.RecursiveKS.Statement
import Proofs.Flock.Claims.Recursion.RecOpen
import Proofs.Flock.Claims.Witnesses.RecursionMeets

/-!
# The proof of the recursion's knowledge soundness at the compiled V\*

* `Flock.SecurityProofs.RecursiveKS : Flock.Guarantees.RecursiveKS` is `RecDeploy.ksLin2_deployed_partAccepts`, its
  `DecompB` premise at the one shape it lists (`Form.Dense`) being `Flock.Assumptions.RecKSDecomp`.

What the statement is not vacuous on:

* `recKSReadsBack_recTr`: the obligation `RecKSReadsBack` holds wherever the brings' transcript is the model of the coin
  server's record (`recTr`), at every layout and every choice of the code side's; `RecDeploy.recRead_honest` reads an
  honest three-round run back.
* `recursiveKS_dense_inScope`: the recursion's readable scope holds a dense statement (`denseStmt false`, the one
  `Claims.Witness.Recursion.Complete_meets` names), so `DenseAt` is not a scope only statements that fail to parse meet.

`RecKSDecomp` and `RecKSReadersTrue` have no witness here: they are open.
-/

namespace Flock.SecurityProofs

/-- **The recursion is knowledge sound at the compiled V\*, at its dense statements**, under its three named
obligations. -/
theorem RecursiveKS : Flock.Guarantees.RecursiveKS := by
  intro b rest wE join final w hrb hD hRT
  refine RecDeploy.ksLin2_deployed_partAccepts b rest wE join final w hrb (fun shape hs => ?_) hRT
  rw [List.mem_singleton] at hs
  subst hs
  exact hD

end Flock.SecurityProofs

namespace RecDeploy

open Flock
open Claims (Msg Out)

/-- **`RecKSReadsBack` at the model of the coin server's record**: wherever the brings' transcript is `recTr` (whatever
its `aux`, `nOf`, `hd` and `rest'`), the obligation holds at every layout and every `wE`, `join`, `final`, `w`
(`readsBack_recTr`). -/
theorem recKSReadsBack_recTr (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest' : Msg → List (Msg × Out) → List UInt8)
    (hb : b.tr = recTr aux nOf hd rest')
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    Flock.Assumptions.RecKSReadsBack b rest wE join final w :=
  readsBack_recTr _ _ _ _ aux nOf hd rest' hb

/-- **The readable scope holds a dense statement**: `denseStmt false` is dense (neither Holo nor sparse) and in the
recursion's readable scope (`Claims.Witness.Recursion.denseStmt_inScope`). -/
theorem recursiveKS_dense_inScope :
    (Claims.Witness.Recursion.denseStmt false).form.Dense ∧
      Claims.Recursion.InScope (Claims.Witness.Recursion.denseStmt false) :=
  ⟨⟨rfl, rfl⟩, Claims.Witness.Recursion.denseStmt_inScope false⟩

end RecDeploy
