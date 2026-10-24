import Proofs.Flock.RecursiveKS.Statement
import Proofs.Flock.Claims.Witnesses.RecursionMeets

/-!
# The proof of the recursion's knowledge-soundness target statement at the compiled V\*

* `Flock.SecurityProofs.RecursiveKS : Flock.Guarantees.RecursiveKS` is `RecDeploy.ksLin2_deployed_partAccepts` (which
  proves `ReadsBackB` at the coin server's record and `ReadersTrue` at `hmSlotsCommit_of_full`), its one premise,
  `DecompB` at `PartAccepts` at the one shape it lists (`Form.Dense`), being `Flock.Assumptions.RecKSDecomp`.

The statement is vacuous at today's V\*: `RecKSDecomp` is not satisfiable there (`IVStar` parses the record JSON, a
message V\* holds under a hiding leaf, which no part can own). What is not vacuous is the rest of its scope:

* `recTr` is the coin server's record: `test_lean_rec_record.py` ties it to the coin server's bytes on the honest
  fixtures, and `RecDeploy.recRead_honest` reads an honest three-round run back.
* `recursiveKS_dense_inScope`: the recursion's readable scope holds a dense statement (`denseStmt false`, the one
  `Claims.Witness.Recursion.Complete_meets` names), so `DenseAt` is not a scope only statements that fail to parse meet.
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)

/-- **The target statement of the recursion's knowledge soundness at the compiled V\* holds**, under its named
obligation, which is not satisfiable at today's V\* (`Flock.Assumptions.RecKSDecomp`): the implication, not knowledge
soundness. -/
theorem _root_.Flock.SecurityProofs.RecursiveKS : Flock.Guarantees.RecursiveKS := by
  intro b aux nOf hd rest fin wE join final w hD
  refine ksLin2_deployed_partAccepts b aux nOf hd rest fin wE join final w (fun shape hs => ?_)
  rw [List.mem_singleton] at hs
  subst hs
  exact hD

/-- **The readable scope holds a dense statement**: `denseStmt false` is dense (neither Holo nor sparse) and in the
recursion's readable scope (`Claims.Witness.Recursion.denseStmt_inScope`). -/
theorem recursiveKS_dense_inScope :
    (Claims.Witness.Recursion.denseStmt false).form.Dense ∧
      Claims.Recursion.InScope (Claims.Witness.Recursion.denseStmt false) :=
  ⟨⟨rfl, rfl⟩, Claims.Witness.Recursion.denseStmt_inScope false⟩

end RecDeploy
