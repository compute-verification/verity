import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.KSLin2Committed

/-!
# The recursive verifier's knowledge soundness at Hello-committed coins (trusted)

`RecursiveKS` reads the recursive verifier's run at coins drawn after each message. Deployed, the coin server commits
them at Hello in an hm96 coin tree (`coin-tree/hm96-sha512`) and opens each round's coin with its answer; a V\* session
before the last one that accepts is answered with `[]`, and no coin is opened there: that round is the session's
`Finish`, which C-Flock's live server answers with its verdict alone (it opens coins only at `Commit` and at each
`Round`, `committed_coins`), and the ZK server's contract lets coins out only as a scheduled round's coins, none at
`finish` (`Flock.Firewall.Contract.step`). `RecursiveKSDeployed` is
`RecursiveKS`'s bound at that run (`KSLin2OnC`, at `RecursiveKS`'s record, bound record, outer record, caps and
statements): at every leaf hiding bound `δ₁ ≠ ∞`, every hm96 tree whose leaves hide at `δ₁` (`Claims.Hm96Hiding`, the
named assumption `hash-derived-key` at the pinned key) has a simulator `S` such that, for every prover that reads the
root and the openings, the committed run accepts at most as often as `RecursiveKS`'s extractor, run on the prover with
`S`'s root and openings, recovers a witness, plus `RecursiveKS`'s bound at that prover and `2 · k(x) · δ₁`.
-/

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open scoped ENNReal

/-- **`RecursiveKSAt` at Hello-committed coins**, at the deployed reader, weights and last part's pieces, the brings
`b` and the coin server's record: `KSLin2OnC` where `RecursiveKSAt` has `KSLin2On`, at the leaf hiding bound `δ₁`. -/
def RecursiveKSDeployedAt (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) (δ₁ : ℝ≥0∞) : Prop :=
  Claims.Recursion.KSLin2OnC FlockDeploy.sha512
    (RecDeploy.recBoundAt _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
      (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR))
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
      RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
      RecDeploy.DeployedR.wR)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt δ₁

/-- **Knowledge soundness of the recursive verifier at Hello-committed coins** (`RecursiveKSDeployedAt` at the
deployed brings' setup), for every leaf hiding bound `δ₁`, every other field of the brings and every record the coin
server keeps. -/
def RecursiveKSDeployed : Prop :=
  ∀ (δ₁ : ℝ≥0∞) (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    RecursiveKSDeployedAt (RecDeploy.deployedB b) aux nOf hd recRest δ₁

end Flock.Guarantees
