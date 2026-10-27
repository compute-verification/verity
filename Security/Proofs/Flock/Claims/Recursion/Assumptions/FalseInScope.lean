import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.LamDeployed

/-!
# A false compiled statement in the deployed scope

The one fact `RecursiveKS`'s nontrivial lemma (`Flock.Guarantees.recursiveKS_nontrivial`) takes, and nothing proves:
some compiled statement the deployed reader takes is dense, in the scope `λ₂`'s numbers are stated at, and false. It
is a fact about one concrete statement, and the deployed reader reads a statement only through `jsonOf`, whose
`Lean.Json.Parser.anyCore` is a `partial def` the kernel can't unfold, and only where every V\* part's circuit is V\*'s
byte for byte (`RecDeploy.partsVStar`, 22 MB at the deployed shape), so no statement can be built in Lean for it.
Wherever no statement reads, `RecursiveKS` holds at the verifier that accepts everything
(`Flock.Guarantees.recursiveKSAtAll_of_readX_none`), so some such fact is needed.
-/

namespace RecDeploy

open Claims (Msg Out)

/-- **A false dense compiled statement in the deployed scope**: a statement `x` and a coin-server record (`aux`,
`nOf`, `hd`, `recRest`) where

* `x` is dense (`DenseAt`: every parse of it has the dense form);
* its inner statement has no witness in Flock's relation at the deployed bound record's inner Flock (`bdDeployed` at
  `RecursiveKS`'s deployed values);
* its inner statement is in Flock's plain scope (`FlockDeploy.PlainIn`), and each V\* part `j < nS x`, staged at the
  record's transcript (`recTr aux nOf hd recRest`), is in the parts' scope (`PartIn`);
* it has at most `2^16` parts and at most `2^56` inner setup inputs.

The scope conditions are `RecDeploy.lamLin2_deployed_ge_dep`'s premises, which give `λ₂ ≥ 100` there. Assumption
record. No claim: a fact about one concrete statement of the deployment, a deterministic computation on public data
that says nothing about the prover or the world, witnessed outside Lean by a fixture. The witness is the deployed
compiled statement over an inner statement whose claimed public output isn't its circuit's: the deployed reader takes it
(`RecDeploy.readX`), Flock's plain setup sets its inner statement up, and evaluating the inner circuit refutes its claim.
No witness run exists yet; this docstring is to name its run id or art id when it passes. -/
def FalseInScope : Prop :=
  ∃ (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8) (s : FlockDeploy.Ins) (spec : _root_.Flock.SessionSpec)
    (sts : Array _root_.Flock.Stmt),
    DenseAt x ∧
    (∀ v, ¬ Claims.Flock.R
      (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
        FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR).deploy.parse (xIOf x) v) ∧
    FlockDeploy.PlainIn (xIOf x) s spec sts ∧
    (∀ j < nS x, ∃ i spec' sts', PartIn (staged (Stage.ofMsg (recTr aux nOf hd recRest)) x
      (pick (Stage.ofMsg (recTr aux nOf hd recRest)) x j) j) i spec' sts') ∧
    nS x ≤ 2 ^ 16 ∧ (innerSetupIns (xIOf x)).length ≤ 2 ^ 56

end RecDeploy
