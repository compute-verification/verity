import Proofs.Flock.Claims.Recursion.KSLin2R
import Proofs.Flock.Claims.Recursion.DeployedR
import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.OpenRowsDeployed

/-!
# The recursive verifier's knowledge soundness at the deployed reader (trusted)

`RecursiveKSAt` is `KSLin2On` at SHA-512 at the deployed relation (`Flock.Readable.relation`), at the bound record
whose steps are the deployed recursive verifier's (`FlockDeploy.nextR rd`) and whose columns are the ones it opens,
trimmed (`FlockDeploy.deployPlainTrimR`), at V\*'s layout of record `DeployedR.layR rest` (every hm96 call the
deployed circuit performs, recorded), the coin server's record
(`recTr aux nOf hd recRest`), the deployed caps (`FlockDeploy.KS.CpDep`) and the dense statements (`DenseAt`).

It is stated at the deployment's parameters (the reader `rd`, V\*'s last message's reader `final`, the last part's
pieces `rest`, the weights `wE` and `w`, the brings `b` and the record's `aux`, `nOf`, `hd` and `recRest`) so that the
guarantee `RecursiveKS` is this at the deployed values: the reader `DeployedR.rdR`, V\*'s last message's reader
`DeployedR.finalR`, the last part's pieces `OpenRows.elemR`, the weights `DeployedR.wER` and `DeployedR.wR` and
the brings' setup `deployedB` (the compiled verifier's own evaluations), for every other field of the brings and every
record the coin server keeps (`aux`, `nOf`, `hd`, `recRest`).
-/

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

/-- **Knowledge soundness of the recursive verifier at a compiled record** `D`: `KSLin2On` at SHA-512, the deployed
relation's bound at the reader `rd` (its steps `FlockDeploy.nextR rd`, its trimmed columns, V\*'s last message's reader
`final` and the weights `wE`, `w`), the deployed caps and the dense statements. `RecursiveKSAt` is this at the
deployment's record (`recBoundAt`); a statement about another compiled record (one that accepts every message, say)
names its own. -/
def RecursiveKSAtRec (D : Claims.Recursion.RecAt Coin) (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ) : Prop :=
  Claims.Recursion.KSLin2On FlockDeploy.sha512 D
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt

/-- **Knowledge soundness of the recursive verifier at the deployed reader**, at the deployment's parameters: for
every inner and outer decoder meeting the list-decoding spec, every dense statement and every prover, the probability
that the recursive verifier accepts and the extractor recovers no witness is at most `κ + √(γ·Pr[B])` plus the inner
verifier's bad-abort, forked-coin and first-coin terms (`KSLin2On`, through `RecursiveKSAtRec` at the deployment's
compiled record). -/
def RecursiveKSAt (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) : Prop :=
  RecursiveKSAtRec
    (RecDeploy.recBoundAt _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
      (RecDeploy.DeployedR.layR rest)) rd final wE w

/-- **Knowledge soundness of the recursive verifier at the deployed reader** (`RecursiveKSAt` at the deployed values),
for every other field of the brings and every record the coin server keeps. -/
def RecursiveKS : Prop :=
  ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    RecursiveKSAt RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.OpenRows.elemR RecDeploy.DeployedR.wER
      RecDeploy.DeployedR.wR (RecDeploy.deployedB b) aux nOf hd recRest

end Flock.Guarantees
