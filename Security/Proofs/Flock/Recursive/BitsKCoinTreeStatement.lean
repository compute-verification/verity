module

public import Proofs.Flock.Recursive.BitsK
public import Proofs.Flock.Claims.Recursion.BitsKCommitted

/-!
# `Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}` at Hello-committed coins (trusted)

`RecDeploy.recursiveBitsK_deployed` bounds the recursive verifier's `Pr[acc ∧ ¬R]` by `κ + t·ε` at the deployment's
sizes, `κ ≤ 2^{−167.7}` once a session and `ε ≤ 2^{−174.95}` once per SHA-512 call of the prover's run, at coins drawn
after each message. Deployed, the coin server commits them at Hello in an hm96 coin tree (`coin-tree/hm96-sha512`) and
opens each round's coin with its answer, except at a V\* session's `Finish` (`Flock.Guarantees.RecursiveKSDeployed`).
`RecursiveBitsKDeployed` is that bound at the committed run (`BitsKAtC`, at `recursiveBitsK_deployed`'s records,
caps, statements, call bounds and numbers): at every leaf hiding bound `δ₁ ≠ ∞` and every statement of the
deployment's sizes (`DenseAt ∧ SizesAt`), every hm96 tree whose leaves hide at `δ₁` (`Claims.Hm96Hiding`, the named
assumption `hash-derived-key` at the pinned key) has a simulator `S` such that, where the statement's inner calls and
setup calls are each at most `2^32`, for every prover that reads the root and the openings and whose run with `S`'s
root and openings makes at most `t` SHA-512 calls, under `StrictBirthday` at `B` and `ExpectedCR` at `B_I`, `Fdev`,
`FCI` and `F0I` there, the committed run accepts at most as often as the extractor, run on that simulating prover,
recovers a witness, plus `2^{−167.7} + t·2^{−174.95}` and `2 · k(x) · δ₁`.
-/

@[expose] public section

namespace Flock.Guarantees

open Claims (Msg Out)
open scoped ENNReal

/-- **`RecDeploy.RecursiveBitsKBoundAt` at Hello-committed coins**, at the deployed reader, weights and last part's
pieces, the brings `b` and the coin server's record: `BitsKAtC` where `RecursiveBitsKBoundAt` has `BitsKAt`, at the leaf
hiding bound `δ₁`. -/
def RecursiveBitsKDeployedAt (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) (δ₁ : ℝ≥0∞) : Prop :=
  Claims.Recursion.BitsKAtC FlockDeploy.sha512
    (RecDeploy.recBoundAt _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
      (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR))
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
      RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
      RecDeploy.DeployedR.wR)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep
    (fun x => RecDeploy.DenseAt x ∧ RecDeploy.SizesAt { b with tr := RecDeploy.recTr aux nOf hd recRest } x)
    (2 ^ 32) (2 ^ 32) (2 ^ (-(167.7 : ℝ))) (2 ^ (-(174.95 : ℝ))) δ₁

/-- **`Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}` at Hello-committed coins** (`RecursiveBitsKDeployedAt` at the
deployed brings' setup), for every leaf hiding bound `δ₁`, every other field of the brings and every record the coin
server keeps. -/
def RecursiveBitsKDeployed : Prop :=
  ∀ (δ₁ : ℝ≥0∞) (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    RecursiveBitsKDeployedAt (RecDeploy.deployedB b) aux nOf hd recRest δ₁

end Flock.Guarantees
