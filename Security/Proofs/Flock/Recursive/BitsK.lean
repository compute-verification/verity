import Proofs.Flock.Recursive.KSProved
import Proofs.Flock.Claims.Recursion.KappaEpsDeployed

/-!
# `Pr[acc ∧ ¬R] ≤ κ + t·ε` at the deployed reader

`recursiveBitsK`: `Flock.SecurityProofs.RecursiveKS` through `Claims.Recursion.bitsLin2KOn_of_ksLin2On`, so at the
deployment's dense statements `Pr[acc ∧ ¬R] ≤ κ + t·ε`, `κ` once a session and `ε` once per SHA-512 call of the
prover's run, under `StrictBirthday` at `B` and `ExpectedCR` at `B_I`, `Fdev`, `FCI` and `F0I`, the named SHA-512
collision assumptions. `recursiveBitsK_deployed` evaluates it at the deployment's sizes (`SizesAt`, call bounds
`n, v ≤ 2^32`) with `kappa_eps_deployed`: `Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}`.

Both are stated at the brings `b` (`RecursiveBitsKAt`, `RecursiveBitsKBoundAt`), as `Flock.Guarantees.RecursiveKSAt`
is, and hold at `deployedB b`. A statement that writes `{ deployedB b with … }` itself carries a `let`, which the
elaborator's `let`-to-`have` pass type-checks through the deployed records past the heartbeat limit.
-/

namespace RecDeploy

open Claims (Msg Out)

/-- **`Pr[acc ∧ ¬R] ≤ κ + t·ε` at the deployed reader**, at the brings `b` and the coin server's record (`aux`, `nOf`,
`hd`, `recRest`): `BitsLin2KOn` where `Flock.Guarantees.RecursiveKSAt` states `KSLin2On`, at the dense statements. -/
def RecursiveBitsKAt (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) : Prop :=
  Claims.Recursion.BitsLin2KOn FlockDeploy.sha512
    (recBoundAt _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd recRest }
      (DeployedR.layR OpenRows.elemR))
    (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
      FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR)
    (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt

/-- **`Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}` at the deployed reader**, at the brings `b` and the coin server's
record: `BitsKAt` at `RecursiveBitsKAt`'s records, at every dense statement at the deployment's sizes (`SizesAt b`) and
call bounds `n ≤ 2^32` and `v ≤ 2^32`. -/
def RecursiveBitsKBoundAt (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) : Prop :=
  Claims.Recursion.BitsKAt FlockDeploy.sha512
    (recBoundAt _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd recRest }
      (DeployedR.layR OpenRows.elemR))
    (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
      FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR)
    (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep (fun x => DenseAt x ∧ SizesAt b x)
    (2 ^ 32) (2 ^ 32) (2 ^ (-(167.7 : ℝ))) (2 ^ (-(174.95 : ℝ)))

/-- **`RecursiveBitsKAt` from `RecursiveKSAt`** at the same brings and record: `bitsLin2KOn_of_ksLin2On`
(`sessionAccepts`). -/
theorem recursiveBitsKAt_of (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (h : _root_.Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR DeployedR.wER
      DeployedR.wR b aux nOf hd recRest) :
    RecursiveBitsKAt b aux nOf hd recRest :=
  Claims.Recursion.bitsLin2KOn_of_ksLin2On
    (sessionAccepts
      (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
        FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR).IVStar
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck
      { b with tr := recTr aux nOf hd recRest, rd := (DeployedR.layR OpenRows.elemR).readers }) h

/-- **`RecursiveBitsKBoundAt` from `RecursiveBitsKAt`** at the same brings and record: `bitsKAt_of` with
`kappa_eps_deployed`. -/
theorem recursiveBitsKBoundAt_of (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (h : RecursiveBitsKAt b aux nOf hd recRest) : RecursiveBitsKBoundAt b aux nOf hd recRest := by
  refine Claims.Recursion.bitsKAt_of h (fun _ hx => hx.1) ?_
  rintro decI x n v ⟨-, ⟨s, spec, sts, hin, h1⟩, hO, hn, hS⟩ hU hV
  exact kappa_eps_deployed _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd recRest }
    (DeployedR.layR OpenRows.elemR) (FlockDeploy.nextR DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
    FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR decI hin h1 hO hn hS hU hV

/-- **`Pr[acc ∧ ¬R] ≤ κ + t·ε` at the deployed reader**, at the deployment's dense statements: `RecursiveKS` through
`recursiveBitsKAt_of`. -/
theorem recursiveBitsK (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    RecursiveBitsKAt (deployedB b) aux nOf hd recRest :=
  recursiveBitsKAt_of (deployedB b) aux nOf hd recRest (_root_.Flock.SecurityProofs.RecursiveKS b aux nOf hd recRest)

/-- **`Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}` at the deployed reader**, at every dense statement at the
deployment's sizes (`SizesAt (deployedB b)`) and call bounds `n ≤ 2^32` and `v ≤ 2^32`: `recursiveBitsK` with
`kappa_eps_deployed`. -/
theorem recursiveBitsK_deployed (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    RecursiveBitsKBoundAt (deployedB b) aux nOf hd recRest :=
  recursiveBitsKBoundAt_of (deployedB b) aux nOf hd recRest (recursiveBitsK b aux nOf hd recRest)

end RecDeploy
