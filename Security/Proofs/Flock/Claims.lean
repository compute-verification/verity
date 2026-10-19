import Models.Flock.Guarantees.Claims
import Proofs.Flock.Claims.RecursionStated
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Bridge
import Proofs.Flock.Claims.Numbers
import Proofs.Flock.Claims.NumbersTable
import Proofs.Flock.Claims.NumbersPads
import Proofs.Flock.Claims.NumbersGamma
import Proofs.Flock.Claims.NumbersNMin
import Proofs.Flock.Claims.Recursion
import Proofs.Flock.Claims.NumbersDeploy
import Proofs.Flock.Claims.NumbersRadius
import Proofs.Flock.Claims.Recursion.InnerAccepts
import Proofs.Flock.Claims.Recursion.InnerAcceptsKS
import Proofs.Flock.Claims.Recursion.DeployInputs
import Proofs.Flock.Claims.Recursion.DeployOuter
import Proofs.Flock.Claims.Recursion.Deploy
import Proofs.Flock.Claims.Recursion.DeployComplete
import Proofs.Flock.Claims.Recursion.DeployScope
import Proofs.Flock.Claims.Recursion.DeployFacts
import Proofs.Flock.Claims.Recursion.DeploySessions
import Proofs.Flock.Claims.Recursion.DeployScopeRuns
import Proofs.Flock.Claims.Recursion.DeployMargin
import Proofs.Flock.Claims.Recursion.DeployPremises
import Proofs.Flock.Claims.Recursion.DeployHonest
import Proofs.Flock.Claims.Recursion.DeployZK
import Proofs.Flock.Claims.Recursion.Seq
import Proofs.Flock.Claims.Recursion.SeqSim
import Proofs.Flock.Claims.Recursion.SeqSession
import Proofs.Flock.Claims.Recursion.SeqPrefix
import Proofs.Flock.Claims.Recursion.SeqCap
import Proofs.Flock.Claims.Recursion.SeqKey
import Proofs.Flock.Claims.Recursion.SeqDeploy
import Proofs.Flock.Claims.Recursion.DeployStop
import Proofs.Flock.Claims.Recursion.VStar
import Proofs.Flock.Claims.Recursion.VStarBound
import Proofs.Flock.Claims.Recursion.BoundDeploy
import Proofs.Flock.Claims.Recursion.PlainKSAbortFails
import Proofs.Flock.Claims.Recursion.BoundRun
import Proofs.Flock.Claims.Recursion.BoundForks
import Proofs.Flock.Claims.Recursion.PlainKSAbortDeployed
import Proofs.Flock.Claims.Recursion.VStarParts
import Proofs.Flock.Claims.Recursion.VStarPartsHolo
import Proofs.Flock.Claims.Recursion.VStarPartsDense
import Proofs.Flock.Claims.Recursion.DeployReg
import Proofs.Flock.Claims.Recursion.VStarPartsSparse
import Proofs.Flock.Claims.Recursion.VStarPartsSparseDeploy
import Proofs.Flock.Claims.Recursion.VStarLinks
import Proofs.Flock.Claims.Recursion.VStarCross
import Proofs.Flock.Claims.Recursion.DeployChecked
import Proofs.Flock.Claims.Recursion.DeployLayout
import Proofs.Flock.Claims.Counted
import Proofs.Flock.Claims.Recursion
import Proofs.Flock.Claims.Decode
import Proofs.Flock.Claims.DecodeFlock
import Proofs.Flock.Claims.ZkDecode
import Proofs.Flock.Claims.DecodeFamily
import Proofs.Flock.Claims.ZkDecodeFlock
import Proofs.Flock.Claims.ZkCols
import Proofs.Flock.Claims.ZkQueryRound
import Proofs.Flock.Claims.Scope
import Proofs.Flock.Claims.DeployOC
import Proofs.Flock.Claims.DeployInputs
import Proofs.Flock.Claims.JsonEq
import Proofs.Flock.Claims.CanonEq
import Proofs.Flock.Claims.Deploy
import Proofs.Flock.Claims.DeployPlainCode
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.DeployTables
import Proofs.Flock.Claims.KS
import Proofs.Flock.Claims.KS.ZkForkClash
import Proofs.Flock.Claims.KS.ZkPadsSched
import Proofs.Flock.Claims.RootFindGF128
import Proofs.Flock.Claims.GS
import Proofs.Flock.Claims.DeployRounds
import Proofs.Flock.Claims.DeployRoundsZk
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.Claims.DeployEfficient
import Proofs.Flock.Claims.Assembled
import Proofs.Flock.Claims.ZK
import Proofs.Flock.Claims.CompleteFails
import Proofs.Flock.Claims.Sampling
import Proofs.Flock.Claims.Witnesses

/-! `Flock.SecurityProofs.ClaimsStated : Flock.Guarantees.ClaimsStated`, by reflexivity: the guarantee is there for its
record, which holds the claims' text.

The claims' lemmas sit under `Proofs.Flock.Claims`, one module per piece, imported above; each module's docstring
says what it proves. -/

/-- **The claims are stated** (`Flock.Guarantees.ClaimsStated`). -/
theorem Flock.SecurityProofs.ClaimsStated : Flock.Guarantees.ClaimsStated := rfl
