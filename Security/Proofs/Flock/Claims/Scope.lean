import Proofs.Flock.Claims.Scope.Readable
import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Sizes
import Proofs.Flock.Claims.Scope.Matches
import Proofs.Flock.Claims.Scope.Wires
import Proofs.Flock.Claims.Scope.Block
import Proofs.Flock.Claims.Scope.System
import Proofs.Flock.Claims.Scope.Tables
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.Deployed
import Proofs.Flock.Claims.Scope.Split

/-! Flock's scope at the deployed verifier: the verifier's statement read as `Claims.Flock.Stmt` (`Readable`),
`ProvedScope.check` both ways (`Check`), what the setup gives the scope (`Setup`), the readable scope against the check
for every statement the verifier parses (`InScope`), the tables' `m` bound (`Sizes`: `ProvedScope.sizesOk`, which with
the check is `InScope`, and which no session `verify` accepts fails), and `Claims.Flock.ScopeMatches` at the verifier's
parser and scope check (`Matches`). The relation's fields: block R1CS over GF(2) as one AND/XOR block circuit (`Wires`, `Block`,
`System`), with `RelS` exactly the system's rows, pins and fixed bits in every block (`Sys.relS_iff`), and the deployed
relation (`Tables`: `relation`), the verifier's tables as that system, whose `RelS` is every table's model statement
(`Layout.model` at `regionsOf`) satisfied by its share of the hidden values (`relS_relation`); at every statement the
verifier parses, every table fits the first's blocks and has the verifier's layout (`Fit`: `tables_fit`,
`relS_layout`), and at the deployed parser (`FlockDeploy.parse`) a parsed statement's `RelS` is its tables' models
satisfied (`Deployed`: `relS_deployed`). The tables' model witnesses as a readable witness that satisfies `RelS`
(`Witness`: `relS_witOf`). The relation splits by table: `RelS` holds exactly when each table's share satisfies that
table alone (`Split`: `relS_only`). `ScopeMatches` at the deployed records is `FlockDeploy.scopeMatches_zk` and
`FlockDeploy.scopeMatches_plain` (`DeployCode`), `Matches`' `scopeMatches` at the verifier's reading. -/
