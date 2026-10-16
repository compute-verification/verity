import Proofs.Flock.Claims.Scope.Readable
import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Matches
import Proofs.Flock.Claims.Scope.Wires
import Proofs.Flock.Claims.Scope.Block
import Proofs.Flock.Claims.Scope.System
import Proofs.Flock.Claims.Scope.Tables
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.Deployed
import Proofs.Flock.Claims.Scope.Witness

/-! Flock's scope at the deployed verifier: the verifier's statement read as `Claims.Flock.Stmt` (`Readable`),
`ProvedScope.check` both ways (`Check`), what the setup gives the scope (`Setup`), the readable scope against the check
for every statement the verifier parses (`InScope`), and `Claims.Flock.ScopeMatches` at the verifier's parser and scope
check (`Matches`). The relation's fields: block R1CS over GF(2) as one AND/XOR block circuit (`Wires`, `Block`,
`System`), with `RelS` exactly the system's rows, pins and fixed bits in every block (`Sys.relS_iff`), and the deployed
relation (`Tables`: `relation`), the verifier's tables as that system, whose `RelS` is every table's model statement
(`Layout.model` at `regionsOf`) satisfied by its share of the hidden values (`relS_relation`); at every statement the
verifier parses, every table fits the first's blocks and has the verifier's layout (`Fit`: `tables_fit`,
`relS_layout`). At the deployed parser and scope check (`FlockDeploy.parse`, `FlockDeploy.scopeCheck`): `ScopeMatches`
exactly when no parsed, checked statement has `m < 22`, and under `--zk` always (`Deployed`). The tables' model
witnesses as a readable witness that satisfies `RelS` (`Witness`: `relS_witOf`). -/
