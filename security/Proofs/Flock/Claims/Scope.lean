module

public import Proofs.Flock.Claims.Scope.Readable
public import Proofs.Flock.Claims.Scope.Check
public import Proofs.Flock.Claims.Scope.Setup
public import Proofs.Flock.Claims.Scope.InScope
public import Proofs.Flock.Claims.Scope.Sizes
public import Proofs.Flock.Claims.Scope.Matches
public import Proofs.Flock.Claims.Scope.System
public import Proofs.Flock.Claims.Scope.Tables
public import Proofs.Flock.Claims.Scope.Fit
public import Proofs.Flock.Claims.Scope.Deployed
public import Proofs.Flock.Claims.Scope.Split

/-! Flock's scope at the deployed verifier: the verifier's statement read as `Claims.Flock.Stmt` (`Readable`),
`ProvedScope.check` both ways (`Check`), what the setup gives the scope (`Setup`), the readable scope against the check
for every statement the verifier parses (`InScope`), the tables' `m` bound (`Sizes`: `ProvedScope.sizesOk`, which with
the check is `InScope`, and which no session `verify` accepts fails), and `Claims.Flock.ScopeMatches` at the verifier's
parser and scope check (`Matches`). The relation's fields: block R1CS over GF(2) as one Boolean block circuit
(`System`), with `RelS` exactly the system's rows, pins and fixed bits in every block (`Sys.relS_iff`), and the deployed
relation (`Tables`: `relation`), the verifier's tables as that system, whose `RelS` is every table's model statement
(`Layout.model` at `regionsOf`) satisfied by its share of the hidden values (`relS_relation`); at every statement the
verifier parses, every table fits the first's blocks and has the verifier's layout (`Fit`: `tables_fit`,
`relS_layout`), and at the deployed parser (`FlockDeploy.parse`) a parsed statement's `RelS` is its tables' models
satisfied (`Deployed`: `relS_deployed`). The relation splits by table: `RelS` holds exactly when each table's share satisfies that
table alone (`Split`: `relS_only`). `ScopeMatches` at the deployed records is `FlockDeploy.scopeMatches_zk` and
`FlockDeploy.scopeMatches_plain` (`DeployCode`), `Matches`' `scopeMatches` at the verifier's reading. -/

@[expose] public section
