import Proofs.Flock.Claims.Scope.Readable
import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Matches

/-! Flock's scope at the deployed verifier: the verifier's statement read as `Claims.Flock.Stmt` (`Readable`),
`ProvedScope.check` both ways (`Check`), what the setup gives the scope (`Setup`), the readable scope against the check
for every statement the verifier parses (`InScope`), and `Claims.Flock.ScopeMatches` at the verifier's parser and scope
check (`Matches`). -/
