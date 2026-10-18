/-
`ProvedScope.statementRules` on plain SHA-512 leaves (`Tags.withPlainLeaves`): a recursive session's inner statement
(`Tags.circuitPlainLeaves`, hidden outputs) passes them without `--zk` and is refused under it; a statement with public
outputs and plain leaves is refused either way.
-/
import Flock.ProvedScope

namespace Flock.Test.PlainLeaves

def why (t : Tags) (zk : Bool) : Option String :=
  ProvedScope.firstFailure (ProvedScope.statementRules t zk 1 false)

#guard why Tags.circuitPlainLeaves false == none
#guard why Tags.circuitPlainLeaves true ==
  some "verity/flock-circuit+plain-leaves, a statement with flock-leaf/sha512-unsalted leaves under --zk"
#guard why Tags.circuit967b8d06.withPlainLeaves false ==
  some "verity/flock-circuit@967b8d06, a statement with flock-leaf/sha512-unsalted leaves"
#guard why Tags.circuit false == none
#guard why Tags.circuit true == none

end Flock.Test.PlainLeaves
