/-
`ProvedScope.check`'s table cap: a `--zk` session of more than `zkTableCap` tables is refused, with the text the live
build's mirror (`proved_scope.rs`, `statement_why`) gives; one of `zkTableCap` tables passes the statement's rules, and
without `--zk` the rule is still at most one table.
-/
import Flock.ProvedScope

namespace Flock.Test.TableCap

def why (zk : Bool) (count : Nat) : String :=
  match ProvedScope.check Tags.circuit zk ByteArray.empty ByteArray.empty [] count false false with
  | .error e => e
  | .ok () => ""

#guard why true 65537 ==
  "refused: 65537 tables in a --zk session, more than 65536: a statement form the soundness proof does not cover"
#guard why false 2 ==
  "refused: 2 tables in a session without --zk: a statement form the soundness proof does not cover"
#guard ProvedScope.firstFailure (ProvedScope.statementRules Tags.circuit true ProvedScope.zkTableCap false) == none
#guard ProvedScope.firstFailure (ProvedScope.statementRules Tags.circuit false 1 false) == none

end Flock.Test.TableCap
