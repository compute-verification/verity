/-
`ProvedScope.check`'s table cap: a `--zk` session of more than `zkTableCap` tables is refused, with the text the live
build's mirror (`proved_scope.rs`, `statement_why`) gives; one of `zkTableCap` tables passes the statement's rules, and
without `--zk` the rule is still at most one table. Under `--plain-tables` (`ProvedScope.checkP`) a plain session of
public outputs passes the cap at up to `Tags.MAX_TABLES` tables and is refused at 65, and a statement with hidden outputs
is refused at any count.
-/
import Flock.ProvedScope

namespace Flock.Test.TableCap

def why (zk : Bool) (count : Nat) : String :=
  match ProvedScope.check Tags.circuit zk ByteArray.empty ByteArray.empty [] count false false with
  | .error e => e
  | .ok () => ""

def whyP (tags : Tags) (count : Nat) : String :=
  match ProvedScope.checkP tags ByteArray.empty ByteArray.empty [] count false false with
  | .error e => e
  | .ok () => ""

#guard why true 65537 ==
  "refused: 65537 tables in a --zk session, more than 65536: a statement form the soundness proof does not cover"
#guard why false 2 ==
  "refused: 2 tables in a session without --zk: a statement form the soundness proof does not cover"
#guard ProvedScope.firstFailure (ProvedScope.statementRules Tags.circuit true ProvedScope.zkTableCap false) == none
#guard ProvedScope.firstFailure (ProvedScope.statementRules Tags.circuit false 1 false) == none
#guard whyP Tags.circuit967b8d06 65 ==
  "refused: 65 tables in a session under --plain-tables, more than 64: a statement form the soundness proof does not cover"
#guard whyP Tags.circuit 2 == "refused: verity/flock-circuit, a statement with hidden outputs under --plain-tables: " ++
  "a statement form the soundness proof does not cover"
#guard ProvedScope.plainTablesWhy Tags.circuit967b8d06 Tags.MAX_TABLES == none
#guard ProvedScope.plainTablesWhy Tags.circuit967b8d06 1 == none

end Flock.Test.TableCap
