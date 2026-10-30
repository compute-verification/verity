/-
`ProvedScope.check`'s table cap: a `--zk` session of more than `zkTableCap` tables is refused, with the text the live
build's mirror (`proved_scope.rs`, `statement_why`) gives; one of `zkTableCap` tables passes the statement's rules, and
without `--zk` the rule is still at most one table. Under `--plain-tables` (`ProvedScope.checkP`) a plain session of
public or hidden outputs passes the cap at up to `Tags.MAX_TABLES` tables and is refused at 65, a statement with hidden
outputs and plain SHA-512 leaves is refused at any count, and the verifier's own program must be the one META binds
(`ProvedScope.programWhyP`). Its draw is by the
subset law alone (`ProvedScope.lawWhy` at `plainTables`), where without the flag a plain session at public outputs draws
by the stratified law alone.
-/
import Flock.ProvedScope

namespace Flock.Test.TableCap

def why (zk : Bool) (count : Nat) : String :=
  match ProvedScope.check Tags.circuit zk ByteArray.empty ByteArray.empty [] count false false with
  | .error e => e
  | .ok () => ""

def whyP (tags : Tags) (count : Nat) : String :=
  match ProvedScope.checkP tags ByteArray.empty ByteArray.empty [] count none false false with
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
#guard whyP Tags.circuitPlainLeaves 2 == "refused: verity/flock-circuit+plain-leaves, a statement with hidden " ++
  "outputs and flock-leaf/sha512-unsalted leaves under --plain-tables: a statement form the soundness proof does not cover"
#guard whyP Tags.circuit 65 ==
  "refused: 65 tables in a session under --plain-tables, more than 64: a statement form the soundness proof does not cover"
#guard Tags.circuit.hiddenOutputs && Tags.circuit.merkleLeaf == "hm96-sha512/v1"
#guard ProvedScope.plainTablesWhy Tags.circuit Tags.MAX_TABLES == none
#guard ProvedScope.plainTablesWhy Tags.circuit967b8d06 Tags.MAX_TABLES == none
#guard ProvedScope.plainTablesWhy Tags.circuit967b8d06 1 == none

/-- A circuit whose META binds `m` as its program, the one field `programWhyP` reads. -/
def bound (m : Option String) : Circuit :=
  { (default : Circuit) with metaJ := Lean.Json.mkObj (m.toList.map fun h => ("program_sha512", Lean.Json.str h)) }

def prog : ByteArray := "{\"root\":\"r\",\"definitions\":[1,2],\"annotations\":{\"x\":1}}".toUTF8

/-- `program_digest` of `{"root": "r", "definitions": [1, 2]}` (`verity.ml.circuits.codec`, which drops `annotations`). -/
def progSha : String :=
  "2ddb1d2afb53742ae98210afa683ceb6842b10625d88ff0ad16eca0de13512b40226451238d1cf430a87db6db4e6d34629a43fd5ee5609bd8e23bd191e6b96a4"

#guard ProvedScope.programDigest prog == some progSha
#guard ProvedScope.programWhyP (bound (some progSha)) (some prog) == none
#guard ProvedScope.programWhyP (bound (some progSha)) (some "{\"root\":\"s\"}".toUTF8) ==
  some s!"refused: the verifier's program (--program, program_sha512 {(ProvedScope.programDigest "{\"root\":\"s\"}".toUTF8).getD ""}) is not the circuit's (META program_sha512 {progSha})"
#guard ProvedScope.programWhyP (bound (some progSha)) none == some ProvedScope.NO_PROGRAM
#guard ProvedScope.programWhyP (bound (some progSha)) (some "not json".toUTF8) ==
  some "refused: the verifier's program (--program) is not a JSON descriptor"
#guard ProvedScope.programWhyP (bound none) (some prog) ==
  some "refused: a circuit whose META binds no program_sha512 under --plain-tables: the verifier cannot hold its program to the circuit"

def subsetD : Draw.UnitDraw := { law := .subset 2, population := 8, units := #[1, 6] }
def stratD : Draw.UnitDraw := { law := .stratified 2 #[], population := 8, units := #[1, 6] }

#guard ProvedScope.lawWhy false Tags.circuit967b8d06 true subsetD == none
#guard ProvedScope.lawWhy false Tags.circuit967b8d06 true stratD == some "a stratified draw under --plain-tables"
#guard ProvedScope.lawWhy false Tags.circuit967b8d06 false stratD == none
#guard ProvedScope.lawWhy false Tags.circuit967b8d06 false subsetD == some "a subset draw without --zk"

end Flock.Test.TableCap
