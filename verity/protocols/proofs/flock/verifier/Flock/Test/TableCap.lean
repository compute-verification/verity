module

public import Flock.ProvedScope
meta import Flock.ProvedScope

/-!
`ProvedScope.check`'s table cap: a `--zk` session of more than `zkTableCap` tables is refused, with the text the live
build's mirror (`proved_scope.rs`, `statement_why`) gives; one of `zkTableCap` tables passes the statement's rules, and
without `--zk` the rule is still at most one table. Under `--plain-tables` (`ProvedScope.checkP`) a plain session of
public or hidden outputs passes the cap at up to `Tags.MAX_TABLES` tables and is refused at 65, a statement with hidden
outputs and plain SHA-512 leaves is refused at any count, and the verifier's own program must be the one META binds
(`ProvedScope.programWhyP`). Its sample is by the
subset sampling design alone (`ProvedScope.samplingDesignWhy` at `plainTables`), where without the flag a plain session at public outputs samples
by the stratified sampling design alone.
-/

@[expose] public section

namespace Flock.Test.TableCap

/-- The served statement with public outputs, the form only a superseded version has. -/
def publicOutputs : Tags := { Tags.circuit with hiddenOutputs := false }

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
#guard whyP publicOutputs 65 ==
  "refused: 65 tables in a session under --plain-tables, more than 64: a statement form the soundness proof does not cover"
#guard whyP Tags.circuitPlainLeaves 2 == "refused: verity/flock-circuit+plain-leaves, a statement with hidden " ++
  "outputs and flock-leaf/sha512-unsalted leaves under --plain-tables: a statement form the soundness proof does not cover"
#guard whyP Tags.circuit 65 ==
  "refused: 65 tables in a session under --plain-tables, more than 64: a statement form the soundness proof does not cover"
#guard Tags.circuit.hiddenOutputs && Tags.circuit.merkleLeaf == "hm96-sha512/v1"
#guard ProvedScope.plainTablesWhy Tags.circuit Tags.MAX_TABLES == none
#guard ProvedScope.plainTablesWhy publicOutputs Tags.MAX_TABLES == none
#guard ProvedScope.plainTablesWhy publicOutputs 1 == none

/-- A circuit whose META binds `m` as its program, the one field `programWhyP` reads. -/
def bound (m : Option String) : Circuit :=
  { (default : Circuit) with metaJ := Lean.Json.mkObj (m.toList.map fun h => ("program_sha512", Lean.Json.str h)) }

/-- A descriptor of one primitive Definition, its root `root`. -/
def desc (root : String) (rest : String := "") : ByteArray :=
  ("{" ++ rest ++ "\"schema\":\"verity-ir/descriptor/v2\",\"callees\":[],\"definitions\":{\"" ++ root ++
    "\":{\"form\":\"primitive\",\"id\":\"" ++ root ++ "\",\"params\":[],\"ret\":[\"Value\",1]}}," ++
    "\"inputs\":[],\"program_fn\":\"" ++ root ++ "\",\"root\":\"" ++ root ++ "\",\"types\":[]}").toUTF8

def prog : ByteArray := desc "R_v1" "\"annotations\":{\"x\":1},"

/-- `program_digest` of `desc "R_v1"` (`verity.circuits.codec`, which drops `annotations`). -/
def progSha : String :=
  "6c1fe3b1e3f2bd59d18e62aae46b1666a7586234ae0490cf13601d1f1fbd84b0cbccbdb0bf38377e2815572b6fb4552416016d70535280b3ab4f7f0821400572"

#guard ProvedScope.programDigest prog == some progSha
#guard ProvedScope.programWhyP (bound (some progSha)) (some prog) == none
#guard ProvedScope.programWhyP (bound (some progSha)) (some (desc "S_v1")) ==
  some s!"refused: the verifier's program (--program, program_sha512 {(ProvedScope.programDigest (desc "S_v1")).getD ""}) is not the circuit's (META program_sha512 {progSha})"
#guard ProvedScope.programWhyP (bound (some progSha)) none == some ProvedScope.NO_PROGRAM
#guard ProvedScope.programWhyP (bound (some progSha)) (some "not json".toUTF8) ==
  some "refused: the verifier's program (--program) is not a JSON descriptor"
#guard ProvedScope.programWhyP (bound none) (some prog) ==
  some "refused: a circuit whose META binds no program_sha512 under --plain-tables: the verifier cannot hold its program to the circuit"

def subsetD : Draw.UnitSample := { samplingDesign := .subset 2, population := 8, units := #[1, 6] }
def stratD : Draw.UnitSample := { samplingDesign := .stratified 2 #[], population := 8, units := #[1, 6] }

#guard ProvedScope.samplingDesignWhy false publicOutputs true subsetD == none
#guard ProvedScope.samplingDesignWhy false publicOutputs true stratD == some "a stratified draw under --plain-tables"
#guard ProvedScope.samplingDesignWhy false publicOutputs false stratD == none
#guard ProvedScope.samplingDesignWhy false publicOutputs false subsetD == some "a subset draw without --zk"

end Flock.Test.TableCap
