/-
The legacy statement forms (PROTOCOL.md §16.3, §16.4, §16.9) are refused: a legacy statement's name is no statement
`verify` sets up and meets `ProvedScope.legacyWhy`, and every statement before hm96-sha512 rows (the netlist statement and
the BLAKE3-row circuit statements up to `631567f7`) fails `ProvedScope.statementRules` first, with or without `--zk`.
-/
import Flock.ProvedScope

namespace Flock.Test.Legacy

def known : List String := (Tags.all ++ Tags.templated).map (·.name)

#guard ProvedScope.legacyStatements.all fun n => !known.contains n && (Tags.find? n).isNone
#guard ProvedScope.legacyStatements.all fun n =>
  ProvedScope.legacyWhy n == some s!"refused: {n}, a legacy statement (§16.9): a statement form the soundness proof does not cover"
#guard known.all fun n => (ProvedScope.legacyWhy n).isNone

def preHm96 : List Tags := Tags.all.filter (!·.hm96Rows)

#guard preHm96.map (·.name) == ["verity/flock-netlist/v1", "verity/flock-circuit@19c7269a", "verity/flock-circuit@fd02e847",
  "verity/flock-circuit@631567f7"]
#guard preHm96.all fun t => [false, true].all fun zk =>
  ProvedScope.firstFailure (ProvedScope.statementRules t zk 1 false) == some s!"{t.name}, a statement before hm96-sha512 rows"

end Flock.Test.Legacy
