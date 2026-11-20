module

public import Flock.ProvedScope
meta import Flock.ProvedScope

/-!
The legacy statement forms (`ProvedScope.legacyStatements`) are refused: a legacy statement's name is no statement
`verify` sets up and meets `ProvedScope.legacyWhy`, and every statement `verify` sets up has hm96-sha512 rows.
-/

@[expose] public section

namespace Flock.Test.Legacy

def known : List String := (Tags.all ++ Tags.templated).map (·.name)

#guard ProvedScope.legacyStatements.all fun n => !known.contains n && (Tags.find? n).isNone
#guard ProvedScope.legacyStatements.all fun n =>
  ProvedScope.legacyWhy n == some s!"refused: {n}, a legacy statement (§16.9): a statement form the soundness proof does not cover"
#guard known.all fun n => (ProvedScope.legacyWhy n).isNone

#guard Tags.all.all (·.hm96Rows)

end Flock.Test.Legacy
