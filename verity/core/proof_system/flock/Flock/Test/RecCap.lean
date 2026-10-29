/-
RecCap at the deployed inner shape (m 25, k_log 22, claims 8): `RecVStar.statements` builds V*'s whole list there (its
parts cut and as many as the verifier's groups) and ends it with the later levels' caps, then level 0's. A whole V* list in
the interpreter is about 20 minutes, so this is not one of Flock's audit roots: verity-flock's
`test_lean_rec_cap_guards.py` builds it after landing (lean-suites). `Flock.Test.RecCapSmall`, a root, holds RecCap's
refusals, and the checks `statements` runs on its list at m 22.
-/
import Flock.Test.RecCapSmall

namespace Flock.Test.RecCap

open Flock Flock.RecVStar Flock.Test.RecCapSmall

#guard match statements inner with
  | .ok ss => (ss.toList.drop (ss.size - 4)).map (·.name) ==
      ["rec-cap/n128-l1", "rec-cap/n128-l2", "rec-cap/n64-l3", "rec-cap/n128"]
  | .error _ => false

end Flock.Test.RecCap
