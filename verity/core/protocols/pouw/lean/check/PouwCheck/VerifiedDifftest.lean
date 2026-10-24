import PouwCheck.CreditDifftest
import PouwCheck.Verified

/-!
# The difftest's function for a served window's statements (`PouwCheck.Verified`)

Beside `PouwCheck.CreditDifftest`'s, behind the same `pouw-check-difftest` (`PouwCheck.DifftestMain` reads `all`):

* `credit-verified` (`statements`, `files`, `descriptor` in hex, `entries`, `window`, `units`, `steps`, `salt` and
  `runs`, as `credit._lean_verified` sends them): `Verified.verified`'s `{"run": [...]}` or `{"outcomes": [...]}`, or
  `{"refused": [code, why]}`. The served path (`credit._lean_verified`, `verifier.lean`) reads the same answers.
-/

namespace PouwCheck.VerifiedDifftest

open Lean (Json)

def verifiedOf (j : Json) : Except String Json := PouwCheck.Difftest.refusal id (Verified.verified j)

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [("credit-verified", verifiedOf)]

/-- `pouw-check-difftest`'s functions: `PouwCheck.CreditDifftest.all`'s, then these. -/
def all (name : String) : Option _root_.Difftest.Fn := PouwCheck.CreditDifftest.all name <|> table name

end PouwCheck.VerifiedDifftest
