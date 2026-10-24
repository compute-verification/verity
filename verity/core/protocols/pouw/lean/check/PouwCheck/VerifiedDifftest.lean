import PouwCheck.CreditDifftest
import PouwCheck.Verified
import PouwCheck.LiveVerified

/-!
# The difftest's functions for a served window's statements (`PouwCheck.Verified`, `PouwCheck.LiveVerified`)

Beside `PouwCheck.CreditDifftest`'s, behind the same `pouw-check-difftest` (`PouwCheck.DifftestMain` reads `all`):

* `credit-verified` (`statements`, `files`, `descriptor` in hex, `entries`, `window`, `units`, `steps`, `salt` and
  `runs`, as `credit._lean_verified` sends them): `Verified.verified`'s `{"run": [...]}` or `{"outcomes": [...]}`, or
  `{"refused": [code, why]}`. The served path (`credit._lean_verified`, `verifier.lean`) reads the same answers.
* `credit-live-verified` (`files`, `statements`, `descriptor` in hex, `entries`, `ms`, `program`, `window`, `session`
  and `runs`, as `credit._lean_live_verified` sends them): `LiveVerified.liveVerified`'s `{"run": [...]}` or
  `{"outcomes": [...]}`, or `{"refused": [code, why]}`, the window's live@<s> statements' verdict.
-/

namespace PouwCheck.VerifiedDifftest

open Lean (Json)

def verifiedOf (j : Json) : Except String Json := PouwCheck.Difftest.refusal id (Verified.verified j)

def liveVerifiedOf (j : Json) : Except String Json := PouwCheck.Difftest.refusal id (LiveVerified.liveVerified j)

def table : String → Option _root_.Difftest.Fn :=
  _root_.Difftest.table [("credit-verified", verifiedOf), ("credit-live-verified", liveVerifiedOf)]

/-- `pouw-check-difftest`'s functions: `PouwCheck.CreditDifftest.all`'s, then these. -/
def all (name : String) : Option _root_.Difftest.Fn := PouwCheck.CreditDifftest.all name <|> table name

end PouwCheck.VerifiedDifftest
