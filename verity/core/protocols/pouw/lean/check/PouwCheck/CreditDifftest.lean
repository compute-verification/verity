import PouwCheck.Credit
import PouwCheck.Difftest

/-!
# The difftest's functions for a served window's credit (`PouwCheck.Credit`)

Beside `PouwCheck.Difftest`'s, behind the same `pouw-check-difftest` (`PouwCheck.DifftestMain` reads both tables):

* `credit` (`key`, `ledger`, `window`, `session` as `rounds` reads them, `salt` the window's Program salt as hex, `calls`
  the manifest's rows, `served` the served trees' digest or null, `rounds` the round log, `draw` the window's draw, each
  as `credit.credited` takes it): `{"credited": [indices]}`, or `{"refused": [code, why]}`.
* `drawn` (`coin`, `stream`, `registration`, `receipt`, `layout`, as `credit.drawn` takes them): `{"drawn": [units]}`,
  or `{"refused": [code, why]}`.
* `outcome` (`drawn`'s and `proofs`, as `credit.outcome` takes them) and `a-link` (`rounds`, `entries`, as
  `credit.a_link` takes them): null, or `{"refused": [code, why]}`.
* `window` (`credit`'s, `layout` and `proofs`, as `credit.window` takes them): `{"credited": [indices]}`, and `period`
  (`key`, `ledger`, `name`, `windows`, as `credit.period_credit` takes them, each window's evidence `window`'s and its
  `descriptor` in hex): `{"credited": [[indices], …]}`; or
  `{"refused": [code, why]}`.
-/

namespace PouwCheck.CreditDifftest

open Lean (Json ToJson toJson)

def creditOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun out => Json.mkObj [("credited", toJson out)])
    (Credit.credited (← f "key") (← f "ledger") (← f "window") (← f "session") (← f "salt") (← f "calls")
      (← f "served") (← f "rounds") (← f "draw"))

def drawnOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun out => Json.mkObj [("drawn", toJson out)])
    (Credit.drawn (← f "coin") (← f "stream") (← f "registration") (← f "receipt") (← f "layout"))

def outcomeOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun _ => Json.null)
    (Credit.outcome (← f "coin") (← f "stream") (← f "registration") (← f "receipt") (← f "layout") (← f "proofs"))

def windowOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun out => Json.mkObj [("credited", toJson out)])
    (Credit.window (← f "key") (← f "ledger") (← f "window") (← f "session") (← f "salt") (← f "calls")
      (← f "served") (← f "rounds") (← f "draw") (← f "layout") (← f "proofs"))

def periodOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun out => Json.mkObj [("credited", toJson out)])
    (Credit.periodCredit (← f "key") (← f "ledger") (← f "name") (← f "windows"))

def aLinkOf (j : Json) : Except String Json := do
  PouwCheck.Difftest.refusal (fun _ => Json.null)
    (Credit.aLink (← PouwCheck.Difftest.get j "rounds") (← PouwCheck.Difftest.get j "entries"))

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("credit", creditOf), ("drawn", drawnOf), ("outcome", outcomeOf), ("window", windowOf), ("period", periodOf),
  ("a-link", aLinkOf)]

/-- `pouw-check-difftest`'s functions: `PouwCheck.Difftest.table`'s, then these. -/
def all (name : String) : Option _root_.Difftest.Fn := PouwCheck.Difftest.table name <|> table name

end PouwCheck.CreditDifftest
