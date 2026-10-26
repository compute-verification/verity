import PouwCheck.Credit
import PouwCheck.Difftest

/-!
# The difftest's functions for a served window's credit (`PouwCheck.Credit`)

Beside `PouwCheck.Difftest`'s, behind the same `pouw-check-difftest` (`PouwCheck.DifftestMain` reads both tables):

* `credit` (`key`, `ledger`, `pins`, `main`, `window`, `session` as `rounds` reads them, `salt` the window's Program salt
  as hex, `calls` the manifest's rows, `served` the served trees' digest or null, `commit` the window's Commit record in
  hex, `rounds` the round log, `draw` the window's draw, each as `credit._credited` takes it,
  `Credit.Internal.credited`): `{"credited": [indices]}`, or `{"refused": [code, why]}`.
* `drawn` (`coin`, `stream`, `registration`, `receipt`, `layout`, as `credit.drawn` takes them): `{"drawn": [units]}`,
  or `{"refused": [code, why]}`.
* `window-drawn` (`window`, `draw`, `descriptor` in hex, `k`): `Credit.descriptorDrawn`, `credit._window_drawn` over
  the strata of the descriptor's `layoutOf`: `{"drawn": [units]}`, or `{"refused": [code, why]}`.
* `outcome` (`drawn`'s and `proofs`, as `credit._outcome` takes them) and `a-link` (`rounds`, `entries`, `calls`, the
  window's call count, and `rows`, `{value: rows}` the layout gives or null for none, as `credit._a_link` takes them),
  `Credit.Internal`'s: null, or `{"refused": [code, why]}`.
* `leaf-words` (`k`): `{"leaf_words": w}`, `Credit.leafWords` (`weights.leaf_words`), or `{"leaf_words": null}` for a K
  `Credit.chunks?` refuses (`PouwCheck.Difftest`'s `chunks` holds the count itself).
* `window` (`credit`'s, `descriptor` in hex, `entries`, `proofs`, `live`, `k` and `rows`, as `credit._window_outcomes`
  takes them): `{"credited": [indices]}`,
  and `period` (`key`, `ledger`, `pins`, `main`, `name`, `windows`, as `credit._period_outcomes` takes them, each window's evidence
  `window`'s): `{"credited": …}`; or
  `{"refused": [code, why]}`.
* `weights` (`window`, `layout`, `entries`, `requests`, `session`, `program`, `prefix`, as `credit._weights` takes them,
  `Credit.Internal.weights`): `{"weights": hex}`, or `{"refused": [code, why]}`.
* `registered-file` (`descriptor` in hex, `call`, `roots` `{value: [root, ...]}`, `window`, the window's open root,
  `steps`, its rounds, and `salt`, its Program salt):
  `{"file": …, "partition": …}`, `Credit.registeredFile` and `Credit.partitionOf` over the descriptor's `layoutOf` and
  `shapeOf` (`credit.registered_file` and `credit.partition_of`), or `{"refused": [code, why]}`.
-/

namespace PouwCheck.CreditDifftest

open Lean (Json ToJson toJson)

def creditOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun out => Json.mkObj [("credited", toJson out)])
    (Credit.Internal.credited (← f "key") (← f "ledger") (← f "pins") (← f "main") (← f "window") (← f "session")
      (← f "salt") (← f "calls") (← f "served") (← f "commit") (← f "rounds") (← f "draw"))

def drawnOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun out => Json.mkObj [("drawn", toJson out)])
    (Credit.drawn (← f "coin") (← f "stream") (← f "registration") (← f "receipt") (← f "layout") (← f "k"))

def windowDrawnOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun out => Json.mkObj [("drawn", toJson out)])
    (Credit.descriptorDrawn (← f "window") (← f "draw") (← f "descriptor") (← f "k"))

def outcomeOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun _ => Json.null)
    (Credit.Internal.outcome (← f "coin") (← f "stream") (← f "registration") (← f "receipt") (← f "layout") (← f "k")
      (← f "proofs"))

def windowOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun out => Json.mkObj [("credited", toJson out)])
    (Credit.window (← f "key") (← f "ledger") (← f "pins") (← f "main") (← f "window") (← f "session") (← f "salt")
      (← f "calls") (← f "served") (← f "commit") (← f "rounds") (← f "draw") (← f "descriptor") (← f "entries")
      (← f "proofs") (← f "live") (← f "k") (← f "rows"))

def periodOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun out => Json.mkObj [("credited", out.toJson)])
    (Credit.periodCredit (← f "key") (← f "ledger") (← f "pins") (← f "main") (← f "name") (← f "windows"))

def aLinkOf (j : Json) : Except String Json := do
  PouwCheck.Difftest.refusal (fun _ => Json.null)
    (Credit.Internal.aLink (← PouwCheck.Difftest.get j "rounds") (← PouwCheck.Difftest.get j "entries")
      ((PouwCheck.nat? (← PouwCheck.Difftest.get j "calls")).getD 0)
      (match PouwCheck.get j "rows" with
        | .obj kvs => kvs.toArray.toList.filterMap fun (v, n) => (v, ·) <$> PouwCheck.nat? n
        | _ => []))

def leafWordsOf (j : Json) : Except String Json := do
  let k ← PouwCheck.Difftest.get j "k"
  pure (match (PouwCheck.nat? k).filter (Credit.chunks? · |>.isSome) with
    | some k => Json.mkObj [("leaf_words", toJson (Credit.leafWords k))]
    | none => Json.mkObj [("leaf_words", Json.null)])

def weightsOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  PouwCheck.Difftest.refusal (fun w => Json.mkObj [("weights", Json.str (Flock.hexOf w))])
    (Credit.Internal.weights (← f "window") (← f "layout") (← f "entries") (← f "requests") (← f "session")
      (← f "program") (← f "prefix"))

def registeredFileOf (j : Json) : Except String Json := do
  let f := PouwCheck.Difftest.get j
  let descriptor ← f "descriptor"
  let call ← f "call"
  let roots ← f "roots"
  let window ← f "window"
  let steps ← f "steps"
  let salt ← f "salt"
  let bytes : Option ByteArray := match descriptor with
    | Json.str s => if Credit.isHexBytes s then PouwCheck.hex? (Json.str s) else none
    | _ => none
  let got : PouwCheck.R Json := do
    let some d := bytes | Credit.layoutRefuse Credit.DESCRIPTOR_HEX
    let layout ← Credit.layoutOf d
    let shape := Credit.shapeOf d
    let program := Flock.hexOf (Flock.Sha512.hash d)
    let file ← Credit.registeredFile program layout shape ((PouwCheck.nat? call).getD 0) roots
      ((PouwCheck.str? window).getD "") ((PouwCheck.nat? steps).getD 1) ((PouwCheck.str? salt).getD "")
    pure (Json.mkObj [("file", file), ("partition", Credit.partitionOf program shape)])
  PouwCheck.Difftest.refusal id got

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("credit", creditOf), ("drawn", drawnOf), ("window-drawn", windowDrawnOf), ("outcome", outcomeOf),
  ("window", windowOf), ("period", periodOf), ("a-link", aLinkOf), ("weights", weightsOf), ("registered-file", registeredFileOf), ("leaf-words", leafWordsOf)]

/-- `pouw-check-difftest`'s functions: `PouwCheck.Difftest.table`'s, then these. -/
def all (name : String) : Option _root_.Difftest.Fn := PouwCheck.Difftest.table name <|> table name

end PouwCheck.CreditDifftest
