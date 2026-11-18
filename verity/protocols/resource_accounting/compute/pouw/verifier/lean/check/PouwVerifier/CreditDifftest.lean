module

public import PouwVerifier.Credit
public import PouwVerifier.CreditOnce
public import PouwVerifier.Difftest
public import PouwPearlPlus.Load

/-!
# The difftest's functions for a served execution phase's credit (`PouwCheck.Credit`)

Beside `PouwCheck.Difftest`'s, behind the same `pouw-check-difftest` (`PouwCheck.DifftestMain` reads both tables):

* `credit` (`key`, `ledger`, `pins`, `main` as `rounds` reads them, `salt` the execution phase's Program salt as hex, `calls`
  the manifest's rows, `served` the served trees' digest or null, `commit` the execution phase's Commit record in hex, `rounds`
  the round log, `draw` the execution phase's sample with its own session's files, each as `credit._credited` takes it,
  `Credit.Internal.credited`): `{"credited": [indices]}`, or `{"refused": [code, why]}`.
* `drawn` (`coin`, `stream`, `registration`, `receipt`, `layout`, as `credit.drawn` takes them): `{"drawn": [units]}`,
  or `{"refused": [code, why]}`.
* `window-drawn` (`window`, `draw`, `descriptor` in hex, `k`): `Credit.descriptorDrawn`, `credit._window_drawn` over
  the strata of the `layoutOf` of the descriptor's Program: `{"drawn": [units]}`, or `{"refused": [code, why]}`.
* `outcome` (`drawn`'s and `proofs`, as `credit._outcome` takes them) and `a-link` (`rounds`, `entries`, `calls`, the
  execution phase's call count, and `rows`, `{value: rows}` the layout gives or null for none, as `credit._a_link` takes them),
  `Credit.Internal`'s: null, or `{"refused": [code, why]}`.
* `leaf-words` (`k`): `{"leaf_words": w}`, `Credit.leafWords` (`weights.leaf_words`), or `{"leaf_words": null}` for a K
  `Credit.chunks?` refuses (`PouwCheck.Difftest`'s `chunks` holds the count itself).
* `window` (`credit`'s, `descriptor` in hex, `entries`, `proofs`, `live`, `k` and `rows`, as `credit._window_outcomes`
  takes them): `{"credited": [indices]}`,
  and `period` (`key`, `ledger`, `pins`, `main`, `name`, `windows`, as `credit._period_outcomes` takes them, each execution phase's evidence
  `window`'s): `{"credited": …}`; or
  `{"refused": [code, why]}`.
* `period-roots` (`key`, `ledger`, `name`, `roots`, as `credit.period` takes them): `Credit.period`, null, or
  `{"refused": [code, why]}`.
* `weights` (`window`, `layout`, `entries`, `requests`, `session`, `program`, `prefix`, as `credit._weights` takes them,
  `Credit.Internal.weights`): `{"weights": hex}`, or `{"refused": [code, why]}`.
* `registered-file` (`descriptor` in hex, `call`, `roots` `{value: [root, ...]}`, `window`, the execution phase's open root,
  `steps`, its rounds, and `salt`, its Program salt):
  `{"file": …, "partition": …}`, `Credit.registeredFile` and `Credit.partitionOf` over the `layoutOf` and `shapeOf` of
  the descriptor's Program (`credit.registered_file` and `credit.partition_of`), or `{"refused": [code, why]}`.
* `passes-distinct` (`passes`, the Pearl-C pass manifests credited together, as `credit.passes_distinct` takes them):
  Pearl+'s `PouwCheck.PearlPlus.passesDistinct`, null, or `{"refused": [code, why]}`.
* `commit-parsers` (`commit`, a Commit record as `credit` takes it): `{"core": [value], "total": [value]}`, the value
  Lean's own JSON parser and the total one (`Flock.JsonParse.parse`, which `Credit.commitOf?` reads the record with)
  each read the record's text as; `[]` for a parser that refuses the text, and for both when `commit` is not bytes in
  lowercase hex (`Credit.isHexBytes`) of UTF-8.
-/

@[expose] public section

namespace PouwVerifier.CreditDifftest

open Lean (Json ToJson toJson)

def creditOf (j : Json) : Except String Json := do
  let f := PouwVerifier.Difftest.get j
  PouwVerifier.Difftest.refusal (fun out => Json.mkObj [("credited", toJson out)])
    (Credit.Internal.credited (← f "key") (← f "ledger") (← f "pins") (← f "main") (← f "salt") (← f "calls") (← f "served") (← f "commit") (← f "rounds") (← f "draw"))

def drawnOf (j : Json) : Except String Json := do
  let f := PouwVerifier.Difftest.get j
  PouwVerifier.Difftest.refusal (fun out => Json.mkObj [("drawn", toJson out)])
    (Credit.drawn (← f "coin") (← f "stream") (← f "registration") (← f "receipt") (← f "layout") (← f "k"))

def executionPhaseDrawnOf (j : Json) : Except String Json := do
  let f := PouwVerifier.Difftest.get j
  PouwVerifier.Difftest.refusal (fun out => Json.mkObj [("drawn", toJson out)])
    (Credit.descriptorDrawn (← f "window") (← f "draw") (← f "descriptor") (← f "k"))

def outcomeOf (j : Json) : Except String Json := do
  let f := PouwVerifier.Difftest.get j
  PouwVerifier.Difftest.refusal (fun _ => Json.null)
    (Credit.Internal.outcome (← f "coin") (← f "stream") (← f "registration") (← f "receipt") (← f "layout") (← f "k")
      (← f "proofs"))

def executionPhaseOf (j : Json) : Except String Json := do
  let f := PouwVerifier.Difftest.get j
  PouwVerifier.Difftest.refusal (fun out => Json.mkObj [("credited", toJson out)])
    (Credit.executionPhase (← f "key") (← f "ledger") (← f "pins") (← f "main") (← f "window") (← f "session") (← f "salt")
      (← f "calls") (← f "served") (← f "commit") (← f "rounds") (← f "draw") (← f "descriptor") (← f "entries")
      (← f "proofs") (← f "live") (← f "k") (← f "rows"))

def periodOf (j : Json) : Except String Json := do
  let f := PouwVerifier.Difftest.get j
  PouwVerifier.Difftest.refusal (fun out => Json.mkObj [("credited", out.toJson)])
    (Credit.periodCredit (← f "key") (← f "ledger") (← f "pins") (← f "main") (← f "name") (← f "windows"))

def periodRootsOf (j : Json) : Except String Json := do
  let f := PouwVerifier.Difftest.get j
  PouwVerifier.Difftest.refusal (fun _ => Json.null) (Credit.period (← f "key") (← f "ledger") (← f "name") (← f "roots"))

def aLinkOf (j : Json) : Except String Json := do
  PouwVerifier.Difftest.refusal (fun _ => Json.null)
    (Credit.Internal.aLink (← PouwVerifier.Difftest.get j "rounds") (← PouwVerifier.Difftest.get j "entries")
      ((PouwVerifier.nat? (← PouwVerifier.Difftest.get j "calls")).getD 0)
      (match PouwVerifier.get j "rows" with
        | .obj kvs => kvs.toArray.toList.filterMap fun (v, n) => (v, ·) <$> PouwVerifier.nat? n
        | _ => []))

def leafWordsOf (j : Json) : Except String Json := do
  let k ← PouwVerifier.Difftest.get j "k"
  pure (match (PouwVerifier.nat? k).filter (Credit.chunks? · |>.isSome) with
    | some k => Json.mkObj [("leaf_words", toJson (Credit.leafWords k))]
    | none => Json.mkObj [("leaf_words", Json.null)])

def weightsOf (j : Json) : Except String Json := do
  let f := PouwVerifier.Difftest.get j
  PouwVerifier.Difftest.refusal (fun w => Json.mkObj [("weights", Json.str (Flock.hexOf w))])
    (Credit.Internal.weights (← f "window") (← f "layout") (← f "entries") (← f "requests") (← f "session")
      (← f "program") (← f "prefix"))

def registeredFileOf (j : Json) : Except String Json := do
  let f := PouwVerifier.Difftest.get j
  let descriptor ← f "descriptor"
  let call ← f "call"
  let roots ← f "roots"
  let window ← f "window"
  let steps ← f "steps"
  let salt ← f "salt"
  let bytes : Option ByteArray := match descriptor with
    | Json.str s => if Credit.isHexBytes s then PouwVerifier.hex? (Json.str s) else none
    | _ => none
  let got : PouwVerifier.R Json := do
    let some d := bytes | Credit.layoutRefuse Credit.DESCRIPTOR_HEX
    let P ← Credit.programOf d
    let layout ← Credit.layoutOf P
    let shape := Credit.shapeOf P
    let program := Flock.hexOf (Flock.Sha512.hash d)
    let file ← Credit.registeredFile program layout shape ((PouwVerifier.nat? call).getD 0) roots
      ((PouwVerifier.str? window).getD "") ((PouwVerifier.nat? steps).getD 1) ((PouwVerifier.str? salt).getD "")
    pure (Json.mkObj [("file", file), ("partition", Credit.partitionOf program shape)])
  PouwVerifier.Difftest.refusal id got

def passesDistinctOf (j : Json) : Except String Json := do
  let .arr passes ← PouwVerifier.Difftest.get j "passes" | throw "passes is not a list"
  pure <| match PouwVerifier.PearlPlus.passesDistinct passes.toList with
    | some (code, why) => Json.mkObj [("refused", toJson [code, why])]
    | none => Json.null

def commitParsersOf (j : Json) : Except String Json := do
  let commit ← PouwVerifier.Difftest.get j "commit"
  let text := (PouwVerifier.str? commit).bind fun s =>
    if Credit.isHexBytes s then (PouwVerifier.hex? commit).bind String.fromUTF8? else none
  let read := fun (parse : String → Except String Json) =>
    match text.bind fun t => (parse t).toOption with
    | some v => Json.arr #[v]
    | none => Json.arr #[]
  pure (Json.mkObj [("core", read Json.parse), ("total", read Flock.JsonParse.parse)])

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("commit-parsers", commitParsersOf), ("credit", creditOf), ("drawn", drawnOf), ("window-drawn", executionPhaseDrawnOf), ("outcome", outcomeOf),
  ("window", executionPhaseOf), ("period", periodOf), ("period-roots", periodRootsOf), ("a-link", aLinkOf), ("weights", weightsOf), ("registered-file", registeredFileOf), ("leaf-words", leafWordsOf),
  ("passes-distinct", passesDistinctOf)]

/-- `pouw-check-difftest`'s functions: `PouwCheck.Difftest.table`'s, then these. -/
def all (name : String) : Option _root_.Difftest.Fn := PouwVerifier.Difftest.table name <|> table name

end PouwVerifier.CreditDifftest
