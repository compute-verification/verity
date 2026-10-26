import Difftest
import PouwCheck.Check
import PouwCheck.Chunks
import PouwCheck.Opening
import PouwCheck.Pass
import PouwCheck.Rounds

/-!
# The difftest's functions: the check's Lean behind `Difftest.serve`

`serve` cases go through `pouw-check serve` (`Core.Protocol.main`); what is not a `serve` case is here, for the shared
harness (`verity/core/difftest`):

* `opening` (`params` as JSON, `commitment` the registration's bytes as hex, `units` a list of integers or null, `unit`,
  `pub` the proof's public file as hex, `opening` as JSON, and `salt`, a certified window's salt as hex, null or absent
  otherwise): `core_check.opening`'s reason, or null. A refusal of the
  parameters or the registration is a reject, as `core_check.params` and `commitment` raise `Refused` (a ValueError).
* `registered` (`params`, `commitment`, as `opening` reads them): PoUW's rule over the registration
  (`core_check.registered`), as `{"reason"}`, null when it holds.
* `pass` (`coins` the challenger's commitment as JSON or null, `raw` the manifest's bytes as hex or null, `context`,
  `recorded`, `reveal`, `committed`, `stream` and `prover`, a certified window's, as hex, null or absent otherwise, and
  `live`, a live Pearl-C pass's verifier's `{key, ledger, main, pins, session}`, null or absent otherwise):
  `core_check.pass_epoch`, as `{"how"}` or `{"refused": [code, why]}`.
* `draw` (`coins`, `recorded`, `opening`, `stream`, `registration`, `receipt`, `k`, `population`, `units`, each as
  `core_check.draw` takes it): null, or `{"refused": [code, why]}`.
* `salt` (`key`, `ledger`, `pins`, `main`, `window` and `session` as `rounds` reads them, `weights` and `requests` the
  window's weights' root and round-0 request table's root, `opening` the salt coin, each as `core_check.salt` takes it):
  null, or `{"refused": [code, why]}`.
* `live-salt` (`key`, `ledger`, `pins`, `main` and `session` as `salt` reads them, `manifest` a live Pearl-C pass's
  manifest as JSON, each as `core_check.live_salt` takes it): `{"salt": hex}`, or `{"refused": [code, why]}`.
* `rounds` (`key` the window's pinned challenger key, `ledger` the auditor's signed ledger's lines, `pins` the challenger
  host's published pins log, `main` the auditor's own first-parent list of main, `window` the window's identity,
  `session` the challenger's session's files, `roots` the round roots, `openings` the round coins, each as
  `core_check.rounds` takes it): null, or `{"refused": [code, why]}`.
* `chunks` (a natural number K, not an object): `PouwCheck.Chunks.chunks` (`weights.chunks`) as `{"chunks": n}`, or
  `{"refused": why}`; any other input is a reject.
-/

namespace PouwCheck.Difftest

open Lean (Json ToJson toJson)

def need {α : Type} (what : String) : Option α → Except String α
  | some v => .ok v
  | none => .error s!"the input's {what} is malformed"

def get (j : Json) (k : String) : Except String Json := need k (PouwWindow.pyGet? j k)

def hex (j : Json) (k : String) : Except String ByteArray := do
  let .str s ← get j k | throw s!"the input's {k} is not a string"
  PouwCommit.pyFromHex s

/-- An optional hex field: absent or null is none. -/
def hexOrNull (j : Json) (k : String) : Except String (Option ByteArray) :=
  match (PouwWindow.pyGet? j k).getD .null with
  | .null => pure Option.none
  | .str s => some <$> PouwCommit.pyFromHex s
  | _ => throw s!"the input's {k} is malformed"

def openingOf (j : Json) : Except String Json := do
  let p ← paramsOf (← get j "params")
  let c ← registrationOf (← hex j "commitment")
  let units ← match ← get j "units" with
    | .null => pure Option.none
    | .arr a => some <$> a.toList.mapM fun u => need "units" (PouwWindow.strictInt? u)
    | _ => throw "the input's units is malformed"
  let unit ← need "unit" (PouwWindow.strictInt? (← get j "unit"))
  pure (toJson (← opening p c units unit (← hex j "pub") (← get j "opening") (← hexOrNull j "salt")))

def refusal {α : Type} (ok : α → Json) : R α → Except String Json
  | .ok a => pure (ok a)
  | .error ("error", e) => throw e
  | .error (code, why) => pure (Json.mkObj [("refused", toJson [code, why])])

def flag (j : Json) (k : String) : Except String Bool := do
  let .bool b ← get j k | throw s!"the input's {k} is not a boolean"
  pure b

def passOf (j : Json) : Except String Json := do
  let raw ← match ← get j "raw" with
    | .null => pure Option.none
    | .str s => some <$> PouwCommit.pyFromHex s
    | _ => throw "the input's raw is malformed"
  let .str context ← get j "context" | throw "the input's context is not a string"
  refusal (fun how => Json.mkObj [("how", how)])
    (passEpoch (← get j "coins") raw context (← flag j "recorded") (← hexOrNull j "reveal") (← hexOrNull j "committed")
      (← hexOrNull j "stream") (← hexOrNull j "prover") (SampledProofs.getD j "live"))

def drawOf (j : Json) : Except String Json := do
  let f := get j
  refusal (fun _ => Json.null) (drawCheck (← f "coins") (← flag j "recorded") (← f "opening") (← f "stream")
    (← f "registration") (← f "receipt") (← f "k") (← f "population") (← f "units") (SampledProofs.getD j "key"))

def saltOf (j : Json) : Except String Json := do
  let f := get j
  refusal (fun _ => Json.null) (saltCheck (← f "key") (← f "ledger") (← f "pins") (← f "main") (← f "window")
    (← f "session") (← f "weights") (← f "requests") (← f "opening"))

def liveSaltOf (j : Json) : Except String Json := do
  let f := get j
  refusal (fun salt => Json.mkObj [("salt", .str (Flock.hexOf salt))]) (liveSaltCheck (← f "key") (← f "ledger")
    (← f "pins") (← f "main") (← f "session") (← f "manifest"))

def roundsOf (j : Json) : Except String Json := do
  let f := get j
  refusal (fun _ => Json.null) (roundsCheck (← f "key") (← f "ledger") (← f "pins") (← f "main") (← f "window")
    (← f "session") (← f "roots") (← f "openings"))

def registeredOf (j : Json) : Except String Json := do
  let p ← paramsOf (← get j "params")
  let c ← registrationOf (← hex j "commitment")
  pure (Json.mkObj [("reason", match registered p c with
    | .ok _ => Json.null
    | .error r => .str r)])

def chunksOf (j : Json) : Except String Json := do
  let some k := (PouwWindow.strictInt? j).bind fun i => if 0 ≤ i then some i.toNat else none
    | throw "the input is not a natural number K"
  pure <| match Chunks.chunks k with
    | .ok n => Json.mkObj [("chunks", toJson n)]
    | .error why => Json.mkObj [("refused", Json.str why)]

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("opening", openingOf), ("registered", registeredOf), ("pass", passOf), ("draw", drawOf), ("salt", saltOf),
  ("live-salt", liveSaltOf), ("rounds", roundsOf), ("chunks", chunksOf)]

end PouwCheck.Difftest
