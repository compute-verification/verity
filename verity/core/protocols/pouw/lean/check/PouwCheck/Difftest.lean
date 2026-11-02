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

The lead's window (`PouwCheck.Lead`, `circuit/lead.py`'s), a refusal as `{"refused": why}` where Python raises
`ValueError`, and a malformed input a reject:

* `lead-record` (`weights_root`, `stream`, `prover`, `key_leaf` as hex, `calls`): `lead_record` as `{"record": hex}`.
* `lead-instance` (`salt` as hex, `position`, `schema`, `values` a list of hex): `instance_domain` over the values, as
  `{"domain", "leaves", "root"}`, leaf r the domain's leaf at rank r of value r and the root the tree's over them.
* `lead-unit` (`shapes` a list of `[rows, cols, K, bn]`, `call`, `R0`, `C0`, `BLK`): `unit_opening` as `{"words": [[t,
  leaf, word], …], "a": [[lo, hi], …], "segments"}`.
* `lead-egress` (`count`, `shapes` the forward's, admitted, `cap`, `fwd_macs`, `buckets` a list of `[r, [bits, …]]`,
  `release`): `egress_refused` and `forward_count` as `{"refused": bool, "count": n}`.
* `lead-window` (`record`, `weights_root`, `stream`, `prover`, `key_leaf` as hex, `calls`, `events` the certifier's
  record): `window_refusal` as `{"reason"}`, null when the window holds.
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

def intOf (j : Json) (k : String) : Except String Int := do need k (PouwWindow.strictInt? (← get j k))

def natOf (j : Json) (k : String) : Except String Nat := do
  let i ← intOf j k
  if i < 0 then throw s!"the input's {k} is negative" else pure i.toNat

def refused (why : String) : Json := Json.mkObj [("refused", .str why)]

def shapesOf (j : Json) (k : String) : Except String (List Lead.Shape) := do
  let .arr a ← get j k | throw s!"the input's {k} is not a list"
  a.toList.mapM fun x => match x with
    | .arr #[r, c, kk, b] => need k do
      pure ⟨← PouwWindow.strictInt? r, ← PouwWindow.strictInt? c, ← PouwWindow.strictInt? kk, ← PouwWindow.strictInt? b⟩
    | _ => throw s!"the input's {k} is malformed"

def leadRecordOf (j : Json) : Except String Json := do
  pure <| match Lead.leadRecord? (← hex j "weights_root") (← hex j "stream") (← hex j "prover") (← hex j "key_leaf")
      (← intOf j "calls") with
    | .ok r => Json.mkObj [("record", .str (Flock.hexOf r))]
    | .error why => refused why

def leadInstanceOf (j : Json) : Except String Json := do
  let .arr vs ← get j "values" | throw "the input's values is not a list"
  let values ← vs.toList.mapM fun v => match v with
    | .str s => PouwCommit.pyFromHex s
    | _ => throw "the input's values is malformed"
  let .str schema ← get j "schema" | throw "the input's schema is not a string"
  pure <| match Lead.instanceDomain? (← hex j "salt") (← intOf j "position") values.length with
    | .ok d =>
      let leaves := values.zipIdx.map fun (v, r) => d.leaf r schema v
      Json.mkObj [("domain", .str (Flock.hexOf d.id)), ("leaves", toJson (leaves.map Flock.hexOf)),
        ("root", match d.root leaves.toArray with
          | some r => .str (Flock.hexOf r)
          | none => .null)]
    | .error why => refused why

def leadUnitOf (j : Json) : Except String Json := do
  let o := Lead.unitOpening (← shapesOf j "shapes") (← intOf j "call") (← intOf j "R0") (← intOf j "C0") (← intOf j "BLK")
  pure <| match o with
    | .ok o => Json.mkObj [("words", toJson (o.words.map fun (t, r, w) => [t, r, w])),
        ("a", toJson (o.a.map fun (lo, hi) => [lo, hi])), ("segments", toJson o.segments)]
    | .error why => refused why

def leadEgressOf (j : Json) : Except String Json := do
  let shapes ← shapesOf j "shapes"
  if let some why := shapes.findSome? Lead.Shape.refuse then throw why
  let nat (x : Json) : Except String Nat := need "buckets" ((PouwWindow.strictInt? x).bind fun i =>
    if 0 ≤ i then some i.toNat else none)
  let .arr bs ← get j "buckets" | throw "the input's buckets is not a list"
  let buckets ← bs.toList.mapM fun b => match b with
    | .arr #[r, .arr fs] => do pure (← nat r, ← fs.toList.mapM nat)
    | _ => throw "the input's buckets is malformed"
  let over := Lead.egressRefused (← natOf j "count") shapes (← natOf j "cap") (← natOf j "fwd_macs")
  pure (Json.mkObj [("refused", .bool over), ("count", toJson (Lead.forwardCount buckets (← natOf j "release")))])

def leadWindowOf (j : Json) : Except String Json := do
  let .arr events ← get j "events" | throw "the input's events is not a list"
  let why := Lead.windowRefusal (← hex j "record") (← hex j "weights_root") (← hex j "stream") (← hex j "prover")
    (← hex j "key_leaf") (← intOf j "calls") events.toList
  pure (Json.mkObj [("reason", match why with
    | some r => .str r
    | none => .null)])

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("opening", openingOf), ("registered", registeredOf), ("pass", passOf), ("draw", drawOf), ("salt", saltOf),
  ("live-salt", liveSaltOf), ("rounds", roundsOf), ("chunks", chunksOf), ("lead-record", leadRecordOf),
  ("lead-instance", leadInstanceOf), ("lead-unit", leadUnitOf), ("lead-egress", leadEgressOf),
  ("lead-window", leadWindowOf)]

end PouwCheck.Difftest
