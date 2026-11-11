module

public import Difftest
public import Flock.Firewall.Honest

/-! The honest worker's requests in an inner session, `Flock.Firewall.honestReqs`, held to `rec_live`'s
(`verity/ml/flock/tests/test_honest_reqs.py`): `honest-reqs` reads `{"schedule", "honest"}`, the pinned schedule
and what the worker brings beyond it (`honestOf`), and answers the requests in hex, in order. Flock's package has no
dependencies, so its port lives beside the harness. -/

@[expose] public section

namespace FirewallHonestDifftest

open Lean (Json)
open Flock Flock.Firewall

def hexAt (a : Array Json) (i : Nat) : V ByteArray :=
  match a[i]? with
  | some (.str x) => unhex x
  | _ => throw "an item's bytes: not hex"

def natAt (a : Array Json) (i : Nat) : V Nat :=
  match a[i]?.bind natOf? with
  | some n => pure n
  | none => throw "an item's number: not a natural number"

/-- An item as `["domain", hex]`, `["label", hex]`, `["bind", digest hex, cap hex]`, `["cap", hex]`, `["nonce", hex]`,
`["observe", kind, [element…]]` or `["squeeze", kind]`. -/
def absorbOf (j : Json) : V Absorb := do
  let a ← arr j
  let tag : String := match a[0]? with
    | some (Json.str t) => t
    | _ => ""
  match tag with
  | "domain" => return .domain (← hexAt a 1)
  | "label" => return .label (← hexAt a 1)
  | "bind" => return .bind (← hexAt a 1) (← hexAt a 2)
  | "cap" => return .cap (← hexAt a 1)
  | "nonce" => return .nonce (← hexAt a 1)
  | "observe" =>
    let es ← (← arr (a.getD 2 .null)).toList.mapM fun e => match natOf? e with
      | some v => pure v
      | none => throw "an element: not a natural number"
    return .observe (← natAt a 1) es
  | "squeeze" => return .squeeze (← natAt a 1)
  | _ => throw "not an item"

def strAt (a : Array Json) (i : Nat) : V String :=
  match a[i]? with
  | some (.str x) => pure x
  | _ => throw "a name: not a string"

/-- What an honest worker brings, as `{"pings": [hex…], "sigma", "root_f": hex, "roots": [[table, hex]…],
"labels": {stream: hex}, "opens": [[stream…]…], "rounds": [{"items": [item…], "kind"}…], "proofs": [[table, rep,
hex]…]}`, `opens[j]` the streams opened at the worker's `j`-th place (`Honest.opens`). -/
def honestOf (j : Json) : V Honest := do
  let pings ← (← arr (← field j "pings")).toList.mapM fun e => match e with
    | Json.str x => unhex x
    | _ => throw "pings: not hex"
  let roots ← (← arr (← field j "roots")).toList.mapM fun e => do
    let p ← arr e
    return (← strAt p 0, ← hexAt p 1)
  let labels ← (← obj (← field j "labels")).mapM fun (n, v) => match v with
    | Json.str x => do return (n, ← unhex x)
    | _ => throw "labels: not hex"
  let opens ← (← arr (← field j "opens")).mapM fun o => do
    (← arr o).toList.mapM fun e => match e with
      | Json.str x => pure x
      | _ => throw "opens: not a stream's name"
  let rounds ← (← arr (← field j "rounds")).mapM fun r => do
    return ({ items := ← (← arr (← field r "items")).toList.mapM absorbOf, kind := ← nat r "kind" } : HonestRound)
  let proofs ← (← arr (← field j "proofs")).toList.mapM fun e => do
    let p ← arr e
    return (← strAt p 0, ← natAt p 1, ← hexAt p 2)
  return { pings, sigma := ← str j "sigma", rootF := ← unhex (← str j "root_f"), roots,
           label := fun st => ((labels.find? (·.1 == st)).map (·.2)).getD .empty,
           opens := fun i => opens.getD i [],
           round := fun i => rounds.getD i {}, proofs }

def honestReqsFn : Difftest.Fn := fun j => do
  let reqs := honestReqs (← schedule (← field j "schedule")) (← honestOf (← field j "honest"))
  return .arr (reqs.map (Json.str ∘ hexOf)).toArray

def table : String → Option Difftest.Fn := Difftest.table [("honest-reqs", honestReqsFn)]

end FirewallHonestDifftest
