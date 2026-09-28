/-
The release side's functions over JSON, as the difftest (`FirewallDifftest`) serves them and Rust's `release::run_op`
writes them (`live/src/firewall_record_vectors.json`):

* `op`, `{"op": [NAME, ARGS…], "record": TEXT}`: one call on the record, `{"record": TEXT', "result": R}`. `open` gives
  the session's id, `{"refused": why}` or `{"error": why}`; `admit` gives `{"release_ms": t or null}` or
  `{"refused": why}`; `stop` whether it recorded the stop; `failures` the count; `finish`, `release` and `slot` null.
* `outer`, `{"now_ms", "part", "prove", "ready", "record", "send", "session"}`: one outer session (`outerSession`),
  `{"outcome", "record", "sent_at", "why"}`. `prove` and `ready` are `"ok"`, `{"err": why}` or `{"panic": msg}`; `send` is
  the challenger's verdict (a boolean) or `{"panic": msg}`.

A record the firewall can't read, and an op that isn't one of these with arguments of their types, are rejected. An open's
statements, budget and schedule are checked before the record is read, as Rust and Python check them.
-/
import Firewall.Release

namespace Core.Firewall

open Lean (Json)
open Flock (V)

private def strAt (a : Array Json) (i : Nat) (what : String) : V String :=
  match a[i]? with
  | some (Json.str s) => pure s
  | _ => throw s!"op: {what} is not a string"

private def obj1 (k : String) (v : Json) : Json := Json.mkObj [(k, v)]

private def optNat : Option Nat → Json
  | some n => Json.num n
  | none => Json.null

private def answer (r : Record) (result : Json) (evs : Events) : Json :=
  Json.mkObj [("record", Json.str (r.extend evs).text), ("result", result)]

def opFn (input : Json) : V Json := do
  let text ← Flock.J.str input "record" "input"
  let op ← Flock.J.arr input "op" "input"
  let name ← strAt op 0 "its name"
  let arg := fun (i : Nat) (what : String) => strAt op i what
  let read : V Record := Record.read text
  match name, op.size with
  | "open", 5 =>
    let stream ← arg 1 "stream"
    let parts ← match op[2]! with
      | Json.arr ps => ps.mapM (Flock.J.asStr · "op: a statement")
      | _ => throw "op: parts is not a list"
    let budget ← match op[3]! with
      | Json.null => pure none
      | j => some <$> Flock.J.asNat j "op: budget"
    match openArgs parts budget (match op[4]! with | Json.null => none | j => some j) with
    | .error e => pure (Json.mkObj [("record", Json.str text), ("result", obj1 "error" (Json.str e))])
    | .ok sched =>
      let r ← read
      match openSession r.events stream parts budget sched with
      | (.ok s, evs) => pure (answer r (Json.str s) evs)
      | (.refused why, evs) => pure (answer r (obj1 "refused" (Json.str why)) evs)
      | (.error e, evs) => pure (answer r (obj1 "error" (Json.str e)) evs)
  | "finish", 2 =>
    let s ← arg 1 "session"
    let r ← read
    pure (answer r Json.null (finish r.events s))
  | "stop", 4 =>
    let (s, p, why) := (← arg 1 "session", ← arg 2 "part", ← arg 3 "why")
    let r ← read
    let (b, evs) := stop r.events s p why
    pure (answer r (Json.bool b) evs)
  | "release", 3 =>
    let (s, p) := (← arg 1 "session", ← arg 2 "part")
    let r ← read
    pure (answer r Json.null (release r.events s p))
  | "admit", 3 =>
    let (s, p) := (← arg 1 "session", ← arg 2 "part")
    let r ← read
    match admit r.events s p with
    | (.ok a, evs) => pure (answer r (obj1 "release_ms" (optNat a.releaseMs)) evs)
    | (.error why, evs) => pure (answer r (obj1 "refused" (Json.str why)) evs)
  | "failures", 2 =>
    let stream ← arg 1 "stream"
    let r ← read
    pure (answer r (Json.num (failures r.events stream)) r.events)
  | "slot", 3 =>
    let (s, p) := (← arg 1 "session", ← arg 2 "part")
    let r ← read
    pure (answer r Json.null (slot r.events s p))
  | name, n => throw s!"op: {name} with {n - 1} arguments is not an op of the record"

private def runOf (j : Json) (what : String) : V Run :=
  match j with
  | Json.str "ok" => pure .ok
  | Json.obj kvs =>
    match kvs.toArray with
    | #[("err", Json.str m)] => pure (.err m)
    | #[("panic", Json.str m)] => pure (.fault m)
    | _ => throw s!"{what}: not \"ok\", \{\"err\": why} or \{\"panic\": msg}"
  | _ => throw s!"{what}: not \"ok\", \{\"err\": why} or \{\"panic\": msg}"

def outerFn (input : Json) : V Json := do
  let ctx := "input"
  let r ← Record.read (← Flock.J.str input "record" ctx)
  let send ← match ← Flock.J.get input "send" ctx with
    | Json.bool b => pure (Except.ok b)
    | j => match runOf j "send" with
      | .ok (.fault m) => pure (Except.error m)
      | _ => throw "send: not a boolean or {\"panic\": msg}"
  let (o, r) ← outerSession r (← Flock.J.str input "session" ctx) (← Flock.J.str input "part" ctx) (← Flock.J.nat input "now_ms" ctx)
    (← runOf (← Flock.J.get input "prove" ctx) "prove") (← runOf (← Flock.J.get input "ready" ctx) "ready") send
  pure (Json.mkObj [("outcome", Json.str o.kind.name), ("record", Json.str r.text),
    ("sent_at", optNat o.sentAt), ("why", match o.why with | some w => Json.str w | none => Json.null)])

end Core.Firewall
