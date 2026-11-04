import Flock.Canon
import RemoteAudit.Flight

/-!
# The verifier's registrations: `audit.py`'s `Registry`

At most one live registration a node (the red team's KB-4). `register` draws a codec seed (`Registration.register`) only
for a node with none live: a second while one is live is refused and counted (`refused`), and draws nothing, so a retry
never redraws. A session runs over its node's live registration, drawn by this registry, as its k-th, k its sessions so
far: the epoch its coins are committed under (`coins.SessionCoins`, epoch k of the one challenger). A registration ends only
when the verifier closes it (`close`), which counts each of its sessions as it stands: "accepted", the failure that
rejected it, or `UNENDED`, counted as failed. `drawn` is R, the registrations a node has had, closed and live.

`toJson` and `ofJson` are `registry/v0`, which keeps the registry across the verifier's processes: a registration read
back is live and closable, and runs no session, its sessions' outcomes as they stood.

A call `Registry` refuses with `RegistryError` is `none`; one it raises `ValueError` on (a node not 1 to 64 bytes, a
registration `Registration.register` refuses) is an error.
-/

namespace RemoteAudit

open Lean (Json)
open Flock

def Registry.SCHEMA : String := "verity/remote-bandwidth/registry/v0"
def Registry.UNENDED : String := "unended"

/-- A live registration's session: one this registry started, as it stands, with its windows in flight (`Flight`)
and the windows a draw's openings coin covers (`Shared`), or the outcome it was read back with. -/
inductive Registry.Run
  | running (s : Session.State) (inFlight perCoin : Nat)
  | read (outcome : String)

def Registry.Run.outcome : Registry.Run → String
  | .running s _ _ => s.outcome.getD Registry.UNENDED
  | .read o => o

/-- A node's live registration: its digest, the registration if this registry drew it (`none` read back), and its
sessions in the order they started. -/
structure Registry.Live where
  node : ByteArray
  digest : ByteArray
  drawn : Option Registration
  sessions : Array Registry.Run := #[]

/-- A registration the verifier closed: its node, its digest, and each of its sessions' outcomes in the order they
started. -/
structure Registry.Closed where
  node : ByteArray
  digest : ByteArray
  sessions : Array String

structure Registry where
  live : Array Registry.Live := #[]
  closed : Array Registry.Closed := #[]
  refused : Array (ByteArray × Nat) := #[]

namespace Registry

def node? (node : ByteArray) : Except String ByteArray :=
  if 1 ≤ node.size && node.size ≤ 64 then pure node else throw "a node's id is 1 to 64 bytes"

private def withLive (r : Registry) (node : ByteArray) (f : Live → Live) : Registry :=
  { r with live := r.live.map fun x => if x.node == node then f x else x }

/-- The digest of the node's live registration, `none` without one. -/
def liveDigest (r : Registry) (node : ByteArray) : Except String (Option ByteArray) := do
  let node ← node? node
  pure ((r.live.find? (·.node == node)).map (·.digest))

/-- R: the registrations the node has had, closed and live, each under a codec seed of its own. -/
def drawn (r : Registry) (node : ByteArray) : Except String Nat := do
  let node ← node? node
  pure ((r.closed.filter (·.node == node)).size + if r.live.any (·.node == node) then 1 else 0)

/-- `Registration.register`'s draw for a node with no live registration, which it makes live; refused, and counted, for
one with. -/
def register (r : Registry) (source node : ByteArray) (blocks : Nat) (payloadRoot : ByteArray) (codec : Codec.Codec)
    (receipt : Kind) (bind : Bool) (layout : String) (rows : Nat) (nonce : ByteArray) :
    Except String (Option Registration × Registry) := do
  let node ← node? node
  if r.live.any (·.node == node) then
    let refused := match r.refused.findIdx? (·.1 == node) with
      | some i => r.refused.modify i fun (n, k) => (n, k + 1)
      | none => r.refused.push (node, 1)
    return (none, { r with refused })
  let reg ← Registration.register source node blocks payloadRoot codec receipt bind layout rows nonce
  pure (some reg, { r with live := r.live.push { node, digest := reg.digest, drawn := some reg } })

/-- The next session over the node's live registration, `Session.State.init` on its parameters and coins, as its k-th
(`index`, if given, must be k), at `inFlight` windows in flight (`Flight`) and `perCoin` windows a draw's openings coin
(`Shared`); refused for a registration that isn't its node's live one, drawn here. -/
def session (r : Registry) (reg : Registration) (params : Session.Params) (coins appendCoins : Array ByteArray)
    (notBeforeNs : Nat) (index : Option Int) (inFlight : Nat := 2) (perCoin : Nat := 1) : Option (Nat × Registry) := do
  let x ← r.live.find? (·.node == reg.node)
  let drawn ← x.drawn
  guard (drawn.digest == reg.digest)
  let k := x.sessions.size
  if let some i := index then guard (i = k)
  let s := Session.State.init params reg coins appendCoins notBeforeNs
  pure (k, r.withLive reg.node ({ · with sessions := x.sessions.push (.running s inFlight perCoin) }))

/-- One event through session `k` of the node's live registration (`Flight.step` at its windows in flight and its
windows a coin, which is `Session.step` at two and one); refused for a session this registry isn't running. -/
def step (r : Registry) (node : ByteArray) (k : Nat) (e : Session.Event) :
    Except String (Option Session.Result × Registry) := do
  let node ← node? node
  let some x := r.live.find? (·.node == node) | return (none, r)
  let some (.running s f g) := x.sessions[k]? | return (none, r)
  let (out, s) ← Flight.step f g s e
  pure (some out, r.withLive node ({ · with sessions := x.sessions.set! k (.running s f g) }))

/-- Session `k` of the node's live registration, if this registry runs it. -/
def running (r : Registry) (node : ByteArray) (k : Nat) : Option Session.State :=
  match (r.live.find? (·.node == node)).bind (·.sessions[k]?) with
  | some (.running s _ _) => some s
  | _ => none

/-- Session `k` of the node's live registration set to `s`, if this registry runs it. -/
def update (r : Registry) (node : ByteArray) (k : Nat) (s : Session.State) : Registry :=
  r.withLive node fun x => { x with sessions := x.sessions.modify k fun | .running _ f g => .running s f g | o => o }

/-- Session `k` of the node's live registration ended with `outcome` where `Session` gives it none: the driver's
(a message it can't read, `malformed`, or the verifiers' `disagreement`). It counts so at the close. -/
def abort (r : Registry) (node : ByteArray) (k : Nat) (outcome : String) : Registry :=
  r.withLive node fun x => { x with sessions := x.sessions.modify k fun _ => .read outcome }

/-- End the node's live registration, its sessions counted as they stand; refused for a node with none live. -/
def close (r : Registry) (node : ByteArray) : Except String (Option Closed × Registry) := do
  let node ← node? node
  let some x := r.live.find? (·.node == node) | return (none, r)
  let c : Closed := { node, digest := x.digest, sessions := x.sessions.map (·.outcome) }
  pure (some c, { r with live := r.live.filter (·.node != node), closed := r.closed.push c })

private def outcomesJ (os : Array String) : Json := .arr (os.map .str)

def Closed.toJson (c : Closed) : Json :=
  Json.mkObj [("node", .str (hexOf c.node)), ("registration", .str (hexOf c.digest)), ("sessions", outcomesJ c.sessions)]

def toJson (r : Registry) : Json :=
  Json.mkObj [("schema", .str SCHEMA), ("closed", .arr (r.closed.map Closed.toJson)),
    ("live", Json.mkObj (r.live.toList.map fun x => (hexOf x.node,
      Json.mkObj [("registration", .str (hexOf x.digest)), ("sessions", outcomesJ (x.sessions.map (·.outcome)))]))),
    ("refused", Json.mkObj (r.refused.toList.map fun (n, k) => (hexOf n, Lean.toJson k)))]

private def entries (j : Json) (ctx : String) : Except String (Array (String × Json)) :=
  match j with
  | .obj kvs => pure kvs.toArray
  | _ => throw s!"{ctx}: not an object"

private def nodeOf (s : String) : Except String ByteArray := do node? (← unhex s)

private def digestOf (j : Json) (ctx : String) : Except String ByteArray := do
  let d ← unhex (← J.str j "registration" ctx)
  if d.size ≠ 64 then throw "a registration's digest is 64 bytes"
  pure d

private def outcomesOf (j : Json) (ctx : String) : Except String (Array String) := do
  (← J.arr j "sessions" ctx).mapM fun o => do
    let s ← J.asStr o ctx
    if s.isEmpty then throw "a registration's sessions are their outcomes, each a nonempty string"
    pure s

/-- A registry read back from `registry/v0`: each live registration closable and running no session. -/
def ofJson (j : Json) : Except String Registry := do
  if (← J.str j "schema" "registry") ≠ SCHEMA then throw s!"a registry is {SCHEMA}"
  let live ← (← entries (← J.get j "live" "registry") "live").mapM fun (n, x) => do
    pure ({ node := ← nodeOf n, digest := ← digestOf x "live", drawn := none,
            sessions := (← outcomesOf x "live").map .read } : Live)
  let closed ← (← J.arr j "closed" "registry").mapM fun c => do
    pure ({ node := ← nodeOf (← J.str c "node" "closed"), digest := ← digestOf c "closed",
            sessions := ← outcomesOf c "closed" } : Closed)
  let refused ← (← entries (← J.get j "refused" "registry") "refused").mapM fun (n, k) => do
    let k ← J.asNat k "refused"
    if k = 0 then throw "a node's refused registrations are a positive count"
    pure (← nodeOf n, k)
  pure { live, closed, refused }

end Registry

end RemoteAudit
