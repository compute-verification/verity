import CoreProtocol.Cli
import CoreProtocol.Ratio
import PoubCheck.Difftest

/-!
# A PoUB session as a core protocol

The verifier's session (`Shared.run g`, which `audit.Session` is held to) behind the core interface (`Core.Protocol`),
the bandwidth component of onsite Verity (`verity/core/protocols/onsite`). Its bytes are canonical JSON, in the forms
the difftest's `session` reads (`Difftest.sessionFn`):

* params `{"not_before_ns": n, "params": {...}}`: the session's parameters (`Difftest.sessionOf`'s, its windows a
  draw's openings coin `groups_per_coin` g among them) and the verifier's clock floor. The check refuses what no
  property of the code covers (`sessionOf`): a session at one window in flight (the pace's `in_flight` 1, `Flight`), so
  what it replays is `Shared.run g` (`Session.run` at g = 1), which `Properties.PoubCheck.Shared` and the memory line
  are stated on;
* the commitment: the registration (`Difftest.registrationOf`: node, blocks, payload root, codec seed, codec, receipt
  kind, `bind`, layout, rows), which the prover's payload root fixed before any coin; its digest `Registration.digest`;
* the evidence `{"append_coins": [hex, …], "coins": [hex, …], "events": [...]}`: the verifier's log of the session, its
  coins (W challenges and W openings coins, then the append coins in the order drawn) and every event it took with
  its clock's reading. The log is the verifier's own, as PoUS's `trusted` keys are: it never comes from the prover;
* the check: the session replayed through the verifier. It rejects with the failure that rejected the session, or
  when the parameters or the log don't decode, as the difftest's port refuses; it is undecided while the session runs
  (a window not yet verified); it accepts once every window is. `Properties.PoubCheck.Component.AcceptIsVerified`
  states that it accepts only what `Shared.run g` accepts, and publishes `certify` of that run's record at g.

On acceptance it publishes the reads it certifies and the error that holds at, `{"eps", "q_bytes"}` (`certify`): the
session's receipts U less the wrong ones its openings leave, W rounded up (`Pooled.pooled` of each draw's receipts and
draws at `DELTA`), each a read of one `BLOCK_BYTES` block, at ε = δ + S/2^64 for S the draws (the per-draw term of
`Accounting.Guarantees.Poub.PooledDraw`; its PRF and coin-commitment hiding terms are named assumptions, not
numbers). The ledger's γ terms, which need the node's measured rates, are not taken off: `q_bytes` is the reads the
session's openings certify. When the pooled bound refuses the windows, `q_bytes` and `eps` are 0 and the profile says
why.
-/

namespace PoubCheck.Protocol

open Lean (Json JsonNumber)
open Flock

def ID : String := "verity/poub/v0"

/-- `audit.DELTA`: the pooled bound's δ, `2^-40`. -/
def DELTA : Rat := 1 / (2 ^ 40 : Nat)

private def natJ (n : Nat) : Json := .num (JsonNumber.fromNat n)

/-- `j` is an object with exactly the keys `want`, in sorted order. -/
private def keys (j : Json) (want : List String) (ctx : String) : V Unit :=
  match j with
  | .obj kvs => if kvs.keys == want then pure () else throw s!"{ctx}: the keys must be {want}, not {kvs.keys}"
  | _ => throw s!"{ctx}: not an object"

structure Params where
  notBeforeNs : Json
  params : Json

def decodeParams (j : Json) : V Params := do
  keys j ["not_before_ns", "params"] "params"
  pure ⟨← J.get j "not_before_ns" "params", ← J.get j "params" "params"⟩

def decodeCommitment (b : ByteArray) : V Registration := do Difftest.registrationOf (← Core.canonicalJson b)

structure Evidence where
  appendCoins : Json
  coins : Json
  events : Array Json

def decodeEvidence (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  keys j ["append_coins", "coins", "events"] "evidence"
  pure ⟨← J.get j "append_coins" "evidence", ← J.get j "coins" "evidence", ← J.arr j "events" "evidence"⟩

/-- The session the log replays: its parameters, its windows a draw's openings coin (`groups_per_coin` g,
`Difftest.perCoinOf`), its coins, append coins and clock floor as `Difftest.sessionOf` reads them, and its events as
`Difftest.eventOf` reads them. It refuses the one form no property of the code covers, one window in flight (the pace's
`in_flight` 1, `Flight`), the one where `Flight.run f g` is not `Shared.run g` (`Flight.run_two`). -/
def sessionOf (p : Params) (e : Evidence) :
    V (Session.Params × Nat × Array ByteArray × Array ByteArray × Nat × Array Session.Event) := do
  let j := Json.mkObj [("append_coins", e.appendCoins), ("coins", e.coins), ("not_before_ns", p.notBeforeNs),
    ("params", p.params)]
  let (params, coins, appendCoins, notBeforeNs) ← Difftest.sessionOf j
  let g ← Difftest.perCoinOf j
  if (← Difftest.flightOf j) != 2 then throw "in_flight is 2: no property covers one window in flight"
  let events ← e.events.mapM Difftest.eventOf
  pure (params, g, coins, appendCoins, notBeforeNs, events)

/-- The session after the log's events through the Lean verifier (`Shared.run g` from `State.init`, `Session.run` at
g = 1), and its windows a draw. -/
def replay (p : Params) (reg : Registration) (e : Evidence) : V (Session.State × Nat) := do
  let (params, g, coins, appendCoins, notBeforeNs, events) ← sessionOf p e
  let (_, s) ← Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) events
  pure (s, g)

/-- Each draw's receipts and draws, a draw being `g` consecutive windows (`audit.pooled_windows`). -/
def windowsOf (sm : Session.Summary) (g : Nat) : Array (Nat × Nat) :=
  let ws := sm.windows
  (Array.range ((ws.size + g - 1) / g)).map fun a =>
    let grp := ws.extract (a * g) (a * g + g)
    (grp.foldl (· + Line.receipts ·) 0, grp.foldl (· + ·.drawn) 0)

def ratio (q : Rat) : Core.Ratio := ⟨q.num.toNat, q.den⟩

/-- The public outputs: the error and the reads certified, in bytes. -/
structure Public where
  eps : Core.Ratio
  qBytes : Nat

def Public.toJson (p : Public) : Json := Json.mkObj [("eps", .str p.eps.toString), ("q_bytes", natJ p.qBytes)]

/-- What an accepted session certifies (module docstring), with its profile. -/
def certify (sm : Session.Summary) (g : Nat) : Public × Json :=
  let ws := windowsOf sm g
  let receipts := (ws.map (·.1)).toList.sum
  let draws := (ws.map (·.2)).toList.sum
  let base := [("delta", Json.str (ratio DELTA).toString), ("receipts", natJ receipts),
    ("windows", .arr (ws.map fun (u, s) => .arr #[natJ u, natJ s]))]
  match Pooled.pooled ws DELTA with
  | .ok b =>
    let wrong := (b.wrong.num.toNat + b.wrong.den - 1) / b.wrong.den
    (⟨(Core.Ratio.mk' (2 ^ 24 + draws) (2 ^ 64)).getD ⟨1, 1⟩, (receipts - wrong) * Codec.BLOCK_BYTES⟩,
     Json.mkObj (base ++ [("gaps", .arr #[]), ("wrong", .str (ratio b.wrong).toString)]))
  | .error why => (⟨⟨0, 1⟩, 0⟩, Json.mkObj (base ++ [("gaps", .arr #[.str why]), ("wrong", .null)]))

def check (p : Params) (reg : Registration) (e : Evidence) : Core.Outcome Public Json :=
  match replay p reg e with
  | .error why => .reject why
  | .ok (s, g) =>
    match s.outcome, s.summary with
    | some "accepted", some sm => let (pub, prof) := certify sm g; .accept pub prof
    | some failure, _ => .reject s!"the session was rejected: {failure}"
    | none, _ => .undecided "the session has a window not yet verified"

def protocol : Core.Protocol where
  id := ID
  Params := Params
  Commitment := Registration
  Evidence := Evidence
  Public := Public
  Profile := Json
  params := decodeParams
  commitment := decodeCommitment
  evidence := decodeEvidence
  digest := Registration.digest
  check := check
  publicJson := Public.toJson
  profileJson := id

#guard (protocol.answer "{}".toUTF8 ByteArray.empty ByteArray.empty).reason ==
  some "params: params: the keys must be [not_before_ns, params], not []"
#guard (ratio DELTA).toString == "1/1099511627776"

end PoubCheck.Protocol
