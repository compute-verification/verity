module

public import CoreProtocol.Account
public import CoreProtocol.Cli
public import CoreProtocol.Ratio
public import Poub.Difftest
meta import CoreProtocol.Account
meta import CoreProtocol.Cli
meta import CoreProtocol.Ratio
meta import Poub.Difftest

/-!
# A PoUB session as a core protocol

The verifier's session (`Shared.run g`, which `audit.Session` is held to) behind the core interface (`Core.Protocol`).
No composite lists it: onsite's bandwidth component is `Poub.P2Protocol.protocol` (`Core.Onsite.components`,
`verity/protocols/compose/verifier/onsite`). Its bytes are canonical JSON, in the forms the difftest's `session` reads
(`Difftest.sessionFn`):

* params `{"not_before_ns": n, "params": {...}}`: the session's parameters (`Difftest.sessionOf`'s, its windows a
  sample's openings coin `groups_per_coin` g among them) and the verifier's clock floor. The check refuses what no
  property of the code covers (`sessionOf`): a session at one window in flight (the pace's `in_flight` 1, `Flight`), so
  what it replays is `Shared.run g` (`Session.run` at g = 1), which `Properties.Poub.Shared` and the memory line
  are stated on;
* the commitment: the registration (`Difftest.registrationOf`: node, blocks, payload root, codec seed, codec, receipt
  kind, `bind`, layout, rows), which the prover's payload root fixed before any coin; its digest `Registration.digest`;
* the evidence `{"append_coins": [hex, …], "coins": [hex, …], "events": [...]}`: the verifier's log of the session, its
  coins (W challenges and W openings coins, then the append coins in the order sampled) and every event it took with
  its clock's reading. The log is the verifier's own, as PoUS's `trusted` keys are: it never comes from the prover;
* the check: the session replayed through the verifier. It rejects with the failure that rejected the session, or
  when the parameters or the log don't decode, as the difftest's port refuses; it is undecided while the session runs
  (a window not yet verified); it accepts once every window is. `Properties.Poub.Subprotocol.AcceptIsVerified`
  states that it accepts only what `Shared.run g` accepts, and publishes `certify` of that run's record at the
  session's parameters and g.

On acceptance it publishes the receipts not shown wrong and the error they hold at, `{"eps", "q_bytes"}` (`certify`):
the session's receipts U less the wrong ones its openings leave, W rounded up (`Pooled.pooled` of each sample's receipts
and openings at `DELTA`), one `BLOCK_BYTES` block each, at the constant `EPS`, 2^-100. `EPS` bounds δ, the samples'
rejection, the sketch, the receipts' binding and AuditRO's slack only inside the caps `refusal` holds a session to
(`Properties.Poub.Subprotocol.PublishedEps`); it leaves out the openings' PRF advantages and the network attesters'
term δnet, which `LineHeldAtSubprotocol` carries beside it. A session outside the caps, or one the pooled bound
refuses, publishes `q_bytes` and `eps` 0 and its profile says why.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Poub.Protocol

open Lean (Json JsonNumber)
open Flock

def ID : String := "verity/remote-bandwidth/v0"

/-- `audit.DELTA`: PoUB's δ, the pooled bound's, `2^-101`. -/
def DELTA : Rat := 1 / (2 ^ 101 : Nat)

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

/-- The session the log replays: its parameters, its windows a sample's openings coin (`groups_per_coin` g,
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
g = 1), its parameters, its windows a sample and its append coins. -/
def replay (p : Params) (reg : Registration) (e : Evidence) :
    V (Session.State × Session.Params × Nat × Array ByteArray) := do
  let (params, g, coins, appendCoins, notBeforeNs, events) ← sessionOf p e
  let (_, s) ← Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) events
  pure (s, params, g, appendCoins)

/-- Each sample's receipts and samples, a sample being `g` consecutive windows (`audit.pooled_windows`). -/
def windowsOf (sm : Session.Summary) (g : Nat) : Array (Nat × Nat) :=
  let ws := sm.windows
  (Array.range ((ws.size + g - 1) / g)).map fun a =>
    let grp := ws.extract (a * g) (a * g + g)
    (grp.foldl (· + Line.receipts ·) 0, grp.foldl (· + ·.drawn) 0)

def ratio (q : Rat) : Core.Ratio := ⟨q.num.toNat, q.den⟩

/-- The public outputs: the error and the receipts not shown wrong, in bytes. -/
structure Public where
  eps : Core.Ratio
  qBytes : Nat

def Public.toJson (p : Public) : Json := Json.mkObj [("eps", .str p.eps.toString), ("q_bytes", natJ p.qBytes)]

/-- `⌈W/g⌉`: the openings coins of a session of `params.windows` windows sampled `g` a coin (`coins.openings_coins`), one
a window at g ≤ 1, as `Properties.Poub.Line.Node.samples`. -/
def openingsCoins (params : Session.Params) (g : Nat) : Nat :=
  if g ≤ 1 then params.windows else (params.windows + g - 1) / g

/-- The error every certifying session publishes, not counting the openings' PRF advantage. -/
def EPS : Rat := 1 / (2 ^ 100 : Nat)

def PASSES_MAX : Nat := 2 ^ 30
def BINDING_SQ_MAX : Nat := 2 ^ 92
def REJECTION_MAX : Nat := 2 ^ 40

/-- The most blocks the store holds: the registered ones, or the codec's `max_blocks` once append coins let it grow. -/
def mostBlocks (reg : Registration) (appendCoins : Nat) : Nat :=
  if appendCoins = 0 then reg.blocks else reg.codec.maxBlocks

/-- U, the most receipts a sample of `g` windows can open, each of its k passes a window receipting each of
`mostBlocks` at most once. `none` unpaced. -/
def sampleReceipts (params : Session.Params) (g : Nat) (reg : Registration) (appendCoins : Nat) : Option Nat :=
  params.pace.map fun pace => max g 1 * pace.k * mostBlocks reg appendCoins

/-- Why `EPS` doesn't bound a session's error, `none` when it does: only a paced 64-row sketch session with at most
`PASSES_MAX` passes, at most 2^64 blocks, `G²·U·s ≤ BINDING_SQ_MAX` and `G·s ≤ REJECTION_MAX` is covered. -/
def refusal (params : Session.Params) (g : Nat) (reg : Registration) (appendCoins : Nat) : Option String :=
  let G := openingsCoins params g
  let s := params.samples
  match params.pace, sampleReceipts params g reg appendCoins with
  | some pace, some u =>
    if !(reg.receipt == .sketch && reg.rows == 64) then
      some s!"{reg.receipt.name} at {reg.rows} rows: the constant error covers 64-row sketch-sha512 receipts only"
    else if mostBlocks reg appendCoins > 2 ^ 64 then
      some s!"a store of {mostBlocks reg appendCoins} blocks: over the 2^64 the constant error covers"
    else if params.windows * pace.k > PASSES_MAX then
      some s!"{params.windows * pace.k} passes: over the 2^30 the constant error covers"
    else if G * G * u * s > BINDING_SQ_MAX then
      some s!"G²·U·s = {G * G * u * s}: over the 2^92 the constant error covers"
    else if G * s > REJECTION_MAX then
      some s!"G·s = {G * s}: over the 2^40 the constant error covers"
    else none
  | _, _ => some "unpaced: a window's receipts have no bound the constant error covers"

/-- What an accepted session certifies (module docstring), with its profile. -/
def certify (params : Session.Params) (g : Nat) (reg : Registration) (appendCoins : Nat) (sm : Session.Summary) :
    Public × Json :=
  let ws := windowsOf sm g
  let receipts := (ws.map (·.1)).toList.sum
  let base := [("delta", Json.str (ratio DELTA).toString), ("receipts", natJ receipts),
    ("windows", .arr (ws.map fun (u, s) => .arr #[natJ u, natJ s]))]
  let none' (why : String) : Public × Json :=
    (⟨⟨0, 1⟩, 0⟩, Json.mkObj (base ++ [("gaps", .arr #[.str why]), ("wrong", .null)]))
  match refusal params g reg appendCoins with
  | some why => none' why
  | none =>
    match Pooled.pooled ws DELTA with
    | .ok b =>
      let wrong := (b.wrong.num.toNat + b.wrong.den - 1) / b.wrong.den
      (⟨ratio EPS, (receipts - wrong) * Codec.BLOCK_BYTES⟩,
       Json.mkObj (base ++ [("gaps", .arr #[]), ("wrong", .str (ratio b.wrong).toString)]))
    | .error why => none' why

def check (p : Params) (reg : Registration) (e : Evidence) : Core.Outcome Public Json :=
  match replay p reg e with
  | .error why => .reject why
  | .ok (s, params, g, appendCoins) =>
    match s.outcome, s.summary with
    | some "accepted", some sm => let (pub, prof) := certify params g reg appendCoins.size sm; .accept pub prof
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

/-- **What a session accounts for** (`Core.Account`): the reads of the node's memory its openings certify (`q_bytes`),
verified from below at `eps`, over the session. A session `refusal` or the pooled bound refuses publishes 0 reads at
`eps` 0, a true lower bound, so nothing is flagged.

Its answers name no auditor's device's key, so they are of the whole site (`witness`), and no span: the session
publishes none. Its part is the session ledger's two tiers, the node's GPUs' memory and its host's DRAM; the answer
names neither (`tiers`). The bytes the node held through the session are the ledger's line, not this answer. -/
def account : Core.Account :=
  { protocol := ID, witness := none
    amounts := [{ metric := "reads", part := .read [.hbm, .dram], key := .published "q_bytes",
                  unit := Core.Account.bytes, side := .verified, time := .unstated, error := .published "eps",
                  uncertified := .safe }] }

#guard (protocol.answer "{}".toUTF8 ByteArray.empty ByteArray.empty).reason ==
  some "params: params: the keys must be [not_before_ns, params], not []"
#guard account.Formed ∧ account.protocol = protocol.id
#guard (ratio DELTA).toString == "1/2535301200456458802993406410752"
#guard (ratio EPS).toString == "1/1267650600228229401496703205376"

end Poub.Protocol
