import Randomness
import PouwCommit.Coins
import Identity.Domains

/-!
# The salts and draws PoUW derives (`verity.core.protocols.pouw.audit`; the clear audit's draws, `benchmarks/pouw/clear`)

Each is `verity.ml.randomness.derive` (`FlockSoundness.Randomness.derive`) under PoUW's domains:

    salt_of        derive(coin, "verity/pouw/epoch-salt/v1", {epoch, coins: root, weights: weights_root}).stream("salt", 32)
    unverified     derive(seed, "verity/pouw/salt/v1", {weights: weights_root}).stream("salt", 32)
    audit key      derive(source, "verity/pouw/audit/v1", {transcript, salt})                       (`Sampled.select`)
    draw_key       derive(coin, "verity/pouw/pearl-c-draw/v1", {transcript, records, salt})
    replay_key     derive(value, "verity/pouw/pearl-c-draw/v1", {round, transcript, records, salt})
    draw_after     tagged("verity/pouw/pearl-c/draw-after/v1", transcript, records)

`epochSalt` is `Epoch.start`'s salt: the opening must be the epoch's salt coin, opened after the weights' root, and is
refused in `coin_reason`'s words, "the salt's …", as `Epoch.start` refuses it. `drawCoin` is `draw_coin`'s check of the
epoch's draw coin. A key's source is nonempty bytes, which `derive` refuses otherwise.

`weightedDraw` is the work-weighted draw with replacement of `Sampled.select` and `clear_work.draw`: draw `j` is
`bisect_right(cumulative, key.uniform(total, j))` over the tickets' running sums. Python reads `uniform`'s stream for as long
as it needs; the port reads `DRAW_WORDS` words of it and throws past them, which an exact draw reaches with probability
below `2^(-64·DRAW_WORDS)`.
-/

namespace PouwCommit

open Flock
open FlockSoundness.Randomness (derive stream uniform)

def SALT_DOMAIN : String := "verity/pouw/salt/v1"
def EPOCH_SALT_DOMAIN : String := "verity/pouw/epoch-salt/v1"
def AUDIT_DOMAIN : String := Domains.POUW_AUDIT_KEY
def DRAW_DOMAIN : String := Domains.POUW_PEARL_C_SAMPLING
def DRAW_AFTER_TAG : String := Domains.POUW_PEARL_C_SAMPLING_AFTER
def SALT_BYTES : Nat := 32

/-- `derive(source, domain, context)`, refusing an empty source as `derive` does. -/
def key? (source : ByteArray) (domain : String) (context : List (String × Part)) : Except String ByteArray :=
  if source.size = 0 then throw "source must be nonempty bytes: a key holder's secret"
  else pure (derive source domain context)

/-- `salt_of`: the epoch's 32-byte salt from its opened salt coin. -/
def saltOf (c : CoinCommitment) (coin weightsRoot : ByteArray) : Except String ByteArray := do
  let k ← key? coin EPOCH_SALT_DOMAIN [("epoch", .int c.epoch), ("coins", .bytes c.root), ("weights", .bytes weightsRoot)]
  pure (stream k [.str "salt"] SALT_BYTES)

/-- `Epoch.unverified`'s salt: keyed by bare bytes, with no coin behind it. -/
def unverifiedSalt (seed weightsRoot : ByteArray) : Except String ByteArray := do
  let k ← key? seed SALT_DOMAIN [("weights", .bytes weightsRoot)]
  pure (stream k [.str "salt"] SALT_BYTES)

/-- `Epoch.start`'s salt: `.ok salt`, or `.error` with `Epoch.start`'s refusal when `o` is not epoch `c.epoch`'s salt coin
opened after `weightsRoot`. It throws where `coin_reason` raises. -/
def epochSalt (c : CoinCommitment) (o : CoinOpening) (weightsRoot : ByteArray) : Except String (Except String ByteArray) := do
  match ← coinReason c o "salt" weightsRoot with
  | some r => pure (.error s!"the salt's {r}")
  | none => pure (.ok (← saltOf c o.coin weightsRoot))

/-- `draw_coin`'s check: `.ok coin` when `o` is the epoch's draw coin opened after `after`, else `.error` with its words. -/
def drawCoin (c : Option CoinCommitment) (o : CoinOpening) (after : ByteArray) :
    Except String (Except String ByteArray) := do
  let some c := c | pure (.error "the epoch has no coin commitment for its draw coin to open against")
  match ← coinReason c o "draw" after with
  | some r => pure (.error s!"the draw's {r}")
  | none => pure (.ok o.coin)

/-- `Sampled.select`'s key. -/
def auditKey (source transcriptRoot salt : ByteArray) : Except String ByteArray :=
  key? source AUDIT_DOMAIN [("transcript", .bytes transcriptRoot), ("salt", .bytes salt)]

/-- `draw_key`: the Pearl-C draws' key from the epoch's draw coin. -/
def drawKey (coin transcriptRoot recordsDigest salt : ByteArray) : Except String ByteArray :=
  key? coin DRAW_DOMAIN [("transcript", .bytes transcriptRoot), ("records", .bytes recordsDigest), ("salt", .bytes salt)]

/-- `replay_key`: a recorded run's draws' key. -/
def replayKey (round : Nat) (value transcriptRoot recordsDigest salt : ByteArray) : Except String ByteArray :=
  key? value DRAW_DOMAIN
    [("round", .int round), ("transcript", .bytes transcriptRoot), ("records", .bytes recordsDigest), ("salt", .bytes salt)]

/-- `draw_after`: what the verifier records before the challenger opens the draw coin. -/
def drawAfter (transcriptRoot recordsDigest : ByteArray) : ByteArray :=
  tagged DRAW_AFTER_TAG [.bytes transcriptRoot, .bytes recordsDigest]

/-- `bisect_right(cumulative, x)` on a nondecreasing list. -/
def bisectRight (cumulative : List Nat) (x : Nat) : Nat := (cumulative.takeWhile (· ≤ x)).length

/-- The words of `uniform`'s stream the port reads per draw. -/
def DRAW_WORDS : Nat := 16

/-- `t` work-weighted draws with replacement over `tickets` under `key`: draw `j` is the ticket whose running sum first
passes `key.uniform(total, j)`. Like Python, no draw reads the tickets when `t = 0`, and one throws when they sum to 0. -/
def weightedDraw (key : ByteArray) (tickets : List Nat) (t : Nat) : Except String (List Nat) := do
  if t = 0 then return []
  let cumulative := (tickets.foldl (fun (acc : Array Nat × Nat) w => (acc.1.push (acc.2 + w), acc.2 + w)) (#[], 0)).1.toList
  let total ← match cumulative.getLast? with
    | none => throw "no tickets to draw from"
    | some 0 => throw "uniform(n) needs n > 0"
    | some n => pure n
  (List.range t).mapM fun j =>
    match uniform key total [.int j] (DRAW_WORDS * Draw.width total) with
    | some r => pure (bisectRight cumulative r)
    | none => throw s!"draw {j} read {DRAW_WORDS} words of its stream without one below the limit"

/-- `clear_work.draw` over tickets numbered by position: the draws among the tickets above 0, by their positions, and none
when no ticket is above 0. -/
def pearlDraw (key : ByteArray) (tickets : List Nat) (t : Nat) : Except String (List Nat) := do
  let tk := tickets.zipIdx.filter (·.1 > 0)
  if tk.isEmpty then return []
  let ids ← weightedDraw key (tk.map (·.1)) t
  pure (ids.map fun i => (tk.getD i (0, 0)).2)

/-! Each separator below keeps its literal while a lock reads its definition, whose record hashes the value as
written: tied here to its entry of the table (`Identity.Domains`), it reads the entry at its next record change. -/
example : SALT_DOMAIN = Domains.POUW_UNVERIFIED_SALT := rfl
example : EPOCH_SALT_DOMAIN = Domains.POUW_EPOCH_SALT := rfl

end PouwCommit
