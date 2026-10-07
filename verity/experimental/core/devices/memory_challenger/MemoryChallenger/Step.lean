import MemoryChallenger.Verdict

/-!
# One session as a state machine

`memory_challenger.MemoryChallenger` as `step : State → Event → State × List Action` (`Core.World`'s shape), with the
link, the clock and the leaf state outside it: an event carries the clock's reading, and an action is what the driver
does next. The Python's `run(respond)` is a call per round; here the answer arrives on a byte stream, so `step` frames it.

* The first `tick` publishes the commitment, before any index is revealed. Each later `tick` with no round open reveals
  the next index (`challenge`): its send time is the tick's reading, which the driver takes before it sends.
* `received now bytes` appends to the open round's answer. The round's receipt time is the reading `now` of the event
  whose bytes complete the `block_bytes`-byte block, which the driver takes after the bytes arrive and before `step`
  hashes them (`receive`). Bytes with no round open, and any past the block, are no answer and are dropped.
* A round is late when its elapsed time passes `limit_ns`; the first late round ends the session (`LATE`), and `k`
  on-time rounds end it `COMPLETE`. No round is retried and no index redrawn.
* Two endings the Python has no counterpart for, since its `respond` always returns: a `tick` past the open round's
  deadline (`send_ns + limit_ns`), and the end of the link, end the session `LATE` with the open round unrecorded.
* An ended session asks for a leaf (`reserve`) once; `reserved q` signs the session at `q` (`close`, its signature
  pure: `Lms.signAt`) and publishes it with the coins of every recorded round (`openings`). A refused reservation ends
  the session unsigned, the reason in `State.refusal`.

On the link a challenge is `u32 j ‖ u64 index`.
-/

namespace MemoryChallenger

open Flock

/-- One session's fixed parts: the coins committed before it (`CoinTree`), the limit `Δ + RTT`, the answer's size, what
the session audits (`binding`), and the signer's key and scheme. -/
structure Config where
  coins : CoinTree
  limitNs : Nat
  blockBytes : Nat
  binding : ByteArray
  key : Lms.PrivateKey
  scheme : String

/-- `MemoryChallenger(coins, signer, limit_ns=, block_bytes=, binding=)`'s refusals, `LmsSigner`'s check of the scheme
against the key, and the widths a challenge is written in: `k` below 2^32 and every index below 2^64. -/
def Config.make (coins : CoinTree) (limitNs blockBytes : Nat) (binding : ByteArray) (key : Lms.PrivateKey) (scheme : String) :
    E Config := do
  if limitNs = 0 || blockBytes = 0 then throw "a session's limit and answer size are positive"
  if (scheme == LMS) != (key.params == Lms.deployed) then throw s!"{LMS} is LMS at the deployed parameters, and only it"
  if coins.k ≥ 2 ^ 32 || coins.blocks > 2 ^ 64 then throw "a challenge is a u32 round and a u64 index"
  pure ⟨coins, limitNs, blockBytes, binding, key, scheme⟩

def Config.commitment (cfg : Config) : SessionCommitment :=
  ⟨cfg.coins.session, cfg.coins.root, cfg.coins.key, cfg.coins.blocks, cfg.coins.k, cfg.limitNs, cfg.blockBytes, cfg.binding⟩

inductive Event where
  /-- The clock read `now`. -/
  | tick (now : Nat)
  /-- `bytes` arrived on the link, and then the clock read `now`. -/
  | received (now : Nat) (bytes : ByteArray)
  /-- The peer closed the link. -/
  | closed
  /-- The leaf state's answer to `reserve`. -/
  | reserved (leaf : Except String Nat)

/-- What the challenger publishes (the World's `Out`): its commitment, then the signed session and its coins. -/
inductive Out where
  | commitment (c : SessionCommitment)
  | session (s : SignedSession) (openings : Array CoinOpening)

inductive Action where
  | send (bytes : ByteArray)
  | publish (o : Out)
  | reserve

inductive Phase where
  | fresh | running | signing | done
  deriving BEq, Repr, Inhabited

/-- The open round: its index, revealed at `sendNs`. -/
structure Open where
  j : Nat
  index : Nat
  sendNs : Nat

structure State where
  phase : Phase := .fresh
  rounds : Array Round := #[]
  current : Option Open := none
  buffer : ByteArray := .empty
  ended : String := ""
  refusal : Option String := none

/-- `x` big-endian in `n` bytes, modulo 2^(8n) (`Config.make` keeps a challenge's fields inside their widths). -/
def be (n x : Nat) : ByteArray := ⟨(Array.range n).reverse.map fun i => (x / 2 ^ (8 * i) % 256).toUInt8⟩

/-- A challenge on the link: `u32 j ‖ u64 index`. -/
def reveal (j index : Nat) : ByteArray := be 4 j ++ be 8 index

/-- `close`'s signed session at leaf `q`, and `openings`. -/
def sign (cfg : Config) (s : State) (q : Nat) : E Out := do
  let unsigned : SignedSession := ⟨cfg.commitment, s.ended, s.rounds, cfg.scheme, cfg.key.publicKey, .empty⟩
  let record ← unsigned.message
  let sig ← Lms.signAt cfg.key q (← Lms.memoryChallengerMessage cfg.key.I cfg.coins.session record)
  pure (.session { unsigned with signature := sig.encode } (s.rounds.map fun r => cfg.coins.opening r.j))

private def finish (s : State) (ended : String) : State × List Action :=
  ({ s with phase := .signing, ended, current := none, buffer := .empty }, [.reserve])

def step (cfg : Config) (s : State) : Event → State × List Action
  | .tick now =>
    match s.phase, s.current with
    | .fresh, _ => ({ s with phase := .running }, [.publish (.commitment cfg.commitment)])
    | .running, none =>
      let j := s.rounds.size
      match cfg.coins.indices[j]? with
      | some i => ({ s with current := some ⟨j, i, now⟩, buffer := .empty }, [.send (reveal j i)])
      | none => (s, [])
    | .running, some o => if now > o.sendNs + cfg.limitNs then finish s LATE else (s, [])
    | _, _ => (s, [])
  | .received now bytes =>
    match s.phase, s.current with
    | .running, some o =>
      let answer := s.buffer ++ bytes
      if answer.size < cfg.blockBytes then ({ s with buffer := answer }, []) else
      let r : Round := ⟨o.j, o.index, o.sendNs, now, blockDigest (answer.extract 0 cfg.blockBytes)⟩
      let s := { s with rounds := s.rounds.push r, current := none, buffer := .empty }
      if r.elapsedNs > cfg.limitNs then finish s LATE
      else if s.rounds.size == cfg.coins.k then finish s COMPLETE
      else (s, [])
    | _, _ => (s, [])
  | .closed =>
    match s.phase with
    | .fresh => ({ s with phase := .done }, [])
    | .running => finish s LATE
    | _ => (s, [])
  | .reserved leaf =>
    match s.phase with
    | .signing =>
      match leaf >>= sign cfg s with
      | .ok out => ({ s with phase := .done }, [.publish out])
      | .error e => ({ s with phase := .done, refusal := some e }, [])
    | _ => (s, [])

/-- `step` over `events` from the initial state: the last state, and each event's actions. -/
def run (cfg : Config) (events : List Event) : State × Array (List Action) :=
  events.foldl (fun (s, out) e => let (s', acts) := step cfg s e; (s', out.push acts)) ({}, #[])

end MemoryChallenger
