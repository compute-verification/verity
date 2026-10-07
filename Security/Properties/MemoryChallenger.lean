import MemoryChallenger.Step

/-!
# PoUS's memory challenger: what its Lean code (`MemoryChallenger.step`) does

The properties of the executable challenger (`verity/experimental/core/devices/memory_challenger/`, the port of
`verity/experimental/protocols/pous/memory_challenger.py`), stated against `step` as it runs, over every list of events
from the initial state (`reach`): the driver's World is adversarial, so an event list is any list. Each is a `Prop`;
none is proved yet (no `Properties.Proofs.MemoryChallenger`).

What a property assumes is in its statement: that the leaf state never hands out a leaf twice (`NoLeafSignedTwice`),
and a key's tree at most 2^31 leaves high, so a leaf fits the signature's `u32 q`. None assumes the clock: a round is
late when its elapsed time is outside `[0, limit_ns]` (`Late`, `verdict`'s condition), so a clock that runs backwards
ends the session as a slow answer does. `MessageBinds` is stated in the form
`Security/Proofs/Flock/Verifier/Hash.lean` uses, with its caveat: `Sha512Collision` is inhabited classically (SHA-512
compresses), so a proof of the disjunction must exhibit a colliding pair taken from the two inputs, as
`Security/Proofs/Flock/Soundness/StrictCR.lean` does, to say anything.
-/

open MemoryChallenger Flock

namespace Properties.MemoryChallenger

/-- The state `step` reaches on `events` from the initial state. -/
def reach (cfg : Config) (events : List Event) : State := (run cfg events).1

/-- Every action `step` takes on `events`, in order. -/
def actionsOf (cfg : Config) (events : List Event) : List Action := (run cfg events).2.toList.flatten

def sends : List Action → List ByteArray := List.filterMap fun | .send b => some b | _ => none

/-- A challenge is sent only with no round open and every recorded round answered on time (receipt neither before its
send nor past the limit), and it reveals the next round's committed index: index `j + 1` is never revealed before round
`j`'s answer is in, and never after its deadline (which ends the session). -/
def RevealsAfterAnswer : Prop :=
  ∀ cfg events e, let s := reach cfg events
    ∀ b ∈ sends (step cfg s e).2,
      s.phase = .running ∧ s.current = none ∧ s.rounds.size < cfg.coins.k ∧
        b = reveal s.rounds.size cfg.coins.indices[s.rounds.size]! ∧
        ∀ r ∈ s.rounds, 0 ≤ r.elapsedNs ∧ r.elapsedNs ≤ (cfg.limitNs : Int)

/-- On any clock, every round's receipt time is at least its send time, but for the last round of a session that ended
`late`: a receipt before its send ends the session with that round. A session signed `complete` has none. -/
def ReceiptAfterSend : Prop :=
  ∀ cfg events, let s := reach cfg events
    ∀ i (h : i < s.rounds.size), (i + 1 < s.rounds.size ∨ s.ended ≠ LATE) →
      s.rounds[i].sendNs ≤ s.rounds[i].receiptNs

/-- A round is late when its elapsed time is outside `[0, limit_ns]`: past the limit, or a receipt before its send. -/
def Late (cfg : Config) (r : MemoryChallenger.Round) : Prop :=
  ¬(0 ≤ r.elapsedNs ∧ r.elapsedNs ≤ (cfg.limitNs : Int))

/-- A late round is the session's last, and the session has ended `late` with it. -/
def EndsAtFirstLate : Prop :=
  ∀ cfg events, let s := reach cfg events
    ∀ i (h : i < s.rounds.size), Late cfg s.rounds[i] →
      i + 1 = s.rounds.size ∧ s.ended = LATE ∧ (s.phase = .signing ∨ s.phase = .done)

/-- Two different inputs with one SHA-512 digest. -/
def Sha512Collision : Prop := ∃ u v : ByteArray, u ≠ v ∧ Sha512.hash u = Sha512.hash v

/-- The signed message binds the header and every round record: two sessions with one message have one header (the
commitment's fields, how the session ended and how many rounds it had) and the same records, round by round. -/
def MessageBinds : Prop :=
  ∀ c ended rounds c' ended' rounds' m, message c ended rounds = .ok m → message c' ended' rounds' = .ok m →
    (header c ended rounds.size = header c' ended' rounds'.size ∧
        rounds.mapM MemoryChallenger.Round.value = rounds'.mapM MemoryChallenger.Round.value) ∨
      Sha512Collision

/-- The leaf a signature names: its first four bytes, big-endian (`u32 q`). -/
def leafOf (sig : ByteArray) : Nat := (sig.extract 0 4).foldl (fun n b => 256 * n + b.toNat) 0

/-- A session asks for a leaf at most once. -/
def ReservesOnce : Prop :=
  ∀ cfg events, ((actionsOf cfg events).filter fun | .reserve => true | _ => false).length ≤ 1

/-- A signed session is published only in answer to a reserved leaf, and signed at that leaf. -/
def SignsAtTheReservedLeaf : Prop :=
  ∀ cfg s e ss os, cfg.key.params.H < 32 → Action.publish (.session ss os) ∈ (step cfg s e).2 →
    ∃ q, e = .reserved (.ok q) ∧ leafOf ss.signature = q

def reservedLeaves (events : List Event) : List Nat := events.filterMap fun | .reserved (.ok q) => some q | _ => none

def signedLeaves (cfg : Config) (events : List Event) : List Nat :=
  (actionsOf cfg events).filterMap fun | .publish (.session s _) => some (leafOf s.signature) | _ => none

/-- No leaf is signed twice: across sessions whose leaf state never hands out a leaf twice, every published signature
names a leaf of its own. -/
def NoLeafSignedTwice : Prop :=
  ∀ sessions : List (Config × List Event), (∀ x ∈ sessions, x.1.key.params.H < 32) →
    (sessions.flatMap fun x => reservedLeaves x.2).Nodup → (sessions.flatMap fun x => signedLeaves x.1 x.2).Nodup

end Properties.MemoryChallenger
