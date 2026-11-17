module

public import MemoryChallenger.Judge

/-! The properties of the memory challenger's verdict (`MemoryChallenger.verdict`), which take no assumption. -/

@[expose] public section

namespace MemoryChallenger.Properties

/-- What an accepted verdict establishes about its inputs. -/
def Accepted (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (s : SignedChallengePhase)
    (c : ChallengePhaseCommitment) (openings : Array CoinOpening) : Prop :=
  (∃ p, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧
      ∃ m, s.message = .ok m ∧ lmsVerify p s.publicKey s.commitment.session m s.signature = .ok true) ∧
    s.commitment = c ∧ s.ended = COMPLETE ∧ s.rounds.size = c.k ∧
    ∀ j (hj : j < s.rounds.size),
      s.rounds[j].j = j ∧ (∃ o ∈ openings, o.j = j ∧ o.index = s.rounds[j].index ∧ verifyCoin c o = true) ∧
        s.rounds[j].index < c.blocks ∧ expected s.rounds[j].index = some s.rounds[j].digest ∧
        0 ≤ s.rounds[j].elapsedNs ∧ s.rounds[j].elapsedNs ≤ c.limitNs

def VerdictAccepts : Prop :=
  ∀ (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (s : SignedChallengePhase)
    (c : ChallengePhaseCommitment) (openings : Array CoinOpening),
    (verdict schemes trusted expected s c openings).accepted = true → Accepted schemes trusted expected s c openings

def JudgeAccepts : Prop :=
  ∀ (r : Judge.JudgeInputs), Verdict.accepted <$> Judge.judge r = .ok true →
    ∃ schemes, Judge.schemesOf r.schemes = .ok schemes ∧
      Accepted schemes r.trusted.toList r.expected.find r.signed r.commitment r.openings

/-! The model reads a key as signing only its sessions' messages: a key used for another role too is outside it, and
the forgery disjunct is an EUF-CMA break only under that reading. -/

/-- What the device observes in a round it records: its clock's reading as it reveals the index, its reading as the
answer arrives, and the answer. -/
structure Observed where
  sendNs : Nat
  receiptNs : Nat
  block : ByteArray

/-- A session the device runs: its coins, its limit `Δ + RTT`, the answer's size, what the session audits, and what it
observes in each round, in order. -/
structure Run where
  coins : CoinTree
  limitNs : Nat
  blockBytes : Nat
  binding : ByteArray
  observed : List Observed

/-- The commitment the device builds before the session (`MemoryChallenger.__init__`) and releases once it is signed. -/
def Run.commitment (r : Run) : ChallengePhaseCommitment :=
  ⟨r.coins.session, r.coins.root, r.coins.key, r.coins.blocks, r.coins.k, r.limitNs, r.blockBytes, r.binding⟩

/-- The device's record of a session: its rounds, and how it ended once it has. -/
structure Recorded where
  rounds : Array Round
  ended : Option String

/-- One round, `challenge()` then `receive(block)`: none after the end (`challenge` refuses), nothing recorded for an
answer of the wrong size (`receive` refuses it), else round `j` with the coin's index, the two readings and
`blockDigest` of the answer, ending the session `late` if it is not within the limit and `complete` at the `k`-th. -/
def Recorded.round (coins : CoinTree) (limitNs blockBytes : Nat) (st : Recorded) (o : Observed) : Recorded :=
  if st.ended.isSome || o.block.size != blockBytes then st else
  let r : Round := ⟨st.rounds.size, coins.indices[st.rounds.size]!, o.sendNs, o.receiptNs, blockDigest o.block⟩
  let rounds := st.rounds.push r
  ⟨rounds, if !(0 ≤ r.elapsedNs && r.elapsedNs ≤ limitNs) then some LATE
    else if rounds.size == coins.k then some COMPLETE else none⟩

/-- What the device records of a session: its observations, round by round. -/
def Run.recorded (r : Run) : Recorded := r.observed.foldl (Recorded.round r.coins r.limitNs r.blockBytes) ⟨#[], none⟩

/-- The memory challengers: the sessions the holder of each key runs under it. -/
structure Challengers where
  sessions : ByteArray → List Run

/-- The messages the device signs under the key `pk`: for each session it runs under `pk` and ends,
`memoryChallengerMessage` of the key's identifier, the session id and the session's `message`. -/
def Signed (dev : Challengers) (pk m : ByteArray) : Prop :=
  ∃ r ∈ dev.sessions pk, ∃ ended rec, r.recorded.ended = some ended ∧
    message r.commitment ended r.recorded.rounds = .ok rec ∧
    memoryChallengerMessage (pk.extract 0 16) r.commitment.session rec = .ok m

/-- The LMS verifier accepts `sig` (once decoded) under the key `pk` over the message `m`. -/
def LmsVerifies (p : Lms.Params) (pk m sig : ByteArray) : Prop :=
  ∃ σ, Lms.Signature.decode p sig = .ok σ ∧ Lms.verify p pk m σ = .ok true

/-- The message a signed session's signature signs: `memoryChallengerMessage` of its key's identifier, its session id
and its `message`. -/
def challengePhaseMessage (s : SignedChallengePhase) : Option ByteArray :=
  (s.message >>= memoryChallengerMessage (s.publicKey.extract 0 16) s.commitment.session).toOption

/-- What SHA-512 hashes for a session `(c, ended, rounds)` signed under the key `pk`: the coin key (the header holds
its digest), the records stream under its `sha512/row/v2` prefix, the input of the session's `message`, and that of
`memoryChallengerMessage`. -/
def challengePhaseStrings (pk : ByteArray) (c : ChallengePhaseCommitment) (ended : String) (rounds : Array Round) : List ByteArray :=
  let head := (header c ended rounds.size).toUTF8
  [c.coinKey] ++
    (match recordsStream rounds with
      | .ok s => [Flock.HmRow.rowPrefixV2 (8 * s.size) ++ s]
      | .error _ => []) ++
    (match recordsDigest rounds with
      | .ok d => [SESSION_TAG ++ Flock.be64 head.size.toUInt64 ++ head ++ d]
      | .error _ => []) ++
    match message c ended rounds with
    | .ok rec =>
      if (pk.extract 0 16).size = 16 then
        [field MEMORY_CHALLENGER_TAG.toUTF8 ++ pk.extract 0 16 ++ field (field c.session) ++ field rec]
      else []
    | .error _ => []

/-- The strings the run hashes: the signed session's, and those of each session the device ends under its key. -/
def runStrings (s : SignedChallengePhase) (dev : Challengers) : List ByteArray :=
  challengePhaseStrings s.publicKey s.commitment s.ended s.rounds ++
    (dev.sessions s.publicKey).flatMap fun r => match r.recorded.ended with
      | some e => challengePhaseStrings s.publicKey r.commitment e r.recorded.rounds
      | none => []

/-- The signed session `s` is one the device ran under its key: its commitment is the one the device built, and
its end and every round (index, digest, send and receipt times) are the ones the device recorded; or two distinct
strings of the run have one SHA-512 digest; or `s`'s signature verifies under a trusted key and an accepted scheme over
its message, which the device never signed under that key. -/
def SignedByDevice (schemes : Schemes) (trusted : List ByteArray) (s : SignedChallengePhase) (dev : Challengers) : Prop :=
  (∃ r ∈ dev.sessions s.publicKey, r.commitment = s.commitment ∧ r.recorded.ended = some s.ended ∧
      r.recorded.rounds = s.rounds) ∨
    (∃ x ∈ runStrings s dev, ∃ y ∈ runStrings s dev, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) ∨
    ∃ p m, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧ challengePhaseMessage s = some m ∧
      LmsVerifies p s.publicKey m s.signature ∧ ¬ Signed dev s.publicKey m

/-- **An accepted session is the device's.** For every device, an accepted verdict's session is `SignedByDevice`. -/
def VerdictSigned : Prop :=
  ∀ (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (s : SignedChallengePhase)
    (c : ChallengePhaseCommitment) (openings : Array CoinOpening) (dev : Challengers),
    (verdict schemes trusted expected s c openings).accepted = true → SignedByDevice schemes trusted s dev

/-- `VerdictSigned` for `Judge.judge` on a record as its caller holds it, under the schemes the record names. The
forgery disjunct is at the record's own scheme parameters: a reader cites `euf-cma/lms-sha512-n64` only when the
record's scheme is `LMS` (`lms-sha512/n64-w1-h20/v0`) at its deployed parameters (`Lms.deployed`, Python
`lms.DEPLOYED`). -/
def JudgeSigned : Prop :=
  ∀ (r : Judge.JudgeInputs) (dev : Challengers), Verdict.accepted <$> Judge.judge r = .ok true →
    ∃ schemes, Judge.schemesOf r.schemes = .ok schemes ∧ SignedByDevice schemes r.trusted.toList r.signed dev

end MemoryChallenger.Properties
