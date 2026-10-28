import MemoryChallenger.Judge

/-!
# The memory challenger's verdict's property, stated against the Lean code

What `MemoryChallenger.verdict` (package `core_memory_challenger`, `memory_challenger.verdict` in the Python) accepts,
stated over its inputs: the accepted schemes, the trusted keys, the expected digests, the signed session, the commitment
released with it and the coin openings. No statement takes an assumption: the verdict is a function of those inputs.

* `VerdictAccepts`: an accepted verdict's scheme is accepted, its key trusted and `lmsVerify` passes on the session's
  message; the signed header is the commitment; the session ended complete after the commitment's `k` rounds; and each
  round `j`'s index is its committed coin (an opening of round `j` to that index verifies under the coin tree), its
  digest is the expected digest of that block of `C`, and it is on time, within `Δ + RTT`.
* `JudgeAccepts`: the same for `Judge.judge` on a record as its caller holds it, under the schemes the record names.

Neither says the signer signed: the conclusion is that the signature verifies, not that it was made by the key's holder.
`VerdictSigned` and `JudgeSigned` say it: the accepted session is one the memory challenger holding its key ran,
recorded and timed itself, or the run's own data carries a break (an LMS forgery, or a SHA-512 collision). All four are
proved (`Proofs.Properties.Pous.MemoryChallenger`).
-/

namespace MemoryChallenger.Properties

/-- What an accepted verdict establishes about its inputs. -/
def Accepted (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (s : SignedSession)
    (c : SessionCommitment) (openings : Array CoinOpening) : Prop :=
  (∃ p, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧
      ∃ m, s.message = .ok m ∧ lmsVerify p s.publicKey s.commitment.session m s.signature = .ok true) ∧
    s.commitment = c ∧ s.ended = COMPLETE ∧ s.rounds.size = c.k ∧
    ∀ j (hj : j < s.rounds.size),
      s.rounds[j].j = j ∧ (∃ o ∈ openings, o.j = j ∧ o.index = s.rounds[j].index ∧ verifyCoin c o = true) ∧
        s.rounds[j].index < c.blocks ∧ expected s.rounds[j].index = some s.rounds[j].digest ∧
        0 ≤ s.rounds[j].elapsedNs ∧ s.rounds[j].elapsedNs ≤ c.limitNs

def VerdictAccepts : Prop :=
  ∀ (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (s : SignedSession)
    (c : SessionCommitment) (openings : Array CoinOpening),
    (verdict schemes trusted expected s c openings).accepted = true → Accepted schemes trusted expected s c openings

def JudgeAccepts : Prop :=
  ∀ (r : Judge.JudgeInputs), Verdict.accepted <$> Judge.judge r = .ok true →
    ∃ schemes, Judge.schemesOf r.schemes = .ok schemes ∧
      Accepted schemes r.trusted.toList r.expected.find r.signed r.commitment r.openings

/-! ## What the memory challenger signs

The device, as `memory_challenger.MemoryChallenger` runs a session (its module docstring): it builds the commitment of
its coins and the session's parameters (`__init__`) and keeps it until the session is signed; each round,
`challenge()` reveals the next coin's index and reads its clock (the send time), and `receive(block)` reads its clock
again (the receipt time), refuses an answer that is not `block_bytes` long, and records
`Round(j, index, send, receipt, block_digest(block))`; the first late round ends the session `late`, the `k`-th on-time
round `complete`; `close()` signs the ended session once (`LmsSigner.sign`), the message
`memoryChallengerMessage(I, session, message(commitment, ended, rounds))` under its key, and only then releases the
commitment.

The model takes the device's clock readings and the answers it receives as they came, and assumes nothing of them: a
round's times are the device's own readings, whatever its clock did. It does not model what refuses a session before
it starts (a limit or answer size that is not positive, a clock coarser than a microsecond): those restrict the runs, not
what a run signs. The device may run any number of sessions under a key, each signed once. A key signs only its
sessions' messages, one role per key: `Signed` is all the model's key signs, so a key also used for another role
(`lms.ROLES` has `network-certificate` too) is outside the model, and the third disjunct is an EUF-CMA break only
under that one-role reading.

Nothing is assumed of SHA-512 or LMS. `VerdictSigned` holds, or the run's own data carries a break: two distinct
strings of `runStrings`, the strings the session and the device's sessions under its key hash, with one SHA-512 digest
(`cr/sha-512`), or a signature that verifies under the session's key over its message (`sessionMessage`) though the
device never signed that message (`euf-cma/lms-sha512-n64`). -/

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
def Run.commitment (r : Run) : SessionCommitment :=
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
structure Device where
  sessions : ByteArray → List Run

/-- The messages the device signs under the key `pk`: for each session it runs under `pk` and ends,
`memoryChallengerMessage` of the key's identifier, the session id and the session's `message`. -/
def Signed (dev : Device) (pk m : ByteArray) : Prop :=
  ∃ r ∈ dev.sessions pk, ∃ ended rec, r.recorded.ended = some ended ∧
    message r.commitment ended r.recorded.rounds = .ok rec ∧
    memoryChallengerMessage (pk.extract 0 16) r.commitment.session rec = .ok m

/-- The LMS verifier accepts `sig` (once decoded) under the key `pk` over the message `m`. -/
def LmsVerifies (p : Lms.Params) (pk m sig : ByteArray) : Prop :=
  ∃ σ, Lms.Signature.decode p sig = .ok σ ∧ Lms.verify p pk m σ = .ok true

/-- The message a signed session's signature signs: `memoryChallengerMessage` of its key's identifier, its session id
and its `message`. -/
def sessionMessage (s : SignedSession) : Option ByteArray :=
  (s.message >>= memoryChallengerMessage (s.publicKey.extract 0 16) s.commitment.session).toOption

/-! ### The strings a run hashes -/

/-- What `FV3.hash tag parts` hashes. -/
def frameInput (tag : String) (parts : List ByteArray) : ByteArray :=
  parts.foldl (fun acc p => acc ++ Flock.be64 p.size.toUInt64 ++ p)
    (Flock.FV3.FRAME ++ Flock.be32 tag.toUTF8.size.toUInt32 ++ tag.toUTF8)

/-- What SHA-512 hashes for the records tree over `values` under the domain `d`: each leaf, and each node of each
level below the root. -/
def treeStrings (d : FrameV3Sha512.Domain) (values : Array ByteArray) : List ByteArray :=
  let levels := FrameV3Sha512.levels d ((Array.range values.size).map fun r => d.leaf r r RECORD_SCHEMA values[r]!)
  (List.range values.size).map (fun r =>
      frameInput "leaf" [d.id, FrameV3Sha512.uint r, FrameV3Sha512.uint r, RECORD_SCHEMA.toUTF8, values[r]!]) ++
    (List.range (levels.size - 1)).flatMap fun l => (List.range (levels[l]!.size / 2)).map fun i =>
      frameInput "node" [d.id, FrameV3Sha512.uint l, FrameV3Sha512.uint i, levels[l]![2 * i]!, levels[l]![2 * i + 1]!]

/-- What SHA-512 hashes for a session `(c, ended, rounds)` signed under the key `pk`: the coin key (the header holds
its digest), the records tree, the input of the session's `message`, and that of `memoryChallengerMessage`. -/
def sessionStrings (pk : ByteArray) (c : SessionCommitment) (ended : String) (rounds : Array Round) : List ByteArray :=
  let head := (header c ended rounds.size).toUTF8
  [c.coinKey] ++
    (match rounds.mapM Round.value with
      | .ok values => treeStrings (domain RECORD_BINDING c.session rounds.size) values
      | .error _ => []) ++
    (match recordsRoot c.session rounds with
      | .ok root => [SESSION_TAG ++ Flock.be64 head.size.toUInt64 ++ head ++ root]
      | .error _ => []) ++
    match message c ended rounds with
    | .ok rec =>
      if (pk.extract 0 16).size = 16 then
        [field MEMORY_CHALLENGER_TAG.toUTF8 ++ pk.extract 0 16 ++ field (field c.session) ++ field rec]
      else []
    | .error _ => []

/-- The strings the run hashes: the signed session's, and those of each session the device ends under its key. -/
def runStrings (s : SignedSession) (dev : Device) : List ByteArray :=
  sessionStrings s.publicKey s.commitment s.ended s.rounds ++
    (dev.sessions s.publicKey).flatMap fun r => match r.recorded.ended with
      | some e => sessionStrings s.publicKey r.commitment e r.recorded.rounds
      | none => []

/-- The signed session `s` is one the device ran under its key: its commitment is the one the device built, and
its end and every round (index, digest, send and receipt times) are the ones the device recorded; or two distinct
strings of the run have one SHA-512 digest; or `s`'s signature verifies under a trusted key and an accepted scheme over
its message, which the device never signed under that key. -/
def SignedByDevice (schemes : Schemes) (trusted : List ByteArray) (s : SignedSession) (dev : Device) : Prop :=
  (∃ r ∈ dev.sessions s.publicKey, r.commitment = s.commitment ∧ r.recorded.ended = some s.ended ∧
      r.recorded.rounds = s.rounds) ∨
    (∃ x ∈ runStrings s dev, ∃ y ∈ runStrings s dev, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) ∨
    ∃ p m, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧ sessionMessage s = some m ∧
      LmsVerifies p s.publicKey m s.signature ∧ ¬ Signed dev s.publicKey m

/-- **An accepted session is the device's.** For every device, an accepted verdict's session is `SignedByDevice`. -/
def VerdictSigned : Prop :=
  ∀ (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (s : SignedSession)
    (c : SessionCommitment) (openings : Array CoinOpening) (dev : Device),
    (verdict schemes trusted expected s c openings).accepted = true → SignedByDevice schemes trusted s dev

/-- `VerdictSigned` for `Judge.judge` on a record as its caller holds it, under the schemes the record names. The
forgery disjunct is at the record's own scheme parameters: a reader cites `euf-cma/lms-sha512-n64` only when the
record's scheme is `LMS` (`lms-sha512/n64-w1-h20/v0`) at its deployed parameters (`Lms.deployed`, Python
`lms.DEPLOYED`). -/
def JudgeSigned : Prop :=
  ∀ (r : Judge.JudgeInputs) (dev : Device), Verdict.accepted <$> Judge.judge r = .ok true →
    ∃ schemes, Judge.schemesOf r.schemes = .ok schemes ∧ SignedByDevice schemes r.trusted.toList r.signed dev

end MemoryChallenger.Properties
