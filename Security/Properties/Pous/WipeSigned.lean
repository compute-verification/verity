import MemoryChallenger.WipeProtocol
import Properties.Pous.MemoryChallenger
import Definitions.Pous.Model.StreamedWipe

/-!
# An accepted wipe is the memory challenger's own run, and that run is the streamed wipe's accepted session

`MemoryChallenger.WipeProtocol.protocol` (`verity/wipe/v0`, the wipe's component) accepts a signed session whose
signature verifies under a trusted memory challenger's key. `WipeSigned` says what that acceptance means, over a model
of the memory challengers (`Challenger`): the session is a wipe the challenger holding the key ran itself, and that run
is the streamed-label wipe's accepted session (`WipeRun.Accepted`): the challenger recorded an answer to each of its
`k` reads, revealed one at a time after the fill, each answer its read's label and within the deadline, so every prover
of the game of `Pous.StreamedWipe` that plays the run is accepted there (`Pous.StreamedWipe.Accepts`). Or the run's own
data carries a break (`WipeBreak`): two distinct strings it hashes with one SHA-512 digest (`cr/sha-512`), or a
signature that verifies under the trusted key over the session's message, which the challenger never signed under it
(`euf-cma/lms-sha512-n64`).

It is the step from the code to the game that `MemoryChallenger.Properties.WipeSurvivorBound` (MEM-84) and the
composition read: the accepted wipe is the game's accepted session at the run's labels and reads, so the streamed
wipe's guarantees (`Pous.Guarantees.StreamedWipe` and its instances at 2152 reads) bound it, with nothing between the
code and the game but the game's shape (C5, C6 and every tier declared), which the challenger's deployment makes true.

## The memory challenger, as `wipe_party` runs a wipe

`streamed_wipe.Challenger`, as `wipe_party.serve` drives it (their module docstrings): it claims the session id and
publishes the commitment (`__init__`); it reads its clock, streams the labels `L_i = SHA-512(LABEL_TAG ‖ K ‖ u64 i)` of
its key `K` (`fill`, or `fill_by` through the native streamer) and reads its clock again; once the fill has ended it
draws its `k` reads and reveals them one at a time (`challenge`, reading its clock: the send time), each only after the
previous one's answer arrived (`challenge` refuses while a batch is open); `receive(answer)` reads its clock (the receipt
time), refuses an answer that is not one label long, and records batch `b` as `(b, b, 1, send, receipt, wrong,
SHA-512(u64 index ‖ answer))`, `wrong` 1 when the answer is not the read's label; the first late batch ends the session
`late`, the first wrong one `wrong`, the `k`-th `complete`; `close(egress_bits)` signs the ended session once, with the
egress the party was given (`--egress-bits`), the message `memoryChallengerMessage(I, session, message(commitment,
ended, fill_start, fill_end, batches, egress))` under its key.

The wipe party reveals one index a batch (`wipe_party.params_of`): a wipe at another batch size under the key is outside
the model. The model takes the challenger's clock readings and the answers it receives as they came, and assumes
nothing of them. It does not model what refuses a session before it starts. The challenger may run any number of wipes
under a key, each signed once, and the key signs PoUS's sessions too: `wipe_party.main` signs with
`period.challenger`, the memory challenger's key, in PoUS's role (`memory-challenger-record`). So `Challenger.Signed`,
all the model's key signs, holds both. A key also used for another role is outside the model, and the forgery disjunct
is an EUF-CMA break only under that one-role reading, as for PoUS's sessions (`Device`).
-/

namespace MemoryChallenger.Properties

open Flock

/-! ## The challenger's wipes -/

/-- `streamed_wipe.LABEL_TAG`. -/
def WIPE_LABEL_TAG : ByteArray := "verity-pous/streamed-wipe/label/v1".toUTF8.push 0

/-- `streamed_wipe.label(K, i)`: label `i` of the key `K`, `SHA-512(LABEL_TAG ‖ K ‖ u64 i)`, at an index below `2^64` as
`label` requires. -/
def wipeLabel (key : ByteArray) (i : Nat) : ByteArray := Sha512.hash (WIPE_LABEL_TAG ++ key ++ be64 i.toUInt64)

/-- `streamed_wipe.batch_digest` of one index and its answer: `SHA-512(u64 index ‖ answer)`. -/
def wipeBatchDigest (index : Nat) (answer : ByteArray) : ByteArray := Sha512.hash (be64 index.toUInt64 ++ answer)

/-- `streamed_wipe.WRONG`. -/
def WIPE_WRONG : String := "wrong"

/-- What the challenger observes of one read it reveals: its clock's reading as it reveals the index, its reading as the
answer arrives, and the answer. -/
structure WipeObserved where
  sendNs : Int
  receiptNs : Int
  answer : ByteArray

/-- A wipe the challenger runs: the commitment it publishes when it opens the session, the labels' key `K`, the `k`
reads it draws once the fill ends, its clock's readings as the fill starts and as it ends, what it observes of each read
it reveals, in order, and the egress it is given for the session, in bits. -/
structure WipeRun where
  commitment : StreamedWipe.Commitment
  key : ByteArray
  reads : Fin commitment.params.k → Fin commitment.params.labels
  fillStartNs : Int
  fillEndNs : Int
  observed : List WipeObserved
  egressBits : Nat

/-- The challenger's record of a wipe: its batch records, the answer each batch records, and how the session ended, once
it has. The challenger keeps only an answer's digest; the model keeps the answer. -/
structure WipeRecorded where
  batches : Array StreamedWipe.Batch
  answers : Array ByteArray
  ended : Option String

/-- One read, `challenge()` then `receive(answer)`, one index a batch: nothing once the session has ended (`challenge`
refuses), nothing recorded for an answer that is not one label long (`receive` refuses it), else batch `b` with read
`b`'s index, the two readings, whether the answer is the read's label, and the digest, ending the session `late` if the
answer is not within the deadline, `wrong` if it is not the label, and `complete` at the `k`-th read. -/
def WipeRun.step (r : WipeRun) (st : WipeRecorded) (o : WipeObserved) : WipeRecorded :=
  let p := r.commitment.params
  let b := st.batches.size
  if h : st.ended = none ∧ b < p.k ∧ o.answer.size = StreamedWipe.LABEL_BYTES then
    let i : Nat := r.reads ⟨b, h.2.1⟩
    let wrong : Int := if o.answer = wipeLabel r.key i then 0 else 1
    let x : StreamedWipe.Batch := ⟨b, b, 1, o.sendNs, o.receiptNs, wrong, wipeBatchDigest i o.answer⟩
    ⟨st.batches.push x, st.answers.push o.answer,
      if !(0 ≤ x.elapsedNs && x.elapsedNs ≤ (p.limitNs : Int)) then some LATE
      else if wrong != 0 then some WIPE_WRONG
      else if b + 1 == p.k then some StreamedWipe.COMPLETE else none⟩
  else st

/-- What the challenger records of a wipe: its observations, read by read. -/
def WipeRun.recorded (r : WipeRun) : WipeRecorded := r.observed.foldl r.step ⟨#[], #[], none⟩

/-- The record message the challenger signs when it closes the wipe, ended `ended` (`close`): `streamed_wipe.message`
of its commitment, how it ended, the fill's readings, its batch records and its egress. -/
def WipeRun.message (r : WipeRun) (ended : String) : E ByteArray :=
  StreamedWipe.message r.commitment ended r.fillStartNs r.fillEndNs r.recorded.batches r.egressBits

/-- The memory challengers: the sessions the holder of each key runs under it, PoUS's (`Device`) and its wipes. -/
structure Challenger where
  device : Device
  wipes : ByteArray → List WipeRun

/-- The messages the challenger signs under the key `pk`: its PoUS sessions' (`Signed`), and for each wipe it runs under
`pk` and ends, `memoryChallengerMessage` of the key's identifier, the session id and the wipe's `message`. -/
def Challenger.Signed (dev : Challenger) (pk m : ByteArray) : Prop :=
  MemoryChallenger.Properties.Signed dev.device pk m ∨
    ∃ r ∈ dev.wipes pk, ∃ ended rec, r.recorded.ended = some ended ∧ r.message ended = .ok rec ∧
      memoryChallengerMessage (pk.extract 0 16) r.commitment.session rec = .ok m

/-- The message a signed wipe's signature signs: `memoryChallengerMessage` of its key's identifier, its session id and
its `message`. -/
def wipeMessage (s : StreamedWipe.Signed) : Option ByteArray :=
  (s.message >>= memoryChallengerMessage (s.publicKey.extract 0 16) s.commitment.session).toOption

/-! ### The strings a wipe hashes -/

/-- What SHA-512 hashes for a wipe's records tree over `values` under the domain `d`: each leaf, and each node of each
level below the root (`treeStrings` at the wipe's record schema). -/
def wipeTreeStrings (d : FrameV3Sha512.Domain) (values : Array ByteArray) : List ByteArray :=
  let levels := FrameV3Sha512.levels d
    ((Array.range values.size).map fun r => d.leaf r r StreamedWipe.RECORD_SCHEMA values[r]!)
  (List.range values.size).map (fun r =>
      frameInput "leaf"
        [d.id, FrameV3Sha512.uint r, FrameV3Sha512.uint r, StreamedWipe.RECORD_SCHEMA.toUTF8, values[r]!]) ++
    (List.range (levels.size - 1)).flatMap fun l => (List.range (levels[l]!.size / 2)).map fun i =>
      frameInput "node" [d.id, FrameV3Sha512.uint l, FrameV3Sha512.uint i, levels[l]![2 * i]!, levels[l]![2 * i + 1]!]

/-- What SHA-512 hashes for a wipe `(c, ended, the fill's readings, batches, egress)` signed under the key `pk`: its
records tree, the input of its `message`, and that of `memoryChallengerMessage`. -/
def wipeSessionStrings (pk : ByteArray) (c : StreamedWipe.Commitment) (ended : String) (fillStartNs fillEndNs : Int)
    (batches : Array StreamedWipe.Batch) (egressBits : Int) : List ByteArray :=
  let head := (StreamedWipe.header c ended fillStartNs fillEndNs batches.size egressBits).toUTF8
  (match batches.mapM StreamedWipe.Batch.value with
    | .ok values => wipeTreeStrings (domain StreamedWipe.RECORD_BINDING c.session batches.size) values
    | .error _ => []) ++
    (match StreamedWipe.recordsRoot c.session batches with
      | .ok root => [StreamedWipe.SESSION_TAG ++ be64 head.size.toUInt64 ++ head ++ root]
      | .error _ => []) ++
    match StreamedWipe.message c ended fillStartNs fillEndNs batches egressBits with
    | .ok rec =>
      if (pk.extract 0 16).size = 16 then
        [field MEMORY_CHALLENGER_TAG.toUTF8 ++ pk.extract 0 16 ++ field (field c.session) ++ field rec]
      else []
    | .error _ => []

/-- The strings the run hashes: the signed wipe's, those of each wipe the challenger ends under its key, and those of
each PoUS session it ends under the key (`sessionStrings`). -/
def wipeRunStrings (s : StreamedWipe.Signed) (dev : Challenger) : List ByteArray :=
  wipeSessionStrings s.publicKey s.commitment s.ended s.fillStartNs s.fillEndNs s.batches s.egressBits ++
    (dev.wipes s.publicKey).flatMap (fun r => match r.recorded.ended with
      | some e => wipeSessionStrings s.publicKey r.commitment e r.fillStartNs r.fillEndNs r.recorded.batches r.egressBits
      | none => []) ++
    (dev.device.sessions s.publicKey).flatMap fun r => match r.recorded.ended with
      | some e => sessionStrings s.publicKey r.commitment e r.recorded.rounds
      | none => []

/-- **A break in a wipe's run**: two distinct strings the run hashes with one SHA-512 digest (`cr/sha-512`), or a
signature of `s` that verifies under a trusted key and an accepted scheme over its message, which the challenger never
signed under that key (`euf-cma/lms-sha512-n64`). -/
def WipeBreak (schemes : Schemes) (trusted : List ByteArray) (s : StreamedWipe.Signed) (dev : Challenger) : Prop :=
  (∃ x ∈ wipeRunStrings s dev, ∃ y ∈ wipeRunStrings s dev, x ≠ y ∧ Sha512.hash x = Sha512.hash y) ∨
    ∃ p m, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧ wipeMessage s = some m ∧
      LmsVerifies p s.publicKey m s.signature ∧ ¬ dev.Signed s.publicKey m

/-! ## The streamed wipe's accepted session -/

/-- A label's 512 bits as the game reads them (`Pous.Bits 512`): bit `i` is bit `7 - i % 8` of byte `i / 8`, most
significant first, the order SHA-512 writes its digest in. -/
def bits512 (b : ByteArray) : Pous.Bits 512 := fun i => (b.get! (i.val / 8)).toNat.testBit (7 - i.val % 8)

/-- The run's labels as the game's (`Fin N → Bits 512`): label `i` is `wipeLabel K i`. -/
def WipeRun.labels (r : WipeRun) : Fin r.commitment.params.labels → Pous.Bits 512 :=
  fun i => bits512 (wipeLabel r.key i)

/-- **A prover of the game plays the run**: with prior content `z` and the run's labels, its answer to each read `j`,
from what it holds when the fill ends, its egress, the reads revealed before `j` in order and `j`'s index, is the answer
the challenger recorded for read `j`. -/
def WipeRun.Plays (r : WipeRun) {q M E : ℕ}
    (P : Pous.StreamedWipe.Prover r.commitment.params.labels r.commitment.params.k 512 q M E) (z : Pous.Bits q) :
    Prop :=
  ∀ j : Fin r.commitment.params.k,
    P.answer (P.fill z r.labels).1 (P.fill z r.labels).2 (Pous.StreamedWipe.revealed r.reads j) (r.reads j) =
      bits512 r.recorded.answers[j.val]!

/-- **The run is the streamed wipe's accepted session.** The challenger ended it `complete` with a batch and an answer
recorded for each of its `k` reads; read `j`'s answer is its label, `wipeLabel K (c j)`, and arrived within the
deadline; the first read was revealed after the fill ended. So every prover of the game (`Pous.StreamedWipe`) that plays
the run is accepted on the run's labels and reads (`Pous.StreamedWipe.Accepts`). -/
def WipeRun.Accepted (r : WipeRun) : Prop :=
  r.recorded.ended = some StreamedWipe.COMPLETE ∧ r.recorded.batches.size = r.commitment.params.k ∧
    r.recorded.answers.size = r.commitment.params.k ∧
    (∀ j : Fin r.commitment.params.k, r.recorded.answers[j.val]! = wipeLabel r.key (r.reads j) ∧
      0 ≤ r.recorded.batches[j.val]!.elapsedNs ∧
      r.recorded.batches[j.val]!.elapsedNs ≤ (r.commitment.params.limitNs : Int)) ∧
    r.fillEndNs ≤ r.recorded.batches[0]!.sendNs ∧
    ∀ {q M E : ℕ} (P : Pous.StreamedWipe.Prover r.commitment.params.labels r.commitment.params.k 512 q M E)
      (z : Pous.Bits q), r.Plays P z → Pous.StreamedWipe.Accepts P z r.labels r.reads

/-- **An accepted wipe is the challenger's own run, and that run is the streamed wipe's accepted session.**
Deterministic and assumption free: for every model `dev` of the memory challengers, if the wipe's compiled check
(`WipeProtocol.protocol.check`, the check the composition runs on the wipe's case) accepts the commitment `c` and the
evidence `e` under the parameters `p`, then `e` carries a signed session `s` under one of `p`'s trusted keys, and either
the challenger holding that key ran the wipe itself, a run `r` under the key whose commitment is `c`, whose fill
readings, batch records and egress are `s`'s, and which is the streamed wipe's accepted session (`WipeRun.Accepted`), or
the run's data carries a break (`WipeBreak`). -/
def WipeSigned : Prop :=
  ∀ (p : WipeProtocol.Params) (c : StreamedWipe.Commitment) (e : WipeProtocol.Evidence) (out : WipeProtocol.Public)
    (prof : StreamedWipe.Params × Int) (dev : Challenger),
    WipeProtocol.protocol.check p c e = .accept out prof →
      ∃ s, e.signed = some s ∧ s.publicKey ∈ p.trusted.toList ∧
        ((∃ r ∈ dev.wipes s.publicKey, r.commitment = c ∧ r.fillStartNs = s.fillStartNs ∧
            r.fillEndNs = s.fillEndNs ∧ r.recorded.batches = s.batches ∧ (r.egressBits : Int) = s.egressBits ∧
            r.Accepted) ∨
          WipeBreak SCHEMES p.trusted.toList s dev)

end MemoryChallenger.Properties
