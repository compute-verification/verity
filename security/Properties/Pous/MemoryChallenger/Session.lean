module

public import Definitions.Accounting.Assumptions.Parties
public import Properties.Pous.MemoryChallenger

/-!
# What the auditor sees of the memory challenger in a session

In a session the memory challenger takes the prover's answers, each a whole block of `C*`, the encoding of the node's
store. This module states what of those answers reaches the auditor from the device: one statement,
`AuditorSeesSkeletonAt`, whose parameter says where the device's messages end, and its instance as PoUS runs a session,
`AuditorSeesDigests`. No soundness statement takes either as a premise.

**The device** is an auditor party (`Accounting.Assumptions.Parties`), stated as code (`code`) over the record
soundness reads of a session (`MemoryChallenger.Properties.Observed`, `Recorded.round`): the party
`challenger_party.Challenger` driving `memory_challenger.MemoryChallenger`, one session a run. An input is one exchange:
a warm-up's answer, which the device drops, or a timed round, its clock's two readings and the answer. It sends
challenges to the prover (a warm-up's frame, a round's index, the frame that ends the session) and, once a round has
ended the session, its report: the session it signed, the commitment it releases with it and the opening of every coin
it revealed.

**The statement.** `Recorded.round` reads an answer only through its size and its `blockDigest`. So an exchange's
skeleton (`Step.skeleton`) holds nothing of a warm-up's answer and, of a round, the two readings, the answer's size and
its `blockDigest`, and `seenCode` is the same code taking skeletons: no answer's bytes are among its inputs.
`AuditorSeesSkeletonAt dest` says that under `Faithful`, `Confined` (at `dest`) and `WitnessPathPrivate`, what the
auditor sees of the device is what `seenCode` sends in the clear on the device's coins and the skeleton of its true
inputs.

* `AuditorSeesDigests`, at `reportPublic`: the session's verdict is checked in the clear on the device's report, so
  the report is public, while the challenges stay with the prover. Of the answers the auditor then learns at most each
  round's size and `blockDigest`.
* At a `dest` where every channel ends with the prover, the statement says the auditor sees nothing of the device.
  That is the path on which the prover proves the session's verdict instead of showing the report; no instance is
  stated for it.

What neither covers is in `AuditorSeesDigests`'s docstring.
-/

@[expose] public section

namespace MemoryChallenger.Session.Properties

open Accounting.Assumptions
open MemoryChallenger.Properties (Observed Recorded)

/-! ## The device's code in a session -/

/-- What the device's code in a session reads besides its inputs. From its coins `ω`, which stand for all the device
holds of its own, its key and the one-time key it signs at among them: the session's coin tree `coins ω`
(`CoinTree.draw`: the session's id, its indices, the tree's hiding key and salts), its signer's public key
`publicKey ω`, and the one signature it makes, `sign ω session record` (`Signer.sign`). The session's parameters: the
limit `Δ + RTT`, the answer's size, the root of the `C*` the session audits (`binding`) and the signer's scheme. -/
structure Params (Ω : Type) where
  coins : Ω → CoinTree
  limitNs : Nat
  blockBytes : Nat
  binding : ByteArray
  scheme : String
  publicKey : Ω → ByteArray
  sign : Ω → ByteArray → ByteArray → ByteArray

/-- The session's commitment, which the device builds before the session and releases once it is signed
(`MemoryChallenger.Properties.Run.commitment`). -/
def Params.commitment {Ω : Type} (p : Params Ω) (ω : Ω) : SessionCommitment :=
  (⟨p.coins ω, p.limitNs, p.blockBytes, p.binding, []⟩ : MemoryChallenger.Properties.Run).commitment

/-- The device's channels in a session: its challenges, one frame an exchange to the prover over the session's
connection, and its report, once the session is signed. -/
inductive Ch
  | challenge
  | report
  deriving DecidableEq

/-- What the device sends. On `Ch.challenge`: a warm-up's frame (on an index no coin chose), a round's index, and the
frame that ends the session. On `Ch.report`: the signed session, the commitment released with it, and the opening of
every coin it revealed (`run.ChallengePhaseRecord`). -/
inductive Msg
  | warmup
  | index (i : Nat)
  | done
  | report (signed : SignedSession) (commitment : SessionCommitment) (openings : Array CoinOpening)

/-- One exchange of a session, over what is taken of a warm-up's answer (`W`) and of a round (`R`): a warm-up, whose
answer the device drops, or a timed round. -/
inductive Step (W R : Type)
  | warmup (answer : W)
  | round (o : R)

/-- What the device's record reads of a round (`Recorded.round`): its clock's reading as it reveals the index, its
reading as the answer arrives, the answer's size, and the answer's `blockDigest`. -/
structure Seen where
  sendNs : Nat
  receiptNs : Nat
  size : Nat
  digest : ByteArray

/-- What the device's record reads of the round it observed: the two readings as they are, and of the answer its size
and its `blockDigest`. -/
def seen (o : Observed) : Seen := ⟨o.sendNs, o.receiptNs, o.block.size, blockDigest o.block⟩

/-- An exchange's skeleton: nothing of a warm-up's answer, and of a round what the record reads of it (`seen`). -/
def Step.skeleton : Step ByteArray Observed → Step Unit Seen
  | .warmup _ => .warmup ()
  | .round o => .round (seen o)

/-- `Recorded.round` on what it reads of a round: the same record, from the answer's size and digest in place of the
answer. -/
def roundSeen (coins : CoinTree) (limitNs blockBytes : Nat) (st : Recorded) (o : Seen) : Recorded :=
  if st.ended.isSome || o.size != blockBytes then st else
  let r : Round := ⟨st.rounds.size, coins.indices[st.rounds.size]!, o.sendNs, o.receiptNs, o.digest⟩
  let rounds := st.rounds.push r
  ⟨rounds, if !(0 ≤ r.elapsedNs && r.elapsedNs ≤ limitNs) then some LATE
    else if rounds.size == coins.k then some COMPLETE else none⟩

/-- What the device sends once a round has ended the session whose record is `rec` (`close`, then `commitment` and
`openings`): the frame that ends it, then its report, the session signed over `message` of its commitment, its end and
its rounds. Nothing while the session is open, and nothing for a record `message` refuses. -/
def Params.closing {Ω : Type} (p : Params Ω) (ω : Ω) (rec : Recorded) : List (Ch × Msg) :=
  match rec.ended with
  | none => []
  | some ended =>
    match message (p.commitment ω) ended rec.rounds with
    | .ok m =>
      [(Ch.challenge, Msg.done),
        (Ch.report, Msg.report
          ⟨p.commitment ω, ended, rec.rounds, p.scheme, p.publicKey ω, p.sign ω (p.commitment ω).session m⟩
          (p.commitment ω) (rec.rounds.map fun r => (p.coins ω).opening r.j))]
    | .error _ => []

/-- The device's code in one session, over the way `record` it records a round: its state is its coins and its record
so far. A warm-up sends its frame and changes nothing. A round, while the session is open, sends the index of the round
the record is at (`challenge`), records the round and, if that ended the session, closes it (`Params.closing`). After
the end a round does nothing (`challenge` refuses). An exchange is one step, so a round's index counts as sent in the
step that takes its answer. -/
def codeOn {Ω W R : Type} (p : Params Ω) (record : CoinTree → Nat → Nat → Recorded → R → Recorded) :
    Parties.Code Ω (Ω × Recorded) (Step W R) Ch Msg where
  init ω := (ω, ⟨#[], none⟩)
  step st x := match x with
    | .warmup _ => (st, [(Ch.challenge, Msg.warmup)])
    | .round o =>
      if st.2.ended.isSome then (st, []) else
      let next := record (p.coins st.1) p.limitNs p.blockBytes st.2 o
      ((st.1, next), (Ch.challenge, Msg.index (p.coins st.1).indices[st.2.rounds.size]!) :: p.closing st.1 next)

/-- The device's code: `codeOn` with soundness's record of a round (`Recorded.round`), on its true inputs, each answer's
bytes among them. -/
def code {Ω : Type} (p : Params Ω) : Parties.Code Ω (Ω × Recorded) (Step ByteArray Observed) Ch Msg :=
  codeOn p Recorded.round

/-- The same code on skeletons: `codeOn` with `roundSeen`. No answer's bytes are among its inputs: of a warm-up it
takes nothing, of a round the two readings, the answer's size and its `blockDigest`. -/
def seenCode {Ω : Type} (p : Params Ω) : Parties.Code Ω (Ω × Recorded) (Step Unit Seen) Ch Msg :=
  codeOn p roundSeen

/-! ## What the auditor sees -/

/-- What a list of messages puts in the clear, where `dest` says each channel ends. -/
def inClear (dest : Ch → Parties.Dest) (ms : List (Ch × Msg)) : List (Ch × Msg) :=
  ms.filter fun m => dest m.1 = .clear

/-- **The auditor sees of the device only what its code sends in the clear on skeletons.** `dest` says where each of
the device's channels ends: with the auditor in the clear, or with the prover only. A run of the device `Faithful` to
its code, `Confined` to its channels as `dest` has them, and with its witness path private (`WitnessPathPrivate`) shows
the auditor what `seenCode` sends in the clear on the device's coins and the skeleton of its true inputs, as it goes on
the wire. No answer's bytes are among `seenCode`'s inputs, so the auditor's view is computed from the device's coins,
its clock's readings, and each round's answer's size and `blockDigest`, and from nothing else of an answer: two runs
from the same coins whose inputs have one skeleton show the auditor the same.

It reads the device's code (`code`: `Recorded.round`, `message`, the coin tree's openings), the party's run and the
auditor's view of it (`Parties.Run`, `Parties.auditorView`), and the three premises. It is stated at every `dest`;
which `dest` a deployment has is that deployment's `Confined`. -/
def AuditorSeesSkeletonAt (dest : Ch → Parties.Dest) : Prop :=
  ∀ {Ω Wi A : Type} (p : Params Ω) (wire : Ch × Msg → Wi)
    (r : Parties.Run Ω (Ω × Recorded) (Step ByteArray Observed) Ch Msg Wi A),
    Parties.Faithful (code p) r → Parties.Confined dest wire r → Parties.WitnessPathPrivate r →
      Parties.auditorView wire r =
        (inClear dest ((seenCode p).messages r.coins (r.truth.map Step.skeleton))).map wire

/-- Where the device's channels end as PoUS runs a session: its challenges with the prover, and its report in the
clear. The device hands the report to the party that started its side and answers its challenges
(`challenger_session.session`), and the session's verdict is checked in the clear on it (`run.check_at`), as evidence
of the run's case: the report is public. The report goes first to the prover's side, which in the run's case passes it
on unchanged; reading that channel as ending with the auditor in the clear gives the auditor more than the path it
takes, so the bound holds of what happens. -/
def reportPublic : Ch → Parties.Dest
  | .challenge => .witness
  | .report => .clear

/-- **Of a session's answers the auditor learns at most each round's digest.** `AuditorSeesSkeletonAt` with the report
public (`reportPublic`): what the auditor sees of the device in a session is the report `seenCode` computes from the
device's coins, its clock's two readings a round, and each round's answer's size and `blockDigest`. No warm-up's answer
and no other function of a round's answer enters it.

Not covered:
* what a digest tells. `blockDigest` is SHA-512 over a public prefix and the answer, with no key and no salt, so
  whoever holds a candidate for the answer can test it. The statement bounds what reaches the auditor by the digest; it
  does not say the digest hides the answer;
* what the prover sends the auditor. The run's case also carries a table of every block's `blockDigest`, and in the
  public-encoder setup `C*` itself (`period.case`, `setup_case`). The view is of the device alone: the statement
  bounds what the device adds to what the auditor holds, not all the auditor holds;
* when a message is sent: the view holds the messages in order, and the clock's readings only as the report has them;
* the device's host: what the device keeps (its state, its logs, the sessions it completed) is not in the view;
* what else the device sends: the port it listens on before the session, another session's messages, the salt's
  source it gives setup, and its signed period roots and liveness records, which
  `MemoryChallenger.Attest.Properties.AuditorSeesSkeleton` covers;
* a session cut short inside an exchange: a challenge the prover leaves unanswered is none of the code's messages, so
  the run is not `Confined` and the statement is silent on it. The device signs nothing and prints no report then;
* that the Python device is `code`: no executable Lean of the device exists to hold it to. -/
def AuditorSeesDigests : Prop := AuditorSeesSkeletonAt reportPublic

end MemoryChallenger.Session.Properties
