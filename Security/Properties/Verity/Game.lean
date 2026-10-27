import CoreOnsite
import Properties.Challenger
import Properties.Composition
import Properties.Flock.Circuit.CircuitOf
import Properties.Pous.MemoryChallenger
import Definitions.Accounting.JointLive
import Definitions.Core.Game
import Definitions.Hash.RowLeaf
import Flock.Hash

/-!
# Soundness as a causal game at onsite's compiled verifier, and each subprotocol's target

The verifier of record is one compiled Lean function, `Core.Onsite.protocol`'s `answer`: NCI's registration check
(`NciCheck.Protocol.protocol`, in the commit phase, on the registration sampled proofs' one-stage audit commits to) and
then the composition of the subprotocols that decide a period over the blobs the prover sent
(`Core.Composition.registeredB`, the single onsite form). Five of its six components are one per `Accounting.Audit`:

| subprotocol | `Audit` | component |
| --- | --- | --- |
| sampled proofs | `.sampled` | `SampledProofs.OneStageZk.protocol` (v2) |
| PoUW | `.pouw` | `PouwCheck.Period.protocol` (the period's windows credited all or nothing) |
| PoUS | `.pous` | `PousCheck.RunProtocol.protocol` |
| the bandwidth audit | `.poub` | `RemoteAudit.Protocol.protocol` |
| the network certifier | `.network` | `NetworkCheck.Period.protocol` |

The sixth, the wipe (`MemoryChallenger.WipeProtocol.protocol`), has its target in `Properties.Verity.Wipe`.

The statement's form is the causal game, the committed files, one fixed extractor and one adversary form:

* **A causal game.** The prover sends the parameters `θ`, the committed files `F` (one blob) and the
  commitment `c` first, before any coin, so every statement is `∀ θ F c`, and then plays
  `play (rounds θ) θ F c`, a `Verity.Game`: in each of the device's rounds the prover sends the bytes (a root,
  a transcript digest) that the round's release answers, and then the device draws and releases that round's coins
  (`releases`). The device is the challenger device (`Properties.Challenger.Device`): its session's log is defined from
  the play's releases, and it signs that session's record and close and the auditor's ledger lines for it
  (`PlayDevice.device`, `Signed`); the evidence and the blobs the components
  fetch by digest come last (`Outcome.sent`: the committed file first, then those blobs).
  A strategy (`Verity.Game.Strategy`) is a function of the coins released so far and never of later ones.
* **The claims read the committed files.** The circuit, its units and its constants are what the verifier reads
  from the files (`Flock.Bridge.circuitOfFile`, `unitsOfFile`, `partsOfFile` at v2's reading of the blob, `read`), and
  the positions are the files' and the commitment's (`place`), never the prover's world. The params hold the circuit
  file's digest (`programDigest`): a file whose SHA-512 isn't it is refused (`Registered`, conjoined beside onsite's
  acceptance until `Properties.Onsite.SampledProofsRegistered` derives it from v2's). The committed file is the
  first blob onsite reads, and the blobs sent with the evidence follow it (`Outcome.sent`).
* **Sampled proofs' claim reads one fixed extractor's transcript.** The extractor (`extract`, the one
  `SampledKSLive*` is stated with, `XplurZC`) is a construction applied to the strategy, never chosen per strategy. The
  claim is that at most an `ε` share of the file's units are unbound in its transcript (`Accounting.unbound`): wrong in
  it, or with a committed input or output the commitment doesn't open to its value there (`Opens`: a registered constant
  to its constant, any other wire by an opening of its row leaf that `Hash.RowLeaf.accepts` passes under the compiled
  SHA-512).
* **One adversary form: every prover, and the breaks two fixed finders output.** There is no strategy class and no
  break in the event. The collision finder `B` (`finder a`, a `Claims.Finder`) is a fixed construction applied to the
  strategy whose SHA-512 calls are at most `c · calls(τ) + d` (`finder_calls`); the LMS forger (`forger a`) is a fixed
  function of the play's outcome, whose output is a forgery (`forgery`, defined here and no field) when it verifies
  under the device's registered LMS key (`registeredKey`) and the play's device never signed the message (`Signed`).
  Subprotocol `a`'s error at a strategy (`error`) is, where `B^τ` outputs a SHA-512 collision with probability at most
  the published `ε_cr` at the strategy's calls (`FinderCR`, quantitative), its published `δ` at those calls plus its
  loss at `Pr[B^τ]` plus the probability that the forger succeeds in the play (`forges`); elsewhere `⊤`. That admits
  C-Flock's conditional bound (`loss = 0`) and the additive one (`crBound = 1`). Two openings of one position and a
  second file with the registered digest (`fileFinder`) enter at `B`, a forged device release at the forger, an event of
  the device's log. The named assumptions are asked only of the finders, each of its own, outside the theorem:
  `ecr/sha-512` of `B` and `euf-cma/lms-sha512-n64` of the forger.
* **The one vocabulary.** The period is `Accounting.PeriodLive` (`live`), and the claim is its `Holds`.
* **Each component's acceptance in what the prover sent.** Subprotocol `a`'s verdict (`Verdict`) is the file check,
  NCI's acceptance of the registration sampled proofs committed to (`RegistrationAccepts`, the conclusion of
  `Properties.Composition.AcceptRegistration`), and `a`'s compiled component's acceptance of its own case. The releases
  each component's check read (`coinsRead`) are the targets' to bound (`Logged`).
* **PoUW counts gates** of the committed circuit the play computed after their window's coin, over the period's windows
  (`computed`), with no pricing.

Each subprotocol's target (`SampledProofsSound`, `PouwSound`, `PousSound`, `BandwidthAuditSound`,
`NetworkCertifierSound`) is `Target a`: at every `θ`, `F`, `c` and strategy, the probability that `a`'s verdict accepts
while its claim fails is at most its error there. `Verity.Soundness` (`Properties.Verity.Soundness`) composes them.

**Any component list (`Deployment`).** A deployment is a composition `registeredB D.id` of a list of components over
NCI's registration check, with the component (if any) that decides each `Audit` (`comp`, each in the list). Onsite
(`onsite`) decides every claim. A deployment that decides only some, such as sampled proofs and the network certifier,
decides the rest by no verdict: its period's `accepts` is false there. Every `…At D` name is the onsite name's
reading at `D`, and the onsite name is it at `onsite` by definition. `Verity.SoundnessAt` composes the targets of the
claims a deployment decides.

The names a subprotocol defines are `Vocab`'s fields: `read` (v2's reading of the blob), `place`, `programDigest` and
`registered`, `rounds`, the device's fixed parameters (`device`) and `coinsRead`, `extract`, the finders and the
forgers, `computed`, `world` (the runs PoUS, the bandwidth audit and the network certifier realize), and `certOf`,
`delta` and `loss` (what the product publishes, functions of the parameters alone).
-/

namespace Verity

open Accounting Core
open scoped ENNReal

/-- The compiled component of onsite that decides `Audit` `a`. -/
def component : Audit → BlobProtocol
  | .sampled => SampledProofs.OneStageZk.protocol
  | .pouw => PouwCheck.Period.protocol
  | .pous => PousCheck.RunProtocol.protocol
  | .poub => RemoteAudit.Protocol.protocol
  | .network => NetworkCheck.Period.protocol

theorem component_mem (a : Audit) : component a ∈ Onsite.components := by
  cases a <;> simp [component, Onsite.components]

/-- **A deployment's verifier of record, as a composition**: NCI's registration check, on the registration sampled
proofs' one-stage audit commits to, and then the components `components` over the blobs the prover sent
(`Core.Composition.registeredB` under the id `id`). `comp a` is the component that decides `Audit` `a`'s claim, if
the deployment composes one; a component that decides no claim here (onsite's wipe) still has to accept. -/
structure Deployment where
  id : String
  components : List BlobProtocol
  comp : Audit → Option BlobProtocol
  comp_mem : ∀ a P, comp a = some P → P ∈ components

namespace Deployment

variable (D : Deployment)

/-- The deployment's compiled verifier. -/
def protocol : BlobProtocol :=
  Core.Composition.registeredB D.id NciCheck.Protocol.protocol SampledProofs.OneStageZk.protocol.id D.components

/-- The deployment composes a component that decides `Audit` `a`'s claim. -/
def Decides (a : Audit) : Prop := (D.comp a).isSome = true

/-- `e` at the claims the deployment decides, and `0` at the others. -/
def restrict (e : Audit → ℝ≥0∞) (a : Audit) : ℝ≥0∞ :=
  match D.comp a with
  | some _ => e a
  | none => 0

end Deployment

/-- **Onsite** (`Core.Onsite.protocol`): its six components, and one per `Audit` (`component`). -/
def onsite : Deployment where
  id := "verity/onsite/v0"
  components := Onsite.components
  comp a := some (component a)
  comp_mem a P h := by cases h; exact component_mem a

/-- What the prover sends a deployment's verifier of record: the parameters, the commitment and the evidence, as
bytes, and the blobs its components read. -/
structure Sent where
  params : ByteArray
  commitment : ByteArray
  evidence : ByteArray
  blobs : Array ByteArray

/-- The deployment's components on the blobs the prover sent. -/
def Sent.componentsAt (D : Deployment) (s : Sent) : List Protocol :=
  D.components.map (·.onBlobs (Blobs.ofArray s.blobs))

/-- Onsite's components on the blobs the prover sent. -/
def Sent.components (s : Sent) : List Protocol := s.componentsAt onsite

/-- **The deployment's verifier of record accepts what the prover sent.** -/
def AcceptsAt (D : Deployment) (s : Sent) : Prop :=
  (D.protocol.answer s.params s.commitment s.evidence s.blobs).verdict = .accept

/-- **Onsite's verifier of record accepts what the prover sent.** -/
def Accepts (s : Sent) : Prop := AcceptsAt onsite s

/-- Onsite's three byte strings decode under its registered composition's formats: the registration's entry `n` and
the components' `ps`, the components' binding `b` and the registration's digest `d`, the registration's case `k` and the
components' `ev`. -/
def Decoded (s : Sent) (n : Composition.Named) (ps : List Composition.Named) (b d : ByteArray) (k : Composition.Case)
    (ev : List Composition.Case) : Prop :=
  (canonicalJson s.params >>= Composition.decodeRegisteredParams) = .ok (n, ps) ∧
    Composition.decodeRegisteredCommitment s.commitment = .ok (b, d) ∧
      Composition.decodeRegisteredEvidence s.evidence = .ok (k, ev)

/-- The protocol `P` accepts its own case in what the prover sent: the three byte strings decode, and one of the
components' cases is `P`'s, as one of the deployment's components, which decodes under the parameters the
deployment names for it, and `P`'s check accepts it. -/
def CaseAcceptsAt (D : Deployment) (P : BlobProtocol) (s : Sent) : Prop :=
  ∃ n ps b d k ev, Decoded s n ps b d k ev ∧
    ∃ x ∈ Properties.Composition.triples (s.componentsAt D) ps ev, x.1 = P.onBlobs (Blobs.ofArray s.blobs) ∧
      Properties.Composition.DecodedAccepted x.1 x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence

/-- `CaseAcceptsAt` at onsite. -/
def CaseAccepts (P : BlobProtocol) (s : Sent) : Prop := CaseAcceptsAt onsite P s

/-- The component that decides `Audit` `a`'s claim in the deployment accepts its own case in what the prover sent;
false where the deployment decides no such claim. -/
def ComponentAcceptsAt (D : Deployment) (a : Audit) (s : Sent) : Prop :=
  match D.comp a with
  | some P => CaseAcceptsAt D P s
  | none => False

/-- `Audit` `a`'s component accepts its own case in what the prover sent to onsite. -/
def ComponentAccepts (a : Audit) (s : Sent) : Prop := CaseAccepts (component a) s

/-- **NCI accepted the registration sampled proofs committed to**: the three byte strings decode, and what they
decode to is `Properties.Composition.RegisteredAccepted` at the deployment (its registration check NCI's, its
committing component sampled proofs' one-stage audit v2, its components on the blobs sent): NCI's check accepts the
registration's case, its digest is the one the commitment ends in and the one sampled proofs' case commits to, and
the components' cases are an accepted composition. -/
def RegistrationAcceptsAt (D : Deployment) (s : Sent) : Prop :=
  ∃ n ps b d k ev, Decoded s n ps b d k ev ∧ ∃ pub prof,
    Properties.Composition.RegisteredAccepted D.id NciCheck.Protocol.protocol
      SampledProofs.OneStageZk.protocol.id (s.componentsAt D) n ps b d k ev pub prof

/-- `RegistrationAcceptsAt` at onsite. -/
def RegistrationAccepts (s : Sent) : Prop := RegistrationAcceptsAt onsite s

/-- One release of the challenger device: the bytes the prover sent it (its request) and the coins it released after
them. -/
structure Release where
  answers : ByteArray
  coins : List Bool

/-- **The device's releases, in order**: for a round of `n` coin bits, the prover sends the bytes the release answers,
then the device draws `n` uniform bits and releases them. What the device writes and signs of them is its session's
(`PlayDevice.device`). -/
def releases : List ℕ → Game (List Release)
  | [] => .ret []
  | n :: ns => .send ByteArray fun b => .coin (Fin n → Bool) fun x =>
      (releases ns).map fun log => ⟨b, List.ofFn x⟩ :: log

/-- How a play ends: the parameters, the committed files and the commitment sent first, the device's log, and the
evidence and the blobs sent last (the files the components fetch by digest beside the committed one: v2's circuit and
public files, each session's record and its rep files, written after the coins). -/
structure Outcome where
  params : ByteArray
  file : ByteArray
  commitment : ByteArray
  log : List Release
  evidence : ByteArray
  blobs : Array ByteArray

/-- What the prover sent onsite in the play that ended with `o`: the committed file first, then the blobs sent last.
A component fetches a blob by the SHA-512 of its bytes, so a blob the prover adds grants nothing the component does not
check against its digest. -/
def Outcome.sent (o : Outcome) : Sent := ⟨o.params, o.commitment, o.evidence, #[o.file] ++ o.blobs⟩

/-- **The causal game after the parameters `θ`, the committed files `F` and the commitment `c`**: the device's rounds
`ns`, each release after the bytes it answers, then the evidence and the blobs. -/
def play (ns : List ℕ) (θ F c : ByteArray) : Game Outcome :=
  (releases ns).bind fun log => .send (ByteArray × Array ByteArray) fun eb => .ret ⟨θ, F, c, log, eb.1, eb.2⟩

/-! ## The play's device

The play's device is the challenger device (`Properties.Challenger.Device`), the one description its properties
(`RoundsByDevice`, `DrawnByDevice`) are stated over. Its fixed parameters at `θ` are a `PlayDevice`; what it writes is
defined from the play's releases: one session under the key, whose release log is its commitment's head line and, for
each release whose request it serves, the root line and the release line `challenger.py` writes, and whose coin at each
position is the bytes of the release that opened it. It signs that session's record and close, and the auditor's
ledger lines for it (`Challenger.signedBy`). -/

/-- The challenger device's fixed parameters at one `θ`: its session's id, host, coin layout, salts by position and
the times it writes on its log's lines by line, and the auditor's ledger lines for the session it ran. -/
structure PlayDevice where
  id : String
  host : String
  scheme : String
  epoch : Nat
  families : SampledProofs.Challenger.Families
  salts : Nat → ByteArray
  time : Nat → Nat
  ledger : Properties.Challenger.Session → List Lean.Json

namespace PlayDevice

open Lean (Json)
open SampledProofs.Challenger

/-- A request's family and its bytes: the label's length in UTF-8 (two bytes, big-endian), the label, then the bytes
the coin is released after. -/
def requestParts (b : ByteArray) : Option (String × ByteArray) :=
  if b.size < 2 then none else
  let n := (b.get! 0).toNat * 256 + (b.get! 1).toNat
  if b.size < 2 + n then none else
  (String.fromUTF8? (b.extract 2 (2 + n))).map fun l => (l, b.extract (2 + n) b.size)

/-- A request's family and the root its coin is released after: the SHA-512 of the bytes it carries. The prover sends
the bytes (the registration, a session's root_B digests, a round's message) and the device answers on their digest,
as the proof service's client, which holds the bytes, asks the device. -/
def requestOf (b : ByteArray) : Option (String × ByteArray) :=
  (requestParts b).map fun (l, x) => (l, Flock.Sha512.hash x)

/-- A coin's bytes from its released bits, bit `t` bit `t mod 8` of byte `⌊t/8⌋` (`Properties.Challenger.bitsOf`'s
order). -/
def bytesOfBits (l : List Bool) : ByteArray :=
  ⟨((List.range (l.length / 8)).map fun j =>
    ((List.range 8).foldl (fun (acc : Nat) i => if l.getD (8 * j + i) false then acc + 2 ^ i else acc) 0).toUInt8).toArray⟩

/-- A line in canonical JSON, as the challenger writes it. -/
def line (j : Json) : String := (Flock.canon j).toOption.getD ""

/-- The head line of the device's release log: its commitment's root, in canonical JSON as `challenger.py` writes it. -/
def headLine (root : ByteArray) : String := "{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf root ++ "\"}"

/-- The root line the device writes for coin `i` of family `l`, received at `t` after `root`. -/
def rootJ (l : String) (i t : Nat) (root : ByteArray) : Json :=
  Json.mkObj [("event", Json.str "root"), ("index", Lean.toJson i), ("label", Json.str l),
    ("received_at", Lean.toJson t), ("root", Json.str (Flock.hexOf root))]

/-- The release line the device writes for coin `i` of family `l`, released at `t` after `root`. -/
def releaseJ (l : String) (i t : Nat) (root : ByteArray) : Json :=
  Json.mkObj [("after", Json.str (Flock.hexOf root)), ("event", Json.str "release"), ("index", Lean.toJson i),
    ("label", Json.str l), ("released_at", Lean.toJson t)]

/-- The releases the device serves, in order, each with its family, its index in the family and the root it answered:
a release whose request names a family of the layout with a coin left, and whose two lines fit the reader's cap
(`MAX_LOG_LINE`; a layout's labels are short names, so the cap holds of every line the deployed device writes); the
others it serves nothing for. Its `k`th served release writes its root line at time `2k+1` and its release at `2k+2`. -/
def served (D : PlayDevice) (rs : List Release) : List (String × Nat × ByteArray × Release) :=
  (rs.foldl (fun (acc : List (String × Nat × ByteArray × Release)) r =>
    match requestOf r.answers with
    | none => acc
    | some (l, root) =>
      let i := (acc.filter (·.1 == l)).length
      let k := acc.length
      if (D.families.find? (·.1 == l)).any (fun f => decide (i < f.2)) &&
          decide ((line (rootJ l i (D.time (2 * k + 1)) root)).length ≤ MAX_LOG_LINE) &&
          decide ((line (releaseJ l i (D.time (2 * k + 2)) root)).length ≤ MAX_LOG_LINE)
      then acc ++ [(l, i, root, r)] else acc) [])

/-- The layout's coin positions, as the commitment numbers them. -/
def layout (D : PlayDevice) : LiveCommitment := ⟨D.scheme, D.epoch, D.families, .empty⟩

/-- The device's session before its log: its coin at each position the bytes of the release served there. -/
def base (D : PlayDevice) (rs : List Release) : Properties.Challenger.Session where
  id := D.id
  host := D.host
  scheme := D.scheme
  epoch := D.epoch
  families := D.families
  coins p := ((D.served rs).find? fun (l, i, _, _) => D.layout.position l i == some p).elim .empty
    fun (_, _, _, r) => bytesOfBits r.coins
  salts := D.salts
  log := []
  ended := none

/-- The release log's lines after its head: for each release served, the root line, then the release line. -/
def lines (D : PlayDevice) (rs : List Release) : List String :=
  ((D.served rs).zipIdx.map fun ((l, i, root, _), k) =>
    [line (rootJ l i (D.time (2 * k + 1)) root), line (releaseJ l i (D.time (2 * k + 2)) root)]).flatten

/-- **The device's session in the play**: its release log is its commitment's head line, then `lines`; its client
closed it. -/
def session (D : PlayDevice) (rs : List Release) : Properties.Challenger.Session :=
  let s := D.base rs
  { s with log := headLine ((s.commitment.map (·.root)).getD .empty) :: D.lines rs }

/-- **The play's device under the key `pk`**: the one session it ran, and the auditor's ledger lines for it. -/
def device (D : PlayDevice) (pk : ByteArray) (rs : List Release) : Properties.Challenger.Device where
  sessions k := if k == pk then [D.session rs] else []
  ledger k := if k == pk then D.ledger (D.session rs) else []

end PlayDevice

/-- **The device's registered LMS key at `θ`**: the challenger key v2's compiled check reads from the registered
components' parameters (`Properties.Challenger.keyOfParams`, on the components `Composition.decodeRegisteredParams`
decodes from `θ`), as LMS's parameters and public key (`Properties.Challenger.lmsKey`). The compiled check's LMS verify
is `LmsVerifies` at it wherever v2 reads a key (`Properties.Challenger.verifies_registered`). -/
def registeredKey (θ : ByteArray) : Lms.Params × ByteArray :=
  Properties.Challenger.lmsKey ((canonicalJson θ >>= Composition.decodeRegisteredParams).toOption.bind
    fun r => Properties.Challenger.keyOfParams r.2)

/-- SHA-512, the compiled one (`Flock.Sha512.hash`), as a 64-byte digest. -/
def sha512 (m : Hash.Msg) : Hash.Digest := fun i => (Flock.Sha512.hash ⟨m.toArray⟩).data.getD i.val 0

/-- SHA-512, the compiled one, as the message-to-message hash `Claims.Finder.Pr` and `ecr/sha-512`
(`Claims.ExpectedCR`) read. -/
def sha512Msg (m : Claims.Msg) : Claims.Msg := (Flock.Sha512.hash ⟨m.toArray⟩).data.toList

/-- **How v2 reads the committed blob**: its statement's tags (derived from the circuit file), the circuit file, its
lookup tables and the public file. -/
structure FileRead where
  tags : Flock.Tags
  circuitFile : ByteArray
  tables : List (String × ByteArray)
  publicFile : ByteArray

namespace FileRead

/-- The committed files as core's circuit, as the verifier reads them (`Flock.Bridge.circuitOfFile`). -/
noncomputable def circuit (r : FileRead) : Verity.Circuit :=
  Flock.Bridge.circuitOfFile r.tags r.circuitFile r.tables r.publicFile

/-- The committed files' number of units, the public file's population. -/
noncomputable def units (r : FileRead) : ℕ := Flock.Bridge.unitsOfFile r.tags r.circuitFile r.tables r.publicFile

/-- The committed files' units, as core's partition of `circuit`. -/
noncomputable def parts (r : FileRead) : Verity.Partition r.circuit r.units :=
  Flock.Bridge.partsOfFile r.tags r.circuitFile r.tables r.publicFile

end FileRead

/-- **A parse's constants**: the key program's constant input gate holds `1` and its zero gate `0`
(`FlockSoundness.Discharge.Placed.oneGate`, `zeroGate`), public circuit structure that no proof unit opens. -/
noncomputable def parsedConsts : (o : Option Flock.Bridge.Parsed) → Fin (Flock.Bridge.Parsed.circuit o).N → Option Bool
  | some r, g =>
    if g.val = (FlockSoundness.Discharge.Placed.oneGate r.n (FlockSoundness.Discharge.Layout.keyB r.D.unroll r.hU)).val
    then some true
    else if g.val =
        (FlockSoundness.Discharge.Placed.zeroGate r.n (FlockSoundness.Discharge.Layout.keyB r.D.unroll r.hU)).val
    then some false
    else none
  | none, _ => none

/-- The committed files' constants. -/
noncomputable def FileRead.consts (r : FileRead) : Fin r.circuit.N → Option Bool :=
  parsedConsts (Flock.Bridge.parsedOf r.tags r.circuitFile r.tables r.publicFile)

/-- A value's registered position: the row-leaf verifier (the frame-v3 domain, the hm96 key and the row format), the
commitment's root it opens against, the leaf's rank, and the value's first bit in the row. -/
structure Place where
  verifier : Hash.RowLeaf.Verifier
  root : Hash.Digest
  rank : ℕ
  offset : ℕ

/-- The fields of a period's run that the prover's world realizes in a play: whose account each resource is charged to,
the window, the device schedule, and what the network certifiers saw (`Accounting.PeriodWorld`, the one definition). -/
abbrev World {ι : Type} (d : Device ι) (α : Type) [Fintype α] := Accounting.PeriodWorld d α

/-- **The names the subprotocols define.** v2's reading of the committed blob; the positions from the files and the
commitment, the circuit file's digest in the parameters and the registered blob; the device's rounds, the device's
fixed parameters, and the releases each component's check read, decoded from the device's signed
log in what the prover sent; the extractor applied to a strategy, and each subprotocol's collision finder, with the
finder's SHA-512 calls bounded by the strategy's (`finder_calls`); each subprotocol's LMS forger on the play's outcome;
the committed circuit's gates a play computed after their window's coin; the world each play realizes; the deployment's
honest prover; and the certificates, errors, collision bounds and losses the product publishes, from the parameters and
the strategy's SHA-512 calls alone.

`cost θ F c σ` is the strategy's SHA-512 calls in one run (sampled proofs' `t' τ`): `Verity.Game` has no call model of
strategies, so it is declared, as `Claims.Finder.cost` is. -/
structure Vocab {ι : Type} (d : Device ι) (α : Type) [Fintype α] where
  read : ByteArray → FileRead
  place : (θ F c : ByteArray) → Fin (read F).circuit.N → Option Place
  programDigest : ByteArray → Option Hash.Digest
  registered : ByteArray → ByteArray
  rounds : ByteArray → List ℕ
  /-- The challenger device's fixed parameters at `θ` (`PlayDevice`): what it writes and signs is defined from the
  play's releases (`PlayDevice.device`). -/
  device : ByteArray → PlayDevice
  coinsRead : Audit → Sent → List Release
  extract : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → (read F).circuit.Transcript
  finder : Audit → (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Claims.Finder
  /-- Subprotocol `a`'s LMS forger at `θ`, a fixed function of the play's outcome (whose `log` the play's device
  writes and signs, `Vocab.deviceOf`). What it must output to succeed is defined, `Vocab.forgery`. -/
  forger : Audit → ByteArray → Outcome → Option (ByteArray × ByteArray)
  cost : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → ℕ
  callsMul : ByteArray → Audit → ℕ
  callsAdd : ByteArray → Audit → ℕ
  /-- **The finder's SHA-512 calls are at most `c · calls(τ) + d`**, on every coin of the finder. -/
  finder_calls : ∀ a θ F c σ (x : (finder a θ F c σ).S),
    (finder a θ F c σ).cost x ≤ callsMul θ a * cost θ F c σ + callsAdd θ a
  computed : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Outcome →
    Finset (Fin (read F).circuit.N)
  world : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Outcome → World d α
  honest : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c)
  certOf : ByteArray → Certificates
  crBound : ByteArray → Audit → ℕ → ℝ
  delta : ByteArray → Audit → ℕ → ℝ≥0∞
  loss : Audit → ByteArray → ℝ≥0∞ → ℝ≥0∞

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- The game after `θ`, `F` and `c`. -/
def game (θ F c : ByteArray) : Game Outcome := play (V.rounds θ) θ F c

/-- The committed files' circuit. -/
noncomputable def circuitOf (F : ByteArray) : Verity.Circuit := (V.read F).circuit

/-- The committed files' number of units. -/
noncomputable def units (F : ByteArray) : ℕ := (V.read F).units

/-- The committed files' units. -/
noncomputable def partsOf (F : ByteArray) : Verity.Partition (V.circuitOf F) (V.units F) := (V.read F).parts

/-- **The commit phase's file check**: the committed blob's SHA-512 is the digest the parameters hold. -/
def Registered (θ F : ByteArray) : Prop := V.programDigest θ = some (sha512 F.data.toList)

/-- **The commitment `c` opens gate `g` to `v`**: a registered constant to its constant; any other gate by an opening
of the row leaf at its registered position that the row-leaf verifier accepts under SHA-512, whose row holds `v`'s bits
from the position's offset. -/
def Opens (θ F c : ByteArray) (g : Fin (V.circuitOf F).N) (v : BitVec ((V.circuitOf F).w g)) : Prop :=
  match (V.read F).consts g with
  | some b => ∀ i : Fin ((V.circuitOf F).w g), v.getLsbD i.val = b
  | none => ∃ p, V.place θ F c g = some p ∧ ∃ o : Hash.RowLeaf.Opening,
      Hash.RowLeaf.accepts sha512 sha512 p.verifier p.root p.rank o ∧
        ∀ i : Fin ((V.circuitOf F).w g), o.row[p.offset + i.val]? = some (v.getLsbD i.val)

/-- **Subprotocol `a`'s verdict**: the committed blob passed the commit phase's check, NCI accepted the registration
sampled proofs committed to, and `a`'s compiled component accepts its own case in what the prover sent. -/
def Verdict (a : Audit) (o : Outcome) : Prop :=
  V.Registered o.params o.file ∧ RegistrationAccepts o.sent ∧ ComponentAccepts a o.sent

/-- **Subprotocol `a`'s verdict in the deployment `D`**: `Verdict` with `D`'s registration and `D`'s component for
`a`; false where `D` decides no claim of `a`'s. -/
def VerdictAt (D : Deployment) (a : Audit) (o : Outcome) : Prop :=
  V.Registered o.params o.file ∧ RegistrationAcceptsAt D o.sent ∧ ComponentAcceptsAt D a o.sent

/-- **The period after `θ`, `F` and `c` in the deployment `D`, as the one vocabulary's `Accounting.PeriodLive`**: the
committed files' units, the commitment's openings, the extractor, the finders, the gates computed, the world and `D`'s
verdicts. -/
noncomputable def liveAt (D : Deployment) (θ F c : ByteArray) : PeriodLive d α (V.circuitOf F) (V.game θ F c) where
  units := V.units F
  parts := V.partsOf F
  opens := V.Opens θ F c
  extract σ _ := V.extract θ F c σ
  finder a σ := V.finder a θ F c σ
  computed := V.computed θ F c
  world := V.world θ F c
  accepts := V.VerdictAt D

/-- **The period after `θ`, `F` and `c`**, at onsite. -/
noncomputable def live (θ F c : ByteArray) : PeriodLive d α (V.circuitOf F) (V.game θ F c) := V.liveAt onsite θ F c

/-- **The claim subprotocol `a` certifies** in the play of `σ` that ended with `o`: `Accounting.PeriodLive.Holds` at the
period and the published certificates. Sampled proofs' on the extractor's transcript at `σ` (at most an `ε` share of
the units unbound in it under the commitment's openings), PoUW's in gates (at least `W` gates of the committed circuit
computed after their window's coin), and the others' on the world the play realized (at least `S` bits of device memory
held on average, at least `Q` bits read, at most `K` bits through the network certifiers). -/
def Holds (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : Audit → Outcome → Prop :=
  (V.live θ F c).Holds (V.certOf θ) σ

/-- Every release `a`'s check read is in the device's log; outside it, the release is a forgery of the device's LMS
signature (`forgery`), which the forger outputs. -/
def Logged (a : Audit) (o : Outcome) : Prop := ∀ r ∈ V.coinsRead a o.sent, r ∈ o.log

/-- **The finder for a second file with the registered digest**: on its one coin it hashes the committed blob and the
registered one, and outputs them when they differ, a SHA-512 collision when the blob passed the file check. -/
def fileFinder (θ F : ByteArray) : Claims.Finder where
  S := Unit
  out _ := if F = V.registered θ then none else some (F.data.toList, (V.registered θ).data.toList)
  cost _ := 2

/-- **`B^τ`'s collision premise, quantitative**: subprotocol `a`'s finder applied to the strategy outputs a SHA-512
collision with probability at most the published bound at the strategy's SHA-512 calls (`ε_cr`). `ecr/sha-512` is asked
at the finder's budget, `callsMul θ a * cost θ F c σ + callsAdd θ a` calls (`finder_calls`), a defined function of the
strategy's `cost` as sampled proofs' `qF`/`qS` are of `t'`. -/
def FinderCR (a : Audit) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : Prop :=
  (V.finder a θ F c σ).Pr sha512Msg ≤ V.crBound θ a (V.cost θ F c σ)

/-- **The play's device at `θ`**, under its registered key, after the releases of the play that ended with `o`. -/
def deviceOf (θ : ByteArray) (o : Outcome) : Properties.Challenger.Device :=
  (V.device θ).device (registeredKey θ).2 o.log

/-- **The device signed `m`** in the play that ended with `o`: `m` is its session's record or close message, or one of
the auditor's ledger lines for it, under its registered key (`Challenger.Signed` of `deviceOf`). -/
def Signed (θ : ByteArray) (o : Outcome) (m : ByteArray) : Prop :=
  Properties.Challenger.Signed (V.deviceOf θ o) (registeredKey θ).2 m

/-- **`sig` is a forgery on `m` at `θ`** in the play that ended with `o`: it verifies under the device's registered
LMS key at `θ` (the compiled `Lms.verify`, through `MemoryChallenger.Properties.LmsVerifies`), and the device never
signed `m` in the play (`Signed`). The challenger's `Witness.Valid` at `.forgery` on the play's device
(`deviceOf`), and the memory challenger's `WipeBreak` on the game's one device and key. It is defined, never a field, so
no vocabulary can empty the forger's term. -/
def forgery (θ : ByteArray) (o : Outcome) (m sig : ByteArray) : Prop :=
  MemoryChallenger.Properties.LmsVerifies (registeredKey θ).1 (registeredKey θ).2 m sig ∧ ¬ V.Signed θ o m

/-- **Subprotocol `a`'s forger succeeds** on the outcome `o`: it outputs a message and a signature that are a forgery
at `θ` and `o`. -/
def Forges (a : Audit) (θ : ByteArray) (o : Outcome) : Prop :=
  ∃ m sig, V.forger a θ o = some (m, sig) ∧ V.forgery θ o m sig

/-- Pr[the LMS forger succeeds] in the play: `euf-cma/lms-sha512-n64`'s term. -/
noncomputable def forges (a : Audit) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : ℝ≥0∞ :=
  Game.prob (V.Forges a θ) (V.game θ F c) σ

/-- **Subprotocol `a`'s error at a strategy**: where its finder meets its collision premise (`FinderCR`), its
period's error (`Accounting.PeriodLive.error`: its published `δ` at the strategy's SHA-512 calls plus its loss at the
probability that the finder outputs a collision) plus the probability that its LMS forger succeeds in the play
(`forges`); elsewhere no bound (`⊤`), as `Accounting.condError` reads it. The two breaks are separate terms, each of
its own fixed finder: `ecr/sha-512` is asked of the collision finder and `euf-cma/lms-sha512-n64` of the forger,
outside the theorem. C-Flock's conditional bound is the case `loss = 0`, the additive one the case `crBound = 1`. -/
noncomputable def error (a : Audit) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : ℝ≥0∞ :=
  condError (V.FinderCR a θ F c)
    (fun σ => (V.live θ F c).error sha512Msg a (V.delta θ a (V.cost θ F c σ)) (V.loss a θ) σ + V.forges a θ F c σ) σ

/-- **Soundness for the acceptance `acc`, in the causal game.** At all parameters `θ`, committed files `F` and
commitments `c`, and for every strategy, the probability that `acc` holds of what the prover sent for a blob that passed
the commit phase's check while one of the period's claims fails (more than an `ε` share of the units unbound in the
extractor's transcript, fewer than `W` gates computed after their coin, fewer than `S` bits held on average, fewer than
`Q` bits read, or more than `K` bits through the network certifiers) is at most the subprotocols' errors at that
strategy summed. -/
def SoundnessFor (acc : Sent → Prop) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => acc o.sent ∧ V.Registered o.params o.file ∧ ∃ a, ¬ V.Holds θ F c σ a o) (V.game θ F c) σ ≤
      ∑ a, V.error a θ F c σ

/-- **Soundness at onsite's compiled verifier** (`Accepts`). -/
def Soundness : Prop := V.SoundnessFor Accepts

/-- **Soundness in the deployment `D` for the acceptance `acc`**: `SoundnessFor` over the claims `D` decides, with
their errors summed (`Deployment.restrict`). -/
def SoundnessAtFor (D : Deployment) (acc : Sent → Prop) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => acc o.sent ∧ V.Registered o.params o.file ∧ ∃ a, D.Decides a ∧ ¬ V.Holds θ F c σ a o)
        (V.game θ F c) σ ≤
      ∑ a, D.restrict (fun b => V.error b θ F c σ) a

/-- **Soundness at the deployment's compiled verifier** (`AcceptsAt D`). -/
def SoundnessAt (D : Deployment) : Prop := V.SoundnessAtFor D (AcceptsAt D)

/-- **Subprotocol `a`'s target in the deployment `D`**: at all `θ`, `F` and `c`, its per-period guarantee over `D`'s
period (`Accounting.PeriodLive.Sound` at `liveAt D`): for every strategy, the probability that `a`'s verdict in `D`
accepts while its claim fails is at most its error at that strategy. -/
def TargetAt (D : Deployment) (a : Audit) : Prop :=
  ∀ θ F c : ByteArray, (V.liveAt D θ F c).Sound (V.certOf θ) a (V.error a θ F c)

/-- **Subprotocol `a`'s target**, at onsite. -/
def Target (a : Audit) : Prop := V.TargetAt onsite a

/-- **Sampled proofs' target**: at most an `ε` share of the units unbound in the extractor's transcript. -/
def SampledProofsSound : Prop := V.Target .sampled

/-- **PoUW's target**: at least `W` gates of the committed circuit computed after their window's coin. -/
def PouwSound : Prop := V.Target .pouw

/-- **PoUS's target**: at least `S` bits of device memory held on average. -/
def PousSound : Prop := V.Target .pous

/-- **The bandwidth audit's target**: at least `Q` bits read from device memory. -/
def BandwidthAuditSound : Prop := V.Target .poub

/-- **The network certifier's target**: at most `K` bits through the network certifiers, where NCI accepted
the registration sampled proofs committed to. -/
def NetworkCertifierSound : Prop := V.Target .network

/-- **Subprotocol `a`'s honest lemma at the deployment**: at registered files `θ`, `F` and the commitment the honest
prover sends from them (`commit θ F`), the deployment's honest prover meets `a`'s claim on almost every play. The
commitment is the honest one, never an arbitrary `c`: at a commitment that opens nothing, sampled proofs' claim fails
on every play. Each subprotocol's piece proves its own (sampled proofs from `Flock.Bridge.card_wrong_lt_of_relS_honest`;
the others from their owners' `_honest` lemmas). -/
def HonestHolds (commit : ByteArray → ByteArray → ByteArray) (a : Audit) : Prop :=
  ∀ θ F : ByteArray, V.Registered θ F →
    Game.prob (fun o => ¬ V.Holds θ F (commit θ F) (V.honest θ F (commit θ F)) a o) (V.game θ F (commit θ F))
      (V.honest θ F (commit θ F)) = 0

end Vocab

end Verity
