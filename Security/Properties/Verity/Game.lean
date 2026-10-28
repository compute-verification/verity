import CoreComposites
import Properties.Composition
import Properties.Flock.Circuit.CircuitOf
import Definitions.Accounting.JointLive
import Definitions.Core.Blocks
import Definitions.Core.Game
import Definitions.Core.Site
import Definitions.Hash.RowLeaf
import Definitions.Nci
import Flock.Hash

/-!
# Soundness as a causal game at a composite's compiled verifier, and each subprotocol's target

The verifier of record is one compiled Lean function per composite (`Verity.Registration.Composite`), the `answer` of
the composite `Core.Composites.composites` lists for it (`verifier`): onsite's `Core.Onsite.protocol`
(`verity/onsite/v0`, `Core.Composition.tiedSameB`) and remote's `Core.Remote.protocol` (`verity/remote/v0`,
`Core.Composition.sameB`), each composing its subprotocols over the blobs the prover sent. Onsite has seven
subprotocols and remote three of them, each subprotocol at the component its composite names for it (`component`);
the five that decide a period are one per `Accounting.Audit` (`Subprotocol.ofAudit`):

| subprotocol | `Audit` | onsite's component | remote's component |
| --- | --- | --- | --- |
| sampled proofs | `.sampled` | `SampledProofs.OneStageZk.protocol` (v2) | the same |
| PoUW | `.pouw` | `PouwCheck.Period.protocol` (its period credit) | `PouwCheck.protocol` (its window) |
| the inference-only policy | | `NciCheck.Protocol.protocol` (`verity/inference-only/v0`) | the same |
| PoUS | `.pous` | `PousCheck.RunProtocol.protocol` | |
| PoUB-P2 | `.poub` | `RemoteAudit.P2Protocol.protocol` (`verity/remote-bandwidth/p2/v0`) | |
| the network certifier | `.network` | `NetworkCheck.Period.protocol` | |
| the wipe | | `MemoryChallenger.WipeProtocol.protocol` (`verity/wipe/v0`) | |

The statement's form is the causal game, the committed files, one fixed extractor and one adversary form:

* **A causal game.** The prover sends the parameters `θ`, the committed files `F` (one blob) and the
  commitment `c` first, before any coin, so every statement is `∀ θ F c`, and then plays `play (rounds θ) θ F c`, a
  `Verity.Game`: in each of the device's rounds the prover sends the bytes (a root, a transcript digest) that the
  round's release answers, and then the device draws and releases that round's coins (`releases`); the evidence comes
  last. A strategy (`Verity.Game.Strategy`) is a function of the coins released so far and never of later ones.
* **The claims read the committed files.** The circuit, its units and its constants are what the verifier reads
  from the files (`Flock.Bridge.circuitOfFile`, `unitsOfFile`, `partsOfFile` at v2's reading of the blob, `read`), and
  the positions are the files' and the commitment's (`place`), never the prover's world. The params hold the circuit
  file's digest (`programDigest`): a file whose SHA-512 isn't it is refused (`Registered`, conjoined beside the
  composite's acceptance). The committed file is the blob the composite reads (`Exchange.sent`).
* **Sampled proofs' claim reads one fixed extractor's transcript.** The extractor (`extract`, the one
  `SampledKSLive*` is stated with, `XplurZC`) is a construction applied to the strategy, never chosen per strategy.
  The claim is that at most an `ε` share of the file's units are unbound in its transcript (`Accounting.unbound`): wrong
  in it, or with a committed input or output the commitment doesn't open to its value there (`Opens`: a registered
  constant to its constant, any other wire by an opening of its row leaf that `Hash.RowLeaf.accepts` passes under the
  compiled SHA-512).
* **One adversary form: every prover, and collisions a fixed finder outputs.** There is no strategy class and no break
  in the event. The finder `B` (`finder a`, a `Claims.Finder`) is a fixed construction applied to the strategy whose
  SHA-512 calls are at most `c · calls(τ) + d` (`finder_calls`). Subprotocol `a`'s error at a strategy (`error`) is,
  where `B^τ` outputs a SHA-512 collision with probability at most the published `ε_cr` at the strategy's calls
  (`FinderCR`, quantitative), its published `δ` at those calls plus its loss at `Pr[B^τ]`, the form of
  `Accounting.PeriodLive.error`; elsewhere `⊤`. That admits C-Flock's conditional bound (`loss = 0`) and the additive
  one (`crBound = 1`). Two openings of one position, a forged device release (LMS is hash-based) and a second file with
  the registered digest (`fileFinder`) each enter that way. The named assumption `ecr/sha-512` is asked only of the
  finder.
* **The one vocabulary.** The period is `Accounting.PeriodLive` (`live`), and the five period claims are its `Holds`.
  The inference-only policy's claim is `Nci.Complies` at the published `Γ` and `α` of the declaration over the committed
  circuit (`declaration`: its registered inputs `θ`, `theta`). The wipe's claim is that the run, read at the wipe's
  boundary as the wipe's game (`Verity.Wipe.OfRun`, `wipeRun`), doesn't keep foreign content past the survivor threshold
  the wipe publishes (`Verity.Wipe.OfRun.Kept` at `foreignBits`).
* **Each component's acceptance in what the prover sent.** Subprotocol `a`'s verdict at composite `k` (`Verdict`) is the
  file check, the composite's own acceptance as a `same` composition (`SameAccepts`: an accepted composition of its
  components whose record subprotocols committed to one record, `Properties.Composition.AcceptSame`'s conclusion, and
  at onsite PoUS and PoUB-P2 on one store, `AcceptTied`'s), and `a`'s compiled component's acceptance of its own case.
  That the releases each component's check read (`coinsRead`) are the device's at the rounds the parameters assign them
  is the targets' to bound (`Logged`): membership alone would let a prover keep the better of two rounds' coins.
* **PoUW counts gates** of the committed circuit the play computed after their window's coin, over the period's windows
  (`computed`), with no pricing.

Each subprotocol's target (`SampledProofsSound`, `PouwSound`, `InferenceOnlySound`, `PousSound`, `BandwidthAuditSound`,
`NetworkCertifierSound`, `WipeSound`) is `Target k a`: at every `θ`, `F`, `c` and strategy, the probability that `a`'s
verdict at `k` accepts while its claim fails is at most its error there. `Verity.Soundness` and `Verity.RemoteSoundness`
(`Properties.Verity.Soundness`) compose them, onsite's seven and remote's three.

The names a subprotocol defines are `Vocab`'s fields: `read` (v2's reading of the blob), `place`, `programDigest` and
`registered`, `rounds` and `coinsRead`, `extract` and the finders, `computed`, `world` (the runs PoUS, PoUB-P2 and the
network certifier realize), the declaration's registered inputs and the policy's budgets, the wipe at the period's
boundary and the run read as its game, and `certOf`, `foreignBits`, `delta` and `loss` (what the composite publishes,
functions of the parameters alone).
-/

namespace Verity

open Accounting Core
open scoped ENNReal

/-- The subprotocols a composite is made of. -/
inductive Subprotocol
  /-- sampled proofs: the integrity of the circuit's committed transcript -/
  | sampled
  /-- PoUW: work -/
  | pouw
  /-- the inference-only policy: the declared circuit divides into inference blocks within their budgets -/
  | inferenceOnly
  /-- PoUS: space -/
  | pous
  /-- PoUB-P2, the remote bandwidth audit's P2 session on PoUS's store: bits read from device memory -/
  | poub
  /-- the network certifier: communication through the network certifiers -/
  | network
  /-- the wipe: the memory challenger's streamed-label wipe that opens the period -/
  | wipe
  deriving DecidableEq

instance instFintypeSubprotocol : Fintype Subprotocol where
  elems := {.sampled, .pouw, .inferenceOnly, .pous, .poub, .network, .wipe}
  complete a := by cases a <;> simp

/-- The subprotocol that decides `Audit` `a`. -/
def Subprotocol.ofAudit : Audit → Subprotocol
  | .sampled => .sampled
  | .pouw => .pouw
  | .pous => .pous
  | .poub => .poub
  | .network => .network

/-- The compiled verifier of record of composite `k`. -/
def verifier : Verity.Registration.Composite → BlobProtocol
  | .onsite => Onsite.protocol
  | .remote => Remote.protocol

/-- Composite `k`'s subprotocols, as its verifier composes them, in the order of their ids. -/
def componentsOf : Verity.Registration.Composite → List BlobProtocol
  | .onsite => Onsite.components
  | .remote => Remote.components

/-- The subprotocols of composite `k` that must commit to one record. -/
def tied : Verity.Registration.Composite → List String
  | .onsite => Onsite.tied
  | .remote => Remote.components.map (·.id)

/-- The ties of composite `k`: at onsite, PoUS and PoUB-P2 on one store. -/
def ties : Verity.Registration.Composite → List Composition.Tie
  | .onsite => [Onsite.store]
  | .remote => []

/-- **The compiled component of composite `k` that decides subprotocol `a`**; `none` when `k` has no such
subprotocol. -/
def component : Verity.Registration.Composite → Subprotocol → Option BlobProtocol
  | .onsite, .sampled => some SampledProofs.OneStageZk.protocol
  | .onsite, .pouw => some PouwCheck.Period.protocol
  | .onsite, .inferenceOnly => some NciCheck.Protocol.protocol
  | .onsite, .pous => some PousCheck.RunProtocol.protocol
  | .onsite, .poub => some RemoteAudit.P2Protocol.protocol
  | .onsite, .network => some NetworkCheck.Period.protocol
  | .onsite, .wipe => some MemoryChallenger.WipeProtocol.protocol
  | .remote, .sampled => some SampledProofs.OneStageZk.protocol
  | .remote, .pouw => some PouwCheck.protocol
  | .remote, .inferenceOnly => some NciCheck.Protocol.protocol
  | .remote, _ => none

/-- Composite `k`'s subprotocols: those it has a component for. Remote's are a subset of onsite's. -/
def subprotocols (k : Verity.Registration.Composite) : Finset Subprotocol :=
  Finset.univ.filter fun a => (component k a).isSome

/-- What the prover sends a composite's verifier of record: the parameters, the commitment and the evidence, as bytes,
and the blobs its components read. -/
structure Sent where
  params : ByteArray
  commitment : ByteArray
  evidence : ByteArray
  blobs : Array ByteArray

/-- Composite `k`'s components on the blobs the prover sent. -/
def Sent.components (k : Verity.Registration.Composite) (s : Sent) : List Protocol :=
  (componentsOf k).map (·.onBlobs (Blobs.ofArray s.blobs))

/-- **Composite `k`'s verifier of record accepts what the prover sent.** -/
def Accepts (k : Verity.Registration.Composite) (s : Sent) : Prop :=
  ((verifier k).answer s.params s.commitment s.evidence s.blobs).verdict = .accept

/-- The three byte strings decode under the composition's formats (`Core.Composition.product`'s, which both composites
read): the components' parameters `ps`, the binding `c` of their digests, and their cases `ev`. -/
def Decoded (s : Sent) (ps : List Composition.Named) (c : ByteArray) (ev : List Composition.Case) : Prop :=
  (canonicalJson s.params >>= Composition.decodeParams) = .ok ps ∧ Composition.decodeCommitment s.commitment = .ok c ∧
    Composition.decodeEvidence s.evidence = .ok ev

/-- The protocol `P` accepts its own case in what the prover sent composite `k`: the three byte strings decode, and
one of the components' cases is `P`'s, as one of `k`'s components, which decodes under the parameters `k` names for
it, and `P`'s check accepts it. -/
def CaseAccepts (k : Verity.Registration.Composite) (P : BlobProtocol) (s : Sent) : Prop :=
  ∃ ps c ev, Decoded s ps c ev ∧
    ∃ x ∈ Properties.Composition.triples (s.components k) ps ev, x.1 = P.onBlobs (Blobs.ofArray s.blobs) ∧
      Properties.Composition.DecodedAccepted x.1 x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence

/-- Subprotocol `a`'s component at composite `k` accepts its own case in what the prover sent. -/
def ComponentAccepts (k : Verity.Registration.Composite) (a : Subprotocol) (s : Sent) : Prop :=
  ∃ P, component k a = some P ∧ CaseAccepts k P s

/-- **Composite `k` accepted what the prover sent as a `same` composition**: the three byte strings decode to an
accepted composition of `k`'s components on the blobs sent, whose record subprotocols (`tied k`) committed to one
record (`Properties.Composition.SameAccepted`), and whose ties (`ties k`: at onsite, PoUS and PoUB-P2 on one store)
each named one value (`Properties.Composition.Tied`). -/
def SameAccepts (k : Verity.Registration.Composite) (s : Sent) : Prop :=
  ∃ ps c ev, Decoded s ps c ev ∧ ∃ pub prof,
    Properties.Composition.SameAccepted (verifier k).id (tied k) (s.components k) ps c ev pub prof ∧
      ∀ t ∈ ties k, Properties.Composition.Tied t ev

/-- One release of the challenger device: the bytes it answers, and the coins it released after them. -/
structure Release where
  answers : ByteArray
  coins : List Bool

/-- **The device's releases, in order**: for a round of `n` coin bits, the prover sends the bytes the release answers,
then the device draws `n` uniform bits and releases them. The output is the device's log. -/
def releases : List ℕ → Game (List Release)
  | [] => .ret []
  | n :: ns => .send ByteArray fun b => .coin (Fin n → Bool) fun x =>
      (releases ns).map fun log => ⟨b, List.ofFn x⟩ :: log

/-- How a play ends: the parameters, the committed files and the commitment sent first, the device's log, and the
evidence sent last. -/
structure Exchange where
  params : ByteArray
  file : ByteArray
  commitment : ByteArray
  log : List Release
  evidence : ByteArray

/-- What the prover sent the composite in the play that ended with `o`; the blob its components read is the committed
file. -/
def Exchange.sent (o : Exchange) : Sent := ⟨o.params, o.commitment, o.evidence, #[o.file]⟩

/-- **The causal game after the parameters `θ`, the committed files `F` and the commitment `c`**: the device's rounds
`ns`, each release after the bytes it answers, then the evidence. -/
def play (ns : List ℕ) (θ F c : ByteArray) : Game Exchange :=
  (releases ns).bind fun log => .send ByteArray fun e => .ret ⟨θ, F, c, log, e⟩

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

/-- **The names the subprotocols define.** v2's reading of the committed blob; the positions from the files and
the commitment, the circuit file's digest in the parameters and the registered blob; the device's rounds, and the
releases each component's check read, decoded from the device's signed log entries in what the prover sent; the
extractor applied to a strategy, and each subprotocol's collision finder, with the finder's SHA-512 calls bounded by the
strategy's (`finder_calls`); the committed circuit's gates a play computed after their window's coin; the world each
play realizes; the declaration's registered inputs and the inference-only policy's budgets; the wipe that opens the
period, at a boundary of a schedule of the committed circuit's blocks, and each play read at that boundary as the wipe's
game; the deployment's honest prover; and the certificates, survivor threshold, errors, collision bounds and losses the
composite publishes, from the parameters and the strategy's SHA-512 calls alone.

`cost θ F c σ` is the strategy's SHA-512 calls in one run (`t' τ`): `Verity.Game` has no call model of
strategies, so it is declared, as `Claims.Finder.cost` is. -/
structure Vocab {ι : Type} (d : Device ι) (α : Type) [Fintype α] where
  read : ByteArray → FileRead
  place : (θ F c : ByteArray) → Fin (read F).circuit.N → Option Place
  programDigest : ByteArray → Option Hash.Digest
  registered : ByteArray → ByteArray
  rounds : ByteArray → List ℕ
  coinsRead : Subprotocol → Sent → List Release
  extract : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → (read F).circuit.Transcript
  finder : Subprotocol → (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Claims.Finder
  cost : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → ℕ
  callsMul : ByteArray → Subprotocol → ℕ
  callsAdd : ByteArray → Subprotocol → ℕ
  /-- **The finder's SHA-512 calls are at most `c · calls(τ) + d`**, on every coin of the finder. -/
  finder_calls : ∀ a θ F c σ (x : (finder a θ F c σ).S),
    (finder a θ F c σ).cost x ≤ callsMul θ a * cost θ F c σ + callsAdd θ a
  computed : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Exchange → Finset (Fin (read F).circuit.N)
  world : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Exchange → World d α
  /-- the window PoUS's and PoUB-P2's claims read, fixed by the parameters before any coin -/
  window : ByteArray → Accounting.Window
  /-- the account PoUS's claim charges device memory to, fixed by the parameters before any coin -/
  capacity : ByteArray → α
  /-- every play's world is read over the parameters' window, never one PoUS's own coins chose -/
  world_window : ∀ θ F c σ o, (world θ F c σ o).win = window θ
  /-- every play's world charges device memory to the parameters' account, never one PoUS's own coins chose -/
  world_capacity : ∀ θ F c σ o, (world θ F c σ o).acct Accounting.Resource.capacity = capacity θ
  /-- the declaration's registered inputs `θ`, input gates of the committed circuit -/
  theta : (θ F : ByteArray) → Finset (Fin (read F).circuit.N)
  theta_input : ∀ θ F, ∀ g ∈ theta θ F, (read F).circuit.isInput g = true
  /-- the inference-only policy's maximum inference block size `Γ`, in gates -/
  gates : ByteArray → ℕ
  /-- the inference-only policy's limit `α` on fresh input, in bits per gate -/
  fresh : ByteArray → ℝ
  /-- the number of blocks of the committed circuit's schedule -/
  spans : ByteArray → ℕ
  /-- the schedule of the committed circuit's blocks the wipe sits in -/
  schedule : (F : ByteArray) → Verity.Blocks (read F).circuit (spans F)
  /-- the boundary the wipe opens -/
  boundary : (F : ByteArray) → Fin (spans F)
  /-- the wipe's seeds -/
  Seed : Type
  /-- the wipe's spot reads -/
  Spots : Type
  /-- the wipe that opens the period, on the device -/
  wipe : (θ F : ByteArray) → Verity.Wipe d (schedule F) (boundary F) Seed Spots
  /-- each play, read at the wipe's boundary as the wipe's game, with the strategy's moves there -/
  wipeRun : (θ F c : ByteArray) → (σ : Game.Strategy (play (rounds θ) θ F c)) →
    (wipe θ F).OfRun (play (rounds θ) θ F c)
  /-- the survivor threshold the wipe publishes with an accepted verdict -/
  foreignBits : ByteArray → ℕ
  honest : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c)
  certOf : ByteArray → Certificates
  crBound : ByteArray → Subprotocol → ℕ → ℝ
  delta : ByteArray → Subprotocol → ℕ → ℝ≥0∞
  /-- **A subprotocol's LMS-forgery term** at the strategy's SHA-512 calls: the chance that a release outside the
  device's log passes its check, a forgery of the device's LMS signature. A forgery needs SHA-512's one-wayness as
  well as its collision resistance, and no lock premise bounds it yet, so this term is open: an instance supplies it. -/
  forgery : ByteArray → Subprotocol → ℕ → ℝ≥0∞
  loss : Subprotocol → ByteArray → ℝ≥0∞ → ℝ≥0∞

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- The game after `θ`, `F` and `c`. -/
def game (θ F c : ByteArray) : Game Exchange := play (V.rounds θ) θ F c

/-- The committed files' circuit. -/
noncomputable def circuitOf (F : ByteArray) : Verity.Circuit := (V.read F).circuit

/-- The committed files' number of units. -/
noncomputable def units (F : ByteArray) : ℕ := (V.read F).units

/-- The committed files' units. -/
noncomputable def partsOf (F : ByteArray) : Verity.Partition (V.circuitOf F) (V.units F) := (V.read F).parts

/-- **The declaration the inference-only policy judges**: the committed circuit, with its registered inputs. -/
noncomputable def declaration (θ F : ByteArray) : Declaration := ⟨V.circuitOf F, V.theta θ F, V.theta_input θ F⟩

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

/-- **Subprotocol `a`'s verdict at composite `k`**: the committed blob passed the commit phase's check, `k` accepted
what the prover sent as a `same` composition, and `a`'s compiled component accepts its own case in it. -/
def Verdict (k : Verity.Registration.Composite) (a : Subprotocol) (o : Exchange) : Prop :=
  V.Registered o.params o.file ∧ SameAccepts k o.sent ∧ ComponentAccepts k a o.sent

/-- **The period after `θ`, `F` and `c` at composite `k`, as the one vocabulary's `Accounting.PeriodLive`**: the
committed files' units, the commitment's openings, the extractor, the finders, the gates computed, the world and the
verdicts. -/
noncomputable def live (k : Verity.Registration.Composite) (θ F c : ByteArray) :
    PeriodLive d α (V.circuitOf F) (V.game θ F c) where
  units := V.units F
  parts := V.partsOf F
  opens := V.Opens θ F c
  extract σ _ := V.extract θ F c σ
  finder a σ := V.finder (.ofAudit a) θ F c σ
  computed := V.computed θ F c
  world := V.world θ F c
  accepts a := V.Verdict k (.ofAudit a)

/-- **The claim subprotocol `a` certifies** in the play of `σ` that ended with `o`. The five period claims are
`Accounting.PeriodLive.Holds` at the period and the published certificates: sampled proofs' on the extractor's
transcript at `σ` (at most an `ε` share of the units unbound in it under the commitment's openings), PoUW's in gates (at
least `W` gates of the committed circuit computed after their window's coin), and the others' on the world the play
realized (at least `S` bits of device memory held on average, at least `Q` bits read, at most `K` bits through the
network certifiers), whose window and capacity account are the parameters' (`world_window`, `world_capacity`), so PoUS's
own coins choose neither. The inference-only policy's: the declaration complies with `Nci.Complies` at the published `Γ` and
`α`. The wipe's: the play, read at the wipe's boundary as the wipe's game, doesn't keep foreign content past the
published survivor threshold (`Verity.Wipe.OfRun.Kept`). -/
def Holds (k : Verity.Registration.Composite) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) :
    Subprotocol → Exchange → Prop
  | .sampled, o => (V.live k θ F c).Holds (V.certOf θ) σ .sampled o
  | .pouw, o => (V.live k θ F c).Holds (V.certOf θ) σ .pouw o
  | .pous, o => (V.live k θ F c).Holds (V.certOf θ) σ .pous o
  | .poub, o => (V.live k θ F c).Holds (V.certOf θ) σ .poub o
  | .network, o => (V.live k θ F c).Holds (V.certOf θ) σ .network o
  | .inferenceOnly, _ => Nci.Complies (V.gates θ) (V.fresh θ) (V.declaration θ F)
  | .wipe, o => ¬ (V.wipeRun θ F c σ).Kept (V.foreignBits θ) o

/-- **Each release `a`'s check read is the device's at its round**: the `k`-th release `a`'s check read is the log's
release at round `pos k`, the round the parameters assign it. Out of place, it is a forgery of the device's LMS
signature over its ledger or a session's close, which `a`'s error bounds by its forgery term (`Vocab.forgery`). -/
def Logged (pos : ℕ → ℕ) (a : Subprotocol) (o : Exchange) : Prop :=
  ∀ k r, (V.coinsRead a o.sent)[k]? = some r → o.log[pos k]? = some r

/-- **The finder for a second file with the registered digest**: on its one coin it hashes the committed blob and the
registered one, and outputs them when they differ, a SHA-512 collision when the blob passed the file check. -/
def fileFinder (θ F : ByteArray) : Claims.Finder where
  S := Unit
  out _ := if F = V.registered θ then none else some (F.data.toList, (V.registered θ).data.toList)
  cost _ := 2

/-- **`B^τ`'s collision premise, quantitative**: subprotocol `a`'s finder applied to the strategy outputs a SHA-512
collision with probability at most the published bound at the strategy's SHA-512 calls (`ε_cr`). `ecr/sha-512` is asked
at the finder's budget, `callsMul θ a * cost θ F c σ + callsAdd θ a` calls (`finder_calls`), a defined function of the
strategy's `cost` as `qF`/`qS` are of `t'`. -/
def FinderCR (a : Subprotocol) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : Prop :=
  (V.finder a θ F c σ).Pr sha512Msg ≤ V.crBound θ a (V.cost θ F c σ)

/-- **Subprotocol `a`'s error at a strategy**: where its finder meets its collision premise (`FinderCR`), its published
`δ` at the strategy's SHA-512 calls, plus its LMS-forgery term there (`forgery`), plus its loss at the probability that
the finder outputs a collision, the form of `Accounting.PeriodLive.error` with the forgery term beside `δ`; elsewhere no
bound (`⊤`), as `Accounting.condError` reads it. C-Flock's conditional bound is the case `loss = 0`, the additive one
the case `crBound = 1`. -/
noncomputable def error (a : Subprotocol) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : ℝ≥0∞ :=
  condError (V.FinderCR a θ F c)
    (fun σ => V.delta θ a (V.cost θ F c σ) + V.forgery θ a (V.cost θ F c σ) +
      V.loss a θ (ENNReal.ofReal ((V.finder a θ F c σ).Pr sha512Msg))) σ

/-- **Soundness at composite `k` for the acceptance `acc`, in the causal game.** At all parameters `θ`, committed files
`F` and commitments `c`, and for every strategy, the probability that `acc` holds of what the prover sent for a blob
that passed the commit phase's check while the claim of one of `k`'s subprotocols fails is at most their errors at that
strategy summed. -/
def SoundnessFor (k : Verity.Registration.Composite) (acc : Sent → Prop) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => acc o.sent ∧ V.Registered o.params o.file ∧ ∃ a ∈ subprotocols k, ¬ V.Holds k θ F c σ a o)
        (V.game θ F c) σ ≤
      ∑ a ∈ subprotocols k, V.error a θ F c σ

/-- **Soundness at composite `k`'s compiled verifier** (`Accepts k`). -/
def SoundnessAt (k : Verity.Registration.Composite) : Prop := V.SoundnessFor k (Accepts k)

/-- **Subprotocol `a`'s target at composite `k`**: at all `θ`, `F` and `c`, for every strategy, the probability that
`a`'s verdict at `k` accepts while its claim fails is at most its error at that strategy. For a subprotocol that decides
`Audit` `b` it is the per-period guarantee `Accounting.PeriodLive.Sound` at `live` (`Vocab.target_audit`). -/
def Target (k : Verity.Registration.Composite) (a : Subprotocol) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ ≤ V.error a θ F c σ

/-- **Sampled proofs' target**: at most an `ε` share of the units unbound in the extractor's transcript. -/
def SampledProofsSound (k : Verity.Registration.Composite) : Prop := V.Target k .sampled

/-- **PoUW's target**: at least `W` gates of the committed circuit computed after their window's coin. -/
def PouwSound (k : Verity.Registration.Composite) : Prop := V.Target k .pouw

/-- **The inference-only policy's target**: the declaration over the committed circuit complies with `Nci.Complies`
at the published `Γ` and `α`. -/
def InferenceOnlySound (k : Verity.Registration.Composite) : Prop := V.Target k .inferenceOnly

/-- **PoUS's target**: at least `S` bits of device memory held on average. -/
def PousSound (k : Verity.Registration.Composite) : Prop := V.Target k .pous

/-- **PoUB-P2's target** (`verity/remote-bandwidth/p2/v0`, on PoUS's store): at least `Q` bits read from device
memory. -/
def BandwidthAuditSound (k : Verity.Registration.Composite) : Prop := V.Target k .poub

/-- **The network certifier's target**: at most `K` bits through the network certifiers. -/
def NetworkCertifierSound (k : Verity.Registration.Composite) : Prop := V.Target k .network

/-- **The wipe's target**: the play doesn't keep foreign content past the survivor threshold the wipe publishes. -/
def WipeSound (k : Verity.Registration.Composite) : Prop := V.Target k .wipe

/-- **Subprotocol `a`'s honest lemma at the deployment**: at registered files, the deployment's honest prover meets
`a`'s claim at composite `k` on almost every play. Each subprotocol's piece proves its own (sampled proofs from
`Flock.Bridge.card_wrong_lt_of_relS_honest`; the others from their owners' `_honest` lemmas). -/
def HonestHolds (k : Verity.Registration.Composite) (a : Subprotocol) : Prop :=
  ∀ θ F c : ByteArray, V.Registered θ F →
    Game.prob (fun o => ¬ V.Holds k θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c) = 0

end Vocab

end Verity
