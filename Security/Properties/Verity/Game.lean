import CoreOnsite
import Properties.Composition
import Properties.Flock.Circuit.CircuitOf
import Definitions.Accounting.JointLive
import Definitions.Core.Game
import Definitions.Hash.RowLeaf
import Flock.Hash

/-!
# Soundness as a causal game at onsite's compiled verifier, and each subprotocol's target

The verifier of record is one compiled Lean function, `Core.Onsite.protocol`'s `answer` on the caller's blobs: the
composition of its components (`Core.Composition.sameB`), whose inference-only policy (`NciCheck.Protocol.protocol`)
and sampled proofs' one-stage audit must commit to one record (`Core.Onsite.tied`). Five of its components decide a
period, one per `Accounting.Audit`:

| subprotocol | `Audit` | component |
| --- | --- | --- |
| sampled proofs | `.sampled` | `SampledProofs.OneStageZk.protocol` |
| PoUW | `.pouw` | `PouwCheck.Period.protocol` |
| PoUS | `.pous` | `PousCheck.RunProtocol.protocol` |
| the bandwidth audit | `.poub` | `RemoteAudit.Protocol.protocol` |
| the network certifier | `.network` | `NetworkCheck.Period.protocol` |

The policy's acceptance is in every verdict (`RecordAccepts`). The wipe (`MemoryChallenger.WipeProtocol.protocol`) is
the seventh component, with no `Audit` and no target here.

The statement's form is soundness's rulings R1, R4′, R5 and R6 and the one adversary form:

* **A causal game (R5, R6).** The prover sends the parameters `θ`, the committed files `F` (one blob) and the
  commitment `c` first, before any coin, so every statement is `∀ θ F c`, and then plays `play (rounds θ) θ F c`, a
  `Verity.Game`: in each of the device's rounds the prover sends the bytes (a root, a transcript digest) that the
  round's release answers, and then the device draws and releases that round's coins (`releases`); the evidence comes
  last. A strategy (`Verity.Game.Strategy`) is a function of the coins released so far and never of later ones.
* **The claims read the committed files (R6).** The circuit, its units and its constants are what the verifier reads
  from the files (`Flock.Bridge.circuitOfFile`, `unitsOfFile`, `partsOfFile` at v2's reading of the blob, `read`), and
  the positions are the files' and the commitment's (`place`), never the prover's world. The params hold the circuit
  file's digest (`programDigest`): a file whose SHA-512 isn't it is refused (`Registered`, conjoined beside onsite's
  acceptance).
* **Sampled proofs' claim reads one fixed extractor's transcript (R4′).** The extractor (`extract`, the one
  `SampledKSLive*` is stated with, `XplurZC`) is a construction applied to the strategy, never chosen per strategy.
  The claim is that at most an `ε` share of the file's units are unbound in its transcript (`Accounting.unbound`): wrong in it, or
  with a committed input or output the commitment doesn't open to its value there (`Opens`: a registered constant to its
  constant, any other wire by an opening of its row leaf that `Hash.RowLeaf.accepts` passes under the compiled SHA-512).
* **One adversary form: every prover, and collisions a fixed finder outputs.** There is no strategy class and no break
  in the event. The finder `B` (`finder a`, a `Claims.Finder`) is a fixed construction applied to the strategy whose
  SHA-512 calls are at most `c · calls(τ) + d` (`finder_calls`). Subprotocol `a`'s error at a strategy (`error`) is,
  where `B^τ` outputs a SHA-512 collision with probability at most the published `ε_cr` at the strategy's calls
  (`FinderCR`, quantitative), its published `δ` at those calls plus its loss at `Pr[B^τ]`; elsewhere `⊤`. That admits
  C-Flock's conditional bound (`loss = 0`, F3) and the additive one (`crBound = 1`). Two openings of one position, a
  forged device release (LMS is hash-based) and a second file with the registered digest (`fileFinder`) each enter
  that way. The named assumption `ecr/sha-512` is asked only of the finder.
* **The one vocabulary.** The period is `Accounting.PeriodLive` (`live`), and the claim is its `Holds`.
* **Each component's acceptance in what the prover sent.** Subprotocol `a`'s verdict (`Verdict`) is the file check,
  onsite's cases an accepted composition whose policy and sampled proofs commit to one record (`RecordAccepts`, from
  `Properties.Composition.AcceptMeans`), and `a`'s compiled component's acceptance of its own case. The blobs are
  content-addressed bytes sent with the evidence, last, and each acceptance counts any blobs. The releases each
  component's check read (`coinsRead`) are the targets' to bound (`Logged`).
* **PoUW counts gates** of the committed circuit the play computed after their window's coin, over the period's windows
  (`computed`), with no pricing.

Each subprotocol's target (`SampledProofsSound`, `PouwSound`, `PousSound`, `BandwidthAuditSound`,
`NetworkCertifierSound`) is `Target a`: at every `θ`, `F`, `c` and strategy, the probability that `a`'s verdict accepts
while its claim fails is at most its error there. `Verity.Soundness` (`Properties.Verity.Soundness`) composes them.

The names a subprotocol defines are `Vocab`'s fields: `read` (v2's reading of the blob, ONE-3b), `place`,
`programDigest` and `registered` (ONE-7), `rounds` and `coinsRead` (ONE-4), `extract` and the finders (ONE-8 and each
target's), `computed` (ONE-16), `world` (the runs PoUS, the bandwidth audit and the network certifier realize: ONE-10,
ONE-11 and ONE-12), and `certOf`, `delta` and `loss` (what the product publishes, functions of the parameters alone:
ONE-19).
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

/-- What the prover sends onsite's verifier of record: the parameters, the commitment and the evidence, as bytes. -/
structure Sent where
  params : ByteArray
  commitment : ByteArray
  evidence : ByteArray

/-- **The verifier of record accepts what the prover sent**, on some blobs. -/
def Accepts (s : Sent) : Prop :=
  ∃ bs, (Onsite.protocol.answer s.params s.commitment s.evidence bs).verdict = .accept

/-- Onsite's components on the blobs `bs`, each reading them. -/
def componentsOn (bs : Array ByteArray) : List Protocol := Onsite.components.map (·.onBlobs (Blobs.ofArray bs))

/-- Onsite's three byte strings decode under its composition's formats: the components' parameters `ps`, the binding
`b` of their digests, and their cases `ev`. -/
def Decoded (s : Sent) (ps : List Composition.Named) (b : ByteArray) (ev : List Composition.Case) : Prop :=
  (canonicalJson s.params >>= Composition.decodeParams) = .ok ps ∧ Composition.decodeCommitment s.commitment = .ok b ∧
    Composition.decodeEvidence s.evidence = .ok ev

/-- The protocol `P` accepts its own case in what the prover sent, on some blobs: onsite's three byte strings decode,
and one of the components' cases is `P`'s, as one of onsite's components on the blobs, which decodes under the
parameters onsite names for it, and `P`'s check on the blobs accepts it. -/
def CaseAccepts (P : BlobProtocol) (s : Sent) : Prop :=
  ∃ bs ps b ev, Decoded s ps b ev ∧
    ∃ x ∈ Properties.Composition.triples (componentsOn bs) ps ev, x.1 = P.onBlobs (Blobs.ofArray bs) ∧
      Properties.Composition.DecodedAccepted x.1 x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence

/-- `Audit` `a`'s component accepts its own case in what the prover sent. -/
def ComponentAccepts (a : Audit) (s : Sent) : Prop := CaseAccepts (component a) s

/-- **Onsite's cases are an accepted composition whose policy and sampled proofs commit to one record**: on some
blobs, onsite's three byte strings decode, the cases are an accepted composition of its components on the blobs
(`Properties.Composition.Accepted`: every component accepts its own case, the policy's among them, and the commitment
binds their digests), and the cases of `Onsite.tied`, the policy's and sampled proofs' one-stage audit's, commit to one
record: none has a digest other than the first's (`Composition.strays`). -/
def RecordAccepts (s : Sent) : Prop :=
  ∃ bs ps b ev, Decoded s ps b ev ∧ ∃ pub prof,
    Properties.Composition.Accepted "verity/onsite/v0" (componentsOn bs) ps b ev pub prof ∧
      Composition.strays Onsite.tied (componentsOn bs) ev = none

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
structure Ending where
  params : ByteArray
  file : ByteArray
  commitment : ByteArray
  log : List Release
  evidence : ByteArray

/-- What the prover sent onsite in the play that ended with `o`. -/
def Ending.sent (o : Ending) : Sent := ⟨o.params, o.commitment, o.evidence⟩

/-- **The causal game after the parameters `θ`, the committed files `F` and the commitment `c`**: the device's rounds
`ns`, each release after the bytes it answers, then the evidence. -/
def play (ns : List ℕ) (θ F c : ByteArray) : Game Ending :=
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

/-- **The names the subprotocols define.** v2's reading of the committed blob (ONE-3b); the positions from the files and
the commitment, the circuit file's digest in the parameters and the registered blob (ONE-7); the device's rounds, and
the releases each component's check read, decoded from the device's signed log entries in what the prover sent
(ONE-4); the extractor applied to a strategy, and each subprotocol's collision finder (ONE-8 and each target's), with
the finder's SHA-512 calls bounded by the strategy's (`finder_calls`); the committed circuit's gates a play computed
after their window's coin (ONE-16); the world each play realizes (ONE-10, ONE-11, ONE-12); the deployment's honest
prover; and the certificates, errors, collision bounds and losses the product publishes, from the parameters and the
strategy's SHA-512 calls alone (ONE-19).

`cost θ F c σ` is the strategy's SHA-512 calls in one run (ONE8's `t' τ`): `Verity.Game` has no call model of
strategies, so it is declared, as `Claims.Finder.cost` is. -/
structure Vocab {ι : Type} (d : Device ι) (α : Type) [Fintype α] where
  read : ByteArray → FileRead
  place : (θ F c : ByteArray) → Fin (read F).circuit.N → Option Place
  programDigest : ByteArray → Option Hash.Digest
  registered : ByteArray → ByteArray
  rounds : ByteArray → List ℕ
  coinsRead : Audit → Sent → List Release
  extract : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → (read F).circuit.Transcript
  finder : Audit → (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Claims.Finder
  cost : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → ℕ
  callsMul : ByteArray → Audit → ℕ
  callsAdd : ByteArray → Audit → ℕ
  /-- **The finder's SHA-512 calls are at most `c · calls(τ) + d`**, on every coin of the finder. -/
  finder_calls : ∀ a θ F c σ (x : (finder a θ F c σ).S),
    (finder a θ F c σ).cost x ≤ callsMul θ a * cost θ F c σ + callsAdd θ a
  computed : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Ending → Finset (Fin (read F).circuit.N)
  world : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Ending → World d α
  honest : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c)
  certOf : ByteArray → Certificates
  crBound : ByteArray → Audit → ℕ → ℝ
  delta : ByteArray → Audit → ℕ → ℝ≥0∞
  loss : Audit → ByteArray → ℝ≥0∞ → ℝ≥0∞

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- The game after `θ`, `F` and `c`. -/
def game (θ F c : ByteArray) : Game Ending := play (V.rounds θ) θ F c

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

/-- **Subprotocol `a`'s verdict**: the committed blob passed the commit phase's check, onsite's cases are an accepted
composition whose policy and sampled proofs commit to one record, and `a`'s compiled component accepts its own case in
what the prover sent. -/
def Verdict (a : Audit) (o : Ending) : Prop :=
  V.Registered o.params o.file ∧ RecordAccepts o.sent ∧ ComponentAccepts a o.sent

/-- **The period after `θ`, `F` and `c`, as the one vocabulary's `Accounting.PeriodLive`**: the committed files'
units, the commitment's openings, the extractor, the finders, the gates computed, the world and the verdicts. -/
noncomputable def live (θ F c : ByteArray) : PeriodLive d α (V.circuitOf F) (V.game θ F c) where
  units := V.units F
  parts := V.partsOf F
  opens := V.Opens θ F c
  extract σ _ := V.extract θ F c σ
  finder a σ := V.finder a θ F c σ
  computed := V.computed θ F c
  world := V.world θ F c
  accepts := V.Verdict

/-- **The claim subprotocol `a` certifies** in the play of `σ` that ended with `o`: `Accounting.PeriodLive.Holds` at the
period and the published certificates. Sampled proofs' on the extractor's transcript at `σ` (at most an `ε` share of
the units unbound in it under the commitment's openings), PoUW's in gates (at least `W` gates of the committed circuit
computed after their window's coin), and the others' on the world the play realized (at least `S` bits of device memory
held on average, at least `Q` bits read, at most `K` bits through the network certifiers). -/
def Holds (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : Audit → Ending → Prop :=
  (V.live θ F c).Holds (V.certOf θ) σ

/-- Every release `a`'s check read is in the device's log; outside it, the release is a forgery of the device's LMS
signature, which `a`'s finder turns into a collision. -/
def Logged (a : Audit) (o : Ending) : Prop := ∀ r ∈ V.coinsRead a o.sent, r ∈ o.log

/-- **The finder for a second file with the registered digest**: on its one coin it hashes the committed blob and the
registered one, and outputs them when they differ, a SHA-512 collision when the blob passed the file check. -/
def fileFinder (θ F : ByteArray) : Claims.Finder where
  S := Unit
  out _ := if F = V.registered θ then none else some (F.data.toList, (V.registered θ).data.toList)
  cost _ := 2

/-- **`B^τ`'s collision premise, quantitative**: subprotocol `a`'s finder applied to the strategy outputs a SHA-512
collision with probability at most the published bound at the strategy's SHA-512 calls (`ε_cr`). `ecr/sha-512` is asked
at the finder's budget, `callsMul θ a * cost θ F c σ + callsAdd θ a` calls (`finder_calls`), a defined function of the
strategy's `cost` as ONE8's `qF`/`qS` are of `t'`. -/
def FinderCR (a : Audit) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : Prop :=
  (V.finder a θ F c σ).Pr sha512Msg ≤ V.crBound θ a (V.cost θ F c σ)

open Classical in
/-- **Subprotocol `a`'s error at a strategy**: where its finder meets its collision premise (`FinderCR`), its
period's error (`Accounting.PeriodLive.error`: its published `δ` at the strategy's SHA-512 calls plus its loss at the
probability that the finder outputs a collision); elsewhere no bound (`⊤`). This is ONE8's
`condError FinderCR δ σ := if FinderCR σ then δ σ else ⊤`; C-Flock's conditional bound is the case `loss = 0` (F3), the
additive one the case `crBound = 1`. -/
noncomputable def error (a : Audit) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : ℝ≥0∞ :=
  if V.FinderCR a θ F c σ then
    (V.live θ F c).error sha512Msg a (V.delta θ a (V.cost θ F c σ)) (V.loss a θ) σ
  else ⊤

/-- **Soundness for the acceptance `acc`, in the causal game.** At all parameters `θ`, committed files `F` and
commitments `c`, and for every strategy, the probability that `acc` holds of what the prover sent for a blob that passed
the commit phase's check while one of the period's claims fails (more than an `ε` share of the units unbound in the
extractor's transcript, fewer than `W` gates computed after their coin, fewer than `S` bits held on average, fewer than
`Q` bits read, or more than `K` bits through the network certifiers) is at most the subprotocols' errors at that strategy
summed. -/
def SoundnessFor (acc : Sent → Prop) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => acc o.sent ∧ V.Registered o.params o.file ∧ ∃ a, ¬ V.Holds θ F c σ a o) (V.game θ F c) σ ≤
      ∑ a, V.error a θ F c σ

/-- **Soundness at onsite's compiled verifier** (`Accepts`). -/
def Soundness : Prop := V.SoundnessFor Accepts

/-- **Subprotocol `a`'s target**: at all `θ`, `F` and `c`, and for every strategy, the probability that `a`'s verdict
accepts while its claim fails is at most its error at that strategy. -/
def Target (a : Audit) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict a o ∧ ¬ V.Holds θ F c σ a o) (V.game θ F c) σ ≤ V.error a θ F c σ

/-- **Sampled proofs' target (ONE-8)**: at most an `ε` share of the units unbound in the extractor's transcript. -/
def SampledProofsSound : Prop := V.Target .sampled

/-- **PoUW's target (ONE-16)**: at least `W` gates of the committed circuit computed after their window's coin. -/
def PouwSound : Prop := V.Target .pouw

/-- **PoUS's target (ONE-10)**: at least `S` bits of device memory held on average. -/
def PousSound : Prop := V.Target .pous

/-- **The bandwidth audit's target (ONE-11)**: at least `Q` bits read from device memory. -/
def BandwidthAuditSound : Prop := V.Target .poub

/-- **The network certifier's target (ONE-12)**: at most `K` bits through the network certifiers, where the
inference-only policy accepted the record sampled proofs committed to. -/
def NetworkCertifierSound : Prop := V.Target .network

/-- **Subprotocol `a`'s honest lemma at the deployment**: at registered files, the deployment's honest prover meets
`a`'s claim on almost every play. Each subprotocol's piece proves its own (sampled proofs from
`Flock.Bridge.card_wrong_lt_of_relS_honest`; the others from their owners' `_honest` lemmas). -/
def HonestHolds (a : Audit) : Prop :=
  ∀ θ F c : ByteArray, V.Registered θ F →
    Game.prob (fun o => ¬ V.Holds θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c) = 0

end Vocab

end Verity
