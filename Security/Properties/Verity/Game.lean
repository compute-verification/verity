import CoreOnsite
import Properties.Composition
import Definitions.Accounting.JointPeriod

/-!
# Soundness as a commit-then-coin game at onsite's compiled verifier, and each subprotocol's target

The verifier of record is one compiled Lean function, `Core.Onsite.protocol`'s `answer`, the composition
(`Core.Composition.product`) of the subprotocols that decide a period. Five of the six are its components today, one per
`Accounting.Audit`:

| subprotocol | `Audit` | component |
| --- | --- | --- |
| sampled proofs | `.sampled` | `SampledProofs.OneStage.protocol` |
| PoUW | `.pouw` | `PouwCheck.protocol` |
| PoUS | `.pous` | `PousCheck.RunProtocol.protocol` |
| the bandwidth audit | `.poub` | `RemoteAudit.Protocol.protocol` |
| the network certifier | `.network` | `NetworkCheck.Period.protocol` |

The sixth, the wipe (`MemoryChallenger.WipeProtocol.protocol`), is composed with them in
`Properties.Verity.Wipe`.

The statement's form is soundness's ruling R1 (26 Oct, 17:50Z):

* **A commit-then-coin game.** The prover's strategy (`Play`) sends the parameters and the commitment before any coin,
  so both are constants of the play, and only the evidence reads the coins.
* **`∃ X` fixed by the commitment.** The transcript the sampled-proofs claim counts is `X τ.commitment`, the same at
  every coin.
* **Each component's coins from the device's log.** Subprotocol `a` reads the coordinate `own a` of the coins the
  challenger device drew and logged. Its verdict at the coins `ξ` (`Vocab.run`) is its compiled component's acceptance
  with the coins that component read (`coinsRead`) equal to the device's `ξ (own a)`. A component that accepts on other
  coins accepted a signature on a message missing from the device's log, an LMS forgery, which is the event's `Break`.
* **The claims read `circuitOf`.** The circuit and its partition are the parameters' (`circuitOf`, `partsOf`), never
  the prover's world.

Each subprotocol's target (`SampledProofsSound`, `PouwSound`, `PousSound`, `BandwidthAuditSound`,
`NetworkCertifierSound`) is its per-period guarantee (`Accounting.PeriodRun.Sound`) at that run, so its verdict is its
compiled component's on the device's coins, not a free field: a prover that picks its own coins is outside it, and
the targets can hold for errors below 1. `Verity.Soundness` (`Properties.Verity.Soundness`) composes them.

The sampled component is the one-stage audit v2 (`SampledProofs.OneStageZk`, #4637) once onsite reads it (ONE-3b):
`ComponentAccepts .sampled` reads onsite's components, so this statement doesn't change when it does.

The names a subprotocol defines are `Vocab`'s fields: `circuitOf`, `units` and `partsOf` (ONE-7), `own` and
`coinsRead` (ONE-4), `world` (the runs PoUW, PoUS, the bandwidth audit and the network certifier realize: ONE-16,
ONE-10, ONE-11 and ONE-12), and `certOf`, `priceOf` and `delta` (what the product publishes, functions of the
parameters alone: ONE-19 for sampled proofs).
-/

namespace Verity

open Accounting Core

/-- The compiled component of onsite that decides `Audit` `a`. -/
def component : Audit → Protocol
  | .sampled => SampledProofs.OneStage.protocol
  | .pouw => PouwCheck.protocol
  | .pous => PousCheck.RunProtocol.protocol
  | .poub => RemoteAudit.Protocol.protocol
  | .network => NetworkCheck.Period.protocol

/-- What the prover sends onsite's verifier of record: the parameters, the commitment and the evidence, as bytes. -/
structure Sent where
  params : ByteArray
  commitment : ByteArray
  evidence : ByteArray

/-- **The verifier of record accepts what the prover sent.** -/
def Accepts (s : Sent) : Prop :=
  (Onsite.protocol.answer s.params s.commitment s.evidence).verdict = .accept

/-- The protocol `P` accepts its own case in what the prover sent: onsite's three byte strings decode, and one of their
cases is `P`'s, as one of onsite's components, which decodes under the parameters onsite names for it, and `P`'s check
accepts it. -/
def CaseAccepts (P : Protocol) (s : Sent) : Prop :=
  ∃ (ps : List Composition.Named) (c : ByteArray) (ev : List Composition.Case),
    (canonicalJson s.params >>= Composition.decodeParams) = .ok ps ∧ Composition.decodeCommitment s.commitment = .ok c ∧
      Composition.decodeEvidence s.evidence = .ok ev ∧
      ∃ x ∈ Properties.Composition.triples Onsite.components ps ev, x.1 = P ∧
        Properties.Composition.DecodedAccepted x.1 x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence

/-- `Audit` `a`'s component accepts its own case in what the prover sent. -/
def ComponentAccepts (a : Audit) (s : Sent) : Prop := CaseAccepts (component a) s

/-- **A prover's strategy in the commit-then-coin game**: the parameters and the commitment are sent before any coin
is drawn, so they are constants; only the evidence reads the coins `ξ` the challenger device drew and logged. -/
structure Play (Ξ : Type) where
  params : ByteArray
  commitment : ByteArray
  evidence : Ξ → ByteArray

/-- What the prover sends onsite at the coins `ξ`. -/
def Play.sent {Ξ : Type} (τ : Play Ξ) (ξ : Ξ) : Sent := ⟨τ.params, τ.commitment, τ.evidence ξ⟩

/-- The fields of a period's run that the prover's world realizes, at the coins: whose account each resource is
charged to, the window, the device schedule, and what the network certifiers saw. -/
structure World {ι : Type} (d : Device ι) (α : Type) [Fintype α] (Ξ : Type) where
  acct : Resource → α
  win : Window
  schedule : Ξ → Schedule ι α
  fits : ∀ ξ, (schedule ξ).Fits d
  commWithin : ℝ → Ξ → Prop

/-- **The names the subprotocols define.** The circuit, its units and partition from the parameters (ONE-7); the
coordinate of the device's log each subprotocol reads, and the coins each component's check read, decoded from the
device's signed log entries in what the prover sent (ONE-4); the world each play realizes (ONE-10, ONE-11, ONE-12,
ONE-16); and the certificates, price and errors the product publishes, from the parameters alone (ONE-19). -/
structure Vocab {ι : Type} (d : Device ι) (α : Type) [Fintype α] {I : Type} (Ω : I → Type) where
  circuitOf : ByteArray → Verity.Circuit
  units : ByteArray → ℕ
  partsOf : (θ : ByteArray) → Verity.Partition (circuitOf θ) (units θ)
  own : Audit → I
  coinsRead : (a : Audit) → Sent → Option (Ω (own a))
  world : Play ((i : I) → Ω i) → World d α ((i : I) → Ω i)
  certOf : ByteArray → Certificates
  priceOf : ByteArray → Price
  delta : ByteArray → Audit → ℝ

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {I : Type} {Ω : I → Type} (V : Vocab d α Ω)

/-- The period the play `τ` realizes, with the transcript `X`: the parameters' circuit and partition, the world's
fields, and subprotocol `a`'s verdict at the coins `ξ` its compiled component's acceptance of what the prover sent, on
the coins the device logged for it. -/
def run (τ : Play ((i : I) → Ω i)) (X : (V.circuitOf τ.params).Transcript) :
    PeriodRun d α (V.circuitOf τ.params) Ω where
  units := V.units τ.params
  parts := V.partsOf τ.params
  own := V.own
  acct := (V.world τ).acct
  win := (V.world τ).win
  schedule := (V.world τ).schedule
  fits := (V.world τ).fits
  transcript := fun _ => X
  commWithin := (V.world τ).commWithin
  accepts := fun a ξ => ComponentAccepts a (τ.sent ξ) ∧ V.coinsRead a (τ.sent ξ) = some (ξ (V.own a))

/-- **The break**: some component accepts what the prover sent on coins other than those the device logged for it,
which is a signature, under the device's key, on a message missing from the device's log. -/
def Break (τ : Play ((i : I) → Ω i)) (ξ : (i : I) → Ω i) : Prop :=
  ∃ a, ComponentAccepts a (τ.sent ξ) ∧ V.coinsRead a (τ.sent ξ) ≠ some (ξ (V.own a))

variable [Fintype I] [DecidableEq I] [∀ i, Fintype (Ω i)]

/-- **Soundness, in the commit-then-coin form.** For every strategy there is a transcript `X`, fixed by the
commitment, such that the probability over the device's coins that onsite's compiled verifier accepts while one of
the period's claims fails on the parameters' circuit (more than an `ε` share of its units wrong in `X`, less than `W`
of work, fewer than `S` bits held on average, fewer than `Q` bits read, or more than `K` bits through the network
certifiers), with no break, is at most the published errors summed. -/
def Soundness : Prop :=
  ∀ τ : Play ((i : I) → Ω i), ∃ X : ByteArray → (V.circuitOf τ.params).Transcript,
    Pouw.pr (fun ξ => Accepts (τ.sent ξ) ∧
        (∃ a, ¬ (V.run τ (X τ.commitment)).Holds (V.certOf τ.params) (V.priceOf τ.params) a ξ) ∧ ¬ V.Break τ ξ)
      ≤ ∑ a, V.delta τ.params a

/-- **Sampled proofs' target (ONE-8)**: for every strategy, a transcript fixed by the commitment at which sampled
proofs, on the device's logged coins, has its per-period guarantee at the published `ε` and error. -/
def SampledProofsSound : Prop :=
  ∀ τ : Play ((i : I) → Ω i), ∃ X : ByteArray → (V.circuitOf τ.params).Transcript,
    (V.run τ (X τ.commitment)).Sound (V.certOf τ.params) (V.priceOf τ.params) .sampled (V.delta τ.params .sampled)

/-- **Subprotocol `a`'s target, for one whose claim doesn't read the transcript**: at every strategy and every
transcript, `a` on the device's logged coins has its per-period guarantee at the published certificate and error. -/
def SoundAt (a : Audit) : Prop :=
  ∀ (τ : Play ((i : I) → Ω i)) (X : (V.circuitOf τ.params).Transcript),
    (V.run τ X).Sound (V.certOf τ.params) (V.priceOf τ.params) a (V.delta τ.params a)

/-- **PoUW's target (ONE-16)**: at least `W` of work at the published price. -/
def PouwSound : Prop := V.SoundAt .pouw

/-- **PoUS's target (ONE-10)**: at least `S` bits of device memory held on average. -/
def PousSound : Prop := V.SoundAt .pous

/-- **The bandwidth audit's target (ONE-11)**: at least `Q` bits read from device memory. -/
def BandwidthAuditSound : Prop := V.SoundAt .poub

/-- **The network certifier's target (ONE-12)**: at most `K` bits through the network certifiers. -/
def NetworkCertifierSound : Prop := V.SoundAt .network

end Vocab

end Verity
