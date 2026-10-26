import Properties.Verity.Soundness

/-!
# Soundness as a commit-then-coin game, and every leg's target

`Verity.Soundness` (`Properties.Verity.Soundness`) bounds the verifier of record's acceptance over runs whose
transcript, coins and world are free fields. Soundness's ruling R1 (26 Oct, 17:50Z) fixes the statement's final form,
which this module states, with its middle layer still `soundness_of_legs`'s:

* **A commit-then-coin game.** The prover's strategy (`Play`) sends the parameters and the commitment before any coin,
  so both are constants of the play, and only the evidence reads the coins.
* **`∃ X` fixed by the commitment.** The transcript the integrity claim counts is `X τ.commitment`, the same at every
  coin.
* **Each component's coins from the device's log.** Audit `a` reads the coordinate `own a` of the coins the challenger
  device drew and logged. Its verdict at the coins `ξ` (`Vocab.run`) is its compiled component's acceptance with the
  coins that component read (`coinsRead`) equal to the device's `ξ (own a)`. A component that accepts on other coins
  accepted a signature on a message missing from the device's log, an LMS forgery, which is the event's `Break`.
* **The claims read `circuitOf`.** The circuit and its partition are the parameters' (`circuitOf`, `partsOf`), never
  the prover's world.

The sampled component is the one-stage audit v2 (`SampledProofs.OneStageZk`, #4637) once onsite reads it (ONE-3b):
`ComponentAccepts .sampled` reads onsite's components, so this statement doesn't change when it does.

The names a leg defines are `Vocab`'s fields: `circuitOf`, `units` and `partsOf` (ONE-7), `own` and `coinsRead`
(ONE-4), `world` (the runs PoUW, PoUS, the bandwidth audit and the network certifier realize: ONE-16, ONE-10, ONE-11
and ONE-12), and `certOf`, `priceOf` and `delta` (what the product publishes, functions of the parameters alone:
ONE-19 for integrity). The legs' targets are `IntegrityLeg` (ONE-8) and `Leg a` for the other audits (ONE-18), and
ONE-20 (`Proofs.Properties.Verity.Game`) proves `Vocab.Soundness` from them.
-/

namespace Verity

open Accounting Core

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

/-- **The names the legs define.** The circuit, its units and partition from the parameters (ONE-7); the coordinate of
the device's log each audit reads, and the coins each component's check read, decoded from the device's signed log
entries in what the prover sent (ONE-4); the world each play realizes (ONE-10, ONE-11, ONE-12, ONE-16); and the
certificates, price and errors the product publishes, from the parameters alone (ONE-19). -/
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
fields, and audit `a`'s verdict at the coins `ξ` its compiled component's acceptance of what the prover sent, on the
coins the device logged for it. -/
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

/-- **ONE-8's target, the integrity leg**: for every strategy, a transcript fixed by the commitment at which the
sampled audit, on the device's logged coins, has its per-period guarantee at the published `ε` and error. -/
def IntegrityLeg : Prop :=
  ∀ τ : Play ((i : I) → Ω i), ∃ X : ByteArray → (V.circuitOf τ.params).Transcript,
    (V.run τ (X τ.commitment)).Sound (V.certOf τ.params) (V.priceOf τ.params) .sampled (V.delta τ.params .sampled)

/-- **Audit `a`'s leg** (work: ONE-16, space: ONE-10, bandwidth: ONE-11, communication: ONE-12, composed by ONE-18):
at every strategy and every transcript, the audit on the device's logged coins has its per-period guarantee at the
published certificate and error. Its claim doesn't read the transcript. -/
def Leg (a : Audit) : Prop :=
  ∀ (τ : Play ((i : I) → Ω i)) (X : (V.circuitOf τ.params).Transcript),
    (V.run τ X).Sound (V.certOf τ.params) (V.priceOf τ.params) a (V.delta τ.params a)

end Vocab

end Verity
