module

public import Definitions.Pouw.Confidence
public import Definitions.Core.Site
public import Definitions.Core.Interaction

/-!
# Acceptance as the verifier's code on the transcript (draft)

`ResourceAccounting.Sound` reads acceptance as a predicate `acc` of the view of the auditor's leaves. Here `acc` is the
verdict the verifier's code writes, and the premise that ties the verdict to what arrived is stated once for every
protocol. The auditor's leaves are two: the challenge device's `Vd`, whose input gates hold the challenge, and the
verifier's `Vv`, which has no input gates and hears the challenge as a message from `Vd`.

* `heard`: the messages that crossed into `Vv` before a tick, each with its source's leaf (its sender), read off the
  run, not the view: this list is the transcript.
* `transcript`: `heard` at the verifier's deadline, the argument of its code.
* `challenge`: the values of a set of leaves' input gates, in order of position: at `Vd`, the challenge it holds.
* `Accepts`: the view holds the verdict bit `1` at the verifier's decision cell at its deadline.
* `DeviceAsWritten` and `AsWritten`, the two parts of the premise on the feasible runs: the device sends its input, at
  or after its tick, and nothing else; the verifier has no input gates, its verdict is its code's answer on the
  transcript, and everything read off it is its code's function of what it had heard.
* `Implements` and `HeardIsModel`: the bridge to `Claims.Transcript`: in a run that implements the interaction
  model, the prover's messages of the model are what the verifier heard from outside the auditor's leaves, in order.
* `RegistrationAccepted`: `Sound` for registration at that acceptance.
* `ProposedSound`, `Holds` and `RegistrationAcceptedFamily`: the proposed replacement of `Sound`, with the feasible runs
  a challenge at a time, and registration's instance of it.

A message carries its source's leaf. A message from `Vd` is the device's input (`DeviceAsWritten`), so a challenge
substituted on a wire from `Vd` into `Vv` is excluded: that wire is inside the auditor's leaves. A challenge relayed
through leaves off the auditor's arrives with a source off them, as any prover's message does, and the code takes it
as the challenge at the soundness gap of the relayed challenge, unless the code authenticates it. Under `Sound`, whose
feasible runs don't depend on the challenge, no premise on them says the device holds the strategy's own challenge;
under `ProposedSound` that is `Holds`, a premise on the feasible runs.
-/

@[expose] public section

namespace Draft.Acceptance

open ResourceAccounting Finset

variable {L : Type*} [LinearOrder L]

open Classical in
/-- **What the leaves `Vv` heard before `d`**: each source placed off `Vv` that a gate of `Vv` placed before `d` reads,
as its leaf and its value at its width, in order of the reading tick, the reading leaf and the source's position. A
message is a source arriving at a leaf at a tick: two gates of one leaf reading one source at one tick hear it once,
since how many of its gates read it is the verifier's own computation; two leaves, or two ticks, hear it twice. -/
noncomputable def heard (R : Actual.Run L) (Vv : Finset L) (d : ℕ) : List (L × List Bool) :=
  ((((univ ×ˢ univ).filter fun p : Fin R.A.N × Fin R.A.N =>
      p.2 ∈ R.A.reads p.1 ∧ R.A.x p.2 ∉ Vv ∧ R.A.x p.1 ∈ Vv ∧ R.A.t p.1 < d).image
    fun p => toLex (R.A.t p.1, toLex (R.A.x p.1, Actual.position (R.seen p.2)))).sort).map
    fun k => ((ofLex (ofLex (ofLex k).2).2).1, Actual.chunk (ofLex (ofLex k).2).2)

/-- **The transcript of the verifier at `Vv` up to its deadline `d`**: what it heard before `d`. -/
noncomputable def transcript (R : Actual.Run L) (Vv : Finset L) (d : ℕ) : List (L × List Bool) := heard R Vv d

open Classical in
/-- **The challenge the leaves `Vd` hold**: their input gates' values, in order of position, as one bit string. -/
noncomputable def challenge (R : Actual.Run L) (Vd : Finset L) : List Bool :=
  (((univ.filter fun g => R.A.isInput g = true ∧ R.A.x g ∈ Vd).image fun g => Actual.position (R.seen g)).sort).flatMap
    Actual.chunk

/-- **The verifier accepts**: the view holds the bit `1` at the decision cell, cell `c` of leaf `y`, at tick `d`. -/
def Accepts (y : L) (c d : ℕ) (v : Set (ℕ × L × ℕ × ℕ × ℕ)) : Prop := (d, y, c, 1, 1) ∈ v

/-- **The challenge device at `Vd` runs as written**, in every feasible run: every gate off `Vd` that reads a gate of
`Vd` reads one of its input gates, at tick `t₀` or later. It sends its input, from its tick, and nothing else. -/
def DeviceAsWritten (𝓕 : Set (Actual.Run L)) (Vd : Finset L) (t₀ : ℕ) : Prop :=
  ∀ R ∈ 𝓕, ∀ g a, a ∈ R.A.reads g → R.A.x a ∈ Vd → R.A.x g ∉ Vd → R.A.isInput a = true ∧ t₀ ≤ R.A.t g

/-- **The verifier at `Vv` runs its code as written**, in every feasible run: `Vv` has no input gates; the decision
cell `c` of `y ∈ Vv` takes a one-bit gate at tick `d`, which holds `1` exactly when `vf` accepts the transcript; and
every gate of `Vv` that a gate off `Vv` reads is at the position and with the width and value that `vsend` gives on
what `Vv` had heard by its tick (up to and including it): what it sends is its code's function of the transcript so
far, and it computes nothing else that is read off it. -/
def AsWritten (𝓕 : Set (Actual.Run L)) (Vv : Finset L) (y : L) (c d : ℕ) (vf : List (L × List Bool) → Bool)
    (vsend : List (L × List Bool) → L × ℕ × ℕ → Option (ℕ × ℕ)) : Prop :=
  y ∈ Vv ∧ ∀ R ∈ 𝓕,
    (∀ g, R.A.x g ∈ Vv → R.A.isInput g = false) ∧
    (∃ g, R.A.x g = y ∧ R.A.cell g = c ∧ R.A.t g = d) ∧
    (∀ g, R.A.x g = y → R.A.cell g = c → R.A.t g = d →
      R.A.w g = 1 ∧ ((R.val g).toNat = 1 ↔ vf (transcript R Vv d) = true)) ∧
    ∀ g a, a ∈ R.A.reads g → R.A.x a ∈ Vv → R.A.x g ∉ Vv →
      vsend (heard R Vv (R.A.t a + 1)) (R.A.x a, R.A.cell a, R.A.t a) = some (R.A.w a, (R.val a).toNat)

/-- **The run implements the interaction model's transcript `τ` at the verifier `Vv`**, beside the device `Vd`, up
to `d`, with the model's messages read as bits by `enc`: what crosses into `Vv` from off the auditor's leaves before
`d` is one source a round, round `j`'s arriving at tick `u j` at the leaf `y j`, the ticks increasing, its value at its
width the prover's message `a_j`. -/
def Implements (R : Actual.Run L) (Vd Vv : Finset L) (d : ℕ) (τ : Claims.Transcript)
    (enc : Claims.Msg → List Bool) : Prop :=
  ∃ (u : ℕ → ℕ) (y : ℕ → L) (s : ℕ → ℕ × L × ℕ × ℕ × ℕ), StrictMono u ∧
    (∀ j < τ.msgs.length, u j < d ∧ (s j).2.1 ∉ Vd ∧
      (τ.msgs[j]?).map (fun m => enc m.1) = some (Actual.bits (s j).2.2.2.1 (s j).2.2.2.2)) ∧
    R.crossing (↑(Vd ∪ Vv))ᶜ ↑Vv (Set.Iio d) = {p | ∃ j < τ.msgs.length, p = (u j, y j, s j)}

/-- **The bridge to `Claims.Transcript`**: in a run that implements the interaction model's transcript `τ`, what
the verifier heard from off the device, in order, is the prover's messages of `τ`, read as bits. The verifier's own
messages are what `AsWritten`'s `vsend` sends. -/
def HeardIsModel : Prop :=
  ∀ {L : Type*} [LinearOrder L] (R : Actual.Run L) (Vd Vv : Finset L) (d : ℕ) (τ : Claims.Transcript)
    (enc : Claims.Msg → List Bool),
    Disjoint Vd Vv → Implements R Vd Vv d τ enc →
      ((heard R Vv d).filter fun m => m.1 ∉ Vd).map Prod.snd = τ.msgs.map fun m => enc m.1

/-- **Registration is sound at its code's acceptance.** For the device at `Vd` and the verifier at `Vv` running as
written (`DeviceAsWritten`, `AsWritten`, the verifier's code `vf` registration's), where every transcript `vf` accepts
decodes by `regOf` to a registration that is valid and whose Site is complete (a premise about the code alone, which a
property of the code discharges at the executable's answer and its decoder), and where every feasible run whose
verifier heard a valid registration has its actual circuit in `𝒞` of it (a premise on the feasible runs): every
accepted run's transcript decodes to a valid registration whose set holds the actual circuit, at error `0`.
Registration draws no challenge, so the device is idle in it. -/
def RegistrationAccepted : Prop :=
  ∀ {L : Type*} [LinearOrder L] (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (Vd Vv : Finset L) (t₀ : ℕ)
    (y : L) (c d : ℕ) (vf : List (L × List Bool) → Bool) (vsend : List (L × List Bool) → L × ℕ × ℕ → Option (ℕ × ℕ))
    (D : Verity.Registration.Digests) (Aud : Verity.Registration.Auditor) (co : Verity.Registration.Composite)
    (regOf : List (L × List Bool) → Option Verity.Registration) (𝒞 : Verity.Registration → Set (Placed L)),
    DeviceAsWritten 𝓕 Vd t₀ → AsWritten 𝓕 Vv y c d vf vsend →
    (∀ τ, vf τ = true → ∃ reg, regOf τ = some reg ∧ Verity.Registration.Valid D Aud co reg ∧
      Verity.Registration.Complete D Aud (Verity.Site.ofLog reg)) →
    (∀ R ∈ 𝓕, ∀ reg, regOf (transcript R Vv d) = some reg → Verity.Registration.Valid D Aud co reg → R.A ∈ 𝒞 reg) →
    Sound Ω 𝓕 (Vd ∪ Vv) (Accepts y c d)
      (fun S r => ∃ reg, regOf (transcript (S r) Vv d) = some reg ∧ Verity.Registration.Valid D Aud co reg ∧
        (S r).A ∈ 𝒞 reg) 0

open Classical in
/-- **The proposed replacement of `ResourceAccounting.Sound`**, the feasible runs a challenge at a time: every strategy
causal from every tick whose run at each challenge `r` is in `𝓕 r`, and whose input gates off `V` don't depend on the
challenge, accepts with `¬E` for at most a `δ` fraction of the challenges. `Sound Ω 𝓕` is this at the constant family
`fun _ => 𝓕`. It is a draft's stand-in, not a second definition: the change that takes it edits `Sound` itself. -/
def ProposedSound (Ω : Type) [Fintype Ω] (𝓕 : Ω → Set (Actual.Run L)) (V : Finset L)
    (acc : Set (ℕ × L × ℕ × ℕ × ℕ) → Prop) (E : (Ω → Actual.Run L) → Ω → Prop) (δ : ℝ) : Prop :=
  ∀ S : Ω → Actual.Run L, (∀ r, S r ∈ 𝓕 r) → CausalFrom S → (∀ r r', (S r).inputs (↑V)ᶜ = (S r').inputs (↑V)ᶜ) →
    ((univ.filter fun r => acc ((S r).view V) ∧ ¬E S r).card : ℝ) ≤ δ * Fintype.card Ω

/-- **The device at `Vd` holds the challenge**: every feasible run at challenge `r` has `Vd`'s input gates holding
`chal r`. A premise on the feasible runs, stated only once they are taken a challenge at a time. -/
def Holds {Ω : Type} (𝓕 : Ω → Set (Actual.Run L)) (Vd : Finset L) (chal : Ω → List Bool) : Prop :=
  ∀ r, ∀ R ∈ 𝓕 r, challenge R Vd = chal r

/-- **Registration's instance of the proposed `Sound`**: `RegistrationAccepted` with the feasible runs a challenge at
a time, each premise on the feasible runs holding at every challenge. Registration draws no challenge, so it takes no
`Holds`; a protocol that draws one adds `Holds 𝓕 Vd chal`. -/
def RegistrationAcceptedFamily : Prop :=
  ∀ {L : Type*} [LinearOrder L] (Ω : Type) [Fintype Ω] (𝓕 : Ω → Set (Actual.Run L)) (Vd Vv : Finset L) (t₀ : ℕ)
    (y : L) (c d : ℕ) (vf : List (L × List Bool) → Bool) (vsend : List (L × List Bool) → L × ℕ × ℕ → Option (ℕ × ℕ))
    (D : Verity.Registration.Digests) (Aud : Verity.Registration.Auditor) (co : Verity.Registration.Composite)
    (regOf : List (L × List Bool) → Option Verity.Registration) (𝒞 : Verity.Registration → Set (Placed L)),
    (∀ r, DeviceAsWritten (𝓕 r) Vd t₀) → (∀ r, AsWritten (𝓕 r) Vv y c d vf vsend) →
    (∀ τ, vf τ = true → ∃ reg, regOf τ = some reg ∧ Verity.Registration.Valid D Aud co reg ∧
      Verity.Registration.Complete D Aud (Verity.Site.ofLog reg)) →
    (∀ r, ∀ R ∈ 𝓕 r, ∀ reg, regOf (transcript R Vv d) = some reg → Verity.Registration.Valid D Aud co reg →
      R.A ∈ 𝒞 reg) →
    ProposedSound Ω 𝓕 (Vd ∪ Vv) (Accepts y c d)
      (fun S r => ∃ reg, regOf (transcript (S r) Vv d) = some reg ∧ Verity.Registration.Valid D Aud co reg ∧
        (S r).A ∈ 𝒞 reg) 0

end Draft.Acceptance
