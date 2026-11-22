module

public import Definitions.Pouw.Confidence
public import Definitions.Core.Site

/-!
# Acceptance as the verifier's code on the transcript (draft)

`ResourceAccounting.Sound` reads acceptance as a predicate `acc` of the verifier's view. Here `acc` is the verdict the
verifier's code writes, and the premise that ties the verdict to the transcript is stated once for every protocol:

* `Actual.Run.heard`: the messages that crossed into the verifier's gates `V` before the deadline `d`, in order of
  arrival tick, reader and source, each the source gate's value at its width. It is read off the run, not the view:
  what crossed is physics, not the verifier's account of it.
* `Actual.Run.challenge`: the verifier's input gates' values, in order of position, as one message.
* `Actual.Run.transcript`: the challenge, then what the verifier heard.
* `Accepts`: the verifier's view holds the verdict bit `1` at its decision cell at tick `d`.
* `AsWritten`, a premise on the feasible runs: in every feasible run the verdict cell holds the code's answer `vf` on
  the transcript, and the input gates hold a challenge (an element of `Chal`) and nothing else.
* `RegistrationAccepted`: `Sound` for registration at that acceptance, with registration's code as `vf` and its
  model's acceptance (`Verity.Registration.Valid`, `Complete`) reached through a premise about the code alone.

`Sound`'s feasible set does not depend on the challenge, so `AsWritten` says the inputs hold some challenge and cannot
say they hold the strategy's own `r`.
-/

@[expose] public section

namespace Draft.Acceptance

open ResourceAccounting Finset

variable {L : Type*} [LinearOrder L]

open Classical in
/-- **What the verifier at `V` heard before `d`**: each wire whose reader is a gate of `V` placed before `d` and whose
source is placed off `V`, in order of the reader's tick, the reader's leaf and the source's position, as the source's
value at its width. A wire read by two gates of `V` is heard twice. -/
noncomputable def Actual.Run.heard (R : Actual.Run L) (V : Finset L) (d : ℕ) : List (List Bool) :=
  ((((univ ×ˢ univ).filter fun p : Fin R.A.N × Fin R.A.N =>
      p.2 ∈ R.A.reads p.1 ∧ R.A.x p.2 ∉ V ∧ R.A.x p.1 ∈ V ∧ R.A.t p.1 < d).image
    fun p => toLex (R.A.t p.1, toLex (R.A.x p.1, Actual.position (R.seen p.2)))).sort).map
    fun k => Actual.chunk (ofLex (ofLex k).2).2

open Classical in
/-- **The challenge the verifier at `V` holds**: its input gates' values, in order of position, as one bit string. -/
noncomputable def Actual.Run.challenge (R : Actual.Run L) (V : Finset L) : List Bool :=
  (((univ.filter fun g => R.A.isInput g = true ∧ R.A.x g ∈ V).image fun g => Actual.position (R.seen g)).sort).flatMap
    Actual.chunk

/-- **The transcript of the verifier at `V` up to its deadline `d`**: its challenge, then the messages it heard. -/
noncomputable def Actual.Run.transcript (R : Actual.Run L) (V : Finset L) (d : ℕ) : List (List Bool) :=
  R.challenge V :: R.heard V d

/-- **The verifier accepts**: its view holds the bit `1` at its decision cell, cell `c` of leaf `y`, at tick `d`. -/
def Accepts (y : L) (c d : ℕ) (v : Set (ℕ × L × ℕ × ℕ × ℕ)) : Prop := (d, y, c, 1, 1) ∈ v

/-- **The verifier at `V` runs its code `vf` as written**, in every feasible run: the decision cell `c` of `y ∈ V`
takes a one-bit gate at tick `d`, which holds `1` exactly when `vf` accepts the transcript; and the input gates of `V`
hold a challenge of `Chal` and nothing else. A premise on the feasible runs, the verifier's device's: it does not say
the challenge is the strategy's `r`, nor what the verifier sends. -/
def AsWritten (𝓕 : Set (Actual.Run L)) (V : Finset L) (y : L) (c d : ℕ) (vf : List (List Bool) → Bool)
    (Chal : Set (List Bool)) : Prop :=
  y ∈ V ∧ ∀ R ∈ 𝓕,
    (∃ g, R.A.x g = y ∧ R.A.cell g = c ∧ R.A.t g = d) ∧
    (∀ g, R.A.x g = y → R.A.cell g = c → R.A.t g = d →
      R.A.w g = 1 ∧ ((R.val g).toNat = 1 ↔ vf (R.transcript V d) = true)) ∧
    R.challenge V ∈ Chal

/-- **Registration is sound at its code's acceptance.** For the verifier at `V` running registration's code `vf` as
written (`AsWritten`, the challenge empty, as registration draws none), where every transcript `vf` accepts decodes by
`regOf` to a registration that is valid and whose Site is complete (a premise about the code alone, which a property of
the code discharges at the executable's answer and its decoder), and where every feasible run whose verifier heard a
valid registration has its actual circuit in `𝒞` of it (a premise on the feasible runs): every accepted run's
transcript decodes to a valid registration whose set holds the actual circuit, at error `0`. Registration draws no
challenge, so the union over challenges is not at issue here; for a protocol that draws one, `AsWritten` does not bind
the challenge held to the strategy's `r`. -/
def RegistrationAccepted : Prop :=
  ∀ {L : Type*} [LinearOrder L] (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V : Finset L) (y : L) (c d : ℕ)
    (vf : List (List Bool) → Bool) (D : Verity.Registration.Digests) (Aud : Verity.Registration.Auditor)
    (co : Verity.Registration.Composite) (regOf : List (List Bool) → Option Verity.Registration)
    (𝒞 : Verity.Registration → Set (Placed L)),
    AsWritten 𝓕 V y c d vf {[]} →
    (∀ τ, vf τ = true → ∃ reg, regOf τ = some reg ∧ Verity.Registration.Valid D Aud co reg ∧
      Verity.Registration.Complete D Aud (Verity.Site.ofLog reg)) →
    (∀ R ∈ 𝓕, ∀ reg, regOf (R.transcript V d) = some reg → Verity.Registration.Valid D Aud co reg → R.A ∈ 𝒞 reg) →
    Sound Ω 𝓕 V (Accepts y c d)
      (fun S r => ∃ reg, regOf ((S r).transcript V d) = some reg ∧ Verity.Registration.Valid D Aud co reg ∧
        (S r).A ∈ 𝒞 reg) 0

end Draft.Acceptance
