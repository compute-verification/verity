module

public import Definitions.Draft.ActualCircuit.ForcedMatmul
public import Definitions.Core.Site
public import Definitions.Registration.Physical

/-!
# Compute accounting's statement (draft)

Compute accounting calls forced matrix multiplication and computes one number from its result and the hardware
description: the node's free compute in the epoch, its registered rate times the epoch's length less the work forced
matrix multiplication credits (`freeNumber`). The registered rate is read off the registered Site (`registeredRate`,
`Verity.Site.device` priced at a price per operation class); the credited work is `Wb · Nmin`, which (a) establishes.

* `freeSet`: the actual circuits in which, if the node took in no help during the epoch, its work on the marking, what
  forced matrix multiplication does not credit, is at most a number.
* `AtRegisteredRate`: the hardware premise, that in every feasible run the node's work over any span is at most the
  registered rate times its length (`Placed.ComputeRate`).
* `ComputeAccountingEpoch`: the protocol's own statement, forced matrix multiplication's challenges and error with the
  free set in place of the forced set.
* `ComputeAccountingGiven`: what it adds to forced matrix multiplication, at error `0` and on no challenge of its own:
  every feasible actual circuit in the forced set lies in the free set. This is how it enters the composition, whose
  protocols draw independent challenges: compute accounting draws none beyond the ones it calls.
-/

@[expose] public section

namespace Draft.ActualCircuit

open ResourceAccounting Verity

variable {L : Type}

/-- **The registered rate** of the device `x` of the Site `S`: its device model (`Site.device`, at ticks of the Site's
unit of time, its tensor-core rate the larger of its sheet's and the auditor's estimate's) priced at `p`
(`Device.priced`), work a tick; `0` where the Site gives no device model. -/
def registeredRate (digest : Site.Sheet → String) (Aud : Registration.Auditor) (S : Site) (x : Site.Loc)
    (cuda oracle : ℕ) (p : Price) : ℕ :=
  ((Site.device (ι := Empty) digest Aud S x cuda oracle Empty.elim).map (·.priced p)).getD 0

/-- **The free number**: the rate `Φ` times the epoch's length `len`, less the credited work `W`; `0` when the credit
is the larger. -/
def freeNumber (Φ len W : ℕ) : ℕ := Φ * len - W

open Classical in
/-- **The free set**: the actual circuits in which, if the node `X` took in no help during the ticks `I`, its computed
gates during `I` on the marking `m` have work at most `n` at the pricing `π`. -/
def freeSet (V X : Finset L) (I : Set ℕ) (π : (A : Placed L) → Fin A.N → ℕ) (m : (A : Placed L) → Finset (Fin A.N))
    (n : ℕ) : Set (Placed L) :=
  {A | A.ingress (X ∪ V) I = 0 → A.work (π A) (A.region X I ∩ m A) ≤ n}

/-- **The node computes at most at the registered rate**, in every feasible run: over any span `[a, b)` its work at the
pricing `π` is at most `Φ (b − a)` (`Placed.ComputeRate` at the one location `X`). A hardware premise, the device's true
peak capacity bounded by the registered figure. Nothing else ties `X` to the Site's device or the per-gate `π` to the
per-class price `registeredRate` reads. -/
def AtRegisteredRate (𝓕 : Set (Actual.Run L)) (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ) (Φ : ℕ) : Prop :=
  ∀ R ∈ 𝓕, R.A.ComputeRate {X} (π R.A) fun _ => Φ

/-- **Compute accounting in one epoch.** If the verifier accepts in the epoch `[P.ts, P.td)` then, except with
probability `δdet + δluck + δskip` over forced matrix multiplication's challenges, the actual circuit lies in the free
set at the number the verifier computes, the registered rate of `x` on the registered Site times the epoch's length
less `Wb · Nmin`: had the node taken in no help during the epoch, its work forced matrix multiplication does not credit
is at most that number. Premises: (a)'s, and that the node computes at most at the registered rate
(`AtRegisteredRate`). -/
def ComputeAccountingEpoch : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (L : Type) [LinearOrder L] (P : PouwWork.Verdict Ω L) (𝓕 : Set (Actual.Run L))
    (acc : Transcript L → Prop) (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ)
    (m : (A : Placed L) → Finset (Fin A.N)) (Wb Nmin ρ : ℕ) (δdet δluck δskip : ℝ)
    (digest : Site.Sheet → String) (Aud : Registration.Auditor) (reg : Registration) (x : Site.Loc)
    (cuda oracle : ℕ) (p : Price),
    AcceptIsVerdict 𝓕 P acc →
    ForStrategies Ω 𝓕 P.V (fun S => Assumptions.PouwDetection S P Nmin δdet ∧ Assumptions.PouwLuck S P Nmin δluck ∧
      Assumptions.PouwFloorOff S P X (fun r => π (S r).A) (fun r => m (S r).A) Wb Nmin ρ δskip) →
    AtRegisteredRate 𝓕 X π (registeredRate digest Aud (Site.ofLog reg) x cuda oracle p) →
    Sound Ω 𝓕 P.V acc
      (Lies P.V fun _ => freeSet P.V X (Set.Ico P.ts P.td) π m
        (freeNumber (registeredRate digest Aud (Site.ofLog reg) x cuda oracle p) (P.td - P.ts) (Wb * Nmin)))
      (δdet + δluck + δskip)

/-- **What compute accounting adds to forced matrix multiplication.** For any challenges and any acceptance, except
with probability `0` the actual circuit lies outside the forced set at `W` or inside the free set at
`Φ · (td − ts) − W`, given that the node computes at most at the rate `Φ` (`AtRegisteredRate`). Deterministic: it is
the inclusion of the forced set in the free set over the feasible actual circuits, the work during the epoch split into
the marking and the rest. -/
def ComputeAccountingGiven : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (L : Type) [LinearOrder L] (𝓕 : Set (Actual.Run L)) (V X : Finset L) (ts td : ℕ)
    (π : (A : Placed L) → Fin A.N → ℕ) (m : (A : Placed L) → Finset (Fin A.N)) (Φ W : ℕ) (acc : Transcript L → Prop),
    AtRegisteredRate 𝓕 X π Φ →
    Sound Ω 𝓕 V acc
      (Lies V fun _ => (forcedSet V X (Set.Ico ts td) π m W)ᶜ ∪ freeSet V X (Set.Ico ts td) π m (freeNumber Φ (td - ts) W))
      0

end Draft.ActualCircuit
