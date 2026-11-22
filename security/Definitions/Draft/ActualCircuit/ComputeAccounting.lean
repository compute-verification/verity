module

public import Definitions.Draft.ActualCircuit.ForcedMatmul
public import Definitions.Core.Site
public import Definitions.Registration.Physical

/-!
# Compute accounting's statement (draft)

Compute accounting calls forced matrix multiplication and computes one number from its result and the hardware
description: the registered rate (`registeredRate`, `Verity.Site.device` priced at a price per operation class) times
the epoch's length, plus `ρ` times network accounting's bound on the node's ingress, less the work forced matrix
multiplication credits, `Wb · Nmin`. That number is `ResourceAccounting.Guarantees.Fit`'s residual at `cap = obs`:
a bound on what the node's work could implement beside the forced work, `req_μ`, read off the actual circuit.

* `AtRegisteredRate`: the hardware premise, that in every feasible run the node's work over any span is at most the
  registered rate times its length (`Placed.ComputeRate`).
* `epochUse`, `creditedRuns`: `Fit`'s measure and set of runs for one epoch.
* `ComputeAccountingEpoch`: the protocol's statement, (a)'s challenges and error with `Fit`'s conclusion in place of
  (a)'s.

The bound on ingress, `h`, is network accounting's own statement; this statement reads it as a hypothesis on the run,
met by intersection with network accounting's set, and does not state it.
-/

@[expose] public section

namespace Draft.ActualCircuit

open ResourceAccounting ResourceAccounting.PouwWork Verity

variable {L : Type}

/-- **The registered rate** of the device `x` of the Site `S`: its device model (`Site.device`, at ticks of the Site's
unit of time, its tensor-core rate the larger of its sheet's and the auditor's estimate's) priced at `p`
(`Device.priced`), work a tick; `0` where the Site gives no device model. -/
def registeredRate (digest : Site.Sheet → String) (Aud : Registration.Auditor) (S : Site) (x : Site.Loc)
    (cuda oracle : ℕ) (p : Price) : ℕ :=
  ((Site.device (ι := Empty) digest Aud S x cuda oracle Empty.elim).map (·.priced p)).getD 0

/-- **The node computes at most at the registered rate**, in every feasible run: over any span `[a, b)` its work at the
pricing `π` is at most `Φ (b − a)` (`Placed.ComputeRate` at the one location `X`). A hardware premise, the device's true
peak capacity bounded by the registered figure. Nothing else ties `X` to the Site's device or the per-gate `π` to the
per-class price `registeredRate` reads. -/
def AtRegisteredRate (𝓕 : Set (Actual.Run L)) (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ) (Φ : ℕ) : Prop :=
  ∀ R ∈ 𝓕, R.A.ComputeRate {X} (π R.A) fun _ => Φ

open Classical in
/-- **`Fit`'s measure in one epoch**: the node's work during the epoch at `π`, the marking included, plus `ρ` times
what it took in. -/
noncomputable def epochUse {Ω : Type} (P : Verdict Ω L) (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ) (ρ : ℕ)
    (R : Actual.Run L) : ℕ :=
  R.A.work (π R.A) (R.A.region X (Set.Ico P.ts P.td)) + ρ * R.A.ingress (X ∪ P.V) (Set.Ico P.ts P.td)

/-- **`Fit`'s runs in one epoch**: the feasible runs in which the node's credits outside the marking plus `ρ` times its
help are at least `Wb · Nmin` ((a)'s conclusion) and its help is at most `h` (network accounting's), `credits` and `help`
read on the one run (the strategy constant at it). -/
def creditedRuns {Ω : Type} (𝓕 : Set (Actual.Run L)) (P : Verdict Ω L) (X : Finset L)
    (π : (A : Placed L) → Fin A.N → ℕ) (m : (A : Placed L) → Finset (Fin A.N)) (Wb Nmin ρ h : ℕ) :
    Set (Actual.Run L) :=
  {R | R ∈ 𝓕 ∧ ∀ r : Ω,
    Wb * Nmin ≤ credits (fun _ : Ω => R) P X (fun _ => π R.A) (fun _ => m R.A) r + ρ * help (fun _ : Ω => R) P X r ∧
    help (fun _ : Ω => R) P X r ≤ h}

/-- **Compute accounting in one epoch.** It bounds what the node's work could implement beside the work credited
outside the marking `m`, and `Additive` is claimed only at a marking at which the new A3 (`PouwFloorAt`) holds; at the
empty marking `Additive` fails unless the program costs nothing, and the statement says nothing.

If the verifier accepts in the epoch `[P.ts, P.td)` then, except with probability `δdet + δskip` over forced matrix
multiplication's challenges, at a view holding the challenge, if the node's help was at most `h` and the run implements
`impl`, then `req_μ(impl) ≤ Φ · (td − ts) + ρ · h − Wb · Nmin` at `μ = epochUse` and `Φ` the registered rate of `x`:
`Fit`'s conclusion at `cap = obs`. Premises: (a)'s, the hardware premise (`AtRegisteredRate`) and `Fit`'s additivity
over `creditedRuns`.

Not vacuous because its conclusion fails for the verifier that accepts every view against the strategy that holds the
challenge and runs a program costing more than `Φ · (td − ts) + ρ · h` less `Wb · Nmin` with no forced work, where the
premise that fails is `PouwDetectionAt`. -/
def ComputeAccountingEpoch : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (L : Type) [LinearOrder L] (P : Verdict Ω L) (𝓕 : Set (Actual.Run L))
    (rd : Readout Ω L) (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ)
    (m : (A : Placed L) → Finset (Fin A.N)) (Wb Nmin ρ h : ℕ) (δdet δskip : ℝ)
    (digest : Site.Sheet → String) (Aud : Registration.Auditor) (reg : Registration) (x : Site.Loc)
    (cuda oracle : ℕ) (p : Price) (impl : Actual.Run L → Prop),
    PouwDetectionAt 𝓕 P rd Nmin δdet → PouwFloorAt 𝓕 P rd X π m Wb ρ δskip →
    AtRegisteredRate 𝓕 X π (registeredRate digest Aud (Site.ofLog reg) x cuda oracle p) →
    Additive (creditedRuns (Ω := Ω) 𝓕 P X π m Wb Nmin ρ h) (epochUse P X π ρ) (Wb * Nmin)
      (req (epochUse P X π ρ) 𝓕 impl) impl →
    Sound Ω 𝓕 P.V (acceptsOn P rd)
      (fun S r => HoldsChallenge rd P.V S r → help S P X r ≤ h → impl (S r) →
        req (epochUse P X π ρ) 𝓕 impl ≤
          ((registeredRate digest Aud (Site.ofLog reg) x cuda oracle p * (P.td - P.ts) + ρ * h - Wb * Nmin : ℕ) : ℕ∞))
      (δdet + δskip)

end Draft.ActualCircuit
