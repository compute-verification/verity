module

public import Definitions.Draft.ActualCircuit.ForcedMatmul
public import Definitions.Core.Site
public import Definitions.Registration.Physical

/-!
# Compute accounting's statement (draft)

Compute accounting calls forced matrix multiplication and computes one number from its result and the hardware
description: the registered rate (`registeredRate`, `Verity.Site.device` priced at a price per operation class) at the
device `x` the prover's one message names for `X`, times the epoch's length, plus `ρ` times network accounting's bound
on the node's ingress, less the credited work, forced matrix multiplication's result message, which forced matrix
multiplication must establish at that number. That number is `ResourceAccounting.Guarantees.Fit`'s residual at
`cap = obs`: a bound on what the node's work could implement beside the forced work, `req_μ`, read off the actual
circuit.

* `AtRegisteredRate`: the hardware premise, that in every feasible run the node's work over any span is at most the
  registered rate of the device the run's view names times its length (`Placed.ComputeRate`).
* `epochUse`, `creditedRuns`: `Fit`'s measure and set of runs for one epoch.
* `ComputeAccountingEpoch`: the protocol's statement: from forced matrix multiplication's statement at the credited
  message, the same challenges and error with `Fit`'s conclusion in its place.

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

/-- **The node computes at most at the registered rate of the device its view names**, in every feasible run: over any
span `[a, b)` its work at the pricing `π` is at most `Φ v (b − a)`, `v` the run's view at `V` (`Placed.ComputeRate` at
the one location `X`). A hardware premise, the device's true peak capacity bounded by the registered figure. Nothing
ties the per-gate `π` to the per-class price `registeredRate` reads but this premise. -/
def AtRegisteredRate (𝓕 : Set (Actual.Run L)) (V X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ)
    (Φ : VerifierView L → ℕ) : Prop :=
  ∀ R ∈ 𝓕, R.A.ComputeRate {X} (π R.A) fun _ => Φ (R.view V)

open Classical in
/-- **`Fit`'s measure in one epoch**: the node's work during the epoch at `π`, the marking included, plus `ρ` times
what it took in. -/
noncomputable def epochUse {Ω : Type} (P : Verdict Ω L) (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ) (ρ : ℕ)
    (R : Actual.Run L) : ℕ :=
  R.A.work (π R.A) (R.A.region X (Set.Ico P.ts P.td)) + ρ * R.A.ingress (X ∪ P.V) (Set.Ico P.ts P.td)

/-- **`Fit`'s runs in one epoch, at the device `d` and the credit `c`**: the feasible runs whose view names the device
`d` and in which the node's credits outside the marking plus `ρ` times its help are at least `c` (forced matrix
multiplication's conclusion) and its help is at most `h` (network accounting's), `credits` and `help` read on the one
run (the strategy constant at it). -/
def creditedRuns {Ω : Type} (𝓕 : Set (Actual.Run L)) (P : Verdict Ω L) (X : Finset L)
    (π : (A : Placed L) → Fin A.N → ℕ) (m : (A : Placed L) → Finset (Fin A.N)) (dev : VerifierView L → Site.Loc)
    (d : Site.Loc) (c ρ h : ℕ) : Set (Actual.Run L) :=
  {R | R ∈ 𝓕 ∧ dev (R.view P.V) = d ∧ ∀ r : Ω,
    c ≤ credits (fun _ : Ω => R) P X (fun _ => π R.A) (fun _ => m R.A) r + ρ * help (fun _ : Ω => R) P X r ∧
    help (fun _ : Ω => R) P X r ≤ h}

/-- **Compute accounting in one epoch.** It bounds what the node's work could implement beside the work credited
outside the marking `m`, and `Additive` is claimed only at a marking at which the new A3 (`PouwFloorAt`) holds; at the
empty marking `Additive` fails unless the program costs nothing, and the statement says nothing.

The verifier reads two messages from its view: the device `dev v` the prover names for `X`, and the credited work
`credit v`, forced matrix multiplication's result. Given forced matrix multiplication's statement at that credit
(accepted at a view holding the challenge, the node's credits plus `ρ` times its help are at least `credit v`, except
`δ`; (a) is it at `credit = Wb · Nmin`, `δ = δdet + δskip`), if the verifier accepts then, except `δ`, at a view
holding the challenge, if the node's help was at most `h` and the run implements `impl`, then
`req_μ(impl) ≤ Φ(dev v) · (td − ts) + ρ · h − credit v` at `μ = epochUse`: `Fit`'s conclusion at `cap = obs`. Premises:
that statement, the hardware premise at the named device (`AtRegisteredRate`) and `Fit`'s additivity over
`creditedRuns` at every device and credit.

Not vacuous because its conclusion fails for the verifier that accepts every view against the strategy that holds the
challenge, sends the credit `1` and runs a program costing more than `Φ · (td − ts) + ρ · h − 1` with no forced work,
where the premise that fails is forced matrix multiplication's statement at that credit. -/
def ComputeAccountingEpoch : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (L : Type) [LinearOrder L] (P : Verdict Ω L) (𝓕 : Set (Actual.Run L))
    (rd : Readout Ω L) (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ)
    (m : (A : Placed L) → Finset (Fin A.N)) (ρ h : ℕ) (δ : ℝ) (credit : VerifierView L → ℕ)
    (dev : VerifierView L → Site.Loc) (digest : Site.Sheet → String) (Aud : Registration.Auditor)
    (reg : Registration) (cuda oracle : ℕ) (p : Price) (impl : Actual.Run L → Prop),
    Sound Ω 𝓕 P.V (acceptsOn P rd)
        (fun S r => HoldsChallenge rd P.V S r → credit ((S r).view P.V) ≤
          credits S P X (fun r => π (S r).A) (fun r => m (S r).A) r + ρ * help S P X r) δ →
      AtRegisteredRate 𝓕 P.V X π (fun v => registeredRate digest Aud (Site.ofLog reg) (dev v) cuda oracle p) →
      (∀ d c, Additive (creditedRuns (Ω := Ω) 𝓕 P X π m dev d c ρ h) (epochUse P X π ρ) c
        (req (epochUse P X π ρ) 𝓕 impl) impl) →
      Sound Ω 𝓕 P.V (acceptsOn P rd)
        (fun S r => HoldsChallenge rd P.V S r → help S P X r ≤ h → impl (S r) →
          req (epochUse P X π ρ) 𝓕 impl ≤
            ((registeredRate digest Aud (Site.ofLog reg) (dev ((S r).view P.V)) cuda oracle p * (P.td - P.ts) +
              ρ * h - credit ((S r).view P.V) : ℕ) : ℕ∞))
        δ

end Draft.ActualCircuit
