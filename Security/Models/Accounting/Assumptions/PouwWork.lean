import Definitions.Accounting.PouwWork

/-!
# PoUW's named premises on one node (trusted)

The premises of CMP-12's converged model (v3.1) that its PoUW work lower bound takes, over `Accounting.PouwWork`'s
vocabulary on the actual circuit. Each is a `Prop` about one strategy `S` and the verifier `P`, taken as a hypothesis by
`Accounting.Guarantees.PouwLocalOff`, never an `axiom`. Their v3.1 names are A2 (`Detection`), A2b (`Luck`), A3
(`FloorOff`) and A5 (`HelpBound`).
-/

namespace Accounting.Assumptions

open Accounting.PouwWork

variable {Ω : Type} [Fintype Ω] {L : Type*} (S : Ω → Actual.Run L) (P : Verdict Ω L)

/-- **A2 Detection** (the live-coin draw; `Pouw.Guarantees.WorkWeightedSampling` for a ticket audit): the verifier
accepts with fewer than `Nmin` blocks right at the coin with probability at most `δ`. For one exact live-coin opening it
is a theorem in the converged model (`detection_exact`, at `δ = (Nmin − 1)/N`), not yet ported. -/
def PouwDetection (Nmin : ℕ) (δ : ℝ) : Prop :=
  Pouw.pr (fun r => Accepts S P r ∧ good S P r < Nmin) ≤ δ

/-- **A2b Luck** (the targets' unpredictability): the verifier accepts with `Nmin` blocks right at the coin but fewer
than `Nmin` right at every coin with probability at most `δ`. A computation right only at some coins is right there by
luck. -/
def PouwLuck (Nmin : ℕ) (δ : ℝ) : Prop :=
  Pouw.pr (fun r => Accepts S P r ∧ Nmin ≤ good S P r ∧ goodAll S P < Nmin) ≤ δ

/-- **A3 the work floor, off a skip event** (v3.1's `FloorOff`; on paper at `L = 32` given a merge price, conjectured at
served η): for the marking `m`, some event `Z` of probability at most `δ` is such that at every coin outside `Z`, `Nmin`
blocks right at every coin force the node's credits outside `m`, plus `ρ` a bit of help, to at least `Wb · Nmin`.

`Z` covers a guarded circuit that is right at every coin yet skips credited gates at some. The floor is meant at a
removable marking: one whose erasure leaves the verdict and the right blocks unchanged at every coin (the converged
model's `Removable`), so that covert work earns no credit. The actual-circuit model has no erasure of gates, so the
marking is taken as given, and the premise is claimed only at markings that are removable. A factor-2 projection form
(`#unmixed + 2·#mixed ≥ floor`) is being proved as a bound that implies this one. -/
def PouwFloorOff (X : Finset L) (π : ∀ r, Fin (S r).A.N → ℕ) (m : ∀ r, Finset (Fin (S r).A.N))
    (Wb Nmin ρ : ℕ) (δ : ℝ) : Prop :=
  ∃ Z : Ω → Prop, Pouw.pr Z ≤ δ ∧
    ∀ r, ¬ Z r → Nmin ≤ goodAll S P → Wb * Nmin ≤ credits S P X π m r + ρ * help S P X r

/-- **A5 HelpBound** (the network certifiers' signed ingress for the window, under `trusted-nebius-host`): at every coin
the verifier accepts, the node took in at most `H` bits of help in `[ts, td)` (`help`). At `H = 0` the node is isolated:
nothing placed off the node and the verifier reaches either of them during the window. -/
def PouwHelpBound (X : Finset L) (H : ℕ) : Prop :=
  ∀ r, Accepts S P r → help S P X r ≤ H

end Accounting.Assumptions
