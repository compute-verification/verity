module

public import Definitions.Pouw.PouwWork
public import Definitions.Pouw.Basic.Prob

/-! The named premises of PoUW's work lower bound on one node, besides its open floor conjecture. -/

@[expose] public section

namespace ResourceAccounting.Assumptions

open ResourceAccounting.PouwWork

variable {Ω : Type} [Fintype Ω] {L : Type*} (S : Ω → Actual.Run L) (P : Verdict Ω L)

/-- **A2 Detection** (the live-coin sample; `Pouw.Guarantees.WorkWeightedSampling` for a ticket audit): the verifier
accepts with fewer than `Nmin` blocks right at the coin with probability at most `δ`. For one exact live-coin opening it
is a theorem in the converged model (`detection_exact`, at `δ = (Nmin − 1)/N`), not yet ported. -/
def PouwDetection (Nmin : ℕ) (δ : ℝ) : Prop :=
  Pouw.pr (fun r => Accepts S P r ∧ good S P r < Nmin) ≤ δ

/-- **A2b Luck** (the targets' unpredictability): the verifier accepts with `Nmin` blocks right at the coin but fewer
than `Nmin` right at every coin with probability at most `δ`. A computation right only at some coins is right there by
luck. -/
def PouwLuck (Nmin : ℕ) (δ : ℝ) : Prop :=
  Pouw.pr (fun r => Accepts S P r ∧ Nmin ≤ good S P r ∧ goodAll S P < Nmin) ≤ δ

/-- **A5 HelpBound** (the network attesters' signed ingress for the window, under `trusted-device`): at every coin
the verifier accepts, the node took in at most `H` bits of help in `[ts, td)` (`help`). At `H = 0` the node is isolated:
nothing placed off the node and the verifier reaches either of them during the window. -/
def PouwHelpBound (X : Finset L) (H : ℕ) : Prop :=
  ∀ r, Accepts S P r → help S P X r ≤ H

end ResourceAccounting.Assumptions
