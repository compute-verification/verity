import Definitions.Pouw.Neekon

/-!
# Assumption H: Dither's transcript costs the honest work (trusted)

The premise Theorem B (`Models.Pouw.Guarantees.Neekon`) rests on, stated about the machine and the problem, not the
protocol: it reads only `Mc`, the draw and the transcript.
-/

namespace Pouw.Neekon.Assumptions

open Pouw.RelativeSecurity Pouw.Neekon

/-- `pouw-neekon-h` (proposed). **H: producing Dither's transcript costs the honest work.** For every instance
`(A*, B*)` and every `M₂` of `Mc` from σ and its own coins to a transcript, if `M₂` outputs the correct transcript of
`(A* + R)(B* + S)` (every 32-step prefix sum over `Z/2³²`) with probability at least `δ` over σ and its coins, then
`time(M₂) ≥ (1 − εH) · tHonest`. `M₂` may depend on the instance in any way (`M₁`'s state is in its program, charged as
its size), and it may be any algorithm of `Mc`: no class restriction.

Rationale (note:20261017T0013Z-report-epsilon-program): the binding instance is plausibly the easiest, `A* = B* = 0`,
whose transcript is the prefix sums of the noise product `R·S` alone. Theorem K proves a lower bound of this kind,
`L` products per certified cell-block, for the straight-line runs of its two scopes in the product count
(`StagedFloorKOneSided`'s `KScopeOneSided`); outside them it fails (Strassen with an indicator, NEEKON-FORM's 275-product
ball-exact circuit), so H is not a consequence of K. `tHonest` is the honest transcript's time; H-FALSIFY's runs on
sm_120 (beat the dense int8 kernel on `R·S`) are its falsification attempts. -/
def DitherH (Mc : Machine) (m K n : ℕ) (εH δ tHonest : ℝ) : Prop :=
  ∀ (x : DInst m K n) (Ω : Type) [Fintype Ω] (M₂ : Mc.Alg (DSeed m K n × Ω) (DTrans m K n)),
    δ ≤ prob (fun ω : DSeed m K n × Ω => Mc.eval M₂ ω = transcript ω.1 x) →
    (1 - εH) * tHonest ≤ Mc.time M₂

end Pouw.Neekon.Assumptions
