module

public import Definitions.Core.Prob
public import Mathlib.Probability.Distributions.Uniform

/-!
# The period's limit on untied words: a premise

The charged count with untied words (`Slowdown.Cut.Guarantees.HEChargedUntied`) bounds what a monitoring period's
blocks read once its untied words are counted as fresh input, at a limit `U` on how many words are untied. That limit
is a sampling statement about PoUW over the period, and this premise states it on the period's coins.
-/

@[expose] public section

open Verity.Game renaming prPMFReal → prob

namespace Slowdown.Assumptions

/-- **A period's untied words are at most `U`, except with probability `ε`.** Over a monitoring period's coins `Ω`,
uniform, the period is accepted (`accepted x`) while more than `U` words of its circuit are untied (`untied x`: the
words whose values nothing ties to registered data) with probability at most `ε`.

It is what PoUW's period bound on untied cells gives for one sequence of execution phases,
`Properties.Pouw.PearlPlus.PeriodUntiedCellsAtW`: `ε = δ + q / 2^256` at `(1 − ε_s)^t ≤ δ` over `t` tickets a phase,
each phase committed before its coin, under SHA-512's collision resistance at the run's finder of `q` evaluations; and
`U` at least `Runs.cellsUpTo` at `ε_s` on every outcome, once each untied credited cell is one untied word of the
circuit, its 32 output bits. Neither that bound nor the step from its cells to the circuit's words is proved. -/
def PeriodUntiedAtMost {Ω : Type} [Fintype Ω] [Nonempty Ω] {N : ℕ} (accepted : Ω → Prop)
    (untied : Ω → Finset (Finset (Fin N))) (U : ℕ) (ε : ℝ) : Prop :=
  prob (PMF.uniformOfFintype Ω) (fun x => accepted x ∧ U < (untied x).card) ≤ ε

end Slowdown.Assumptions
