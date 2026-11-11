module

public import Definitions.WeightSecurity.Assumptions

/-!
# Weight security's model claim under faults

Weight security's bound under SPOT's fault model (`TheftFaulted`) rests on `FaultedUseHard`: setting at most `M`
faults in the served circuit, beside honest queries, hands out the weights at most `N^(λ - 1)` times faster than honest
use, `N` the number of fault pattern sequences of total size at most `M` (`patternCount`), so `log₂ N` is the faults'
arbitrary-data capacity and the factor is `2^((λ - 1) · log₂ N)`. It is about the served function alone, as the
honest-use rate it multiplies (`HonestUseHard`) is. It is a named assumption under `Definitions.Assumptions`, in a
module of its own that reads only weight security's definitions, so the area's readers import it without the rest of
the named assumptions.
-/

@[expose] public section

namespace WeightSecurity.Assumptions

open WeightSecurity

variable {Ω R X Y Θ Θ' F : Type}

/-- **Faults hand out the weights at most `N^(λ - 1)` times faster than honest use.** `Cf θ (x, φ)` is the served
function `C`'s circuit run at the weights `θ` and the input `x` with the proof blocks the fault pattern `φ` names
writing `φ`'s values instead, and `Af` a class of parties that query it. Wherever every party of `A` steals the weights
from `q` honest queries with probability at most `p`, every party of `Af` that sets at most `M` faults over its `q`
queries (`FaultBudget`) steals them with probability at most `N^(λ - 1) · p`, `N = patternCount size q M`: that is,
`2^((λ - 1) · C_arb)` with `C_arb = log₂ N` the faults' arbitrary-data capacity. At `λ = 1` faults are worth no more
than honest use; at `λ` large they are worth as much as `λ · C_arb` free covert bits.

It is a claim about the model, not about Verity: whether a server that may corrupt `M` proof blocks' outputs, each
within its format, learns to steal the weights faster than by asking. With `Stolen` restricted to a typical set of
weights (`θ ∈ Typ ∧ Stolen θ θ'`), it need hold only there. -/
def FaultedUseHard [Fintype Ω] [Fintype R] [Fintype F] (C : Θ → X → Y) (Cf : Θ → X × F → Y) (size : F → ℕ)
    (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop) (q M : ℕ) (A : Set (Outsider R X Y Θ'))
    (Af : Set (Outsider R (X × F) Y Θ')) (lam : ℝ) : Prop :=
  ∀ p, HonestUseHard C gen Stolen q A p →
    HonestUseHard Cf gen Stolen q (Af ∩ FaultBudget size q M) ((patternCount size q M : ℝ) ^ (lam - 1) * p)

end WeightSecurity.Assumptions
