import Definitions.NetworkCertifier.AdviceString
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# The advice string's named assumptions: the charge, and the compiler

The leak bound (`Guarantees.AdviceString`) takes three named assumptions. One is about the charge, `bits`, which
Python's `bits` must meet. Two are about the registered compiler: it is precommitted, and the advice is its only free
input.
-/

namespace NetTiming

variable {Adv : Type}

/-- **`bits` counts.** At most `2^c` advice values are charged `c` bits or fewer over all windows. It holds when the
charges sum to at least the length of a prefix-free encoding of the advice (`bitsCount_of_prefixFree`), or to at least
the length plus one of any injective encoding (`bitsCount_of_lengthSucc`). Advice v1's charge meets it under any
registered book (`AdviceV1.bitsCount`, from `Guarantees.EncardChargedLe`). -/
def BitsCount {W : ℕ} (bits : Adv → Fin W → ℕ) : Prop :=
  ∀ c : ℕ, {a | ∑ w, bits a w ≤ c}.encard ≤ ((2 ^ c : ℕ) : ℕ∞)

/-- **The compiler is deterministic, and the advice is its only free input**: its frames depend on the advice and the
certified ingress, never on its other inputs `Ω` (a clock, coins, or state of the run inside X). -/
def CompilerDeterministic {L : ℕ} {In Ω : Type} (C : Compiler L In Adv Ω) : Prop :=
  ∀ a ω ω', C a ω = C a ω'

/-- **The compiler is precommitted**: it is registered before the run, so the compiler of the run with secret `z` is
the same for every secret. What the registration itself carries is charged separately, once. -/
def CompilerPrecommitted {Z : Type} {L : ℕ} {In Ω : Type} (compilerOf : Z → Compiler L In Adv Ω) : Prop :=
  ∀ z z', compilerOf z = compilerOf z'

end NetTiming
