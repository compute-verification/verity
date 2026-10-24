import Definitions.NetworkCertifier.AdviceString
import Mathlib.Data.Set.Card

/-!
# The advice string's named assumptions: the codec, and the compiler

The leak bound (`Guarantees.AdviceString`) takes four named assumptions. Two are about the codec: Python's `encode`
and `bits` must meet them. Two are about the registered compiler: it is precommitted, and the advice is its only free
input.
-/

namespace NetTiming

variable {Adv : Type} {W : ℕ}

/-- **The segments decode** (`decode(encode(a)) = a`): two advice values with the same segment in every window are
equal. -/
def SegmentsDecode (code : AdviceCodec Adv W) : Prop := Function.Injective code.seg

/-- **`bits` counts.** In each window, at most `2^c` segments are charged `c` bits or fewer. A self-delimiting code
charged at least its length meets it (`bitsCount_of_prefixFree`), and so does any code charged its length plus one
(`bitsCount_of_lengthSucc`), which covers a public cap `ℓ` charged `ℓ + 1`. -/
def BitsCount (code : AdviceCodec Adv W) : Prop :=
  ∀ (w : Fin W) (c : ℕ), {s | ∃ a, code.seg a w = s ∧ code.bits a w ≤ c}.encard ≤ ((2 ^ c : ℕ) : ℕ∞)

/-- **The compiler is deterministic, and the advice is its only free input**: its frames depend on the advice and the
certified ingress, never on its other inputs `Ω` (a clock, coins, or state of the run inside X). -/
def CompilerDeterministic {L : ℕ} {In Ω : Type} (C : Compiler L In Adv Ω) : Prop :=
  ∀ a ω ω', C a ω = C a ω'

/-- **The compiler is precommitted**: it is registered before the run, so the compiler of the run with secret `z` is
the same for every secret. What the registration itself carries is charged separately, once. -/
def CompilerPrecommitted {Z : Type} {L : ℕ} {In Ω : Type} (compilerOf : Z → Compiler L In Adv Ω) : Prop :=
  ∀ z z', compilerOf z = compilerOf z'

end NetTiming
