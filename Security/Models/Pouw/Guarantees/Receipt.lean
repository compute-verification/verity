import Definitions.Pouw.Receipt.Duo

/-!
# The receipt fold duo's binding (trusted)

What is proved about duo's words (`Definitions/Pouw/Receipt/Duo.lean`), at every width `n`:

* `DuoWordInjective`: one word is bound at every start state, for every odd multiplier and any rotation and XOR key.
* `DuoPairNoKernel`: two words have no all-state kernel: two different word pairs leave different final states on some
  start state, for an odd first multiplier and a first rotation `0 < r₁ < n`, whatever the second word's keys.

Longer runs are not covered. At four words the statement fails for some odd keys: with `r₁ + r₃ ≡ 0 (mod n)`,
`a₂ ≡ −1 (mod 2^r₁)` and `a₃ ≡ ±1 (mod 2^(n+1−r₁))`, flipping the top bit of all four words leaves every final state
unchanged. pfill's `key_ok` and both deployed rotation schedules exclude that family; whether the deployed keys bind
runs of three or more words is open.
-/

namespace Pouw.Guarantees.Receipt

open Pouw.Receipt

/-- **One duo word is bound at every state.** For every width, XOR key, odd multiplier, rotation and start state, the
state after one word determines the word. -/
def DuoWordInjective : Prop :=
  ∀ (n : ℕ) (k : BitVec n) (a : OddKey n) (r : ℕ) (s : BitVec n × BitVec n), Function.Injective (duoWord k a.1 r s)

/-- **Two duo words have no all-state kernel.** For every width, an odd first multiplier, a first rotation that moves
every bit and any keys for the second word, two word pairs whose final states agree on every start state are equal. -/
def DuoPairNoKernel : Prop :=
  ∀ (n : ℕ) (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (k₂ a₂ : BitVec n) (r₂ : ℕ),
    Function.Injective (duoPair k₁ a₁.1 r₁.1 k₂ a₂ r₂)

end Pouw.Guarantees.Receipt
