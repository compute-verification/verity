import Definitions.Pouw.Receipt.Duo

/-!
# The receipt fold duo's binding (trusted)

What is proved about duo's words (`Definitions/Pouw/Receipt/Duo.lean`), at every width `n`:

* `DuoWordInjective`: one word is bound at every start state, for every odd multiplier and any rotation and XOR key.
* `DuoPairNoKernel`: two words have no all-state kernel: two different word pairs leave different final states on some
  start state, for an odd first multiplier and a first rotation `0 < r₁ < n`, whatever the second word's keys.
* `DuoTripleNoKernel`: three words have no all-state kernel, for odd first and second multipliers and first and second
  rotations `0 < r < n`, whatever the third word's keys.
* `DuoQuadNoKernel`: four words have no all-state kernel, for odd first, second and third multipliers, second and third
  rotations `0 < r < n`, a first rotation `0 < r₁ ≤ n − 3` and a second multiplier `a₂ ≢ −1 (mod 2^r₁)`, whatever
  the fourth word's keys.

`DuoQuadNoKernel`'s condition on `a₂` is needed: with `r₁ + r₃ ≡ 0 (mod n)`, `a₂ ≡ −1 (mod 2^r₁)` and
`a₃ ≡ ±1 (mod 2^(n+1−r₁))`, flipping the top bit of all four words leaves every final state unchanged. pfill's `key_ok`
rejects `a ≡ −1 (mod 32)` and its rotations are 5 to 29, and no served word's multiplier is `≡ −1` modulo two to the
previous word's rotation, so both deployed schedules meet it at `n = 32`. Runs of five or more words are not covered,
nor is the served receipt `y ⊕ rotl(z, 16)`, which commits less than the whole state.
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

/-- **Three duo words have no all-state kernel.** For every width, odd first and second multipliers, first and second
rotations that move every bit and any keys for the third word, two word triples whose final states agree on every start
state are equal. -/
def DuoTripleNoKernel : Prop :=
  ∀ (n : ℕ) (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (k₂ : BitVec n) (a₂ : OddKey n) (r₂ : Rotation n)
    (k₃ a₃ : BitVec n) (r₃ : ℕ), Function.Injective (duoTriple k₁ a₁.1 r₁.1 k₂ a₂.1 r₂.1 k₃ a₃ r₃)

/-- **Four duo words have no all-state kernel.** For every width, odd first and third multipliers, a first rotation
`0 < r₁ ≤ n − 3`, an odd second multiplier that is not `−1` modulo `2^r₁`, second and third rotations that move every
bit and any keys for the fourth word, two word quadruples whose final states agree on every start state are equal. -/
def DuoQuadNoKernel : Prop :=
  ∀ (n : ℕ) (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : ShortRotation n) (k₂ : BitVec n) (a₂ : OddKeyNotNegOne n r₁.1)
    (r₂ : Rotation n) (k₃ : BitVec n) (a₃ : OddKey n) (r₃ : Rotation n) (k₄ a₄ : BitVec n) (r₄ : ℕ),
    Function.Injective (duoQuad k₁ a₁.1 r₁.1 k₂ a₂.1 r₂.1 k₃ a₃.1 r₃.1 k₄ a₄ r₄)

end Pouw.Guarantees.Receipt
