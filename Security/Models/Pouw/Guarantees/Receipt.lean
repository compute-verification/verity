import Definitions.Pouw.Receipt.Duo
import Definitions.Pouw.Receipt.Li

/-!
# The receipt folds duo's and li's binding (trusted)

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

What is proved about li's half steps and checkpoints (`Definitions/Pouw/Receipt/Li.lean`), at every width `n` and for
odd multipliers alone:

* `LiHalfBijective`: a half step is a bijection of the state.
* `LiCheckpointWordInjective`: a checkpoint, the XOR of several chains' states, is injective in each word any chain
  absorbed, whatever the other chains, the words before it and the words after it.
* `LiCheckpointLowBit`: changing one word by XOR with `x` changes the checkpoint at `x`'s lowest set bit and at no bit
  below it.

With every absorbed word recomputed from committed operands and a pinned start, these make a checkpoint taken every
segment catch every skipped value that lands on one word of the segment, with no key rule. One receipt word a window is
not covered: a persistent skip's copies all differ first at the same bit, and an even number of them can cancel there.
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

/-- **One li half step is a bijection of the state.** For every width, XOR key, odd multiplier and word pair, the state
before the half step is determined by the state after it. -/
def LiHalfBijective : Prop :=
  ∀ (n : ℕ) (k : BitVec n) (a : OddKey n) (d : BitVec n × BitVec n), Function.Bijective (fun y => liHalf k a.1 y d)

/-- **A checkpoint binds each word it absorbed.** For every width, chains `before` and `after`, and a chain that
absorbs the half steps `pre`, then a half step with an odd multiplier, then the half steps `post` with odd multipliers,
from the state `y`: the checkpoint of all the chains, as a function of either word of the middle half step with
everything else fixed, is injective. -/
def LiCheckpointWordInjective : Prop :=
  ∀ (n : ℕ) (before after : List (List (LiHalfStep n) × BitVec n)) (pre : List (LiHalfStep n)) (k : BitVec n)
    (a : OddKey n) (d : BitVec n × BitVec n) (post : List (LiOddHalfStep n)) (y : BitVec n),
    Function.Injective (fun d₀ =>
      liCheckpoint (before ++ (pre ++ (k, a.1, (d₀, d.2)) :: post.map LiOddHalfStep.val, y) :: after)) ∧
    Function.Injective (fun d₁ =>
      liCheckpoint (before ++ (pre ++ (k, a.1, (d.1, d₁)) :: post.map LiOddHalfStep.val, y) :: after))

/-- **A word's change moves the checkpoint at the change's lowest bit and below it nowhere.** As
`LiCheckpointWordInjective`'s chains: changing either word of the middle half step by XOR with `x`, whose lowest set bit
is bit `v`, changes the checkpoint at bit `v` and at no bit below it. Applied one word at a time, this gives the parity
rule: when two runs of the same chains differ in words whose lowest differing bit is at least `v`, their checkpoints
differ at bit `v` when an odd number of those words differ at bit `v`. -/
def LiCheckpointLowBit : Prop :=
  ∀ (n v : ℕ) (x : LowBit n v) (before after : List (List (LiHalfStep n) × BitVec n)) (pre : List (LiHalfStep n))
    (k : BitVec n) (a : OddKey n) (d : BitVec n × BitVec n) (post : List (LiOddHalfStep n)) (y : BitVec n),
    (liCheckpoint (before ++ (pre ++ (k, a.1, (d.1 ^^^ x.1, d.2)) :: post.map LiOddHalfStep.val, y) :: after) ^^^
        liCheckpoint (before ++ (pre ++ (k, a.1, d) :: post.map LiOddHalfStep.val, y) :: after)).setWidth (v + 1) =
      BitVec.twoPow (v + 1) v ∧
    (liCheckpoint (before ++ (pre ++ (k, a.1, (d.1, d.2 ^^^ x.1)) :: post.map LiOddHalfStep.val, y) :: after) ^^^
        liCheckpoint (before ++ (pre ++ (k, a.1, d) :: post.map LiOddHalfStep.val, y) :: after)).setWidth (v + 1) =
      BitVec.twoPow (v + 1) v

end Pouw.Guarantees.Receipt
