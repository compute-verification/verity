import Definitions.Pouw.Receipt.Duo
import Definitions.Pouw.Receipt.Fx1

/-!
# The receipt folds duo's and fx1's binding (trusted)

What is proved about duo's words (`Definitions/Pouw/Receipt/Duo.lean`), at every width `n`:

* `DuoWordInjective`: one word is bound at every start state, for every odd multiplier and any rotation and XOR key.
* `DuoPairNoKernel`: two words have no all-state kernel: two different word pairs leave different final states on some
  start state, for an odd first multiplier and a first rotation `0 < r₁ < n`, whatever the second word's keys.
* `DuoTripleNoKernel`: three words have no all-state kernel, for odd first and second multipliers and first and second
  rotations `0 < r < n`, whatever the third word's keys.
* `DuoQuadNoKernel`: four words have no all-state kernel, for odd first, second and third multipliers, second and third
  rotations `0 < r < n`, a first rotation `0 < r₁ ≤ n − 3` and a second multiplier `a₂ ≢ −1 (mod 2^r₁)`, whatever
  the fourth word's keys.
* `DuoStepBijective`: one step permutes the states, for every key, and is injective in the word for an odd
  multiplier.
* `DuoRunWordInjective`: a run of any length binds each of its words at every start state when the others are
  fixed, for odd multipliers: Lemma B's (Inj) with the full state committed (LEDGER's case (a),
  note:20261017T1723Z-draft-ledger §1.3).

`DuoQuadNoKernel`'s condition on `a₂` is needed: with `r₁ + r₃ ≡ 0 (mod n)`, `a₂ ≡ −1 (mod 2^r₁)` and
`a₃ ≡ ±1 (mod 2^(n+1−r₁))`, flipping the top bit of all four words leaves every final state unchanged. pfill's `key_ok`
rejects `a ≡ −1 (mod 32)` and its rotations are 5 to 29, and no served word's multiplier is `≡ −1` modulo two to the
previous word's rotation, so both deployed schedules meet it at `n = 32`. Runs of five or more words have no
all-state statement here (`DuoRunWordInjective` fixes all words but one), nor does the served receipt
`y ⊕ rotl(z, 16)`, which commits less than the whole state.

fx1 (`Definitions/Pouw/Receipt/Fx1.lean`) is duo on the word times an odd multiplier `m`, a bijection of the words, so
each duo statement holds for fx1 with every IMAD multiplier odd (`Fx1WordInjective`, `Fx1PairNoKernel`,
`Fx1TripleNoKernel`, `Fx1QuadNoKernel`, `Fx1StepBijective`, `Fx1RunWordInjective`).
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

/-- **One duo step permutes the states and is injective in the word.** For every width, XOR key and rotation: for
every multiplier and word, the map from start states to states after the word is a bijection (`u = rotr(z′ − y′, r)`,
`z = y′ − u·a`, `y = u ⊕ d ⊕ k`); and for every odd multiplier and start state, the state after the word determines
the word. -/
def DuoStepBijective : Prop :=
  ∀ (n : ℕ) (k : BitVec n) (r : ℕ),
    (∀ a d : BitVec n, Function.Bijective (fun s => duoWord k a r s d)) ∧
      ∀ (a : OddKey n) (s : BitVec n × BitVec n), Function.Injective (duoWord k a.1 r s)

/-- **A duo run binds each of its words at every state.** For every width and length, any XOR keys and rotations,
odd multipliers, every start state, every run of words and every position `i`, the final state of the run with word
`i` replaced by `x` determines `x`: the words before `i` reach one state, the step at `i` is injective in its word and
the words after it permute the states. -/
def DuoRunWordInjective : Prop :=
  ∀ (n L : ℕ) (k : Fin L → BitVec n) (a : Fin L → OddKey n) (r : Fin L → ℕ) (s : BitVec n × BitVec n)
    (w : Fin L → BitVec n) (i : Fin L),
    Function.Injective (fun x => duoRun k (fun j => (a j).1) r (Function.update w i x) s)

/-- **One fx1 word is bound at every state.** For every width, XOR key, odd multiplier, rotation, odd IMAD
multiplier and start state, the state after one word determines the word. -/
def Fx1WordInjective : Prop :=
  ∀ (n : ℕ) (k : BitVec n) (a : OddKey n) (r : ℕ) (m : OddKey n) (s : BitVec n × BitVec n),
    Function.Injective (fx1Word k a.1 r m.1 s)

/-- **Two fx1 words have no all-state kernel.** `DuoPairNoKernel`'s keys, with odd IMAD multipliers. -/
def Fx1PairNoKernel : Prop :=
  ∀ (n : ℕ) (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (m₁ : OddKey n) (k₂ a₂ : BitVec n) (r₂ : ℕ)
    (m₂ : OddKey n), Function.Injective (fx1Pair k₁ a₁.1 r₁.1 m₁.1 k₂ a₂ r₂ m₂.1)

/-- **Three fx1 words have no all-state kernel.** `DuoTripleNoKernel`'s keys, with odd IMAD multipliers. -/
def Fx1TripleNoKernel : Prop :=
  ∀ (n : ℕ) (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (m₁ : OddKey n) (k₂ : BitVec n) (a₂ : OddKey n)
    (r₂ : Rotation n) (m₂ : OddKey n) (k₃ a₃ : BitVec n) (r₃ : ℕ) (m₃ : OddKey n),
    Function.Injective (fx1Triple k₁ a₁.1 r₁.1 m₁.1 k₂ a₂.1 r₂.1 m₂.1 k₃ a₃ r₃ m₃.1)

/-- **Four fx1 words have no all-state kernel.** `DuoQuadNoKernel`'s keys, with odd IMAD multipliers. -/
def Fx1QuadNoKernel : Prop :=
  ∀ (n : ℕ) (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : ShortRotation n) (m₁ : OddKey n) (k₂ : BitVec n)
    (a₂ : OddKeyNotNegOne n r₁.1) (r₂ : Rotation n) (m₂ : OddKey n) (k₃ : BitVec n) (a₃ : OddKey n) (r₃ : Rotation n)
    (m₃ : OddKey n) (k₄ a₄ : BitVec n) (r₄ : ℕ) (m₄ : OddKey n),
    Function.Injective (fx1Quad k₁ a₁.1 r₁.1 m₁.1 k₂ a₂.1 r₂.1 m₂.1 k₃ a₃.1 r₃.1 m₃.1 k₄ a₄ r₄ m₄.1)

/-- **One fx1 step permutes the states and is injective in the word.** For every width, XOR key and rotation: for
every multiplier, IMAD multiplier and word, the step is a bijection of the states; for odd multipliers and every
start state, it is injective in the word. -/
def Fx1StepBijective : Prop :=
  ∀ (n : ℕ) (k : BitVec n) (r : ℕ),
    (∀ a m d : BitVec n, Function.Bijective (fun s => fx1Word k a r m s d)) ∧
      ∀ (a m : OddKey n) (s : BitVec n × BitVec n), Function.Injective (fx1Word k a.1 r m.1 s)

/-- **An fx1 run binds each of its words at every state.** `DuoRunWordInjective` with odd IMAD multipliers. -/
def Fx1RunWordInjective : Prop :=
  ∀ (n L : ℕ) (k : Fin L → BitVec n) (a : Fin L → OddKey n) (r : Fin L → ℕ) (m : Fin L → OddKey n)
    (s : BitVec n × BitVec n) (w : Fin L → BitVec n) (i : Fin L),
    Function.Injective (fun x => fx1Run k (fun j => (a j).1) r (fun j => (m j).1) (Function.update w i x) s)

end Pouw.Guarantees.Receipt
