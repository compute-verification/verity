module

public import Definitions.Pouw.Receipt.Duo
public import Definitions.Pouw.Receipt.Li
public import Definitions.Pouw.Receipt.Fx1

/-! The binding statements of duo's, fx1's and li's receipt folds; none binds a whole run's final state at every length, and the served receipt `y ⊕ rotl(z, 16)` is not a full-state commitment. -/

@[expose] public section

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

/-- **Four duo words have no all-state kernel anywhere in a run.** For every width, any `p` words and keys before
and any `t` words and keys after, and four middle words whose keys meet `DuoQuadKeysOk`: two runs of the `p` words,
four middle words, then the `t` words, whose final states agree on every start state, have the same middle words. -/
def DuoWindowNoKernel : Prop :=
  ∀ (n p t : ℕ) (k a : Fin p → BitVec n) (r : Fin p → ℕ) (w : Fin p → BitVec n) (q : DuoQuadKeysOk n)
    (k' a' : Fin t → BitVec n) (r' : Fin t → ℕ) (w' : Fin t → BitVec n),
    Function.Injective (fun d s => duoRun k' a' r' w' (duoQuadWith q.1 d (duoRun k a r w s)))

/-- **A duo run checkpointed every four words is bound at every length.** For every width and number of blocks `m`,
and keys for each block that meet `DuoQuadKeysOk`: two runs of `m` four-word blocks whose checkpoints, the states after
each block, agree on every start state are equal. -/
def DuoCheckpointNoKernel : Prop :=
  ∀ (n m : ℕ) (q : Fin m → DuoQuadKeysOk n), Function.Injective (duoCheckpoints (fun j => (q j).1))

/-- **Every duo run longer than `2·4ⁿ` words has an all-state kernel.** For every width `n + 1`, length
`2·4ⁿ⁺¹ + 1 + e` and keys: two different runs of that many words leave the same final state on every start state. -/
def DuoLongRunKernel : Prop :=
  ∀ (n e : ℕ) (k a : Fin (2 * 4 ^ (n + 1) + 1 + e) → BitVec (n + 1)) (r : Fin (2 * 4 ^ (n + 1) + 1 + e) → ℕ),
    ∃ w w', w ≠ w' ∧ duoRun k a r w = duoRun k a r w'

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

/-- **One duo word's served receipt binds it up to no XOR constant.** For every width, XOR key, odd multiplier,
rotation `0 < r < n` and receipt rotation `0 < h < n`, two words whose receipts `y ⊕ rotl(z, h)` differ by one XOR
constant on every start state (equal `duoWordSpread`s) are equal. -/
def DuoWordProjNoKernel : Prop :=
  ∀ (n : ℕ) (k : BitVec n) (a : OddKey n) (r h : Rotation n), Function.Injective (duoWordSpread k a.1 r.1 h.1)

/-- **No whole state binds three duo words at a state.** For every width `n + 1`, every run of `L + 3` words with any
keys and every start state, two different runs reach one final state: `2^((n+1)(L+3))` runs, `2^(2(n+1))` states. -/
def DuoRunAtStateNotInjective : Prop :=
  ∀ (n L : ℕ) (k a : Fin (L + 3) → BitVec (n + 1)) (r : Fin (L + 3) → ℕ) (s : BitVec (n + 1) × BitVec (n + 1)),
    ¬ Function.Injective (fun w : Fin (L + 3) → BitVec (n + 1) => duoRun k a r w s)

/-- **Two duo words have no all-state XOR kernel.** For every width, odd multipliers and rotations that move every
bit, two word pairs whose final states differ by one XOR constant on every start state (equal `stateSpread`s) are
equal. -/
def DuoPairXorNoKernel : Prop :=
  ∀ (n : ℕ) (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (k₂ : BitVec n) (a₂ : OddKey n) (r₂ : Rotation n),
    Function.Injective (fun d => stateSpread (duoPair k₁ a₁.1 r₁.1 k₂ a₂.1 r₂.1 d))

/-- **Two fx1 words have no all-state XOR kernel.** `DuoPairXorNoKernel`'s keys, with odd IMAD multipliers. -/
def Fx1PairXorNoKernel : Prop :=
  ∀ (n : ℕ) (k₁ : BitVec n) (a₁ : OddKey n) (r₁ : Rotation n) (m₁ : OddKey n) (k₂ : BitVec n) (a₂ : OddKey n)
    (r₂ : Rotation n) (m₂ : OddKey n),
    Function.Injective (fun d => stateSpread (fx1Pair k₁ a₁.1 r₁.1 m₁.1 k₂ a₂.1 r₂.1 m₂.1 d))

end Pouw.Guarantees.Receipt
