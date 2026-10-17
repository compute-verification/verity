import Definitions.Pouw.Receipt.Run

/-!
# duo and fx1 bind each word of a run given the others (trusted)

What is proved about runs of duo and fx1 words (`Definitions/Pouw/Receipt/Run.lean`), at every width `n` and length
`N`:

* `DuoStepBijective`: one duo word is a bijection of the state, for every XOR key, multiplier and rotation.
* `DuoRunWordInjective`: at every start state, a duo run's final state determines each word, given the others, for any
  keys.
* `Fx1RunWordInjective`: the same for fx1, with odd IMAD keys.

duo needs no condition on its keys here: `z′ − y′ = rotl(u, r)` recovers `u = y ⊕ d ⊕ k`, so one word's step is
injective in its word and in its state. This binds one word given the rest of the run (what the fan-in floor,
`Accounting.Guarantees.DuoFoldFanIn`, needs), not a whole run as `DuoPairNoKernel` does.
-/

namespace Pouw.Guarantees.Receipt

open Pouw.Receipt

/-- **One duo word is a bijection of the state.** For every width, XOR key, multiplier, rotation and word. -/
def DuoStepBijective : Prop :=
  ∀ (n : ℕ) (k a : BitVec n) (r : ℕ) (d : BitVec n), Function.Bijective fun s => duoWord k a r s d

/-- **A duo run binds each word given the others.** For every width and length, any keys for each word, start state,
run of words and position, the final state determines the word at that position. -/
def DuoRunWordInjective : Prop :=
  ∀ (n N : ℕ) (k a : Fin N → BitVec n) (r : Fin N → ℕ) (s : BitVec n × BitVec n) (d : Fin N → BitVec n)
    (i : Fin N), Function.Injective fun x => duoRun k a r s (Function.update d i x)

/-- **An fx1 run binds each word given the others.** For every width and length, an odd IMAD key and any duo keys for
each word, start state, run of words and position, the final state determines the word at that position. -/
def Fx1RunWordInjective : Prop :=
  ∀ (n N : ℕ) (m₁ : Fin N → OddKey n) (k a : Fin N → BitVec n) (r : Fin N → ℕ) (s : BitVec n × BitVec n)
    (d : Fin N → BitVec n) (i : Fin N),
    Function.Injective fun x => fx1Run (fun j => (m₁ j).1) k a r s (Function.update d i x)

end Pouw.Guarantees.Receipt
