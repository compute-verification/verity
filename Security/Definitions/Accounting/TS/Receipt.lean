import Definitions.Accounting.TS.Play
import Definitions.Pouw.Receipt.Fx1

/-!
# The ideal game split at the receipt (trusted)

LEDGER §1.3 (note:20261017T1723Z-draft-ledger): with every absorbed word pinned in the proof, a round's ideal check
recomputes the honest committed value from committed data and passes only if the committed value equals it. A round
then carries two more events on its coins:

* `right ω`: every deficient word's value is right, the operand luck's event (`luck_quad`'s ¾ for affine operands);
* `same ω`: the committed value is the honest one.

* `ReceiptGame`: the ideal game with each round split so; `ReceiptGame.game` forgets the split.
* `ReceiptGame.Pinned`: a round passes only if `same` (the openings are pinned).
* `ReceiptGame.OperandLuck p`: (L) at `p` for the event `right`: every round's share lies in `[0, 1]`, and given the
  past a deficient round's deficient words are all right with probability at most `p`.
* `ReceiptGame.ReceiptLuck pr`: given the past, a deficient round whose deficient words are not all right keeps the
  honest committed value with probability at most `pr`.
* `DuoUnits`, `Fx1Units`: case (a)'s committed units: each commits the full state of a duo (or fx1) chain after its
  words, from a start state the proof pins, and the run's words in a unit are honest but for one position.
  `ReceiptGame.CaseA`: every round's split is that of such units.
* `ReceiptGame.CubeOperands R`: every round draws uniform cube bits after fixing its share and a degree-2 operand
  error, nonzero when the round is deficient (`luck_quad`'s rounds).
-/

namespace Accounting.TS

open Pouw.RelativeSecurity Pouw.Receipt

/-- **Case (a)'s committed units** on a round's coins (LEDGER §1.3, case (a)): unit `u` commits the full state of a
duo chain after its `L` words, word `j` with keys `(k u j, a u j, r u j)` and an odd multiplier, from the start state
`start ω u` that the proof pins. On the coins `ω` the run folds the honest words `honest ω u` but at the one position
`pos ω u`, where it folds `value ω u`: each deficient value sits on one word of each unit. With `L = 1`, a full-state
checkpoint after every word, every run has this form. -/
structure DuoUnits (Ω U : Type) (n L : ℕ) where
  k : U → Fin L → BitVec n
  a : U → Fin L → OddKey n
  r : U → Fin L → ℕ
  start : Ω → U → BitVec n × BitVec n
  honest : Ω → U → Fin L → BitVec n
  pos : Ω → U → Fin L
  value : Ω → U → BitVec n

namespace DuoUnits

variable {Ω U : Type} {n L : ℕ}

/-- The run's words in unit `u`: the honest words, with `value ω u` at `pos ω u`. -/
def word (D : DuoUnits Ω U n L) (ω : Ω) (u : U) : Fin L → BitVec n :=
  Function.update (D.honest ω u) (D.pos ω u) (D.value ω u)

/-- Every deficient word's value is right. -/
def right (D : DuoUnits Ω U n L) (ω : Ω) : Prop := ∀ u, D.value ω u = D.honest ω u (D.pos ω u)

/-- Every unit's committed state is the honest words' state from its start. -/
def same (D : DuoUnits Ω U n L) (ω : Ω) : Prop :=
  ∀ u, duoRun (D.k u) (fun j => (D.a u j).1) (D.r u) (D.word ω u) (D.start ω u) =
    duoRun (D.k u) (fun j => (D.a u j).1) (D.r u) (D.honest ω u) (D.start ω u)

end DuoUnits

/-- **Case (a)'s committed units for fx1**: `DuoUnits` whose chains fold with fx1, word `j` of unit `u` multiplied by
the odd IMAD multiplier `m u j` before duo's step. -/
structure Fx1Units (Ω U : Type) (n L : ℕ) extends DuoUnits Ω U n L where
  m : U → Fin L → OddKey n

/-- Every unit's committed fx1 state is the honest words' state from its start. -/
def Fx1Units.same {Ω U : Type} {n L : ℕ} (D : Fx1Units Ω U n L) (ω : Ω) : Prop :=
  ∀ u, fx1Run (D.k u) (fun j => (D.a u j).1) (D.r u) (fun j => (D.m u j).1) (D.word ω u) (D.start ω u) =
    fx1Run (D.k u) (fun j => (D.a u j).1) (D.r u) (fun j => (D.m u j).1) (D.honest ω u) (D.start ω u)

/-- **The ideal game split at the receipt** over `n` rounds of coins on `Ω`: per round, its share `d`, its check
`pass`, and the events `right` (every deficient word's value is right) and `same` (the committed value is the honest
one), then the rest of the play, which may depend on the round's coins. -/
inductive ReceiptGame (Ω : Type) : ℕ → Type
  | done : ReceiptGame Ω 0
  | round {n : ℕ} (d : Ω → ℝ) (pass right same : Ω → Prop) (next : Ω → ReceiptGame Ω n) : ReceiptGame Ω (n + 1)

namespace ReceiptGame

variable {Ω : Type}

/-- The ideal game: each round's share and check. -/
def game : {n : ℕ} → ReceiptGame Ω n → RoundGame Ω n
  | _, .done => .done
  | _, .round d pass _ _ next => .round d pass fun ω => (next ω).game

/-- **The openings are pinned**: at every round, whatever the past, the round passes only if its committed value is
the honest one. -/
def Pinned : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round _ pass _ same next => (∀ ω, pass ω → same ω) ∧ ∀ ω, (next ω).Pinned

/-- **The operand luck at `p`**: every round's share lies in `[0, 1]`, and at every round, whatever the past, a
deficient round's deficient words are all right with probability at most `p`:
`Pr[right ∧ d_w > 0] ≤ p·Pr[d_w > 0]` over the round's coins. -/
def OperandLuck [Fintype Ω] (p : ℝ) : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round d _ right _ next =>
      (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧ prob (fun ω => right ω ∧ 0 < d ω) ≤ p * prob (fun ω => 0 < d ω) ∧
        ∀ ω, OperandLuck p (next ω)

/-- **The receipt luck at `pr`**: at every round, whatever the past, a deficient round whose deficient words are not
all right keeps the honest committed value with probability at most `pr`:
`Pr[same ∧ ¬right ∧ d_w > 0] ≤ pr·Pr[¬right ∧ d_w > 0]` over the round's coins. -/
def ReceiptLuck [Fintype Ω] (pr : ℝ) : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round d _ right same next =>
      prob (fun ω => same ω ∧ ¬right ω ∧ 0 < d ω) ≤ pr * prob (fun ω => ¬right ω ∧ 0 < d ω) ∧
        ∀ ω, ReceiptLuck pr (next ω)

/-- **Case (a)** (LEDGER §1.3): at every round, whatever the past, the round's `right` and `same` are those of
committed duo or fx1 units (`DuoUnits`, `Fx1Units`): each unit's full final state from a pinned start, the run's
words honest but for one position a unit. -/
def CaseA : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round _ _ right same next =>
      ((∃ (U : Type) (w L : ℕ) (D : DuoUnits Ω U w L), ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∨
        ∃ (U : Type) (w L : ℕ) (D : Fx1Units Ω U w L), ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∧
        ∀ ω, (next ω).CaseA

/-- **Degree-2 operands on cube bits** (`luck_quad`'s rounds, full checking): a round's coins are other coins `a`
and uniform cube bits `S`; at every round, whatever the past, its share lies in `[0, 1]` and is fixed before the bits
(`δ a`), the prover's operand error `e a` is a function of the bits of degree at most 2 (`CubeQuad`), a deficient
share means a nonzero error, and the deficient words are all right only where the error vanishes at the drawn bits. -/
def CubeOperands (R : Type) [CommRing R] {α V : Type} [Fintype V] [DecidableEq V] :
    {n : ℕ} → ReceiptGame (α × Finset V) n → Prop
  | _, .done => True
  | _, .round d _ right _ next =>
      (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧
        (∃ (δ : α → ℝ) (e : α → (V → R) → R), (∀ a, CubeQuad (e a)) ∧ (∀ ω, d ω = δ ω.1) ∧
          (∀ a, 0 < δ a → ∃ S : Finset V, e a (bitInd S) ≠ 0) ∧ ∀ ω, right ω → e ω.1 (bitInd ω.2) = 0) ∧
        ∀ ω, (next ω).CubeOperands R

end ReceiptGame

end Accounting.TS
