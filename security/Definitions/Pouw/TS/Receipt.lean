module

public import Definitions.Pouw.TS.Play
public import Definitions.Pouw.Receipt.Fx1
public import Definitions.Pouw.Receipt.Li
public import Definitions.Pouw.RelativeSecurity
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! The ideal game split at the receipt into the operand luck's event `right` and the event `same`, that the committed value is the honest one. -/

@[expose] public section

namespace ResourceAccounting.TS

open Pouw.RelativeSecurity Pouw.Receipt

/-- With `L = 1`, a full-state checkpoint after every word, every run has this form. -/
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

def word (D : DuoUnits Ω U n L) (ω : Ω) (u : U) : Fin L → BitVec n :=
  Function.update (D.honest ω u) (D.pos ω u) (D.value ω u)

def right (D : DuoUnits Ω U n L) (ω : Ω) : Prop := ∀ u, D.value ω u = D.honest ω u (D.pos ω u)

def same (D : DuoUnits Ω U n L) (ω : Ω) : Prop :=
  ∀ u, duoRun (D.k u) (fun j => (D.a u j).1) (D.r u) (D.word ω u) (D.start ω u) =
    duoRun (D.k u) (fun j => (D.a u j).1) (D.r u) (D.honest ω u) (D.start ω u)

end DuoUnits

structure Fx1Units (Ω U : Type) (n L : ℕ) extends DuoUnits Ω U n L where
  m : U → Fin L → OddKey n

def Fx1Units.same {Ω U : Type} {n L : ℕ} (D : Fx1Units Ω U n L) (ω : Ω) : Prop :=
  ∀ u, fx1Run (D.k u) (fun j => (D.a u j).1) (D.r u) (fun j => (D.m u j).1) (D.word ω u) (D.start ω u) =
    fx1Run (D.k u) (fun j => (D.a u j).1) (D.r u) (fun j => (D.m u j).1) (D.honest ω u) (D.start ω u)

/-- `0` is `d₀`, XORed into the state, and `1` is `d₁`, added. -/
abbrev LiPlace (W C H : ℕ) : Type := Fin W × Fin C × Fin H × Fin 2

/-- A persistent skip, which moves one word of every segment to the window's end, has this form; a keyed round
(`y ← (y ⊕ k)·a + c`, absorbing no word) is a half step whose honest pair is `(0, c)`. -/
structure LiUnits (Ω U : Type) (n S W C H : ℕ) where
  k : Ω → U → ℕ → Fin W × Fin C × Fin H → BitVec n
  a : Ω → U → ℕ → Fin W × Fin C × Fin H → OddKey n
  start : Ω → U → Fin W → Fin C → BitVec n
  honest : Ω → U → ℕ → LiPlace W C H → BitVec n
  pos : Ω → U → ℕ → LiPlace W C H
  value : Ω → U → ℕ → BitVec n

namespace LiUnits

variable {Ω U : Type} {n S W C H : ℕ}

def word (D : LiUnits Ω U n S W C H) (ω : Ω) (u : U) (s : ℕ) : LiPlace W C H → BitVec n :=
  Function.update (D.honest ω u s) (D.pos ω u s) (D.value ω u s)

def steps (D : LiUnits Ω U n S W C H) (ω : Ω) (u : U) (s : ℕ) (w : LiPlace W C H → BitVec n) (ℓ : Fin W) (c : Fin C) :
    List (LiHalfStep n) :=
  List.ofFn fun j => (D.k ω u s (ℓ, c, j), (D.a ω u s (ℓ, c, j)).1, (w (ℓ, c, j, 0), w (ℓ, c, j, 1)))

def state (D : LiUnits Ω U n S W C H) (ω : Ω) (u : U) (w : ℕ → LiPlace W C H → BitVec n) :
    ℕ → Fin W → Fin C → BitVec n
  | 0 => D.start ω u
  | s + 1 => fun ℓ c => liChain (D.steps ω u s (w s) ℓ c) (D.state ω u w s ℓ c)

def checkpoint (D : LiUnits Ω U n S W C H) (ω : Ω) (u : U) (w : ℕ → LiPlace W C H → BitVec n) (s : ℕ) : BitVec n :=
  (List.ofFn fun ℓ => liCheckpoint (List.ofFn fun c => (D.steps ω u s (w s) ℓ c, D.state ω u w s ℓ c))).foldl
    (· ^^^ ·) 0

def right (D : LiUnits Ω U n S W C H) (ω : Ω) : Prop := ∀ u, ∀ s < S, D.value ω u s = D.honest ω u s (D.pos ω u s)

def same (D : LiUnits Ω U n S W C H) (ω : Ω) : Prop :=
  ∀ u, ∀ s < S, D.checkpoint ω u (D.word ω u) s = D.checkpoint ω u (D.honest ω u) s

def sameExecutionPhase (D : LiUnits Ω U n S W C H) (ω : Ω) : Prop :=
  ∀ u, D.checkpoint ω u (D.word ω u) (S - 1) = D.checkpoint ω u (D.honest ω u) (S - 1)

end LiUnits

def xorAll {n m : ℕ} (f : Fin m → BitVec n) : BitVec n := (List.ofFn f).foldl (· ^^^ ·) 0

abbrev CPPlace (W C H : ℕ) : Type := Fin W × Fin C × Fin H

/-- The served GEMM's layout (`W = 32`, `C = 8`, `H = 4`, one segment a k32 step) has this form for a skip that lands
on one word of a segment; the served receipt, `y ⊕ rotl(z, 16)` a thread a window, is not such a unit. -/
structure DuoCPUnits (Ω U : Type) (n S W C H : ℕ) where
  k : Ω → U → ℕ → CPPlace W C H → BitVec n
  a : Ω → U → ℕ → CPPlace W C H → OddKey n
  r : Ω → U → ℕ → CPPlace W C H → ℕ
  start : Ω → U → Fin W → Fin C → BitVec n × BitVec n
  honest : Ω → U → ℕ → CPPlace W C H → BitVec n
  pos : Ω → U → ℕ → CPPlace W C H
  value : Ω → U → ℕ → BitVec n

namespace DuoCPUnits

variable {Ω U : Type} {n S W C H : ℕ}

def word (D : DuoCPUnits Ω U n S W C H) (ω : Ω) (u : U) (s : ℕ) : CPPlace W C H → BitVec n :=
  Function.update (D.honest ω u s) (D.pos ω u s) (D.value ω u s)

def state (D : DuoCPUnits Ω U n S W C H) (ω : Ω) (u : U) (w : ℕ → CPPlace W C H → BitVec n) :
    ℕ → Fin W → Fin C → BitVec n × BitVec n
  | 0 => D.start ω u
  | s + 1 => fun ℓ c =>
      duoRun (fun j => D.k ω u s (ℓ, c, j)) (fun j => (D.a ω u s (ℓ, c, j)).1) (fun j => D.r ω u s (ℓ, c, j))
        (fun j => w s (ℓ, c, j)) (D.state ω u w s ℓ c)

def checkpoint (D : DuoCPUnits Ω U n S W C H) (ω : Ω) (u : U) (w : ℕ → CPPlace W C H → BitVec n) (s : ℕ) :
    BitVec n × BitVec n :=
  (xorAll fun ℓ => xorAll fun c => (D.state ω u w (s + 1) ℓ c).1,
    xorAll fun ℓ => xorAll fun c => (D.state ω u w (s + 1) ℓ c).2)

def right (D : DuoCPUnits Ω U n S W C H) (ω : Ω) : Prop := ∀ u, ∀ s < S, D.value ω u s = D.honest ω u s (D.pos ω u s)

def same (D : DuoCPUnits Ω U n S W C H) (ω : Ω) : Prop :=
  ∀ u, ∀ s < S, D.checkpoint ω u (D.word ω u) s = D.checkpoint ω u (D.honest ω u) s

end DuoCPUnits

structure Fx1CPUnits (Ω U : Type) (n S W C H : ℕ) extends DuoCPUnits Ω U n S W C H where
  m : Ω → U → ℕ → CPPlace W C H → OddKey n

namespace Fx1CPUnits

variable {Ω U : Type} {n S W C H : ℕ}

def imad (D : Fx1CPUnits Ω U n S W C H) (ω : Ω) (u : U) (w : ℕ → CPPlace W C H → BitVec n) :
    ℕ → CPPlace W C H → BitVec n :=
  fun s p => w s p * (D.m ω u s p).1

def same (D : Fx1CPUnits Ω U n S W C H) (ω : Ω) : Prop :=
  ∀ u, ∀ s < S, D.checkpoint ω u (D.imad ω u (D.word ω u)) s = D.checkpoint ω u (D.imad ω u (D.honest ω u)) s

end Fx1CPUnits

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

def CaseA : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round _ _ right same next =>
      ((∃ (U : Type) (w L : ℕ) (D : DuoUnits Ω U w L), ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∨
        (∃ (U : Type) (w L : ℕ) (D : Fx1Units Ω U w L), ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∨
        (∃ (U : Type) (w S W C H : ℕ) (D : LiUnits Ω U w S W C H),
          ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∨
        (∃ (U : Type) (w S W C H : ℕ) (D : DuoCPUnits Ω U w S W C H),
          ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∨
        ∃ (U : Type) (w S W C H : ℕ) (D : Fx1CPUnits Ω U w S W C H),
          ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∧
        ∀ ω, (next ω).CaseA

def CubeOperands (R : Type) [CommRing R] {α V : Type} [Fintype V] [DecidableEq V] :
    {n : ℕ} → ReceiptGame (α × Finset V) n → Prop
  | _, .done => True
  | _, .round d _ right _ next =>
      (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧
        (∃ (δ : α → ℝ) (e : α → (V → R) → R), (∀ a, CubeQuad (e a)) ∧ (∀ ω, d ω = δ ω.1) ∧
          (∀ a, 0 < δ a → ∃ S : Finset V, e a (bitInd S) ≠ 0) ∧ ∀ ω, right ω → e ω.1 (bitInd ω.2) = 0) ∧
        ∀ ω, (next ω).CubeOperands R

/-- For keyed rounds a deficient word is a round whose state the run did not compute, and FOLD-OPT prices a right one at
the round's work, so the share counts the wrong ones. -/
def IncorrectWords : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round d _ right _ next => (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧ (∀ ω, 0 < d ω → ¬right ω) ∧ ∀ ω, (next ω).IncorrectWords

end ReceiptGame

end ResourceAccounting.TS
