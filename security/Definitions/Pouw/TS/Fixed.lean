module

public import Definitions.Pouw.TS.Sampled
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

/-! A fixed word under option A: each multiplicative operand is bound before its noise's coin or written by an always-proved or priced unit, and every unit that steers it is computed. -/

@[expose] public section

namespace ResourceAccounting.TS

open Pouw.RelativeSecurity

/-- `mul`: an operand the noise meets or that meets another operand; `add`: an operand only added to the work (a
K-chunk's running sum); `other`: no operand. -/
inductive SlotRole
  | mul
  | add
  | other
  deriving DecidableEq

/-- `acc` reads neither the noise nor a multiplicative operand, so an operand the noise meets is multiplicative. -/
structure WordLayer (C B : Type) (S m s : ℕ) (J Q Pos V N W : Type) extends SampledLayer C B S m s J Q Pos V where
  role : Fin S × J → Q → SlotRole
  noise : Paths (Paths C m × B) S → Fin S × J → N
  words : Paths (Paths C m × B) S → Fin S × J → W
  prod : (u : Fin S × J) → ({k : Q // role u k = .mul} → V) → N → W
  acc : (u : Fin S × J) → ({k : Q // role u k = .add} → V) → W
  output : Fin S × J → (Q → V) → N → Q → V
  receipt : Fin S × J → Q
  fold : Fin S × J → W → V

namespace WordLayer

variable {C B : Type} {S m s : ℕ} {J Q Pos V N W : Type} (pf : WordLayer C B S m s J Q Pos V N W)

def Bound (u : Fin S × J) (k : Q) : Prop :=
  (pf.cut (pf.pos u.1 u.2 k)).1 < u.1 ∨
    (pf.cut (pf.pos u.1 u.2 k)).1 = u.1 ∧ (pf.cut (pf.pos u.1 u.2 k)).2 ≤ (pf.coin u.1 u.2).castSucc

def AlwaysProved (u : Fin S × J) : Prop := ∀ b, ∃ i, pf.draw b i = u.2

section Priced

variable [Fintype B] [Fintype J]

/-- Over the mean round weight `N_w = Σ wt / S`, not `v`'s own round's weight: a reader composing it with
`SampledLayer.Covers` passes `Covers` the same `Nw`. -/
def Priced (wt : Fin S → J → ℝ) (v : Fin S × J) : Prop :=
  ∃ i, pf.coneWt wt v ≤ prob (fun b : B => pf.draw b i = v.2) * ((∑ w, ∑ j, wt w j) / S)

/-- Its additive operands don't enter. -/
def Fixed (wt : Fin S → J → ℝ) (u : Fin S × J) : Prop :=
  ∀ k, pf.role u k = .mul → pf.Bound u k ∨
    ∃ v, pf.src (pf.pos u.1 u.2 k) = some v ∧ (pf.AlwaysProved v ∨ pf.Priced wt v)

end Priced

def MulLate (v u : Fin S × J) : Prop :=
  ∃ k, pf.role u k = .mul ∧ pf.src (pf.pos u.1 u.2 k) = some v ∧ ¬ pf.Bound u k

def Steers (t u : Fin S × J) : Prop := ∃ v, Relation.ReflTransGen pf.Late t v ∧ pf.MulLate v u

variable [Fintype C] [Fintype B] [Nonempty V]

noncomputable def mulView (x : Paths (Paths C m × B) S) (u : Fin S × J) : {k : Q // pf.role u k = .mul} → V :=
  fun k => pf.view x u.1 u.2 k.1

noncomputable def addView (x : Paths (Paths C m × B) S) (u : Fin S × J) : {k : Q // pf.role u k = .add} → V :=
  fun k => pf.view x u.1 u.2 k.1

def Folds (x : Paths (Paths C m × B) S) (u : Fin S × J) : Prop :=
  pf.committed x (pf.pos u.1 u.2 (pf.receipt u)) = pf.fold u (pf.words x u)

variable [AddCommGroup W]

def derive (u : Fin S × J) (v : Q → V) (n : N) : W := pf.prod u (fun k => v k.1) n + pf.acc u (fun k => v k.1)

noncomputable def work (x : Paths (Paths C m × B) S) (u : Fin S × J) : W := pf.words x u - pf.acc u (pf.addView x u)

def Computed (x : Paths (Paths C m × B) S) (u : Fin S × J) : Prop :=
  pf.words x u = pf.derive u (pf.view x u.1 u.2) (pf.noise x u) ∧
    ∀ k, pf.src (pf.pos u.1 u.2 k) = some u →
      pf.committed x (pf.pos u.1 u.2 k) = pf.output u (pf.view x u.1 u.2) (pf.noise x u) k

variable [Fintype J] in
def FixedWords (wt : Fin S → J → ℝ) (x : Paths (Paths C m × B) S) (u : Fin S × J) : Prop :=
  pf.Fixed wt u ∧ pf.Computed x u ∧ ∀ t, pf.Steers t u → pf.Computed x t

end WordLayer

end ResourceAccounting.TS
