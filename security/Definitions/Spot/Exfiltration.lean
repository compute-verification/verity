module

public import Definitions.Core.Coins
public import Mathlib.Analysis.Complex.Exponential
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Analysis.Complex.Order
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Data.EReal.Inv

/-! Weight exfiltration before detection: the windowed audit with live coins, as a game over a finite coin space. -/

@[expose] public section

namespace Population.Exfil

open Finset

/-- Both `leak h` and `esc h` are fixed before the window's coin is flipped: that is what live coins mean. -/
structure Strategy (Ω : Type) where
  leak : List Ω → ℝ
  esc : List Ω → Finset Ω

variable {Ω : Type}

noncomputable def ex [Fintype Ω] (μ : PMF Ω) (n : ℕ) (F : (Fin n → Ω) → ℝ) : ℝ :=
  ∑ cs, (Verity.Game.iid μ n cs).toReal * F cs

noncomputable abbrev prob [Fintype Ω] (μ : PMF Ω) (n : ℕ) (E : (Fin n → Ω) → Prop) : ℝ :=
  Verity.Game.prPMFReal (Verity.Game.iid μ n) E

/-- The caught window's egress left before its verdict. -/
noncomputable def Strategy.cum [DecidableEq Ω] (A : Strategy Ω) : (n : ℕ) → List Ω → (Fin n → Ω) → ℝ
  | 0, _, _ => 0
  | n + 1, h, cs => A.leak h + if cs 0 ∈ A.esc h then A.cum n (cs 0 :: h) (Fin.tail cs) else 0

noncomputable def Strategy.cumPrev [DecidableEq Ω] (A : Strategy Ω) : (n : ℕ) → List Ω → (Fin n → Ω) → ℝ
  | 0, _, _ => 0
  | n + 1, h, cs => if cs 0 ∈ A.esc h then A.leak h + A.cumPrev n (cs 0 :: h) (Fin.tail cs) else 0

/-- After a detection the prover knows it, because the history holds the coin. -/
noncomputable def Strategy.cumUpTo [DecidableEq Ω] (A : Strategy Ω) : (n j : ℕ) → List Ω → (Fin n → Ω) → ℝ
  | 0, _, _, _ => 0
  | _ + 1, 0, _, _ => 0
  | n + 1, j + 1, h, cs => A.leak h + if cs 0 ∈ A.esc h then A.cumUpTo n (j + 1) (cs 0 :: h) (Fin.tail cs)
      else A.cumUpTo n j (cs 0 :: h) (Fin.tail cs)

/-- Live coins and a sample that includes each unit with probability at least proportional to its cone egress give it. -/
def DetectionPressure (μ : PMF Ω) (κ : ℝ) (A : Strategy Ω) : Prop :=
  ∀ h, Verity.Game.prPMFReal μ (· ∈ A.esc h) ≤ Real.exp (-(κ * A.leak h))

def LeakBounded (ℓ : ℝ) (A : Strategy Ω) : Prop := ∀ h, 0 ≤ A.leak h ∧ A.leak h ≤ ℓ

/-- A trigger that makes the honest model emit weight bits fails it. -/
def TrustedModel {W H T : Type} (ρ : PMF W) (S : W → T) (hon : W → H) (g : ℝ) : Prop :=
  ∀ γ : H → T, Verity.Game.prPMFReal ρ (fun w => γ (hon w) = S w) ≤ g

/-- A coin says, per unit, whether an incorrect `u` would be caught (sampled, and not lucky). -/
noncomputable abbrev indepSample {U : Type} [Fintype U] [DecidableEq U] (p : U → PMF Bool) : PMF (U → Bool) :=
  Verity.Game.piDistribution p

def acceptCoins {U : Type} [Fintype U] [DecidableEq U] (B : Finset U) : Finset (U → Bool) :=
  univ.filter fun c => ∀ u ∈ B, c u = false

noncomputable def changedThrough [DecidableEq Ω] {O : Type} [DecidableEq O] (A : Strategy Ω) (chg : List Ω → Finset O) :
    (n : ℕ) → List Ω → (Fin n → Ω) → Finset (Fin n × O)
  | 0, _, _ => ∅
  | n + 1, h, cs => (chg h).map ⟨fun o => (0, o), fun _ _ e => (Prod.mk.inj e).2⟩ ∪
      if cs 0 ∈ A.esc h then (changedThrough A chg n (cs 0 :: h) (Fin.tail cs)).map
        ⟨fun p => (p.1.succ, p.2), fun _ _ e => Prod.ext (Fin.succ_injective _ (Prod.mk.inj e).1) (Prod.mk.inj e).2⟩
      else ∅

abbrev Dev (Slot V : Type) : Type := Σ C : Finset Slot, (C → V)

def Dev.apply {Slot V : Type} [DecidableEq Slot] (d : Dev Slot V) (e : Slot → V) : Slot → V :=
  fun s => if h : s ∈ d.1 then d.2 ⟨s, h⟩ else e s

end Population.Exfil
