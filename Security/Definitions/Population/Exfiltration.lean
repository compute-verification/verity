import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Fin.Tuple.Basic

/-!
# Weight exfiltration before detection: the windowed audit as a game

A lab serves its model through sampled proofs. The adversary controls the serving stack and knows the weights. The
deployment is a sequence of audit windows (a request, or a batch of them). In each window the prover commits the
window's values; the commitment fixes which units are incorrect, and so what the window's egress carries beyond the honest
run. Then the challenger draws the window's coins, live and fresh, and a drawn incorrect unit fails visibly. The egress of
a window leaves before its verdict (detection, not prevention), and the deployment halts at the first failure.

This file is the vocabulary for that game, over a finite coin space with weights `μ`, with no measure theory.

* A `Strategy` is adaptive: at a window it has seen every earlier window's coins (the history `h`, newest first). It
  fixes the window's leak `leak h` (in a population, the egress slots in its incorrect units' cones, `Dataflow.coneEgress`)
  and the coins on which the window escapes detection, `esc h`. Neither depends on the window's own coin: that is what
  live coins, drawn after the commitment, mean.
* `ex μ n F` and `prob μ n E` are the expectation and the probability over `n` windows' coins, each window's coin drawn
  independently from `μ`: an explicit sum over coin paths with product weights.
* `Strategy.cum` is what leaves up to and including the first detection (its egress left before its verdict);
  `Strategy.cumPrev` what leaves when a caught window's egress is withheld (egress waits for the verdict);
  `Strategy.cumUpTo j` what leaves before the `j`-th detection, for a deployment that tolerates `j - 1` failures.

The named assumptions:

* `DetectionPressure μ κ A`: whatever the prover has seen, a window whose egress leaks `ℓ` escapes with probability at
  most `exp (-κ ℓ)`. Live coins and a draw that includes each unit with probability at least `κ E(u) / (1 - p)`, `E(u)`
  its cone egress and `p` a drawn incorrect unit's luck, give it (`Exfil.acceptProb_le_exp`).
* `TrustedModel ρ S hon g`: from the honest egress alone no guess of the target `S` (a function of the weights drawn
  from `ρ`) succeeds with probability above `g`. It is what trusted training buys: a model with a trigger that makes its
  honest outputs carry weight bits fails it.

`indepDraw a` is the draw that catches unit `u` independently with probability `a u`, and `acceptCoins B` its coins that
miss the incorrect set `B`. A deployment's egress is its honest egress with a deviation (`Dev`, `Dev.apply`) applied: the
slots that differ, and their values. `changedThrough` collects the slots each window changes, through the first
detection.

When a guarantee states these, they move to a `Models/Population/Assumptions` module with their records.
-/

namespace Population.Exfil

open Finset

/-- An adaptive exfiltration strategy against a windowed audit with live coins. At the window after the history `h` of
earlier windows' coins (newest first) the prover's commitment fixes `leak h`, what the window's egress carries beyond
the honest run, and `esc h`, the window's coins that miss its incorrect units. Both are fixed before the window's coin is
drawn. -/
structure Strategy (Ω : Type) where
  leak : List Ω → ℝ
  esc : List Ω → Finset Ω

variable {Ω : Type}

/-- The weight of a path of coins, one per window, each drawn independently from `μ`. -/
def pathWeight (μ : Ω → ℝ) {n : ℕ} (cs : Fin n → Ω) : ℝ := ∏ i, μ (cs i)

/-- The expectation of `F` over `n` windows' coins. -/
noncomputable def ex [Fintype Ω] (μ : Ω → ℝ) (n : ℕ) (F : (Fin n → Ω) → ℝ) : ℝ :=
  ∑ cs, pathWeight μ cs * F cs

open Classical in
/-- The probability of the event `E` over `n` windows' coins. -/
noncomputable def prob [Fintype Ω] (μ : Ω → ℝ) (n : ℕ) (E : (Fin n → Ω) → Prop) : ℝ :=
  ex μ n fun cs => if E cs then 1 else 0

/-- **What leaves up to and including the first detection**, over `n` windows from the history `h`: each window's leak,
until the first window whose coin catches an incorrect unit. That window's egress left before its verdict. -/
noncomputable def Strategy.cum [DecidableEq Ω] (A : Strategy Ω) : (n : ℕ) → List Ω → (Fin n → Ω) → ℝ
  | 0, _, _ => 0
  | n + 1, h, cs => A.leak h + if cs 0 ∈ A.esc h then A.cum n (cs 0 :: h) (Fin.tail cs) else 0

/-- **What leaves when a caught window's egress is withheld** (egress waits for the verdict): the leaks of the windows
before the first detection. -/
noncomputable def Strategy.cumPrev [DecidableEq Ω] (A : Strategy Ω) : (n : ℕ) → List Ω → (Fin n → Ω) → ℝ
  | 0, _, _ => 0
  | n + 1, h, cs => if cs 0 ∈ A.esc h then A.leak h + A.cumPrev n (cs 0 :: h) (Fin.tail cs) else 0

/-- **What leaves before the `j`-th detection**, for a deployment that tolerates `j - 1` failures: each window's leak,
up to and including the `j`-th window that is caught. After a detection the prover knows it, since the history holds
the coin. -/
noncomputable def Strategy.cumUpTo [DecidableEq Ω] (A : Strategy Ω) : (n j : ℕ) → List Ω → (Fin n → Ω) → ℝ
  | 0, _, _, _ => 0
  | _ + 1, 0, _, _ => 0
  | n + 1, j + 1, h, cs => A.leak h + if cs 0 ∈ A.esc h then A.cumUpTo n (j + 1) (cs 0 :: h) (Fin.tail cs)
      else A.cumUpTo n j (cs 0 :: h) (Fin.tail cs)

/-- The window's coins are drawn from a probability distribution. -/
def IsDist [Fintype Ω] (μ : Ω → ℝ) : Prop := (∀ c, 0 ≤ μ c) ∧ ∑ c, μ c = 1

/-- **Detection pressure** (`DetectionPressure`): whatever the prover has seen, a window whose egress leaks `ℓ` escapes
detection with probability at most `exp (-κ ℓ)`. Live coins and a draw that includes each unit with probability at
least proportional to its cone egress give it. -/
def DetectionPressure (μ : Ω → ℝ) (κ : ℝ) (A : Strategy Ω) : Prop :=
  ∀ h, ∑ c ∈ A.esc h, μ c ≤ Real.exp (-(κ * A.leak h))

/-- Every window leaks between `0` and `ℓ`: `ℓ` is the most one window's egress can carry beyond the honest run. -/
def LeakBounded (ℓ : ℝ) (A : Strategy Ω) : Prop := ∀ h, 0 ≤ A.leak h ∧ A.leak h ≤ ℓ

open Classical in
/-- **Trusted model** (`TrustedModel`): with the weights drawn from `ρ`, no guess of the target `S` from the honest
egress `hon` succeeds with probability above `g`. A trigger that makes the honest model emit weight bits fails it. -/
def TrustedModel {W H T : Type} [Fintype W] (ρ : W → ℝ) (S : W → T) (hon : W → H) (g : ℝ) : Prop :=
  ∀ γ : H → T, ∑ w, ρ w * (if γ (hon w) = S w then 1 else 0) ≤ g

/-- The draw that catches each unit `u` independently with probability `a u`: a coin says, per unit, whether an
incorrect `u` would be caught (drawn, and not lucky). -/
def indepDraw {U : Type} [Fintype U] (a : U → ℝ) (c : U → Bool) : ℝ := ∏ u, if c u then a u else 1 - a u

/-- The coins of the independent draw on which the incorrect set `B` escapes: none of its units is caught. -/
def acceptCoins {U : Type} [Fintype U] [DecidableEq U] (B : Finset U) : Finset (U → Bool) :=
  univ.filter fun c => ∀ u ∈ B, c u = false

/-- **The slots changed through the first detection**: window `i` of `n` changes its egress slots `chg h ⊆ O`, which
the deployment's slot `(i, o)` carries, for each window up to and including the first that is caught. -/
noncomputable def changedThrough [DecidableEq Ω] {O : Type} [DecidableEq O] (A : Strategy Ω) (chg : List Ω → Finset O) :
    (n : ℕ) → List Ω → (Fin n → Ω) → Finset (Fin n × O)
  | 0, _, _ => ∅
  | n + 1, h, cs => (chg h).map ⟨fun o => (0, o), fun _ _ e => (Prod.mk.inj e).2⟩ ∪
      if cs 0 ∈ A.esc h then (changedThrough A chg n (cs 0 :: h) (Fin.tail cs)).map
        ⟨fun p => (p.1.succ, p.2), fun _ _ e => Prod.ext (Fin.succ_injective _ (Prod.mk.inj e).1) (Prod.mk.inj e).2⟩
      else ∅

/-- **A deviation of a deployment's egress**: the slots it changes and what it writes there. -/
abbrev Dev (Slot V : Type) : Type := Σ C : Finset Slot, (C → V)

/-- The egress `e` with the deviation `d` applied: `d`'s values on its slots, `e`'s elsewhere. -/
def Dev.apply {Slot V : Type} [DecidableEq Slot] (d : Dev Slot V) (e : Slot → V) : Slot → V :=
  fun s => if h : s ∈ d.1 then d.2 ⟨s, h⟩ else e s

end Population.Exfil
