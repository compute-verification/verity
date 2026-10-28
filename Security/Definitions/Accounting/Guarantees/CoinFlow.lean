import Definitions.Accounting.CoinFlow

/-!
# The coin rules give (H-μ)'s view independence (trusted)

The statements, over `Definitions/Accounting/CoinFlow.lean`. The proofs are in `Security/Proofs/Accounting/CoinFlow/`.
Each holds of every Program the Build admits (`Dataflow.admits`), with no hypothesis on its nodes' operations.

* `ViewIndependent`, from rule (b): a noise injection's fresh coins reach no value outside its forward cone. Every node
  that doesn't read the injection's value, through any chain, has the same value under any two draws that differ only
  on those coins.
* `CoinsThroughNoise`, from rule (a) and sink confinement: a value that isn't a sink depends on the coins only through
  the noise injections it reads. Two draws that agree on the fresh coins of every noise injection in its backward
  cone give it the same value.
* `OneReader`, from rule (b): a noise injection's forward cone is the injection and its one reader's forward cone. So a
  view may hold every value except the noised operand and what reads its consumer.
* `PointMass`: `Accounting.Guarantees.HMu.CoinPointMass` applies outright. A skipped value `a + Σ_j [e_j]·v_j` over a
  noise injection's `u` fresh coins, with `a`, the nonzero integer weights `v` and the view `F` functions of `View k`
  (every value outside its forward cone and every other coin), takes any value with probability at most
  `C(u, ⌊u/2⌋)/2^u` given the view. The bits are uniform: the probability is over every coin of the run.

What stays outside, by construction: the values in the injection's forward cone. Those are its reader's other words,
which share the noised operand (a dithered row's words in every column). So the view independence `CoinPointMass`
needs is checked for every view built from the rest of the Program. A view holding a word that shares the operand still
determines the skipped value, as `Accounting.Guarantees.HMu` says.
-/

namespace Accounting.Guarantees.CoinFlow

open Accounting.CoinFlow Accounting.HMu Pouw.RelativeSecurity

/-- **A noise injection's coins reach only its forward cone (rule (b)).** In a Program the Build admits, two coin draws
that agree off a noise injection's fresh coins give every node outside its forward cone the same value. -/
def ViewIndependent : Prop :=
  ∀ (m : ℕ) (C V : Type) [DecidableEq C] [Inhabited V] (P : Dataflow m C V), P.admits = true →
    ∀ (k : Fin m) (σ σ' : C → Bool), (∀ c, c ∉ P.noiseCoins k → σ c = σ' c) →
      ∀ g, ¬P.Desc k g → P.eval σ g = P.eval σ' g

/-- **A value that isn't a sink reads the coins only through noise injections (rule (a)).** In a Program the Build
admits, two coin draws that agree on the fresh coins of every noise injection a non-fold node reads, through any chain,
give it the same value. -/
def CoinsThroughNoise : Prop :=
  ∀ (m : ℕ) (C V : Type) [DecidableEq C] [Inhabited V] (P : Dataflow m C V), P.admits = true →
    ∀ (g : Fin m) (σ σ' : C → Bool), P.fold g = false →
      (∀ k, P.Desc k g → ∀ c ∈ P.noiseCoins k, σ c = σ' c) → P.eval σ g = P.eval σ' g

/-- **A noise injection's forward cone is its one reader's (rule (b)).** In a Program the Build admits, a node that
reads a noise injection's value through some chain is the injection or reads the injection's one reader. -/
def OneReader : Prop :=
  ∀ (m : ℕ) (C V : Type) [DecidableEq C] (P : Dataflow m C V), P.admits = true →
    ∀ (k r g : Fin m), P.noise k = true → k ∈ P.reads r → P.Desc k g → g = k ∨ P.Desc r g

/-- **(H-μ) from a noise injection's coins, outright.** In a Program the Build admits, over uniform coins, a skipped
value `a + Σ_j [e_j]·v_j` on the `u` fresh coins of node `k`, whose base, nonzero integer weights and view are functions
of `View k`, takes any value `t` with probability at most `C(u, ⌊u/2⌋)/2^u` given the view. -/
def PointMass : Prop :=
  ∀ (m : ℕ) (C V : Type) [Fintype C] [DecidableEq C] [Inhabited V] (P : Dataflow m C V), P.admits = true →
    ∀ (k : Fin m) (O : Type) (F : P.View k → O) (a : P.View k → ℤ) (v : P.View k → {c // c ∈ P.noiseCoins k} → ℤ)
      (o : O) (t : ℤ),
      prob (fun σ : C → Bool => F (P.view k σ) = o ∧ (∀ j, v (P.view k σ) j ≠ 0) ∧
          a (P.view k σ) + coinSum (v (P.view k σ)) (fun j => σ j.1) = t) ≤
        (((P.noiseCoins k).card).choose ((P.noiseCoins k).card / 2) : ℝ) / 2 ^ (P.noiseCoins k).card *
          prob (fun σ : C → Bool => F (P.view k σ) = o)

end Accounting.Guarantees.CoinFlow
