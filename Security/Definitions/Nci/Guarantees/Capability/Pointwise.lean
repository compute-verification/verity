import Definitions.Nci.Capability.Pointwise
import Definitions.Core.Game

/-!
# The pointwise slowdown

The slowdown read one execution at a time, over a decomposition of its actual circuit (`Nci.Capability.Counted`,
`Counted.Decomposition`): `B` disjoint qualifying regions of `c` gates and a remainder of `U`, `N = c·B + U`
(`DecompositionCount`), each counted occurrence represented by one gate. Under the whole cap `q_whole`, the split cap
`q_split` (`q = q_whole + q_split`), the remainder ceiling `ν_out` and utilization `U ≤ f·N`:

* **Theorem A** (`PointwiseCount`): `Ω ≤ Γ·N`, `Γ = q/c + f·max(ν_out − q/c, 0)`; and where `q/c ≤ ν_out`,
  `Γ = (q/c)(1 − f) + ν_out·f` (`GammaBelow`);
* **Theorem B** (`PointwiseThreshold`): an execution carrying `Ω ≥ Ω_*` has `N ≥ L = Ω_*/Γ`;
* **Theorem C** (`PointwiseEvent`): over the executions of any game, if the decomposition hypotheses fail on an accepted
  execution with probability at most `δ`, and coverage puts success below `N_max` without `Ω ≥ Ω_*` at probability at
  most `η`, an accepted execution succeeds with fewer than `min(L, N_max)` operation gates with probability at most
  `δ + η`.

`WholeCapLocal` gives `q_whole` from the local bounds, for regions that are proof blocks of a `(c, k + k_F)`-partition:
`q_whole = r·h·((k + k_F) + c/g)`.
-/

namespace Nci.Guarantees

open Verity Finset Nci.Capability
open scoped ENNReal

/-- **`N = c·B + U`**: a decomposition's regions and remainder count every operation gate once. -/
def DecompositionCount : Prop :=
  ∀ (X : Counted) (c : ℕ) (D : X.Decomposition c), X.N = c * D.B + D.U

/-- **Theorem A (counting).** A decomposition of the counted computation `X` at block size `c` meeting the whole cap
`q_whole`, the split cap `q_split`, the remainder ceiling `ν_out` and utilization `f` bounds its weighted total by
`Γ` a gate: `Ω ≤ Γ·N`, with `Γ = q/c + f·max(ν_out − q/c, 0)` and `q = q_whole + q_split`. Every counted occurrence is
represented in one region or in the remainder, the regions carry at most `q` each and the remainder `ν_out` a gate. -/
def PointwiseCount : Prop :=
  ∀ (X : Counted) (c : ℕ) (D : X.Decomposition c) (qWhole qSplit ν f : ℝ),
    D.WholeCap qWhole → D.SplitCap qSplit → D.RemainderCeiling ν → D.Utilization f →
      X.Ω ≤ Gamma (qWhole + qSplit) c ν f * X.N

/-- **Theorem A's closed form.** Where the per-gate cap of a region is at most the remainder's, `q/c ≤ ν_out`,
`Γ = (q/c)(1 − f) + ν_out·f`: the regions' rate on the share `1 − f` and the remainder's on `f`. -/
def GammaBelow : Prop :=
  ∀ (q : ℝ) (c : ℕ) (ν f : ℝ), q / c ≤ ν → Gamma q c ν f = q / c * (1 - f) + ν * f

/-- **Theorem B (threshold).** Under Theorem A's hypotheses, an execution carrying at least `Ω_*` of the weighted count,
with `Γ > 0`, has at least `L = Ω_*/Γ` operation gates. -/
def PointwiseThreshold : Prop :=
  ∀ (X : Counted) (c : ℕ) (D : X.Decomposition c) (qWhole qSplit ν f Ωstar : ℝ),
    D.WholeCap qWhole → D.SplitCap qSplit → D.RemainderCeiling ν → D.Utilization f →
      Ωstar ≤ X.Ω → 0 < Gamma (qWhole + qSplit) c ν f → Ωstar / Gamma (qWhole + qSplit) c ν f ≤ X.N

/-- **Theorem C (the security event).** Over the executions of a game `g` played by the strategy `s`, each with its
counted computation `X a`: if an accepted execution fails the decomposition hypotheses with probability at most `δ`
(`Pr[accept ∧ ¬ Decomposes] ≤ δ`, what soundness and the accounting give), and a successful execution below `N_max`
operation gates carries less than `Ω_*` with probability at most `η` (`Pr[success ∧ N < N_max ∧ ¬ Ω ≥ Ω_*] ≤ η`,
coverage at cost `N`), then with `Γ > 0` an accepted execution succeeds below `min(L, N_max)` gates, `L = Ω_*/Γ`, with
probability at most `δ + η`. -/
def PointwiseEvent : Prop :=
  ∀ {α : Type} (g : Game α) (s : Game.Strategy g) (X : α → Counted) (accept success : α → Prop) (c : ℕ)
    (qWhole qSplit ν f Ωstar Nmax : ℝ) (δ η : ℝ≥0∞),
    0 < Gamma (qWhole + qSplit) c ν f →
    Game.prob (fun a => accept a ∧ ¬ (X a).Decomposes c qWhole qSplit ν f) g s ≤ δ →
    Game.prob (fun a => success a ∧ ((X a).N : ℝ) < Nmax ∧ ¬ Ωstar ≤ (X a).Ω) g s ≤ η →
    Game.prob (fun a => accept a ∧ success a ∧
      ((X a).N : ℝ) < min (Ωstar / Gamma (qWhole + qSplit) c ν f) Nmax) g s ≤ δ + η

/-- **The whole cap from the local bounds.** In a `(c, k + k_F)`-partition of the committed declaration `R` (`k_F` the
bits faulty proof blocks inject, a parameter here), where every proof block holds slots priced at most `h`, the uses'
upstream certificate holds with overlap weights `ω` and every use requires at least `g > 0` gates, take counted
occurrences each a counted product `prod o` computed by gates `G o` containing the product's, at most `r` per product.
Every decomposition whose regions are proof blocks of `R` meets the whole cap `q_whole = r·h·((k + k_F) + c/g)`: a
region's whole occurrences are at most `r` per product it completes (`WeightedProductCap`), it serves at most `k + k_F`
imported uses (`ImportOrRecompute`) and `c/g` recomputed ones (`RecomputeByGates`).

`HeldBudget`, at the `(c, k + k_F)`-partition with the evaluator's length `aBlk = a_ext` and the deficit `Λ = λ`, gives
the budget `h = (k + k_F) + ⌈log₂ B⌉ + a_ext + λ`, at most its value at `⌈log₂ B_max⌉` for `B ≤ B_max`; the advice's
length is inside `a_ext`. Neither `k_F` nor `a_ext` is bounded here. -/
def WholeCapLocal : Prop :=
  ∀ (R : Committed) (c k kF : ℕ) (h : ℝ), R.P.IsCK R.d c (k + kF) →
    ∀ (B : Nci.Covering.MulBlocks R.d) (W : Items R.d.D (Fin B.nw)), SlotsRead B W → HeldPriceAtMost R W h →
    ∀ (A G : Fin B.nt → Finset (Fin R.d.D.N)) (hG : ∀ j, G j ⊆ A j) (hA : ∀ j, Disjoint (A j) R.d.θ)
      (ω : Fin B.nt → ℝ) (g : ℝ),
      (Consumers.ofUses B A G hG hA).Upstream → (Consumers.ofUses B A G hG hA).OverlapWeights ω → 0 < g →
      (∀ j, g ≤ (G j).card) →
    ∀ {κ : Type} [Fintype κ] (prod : κ → Fin B.nw × Fin B.nt) (Go : κ → Finset (Fin R.d.D.N)) (r : ℕ),
      (∀ o, prod o ∈ B.P) → (∀ o, B.m (prod o) ⊆ Go o) → (∀ q, (univ.filter fun o => prod o = q).card ≤ r) →
    ∀ D : (Counted.ofProducts B W ω prod Go).Decomposition c, (∀ b, ∃ u, D.region b = R.P.gates u) →
      D.WholeCap (r * h * ((k + kF : ℕ) + c / g))

end Nci.Guarantees
