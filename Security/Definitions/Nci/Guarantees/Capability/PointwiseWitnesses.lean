import Definitions.Nci.Guarantees.Capability.Pointwise
import Definitions.Nci.Capability.PointwiseToy

/-!
# The pointwise slowdown: its witnesses

Each statement of `Definitions.Nci.Guarantees.Capability.Pointwise` has two witnesses at the toy instance
(`Nci.Capability.PointwiseToy`), listed under the lock's `witnesses`:

* `XHonest`: its premises hold together at the toy, at the parameters its conclusion is read at, and each parameter is
  spent: the premise fails at the next smaller value the toy distinguishes (`0`, or the toy's own value);
* `XNontrivial`: its conclusion fails at the toy where a premise is dropped.

At `D`, every cap is tight: the region computes `1` whole and splits `1`, and the remainder's one gate represents `1`,
so the caps `q_whole = q_split = ν_out = 1` and the utilization `f = 1/2` all fail at `0`. Without the split cap
(`q_split = 0`), `Γ` falls to `1` and the three occurrences exceed `Γ·N = 2`. Theorem C's executions are tight too: the
heavy execution, accepted, has no decomposition (`δ = 1/2`), and `X`, successful, carries `3 < Ω_* = 6` (`η = 1/2`).
-/

namespace Nci.Guarantees

open Verity Finset Nci.Capability Nci.Capability.PointwiseToy
open scoped ENNReal

/-- **`DecompositionCount`'s premise holds** at the toy: `X` has a decomposition at `c = 1`. -/
def DecompositionCountHonest : Prop := Nonempty (X.Decomposition 1)

/-- **`DecompositionCount`'s conclusion can fail**: counted at `c = 2`, above its region's one gate, `D` gives
`c·B + U = 3`, more than `X`'s two operation gates. -/
def DecompositionCountNontrivial : Prop := ¬ X.N = 2 * D.B + D.U

/-- **`PointwiseCount`'s premises hold together** at the toy, at `q_whole = q_split = ν_out = 1` and `f = 1/2`, each
tight: at `0` each fails. -/
def PointwiseCountHonest : Prop :=
  D.WholeCap 1 ∧ D.SplitCap 1 ∧ D.RemainderCeiling 1 ∧ D.Utilization (1 / 2) ∧
    ¬ D.WholeCap 0 ∧ ¬ D.SplitCap 0 ∧ ¬ D.RemainderCeiling 0 ∧ ¬ D.Utilization 0

/-- **`PointwiseCount`'s conclusion can fail**: without the split cap (`q_split = 0`), `Γ = 1` and `X`'s three
occurrences exceed `Γ·N = 2`. -/
def PointwiseCountNontrivial : Prop := ¬ X.Ω ≤ Gamma (1 + 0) 1 1 (1 / 2) * X.N

/-- **`GammaBelow`'s premise holds** at the toy's caps `q = 2`, `c = 1`, with the remainder ceiling `ν_out = 2`, which
`D` meets: `q/c ≤ ν_out`. It fails at the toy's tight ceiling `ν_out = 1`. -/
def GammaBelowHonest : Prop :=
  D.RemainderCeiling 2 ∧ (1 + 1 : ℝ) / (1 : ℕ) ≤ 2 ∧ ¬ (1 + 1 : ℝ) / (1 : ℕ) ≤ 1

/-- **`GammaBelow`'s conclusion can fail**: at `ν_out = 1`, below `q/c = 2`, `Γ = 2` is not
`(q/c)(1 − f) + ν_out·f = 3/2`. -/
def GammaBelowNontrivial : Prop :=
  ¬ Gamma (1 + 1) 1 1 (1 / 2) = (1 + 1) / (1 : ℕ) * (1 - 1 / 2) + 1 * (1 / 2)

/-- **`PointwiseThreshold`'s premises hold together** at the toy: `PointwiseCount`'s, `Γ = 2 > 0` and `Ω_* = 3`, which
`X`'s three occurrences carry, and not `4`. -/
def PointwiseThresholdHonest : Prop :=
  D.WholeCap 1 ∧ D.SplitCap 1 ∧ D.RemainderCeiling 1 ∧ D.Utilization (1 / 2) ∧ (3 : ℝ) ≤ X.Ω ∧
    0 < Gamma (1 + 1) 1 1 (1 / 2) ∧ ¬ (4 : ℝ) ≤ X.Ω

/-- **`PointwiseThreshold`'s conclusion can fail**: at `Ω_* = 5`, more than `X` carries, `L = 5/2` exceeds `X`'s two
operation gates. -/
def PointwiseThresholdNontrivial : Prop := ¬ (5 : ℝ) / Gamma (1 + 1) 1 1 (1 / 2) ≤ X.N

/-- **`PointwiseEvent`'s premises hold together** at the toy's executions, with every execution accepted and
successful, `c = 1`, `q_whole = q_split = ν_out = 1`, `f = 1/2` (so `Γ = 2`), `Ω_* = 6`, `N_max = 10` and
`δ = η = 1/2`, each spent: the heavy execution fails the decomposition hypotheses, and `X` carries less than `Ω_*`, each
on half the coins. -/
def PointwiseEventHonest : Prop :=
  0 < Gamma (1 + 1) 1 1 (1 / 2) ∧
    Game.prob (fun a => True ∧ ¬ (exec a).Decomposes 1 1 1 1 (1 / 2)) game strat ≤ 2⁻¹ ∧
    Game.prob (fun a => True ∧ ((exec a).N : ℝ) < 10 ∧ ¬ (6 : ℝ) ≤ (exec a).Ω) game strat ≤ 2⁻¹ ∧
    ¬ Game.prob (fun a => True ∧ ¬ (exec a).Decomposes 1 1 1 1 (1 / 2)) game strat ≤ 0 ∧
    ¬ Game.prob (fun a => True ∧ ((exec a).N : ℝ) < 10 ∧ ¬ (6 : ℝ) ≤ (exec a).Ω) game strat ≤ 0

/-- **`PointwiseEvent`'s conclusion can fail**: at `δ = 0`, below the heavy execution's half, every execution, with its
two operation gates below `min(L, N_max) = 3`, is accepted and succeeds, more often than `δ + η = 1/2`. -/
def PointwiseEventNontrivial : Prop :=
  ¬ Game.prob (fun a => True ∧ True ∧ ((exec a).N : ℝ) < min (6 / Gamma (1 + 1) 1 1 (1 / 2)) 10) game strat ≤
    0 + 2⁻¹

/-- **`WholeCapLocal`'s premises hold together** at the toy: `R1` is a `(2, 1 + 0)`-partition, the product reads its
slot, the block holds slots priced at most `h = 1`, the use's certificate and weight `1` hold, it requires one gate
(`g = 1`), and the one occurrence is the product, at its gate (`r = 1`); `Dp`'s region is `R1`'s proof block. The budget
and the multiplicity are spent: at `h = 0` and at `r = 0` they fail. -/
def WholeCapLocalHonest : Prop :=
  Toy.R1.P.IsCK Toy.R1.d 2 (1 + 0) ∧ SlotsRead Toy.B Toy.W ∧ HeldPriceAtMost Toy.R1 Toy.W 1 ∧ Toy.K.Upstream ∧
    Toy.K.OverlapWeights (fun _ => 1) ∧ (0 : ℝ) < 1 ∧ (∀ j, (1 : ℝ) ≤ (Toy.K.G j).card) ∧
    ((0, 0) : Fin 1 × Fin 1) ∈ Toy.B.P ∧ Toy.B.m (0, 0) ⊆ {2} ∧
    (∀ q, (univ.filter fun _ : Fin 1 => ((0, 0) : Fin 1 × Fin 1) = q).card ≤ 1) ∧
    (∀ b, ∃ u, Dp.region b = Toy.R1.P.gates u) ∧
    ¬ HeldPriceAtMost Toy.R1 Toy.W 0 ∧ ¬ ∀ q, (univ.filter fun _ : Fin 1 => ((0, 0) : Fin 1 × Fin 1) = q).card ≤ 0

/-- **`WholeCapLocal`'s conclusion can fail**: at `r = 0`, below the product's one occurrence, the region computes
whole more than `r·h·((k + k_F) + c/g) = 0`. -/
def WholeCapLocalNontrivial : Prop :=
  ¬ Dp.WholeCap (((0 : ℕ) : ℝ) * 1 * (((1 + 0 : ℕ) : ℝ) + ((2 : ℕ) : ℝ) / 1))

end Nci.Guarantees
