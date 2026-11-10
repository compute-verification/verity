module

public import Mathlib
public import Mathlib.Basic.Real.Basic
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

/-!
# The compute dampener: the vocabulary

The training-efficiency bound of `benchmarks/nci/tiers.py`. A developer spends `C` FLOP on monitored hardware and `U` on
unmonitored hardware, and must still compute the counted multiplications `M` of training (an updated weight times an
activation or an activation gradient). Each stands for `2/κ` FLOP of an unconstrained run, `κ` being the share of
training FLOP that are counted multiplications at 2 FLOP each, so the compute efficiency is `η = (2/κ) M / (C + U)`
(`efficiency`). The dampener bounds it by `s + min(r, μ) + ν + χ` (`Nci.Guarantees.Dampener`), with:

* `s = U/(C + U)`, the unmonitored compute share (`unregisteredShare`);
* `r = 1 − W_cert/C`, the monitored compute residual (`residualShare`);
* `ν = (1 − s) · 2/(κ ρ)`, the declared computation at `ρ` declared gates per counted multiplication (`declaredShare`);
  at NCI's thread bound `ρ = ρ_T = b_a/(α Λ(K))` (`rhoT`), `Λ(K) = max(e, reuse · K/(4 d b_w))` (`Lambda`, which is
  `Nci.Threading.bound`);
* `μ = 2 reuse · R α/(κ d b_w b_a)`, the memory cap at wipe state `R` (`memoryCap`);
* `χ`, the cheating share.
-/

@[expose] public section

namespace Nci.Dampener

/-- **Compute efficiency** `η = (2/κ) M / (C + U)`: the FLOP of an unconstrained run that `M` counted multiplications
stand for, `2/κ` each, over the FLOP the developer spends, `C` monitored and `U` unmonitored. -/
noncomputable def efficiency (κ M C U : ℝ) : ℝ := 2 / κ * M / (C + U)

/-- **The unmonitored compute share** `s = U/(C + U)`. -/
noncomputable def unregisteredShare (C U : ℝ) : ℝ := U / (C + U)

/-- **The monitored compute residual** `r = 1 − W_cert/C`: the share of the monitored compute `C` outside the compute
certificate `W_cert`. -/
noncomputable def residualShare (W C : ℝ) : ℝ := 1 - W / C

/-- **Counted multiplications per `b_a` bits read** at read cap `K` (Theorem 1 over threads,
`Nci.Guarantees.ThreadedReads`): `Λ(K) = max(e, reuse · K/(4 d b_w))`, `max(e, K/(2 d b_w))` at `reuse = 2`. It is
`Nci.Threading.bound` with real arguments: `T.bound K b_w d e = Lambda T.reuse e d b_w K`. -/
noncomputable def Lambda (reuse e d b_w K : ℝ) : ℝ := max e (reuse * K / (4 * d * b_w))

/-- **Declared gates per counted multiplication** at NCI's thread bound, `ρ_T = b_a/(α Λ)`:
`Nci.Guarantees.ThreadedThrottled` gives `b_a |M| ≤ α Λ size(d)`, that is `|M| ≤ size(d)/ρ_T`. -/
noncomputable def rhoT (α b_a Λ : ℝ) : ℝ := b_a / (α * Λ)

/-- **The declared computation's share** `ν = (1 − s) · 2/(κ ρ)`: the monitored compute, at `ρ` declared gates per counted
multiplication and `2/κ` FLOP per counted multiplication, over the FLOP spent. -/
noncomputable def declaredShare (κ ρ C U : ℝ) : ℝ := (1 - unregisteredShare C U) * (2 / (κ * ρ))

/-- **The memory cap** `μ = 2 reuse · R α/(κ d b_w b_a)`, `4 R α/(κ d b_w b_a)` at `reuse = 2`: the efficiency of the
monitored servers when each window of `R/(αF)` ticks starts from at most `R` bits of state, takes in at most `αF` bits
a tick, and holds at most `K Λ(K)/b_a` counted multiplications at `K = 2R`, in the quadratic regime of `Λ`. -/
noncomputable def memoryCap (reuse κ R α d b_w b_a : ℝ) : ℝ := 2 * reuse * R * α / (κ * d * b_w * b_a)

end Nci.Dampener
