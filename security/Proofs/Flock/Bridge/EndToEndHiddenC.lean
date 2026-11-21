module

public import Proofs.Flock.Bridge.Canon
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
import Proofs.Spot.Sampling.SamplingDesign
import Proofs.Flock.Soundness.Discharge.Composed.PublicJ
import Proofs.Flock.Soundness.Discharge.Tables.Headline
import Proofs.Flock.Soundness.Game

/-!
# `EndToEndHiddenC`, from `OpeningsCanonTarget`

`EndToEndHidden`'s argument with "wrong" `incorrectRegZc`:

* `zk_session_soundHJ_custody_regC`: `zk_session_soundHJ_custody_reg` at `zk_flock_countC_regC`, for any `canon`
  under `OpeningsCanon`, at the same bound;
* `Flock.Bridge.endToEndHiddenC_of`: `OpeningsCanonTarget` gives `EndToEndHiddenC`. It is
  `zk_session_soundHJ_custody_regC` at `Bridge.canonAt` with its first and last terms in closed form, as
  `EndToEndHidden` is `zk_session_soundHJ_custody_reg`'s (`Spot.Sampling.SamplingDesign.subset_miss`, and `linkBoundZC` at A2);
* `Flock.Bridge.endToEndHidden_of_c`: `EndToEndHiddenC` gives `EndToEndHidden` (`incorrectRegZ_subset_incorrectRegZc`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open Game Spot.Sampling Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

end FlockSoundness.Discharge.ZkHidden

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.ZkLink

/-- **`EndToEndHiddenC` gives `EndToEndHidden`**: a unit wrong on the registration is wrong on it with canonical
openings (`incorrectRegZ_subset_incorrectRegZc`), so `EndToEndHidden`'s event implies `EndToEndHiddenC`'s. -/
theorem endToEndHidden_of_c (h : EndToEndHiddenC) : Flock.Guarantees.EndToEndHidden := by
  intro n Reg I ht D hD hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk
    hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P vb σ hCR hA2
  have h' := h n Reg I ht D hD hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust k Rw M
    hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS hCR hA2
  refine le_trans (prob_mono ?_ _ _) h'
  intro o ho
  exact ⟨ho.1, ho.2.trans (Finset.card_le_card (incorrectRegZ_subset_incorrectRegZc _ _ _ _ _ _ _ _))⟩

end Flock.Bridge
