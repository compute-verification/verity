module

public import Definitions.NetworkCertifier.Carried
public import Proofs.NetworkCertifier.Witnesses
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
import Proofs.Flock.Claims.Prob

/-!
# The nontrivial lemma of the leak bound over a carried pool

`PooledLeakCarried`. At one run whose audit always accepts, `q = 1`, `Fmax = 1`, no error, every row the empty pool's,
and its observations empty where the horizon is one step: no run meets its leak relation, no opening collides (the two
digests of the empty row hash one message, `poolColl_empty`), and the escape bound reads `1 ≤ 0`.

Its honest lemma is open: it needs a deployed check that pins its checked inputs (`CheckedUnique`), its closing row
(`CheckedClosing`) and the roots it opens and leaves (`CarriedRoots`).
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling

/-- **No collision at an opening of the empty pool**: the digests of the empty row hash one message, twice. -/
theorem poolColl_empty (G : Hash.Msg → Hash.Digest) {m : ℕ} (chains : Fin m → Bool) (i : Fin m) :
    ¬PoolColl G chains (fun _ => []) (fun _ => []) (fun _ => []) i := by
  rintro ⟨x, hx, y, hy, hne, -⟩
  have hseg : Hash.RowSeg.segmentMsgs [] = [] := by
    simp [Hash.RowSeg.segmentMsgs, Hash.RowSeg.segCount, Hash.RowSeg.SEG_BITS]
  have hopen : poolOpen chains (fun _ => ([] : Hash.Row)) (fun _ => []) i = [] := by
    unfold poolOpen
    split_ifs <;> rfl
  simp [Hash.RowSeg.hashed, hseg, hopen] at hx hy
  exact hne (hx.trans hy.symm)

/-- **`PooledLeakCarried`'s escape bound fails at an audit that always accepts**: one run, `q = 1`, `Fmax = 1`, no
error, every row the empty pool's, and the run's observations empty where its horizon is one step, so its good event
fails at every sample and no opening collides. -/
theorem PooledLeakCarried_nontrivial :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬PooledGoodCarried (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) (fun _ _ => True) (fun _ => [])
          (fun _ => []) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart) (fun _ _ => false)
          (fun _ _ _ _ => [false]) (fun _ => false) (fun _ => []) 1 (fun _ => ∅) (fun _ => [])) ≤
      (1 - 1 : ℝ) ^ 1 + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) +
        ∑ i : Fin 1, Pr (fun _ : Fin 1 → Unit =>
          PoolColl (fun _ _ => 0) (fun _ => false) (fun _ => []) (fun _ => []) (fun _ => []) i) := by
  have hp : (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
      ¬PooledGoodCarried (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) (fun _ _ => True) (fun _ => [])
        (fun _ => []) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart) (fun _ _ => false)
        (fun _ _ _ _ => [false]) (fun _ => false) (fun _ => []) 1 (fun _ => ∅) (fun _ => [])) = fun _ => True :=
    funext fun _ => propext (iff_true_intro ⟨fun _ => trivial, fun h => by
      obtain ⟨-, rin, rout, -, hl⟩ := h
      exact absurd (hl 0).2.2.1 (by simp)⟩)
  have hc : ∀ i : Fin 1, (fun _ : Fin 1 → Unit =>
      PoolColl (fun _ _ => 0) (fun _ => false) (fun _ => []) (fun _ => []) (fun _ => []) i) = fun _ => False :=
    fun i => funext fun _ => propext (iff_false_intro (poolColl_empty _ _ i))
  rw [hp]
  simp only [hc]
  simp [Pr, Claims.prUniform]

end NetTiming.Witness
